#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import message_filters
import math

class Rebar3DDetector(Node):
    def __init__(self):
        super().__init__('rebar_3d_detector')

        # --- CONFIGURATION ---
        self.rgb_topic = '/rgb'
        self.depth_topic = '/depth'
        self.info_topic = '/camera_info'
        
        # --- TUNING PARAMETERS ---
        self.REBAR_HEIGHT_MIN = 0.02
        self.MORPH_KERNEL_SIZE = 25 
        self.LINE_MERGE_DIST = 30  
        self.CLUSTER_RADIUS = 100 
        
        self.marker_pub = self.create_publisher(MarkerArray, '/rebar/intersections', 10)
        self.debug_image_pub = self.create_publisher(Image, '/rebar/debug_image', 10)
        self.slice_pub = self.create_publisher(Image, '/rebar/sliced_view', 10)
        
        self.bridge = CvBridge()
        
        self.get_logger().info(f"Subscribing to: RGB={self.rgb_topic}, Depth={self.depth_topic}, Info={self.info_topic}")
        
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.info_sub = self.create_subscription(CameraInfo, self.info_topic, self.info_callback, qos_profile_sensor_data)

        # Increased SLOP to 1.0 second to handle laggy simulation
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=1.0)
        self.ts.registerCallback(self.image_callback)

        self.camera_model = None
        self.get_logger().info('>>> Rebar Detector (Merged Lines) Started...')

    def info_callback(self, msg):
        if self.camera_model is None:
            self.camera_model = msg
            self.get_logger().info(f'SUCCESS: Camera Intrinsics Loaded! Frame: {msg.header.frame_id}')

    def image_callback(self, rgb_msg, depth_msg):
        # DEBUG: Check if we are even entering the callback
        if self.camera_model is None:
            self.get_logger().warn_once("Callback Triggered, but WAITING FOR CAMERA INFO...")
            return 

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return
        
        # DEBUG: Confirm we have valid images
        # self.get_logger().info(f"Processing Frame: {rgb_msg.header.stamp.sec}")

        cv_depth = np.nan_to_num(cv_depth, nan=0.0)

        # 1. FLOOR REMOVAL
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.MORPH_KERNEL_SIZE, self.MORPH_KERNEL_SIZE))
        estimated_floor_depth = cv2.dilate(cv_depth, kernel)
        height_map = estimated_floor_depth - cv_depth
        rebar_mask = cv2.inRange(height_map, self.REBAR_HEIGHT_MIN, 0.30)
        rebar_mask = cv2.morphologyEx(rebar_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        
        # 2. EDGE DETECTION
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=rebar_mask)
        blur = cv2.GaussianBlur(masked_gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120, apertureSize=3)

        # 3. HOUGH LINES & MERGING
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=20)

        raw_points = []
        clean_image = cv_image.copy()

        if lines is not None:
            horizontal_raw, vertical_raw = self.sort_lines(lines)
            
            horizontal = self.merge_close_lines(horizontal_raw, orientation='h')
            vertical = self.merge_close_lines(vertical_raw, orientation='v')

            # Draw lines for debug
            for x1, y1, x2, y2 in horizontal:
                cv2.line(clean_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            for x1, y1, x2, y2 in vertical:
                cv2.line(clean_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            for h in horizontal:
                for v in vertical:
                    pt = self.compute_intersection(h, v)
                    if pt:
                        if (0 <= pt[0] < cv_image.shape[1] and 
                            0 <= pt[1] < cv_image.shape[0] and
                            rebar_mask[pt[1], pt[0]] > 0):
                            raw_points.append(pt)

        # 4. CLUSTERING
        clean_points = self.cluster_points(raw_points, radius=self.CLUSTER_RADIUS)
        
        # 5. PUBLISH MARKERS
        marker_array = MarkerArray()
        fx, fy, cx, cy = self.camera_model.k[0], self.camera_model.k[4], self.camera_model.k[2], self.camera_model.k[5]

        for i, (u, v) in enumerate(clean_points):
            z_depth = self.get_neighbor_depth(cv_depth, u, v)

            if z_depth > 0.1:
                x = (u - cx) * z_depth / fx
                y = (v - cy) * z_depth / fy
                z = z_depth

                marker = Marker()
                marker.header.frame_id = rgb_msg.header.frame_id 
                marker.header.stamp = rgb_msg.header.stamp
                marker.ns = "rebar_targets"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(x)
                marker.pose.position.y = float(y)
                marker.pose.position.z = float(z)
                marker.scale.x = 0.03; marker.scale.y = 0.03; marker.scale.z = 0.03
                marker.color.a = 1.0; marker.color.g = 1.0 # Green Dots
                marker_array.markers.append(marker)
                
                cv2.circle(clean_image, (u, v), 8, (0, 255, 0), -1)

        self.marker_pub.publish(marker_array)
        
        try:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(clean_image, encoding="bgr8"))
        except CvBridgeError: pass

        # Show Window
        cv2.imshow("Rebar Vision (Merged)", clean_image)
        cv2.waitKey(1)

    # --- HELPERS ---
    def merge_close_lines(self, lines, orientation='h'):
        if not lines: return []
        idx = 1 if orientation == 'h' else 0
        lines_with_pos = []
        for l in lines:
            mid = (l[idx] + l[idx+2]) / 2
            lines_with_pos.append((l, mid))
        lines_with_pos.sort(key=lambda x: x[1])

        merged = []
        current_group = [lines_with_pos[0][0]]
        for i in range(1, len(lines_with_pos)):
            prev_line, prev_pos = lines_with_pos[i-1]
            curr_line, curr_pos = lines_with_pos[i]
            if abs(curr_pos - prev_pos) < self.LINE_MERGE_DIST:
                current_group.append(curr_line)
            else:
                merged.append(self.average_lines(current_group))
                current_group = [curr_line]
        merged.append(self.average_lines(current_group))
        return merged

    def average_lines(self, group):
        x1 = sum([l[0] for l in group]) // len(group)
        y1 = sum([l[1] for l in group]) // len(group)
        x2 = sum([l[2] for l in group]) // len(group)
        y2 = sum([l[3] for l in group]) // len(group)
        return (x1, y1, x2, y2)
    
    def get_neighbor_depth(self, depth_img, u, v, radius=10):
        y_min = max(0, v - radius); y_max = min(depth_img.shape[0], v + radius)
        x_min = max(0, u - radius); x_max = min(depth_img.shape[1], u + radius)
        patch = depth_img[y_min:y_max, x_min:x_max]
        valid = patch[(patch > 0.05) & (patch < 10.0)]
        if len(valid) == 0: return 0.0
        return np.percentile(valid, 10) 

    def sort_lines(self, lines):
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 30: h_lines.append(line[0])
            elif abs(angle) > 60: v_lines.append(line[0])
        return h_lines, v_lines

    def compute_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0: return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return (int(px), int(py))

    def cluster_points(self, points, radius):
        if not points: return []
        clusters = []
        for p in points:
            matched = False
            for i, c in enumerate(clusters):
                dist = math.sqrt((p[0] - c[0])**2 + (p[1] - c[1])**2)
                if dist < radius:
                    clusters[i] = ((c[0] + p[0]) // 2, (c[1] + p[1]) // 2)
                    matched = True; break
            if not matched: clusters.append(p)
        return clusters

def main(args=None):
    rclpy.init(args=args)
    node = Rebar3DDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()