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
        # 1. Height Threshold: How high (in meters) must an object be off the "floor" to be kept?
        self.REBAR_HEIGHT_MIN = 0.02  # 2cm (Rebar thickness)
        
        # 2. Kernel Size: Must be wider than the rebar thickness in pixels.
        #    If rebar is ~10-15px wide, use 25 or 31.
        self.MORPH_KERNEL_SIZE = 25 

        self.CLUSTER_RADIUS = 100 
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/rebar/intersections', 10)
        self.debug_image_pub = self.create_publisher(Image, '/rebar/debug_image', 10)
        self.slice_pub = self.create_publisher(Image, '/rebar/sliced_view', 10)
        
        self.bridge = CvBridge()
        
        # Subscribers
        self.get_logger().info(f"Subscribing to: {self.rgb_topic}, {self.depth_topic}")
        
        self.rgb_sub = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        
        self.depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
            
        self.info_sub = self.create_subscription(
            CameraInfo, self.info_topic, self.info_callback, qos_profile_sensor_data)

        # Sync
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.image_callback)

        self.camera_model = None
        self.get_logger().info('>>> Rebar Detector (Morphological) Started...')

    def info_callback(self, msg):
        if self.camera_model is None:
            self.camera_model = msg
            self.get_logger().info(f'Camera Intrinsics Loaded! Frame: {msg.header.frame_id}')

    def image_callback(self, rgb_msg, depth_msg):
        if self.camera_model is None:
            return 

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            # Use 32FC1 for depth in meters
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return
        
        # Replace NaNs/Infs with 0 to prevent crashes
        cv_depth = np.nan_to_num(cv_depth, nan=0.0, posinf=0.0, neginf=0.0)

        # =========================================================
        # 1. ROBUST TILT-AGNOSTIC SLICING (Morphological)
        # =========================================================
        
        # Step A: Estimate the "Background" (Floor)
        # We use Dilation (Max Filter). Since Floor is "deeper" (larger value) than Rebar,
        # Dilation will overwrite the rebar pixels with the neighboring floor depth.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.MORPH_KERNEL_SIZE, self.MORPH_KERNEL_SIZE))
        estimated_floor_depth = cv2.dilate(cv_depth, kernel)

        # Step B: Calculate Height Above Floor
        # Diff = Floor - Original.
        # Where there is Rebar: Floor (1.5m) - Rebar (1.4m) = 0.1m (Positive)
        # Where there is Floor: Floor (1.5m) - Floor (1.5m) = 0.0m
        height_map = estimated_floor_depth - cv_depth

        # Step C: Threshold
        # We keep only pixels that are "standing up" from the floor by at least REBAR_HEIGHT_MIN
        # Also limit max range to avoid sky/noise
        rebar_mask = cv2.inRange(height_map, self.REBAR_HEIGHT_MIN, 0.30) # 0.30m max height
        
        # Step D: Cleanup Mask
        # Remove small noise specs
        rebar_mask = cv2.morphologyEx(rebar_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        
        # --- Publish Sliced View ---
        masked_rgb = cv2.bitwise_and(cv_image, cv_image, mask=rebar_mask)
        try:
            slice_msg = self.bridge.cv2_to_imgmsg(masked_rgb, encoding="bgr8")
            slice_msg.header = rgb_msg.header
            self.slice_pub.publish(slice_msg)
        except CvBridgeError: pass

        # =========================================================
        # 2. STANDARD DETECTION LOGIC
        # =========================================================
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=rebar_mask)
        
        blur = cv2.GaussianBlur(masked_gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=40, maxLineGap=15)

        raw_points = []
        if lines is not None:
            horizontal, vertical = self.sort_lines(lines)
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            for h in horizontal:
                for v in vertical:
                    pt = self.compute_intersection(h, v)
                    if pt:
                        if 0 <= pt[0] < cv_image.shape[1] and 0 <= pt[1] < cv_image.shape[0]:
                            # Double check the point is actually on the mask
                            if rebar_mask[pt[1], pt[0]] > 0:
                                raw_points.append(pt)

        clean_points = self.cluster_points(raw_points, radius=self.CLUSTER_RADIUS)
        
        # Create Markers
        marker_array = MarkerArray()
        fx, fy, cx, cy = self.camera_model.k[0], self.camera_model.k[4], self.camera_model.k[2], self.camera_model.k[5]

        for i, (u, v) in enumerate(clean_points):
            z_depth = cv_depth[v, u]
            
            # Sanity check depth
            if z_depth <= 0.1: 
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
                marker.scale.x = 0.02 
                marker.scale.y = 0.02
                marker.scale.z = 0.02
                marker.color.a = 1.0; marker.color.g = 1.0
                marker_array.markers.append(marker)
                
                cv2.circle(cv_image, (u, v), 5, (0, 255, 0), -1)

        self.marker_pub.publish(marker_array)

        # Publish Debug
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            debug_msg.header = rgb_msg.header
            self.debug_image_pub.publish(debug_msg)
        except CvBridgeError: pass

        # Live Display
        cv2.imshow("Rebar Vision (Result)", cv_image)
        cv2.imshow("Sliced View (Masked)", masked_rgb)
        cv2.waitKey(1)

    # --- HELPERS ---
    def get_neighbor_depth(self, depth_img, u, v, radius=5):
        y_min = max(0, v - radius); y_max = min(depth_img.shape[0], v + radius)
        x_min = max(0, u - radius); x_max = min(depth_img.shape[1], u + radius)
        patch = depth_img[y_min:y_max, x_min:x_max]
        valid = patch[(patch > 0.1) & (patch < 10.0)]
        return np.median(valid) if len(valid) > 0 else 0.0

    def sort_lines(self, lines):
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 35: h_lines.append(line[0])
            elif abs(angle) > 55: v_lines.append(line[0])
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
