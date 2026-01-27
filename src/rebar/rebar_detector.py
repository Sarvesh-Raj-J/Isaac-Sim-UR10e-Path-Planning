#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
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
        
        # FILTER SETTINGS
        self.FLOOR_CUTOFF = 1.5   # Meters
        self.CLUSTER_RADIUS = 50  # Pixels
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/rebar/intersections', 10)
        
        # --- NEW: Publisher for the processed image ---
        self.debug_image_pub = self.create_publisher(Image, '/rebar/debug_image', 10)
        
        # Setup CV Bridge
        self.bridge = CvBridge()
        
        # Synchronized Subscribers
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self.info_sub = self.create_subscription(CameraInfo, self.info_topic, self.info_callback, 10)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.camera_model = None
        self.get_logger().info('Rebar 3D Detector Started. Waiting for images...')

    def info_callback(self, msg):
        if self.camera_model is None:
            self.camera_model = msg
            self.get_logger().info(f'Camera Intrinsics Loaded! Frame: {msg.header.frame_id}')

    def image_callback(self, rgb_msg, depth_msg):
        if self.camera_model is None:
            return 

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # 2. DEPTH MASKING
        depth_mask = cv2.inRange(cv_depth, 0.1, self.FLOOR_CUTOFF)

        # 3. Image Processing
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        masked_gray = cv2.bitwise_and(blur, blur, mask=depth_mask)

        # 4. Edge & Line Detection
        edges = cv2.Canny(masked_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

        raw_points = []
        if lines is not None:
            horizontal, vertical = self.sort_lines(lines)
            
            # Draw detected lines (Blue) on cv_image
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Find intersections
            for h in horizontal:
                for v in vertical:
                    pt = self.compute_intersection(h, v)
                    if pt:
                        if 0 <= pt[0] < cv_image.shape[1] and 0 <= pt[1] < cv_image.shape[0]:
                            raw_points.append(pt)

        # 5. CLUSTERING
        clean_points = self.cluster_points(raw_points, radius=self.CLUSTER_RADIUS)

        # 6. Create 3D Markers
        marker_array = MarkerArray()
        
        fx = self.camera_model.k[0]
        fy = self.camera_model.k[4]
        cx = self.camera_model.k[2]
        cy = self.camera_model.k[5]

        for i, (u, v) in enumerate(clean_points):
            z_depth = cv_depth[v, u]

            if z_depth <= 0 or math.isinf(z_depth) or math.isnan(z_depth):
                continue

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
            marker.scale.x = 0.03 
            marker.scale.y = 0.03
            marker.scale.z = 0.03
            marker.color.a = 1.0
            marker.color.r = 0.0 
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

            # Draw Green Dot on cv_image
            cv2.circle(cv_image, (u, v), 5, (0, 255, 0), -1)

        # Publish 3D Markers
        self.marker_pub.publish(marker_array)

        # --- NEW: Publish the Debug Image ---
        try:
            # Convert OpenCV image back to ROS message
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            
            # CRITICAL: Copy header so RViz knows this image matches the markers/camera info
            debug_msg.header = rgb_msg.header
            
            self.debug_image_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

        # Optional: Keep local window if you are running locally
        cv2.imshow("Rebar Vision", cv_image)
        cv2.waitKey(1)

    def sort_lines(self, lines):
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 25: 
                h_lines.append(line[0])
            elif abs(angle) > 65: 
                v_lines.append(line[0])
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
                    new_x = (c[0] + p[0]) // 2
                    new_y = (c[1] + p[1]) // 2
                    clusters[i] = (new_x, new_y)
                    matched = True
                    break
            if not matched:
                clusters.append(p)
        return clusters

def main(args=None):
    rclpy.init(args=args)
    node = Rebar3DDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
