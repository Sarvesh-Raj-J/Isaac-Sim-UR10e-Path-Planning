#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, JointConstraint
from shape_msgs.msg import SolidPrimitive
import tf2_ros
import tf2_geometry_msgs
import math
import time
import threading
import copy
import numpy as np

class RebarMoverBatch(Node):
    def __init__(self):
        super().__init__('rebar_mover_batch')
        
        # --- CONFIGURATION ---
        self.arm_group = "panda_arm"
        self.hand_group = "hand" 
        self.base_frame = "panda_link0"
        self.ee_link = "panda_hand"

        # --- HEIGHT SETTINGS ---
        self.SCAN_HEIGHT = 0.50     
        self.HOVER_HEIGHT = 0.15    
        self.TIE_HEIGHT = 0.15      # Set to 0.02 to actually touch rebar

        self.dedup_radius = 0.05    # 5cm radius to merge duplicate detections
        
        # --- SCAN POSES ---
        self.scan_poses = {
            "center": [0.2409, 0.3304, -0.3162, -0.5634, 0.1061, 0.8759, 0.8945]
        }

        # State
        self.detected_points = []      # Raw points collected during scan
        self.final_points = []         # Sorted, unique points to visit
        self.is_scanning = False       # Flag to enable/disable data collection
        
        # Threading & TF
        self.cb_group = ReentrantCallbackGroup()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Action Client (MoveGroup)
        self._action_client = ActionClient(self, MoveGroup, 'move_action', callback_group=self.cb_group)
        self.get_logger().info(">>> Waiting for MoveIt Action Server...")
        self._action_client.wait_for_server()
        self.get_logger().info(">>> MoveIt Connected.")

        # Subscribers
        self.subscription = self.create_subscription(
            MarkerArray, '/rebar/intersections', self.points_callback, 10, callback_group=self.cb_group)
        
        self.finished_pub = self.create_publisher(MarkerArray, '/rebar/finished', 10)
        
        # Start Main Logic
        threading.Thread(target=self.logic_loop, daemon=True).start()

    def logic_loop(self):
        """
        Main Routine:
        1. Move to Scan Pose
        2. Wait & Collect All Points
        3. Sort Points
        4. Execute Tying Loop
        """
        time.sleep(1.0) 
        
        # --- STEP 1: MOVE TO SCAN POSE ---
        self.get_logger().info("--- PHASE 1: Moving to Scan Position ---")
        if not self.move_to_joints(self.scan_poses["center"]):
            self.get_logger().error("CRITICAL: Failed to reach Scan Pose.")
            return

        # --- STEP 2: SCANNING ---
        self.get_logger().info("--- PHASE 2: Scanning... (Waiting 5s) ---")
        self.detected_points = [] # Clear memory
        self.is_scanning = True   # Enable callback
        time.sleep(5.0)           # Wait for camera to see everything
        self.is_scanning = False  # Stop collecting
        
        if len(self.detected_points) == 0:
            self.get_logger().error("NO POINTS FOUND! Check camera or threshold.")
            return

        # --- STEP 3: SORTING & DEDUPLICATION ---
        self.get_logger().info(f"--- PHASE 3: Processing {len(self.detected_points)} Raw Points ---")
        self.final_points = self.process_and_sort_points(self.detected_points)
        self.get_logger().info(f"final_points count: {len(self.final_points)}")
        
        self.publish_markers(self.final_points, color=(1.0, 1.0, 0.0)) # Show Yellow "To-Do" markers

        # --- STEP 4: EXECUTION LOOP ---
        self.get_logger().info("--- PHASE 4: Starting Execution Sequence ---")
        
        for i, pt in enumerate(self.final_points):
            self.get_logger().info(f">>> Target {i+1}/{len(self.final_points)}")
            self.execute_tying_sequence(pt)
            
        self.get_logger().info("--- MISSION COMPLETE: Returning to Home ---")
        self.move_to_joints(self.scan_poses["center"])

    def points_callback(self, msg):
        """Only collects points if is_scanning is True"""
        if not self.is_scanning or len(msg.markers) == 0:
            return

        for marker in msg.markers:
            try:
                # Transform Camera -> Base
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame, marker.header.frame_id, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                pose_stamped = PoseStamped()
                pose_stamped.header = marker.header
                pose_stamped.pose = marker.pose
                pose_base = tf2_geometry_msgs.do_transform_pose(pose_stamped.pose, transform)
                
                # Store point (We filter duplicates later)
                p = pose_base.position
                self.detected_points.append([p.x, p.y, p.z])
                
            except Exception:
                continue

    def process_and_sort_points(self, raw_points):
        """
        1. Remove duplicates (Dedup)
        2. Sort sequentially (Row by Row)
        """
        if not raw_points: return []
        
        # A. Simple Deduplication
        unique_points = []
        for p in raw_points:
            is_new = True
            for u in unique_points:
                dist = math.sqrt((p[0]-u[0])**2 + (p[1]-u[1])**2)
                if dist < self.dedup_radius:
                    is_new = False
                    break
            if is_new:
                unique_points.append(p)
                
        # B. Sort Logic: "Row by Row"
        # 1. Sort primarily by X (Rows)
        # 2. Sort secondarily by Y (Columns)
        # Rounding X puts points into "Buckets" (Rows) so slight noise doesn't break the row order
        
        # Sort key: (Row Index (10cm chunks), Y value)
        unique_points.sort(key=lambda p: (round(p[0] * 10), p[1]))
        
        return unique_points

    def execute_tying_sequence(self, point_list):
        # Create Pose
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.base_frame
        target_pose.pose.position.x = point_list[0]
        target_pose.pose.position.y = point_list[1]
        target_pose.pose.position.z = point_list[2]
        
        # Orientation: Down
        target_pose.pose.orientation.x = 1.0
        target_pose.pose.orientation.y = 0.0
        target_pose.pose.orientation.z = 0.0
        target_pose.pose.orientation.w = 0.0

        # 1. HOVER
        hover_pose = copy.deepcopy(target_pose.pose)
        hover_pose.position.z += self.HOVER_HEIGHT
        
        self.get_logger().info(f"   Moving to Hover: X={hover_pose.position.x:.2f} Y={hover_pose.position.y:.2f}")
        if not self.move_to_pose(hover_pose):
            self.get_logger().warn("   Skipping (Unreachable Hover)")
            return

        # 2. DIVE
        tie_pose = copy.deepcopy(target_pose.pose)
        tie_pose.position.z += self.TIE_HEIGHT
        
        if abs(self.TIE_HEIGHT - self.HOVER_HEIGHT) > 0.01:
            self.get_logger().info("   Diving...")
            if not self.move_to_pose(tie_pose):
                self.get_logger().warn("   Dive Failed. Resetting.")
                self.move_to_pose(hover_pose)
                return

        # 3. TIE
        self.control_gripper("close")
        time.sleep(1.0)
        self.control_gripper("open")
        
        # 4. RETREAT
        self.move_to_pose(hover_pose)

    # ================= HELPERS =================

    def move_to_joints(self, joint_values):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.arm_group
        goal.request.allowed_planning_time = 5.0
        
        constraints = Constraints()
        joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
        
        for i, val in enumerate(joint_values):
            jc = JointConstraint()
            jc.joint_name = joint_names[i]
            jc.position = val
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)
        return self.send_goal_and_wait(goal)

    def move_to_pose(self, pose):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.arm_group
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_link
        pc.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01]))
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.ee_link
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0

        goal.request.goal_constraints.append(Constraints(position_constraints=[pc], orientation_constraints=[oc]))
        return self.send_goal_and_wait(goal)

    def control_gripper(self, command):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.hand_group
        val = 0.0 if command == "close" else 0.04
        j = JointConstraint()
        j.joint_name = "panda_finger_joint1"
        j.position = val
        j.tolerance_above = 0.01
        j.tolerance_below = 0.01
        j.weight = 1.0
        goal.request.goal_constraints.append(Constraints(joint_constraints=[j]))
        self.send_goal_and_wait(goal)

    def send_goal_and_wait(self, goal):
        future = self._action_client.send_goal_async(goal)
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > 5.0: return False
            time.sleep(0.05)
        goal_handle = future.result()
        if not goal_handle.accepted: return False
        
        res_future = goal_handle.get_result_async()
        start_time = time.time()
        while not res_future.done():
            if time.time() - start_time > 15.0: return True 
            time.sleep(0.05)
        return res_future.result().result.error_code.val == 1

    def publish_markers(self, points, color=(1.0, 0.0, 0.0)):
        marker_array = MarkerArray()
        for i, point in enumerate(points):
            marker = Marker()
            marker.header.frame_id = self.base_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "processed_rebar"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.scale.x = 0.04; marker.scale.y = 0.04; marker.scale.z = 0.04
            marker.color.a = 1.0; marker.color.r = color[0]; marker.color.g = color[1]; marker.color.b = color[2]
            marker_array.markers.append(marker)
        self.finished_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = RebarMoverBatch()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()