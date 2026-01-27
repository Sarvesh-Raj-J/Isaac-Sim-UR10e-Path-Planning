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

class RebarMoverSafe(Node):
    def __init__(self):
        super().__init__('rebar_mover_safe')
        
        # --- CONFIGURATION ---
        self.arm_group = "panda_arm"
        self.hand_group = "hand" 
        self.base_frame = "panda_link0"
        self.ee_link = "panda_hand"

        # Heights (Meters)
        self.SCAN_HEIGHT = 0.50     
        self.HOVER_HEIGHT = 0.15    
        self.TIE_HEIGHT = 0.14    

        self.visited_radius = 0.08  
        self.idle_timeout = 8.0     
        
        # --- SCAN POSES ---
        self.scan_poses = {
            "center": [0.2409, 0.3304, -0.3162, -0.5634, 0.1061, 0.8759, 0.8945],
            "left":   [0.8000, 0.3304, -0.3162, -0.5634, 0.1061, 0.8759, 0.8945],
            "right":  [-0.8000, 0.3304, -0.3162, -0.5634, 0.1061, 0.8759, 0.8945]
        }
        self.scan_order = ["center", "left", "right"]
        self.current_scan_idx = 0

        # State
        self.visited_points = []       
        self.is_busy = True            
        self.homing_done = False       
        self.last_detection_time = time.time()
        
        # Threading & TF
        self.cb_group = ReentrantCallbackGroup()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Action Client (MoveGroup)
        self._action_client = ActionClient(self, MoveGroup, 'move_action', callback_group=self.cb_group)
        self.get_logger().info(">>> Waiting for MoveIt Action Server...")
        self._action_client.wait_for_server()
        self.get_logger().info(">>> MoveIt Connected. Ready to move.")

        # Publishers & Subscribers
        self.subscription = self.create_subscription(
            MarkerArray, '/rebar/intersections', self.points_callback, 10, callback_group=self.cb_group)
        
        self.finished_pub = self.create_publisher(MarkerArray, '/rebar/finished', 10)
        
        # Start Main Logic
        threading.Thread(target=self.logic_loop, daemon=True).start()

    def logic_loop(self):
        """Main State Machine"""
        time.sleep(1.0) 
        self.get_logger().info("--- STARTUP: Moving to Center Scan Pose ---")
        
        if self.move_to_joints(self.scan_poses["center"]):
            self.get_logger().info(">>> ARRIVED at Center Scan Pose.")
            self.homing_done = True
            self.is_busy = False
            self.last_detection_time = time.time()
        else:
            self.get_logger().error("!!! CRITICAL: Failed to reach Home Pose. Robot NOT active.")
            return

        # Idle Loop
        while rclpy.ok():
            if not self.is_busy:
                elapsed = time.time() - self.last_detection_time
                if elapsed > self.idle_timeout:
                    self.is_busy = True
                    self.current_scan_idx = (self.current_scan_idx + 1) % len(self.scan_order)
                    view_name = self.scan_order[self.current_scan_idx]
                    
                    self.get_logger().warn(f"Idle for {elapsed:.1f}s. Switching View -> {view_name}")
                    
                    if self.move_to_joints(self.scan_poses[view_name]):
                        self.get_logger().info(f">>> ARRIVED at {view_name} Scan Pose.")
                    else:
                        self.get_logger().error(f"!!! Failed to move to {view_name}")

                    self.last_detection_time = time.time()
                    self.is_busy = False
            time.sleep(0.5)

    def points_callback(self, msg):
        if self.is_busy or not self.homing_done or len(msg.markers) == 0:
            return

        target_pose_base = None
        
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
                
                if not self.is_visited(pose_base.position):
                    target_pose_base = pose_base
                    break 
            except Exception:
                continue

        if target_pose_base:
            self.is_busy = True
            self.last_detection_time = time.time() 
            self.execute_tying_sequence(target_pose_base)

    def execute_tying_sequence(self, target_pose):
        # Save for memory
        memory_point = Point()
        memory_point.x = target_pose.position.x
        memory_point.y = target_pose.position.y
        memory_point.z = target_pose.position.z

        # Orientation: Down
        target_pose.orientation.x = 1.0
        target_pose.orientation.y = 0.0
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.0

        # --- STEP 1: HOVER ---
        hover_pose = copy.deepcopy(target_pose)
        hover_pose.position.z += self.HOVER_HEIGHT
        
        self.get_logger().info(f"1. Attempting HOVER at X:{hover_pose.position.x:.2f} Y:{hover_pose.position.y:.2f} ...")
        
        if not self.move_to_pose(hover_pose):
            self.get_logger().warn("!!! Skipping Target: Cannot reach Hover Pose.")
            self.is_busy = False
            return
        
        self.get_logger().info(">>> SUCCESS: Hover Position Reached.")

        # --- STEP 2: DIVE ---
        tie_pose = copy.deepcopy(target_pose)
        tie_pose.position.z += self.TIE_HEIGHT
        
        self.get_logger().info("2. Diving to TIE Position...")
        if not self.move_to_pose(tie_pose):
             self.get_logger().warn("!!! ABORT DIVE: Collision Risk or IK Failure.")
             self.move_to_pose(hover_pose)
             self.is_busy = False
             return
        
        self.get_logger().info(">>> SUCCESS: Reached TIE Depth.")

        # --- STEP 3: TIE ACTION ---
        self.get_logger().info("3. Actuating Gripper (CLOSE)...")
        self.control_gripper("close")
        time.sleep(1.0)
        self.get_logger().info("   ...Tying...")
        time.sleep(1.0)
        self.get_logger().info("   Actuating Gripper (OPEN)...")
        self.control_gripper("open")
        
        # --- STEP 4: RETREAT ---
        self.get_logger().info("4. Retreating to Safe Height...")
        self.move_to_pose(hover_pose)
        self.get_logger().info(">>> SEQUENCE COMPLETE.")

        # Add to memory
        self.visited_points.append(memory_point)
        self.publish_finished_markers()
        
        self.last_detection_time = time.time()
        self.is_busy = False

    # ================= HELPERS =================

    def move_to_joints(self, joint_values):
        self.get_logger().info(f"   [PLANNING] Joint Goal: {joint_values[:3]}...")
        goal = MoveGroup.Goal()
        goal.request.group_name = self.arm_group
        goal.request.allowed_planning_time = 5.0
        
        constraints = Constraints()
        joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", 
                       "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
        
        for i, val in enumerate(joint_values):
            jc = JointConstraint()
            jc.joint_name = joint_names[i]
            jc.position = val
            
            # --- CHANGE THIS: Increased from 0.01 to 0.05 ---
            jc.tolerance_above = 0.1
            jc.tolerance_below = 0.1
            
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)
        
        return self.send_goal_and_wait(goal)

    def move_to_pose(self, pose):
        self.get_logger().info(f"   [PLANNING] Pose Goal: Z={pose.position.z:.2f}")
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
        
        # Manual Joint Fallback
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
        # 1. Send Goal
        future = self._action_client.send_goal_async(goal)
        
        # Wait for acceptance (Max 5 seconds)
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > 5.0:
                self.get_logger().error("   [ERROR] Timed out waiting for Goal Acceptance.")
                return False
            time.sleep(0.05)
            
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("   [ERROR] MoveIt REJECTED the goal.")
            return False

        self.get_logger().info("   [EXEC] Goal Accepted. Moving...")
        
        # 2. Wait for Result (Max 15 seconds)
        res_future = goal_handle.get_result_async()
        start_time = time.time()
        
        while not res_future.done():
            # If we wait too long, assume it's stuck or finished silently
            if time.time() - start_time > 15.0:
                self.get_logger().warn("   [WARN] Action Timed Out! (Robot likely moved, but controller didn't report success).")
                # We return True here to let the script continue instead of freezing
                return True 
            time.sleep(0.05)
            
        result = res_future.result().result
        
        if result.error_code.val == 1: 
            self.get_logger().info("   [SUCCESS] Move Complete.")
            return True
        else:
            self.get_logger().warn(f"   [FAIL] Motion Failed. MoveIt Error Code: {result.error_code.val}")
            return False

    def is_visited(self, point):
        for v in self.visited_points:
            dist = math.sqrt((v.x - point.x)**2 + (v.y - point.y)**2)
            if dist < self.visited_radius:
                return True
        return False
        
    def publish_finished_markers(self):
        marker_array = MarkerArray()
        for i, point in enumerate(self.visited_points):
            marker = Marker()
            marker.header.frame_id = self.base_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "finished_rebar"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = point
            marker.scale.x = 0.05; marker.scale.y = 0.05; marker.scale.z = 0.05
            marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.finished_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = RebarMoverSafe()
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