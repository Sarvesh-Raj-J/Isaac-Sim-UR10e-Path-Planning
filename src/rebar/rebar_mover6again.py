#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point, Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, JointConstraint, CollisionObject
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
        self.arm_group = "arm"
        self.hand_group = "hand" 
        self.base_frame = "base_link"
        self.ee_link = "tool0"

        # --- HEIGHT SETTINGS ---
        self.SCAN_HEIGHT = 0.50     
        self.HOVER_HEIGHT = 0.15    
        self.TIE_HEIGHT = 0.14      # 0.15 = Tie from air (Change to 0.02 to touch)

        self.dedup_radius = 0.05    
        
        # --- SCAN POSES ---
        self.scan_poses = {
           "center": [2.1929, -2.1073, -0.883, -4.8829, -1.5293, -1.1045]
            }

        # State
        self.detected_points = []      
        self.final_points = []         
        self.is_scanning = False       
        
        # Threading & TF
        self.cb_group = ReentrantCallbackGroup()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Action Client (MoveGroup)
        self._action_client = ActionClient(self, MoveGroup, 'move_action', callback_group=self.cb_group)
        self.get_logger().info(">>> Waiting for MoveIt Action Server...")
        self._action_client.wait_for_server()
        self.get_logger().info(">>> MoveIt Connected.")

        # Publishers
        self.subscription = self.create_subscription(
            MarkerArray, '/rebar/intersections', self.points_callback, 10, callback_group=self.cb_group)
        
        self.finished_pub = self.create_publisher(MarkerArray, '/rebar/finished', 10)
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        
        # Start Main Logic
        threading.Thread(target=self.logic_loop, daemon=True).start()

    def logic_loop(self):
        time.sleep(1.0) 
        
        # 1. SCAN POSE
        self.get_logger().info("--- PHASE 1: Moving to Scan Position ---")
        if not self.move_to_joints(self.scan_poses["center"]):
            self.get_logger().error("CRITICAL: Failed to reach Scan Pose.")
            return

        # 2. SCANNING
        self.get_logger().info("--- PHASE 2: Scanning... (Waiting 5s) ---")
        self.detected_points = [] 
        self.is_scanning = True   
        time.sleep(5.0)           
        self.is_scanning = False  
        
        if len(self.detected_points) == 0:
            self.get_logger().error("NO POINTS FOUND!")
            return

        # 3. SORTING & SAFETY
        self.get_logger().info(f"--- PHASE 3: Processing {len(self.detected_points)} Raw Points ---")
        self.final_points = self.process_and_sort_points(self.detected_points)
        self.add_safety_floor_from_points(self.final_points)
        
        self.get_logger().info(f"final_points count: {len(self.final_points)}")
        self.publish_markers(self.final_points, color=(1.0, 1.0, 0.0)) 

        # 4. EXECUTION
        self.get_logger().info("--- PHASE 4: Starting Execution Sequence ---")
        
        for i, pt in enumerate(self.final_points):
            self.get_logger().info(f">>> Target {i+1}/{len(self.final_points)}")
            self.execute_tying_sequence(pt)
            
        self.get_logger().info("--- MISSION COMPLETE: Returning to Home ---")
        self.move_to_joints(self.scan_poses["center"])

    def execute_tying_sequence(self, point_list):
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.base_frame
        target_pose.pose.position.x = point_list[0]
        target_pose.pose.position.y = point_list[1]
        target_pose.pose.position.z = point_list[2]
        
        target_pose.pose.orientation.x = 1.0
        target_pose.pose.orientation.y = 0.0
        target_pose.pose.orientation.z = 0.0
        target_pose.pose.orientation.w = 0.0

        # 1. HOVER
        hover_pose = copy.deepcopy(target_pose.pose)
        hover_pose.position.z += self.HOVER_HEIGHT
        
        if not self.move_to_pose(hover_pose):
            self.get_logger().warn("   Skipping (Unreachable Hover)")
            return

        # 2. DIVE
        tie_pose = copy.deepcopy(target_pose.pose)
        tie_pose.position.z += self.TIE_HEIGHT
        
        if abs(self.TIE_HEIGHT - self.HOVER_HEIGHT) > 0.01:
            if not self.move_to_pose(tie_pose):
                self.get_logger().warn("   Dive Failed. Resetting.")
                self.move_to_pose(hover_pose)
                return

        # --- 3. GRIPPER LOGIC (Your Requested Code) ---
        self.get_logger().info(">> Closing Gripper")
        self.send_named_goal_fire_and_forget(self.hand_group, "close")
        time.sleep(3.0)
        
        self.get_logger().info("   (Tying...)")
        time.sleep(2.0)
        
        self.get_logger().info(">> Opening Gripper")
        self.send_named_goal_fire_and_forget(self.hand_group, "open")
        time.sleep(5.0)
        # -----------------------------------------------
        
        # 4. RETREAT
        self.move_to_pose(hover_pose)

    # ================= HELPERS =================

    def send_named_goal_fire_and_forget(self, group_name, target_name):
        """Builds a joint constraint manually for Open/Close and sends without waiting"""
        goal = MoveGroup.Goal()
        goal.request.group_name = group_name
        
        # "close" = 0.0, "open" = 0.035
        val = 0.0 if target_name == "close" else 0.035
        
        j = JointConstraint()
        j.joint_name = "panda_finger_joint1" # Ensure this matches your robot
        j.position = val
        j.tolerance_above = 0.01
        j.tolerance_below = 0.01
        j.weight = 1.0
        
        goal.request.goal_constraints.append(Constraints(joint_constraints=[j]))
        
        # Fire and Forget (Async)
        self._action_client.send_goal_async(goal)

    def calculate_dynamic_timeout(self, target_pose):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_link, rclpy.time.Time())
            curr = trans.transform.translation
            dist = math.sqrt(
                (target_pose.position.x - curr.x)**2 + 
                (target_pose.position.y - curr.y)**2 + 
                (target_pose.position.z - curr.z)**2
            )
            estimated_time = (dist / 0.15) + 9.0
            timeout = max(5.0, min(estimated_time, 25.0))
            self.get_logger().info(f"   [CALC] Dist: {dist:.2f}m -> Timeout: {timeout:.1f}s")
            return timeout
        except Exception:
            return 10.0

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
        return self.send_goal_and_wait(goal, timeout_sec=15.0)

    def move_to_pose(self, pose):
        dyn_timeout = self.calculate_dynamic_timeout(pose)
        
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
        return self.send_goal_and_wait(goal, timeout_sec=dyn_timeout)

    def send_goal_and_wait(self, goal, timeout_sec=15.0):
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
            if time.time() - start_time > timeout_sec: 
                self.get_logger().warn(f"   [TIMEOUT] Reached limit of {timeout_sec:.1f}s. Proceeding.")
                return True 
            time.sleep(0.05)
        return res_future.result().result.error_code.val == 1

    def add_safety_floor_from_points(self, points):
        if not points: return
        z_values = [p[2] for p in points]
        avg_z = sum(z_values) / len(z_values)
        table_z = avg_z - 0.10
        
        self.get_logger().info(f"   [SAFETY] Adding Virtual Table at Z={table_z:.3f}")
        
        co = CollisionObject()
        co.header.frame_id = self.base_frame
        co.id = "safety_table"
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2.0, 2.0, 0.1]
        pose = Pose()
        pose.position.x = 0.5
        pose.position.z = table_z
        pose.orientation.w = 1.0
        co.primitives.append(box)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        for _ in range(5):
            self.collision_pub.publish(co)
            time.sleep(0.1)

    def process_and_sort_points(self, raw_points):
        if not raw_points: return []
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
        unique_points.sort(key=lambda p: (round(p[0] * 10), p[1]))
        return unique_points

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

    def points_callback(self, msg):
        if not self.is_scanning or len(msg.markers) == 0:
            return
        for marker in msg.markers:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame, marker.header.frame_id, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                pose_stamped = PoseStamped()
                pose_stamped.header = marker.header
                pose_stamped.pose = marker.pose
                pose_base = tf2_geometry_msgs.do_transform_pose(pose_stamped.pose, transform)
                p = pose_base.position
                self.detected_points.append([p.x, p.y, p.z])
            except Exception:
                continue

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
