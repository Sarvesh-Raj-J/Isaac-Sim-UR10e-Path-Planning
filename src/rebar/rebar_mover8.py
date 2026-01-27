#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, CollisionObject, RobotState
from moveit_msgs.srv import GetCartesianPath
from shape_msgs.msg import SolidPrimitive
import tf2_ros
import tf2_geometry_msgs
import math
import time
import threading
import copy

class RebarMoverBatch(Node):
    def __init__(self):
        super().__init__('rebar_mover_batch')
        
        # --- CONFIGURATION ---
        self.arm_group = "arm"           
        self.base_frame = "base_link"    
        self.ee_link = "tool0"           

        self.HOVER_HEIGHT = 0.15    
        self.TIE_HEIGHT = 0.02      
        self.dedup_radius = 0.05    
        
        # Scan Pose (Crane)
        self.scan_poses = {
            "center": [2.1929, -2.1073, -0.883, -4.8829, -1.5293, -1.1045]
        }

        self.detected_points = []      
        self.final_points = []         
        self.is_scanning = False       
        
        self.cb_group = ReentrantCallbackGroup()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 1. MoveGroup Action (For big moves)
        self._action_client = ActionClient(self, MoveGroup, 'move_action', callback_group=self.cb_group)
        
        # 2. Cartesian Path Service (For straight lines)
        self._cartesian_client = self.create_client(GetCartesianPath, 'compute_cartesian_path', callback_group=self.cb_group)
        
        # 3. Execute Trajectory Action (To run the straight line)
        self._execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory', callback_group=self.cb_group)

        self.get_logger().info(">>> Waiting for MoveIt Servers...")
        self._action_client.wait_for_server()
        self._execute_client.wait_for_server()
        self.get_logger().info(">>> Connected to MoveIt.")

        self.subscription = self.create_subscription(MarkerArray, '/rebar/intersections', self.points_callback, 10, callback_group=self.cb_group)
        self.finished_pub = self.create_publisher(MarkerArray, '/rebar/finished', 10)
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        
        threading.Thread(target=self.logic_loop, daemon=True).start()

    def logic_loop(self):
        time.sleep(2.0) 
        
        # 1. SCAN POSE
        self.get_logger().info("--- PHASE 1: Moving to Scan Position ---")
        if not self.move_to_joints(self.scan_poses["center"]):
            self.get_logger().error("CRITICAL: Failed to reach Scan Pose.")
            # return 

        # 2. SCANNING
        self.get_logger().info("--- PHASE 2: Scanning... (Waiting 5s) ---")
        self.detected_points = [] 
        self.is_scanning = True   
        time.sleep(5.0)           
        self.is_scanning = False  
        
        if len(self.detected_points) == 0:
            self.get_logger().warn("NO POINTS FOUND!")
            return

        # 3. PROCESSING
        self.final_points = self.process_and_sort_points(self.detected_points)
        self.add_safety_floor_from_points(self.final_points)
        self.publish_markers(self.final_points, color=(0.0, 1.0, 0.0)) 

        # 4. EXECUTION
        self.get_logger().info(f"--- PHASE 4: Executing {len(self.final_points)} Targets ---")
        
        for i, pt in enumerate(self.final_points):
            self.get_logger().info(f">>> Target {i+1}")
            self.execute_tying_sequence(pt)
            
        self.get_logger().info("--- DONE. Returning Home. ---")
        self.move_to_joints(self.scan_poses["center"])

    def execute_tying_sequence(self, point_list):
        # We need the current orientation (Down) to ensure we don't twist
        try:
            current = self.tf_buffer.lookup_transform(self.base_frame, self.ee_link, rclpy.time.Time())
            current_quat = current.transform.rotation
        except:
            # Fallback to standard down
            current_quat = Pose().orientation
            current_quat.x = 1.0

        target_pose = Pose()
        target_pose.position.x = point_list[0]
        target_pose.position.y = point_list[1]
        target_pose.position.z = point_list[2]
        target_pose.orientation = current_quat # KEEP SAME ROTATION

        # 1. HOVER (Using Service)
        hover_pose = copy.deepcopy(target_pose)
        hover_pose.position.z += self.HOVER_HEIGHT
        
        if not self.move_linear_service(hover_pose):
            self.get_logger().warn("   Hover path blocked/failed.")
            return

        # 2. DIVE (Using Service)
        tie_pose = copy.deepcopy(target_pose)
        tie_pose.position.z += self.TIE_HEIGHT
        
        if not self.move_linear_service(tie_pose):
            self.get_logger().warn("   Dive failed.")
            self.move_linear_service(hover_pose)
            return

        # 3. ACTIVATE
        self.get_logger().info(">> Tying...")
        time.sleep(2.0) 
        
        # 4. RETREAT
        self.move_linear_service(hover_pose)

    # ================= REAL LINEAR MOVEMENT =================
    def move_linear_service(self, target_pose):
        """Calls the GetCartesianPath service for a true straight line."""
        
        # 1. Build Request
        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = self.arm_group
        req.waypoints = [target_pose]
        req.max_step = 0.01       # 1cm resolution
        req.jump_threshold = 0.0  # Disable jump check (or use 5.0)
        req.avoid_collisions = True

        # 2. Call Service
        future = self._cartesian_client.call_async(req)
        try:
            # Wait manually for service
            start = time.time()
            while not future.done():
                if time.time() - start > 2.0: 
                    self.get_logger().error("Cartesian Service Timeout")
                    return False
                time.sleep(0.01)
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Service Call Failed: {e}")
            return False

        # 3. Check Result
        if response.fraction < 0.90:
            self.get_logger().warn(f"   Path incomplete (Fraction: {response.fraction})")
            return False

        # 4. Execute Trajectory
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        
        exec_future = self._execute_client.send_goal_async(goal)
        
        # Wait for acceptance
        start = time.time()
        while not exec_future.done():
            if time.time() - start > 2.0: return False
            time.sleep(0.01)
        
        goal_handle = exec_future.result()
        if not goal_handle.accepted: return False
        
        # Wait for completion
        res_future = goal_handle.get_result_async()
        start = time.time()
        while not res_future.done():
            if time.time() - start > 5.0: return False
            time.sleep(0.05)
            
        return res_future.result().result.error_code.val == 1

    # ================= HELPERS =================
    def move_to_joints(self, joint_values):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.arm_group
        goal.request.allowed_planning_time = 5.0
        
        constraints = Constraints()
        joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        for i, val in enumerate(joint_values):
            jc = JointConstraint()
            jc.joint_name = joint_names[i]
            jc.position = val
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)
        
        future = self._action_client.send_goal_async(goal)
        while not future.done(): time.sleep(0.1)
        goal_handle = future.result()
        if not goal_handle.accepted: return False
        res_future = goal_handle.get_result_async()
        while not res_future.done(): time.sleep(0.1)
        return res_future.result().result.error_code.val == 1

    def add_safety_floor_from_points(self, points):
        co = CollisionObject()
        co.header.frame_id = self.base_frame
        co.id = "safety_table"
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2.0, 2.0, 0.05] 
        pose = Pose()
        pose.position.x = 0.5
        pose.position.z = -0.026 
        pose.orientation.w = 1.0
        co.primitives.append(box)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        for _ in range(3):
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