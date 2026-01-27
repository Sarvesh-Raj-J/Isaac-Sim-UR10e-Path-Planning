#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
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
import copy # Import copy to fix the memory bug

class RebarMover(Node):
    def __init__(self):
        super().__init__('rebar_mover')
        
        # --- CONFIGURATION ---
        self.arm_group = "panda_arm"
        self.hand_group = "hand" 
        
        self.work_height = 0.15        
        self.scan_height = 0.50        
        self.visited_radius = 0.10     # INCREASED to 10cm to stop duplicate hits
        self.move_time_limit = 4.0     
        self.idle_timeout = 8.0        
        
        # Memory & State
        self.visited_points = []       
        self.is_busy = True            
        self.homing_done = False       
        self.last_detection_time = time.time()
        
        # Scanning State
        self.scan_index = 0
        self.scan_names = ["scan_center", "scan_left", "scan_right"]

        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Action Client
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.get_logger().info("Waiting for MoveIt Server...")
        self._action_client.wait_for_server()

        # Publishers & Subscribers
        self.subscription = self.create_subscription(
            MarkerArray, '/rebar/intersections', self.points_callback, 10)
        
        self.finished_pub = self.create_publisher(MarkerArray, '/rebar/finished', 10)
        
        self.create_timer(1.0, self.check_idle_status)
        
        self.get_logger().info("Rebar Mover V10 (Sticky Memory) Initialized.")
        
        # Launch Startup
        threading.Thread(target=self.startup_sequence).start()

    def startup_sequence(self):
        self.get_logger().info("--- STARTUP: Moving to Center Scan Position ---")
        self.send_named_goal_fire_and_forget(self.arm_group, "scan_center")
        time.sleep(5.0) 
        self.get_logger().info("--- STARTUP COMPLETE: Scanning... ---")
        self.homing_done = True
        self.is_busy = False 
        self.last_detection_time = time.time()

    def check_idle_status(self):
        if self.is_busy or not self.homing_done:
            return

        elapsed = time.time() - self.last_detection_time
        
        if elapsed > self.idle_timeout:
            self.is_busy = True 
            self.scan_index = (self.scan_index + 1) % len(self.scan_names)
            next_view = self.scan_names[self.scan_index]
            self.get_logger().warn(f"No targets found for {elapsed:.0f}s. Switching View -> {next_view}")
            threading.Thread(target=self.execute_scan_move, args=(next_view,)).start()

    def execute_scan_move(self, view_name):
        self.send_named_goal_fire_and_forget(self.arm_group, view_name)
        time.sleep(5.0) 
        self.get_logger().info(f"Arrived at {view_name}. Looking for points...")
        self.last_detection_time = time.time() 
        self.is_busy = False

    def points_callback(self, msg):
        if self.is_busy or not self.homing_done or len(msg.markers) == 0:
            return

        target_pose_base = None
        
        for marker in msg.markers:
            try:
                p = PoseStamped()
                p.header = marker.header
                p.pose = marker.pose
                
                transform = self.tf_buffer.lookup_transform(
                    "panda_link0", p.header.frame_id, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                pose_base = tf2_geometry_msgs.do_transform_pose(p.pose, transform)
                
                # Check with INCREASED radius
                if not self.is_visited(pose_base.position):
                    target_pose_base = pose_base
                    break 
            except Exception as e:
                continue

        if target_pose_base:
            self.last_detection_time = time.time()
            self.is_busy = True
            threading.Thread(target=self.execute_sequence_thread, args=(target_pose_base,)).start()

    def execute_sequence_thread(self, pose):
        # FIX: Create a clean copy of the target point BEFORE we modify 'pose'
        # This ensures our memory stays accurate to where the rebar actually is
        memory_point = Point()
        memory_point.x = pose.position.x
        memory_point.y = pose.position.y
        memory_point.z = pose.position.z

        pose.orientation.x = 1.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 0.0

        # 1. DESCEND
        pose.position.z += self.work_height
        self.get_logger().info(f">> Found Target! Descending...")
        self.send_move_goal_fire_and_forget(self.arm_group, pose)
        time.sleep(self.move_time_limit) 
        
        # 2. CLOSE
        self.get_logger().info(">> Closing Gripper")
        self.send_named_goal_fire_and_forget(self.hand_group, "close")
        time.sleep(1.0)
        
        # 3. TIE
        self.get_logger().info("   (Tying...)")
        time.sleep(2.0)
        
        # 4. OPEN
        self.get_logger().info(">> Opening Gripper")
        self.send_named_goal_fire_and_forget(self.hand_group, "open")
        time.sleep(1.0)
        
        # 5. RETREAT & MARK
        self.get_logger().info(">> Done. Retracting...")
        
        # Add the UNMODIFIED point to memory
        self.visited_points.append(memory_point)
        self.publish_finished_markers()
        
        pose.position.z = self.scan_height 
        self.send_move_goal_fire_and_forget(self.arm_group, pose)
        time.sleep(3.0) 
        
        self.last_detection_time = time.time()
        self.is_busy = False 

    def publish_finished_markers(self):
        marker_array = MarkerArray()
        for i, point in enumerate(self.visited_points):
            marker = Marker()
            marker.header.frame_id = "panda_link0"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "finished_rebar"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = point
            marker.scale.x = 0.08  # Make red markers bigger
            marker.scale.y = 0.08
            marker.scale.z = 0.08
            marker.color.a = 1.0
            marker.color.r = 1.0 
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.finished_pub.publish(marker_array)

    def is_visited(self, point):
        for v in self.visited_points:
            # Check 2D distance
            dist = math.sqrt((v.x - point.x)**2 + (v.y - point.y)**2)
            if dist < self.visited_radius:
                return True
        return False

    def send_move_goal_fire_and_forget(self, group_name, pose):
        goal = MoveGroup.Goal()
        goal.request.group_name = group_name
        goal.request.allowed_planning_time = 2.0
        
        pc = PositionConstraint()
        pc.header.frame_id = "panda_link0"
        pc.link_name = "panda_hand"
        pc.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.05]))
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = "panda_link0"
        oc.link_name = "panda_hand"
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.5 
        oc.absolute_y_axis_tolerance = 0.5
        oc.absolute_z_axis_tolerance = 3.14 
        oc.weight = 1.0

        goal.request.goal_constraints.append(Constraints(position_constraints=[pc], orientation_constraints=[oc]))
        self._action_client.send_goal_async(goal)

    def send_named_goal_fire_and_forget(self, group_name, target_name):
        goal = MoveGroup.Goal()
        goal.request.group_name = group_name
        
        target_joints = []
        if target_name == "scan_center":
            target_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        elif target_name == "scan_left":
            target_joints = [0.8, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        elif target_name == "scan_right":
            target_joints = [-0.8, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

        if target_joints:
            joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
            constraints = Constraints()
            for i in range(7):
                jc = JointConstraint()
                jc.joint_name = joint_names[i]
                jc.position = target_joints[i]
                jc.tolerance_above = 0.05
                jc.tolerance_below = 0.05
                jc.weight = 1.0
                constraints.joint_constraints.append(jc)
            goal.request.goal_constraints.append(constraints)
        else:
            val = 0.0 if target_name == "close" else 0.035
            j = JointConstraint()
            j.joint_name = "panda_finger_joint1"
            j.position = val
            j.tolerance_above = 0.01
            j.tolerance_below = 0.01
            j.weight = 1.0
            goal.request.goal_constraints.append(Constraints(joint_constraints=[j]))

        self._action_client.send_goal_async(goal)

def main(args=None):
    rclpy.init(args=args)
    node = RebarMover()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()

if __name__ == '__main__':
    main()