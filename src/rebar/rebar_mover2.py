#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
import numpy as np
import time
import threading

class RebarTyingController(Node):
    def __init__(self):
        super().__init__('rebar_tying_controller')

        # --- CONFIGURATION ---
        self.planning_group = "panda_arm"
        self.base_frame = "panda_link0"
        self.ee_frame = "panda_hand"
        
        # NOTE: Ensure these match your robot's actual joint names!
        self.joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", 
                            "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]

        self.survey_joint_values = [
            0.2409,   # panda_joint1
            0.3304,   # panda_joint2
            -0.3162,  # panda_joint3
            -0.5634,  # panda_joint4
            0.1061,   # panda_joint5
            0.8759,   # panda_joint6
            0.8945    # panda_joint7
        ]
        self.APPROACH_HEIGHT = 0.15 
        self.TYING_HEIGHT = 0.02    

        self.target_points = []
        self.scan_complete = False

        # --- MULTI-THREADING SETUP ---
        # Use ReentrantCallbackGroup to allow callbacks to run in parallel
        self.cb_group = ReentrantCallbackGroup()

        # --- SUBSCRIBERS ---
        self.subscription = self.create_subscription(
            MarkerArray, 
            '/rebar/intersections', 
            self.rebar_callback, 
            10,
            callback_group=self.cb_group) # Add to callback group

        # --- MOVEIT ACTION CLIENT ---
        self._action_client = ActionClient(
            self, 
            MoveGroup, 
            'move_action', 
            callback_group=self.cb_group) # Add to callback group
            
        self.get_logger().info('Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('MoveGroup Action Server Found!')

    def start_routine(self):
        """Runs on a separate thread to avoid blocking ROS callbacks"""
        self.get_logger().info("1. Moving to SURVEY POSE...")
        if not self.move_to_joints(self.survey_joint_values):
            self.get_logger().error("Failed to reach survey pose. Check Controllers!")
            return

        self.get_logger().info("2. Waiting for Rebar Detection...")
        while rclpy.ok() and not self.scan_complete:
            time.sleep(0.5) # Use time.sleep instead of spin_once

        self.get_logger().info(f"3. Detection Complete. Found {len(self.target_points)} points.")
        sorted_points = self.optimize_path(self.target_points)

        self.get_logger().info("4. Starting Tying Sequence...")
        for i, point in enumerate(sorted_points):
            self.get_logger().info(f"   --- Tying Rebar {i+1}/{len(sorted_points)} ---")
            self.perform_tying_action(point)

        self.get_logger().info("DONE! Returning to Home.")
        self.move_to_joints(self.survey_joint_values)

    def rebar_callback(self, msg):
        if self.scan_complete: return 
        if len(msg.markers) > 0:
            self.target_points = []
            for marker in msg.markers:
                p = marker.pose.position
                self.target_points.append([p.x, p.y, p.z])
            self.scan_complete = True
            self.get_logger().info("Points Acquired!")

    def optimize_path(self, points):
        pts = np.array(points)
        ind = np.lexsort((pts[:, 1], np.round(pts[:, 0], 2)))
        return pts[ind]

    def perform_tying_action(self, target_point):
        x, y, z = target_point
        
        # 1. Approach
        hover_pose = self.create_pose(x, y, z + self.APPROACH_HEIGHT)
        if not self.move_to_pose(hover_pose):
            self.get_logger().warn("Skipping point (Unreachable)")
            return

        # 2. Dive
        tie_pose = self.create_pose(x, y, z + self.TYING_HEIGHT)
        self.move_to_pose(tie_pose)

        # 3. Tie (Simulated)
        self.get_logger().info("   [GRIPPER] Tying...")
        time.sleep(0.5)

        # 4. Retreat
        self.move_to_pose(hover_pose)

    def create_pose(self, x, y, z):
        p = PoseStamped()
        p.header.frame_id = self.base_frame
        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.position.z = z
        p.pose.orientation.x = 1.0
        p.pose.orientation.y = 0.0
        p.pose.orientation.z = 0.0
        p.pose.orientation.w = 0.0
        return p

    def move_to_joints(self, joint_values):
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.planning_group
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.3
        goal_msg.request.max_acceleration_scaling_factor = 0.3

        constraints = Constraints()
        for i, val in enumerate(joint_values):
            jc = JointConstraint()
            jc.joint_name = self.joint_names[i]
            jc.position = val
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal_msg.request.goal_constraints.append(constraints)
        
        return self.send_goal_and_wait(goal_msg)

    def move_to_pose(self, pose_stamped):
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.planning_group
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.4
        goal_msg.request.max_acceleration_scaling_factor = 0.4

        pc = PositionConstraint()
        pc.header = pose_stamped.header
        pc.link_name = self.ee_frame
        pc.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01]))
        pc.constraint_region.primitive_poses.append(pose_stamped.pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header = pose_stamped.header
        oc.link_name = self.ee_frame
        oc.orientation = pose_stamped.pose.orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)
        goal_msg.request.goal_constraints.append(constraints)

        return self.send_goal_and_wait(goal_msg)

    def send_goal_and_wait(self, goal_msg):
        self.get_logger().info("Sending Goal...")
        
        # Send goal
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        
        # Wait for Goal Acceptance (with timeout)
        while not send_goal_future.done():
            time.sleep(0.1)
            
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal Rejected!')
            return False

        self.get_logger().info("Goal Accepted. Moving...")
        
        # Wait for Execution Result
        get_result_future = goal_handle.get_result_async()
        
        while not get_result_future.done():
            time.sleep(0.1)
            
        result = get_result_future.result().result
        
        if result.error_code.val == 1:
            return True
        else:
            self.get_logger().error(f'Failed (Error: {result.error_code.val})')
            return False

def main(args=None):
    rclpy.init(args=args)
    node = RebarTyingController()
    
    # Use MultiThreadedExecutor so the Loop and the Subscriber run in parallel
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # Start the logic in a separate background thread
    logic_thread = threading.Thread(target=node.start_routine, daemon=True)
    logic_thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
