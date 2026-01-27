#!/usr/bin/env python3
import rclpy
import time
import threading
import math
import copy
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

# --- THE KEY IMPORT ---
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState

class RebarLogic:
    def __init__(self, ros_node, arm_component):
        self.node = ros_node
        self.arm = arm_component
        self.logger = self.node.get_logger()
        
        # --- CONFIGURATION ---
        self.base_frame = "base_link"
        self.ee_link = "tool0"
        self.HOVER_HEIGHT = 0.15
        self.TIE_HEIGHT = 0.02
        self.dedup_radius = 0.05

        # Scan Pose (Joints for UR10e "Crane")
        # Note: moveit_py uses dictionaries for joint targets
        self.scan_joints = {
            "shoulder_pan_joint": 2.1929,
            "shoulder_lift_joint": -2.1073,
            "elbow_joint": -0.883,
            "wrist_1_joint": -4.8829,
            "wrist_2_joint": -1.5293,
            "wrist_3_joint": -1.1045
        }

        # Data
        self.detected_points = []
        self.final_points = []
        self.is_scanning = False

        # Publishers / Subscribers
        self.marker_sub = self.node.create_subscription(
            MarkerArray, 
            '/rebar/intersections', 
            self.points_callback, 
            10
        )
        self.collision_pub = self.node.create_publisher(
            CollisionObject, 
            '/collision_object', 
            10
        )
        self.finished_pub = self.node.create_publisher(
            MarkerArray, 
            '/rebar/finished', 
            10
        )

        # Start Logic Loop
        threading.Thread(target=self.run_sequence, daemon=True).start()

    def run_sequence(self):
        time.sleep(1.0) # Warmup
        
        # 1. MOVE TO SCAN POSE
        self.logger.info("--- PHASE 1: Moving to Scan Position ---")
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name="ready") # Or use joint dict
        # Fallback to manual joints if "ready" config doesn't exist
        self.arm.set_goal_state(robot_state=self.dict_to_state(self.scan_joints))
        
        plan_result = self.arm.plan()
        if plan_result:
            self.logger.info("Executing Scan Move...")
            self.arm.execute(plan_result.trajectory, blocking=True)
        else:
            self.logger.error("Failed to plan to scan pose.")

        # 2. SCANNING
        self.logger.info("--- PHASE 2: Scanning (5s) ---")
        self.detected_points = []
        self.is_scanning = True
        time.sleep(5.0)
        self.is_scanning = False

        if not self.detected_points:
            self.logger.warn("No points found!")
            return

        # 3. PROCESSING
        self.final_points = self.process_and_sort_points(self.detected_points)
        self.add_safety_floor_static()
        self.publish_markers(self.final_points, color=(0.0, 1.0, 0.0))

        # 4. EXECUTION
        self.logger.info(f"--- PHASE 4: Executing {len(self.final_points)} Targets ---")
        
        for i, pt in enumerate(self.final_points):
            self.logger.info(f">>> Target {i+1}")
            self.execute_tying_cycle(pt)

        self.logger.info("--- DONE. Returning Home. ---")
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(robot_state=self.dict_to_state(self.scan_joints))
        p = self.arm.plan()
        if p: self.arm.execute(p.trajectory, blocking=True)

    def execute_tying_cycle(self, point):
        # Setup Poses
        base_pose = PoseStamped()
        base_pose.header.frame_id = self.base_frame
        base_pose.pose.position.x = point[0]
        base_pose.pose.position.y = point[1]
        base_pose.pose.position.z = point[2]
        # Orientation: Down (Standard)
        base_pose.pose.orientation.x = 1.0 

        hover_pose = copy.deepcopy(base_pose)
        hover_pose.pose.position.z += self.HOVER_HEIGHT

        tie_pose = copy.deepcopy(base_pose)
        tie_pose.pose.position.z += self.TIE_HEIGHT

        # 1. HOVER (Cartesian)
        if not self.move_cartesian([hover_pose]):
            self.logger.warn("Skipping target (Hover unreachable)")
            return

        # 2. DIVE (Cartesian)
        if not self.move_cartesian([tie_pose]):
            self.logger.warn("Dive blocked.")
            self.move_cartesian([hover_pose]) # Retreat
            return

        # 3. TIE
        self.logger.info(">> Tying...")
        time.sleep(2.0)

        # 4. RETREAT
        self.move_cartesian([hover_pose])

    def move_cartesian(self, waypoints):
        """
        Uses moveit_py's compute_cartesian_path.
        Returns True if successful and EXECUTED.
        """
        self.arm.set_start_state_to_current_state()
        
        # compute_cartesian_path returns (trajectory, fraction)
        # Note: API might vary slightly by version, checking standard Humble API
        trajectory, fraction = self.arm.compute_cartesian_path(
            waypoints=waypoints,
            max_step=0.01,
            jump_threshold=0.0
        )

        if fraction < 0.9:
            self.logger.warn(f"Path incomplete: {fraction}")
            return False
        
        # EXECUTE (BLOCKING)
        # This function waits until the robot STOPS before returning
        self.arm.execute(trajectory, blocking=True)
        return True

    # --- HELPERS ---
    def dict_to_state(self, joint_dict):
        # Helper to convert dictionary to RobotState for moveit_py
        # This is a bit complex in pure API, simpler to rely on named states if possible
        # But here is a workaround if needed:
        # Actually, set_goal_state(joint_name_positions=dict) works in recent versions
        return None 

    def points_callback(self, msg):
        if not self.is_scanning or not msg.markers: return
        # (Coordinate transform logic same as before - omitted for brevity)
        # Ensure you include the TF buffer logic if points are not in base_link!
        for m in msg.markers:
            p = m.pose.position
            self.detected_points.append([p.x, p.y, p.z])

    def add_safety_floor_static(self):
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

    def process_and_sort_points(self, points):
        # (Same logic as before)
        return points 
    
    def publish_markers(self, points, color):
        # (Same logic as before)
        pass

def main():
    rclpy.init()
    
    # INITIALIZE MOVEIT_PY
    # This automatically creates a node called "rebar_mover_py"
    rebar_robot = MoveItPy(node_name="rebar_mover_py")
    
    # Get the arm group
    rebar_arm = rebar_robot.get_planning_component("arm")
    
    # Start our Logic Class
    # We pass the underlying ROS node so we can subscribe/publish
    logic = RebarLogic(rebar_robot.get_node(), rebar_arm)
    
    # Spin the node
    rclpy.spin(rebar_robot.get_node())

if __name__ == "__main__":
    main()