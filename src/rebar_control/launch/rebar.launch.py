from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 1. LOAD ROBOT DESCRIPTION (UR10e)
    # CHANGE "ur10e_moveit_config" to your actual moveit config package name if different!
    moveit_config = MoveItConfigsBuilder("ur10e", package_name="moveit_ur10").to_moveit_configs()

    # 2. DEFINE THE NODE
    rebar_node = Node(
        package="rebar_control",
        executable="rebar_mover",
        output="screen",
        parameters=[
            moveit_config.to_dict(),  # <--- CRITICAL: Passes URDF/SRDF to moveit_py
            {"use_sim_time": True},   # Set to False if on real robot
        ],
    )

    return LaunchDescription([
        rebar_node
    ])