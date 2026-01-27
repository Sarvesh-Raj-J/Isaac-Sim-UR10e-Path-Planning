from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 1. Build the Configuration
    #moveit_config = MoveItConfigsBuilder("ur10e_robot", package_name="moveit_ur10").to_moveit_configs()
    # 1. Build the Configuration
    # Add .sensors_3d() to automatically load your sensors_3d.yaml file correctly
    moveit_config = (
        MoveItConfigsBuilder("ur10e_robot", package_name="moveit_ur10")
        .robot_description(file_path="config/ur10e_robot.urdf.xacro")
        .robot_description_semantic(file_path="config/ur10e_robot.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .sensors_3d(file_path="config/sensors_3d.yaml")  # <--- THIS IS THE CRITICAL LINE
        .to_moveit_configs()
    )
    # 2. Get the path to ros2_controllers.yaml explicitly
    # Note: Check if your file is named 'ros2_controllers.yaml' or just 'controllers.yaml' in your config folder!
    ros2_controllers_path = str(moveit_config.package_path / "config/ros2_controllers.yaml")

    # 3. Define the Nodes
    
    # A. The "Brain" (Move Group)
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # B. The "Visualizer" (RViz)
    rviz_config_file = str(moveit_config.package_path / "config/moveit.rviz")
    
    run_rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
    )

    # C. The "Bridge" (ROS 2 Control Node)
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            ros2_controllers_path, # Loaded manually
        ],
        output="screen",
    )

    # D. The "Transform Publisher" (Robot State Publisher)
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # E. Spawn the Controllers
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    # CHECK: Ensure this matches the name in your ros2_controllers.yaml
    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
    )

    # F. Sequencing
    delay_rviz_after_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[run_rviz_node],
        )
    )

    return LaunchDescription([
        ros2_control_node,
        robot_state_publisher,
        run_move_group_node,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        delay_rviz_after_spawner, 
    ])