from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_camera = LaunchConfiguration("start_camera")
    start_control_server = LaunchConfiguration("start_control_server")
    start_moveit_shadow = LaunchConfiguration("start_moveit_shadow")
    show_m7_rviz = LaunchConfiguration("show_m7_rviz")
    camera_x = LaunchConfiguration("camera_x")
    camera_y = LaunchConfiguration("camera_y")
    camera_z = LaunchConfiguration("camera_z")
    camera_roll = LaunchConfiguration("camera_roll")
    camera_pitch = LaunchConfiguration("camera_pitch")
    camera_yaw = LaunchConfiguration("camera_yaw")
    robot_host = LaunchConfiguration("robot_host")
    robot_user = LaunchConfiguration("robot_user")
    ssh_command = LaunchConfiguration("ssh_command")
    target_path = LaunchConfiguration("target_path")

    d435_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ghost_mgg_real"), "launch", "d435_realsense.launch.py"]
            )
        ),
        launch_arguments={
            "depth_profile": LaunchConfiguration("depth_profile"),
            "color_profile": LaunchConfiguration("color_profile"),
            "infra_profile": LaunchConfiguration("infra_profile"),
            "enable_infra": "true",
            "enable_pointcloud": "true",
            "enable_align_depth": "true",
            "initial_reset": LaunchConfiguration("initial_reset"),
        }.items(),
        condition=IfCondition(start_camera),
    )

    control_server = Node(
        package="ghost_mgg_real",
        executable="m7_mycobot_control_server.py",
        name="m7_mycobot_control_server",
        output="screen",
        parameters=[
            {
                "robot_host": robot_host,
                "robot_user": robot_user,
                "serial_port": LaunchConfiguration("serial_port"),
                "baud": LaunchConfiguration("baud"),
                "sample_hz": LaunchConfiguration("sample_hz"),
                "connect_timeout_s": LaunchConfiguration("connect_timeout_s"),
                "ssh_command": ssh_command,
                "publish_shadow_gripper_joints": True,
                "shadow_gripper_position": 0.15,
            }
        ],
        condition=IfCondition(start_control_server),
    )

    moveit_shadow_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ghost_mgg_moveit_config"),
                    "launch",
                    "m6_shadow_move_group.launch.py",
                ]
            )
        ),
        launch_arguments={
            "show_rviz": "false",
            "use_fake_joint_states": "false",
        }.items(),
        condition=IfCondition(start_moveit_shadow),
    )

    rough_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="m7_rough_camera_to_base_tf",
        output="screen",
        arguments=[
            "--x",
            camera_x,
            "--y",
            camera_y,
            "--z",
            camera_z,
            "--roll",
            camera_roll,
            "--pitch",
            camera_pitch,
            "--yaw",
            camera_yaw,
            "--frame-id",
            "base_link",
            "--child-frame-id",
            "camera_link",
        ],
    )

    grasp_markers = Node(
        package="ghost_mgg_real",
        executable="m7_real_grasp_marker_node.py",
        name="m7_real_grasp_marker_node",
        output="screen",
        parameters=[{"target_path": target_path, "frame_id": "base_link"}],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="m7_real_control_rviz",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("ghost_mgg_real"), "rviz", "m7_real_control.rviz"]
            ),
        ],
        condition=IfCondition(show_m7_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument("start_control_server", default_value="true"),
            DeclareLaunchArgument("start_moveit_shadow", default_value="true"),
            DeclareLaunchArgument("show_m7_rviz", default_value="true"),
            DeclareLaunchArgument("robot_host", default_value="10.42.0.169"),
            DeclareLaunchArgument("robot_user", default_value="elephant"),
            DeclareLaunchArgument("serial_port", default_value="/dev/ttyAMA0"),
            DeclareLaunchArgument("baud", default_value="1000000"),
            DeclareLaunchArgument("sample_hz", default_value="10.0"),
            DeclareLaunchArgument("connect_timeout_s", default_value="5"),
            DeclareLaunchArgument("ssh_command", default_value="ssh -o BatchMode=yes"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
            DeclareLaunchArgument("color_profile", default_value="640x480x30"),
            DeclareLaunchArgument("infra_profile", default_value="640x480x30"),
            DeclareLaunchArgument("initial_reset", default_value="false"),
            DeclareLaunchArgument("camera_x", default_value="0.00"),
            DeclareLaunchArgument("camera_y", default_value="0.39"),
            DeclareLaunchArgument("camera_z", default_value="0.16"),
            DeclareLaunchArgument("camera_roll", default_value="0.0"),
            DeclareLaunchArgument("camera_pitch", default_value="0.83"),
            DeclareLaunchArgument("camera_yaw", default_value="-1.57"),
            DeclareLaunchArgument(
                "target_path",
                default_value=(
                    "reports/m6_shadow_grasp_targets/"
                    "m7_real_current/m6_shadow_grasp_target.json"
                ),
            ),
            d435_launch,
            control_server,
            moveit_shadow_launch,
            rough_camera_tf,
            grasp_markers,
            rviz,
        ]
    )
