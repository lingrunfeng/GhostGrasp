from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_camera = LaunchConfiguration("start_camera")
    start_state_bridge = LaunchConfiguration("start_state_bridge")
    start_moveit_shadow = LaunchConfiguration("start_moveit_shadow")
    show_inspect_rviz = LaunchConfiguration("show_inspect_rviz")
    use_fake_joint_states = LaunchConfiguration("use_fake_joint_states")

    camera_x = LaunchConfiguration("camera_x")
    camera_y = LaunchConfiguration("camera_y")
    camera_z = LaunchConfiguration("camera_z")
    camera_roll = LaunchConfiguration("camera_roll")
    camera_pitch = LaunchConfiguration("camera_pitch")
    camera_yaw = LaunchConfiguration("camera_yaw")
    parent_frame = LaunchConfiguration("parent_frame")
    child_frame = LaunchConfiguration("child_frame")
    robot_host = LaunchConfiguration("robot_host")
    robot_user = LaunchConfiguration("robot_user")
    ssh_command = LaunchConfiguration("ssh_command")

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

    state_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ghost_mgg_real"),
                    "launch",
                    "m6_mycobot_state_bridge.launch.py",
                ]
            )
        ),
        launch_arguments={
            "robot_host": robot_host,
            "robot_user": robot_user,
            "ssh_command": ssh_command,
        }.items(),
        condition=IfCondition(start_state_bridge),
    )

    # m6_shadow_move_group.launch.py sets allow_trajectory_execution to False.
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
            "use_fake_joint_states": use_fake_joint_states,
        }.items(),
        condition=IfCondition(start_moveit_shadow),
    )

    rough_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="m6_rough_camera_to_base_tf",
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
            parent_frame,
            "--child-frame-id",
            child_frame,
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="m6_rough_tf_shadow_rviz",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("ghost_mgg_real"), "rviz", "m6_rough_tf_shadow.rviz"]
            ),
        ],
        condition=IfCondition(show_inspect_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument("start_state_bridge", default_value="true"),
            DeclareLaunchArgument("start_moveit_shadow", default_value="true"),
            DeclareLaunchArgument("show_inspect_rviz", default_value="true"),
            DeclareLaunchArgument("use_fake_joint_states", default_value="false"),
            DeclareLaunchArgument("robot_host", default_value="10.42.0.169"),
            DeclareLaunchArgument("robot_user", default_value="elephant"),
            DeclareLaunchArgument("ssh_command", default_value="ssh -o BatchMode=yes"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
            DeclareLaunchArgument("color_profile", default_value="640x480x30"),
            DeclareLaunchArgument("infra_profile", default_value="640x480x30"),
            DeclareLaunchArgument("initial_reset", default_value="false"),
            DeclareLaunchArgument("parent_frame", default_value="base_link"),
            DeclareLaunchArgument("child_frame", default_value="camera_link"),
            DeclareLaunchArgument("camera_x", default_value="0.00"),
            DeclareLaunchArgument("camera_y", default_value="0.39"),
            DeclareLaunchArgument("camera_z", default_value="0.16"),
            DeclareLaunchArgument("camera_roll", default_value="0.0"),
            DeclareLaunchArgument("camera_pitch", default_value="0.83"),
            DeclareLaunchArgument("camera_yaw", default_value="-1.57"),
            d435_launch,
            state_bridge_launch,
            moveit_shadow_launch,
            rough_camera_tf,
            rviz,
        ]
    )
