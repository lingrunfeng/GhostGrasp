from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_host = LaunchConfiguration("robot_host")
    robot_user = LaunchConfiguration("robot_user")
    serial_port = LaunchConfiguration("serial_port")
    baud = LaunchConfiguration("baud")
    sample_hz = LaunchConfiguration("sample_hz")
    connect_timeout_s = LaunchConfiguration("connect_timeout_s")
    ssh_command = LaunchConfiguration("ssh_command")
    publish_shadow_gripper_joints = LaunchConfiguration("publish_shadow_gripper_joints")
    shadow_gripper_position = LaunchConfiguration("shadow_gripper_position")

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_host", default_value="10.42.0.169"),
            DeclareLaunchArgument("robot_user", default_value="elephant"),
            DeclareLaunchArgument("serial_port", default_value="/dev/ttyAMA0"),
            DeclareLaunchArgument("baud", default_value="1000000"),
            DeclareLaunchArgument("sample_hz", default_value="10.0"),
            DeclareLaunchArgument("connect_timeout_s", default_value="5"),
            DeclareLaunchArgument("ssh_command", default_value="ssh"),
            DeclareLaunchArgument("publish_shadow_gripper_joints", default_value="true"),
            DeclareLaunchArgument("shadow_gripper_position", default_value="0.15"),
            Node(
                package="ghost_mgg_real",
                executable="m6_ssh_joint_state_bridge.py",
                name="m6_ssh_joint_state_bridge",
                output="screen",
                parameters=[
                    {
                        "robot_host": robot_host,
                        "robot_user": robot_user,
                        "serial_port": serial_port,
                        "baud": baud,
                        "sample_hz": sample_hz,
                        "connect_timeout_s": connect_timeout_s,
                        "ssh_command": ssh_command,
                        "publish_shadow_gripper_joints": publish_shadow_gripper_joints,
                        "shadow_gripper_position": shadow_gripper_position,
                    }
                ],
            ),
        ]
    )
