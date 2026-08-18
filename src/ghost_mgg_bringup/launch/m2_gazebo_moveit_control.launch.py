from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    visual_scene = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('ghost_mgg_sim'),
                    'launch',
                    'm2_visual_scene.launch.py',
                ]
            )
        )
    )

    world_to_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_world_to_base_link_tf',
        arguments=[
            '--x', '-0.171207',
            '--y', '0.228790',
            '--z', '0.765000',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '-2.180640',
            '--frame-id', 'world',
            '--child-frame-id', 'base_link',
        ],
    )

    moveit_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('ghost_mgg_moveit_config'),
                    'launch',
                    'm2_moveit_rviz.launch.py',
                ]
            )
        )
    )

    # Controllers spawn at 6s and the pregrasp command runs at 10s, so
    # MoveIt/RViz starts after the arm state is stable.
    return LaunchDescription(
        [
            visual_scene,
            world_to_base_tf,
            TimerAction(period=13.0, actions=[moveit_rviz]),
        ]
    )
