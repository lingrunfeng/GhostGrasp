from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
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

    return LaunchDescription(
        [
            visual_scene,
            TimerAction(period=5.0, actions=[moveit_rviz]),
        ]
    )
