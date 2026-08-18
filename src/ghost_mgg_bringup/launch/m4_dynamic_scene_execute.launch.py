from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    headless = LaunchConfiguration('headless')
    show_rviz = LaunchConfiguration('show_rviz')
    dynamic_targets_path = LaunchConfiguration('dynamic_targets_path')

    execute_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('ghost_mgg_bringup'),
                    'launch',
                    'm4_joint_hypothesis_moveit_execute.launch.py',
                ]
            )
        ),
        launch_arguments={
            'headless': headless,
            'show_rviz': show_rviz,
            'sim_grasp_targets_path': dynamic_targets_path,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('headless', default_value='true'),
            DeclareLaunchArgument('show_rviz', default_value='false'),
            DeclareLaunchArgument(
                'dynamic_targets_path',
                default_value='reports/m4_scene_snapshot/current_targets.json',
            ),
            execute_launch,
        ]
    )
