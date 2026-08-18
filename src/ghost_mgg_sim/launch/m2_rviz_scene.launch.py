from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_share = FindPackageShare('ghost_mgg_sim')
    scene_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_share, 'launch', 'm2_visual_scene.launch.py'])
        )
    )

    marker_node = Node(
        package='ghost_mgg_sim',
        executable='m2_scene_markers_node',
        output='screen',
    )

    rviz_config = PathJoinSubstitution([sim_share, 'rviz', 'm2_scene.rviz'])
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    return LaunchDescription(
        [
            scene_launch,
            marker_node,
            rviz,
        ]
    )
