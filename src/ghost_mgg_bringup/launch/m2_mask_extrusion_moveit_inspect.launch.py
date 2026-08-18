from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    preferred_hypothesis_id = LaunchConfiguration('preferred_hypothesis_id')
    moveit_loop = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('ghost_mgg_bringup'),
                    'launch',
                    'm2_mask_extrusion_moveit_loop.launch.py',
                ]
            )
        ),
        launch_arguments={
            'headless': 'false',
            'preferred_hypothesis_id': preferred_hypothesis_id,
        }.items(),
    )

    rviz_config = PathJoinSubstitution(
        [
            FindPackageShare('ghost_mgg_sim'),
            'rviz',
            'm2_camera_inspect.rviz',
        ]
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='m2_mask_extrusion_moveit_inspect_rviz',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('preferred_hypothesis_id', default_value=''),
            moveit_loop,
            rviz,
        ]
    )
