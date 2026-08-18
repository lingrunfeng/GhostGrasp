from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera_namespace = LaunchConfiguration('camera_namespace')
    camera_name = LaunchConfiguration('camera_name')
    serial_no = LaunchConfiguration('serial_no')
    depth_profile = LaunchConfiguration('depth_profile')
    color_profile = LaunchConfiguration('color_profile')
    infra_profile = LaunchConfiguration('infra_profile')

    d435_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('ghost_mgg_real'),
                    'launch',
                    'd435_realsense.launch.py',
                ]
            )
        ),
        launch_arguments={
            'camera_namespace': camera_namespace,
            'camera_name': camera_name,
            'serial_no': serial_no,
            'depth_profile': depth_profile,
            'color_profile': color_profile,
            'infra_profile': infra_profile,
            'enable_infra': 'true',
            'enable_pointcloud': 'true',
            'enable_align_depth': 'true',
            'initial_reset': 'true',
        }.items(),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='ghost_mgg_d435_realsense_rviz',
        arguments=[
            '-d',
            PathJoinSubstitution(
                [
                    FindPackageShare('ghost_mgg_real'),
                    'rviz',
                    'd435_realsense.rviz',
                ]
            ),
        ],
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('camera_namespace', default_value='camera'),
            DeclareLaunchArgument('camera_name', default_value='camera'),
            DeclareLaunchArgument('serial_no', default_value=''),
            DeclareLaunchArgument('depth_profile', default_value='1280x720x30'),
            DeclareLaunchArgument('color_profile', default_value='1280x720x30'),
            DeclareLaunchArgument('infra_profile', default_value='1280x720x30'),
            d435_launch,
            rviz,
        ]
    )
