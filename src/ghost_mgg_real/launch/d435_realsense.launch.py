from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera_namespace = LaunchConfiguration('camera_namespace')
    camera_name = LaunchConfiguration('camera_name')
    serial_no = LaunchConfiguration('serial_no')
    depth_profile = LaunchConfiguration('depth_profile')
    color_profile = LaunchConfiguration('color_profile')
    infra_profile = LaunchConfiguration('infra_profile')
    enable_infra = LaunchConfiguration('enable_infra')
    enable_pointcloud = LaunchConfiguration('enable_pointcloud')
    enable_align_depth = LaunchConfiguration('enable_align_depth')
    initial_reset = LaunchConfiguration('initial_reset')

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('realsense2_camera'),
                    'launch',
                    'rs_launch.py',
                ]
            )
        ),
        launch_arguments={
            'camera_namespace': camera_namespace,
            'camera_name': camera_name,
            'serial_no': serial_no,
            'depth_module.depth_profile': depth_profile,
            'rgb_camera.color_profile': color_profile,
            'depth_module.infra_profile': infra_profile,
            'enable_infra1': enable_infra,
            'enable_infra2': enable_infra,
            'pointcloud.enable': enable_pointcloud,
            'align_depth.enable': enable_align_depth,
            'initial_reset': initial_reset,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('camera_namespace', default_value='camera'),
            DeclareLaunchArgument('camera_name', default_value='camera'),
            DeclareLaunchArgument('serial_no', default_value=''),
            DeclareLaunchArgument('depth_profile', default_value='1280x720x30'),
            DeclareLaunchArgument('color_profile', default_value='1280x720x30'),
            DeclareLaunchArgument('infra_profile', default_value='1280x720x30'),
            DeclareLaunchArgument('enable_infra', default_value='true'),
            DeclareLaunchArgument('enable_pointcloud', default_value='true'),
            DeclareLaunchArgument('enable_align_depth', default_value='true'),
            DeclareLaunchArgument('initial_reset', default_value='true'),
            realsense_launch,
        ]
    )
