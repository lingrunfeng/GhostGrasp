from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    description_share = FindPackageShare('ghost_mgg_mycobot_description')
    mycobot_xacro = PathJoinSubstitution(
        [
            description_share,
            'urdf',
            'robots',
            'mycobot_280.urdf.xacro',
        ]
    )
    robot_description = Command(
        [
            'xacro',
            ' ',
            mycobot_xacro,
            ' ',
            'add_world:=false',
            ' ',
            'use_camera:=false',
            ' ',
            'use_gazebo:=true',
            ' ',
            'use_gripper:=true',
        ]
    )

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

    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='m2_camera_bridge',
        arguments=[
            '/ghost_mgg/d435/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/ghost_mgg/d435/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/ghost_mgg/d435/depth/image_rect_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/ghost_mgg/d435/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
    )

    robot_description_publisher = Node(
        package='ghost_mgg_sim',
        executable='robot_description_publisher_node',
        name='robot_description_publisher_node',
        parameters=[
            {
                'use_sim_time': True,
                'robot_description': robot_description,
            }
        ],
        output='screen',
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

    world_to_d435_link_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_world_to_d435_link_tf',
        arguments=[
            '--x', '0.286775',
            '--y', '-0.079936',
            '--z', '1.105590',
            '--roll', '0.0',
            '--pitch', '0.950000',
            '--yaw', '2.754700',
            '--frame-id', 'world',
            '--child-frame-id', 'd435_link',
        ],
    )

    d435_link_to_color_frame_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_d435_link_to_color_frame_tf',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0',
            '--frame-id', 'd435_link',
            '--child-frame-id', 'd435_color_frame',
        ],
    )

    d435_link_to_depth_frame_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_d435_link_to_depth_frame_tf',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0',
            '--frame-id', 'd435_link',
            '--child-frame-id', 'd435_depth_frame',
        ],
    )

    color_frame_to_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_d435_color_frame_to_optical_tf',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '-1.5708',
            '--pitch', '0.0',
            '--yaw', '-1.5708',
            '--frame-id', 'd435_color_frame',
            '--child-frame-id', 'd435_color_optical_frame',
        ],
    )

    depth_frame_to_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_d435_depth_frame_to_optical_tf',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '-1.5708',
            '--pitch', '0.0',
            '--yaw', '-1.5708',
            '--frame-id', 'd435_depth_frame',
            '--child-frame-id', 'd435_depth_optical_frame',
        ],
    )

    depth_to_points = Node(
        package='ghost_mgg_sim',
        executable='depth_to_point_cloud_node',
        name='depth_to_point_cloud_node',
        parameters=[
            {
                'use_sim_time': True,
                'depth_topic': '/ghost_mgg/d435/depth/image_rect_raw',
                'camera_info_topic': '/ghost_mgg/d435/depth/camera_info',
                'points_topic': '/ghost_mgg/d435/depth/points',
                'output_frame': 'd435_depth_optical_frame',
            }
        ],
        output='screen',
    )

    depth_preview = Node(
        package='ghost_mgg_sim',
        executable='depth_to_mono8_node',
        name='m2_depth_to_mono8_node',
        parameters=[
            {
                'use_sim_time': True,
                'depth_topic': '/ghost_mgg/d435/depth/image_rect_raw',
                'preview_topic': '/ghost_mgg/d435/depth/image_viz',
                'min_depth_m': 0.2,
                'max_depth_m': 1.4,
            }
        ],
        output='screen',
    )

    observation_cache = Node(
        package='ghost_mgg_sim',
        executable='observation_cache_node',
        name='m2_observation_cache_node',
        parameters=[
            {
                'use_sim_time': True,
                'cache_namespace': 'm2_d435',
                'observation_ref_topic': '/ghost_mgg/observations/latest',
                'rgb_topic': '/ghost_mgg/d435/color/image_raw',
                'depth_topic': '/ghost_mgg/d435/depth/image_rect_raw',
                'color_camera_info_topic': '/ghost_mgg/d435/color/camera_info',
                'depth_camera_info_topic': '/ghost_mgg/d435/depth/camera_info',
                'rgb_frame_id': 'd435_color_optical_frame',
                'depth_frame_id': 'd435_depth_optical_frame',
                'mask_frame_id': 'd435_color_optical_frame',
                'max_age_sec': 1.0,
            }
        ],
        output='screen',
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
        name='m2_camera_inspect_rviz',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription(
        [
            visual_scene,
            camera_bridge,
            robot_description_publisher,
            world_to_base_tf,
            world_to_d435_link_tf,
            d435_link_to_color_frame_tf,
            d435_link_to_depth_frame_tf,
            color_frame_to_optical_tf,
            depth_frame_to_optical_tf,
            depth_to_points,
            depth_preview,
            observation_cache,
            rviz,
        ]
    )
