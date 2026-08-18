from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    headless = LaunchConfiguration('headless')
    show_rviz = LaunchConfiguration('show_rviz')
    world_file = LaunchConfiguration('world_file')
    world_name = LaunchConfiguration('world_name')
    scenario_id = LaunchConfiguration('scenario_id')
    failure_mode = LaunchConfiguration('failure_mode')
    roi_center_u_ratio = LaunchConfiguration('roi_center_u_ratio')
    roi_center_v_ratio = LaunchConfiguration('roi_center_v_ratio')
    roi_width_ratio = LaunchConfiguration('roi_width_ratio')
    roi_height_ratio = LaunchConfiguration('roi_height_ratio')
    table_leak_depth_m = LaunchConfiguration('table_leak_depth_m')
    flying_point_offset_m = LaunchConfiguration('flying_point_offset_m')
    biased_depth_offset_m = LaunchConfiguration('biased_depth_offset_m')
    edge_band_pixels = LaunchConfiguration('edge_band_pixels')
    flying_point_stride = LaunchConfiguration('flying_point_stride')
    point_cloud_stride = LaunchConfiguration('point_cloud_stride')
    pattern_seed = LaunchConfiguration('pattern_seed')
    enable_target_mask_emulator = LaunchConfiguration('enable_target_mask_emulator')

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
        ),
        launch_arguments={
            'headless': headless,
            'world_file': world_file,
            'world_name': world_name,
        }.items(),
    )

    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='m3_camera_bridge',
        arguments=[
            '/ghost_mgg/d435/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/ghost_mgg/d435/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/ghost_mgg/d435/depth/image_rect_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/ghost_mgg/d435/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/ghost_mgg/d435/infra1/image_rect_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/ghost_mgg/d435/infra1/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/ghost_mgg/d435/infra2/image_rect_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/ghost_mgg/d435/infra2/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
    )

    robot_description_publisher = Node(
        package='ghost_mgg_sim',
        executable='robot_description_publisher_node',
        name='m3_robot_description_publisher_node',
        parameters=[
            {
                'use_sim_time': True,
                'robot_description': robot_description,
            }
        ],
        output='screen',
    )

    static_tfs = [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='m3_world_to_base_link_tf',
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
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='m3_world_to_d435_link_tf',
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
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='m3_d435_link_to_color_frame_tf',
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
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='m3_d435_link_to_depth_frame_tf',
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
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='m3_d435_color_frame_to_optical_tf',
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
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='m3_d435_depth_frame_to_optical_tf',
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
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='m3_d435_link_to_infra1_frame_tf',
            arguments=[
                '--x', '0.0',
                '--y', '-0.026',
                '--z', '0.0',
                '--roll', '0.0',
                '--pitch', '0.0',
                '--yaw', '0.0',
                '--frame-id', 'd435_link',
                '--child-frame-id', 'd435_infra1_frame',
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='m3_d435_link_to_infra2_frame_tf',
            arguments=[
                '--x', '0.0',
                '--y', '0.026',
                '--z', '0.0',
                '--roll', '0.0',
                '--pitch', '0.0',
                '--yaw', '0.0',
                '--frame-id', 'd435_link',
                '--child-frame-id', 'd435_infra2_frame',
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='m3_d435_infra1_frame_to_optical_tf',
            arguments=[
                '--x', '0.0',
                '--y', '0.0',
                '--z', '0.0',
                '--roll', '-1.5708',
                '--pitch', '0.0',
                '--yaw', '-1.5708',
                '--frame-id', 'd435_infra1_frame',
                '--child-frame-id', 'd435_infra1_optical_frame',
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='m3_d435_infra2_frame_to_optical_tf',
            arguments=[
                '--x', '0.0',
                '--y', '0.0',
                '--z', '0.0',
                '--roll', '-1.5708',
                '--pitch', '0.0',
                '--yaw', '-1.5708',
                '--frame-id', 'd435_infra2_frame',
                '--child-frame-id', 'd435_infra2_optical_frame',
            ],
        ),
    ]

    raw_depth_to_points = Node(
        package='ghost_mgg_sim',
        executable='depth_to_point_cloud_node',
        name='m3_raw_depth_to_point_cloud_node',
        parameters=[
            {
                'use_sim_time': True,
                'depth_topic': '/ghost_mgg/d435/depth/image_rect_raw',
                'camera_info_topic': '/ghost_mgg/d435/depth/camera_info',
                'points_topic': '/ghost_mgg/d435/depth/points',
                'output_frame': 'd435_depth_optical_frame',
                'pixel_stride': ParameterValue(point_cloud_stride, value_type=int),
            }
        ],
        output='screen',
    )

    raw_depth_preview = Node(
        package='ghost_mgg_sim',
        executable='depth_to_mono8_node',
        name='m3_raw_depth_to_mono8_node',
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

    target_mask_emulator = Node(
        package='ghost_mgg_sim',
        executable='target_mask_emulator_node',
        name='m3_target_mask_emulator_node',
        condition=IfCondition(enable_target_mask_emulator),
        parameters=[
            {
                'use_sim_time': True,
                'input_image_topic': '/ghost_mgg/d435/depth/image_rect_raw',
                'mask_topic': '/ghost_mgg/d435/target_mask',
                'roi_center_u_ratio': ParameterValue(roi_center_u_ratio, value_type=float),
                'roi_center_v_ratio': ParameterValue(roi_center_v_ratio, value_type=float),
                'roi_width_ratio': ParameterValue(roi_width_ratio, value_type=float),
                'roi_height_ratio': ParameterValue(roi_height_ratio, value_type=float),
            }
        ],
        output='screen',
    )

    failure_injector = Node(
        package='ghost_mgg_sim',
        executable='depth_failure_injector_node',
        name='m3_depth_failure_injector_node',
        parameters=[
            {
                'use_sim_time': True,
                'failure_mode': failure_mode,
                'input_depth_topic': '/ghost_mgg/d435/depth/image_rect_raw',
                'output_depth_topic': '/ghost_mgg/d435/depth/m3_corrupted',
                'hole_mask_topic': '/ghost_mgg/d435/evidence/hole_mask',
                'table_leakage_mask_topic': '/ghost_mgg/d435/evidence/table_leakage_mask',
                'edge_mask_topic': '/ghost_mgg/d435/evidence/edge_mask',
                'flying_point_mask_topic': '/ghost_mgg/d435/evidence/flying_point_mask',
                'biased_depth_mask_topic': '/ghost_mgg/d435/evidence/biased_depth_mask',
                'summary_topic': '/ghost_mgg/d435/evidence/summary',
                'use_mask_topic': True,
                'target_mask_topic': '/ghost_mgg/d435/target_mask',
                'roi_center_u_ratio': ParameterValue(roi_center_u_ratio, value_type=float),
                'roi_center_v_ratio': ParameterValue(roi_center_v_ratio, value_type=float),
                'roi_width_ratio': ParameterValue(roi_width_ratio, value_type=float),
                'roi_height_ratio': ParameterValue(roi_height_ratio, value_type=float),
                'table_leak_depth_m': ParameterValue(table_leak_depth_m, value_type=float),
                'flying_point_offset_m': ParameterValue(flying_point_offset_m, value_type=float),
                'biased_depth_offset_m': ParameterValue(biased_depth_offset_m, value_type=float),
                'edge_band_pixels': ParameterValue(edge_band_pixels, value_type=int),
                'flying_point_stride': ParameterValue(flying_point_stride, value_type=int),
                'pattern_seed': ParameterValue(pattern_seed, value_type=int),
            }
        ],
        output='screen',
    )

    corrupted_depth_to_points = Node(
        package='ghost_mgg_sim',
        executable='depth_to_point_cloud_node',
        name='m3_corrupted_depth_to_point_cloud_node',
        parameters=[
            {
                'use_sim_time': True,
                'depth_topic': '/ghost_mgg/d435/depth/m3_corrupted',
                'camera_info_topic': '/ghost_mgg/d435/depth/camera_info',
                'points_topic': '/ghost_mgg/d435/depth/m3_points',
                'output_frame': 'd435_depth_optical_frame',
                'pixel_stride': ParameterValue(point_cloud_stride, value_type=int),
            }
        ],
        output='screen',
    )

    corrupted_depth_preview = Node(
        package='ghost_mgg_sim',
        executable='depth_to_mono8_node',
        name='m3_corrupted_depth_to_mono8_node',
        parameters=[
            {
                'use_sim_time': True,
                'depth_topic': '/ghost_mgg/d435/depth/m3_corrupted',
                'preview_topic': '/ghost_mgg/d435/depth/m3_image_viz',
                'min_depth_m': 0.2,
                'max_depth_m': 1.4,
            }
        ],
        output='screen',
    )

    observation_cache = Node(
        package='ghost_mgg_sim',
        executable='observation_cache_node',
        name='m3_observation_cache_node',
        parameters=[
            {
                'use_sim_time': True,
                'cache_namespace': 'm3_d435_failure',
                'observation_prefix': 'm3_obs',
                'observation_ref_topic': '/ghost_mgg/observations/latest',
                'rgb_topic': '/ghost_mgg/d435/color/image_raw',
                'depth_topic': '/ghost_mgg/d435/depth/m3_corrupted',
                'color_camera_info_topic': '/ghost_mgg/d435/color/camera_info',
                'depth_camera_info_topic': '/ghost_mgg/d435/depth/camera_info',
                'target_mask_topic': '/ghost_mgg/d435/target_mask',
                'infra1_topic': '/ghost_mgg/d435/infra1/image_rect_raw',
                'infra2_topic': '/ghost_mgg/d435/infra2/image_rect_raw',
                'rgb_frame_id': 'd435_color_optical_frame',
                'depth_frame_id': 'd435_depth_optical_frame',
                'mask_frame_id': 'd435_color_optical_frame',
                'max_age_sec': 1.0,
            }
        ],
        output='screen',
    )

    evidence_summary_logger = Node(
        package='ghost_mgg_sim',
        executable='evidence_summary_logger_node',
        name='m3_evidence_summary_logger_node',
        parameters=[
            {
                'use_sim_time': True,
                'scenario_id': scenario_id,
                'summary_topic': '/ghost_mgg/d435/evidence/summary',
                'log_dir': 'log/ghost_mgg_trials/m3_failure',
            }
        ],
        output='screen',
    )

    rviz_config = PathJoinSubstitution(
        [
            FindPackageShare('ghost_mgg_sim'),
            'rviz',
            'm3_failure_camera_inspect.rviz',
        ]
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='m3_failure_camera_inspect_rviz',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(show_rviz),
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('headless', default_value='true'),
            DeclareLaunchArgument('show_rviz', default_value='false'),
            DeclareLaunchArgument('world_file', default_value='m2_tabletop_visual.sdf'),
            DeclareLaunchArgument('world_name', default_value='ghost_mgg_m2_visual'),
            DeclareLaunchArgument('scenario_id', default_value='custom'),
            DeclareLaunchArgument('failure_mode', default_value='mixed'),
            DeclareLaunchArgument('roi_center_u_ratio', default_value='0.50'),
            DeclareLaunchArgument('roi_center_v_ratio', default_value='0.58'),
            DeclareLaunchArgument('roi_width_ratio', default_value='0.22'),
            DeclareLaunchArgument('roi_height_ratio', default_value='0.22'),
            DeclareLaunchArgument('table_leak_depth_m', default_value='1.20'),
            DeclareLaunchArgument('flying_point_offset_m', default_value='0.12'),
            DeclareLaunchArgument('biased_depth_offset_m', default_value='-0.04'),
            DeclareLaunchArgument('edge_band_pixels', default_value='2'),
            DeclareLaunchArgument('flying_point_stride', default_value='5'),
            DeclareLaunchArgument('point_cloud_stride', default_value='1'),
            DeclareLaunchArgument('pattern_seed', default_value='0'),
            DeclareLaunchArgument('enable_target_mask_emulator', default_value='true'),
            visual_scene,
            camera_bridge,
            robot_description_publisher,
            *static_tfs,
            raw_depth_to_points,
            raw_depth_preview,
            target_mask_emulator,
            failure_injector,
            corrupted_depth_to_points,
            corrupted_depth_preview,
            observation_cache,
            evidence_summary_logger,
            rviz,
        ]
    )
