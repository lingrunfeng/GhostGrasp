from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scenario_id = LaunchConfiguration('scenario_id')
    headless = LaunchConfiguration('headless')
    show_rviz = LaunchConfiguration('show_rviz')
    world_file = LaunchConfiguration('world_file')
    world_name = LaunchConfiguration('world_name')
    live_hypothesis_topic = LaunchConfiguration('live_hypothesis_topic')
    live_marker_topic = LaunchConfiguration('live_marker_topic')
    top_k = LaunchConfiguration('top_k')
    max_visible_hypotheses = LaunchConfiguration('max_visible_hypotheses')
    depth_margin_m = LaunchConfiguration('depth_margin_m')
    min_component_area_px = LaunchConfiguration('min_component_area_px')
    primitive_height_m = LaunchConfiguration('primitive_height_m')
    publish_rate_hz = LaunchConfiguration('publish_rate_hz')
    processing_stride = LaunchConfiguration('processing_stride')
    point_cloud_stride = LaunchConfiguration('point_cloud_stride')
    target_color_hint = LaunchConfiguration('target_color_hint')
    enable_target_lock = LaunchConfiguration('enable_target_lock')
    external_target_mask_topic = LaunchConfiguration('external_target_mask_topic')
    require_external_target_mask = LaunchConfiguration('require_external_target_mask')
    base_frame_id = LaunchConfiguration('base_frame_id')
    table_z_m = LaunchConfiguration('table_z_m')

    m3_scene = GroupAction(
        scoped=True,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare('ghost_mgg_bringup'),
                            'launch',
                            'm3_failure_scenario.launch.py',
                        ]
                    )
                ),
                launch_arguments={
                    'scenario_id': scenario_id,
                    'headless': headless,
                    'show_rviz': 'false',
                    'world_file': world_file,
                    'world_name': world_name,
                    'enable_target_mask_emulator': 'false',
                    'point_cloud_stride': point_cloud_stride,
                }.items(),
            )
        ],
    )

    live_hypothesis_publisher = Node(
        package='ghost_mgg_sim',
        executable='m4_live_hypothesis_publisher_node.py',
        name='m4_live_hypothesis_publisher_node',
        parameters=[
            {
                'use_sim_time': True,
                'raw_depth_topic': '/ghost_mgg/d435/depth/image_rect_raw',
                'corrupted_depth_topic': '/ghost_mgg/d435/depth/m3_corrupted',
                'color_topic': '/ghost_mgg/d435/color/image_raw',
                'camera_info_topic': '/ghost_mgg/d435/depth/camera_info',
                'mask_topic': '/ghost_mgg/d435/target_mask',
                'external_mask_topic': external_target_mask_topic,
                'require_external_mask': ParameterValue(require_external_target_mask, value_type=bool),
                'hypothesis_topic': live_hypothesis_topic,
                'frame_id': 'd435_depth_optical_frame',
                'base_frame_id': base_frame_id,
                'table_z_m': ParameterValue(table_z_m, value_type=float),
                'top_k': ParameterValue(top_k, value_type=int),
                'depth_margin_m': ParameterValue(depth_margin_m, value_type=float),
                'min_component_area_px': ParameterValue(min_component_area_px, value_type=int),
                'primitive_height_m': ParameterValue(primitive_height_m, value_type=float),
                'publish_rate_hz': ParameterValue(publish_rate_hz, value_type=float),
                'processing_stride': ParameterValue(processing_stride, value_type=int),
                'target_color_hint': target_color_hint,
                'enable_target_lock': ParameterValue(enable_target_lock, value_type=bool),
                'max_locked_center_distance_px': 60.0,
            }
        ],
        output='screen',
    )

    live_contract_markers = Node(
        package='ghost_mgg_sim',
        executable='hypothesis_markers_node',
        name='m4_live_contract_markers_node',
        parameters=[
            {
                'use_sim_time': True,
                'hypotheses_topic': live_hypothesis_topic,
                'marker_topic': live_marker_topic,
                'hide_executed_hypotheses': False,
                'max_visible_hypotheses': ParameterValue(max_visible_hypotheses, value_type=int),
            }
        ],
        output='screen',
    )

    rviz_config = PathJoinSubstitution(
        [
            FindPackageShare('ghost_mgg_sim'),
            'rviz',
            'm4_no_truth_live_perception.rviz',
        ]
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='m4_no_truth_live_rviz',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(show_rviz),
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('scenario_id', default_value='S6'),
            DeclareLaunchArgument('headless', default_value='true'),
            DeclareLaunchArgument('show_rviz', default_value='true'),
            DeclareLaunchArgument('world_file', default_value='m2_tabletop_visual.sdf'),
            DeclareLaunchArgument('world_name', default_value='ghost_mgg_m2_visual'),
            DeclareLaunchArgument('live_hypothesis_topic', default_value='/ghost_mgg/m4_live_hypotheses'),
            DeclareLaunchArgument('live_marker_topic', default_value='/ghost_mgg/m4_joint_contract_markers'),
            DeclareLaunchArgument('top_k', default_value='5'),
            DeclareLaunchArgument('max_visible_hypotheses', default_value='5'),
            DeclareLaunchArgument('depth_margin_m', default_value='0.035'),
            DeclareLaunchArgument('min_component_area_px', default_value='40'),
            DeclareLaunchArgument('primitive_height_m', default_value='0.04'),
            DeclareLaunchArgument('publish_rate_hz', default_value='5.0'),
            DeclareLaunchArgument('processing_stride', default_value='2'),
            DeclareLaunchArgument('point_cloud_stride', default_value='2'),
            DeclareLaunchArgument('target_color_hint', default_value='red'),
            DeclareLaunchArgument('enable_target_lock', default_value='true'),
            DeclareLaunchArgument('external_target_mask_topic', default_value='/ghost_mgg/d435/external_target_mask'),
            DeclareLaunchArgument('require_external_target_mask', default_value='false'),
            DeclareLaunchArgument('base_frame_id', default_value='world'),
            DeclareLaunchArgument('table_z_m', default_value='0.7400'),
            m3_scene,
            live_hypothesis_publisher,
            live_contract_markers,
            rviz,
        ]
    )
