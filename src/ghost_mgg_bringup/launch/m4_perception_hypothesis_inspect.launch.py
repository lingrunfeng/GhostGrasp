from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scenario_id = LaunchConfiguration('scenario_id')
    scene_id = LaunchConfiguration('scene_id')
    headless = LaunchConfiguration('headless')
    show_rviz = LaunchConfiguration('show_rviz')
    joint_hypothesis_report_path = LaunchConfiguration('joint_hypothesis_report_path')
    metric_proxy_report_path = LaunchConfiguration('metric_proxy_report_path')
    graspability_report_path = LaunchConfiguration('graspability_report_path')
    perception_hypothesis_topic = LaunchConfiguration('perception_hypothesis_topic')
    executed_hypotheses_topic = LaunchConfiguration('executed_hypotheses_topic')
    include_rejected = LaunchConfiguration('include_rejected')

    visual_scene = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('ghost_mgg_bringup'),
                    'launch',
                    'm4_offline_ranking_inspect.launch.py',
                ]
            )
        ),
        launch_arguments={
            'scenario_id': scenario_id,
            'headless': headless,
            'show_rviz': show_rviz,
            'real_scene_id': scene_id,
            'graspability_scene_id': scene_id,
            'joint_hypothesis_report_path': joint_hypothesis_report_path,
            'joint_hypothesis_topic': perception_hypothesis_topic,
            'executed_hypotheses_topic': executed_hypotheses_topic,
            'enable_joint_hypothesis_contract': 'false',
        }.items(),
    )

    perception_publisher = Node(
        package='ghost_mgg_sim',
        executable='m4_perception_hypothesis_publisher_node.py',
        name='m4_perception_hypothesis_publisher_node',
        parameters=[
            {
                'use_sim_time': True,
                'scene_id': scene_id,
                'joint_report_path': joint_hypothesis_report_path,
                'metric_proxy_report_path': metric_proxy_report_path,
                'graspability_report_path': graspability_report_path,
                'hypothesis_topic': perception_hypothesis_topic,
                'frame_id': 'world',
                'trial_id': 'm4_perception',
                'observation_id': 'm4_perception_hypotheses',
                'backend_name': 'ghost_mgg_m4_perception',
                'include_rejected': include_rejected,
            }
        ],
        output='screen',
    )

    perception_contract_markers = Node(
        package='ghost_mgg_sim',
        executable='hypothesis_markers_node',
        name='m4_perception_contract_markers_node',
        parameters=[
            {
                'use_sim_time': True,
                'hypotheses_topic': perception_hypothesis_topic,
                'marker_topic': '/ghost_mgg/m4_joint_contract_markers',
                'executed_hypotheses_topic': executed_hypotheses_topic,
                'hide_executed_hypotheses': False,
            }
        ],
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('scenario_id', default_value='S6'),
            DeclareLaunchArgument('scene_id', default_value='daylight_transparent_jelly_cup_001'),
            DeclareLaunchArgument('headless', default_value='true'),
            DeclareLaunchArgument('show_rviz', default_value='true'),
            DeclareLaunchArgument(
                'joint_hypothesis_report_path',
                default_value='reports/m4_joint_hypotheses/joint_hypotheses.json',
            ),
            DeclareLaunchArgument(
                'metric_proxy_report_path',
                default_value='reports/m4_metric_proxies/metric_proxies.json',
            ),
            DeclareLaunchArgument(
                'graspability_report_path',
                default_value='reports/m4_graspability_dryrun/graspability.json',
            ),
            DeclareLaunchArgument(
                'perception_hypothesis_topic',
                default_value='/ghost_mgg/m4_perception_hypotheses',
            ),
            DeclareLaunchArgument(
                'executed_hypotheses_topic',
                default_value='/ghost_mgg/m4_executed_hypotheses',
            ),
            DeclareLaunchArgument('include_rejected', default_value='false'),
            visual_scene,
            perception_publisher,
            perception_contract_markers,
        ]
    )
