from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scenario_id = LaunchConfiguration('scenario_id')
    headless = LaunchConfiguration('headless')
    show_rviz = LaunchConfiguration('show_rviz')
    ranking_report_path = LaunchConfiguration('ranking_report_path')
    real_ranking_report_path = LaunchConfiguration('real_ranking_report_path')
    graspability_report_path = LaunchConfiguration('graspability_report_path')
    sim_grasp_targets_path = LaunchConfiguration('sim_grasp_targets_path')
    moveit_plan_report_path = LaunchConfiguration('moveit_plan_report_path')
    joint_hypothesis_report_path = LaunchConfiguration('joint_hypothesis_report_path')
    joint_hypothesis_topic = LaunchConfiguration('joint_hypothesis_topic')
    executed_hypotheses_topic = LaunchConfiguration('executed_hypotheses_topic')
    enable_joint_hypothesis_contract = LaunchConfiguration('enable_joint_hypothesis_contract')
    real_scene_id = LaunchConfiguration('real_scene_id')
    graspability_scene_id = LaunchConfiguration('graspability_scene_id')
    top_k = LaunchConfiguration('top_k')

    m3_scene = IncludeLaunchDescription(
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
            'show_rviz': show_rviz,
        }.items(),
    )

    offline_ranking_markers = Node(
        package='ghost_mgg_sim',
        executable='m4_offline_ranking_markers_node',
        name='m4_offline_ranking_markers_node',
        parameters=[
            {
                'use_sim_time': True,
                'scenario_id': scenario_id,
                'ranking_report_path': ranking_report_path,
                'marker_topic': '/ghost_mgg/hypothesis_markers',
                'frame_id': 'd435_depth_optical_frame',
                'top_k': ParameterValue(top_k, value_type=int),
                'fx': 554.0,
                'fy': 554.0,
                'cx': 320.0,
                'cy': 240.0,
                'marker_depth_m': 1.12,
                'marker_thickness_m': 0.012,
            }
        ],
        output='screen',
    )

    real_ranking_markers = Node(
        package='ghost_mgg_sim',
        executable='m4_real_ranking_markers_node',
        name='m4_real_ranking_markers_node',
        parameters=[
            {
                'use_sim_time': True,
                'scene_id': real_scene_id,
                'ranking_report_path': real_ranking_report_path,
                'marker_topic': '/ghost_mgg/m4_real_ranking_markers',
                'frame_id': 'd435_depth_optical_frame',
                'top_k': 1,
                'fx': 554.0,
                'fy': 554.0,
                'cx': 320.0,
                'cy': 240.0,
                'marker_depth_m': 1.12,
                'marker_thickness_m': 0.016,
            }
        ],
        output='screen',
    )

    graspability_markers = Node(
        package='ghost_mgg_sim',
        executable='m4_graspability_markers_node',
        name='m4_graspability_markers_node',
        parameters=[
            {
                'use_sim_time': True,
                'scene_id': graspability_scene_id,
                'graspability_report_path': graspability_report_path,
                'marker_topic': '/ghost_mgg/m4_graspability_markers',
                'frame_id': 'world',
            }
        ],
        output='screen',
    )

    sim_grasp_markers = Node(
        package='ghost_mgg_sim',
        executable='m4_sim_grasp_markers_node',
        name='m4_sim_grasp_markers_node',
        parameters=[
            {
                'use_sim_time': True,
                'targets_path': sim_grasp_targets_path,
                'marker_topic': '/ghost_mgg/m4_sim_grasp_markers',
                'frame_id': 'world',
            }
        ],
        output='screen',
    )

    moveit_plan_markers = Node(
        package='ghost_mgg_sim',
        executable='m4_sim_moveit_plan_markers_node.py',
        name='m4_sim_moveit_plan_markers_node',
        parameters=[
            {
                'use_sim_time': True,
                'report_path': moveit_plan_report_path,
                'marker_topic': '/ghost_mgg/m4_sim_moveit_plan_markers',
                'frame_id': 'world',
            }
        ],
        output='screen',
    )

    joint_hypothesis_markers = Node(
        package='ghost_mgg_sim',
        executable='m4_joint_hypothesis_markers_node.py',
        name='m4_joint_hypothesis_markers_node',
        condition=IfCondition(enable_joint_hypothesis_contract),
        parameters=[
            {
                'use_sim_time': True,
                'report_path': joint_hypothesis_report_path,
                'targets_path': sim_grasp_targets_path,
                'marker_topic': '/ghost_mgg/m4_joint_hypothesis_markers',
                'executed_hypotheses_topic': executed_hypotheses_topic,
                'frame_id': 'world',
            }
        ],
        output='screen',
    )

    joint_hypothesis_publisher = Node(
        package='ghost_mgg_sim',
        executable='m4_joint_hypothesis_publisher_node.py',
        name='m4_joint_hypothesis_publisher_node',
        condition=IfCondition(enable_joint_hypothesis_contract),
        parameters=[
            {
                'use_sim_time': True,
                'report_path': joint_hypothesis_report_path,
                'targets_path': sim_grasp_targets_path,
                'hypothesis_topic': joint_hypothesis_topic,
                'frame_id': 'world',
                'backend_name': 'ghost_mgg_m4_joint',
                'trial_id': 'm4_tabletop',
                'observation_id': 'm4_joint_hypotheses',
            }
        ],
        output='screen',
    )

    joint_contract_markers = Node(
        package='ghost_mgg_sim',
        executable='hypothesis_markers_node',
        name='m4_joint_contract_markers_node',
        condition=IfCondition(enable_joint_hypothesis_contract),
        parameters=[
            {
                'use_sim_time': True,
                'hypotheses_topic': joint_hypothesis_topic,
                'marker_topic': '/ghost_mgg/m4_joint_contract_markers',
                'executed_hypotheses_topic': executed_hypotheses_topic,
                'hide_executed_hypotheses': True,
            }
        ],
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('scenario_id', default_value='S6'),
            DeclareLaunchArgument('headless', default_value='true'),
            DeclareLaunchArgument('show_rviz', default_value='true'),
            DeclareLaunchArgument('ranking_report_path', default_value='reports/m4_offline_ranking.json'),
            DeclareLaunchArgument('real_ranking_report_path', default_value='reports/m5_real_d435_ranking/m5_real_ranking.json'),
            DeclareLaunchArgument('graspability_report_path', default_value='reports/m4_graspability_dryrun/graspability.json'),
            DeclareLaunchArgument('moveit_plan_report_path', default_value='reports/m4_sim_moveit_dryrun/plan_results.json'),
            DeclareLaunchArgument('joint_hypothesis_report_path', default_value='reports/m4_joint_hypotheses/joint_hypotheses.json'),
            DeclareLaunchArgument('joint_hypothesis_topic', default_value='/ghost_mgg/m4_joint_hypotheses'),
            DeclareLaunchArgument(
                'executed_hypotheses_topic',
                default_value='/ghost_mgg/m4_executed_hypotheses',
            ),
            DeclareLaunchArgument('enable_joint_hypothesis_contract', default_value='true'),
            DeclareLaunchArgument(
                'sim_grasp_targets_path',
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare('ghost_mgg_sim'),
                        'config',
                        'm4_sim_grasp_targets.json',
                    ]
                ),
            ),
            DeclareLaunchArgument('real_scene_id', default_value='daylight_transparent_jelly_cup_001'),
            DeclareLaunchArgument('graspability_scene_id', default_value='daylight_transparent_jelly_cup_001'),
            DeclareLaunchArgument('top_k', default_value='3'),
            m3_scene,
            offline_ranking_markers,
            real_ranking_markers,
            graspability_markers,
            sim_grasp_markers,
            moveit_plan_markers,
            joint_hypothesis_markers,
            joint_hypothesis_publisher,
            joint_contract_markers,
        ]
    )
