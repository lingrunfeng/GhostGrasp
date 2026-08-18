from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def build_m4_moveit_config():
    description_share = get_package_share_directory('ghost_mgg_mycobot_description')
    moveit_share = get_package_share_directory('ghost_mgg_moveit_config')
    mycobot_xacro = str(
        Path(description_share) / 'urdf' / 'robots' / 'mycobot_280.urdf.xacro'
    )

    return (
        MoveItConfigsBuilder('mycobot_280', package_name='ghost_mgg_moveit_config')
        .robot_description(
            file_path=mycobot_xacro,
            mappings={
                'add_world': 'false',
                'use_camera': 'false',
                'use_gazebo': 'true',
                'use_gripper': 'true',
            },
        )
        .robot_description_semantic(
            file_path=str(Path(moveit_share) / 'config' / 'ghost_mgg_mycobot_280.srdf')
        )
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .planning_pipelines(
            default_planning_pipeline='ompl',
            pipelines=['ompl'],
            load_all=False,
        )
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .joint_limits(file_path='config/joint_limits.yaml')
        .to_moveit_configs()
    )


def generate_launch_description():
    scenario_id = LaunchConfiguration('scenario_id')
    headless = LaunchConfiguration('headless')
    show_rviz = LaunchConfiguration('show_rviz')
    auto_execute = LaunchConfiguration('auto_execute')
    auto_execute_output_json = LaunchConfiguration('auto_execute_output_json')
    world_file = LaunchConfiguration('world_file')
    world_name = LaunchConfiguration('world_name')
    live_hypothesis_topic = LaunchConfiguration('live_hypothesis_topic')
    live_marker_topic = LaunchConfiguration('live_marker_topic')
    executed_hypotheses_topic = LaunchConfiguration('executed_hypotheses_topic')
    verify_lift_and_hold = LaunchConfiguration('verify_lift_and_hold')
    top_k = LaunchConfiguration('top_k')
    max_visible_hypotheses = LaunchConfiguration('max_visible_hypotheses')
    enable_target_lock = LaunchConfiguration('enable_target_lock')
    point_cloud_stride = LaunchConfiguration('point_cloud_stride')
    target_color_hint = LaunchConfiguration('target_color_hint')
    external_target_mask_topic = LaunchConfiguration('external_target_mask_topic')
    require_external_target_mask = LaunchConfiguration('require_external_target_mask')
    moveit_config = build_m4_moveit_config()

    live_perception = GroupAction(
        scoped=True,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare('ghost_mgg_bringup'),
                            'launch',
                            'm4_no_truth_live_perception.launch.py',
                        ]
                    )
                ),
                launch_arguments={
                    'scenario_id': scenario_id,
                    'headless': headless,
                    'show_rviz': 'false',
                    'world_file': world_file,
                    'world_name': world_name,
                    'live_hypothesis_topic': live_hypothesis_topic,
                    'live_marker_topic': '/ghost_mgg/m4_live_debug_markers',
                    'top_k': top_k,
                    'max_visible_hypotheses': max_visible_hypotheses,
                    'enable_target_lock': enable_target_lock,
                    'point_cloud_stride': point_cloud_stride,
                    'target_color_hint': target_color_hint,
                    'external_target_mask_topic': external_target_mask_topic,
                    'require_external_target_mask': require_external_target_mask,
                }.items(),
            )
        ],
    )

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('ghost_mgg_moveit_config'),
                    'launch',
                    'm2_move_group.launch.py',
                ]
            )
        )
    )

    live_contract_markers = Node(
        package='ghost_mgg_sim',
        executable='hypothesis_markers_node',
        name='m4_no_truth_live_moveit_contract_markers_node',
        parameters=[
            {
                'use_sim_time': True,
                'hypotheses_topic': live_hypothesis_topic,
                'marker_topic': live_marker_topic,
                'executed_hypotheses_topic': executed_hypotheses_topic,
                'hide_executed_hypotheses': True,
                'hide_all_after_success': True,
                'max_visible_hypotheses': ParameterValue(max_visible_hypotheses, value_type=int),
            }
        ],
        output='screen',
    )

    moveit_executor = Node(
        package='ghost_mgg_backends',
        executable='moveit_sim_execute_server',
        name='moveit_sim_execute_server',
        parameters=[
            moveit_config.to_dict(),
            {
                'use_sim_time': True,
                'action_name': '/grasp_executors/moveit_sim/execute',
                'planning_group': 'arm',
                'end_effector_link': 'gripper_tcp',
                'gripper_trajectory_action_name': '/gripper_group_controller/follow_joint_trajectory',
                'enable_gripper': True,
                'gripper_open_position': 0.15,
                'gripper_close_position': -0.38,
                'gripper_max_effort': 45.0,
                'gripper_motion_duration_sec': 0.60,
                'wait_for_close_gripper_result': True,
                'settle_time_sec': 0.80,
                'verify_lift_and_hold': ParameterValue(verify_lift_and_hold, value_type=bool),
                'planning_time_sec': 5.0,
                'velocity_scaling': 0.20,
                'acceleration_scaling': 0.20,
                'use_grasp_orientation': True,
                'goal_position_tolerance_m': 0.006,
                'goal_orientation_tolerance_rad': 0.45,
                # No-truth live execution must not add hard-coded target obstacles.
                # Table collision stays enabled inside the executor.
                'add_m2_scene_obstacles': False,
                'scene_object_padding_m': 0.006,
                'use_cartesian_top_grasp_segments': False,
            },
        ],
        output='screen',
    )

    auto_execute_client = Node(
        package='ghost_mgg_bringup',
        executable='execute_m4_joint_hypothesis_action.py',
        name='m4_no_truth_auto_execute_client',
        arguments=[
            '--hypotheses-topic',
            live_hypothesis_topic,
            '--execute-action',
            '/grasp_executors/moveit_sim/execute',
            '--executed-topic',
            executed_hypotheses_topic,
            '--fallback-until-success',
            '--reset-executed-before-run',
            '--max-attempts',
            top_k,
            '--fallback-delay-sec',
            '1.0',
            '--stable-hypotheses-count',
            '3',
            '--stable-center-tolerance-m',
            '0.015',
            '--hypotheses-timeout-sec',
            '45.0',
            '--action-server-timeout-sec',
            '45.0',
            '--result-timeout-sec',
            '110.0',
            '--max-runtime-sec',
            '90.0',
            '--output-json',
            auto_execute_output_json,
        ],
        condition=IfCondition(auto_execute),
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
        name='m4_no_truth_live_moveit_rviz',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(show_rviz),
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('scenario_id', default_value='S0'),
            DeclareLaunchArgument('headless', default_value='true'),
            DeclareLaunchArgument('show_rviz', default_value='false'),
            DeclareLaunchArgument('world_file', default_value='m2_tabletop_visual.sdf'),
            DeclareLaunchArgument('world_name', default_value='ghost_mgg_m2_visual'),
            DeclareLaunchArgument('live_hypothesis_topic', default_value='/ghost_mgg/m4_live_hypotheses'),
            DeclareLaunchArgument('live_marker_topic', default_value='/ghost_mgg/m4_joint_contract_markers'),
            DeclareLaunchArgument('executed_hypotheses_topic', default_value='/ghost_mgg/m4_executed_hypotheses'),
            DeclareLaunchArgument('top_k', default_value='5'),
            DeclareLaunchArgument('max_visible_hypotheses', default_value='5'),
            DeclareLaunchArgument('enable_target_lock', default_value='false'),
            DeclareLaunchArgument('point_cloud_stride', default_value='2'),
            DeclareLaunchArgument('target_color_hint', default_value='red'),
            DeclareLaunchArgument('external_target_mask_topic', default_value='/ghost_mgg/d435/external_target_mask'),
            DeclareLaunchArgument('require_external_target_mask', default_value='false'),
            DeclareLaunchArgument('verify_lift_and_hold', default_value='false'),
            DeclareLaunchArgument('auto_execute', default_value='false'),
            DeclareLaunchArgument(
                'auto_execute_output_json',
                default_value='reports/m4_no_truth_live_execute/manual_auto_execute.json',
            ),
            live_perception,
            TimerAction(period=13.0, actions=[move_group]),
            TimerAction(period=15.0, actions=[live_contract_markers]),
            TimerAction(period=17.0, actions=[moveit_executor]),
            TimerAction(period=18.0, actions=[rviz]),
            TimerAction(period=24.0, actions=[auto_execute_client]),
        ]
    )
