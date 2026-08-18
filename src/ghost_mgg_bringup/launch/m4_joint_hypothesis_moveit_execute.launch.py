from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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
    headless = LaunchConfiguration('headless')
    show_rviz = LaunchConfiguration('show_rviz')
    sim_grasp_targets_path = LaunchConfiguration('sim_grasp_targets_path')
    joint_hypothesis_report_path = LaunchConfiguration('joint_hypothesis_report_path')
    joint_hypothesis_topic = LaunchConfiguration('joint_hypothesis_topic')
    executed_hypotheses_topic = LaunchConfiguration('executed_hypotheses_topic')
    moveit_config = build_m4_moveit_config()
    m4_dryrun_scene = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('ghost_mgg_bringup'),
                    'launch',
                    'm4_sim_grasp_moveit_dryrun.launch.py',
                ]
            )
        ),
        launch_arguments={
            'headless': headless,
            'sim_grasp_targets_path': sim_grasp_targets_path,
            'joint_hypothesis_report_path': joint_hypothesis_report_path,
            'joint_hypothesis_topic': joint_hypothesis_topic,
        }.items(),
    )

    rviz_config = PathJoinSubstitution(
        [
            FindPackageShare('ghost_mgg_sim'),
            'rviz',
            'm4_joint_execute.rviz',
        ]
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='m4_joint_execute_rviz',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(show_rviz),
    )

    joint_hypothesis_markers = Node(
        package='ghost_mgg_sim',
        executable='m4_joint_hypothesis_markers_node.py',
        name='m4_joint_hypothesis_markers_node',
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

    joint_contract_markers = Node(
        package='ghost_mgg_sim',
        executable='hypothesis_markers_node',
        name='m4_joint_contract_markers_node',
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
                'verify_lift_and_hold': False,
                'planning_time_sec': 5.0,
                'velocity_scaling': 0.20,
                'acceleration_scaling': 0.20,
                'use_grasp_orientation': True,
                'goal_position_tolerance_m': 0.006,
                'goal_orientation_tolerance_rad': 0.45,
                'add_m2_scene_obstacles': True,
                'scene_object_padding_m': 0.006,
                'use_cartesian_top_grasp_segments': False,
            },
        ],
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('headless', default_value='true'),
            DeclareLaunchArgument('show_rviz', default_value='false'),
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
            DeclareLaunchArgument(
                'joint_hypothesis_report_path',
                default_value='reports/m4_joint_hypotheses/joint_hypotheses.json',
            ),
            DeclareLaunchArgument(
                'joint_hypothesis_topic',
                default_value='/ghost_mgg/m4_joint_hypotheses',
            ),
            DeclareLaunchArgument(
                'executed_hypotheses_topic',
                default_value='/ghost_mgg/m4_executed_hypotheses',
            ),
            m4_dryrun_scene,
            TimerAction(period=15.0, actions=[joint_hypothesis_markers]),
            TimerAction(period=15.0, actions=[joint_contract_markers]),
            TimerAction(period=17.0, actions=[moveit_executor]),
            TimerAction(period=18.0, actions=[rviz]),
        ]
    )
