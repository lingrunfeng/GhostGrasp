from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    headless = LaunchConfiguration('headless')
    sim_grasp_targets_path = LaunchConfiguration('sim_grasp_targets_path')
    joint_hypothesis_report_path = LaunchConfiguration('joint_hypothesis_report_path')
    joint_hypothesis_topic = LaunchConfiguration('joint_hypothesis_topic')

    m2_scene = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('ghost_mgg_sim'),
                    'launch',
                    'm2_visual_scene.launch.py',
                ]
            )
        ),
        launch_arguments={'headless': headless}.items(),
    )

    world_to_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m4_world_to_base_link_tf',
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

    joint_hypothesis_publisher = Node(
        package='ghost_mgg_sim',
        executable='m4_joint_hypothesis_publisher_node.py',
        name='m4_joint_hypothesis_publisher_node',
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

    return LaunchDescription(
        [
            DeclareLaunchArgument('headless', default_value='true'),
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
            m2_scene,
            world_to_base_tf,
            TimerAction(period=13.0, actions=[move_group]),
            TimerAction(period=14.0, actions=[sim_grasp_markers]),
            TimerAction(period=14.0, actions=[joint_hypothesis_publisher]),
        ]
    )
