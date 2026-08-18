from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def build_m6_shadow_moveit_config():
    description_share = get_package_share_directory('ghost_mgg_mycobot_description')
    moveit_share = get_package_share_directory('ghost_mgg_moveit_config')

    mycobot_xacro = str(
        Path(description_share) / 'urdf' / 'robots' / 'mycobot_280.urdf.xacro'
    )

    moveit_config = (
        MoveItConfigsBuilder('mycobot_280', package_name='ghost_mgg_moveit_config')
        .robot_description(
            file_path=mycobot_xacro,
            mappings={
                'add_world': 'false',
                'use_camera': 'false',
                'use_gazebo': 'false',
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
        .joint_limits(file_path='config/joint_limits.yaml')
        .to_moveit_configs()
    )
    moveit_config.trajectory_execution = {}
    return moveit_config


def generate_launch_description():
    show_rviz = LaunchConfiguration('show_rviz')
    use_fake_joint_states = LaunchConfiguration('use_fake_joint_states')
    moveit_config = build_m6_shadow_moveit_config()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='m6_shadow_robot_state_publisher',
        output='screen',
        parameters=[
            moveit_config.robot_description,
            {
                'publish_frequency': 30.0,
                'use_sim_time': False,
            },
        ],
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='m6_shadow_joint_state_publisher',
        output='screen',
        parameters=[
            moveit_config.robot_description,
            {
                'rate': 30,
                'publish_default_positions': True,
                'use_mimic_tags': False,
                'use_smallest_joint_limits': True,
                'use_sim_time': False,
            },
        ],
        condition=IfCondition(use_fake_joint_states),
    )

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            {
                'allow_trajectory_execution': False,
                'moveit_manage_controllers': False,
                'publish_robot_description': True,
                'publish_robot_description_semantic': True,
                'use_sim_time': False,
            },
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='m6_shadow_moveit_rviz',
        output='screen',
        arguments=[
            '-d',
            str(
                Path(get_package_share_directory('ghost_mgg_moveit_config'))
                / 'rviz'
                / 'm2_moveit.rviz'
            ),
        ],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {'use_sim_time': False},
        ],
        condition=IfCondition(show_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('show_rviz', default_value='false'),
            DeclareLaunchArgument('use_fake_joint_states', default_value='true'),
            robot_state_publisher,
            joint_state_publisher,
            move_group,
            rviz,
        ]
    )
