from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def build_m2_moveit_config():
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
    moveit_config = build_m2_moveit_config()

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict(), {'use_sim_time': True}],
    )

    return LaunchDescription([move_group])
