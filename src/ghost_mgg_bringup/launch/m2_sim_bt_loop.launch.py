from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    m2_scene = IncludeLaunchDescription(
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

    m2_tree = PathJoinSubstitution(
        [
            FindPackageShare('ghost_mgg_bt'),
            'trees',
            'm2_sim_closed_loop.xml',
        ]
    )

    dummy_recovery = Node(
        package='ghost_mgg_backends',
        executable='dummy_recovery_server',
        name='m2_dummy_recovery_server',
        parameters=[{'mode': 'success', 'response_delay_ms': 10}],
        output='screen',
    )

    mycobot_executor = Node(
        package='ghost_mgg_backends',
        executable='mycobot_sim_execute_server',
        name='mycobot_sim_execute_server',
        parameters=[
            {
                'action_name': '/grasp_executors/mycobot_sim/execute',
                'trajectory_action_name': '/arm_controller/follow_joint_trajectory',
                'trajectory_server_timeout_sec': 6.0,
            }
        ],
        output='screen',
    )

    bt_runner = Node(
        package='ghost_mgg_bt',
        executable='bt_runner_node',
        name='m2_bt_runner',
        parameters=[
            {
                'tree_path': m2_tree,
                'backend_name': 'dummy',
                'executor_name': 'mycobot_sim',
                'target_label': 'm2_sim_target',
                'shape_hint': 'unknown',
                'recover_timeout_sec': 2.0,
                'execute_timeout_sec': 8.0,
                'max_hypotheses': 3,
                'trial_log_dir': 'log/ghost_mgg_trials/m2_sim',
            }
        ],
        output='screen',
    )

    return LaunchDescription(
        [
            m2_scene,
            TimerAction(period=11.0, actions=[dummy_recovery, mycobot_executor]),
            TimerAction(period=12.0, actions=[bt_runner]),
        ]
    )
