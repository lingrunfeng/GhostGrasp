from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ghost_mgg_backends',
            executable='dummy_recovery_server',
            name='dummy_recovery_server',
            parameters=[{'mode': 'success', 'response_delay_ms': 10}],
        ),
        Node(
            package='ghost_mgg_backends',
            executable='dummy_execute_server',
            name='dummy_execute_server',
            parameters=[{
                'mode': 'fail_first_then_succeed',
                'response_delay_ms': 10,
            }],
        ),
        Node(
            package='ghost_mgg_bt',
            executable='bt_runner_node',
            name='m0_bt_runner',
            parameters=[{
                'backend_name': 'dummy',
                'executor_name': 'dummy',
                'target_label': 'm0_dummy_target',
                'shape_hint': 'unknown',
                'recover_timeout_sec': 2.0,
                'execute_timeout_sec': 2.0,
                'max_hypotheses': 3,
            }],
        ),
    ])
