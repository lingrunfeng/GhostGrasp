import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_share = FindPackageShare('ghost_mgg_sim')
    description_share = FindPackageShare('ghost_mgg_mycobot_description')
    headless = LaunchConfiguration('headless')
    world_file = LaunchConfiguration('world_file')
    world_name = LaunchConfiguration('world_name')

    models_path = PathJoinSubstitution([sim_share, 'models'])
    world_path = PathJoinSubstitution([sim_share, 'worlds', world_file])
    mycobot_xacro = PathJoinSubstitution(
        [
            description_share,
            'urdf',
            'robots',
            'mycobot_280.urdf.xacro',
        ]
    )

    existing_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    gz_resource_path = [models_path]
    if existing_resource_path:
        gz_resource_path.extend([os.pathsep, existing_resource_path])

    robot_description = Command(
        [
            'xacro',
            ' ',
            mycobot_xacro,
            ' ',
            'add_world:=false',
            ' ',
            'use_camera:=false',
            ' ',
            'use_gazebo:=true',
            ' ',
            'use_gripper:=true',
        ]
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('ros_gz_sim'),
                    'launch',
                    'gz_sim.launch.py',
                ]
            )
        ),
        launch_arguments={
            'gz_args': [
                PythonExpression(["'-s -r ' if '", headless, "' == 'true' else '-r '"]),
                world_path,
            ]
        }.items(),
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='m2_clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {
                'use_sim_time': True,
                'robot_description': robot_description,
            }
        ],
    )

    spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world',
            world_name,
            '-param',
            'robot_description',
            '-name',
            'ghost_mgg_mycobot_280',
            '-x',
            '-0.171207',
            '-y',
            '0.228790',
            '-z',
            '0.765000',
            '-Y',
            '-2.180640',
        ],
        parameters=[{'robot_description': robot_description}],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_group_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    pregrasp_goal = (
        '{trajectory: {joint_names: [link1_to_link2, link2_to_link3, '
        'link3_to_link4, link4_to_link5, link5_to_link6, link6_to_link6_flange], '
        'points: [{positions: [0.0, -0.85, 1.35, -0.85, 0.65, 0.0], '
        'time_from_start: {sec: 2, nanosec: 0}}]}}'
    )
    send_pregrasp = ExecuteProcess(
        cmd=[
            'ros2',
            'action',
            'send_goal',
            '/arm_controller/follow_joint_trajectory',
            'control_msgs/action/FollowJointTrajectory',
            pregrasp_goal,
        ],
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'headless',
                default_value='false',
                description='Run GZ Sim server-only without opening the GUI.',
            ),
            DeclareLaunchArgument('world_file', default_value='m2_tabletop_visual.sdf'),
            DeclareLaunchArgument('world_name', default_value='ghost_mgg_m2_visual'),
            SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path),
            gz_sim,
            clock_bridge,
            robot_state_publisher,
            TimerAction(period=3.0, actions=[spawn_node]),
            TimerAction(
                period=6.0,
                actions=[
                    joint_state_broadcaster_spawner,
                    arm_controller_spawner,
                    gripper_controller_spawner,
                ],
            ),
            TimerAction(period=10.0, actions=[send_pregrasp]),
        ]
    )
