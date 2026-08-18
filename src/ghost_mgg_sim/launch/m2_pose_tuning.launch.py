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


def static_tf_node(name, parent, child, x, y, z, roll, pitch, yaw):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=name,
        arguments=[
            '--x', x,
            '--y', y,
            '--z', z,
            '--roll', roll,
            '--pitch', pitch,
            '--yaw', yaw,
            '--frame-id', parent,
            '--child-frame-id', child,
        ],
    )


def declare_pose_arguments():
    return [
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('arm_x', default_value='-0.171207'),
        DeclareLaunchArgument('arm_y', default_value='0.228790'),
        DeclareLaunchArgument('arm_z', default_value='0.765000'),
        DeclareLaunchArgument('arm_yaw', default_value='-2.180640'),
        DeclareLaunchArgument('stand_x', default_value='0.410788'),
        DeclareLaunchArgument('stand_y', default_value='-0.0791765'),
        DeclareLaunchArgument('stand_z', default_value='0.74'),
        DeclareLaunchArgument('stand_yaw', default_value='-0.386880'),
        DeclareLaunchArgument('camera_x', default_value='0.286775'),
        DeclareLaunchArgument('camera_y', default_value='-0.079936'),
        DeclareLaunchArgument('camera_z', default_value='1.105590'),
        DeclareLaunchArgument('camera_roll', default_value='0.0'),
        DeclareLaunchArgument('camera_pitch', default_value='0.95'),
        DeclareLaunchArgument('camera_yaw', default_value='2.754700'),
    ]


def generate_launch_description():
    sim_share = FindPackageShare('ghost_mgg_sim')
    description_share = FindPackageShare('ghost_mgg_mycobot_description')
    headless = LaunchConfiguration('headless')

    models_path = PathJoinSubstitution([sim_share, 'models'])
    world_path = PathJoinSubstitution([sim_share, 'worlds', 'm2_tabletop_tuning.sdf'])
    stand_model_path = PathJoinSubstitution(
        [sim_share, 'models', 'd435_stand_only', 'model.sdf']
    )
    camera_model_path = PathJoinSubstitution(
        [sim_share, 'models', 'd435_camera_head', 'model.sdf']
    )
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
        name='m2_pose_tuning_clock_bridge',
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

    robot_description_publisher = Node(
        package='ghost_mgg_sim',
        executable='robot_description_publisher_node',
        name='m2_pose_tuning_robot_description_publisher',
        parameters=[
            {
                'use_sim_time': True,
                'robot_description': robot_description,
            }
        ],
        output='screen',
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world',
            'ghost_mgg_m2_tuning',
            '-param',
            'robot_description',
            '-name',
            'ghost_mgg_mycobot_280',
            '-x',
            LaunchConfiguration('arm_x'),
            '-y',
            LaunchConfiguration('arm_y'),
            '-z',
            LaunchConfiguration('arm_z'),
            '-Y',
            LaunchConfiguration('arm_yaw'),
        ],
        parameters=[{'robot_description': robot_description}],
        output='screen',
    )

    spawn_stand = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world',
            'ghost_mgg_m2_tuning',
            '-file',
            stand_model_path,
            '-name',
            'd435_stand_only',
            '-x',
            LaunchConfiguration('stand_x'),
            '-y',
            LaunchConfiguration('stand_y'),
            '-z',
            LaunchConfiguration('stand_z'),
            '-Y',
            LaunchConfiguration('stand_yaw'),
        ],
        output='screen',
    )

    spawn_camera = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world',
            'ghost_mgg_m2_tuning',
            '-file',
            camera_model_path,
            '-name',
            'd435_camera_head',
            '-x',
            LaunchConfiguration('camera_x'),
            '-y',
            LaunchConfiguration('camera_y'),
            '-z',
            LaunchConfiguration('camera_z'),
            '-R',
            LaunchConfiguration('camera_roll'),
            '-P',
            LaunchConfiguration('camera_pitch'),
            '-Y',
            LaunchConfiguration('camera_yaw'),
        ],
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

    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='m2_pose_tuning_camera_bridge',
        arguments=[
            '/ghost_mgg/d435/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/ghost_mgg/d435/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/ghost_mgg/d435/depth/image_rect_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/ghost_mgg/d435/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
    )

    world_to_base_tf = static_tf_node(
        'm2_pose_tuning_world_to_base_link_tf',
        'world',
        'base_link',
        LaunchConfiguration('arm_x'),
        LaunchConfiguration('arm_y'),
        LaunchConfiguration('arm_z'),
        '0.0',
        '0.0',
        LaunchConfiguration('arm_yaw'),
    )

    world_to_d435_link_tf = static_tf_node(
        'm2_pose_tuning_world_to_d435_link_tf',
        'world',
        'd435_link',
        LaunchConfiguration('camera_x'),
        LaunchConfiguration('camera_y'),
        LaunchConfiguration('camera_z'),
        LaunchConfiguration('camera_roll'),
        LaunchConfiguration('camera_pitch'),
        LaunchConfiguration('camera_yaw'),
    )

    d435_link_to_color_frame_tf = static_tf_node(
        'm2_pose_tuning_d435_link_to_color_frame_tf',
        'd435_link',
        'd435_color_frame',
        '0.0',
        '0.0',
        '0.0',
        '0.0',
        '0.0',
        '0.0',
    )

    d435_link_to_depth_frame_tf = static_tf_node(
        'm2_pose_tuning_d435_link_to_depth_frame_tf',
        'd435_link',
        'd435_depth_frame',
        '0.0',
        '0.0',
        '0.0',
        '0.0',
        '0.0',
        '0.0',
    )

    color_frame_to_optical_tf = static_tf_node(
        'm2_pose_tuning_d435_color_frame_to_optical_tf',
        'd435_color_frame',
        'd435_color_optical_frame',
        '0.0',
        '0.0',
        '0.0',
        '-1.5708',
        '0.0',
        '-1.5708',
    )

    depth_frame_to_optical_tf = static_tf_node(
        'm2_pose_tuning_d435_depth_frame_to_optical_tf',
        'd435_depth_frame',
        'd435_depth_optical_frame',
        '0.0',
        '0.0',
        '0.0',
        '-1.5708',
        '0.0',
        '-1.5708',
    )

    depth_to_points = Node(
        package='ghost_mgg_sim',
        executable='depth_to_point_cloud_node',
        name='m2_pose_tuning_depth_to_point_cloud_node',
        parameters=[
            {
                'use_sim_time': True,
                'depth_topic': '/ghost_mgg/d435/depth/image_rect_raw',
                'camera_info_topic': '/ghost_mgg/d435/depth/camera_info',
                'points_topic': '/ghost_mgg/d435/depth/points',
                'output_frame': 'd435_depth_optical_frame',
            }
        ],
        output='screen',
    )

    depth_preview = Node(
        package='ghost_mgg_sim',
        executable='depth_to_mono8_node',
        name='m2_pose_tuning_depth_to_mono8_node',
        parameters=[
            {
                'use_sim_time': True,
                'depth_topic': '/ghost_mgg/d435/depth/image_rect_raw',
                'preview_topic': '/ghost_mgg/d435/depth/image_viz',
                'min_depth_m': 0.2,
                'max_depth_m': 1.4,
            }
        ],
        output='screen',
    )

    observation_cache = Node(
        package='ghost_mgg_sim',
        executable='observation_cache_node',
        name='m2_pose_tuning_observation_cache_node',
        parameters=[
            {
                'use_sim_time': True,
                'cache_namespace': 'm2_d435_tuning',
                'observation_ref_topic': '/ghost_mgg/observations/latest',
                'rgb_topic': '/ghost_mgg/d435/color/image_raw',
                'depth_topic': '/ghost_mgg/d435/depth/image_rect_raw',
                'color_camera_info_topic': '/ghost_mgg/d435/color/camera_info',
                'depth_camera_info_topic': '/ghost_mgg/d435/depth/camera_info',
                'rgb_frame_id': 'd435_color_optical_frame',
                'depth_frame_id': 'd435_depth_optical_frame',
                'mask_frame_id': 'd435_color_optical_frame',
                'max_age_sec': 1.0,
            }
        ],
        output='screen',
    )

    rviz_config = PathJoinSubstitution(
        [
            sim_share,
            'rviz',
            'm2_camera_inspect.rviz',
        ]
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='m2_pose_tuning_rviz',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription(
        [
            *declare_pose_arguments(),
            SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path),
            gz_sim,
            clock_bridge,
            robot_state_publisher,
            robot_description_publisher,
            TimerAction(period=3.0, actions=[spawn_robot, spawn_stand, spawn_camera]),
            TimerAction(
                period=6.0,
                actions=[
                    joint_state_broadcaster_spawner,
                    arm_controller_spawner,
                    gripper_controller_spawner,
                ],
            ),
            TimerAction(period=10.0, actions=[send_pregrasp]),
            camera_bridge,
            world_to_base_tf,
            world_to_d435_link_tf,
            d435_link_to_color_frame_tf,
            d435_link_to_depth_frame_tf,
            color_frame_to_optical_tf,
            depth_frame_to_optical_tf,
            depth_to_points,
            depth_preview,
            observation_cache,
            rviz,
        ]
    )
