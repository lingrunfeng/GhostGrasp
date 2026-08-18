from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    preferred_hypothesis_id = LaunchConfiguration('preferred_hypothesis_id')
    description_share = FindPackageShare('ghost_mgg_mycobot_description')
    mycobot_xacro = PathJoinSubstitution(
        [
            description_share,
            'urdf',
            'robots',
            'mycobot_280.urdf.xacro',
        ]
    )
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
        launch_arguments={'headless': 'true'}.items(),
    )

    m2_tree = PathJoinSubstitution(
        [
            FindPackageShare('ghost_mgg_bt'),
            'trees',
            'm2_sim_closed_loop.xml',
        ]
    )

    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='m2_mask_extrusion_camera_bridge',
        arguments=[
            '/ghost_mgg/d435/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/ghost_mgg/d435/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/ghost_mgg/d435/depth/image_rect_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/ghost_mgg/d435/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
    )

    robot_description_publisher = Node(
        package='ghost_mgg_sim',
        executable='robot_description_publisher_node',
        name='m2_mask_extrusion_robot_description_publisher',
        parameters=[
            {
                'use_sim_time': True,
                'robot_description': robot_description,
            }
        ],
        output='screen',
    )

    world_to_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_world_to_base_link_tf',
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

    world_to_d435_link_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_world_to_d435_link_tf',
        arguments=[
            '--x', '0.286775',
            '--y', '-0.079936',
            '--z', '1.105590',
            '--roll', '0.0',
            '--pitch', '0.950000',
            '--yaw', '2.754700',
            '--frame-id', 'world',
            '--child-frame-id', 'd435_link',
        ],
    )

    d435_link_to_color_frame_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_d435_link_to_color_frame_tf',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0',
            '--frame-id', 'd435_link',
            '--child-frame-id', 'd435_color_frame',
        ],
    )

    d435_link_to_depth_frame_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_d435_link_to_depth_frame_tf',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0',
            '--frame-id', 'd435_link',
            '--child-frame-id', 'd435_depth_frame',
        ],
    )

    color_frame_to_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_d435_color_frame_to_optical_tf',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '-1.5708',
            '--pitch', '0.0',
            '--yaw', '-1.5708',
            '--frame-id', 'd435_color_frame',
            '--child-frame-id', 'd435_color_optical_frame',
        ],
    )

    depth_frame_to_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='m2_d435_depth_frame_to_optical_tf',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '-1.5708',
            '--pitch', '0.0',
            '--yaw', '-1.5708',
            '--frame-id', 'd435_depth_frame',
            '--child-frame-id', 'd435_depth_optical_frame',
        ],
    )

    depth_to_points = Node(
        package='ghost_mgg_sim',
        executable='depth_to_point_cloud_node',
        name='m2_mask_extrusion_depth_to_point_cloud_node',
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
        name='m2_mask_extrusion_depth_to_mono8_node',
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
        name='m2_observation_cache_node',
        parameters=[
            {
                'use_sim_time': True,
                'cache_namespace': 'm2_d435',
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

    mask_extrusion_recovery = Node(
        package='ghost_mgg_backends',
        executable='mask_extrusion_recovery_server',
        name='m2_mask_extrusion_recovery_server',
        parameters=[
            {
                'action_name': '/geometry_backends/mask_extrusion/recover',
                'hypotheses_topic': '/ghost_mgg/hypotheses/mask_extrusion',
                'mode': 'success',
                'preferred_hypothesis_id': preferred_hypothesis_id,
                'response_delay_ms': 10,
            }
        ],
        output='screen',
    )

    hypothesis_markers = Node(
        package='ghost_mgg_sim',
        executable='hypothesis_markers_node',
        name='m2_hypothesis_markers_node',
        parameters=[
            {
                'hypotheses_topic': '/ghost_mgg/hypotheses/mask_extrusion',
                'marker_topic': '/ghost_mgg/hypothesis_markers',
            }
        ],
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
        name='m2_mask_extrusion_bt_runner',
        parameters=[
            {
                'tree_path': m2_tree,
                'backend_name': 'mask_extrusion',
                'executor_name': 'mycobot_sim',
                'target_label': 'm2_sim_target',
                'shape_hint': 'unknown',
                'recover_timeout_sec': 6.0,
                'execute_timeout_sec': 12.0,
                'max_hypotheses': 4,
                'trial_log_dir': 'log/ghost_mgg_trials/m2_mask_extrusion',
            }
        ],
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('preferred_hypothesis_id', default_value=''),
            m2_scene,
            camera_bridge,
            robot_description_publisher,
            world_to_base_tf,
            world_to_d435_link_tf,
            d435_link_to_color_frame_tf,
            d435_link_to_depth_frame_tf,
            color_frame_to_optical_tf,
            depth_frame_to_optical_tf,
            depth_to_points,
            depth_preview,
            observation_cache,
            TimerAction(period=12.0, actions=[mask_extrusion_recovery, mycobot_executor, hypothesis_markers]),
            TimerAction(period=16.0, actions=[bt_runner]),
        ]
    )
