from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
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
    headless = LaunchConfiguration('headless')
    preferred_hypothesis_id = LaunchConfiguration('preferred_hypothesis_id')
    strict_preferred_hypothesis = LaunchConfiguration('strict_preferred_hypothesis')
    max_hypotheses = LaunchConfiguration('max_hypotheses')
    moveit_config = build_m2_moveit_config()
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
        launch_arguments={'headless': headless}.items(),
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
        name='m2_mask_extrusion_moveit_camera_bridge',
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
        name='m2_moveit_loop_robot_description_publisher',
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
        name='m2_moveit_loop_depth_to_point_cloud_node',
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
        name='m2_moveit_loop_depth_to_mono8_node',
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
                'strict_preferred_hypothesis': strict_preferred_hypothesis,
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
                # Full close overdrives the adaptive gripper through 25 mm targets.
                # This value leaves the fingertip inner gap just below target width
                # so Gazebo contacts can retain the object instead of ejecting it.
                'gripper_close_position': -0.38,
                'gripper_max_effort': 45.0,
                'gripper_motion_duration_sec': 0.60,
                'wait_for_close_gripper_result': True,
                'settle_time_sec': 0.80,
                'verify_lift_and_hold': True,
                'min_lift_z_delta': 0.010,
                'lift_hold_duration_sec': 0.50,
                'lift_hold_sample_period_sec': 0.10,
                'planning_time_sec': 5.0,
                'velocity_scaling': 0.20,
                'acceleration_scaling': 0.20,
                'use_grasp_orientation': True,
                'goal_position_tolerance_m': 0.006,
                'goal_orientation_tolerance_rad': 0.45,
                'add_m2_scene_obstacles': True,
                'scene_object_padding_m': 0.006,
                'use_cartesian_top_grasp_segments': False,
                'cartesian_step_m': 0.004,
                'cartesian_jump_threshold': 0.0,
                'min_cartesian_fraction': 0.95,
            }
        ],
        output='screen',
    )

    bt_runner = Node(
        package='ghost_mgg_bt',
        executable='bt_runner_node',
        name='m2_mask_extrusion_moveit_bt_runner',
        parameters=[
            {
                'tree_path': m2_tree,
                'backend_name': 'mask_extrusion',
                'executor_name': 'moveit_sim',
                'target_label': 'm2_sim_target',
                'shape_hint': 'unknown',
                'recover_timeout_sec': 6.0,
                'execute_timeout_sec': 65.0,
                'max_hypotheses': max_hypotheses,
                'trial_log_dir': 'log/ghost_mgg_trials/m2_mask_extrusion_moveit',
            }
        ],
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('headless', default_value='true'),
            DeclareLaunchArgument('preferred_hypothesis_id', default_value=''),
            DeclareLaunchArgument('strict_preferred_hypothesis', default_value='false'),
            DeclareLaunchArgument('max_hypotheses', default_value='4'),
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
            TimerAction(period=13.0, actions=[move_group]),
            TimerAction(period=17.0, actions=[mask_extrusion_recovery, moveit_executor, hypothesis_markers]),
            TimerAction(period=21.0, actions=[bt_runner]),
        ]
    )
