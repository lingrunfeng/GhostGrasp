from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_camera = LaunchConfiguration("start_camera")
    start_state_bridge = LaunchConfiguration("start_state_bridge")
    start_moveit_shadow = LaunchConfiguration("start_moveit_shadow")
    show_rviz = LaunchConfiguration("show_rviz")
    use_fake_joint_states = LaunchConfiguration("use_fake_joint_states")

    camera_x = LaunchConfiguration("camera_x")
    camera_y = LaunchConfiguration("camera_y")
    camera_z = LaunchConfiguration("camera_z")
    camera_roll = LaunchConfiguration("camera_roll")
    camera_pitch = LaunchConfiguration("camera_pitch")
    camera_yaw = LaunchConfiguration("camera_yaw")
    robot_host = LaunchConfiguration("robot_host")
    robot_user = LaunchConfiguration("robot_user")
    ssh_command = LaunchConfiguration("ssh_command")

    live_hypothesis_topic = LaunchConfiguration("live_hypothesis_topic")
    live_marker_topic = LaunchConfiguration("live_marker_topic")
    table_z_m = LaunchConfiguration("table_z_m")
    top_k = LaunchConfiguration("top_k")
    max_visible_hypotheses = LaunchConfiguration("max_visible_hypotheses")
    depth_margin_m = LaunchConfiguration("depth_margin_m")
    min_component_area_px = LaunchConfiguration("min_component_area_px")
    primitive_height_m = LaunchConfiguration("primitive_height_m")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")
    processing_stride = LaunchConfiguration("processing_stride")
    target_color_hint = LaunchConfiguration("target_color_hint")
    enable_target_lock = LaunchConfiguration("enable_target_lock")
    enable_table_foreground_gate = LaunchConfiguration("enable_table_foreground_gate")
    foreground_min_height_m = LaunchConfiguration("foreground_min_height_m")
    foreground_max_height_m = LaunchConfiguration("foreground_max_height_m")
    workspace_min_x_m = LaunchConfiguration("workspace_min_x_m")
    workspace_max_x_m = LaunchConfiguration("workspace_max_x_m")
    workspace_min_y_m = LaunchConfiguration("workspace_min_y_m")
    workspace_max_y_m = LaunchConfiguration("workspace_max_y_m")
    stable_foreground_min_observations = LaunchConfiguration(
        "stable_foreground_min_observations"
    )
    stable_foreground_max_center_jump_m = LaunchConfiguration(
        "stable_foreground_max_center_jump_m"
    )
    stable_foreground_max_misses = LaunchConfiguration("stable_foreground_max_misses")
    stable_shape_switch_observations = LaunchConfiguration("stable_shape_switch_observations")
    stable_smoothing_alpha = LaunchConfiguration("stable_smoothing_alpha")
    stable_dimension_smoothing_alpha = LaunchConfiguration("stable_dimension_smoothing_alpha")
    stable_dimension_max_step_ratio = LaunchConfiguration("stable_dimension_max_step_ratio")
    external_target_mask_topic = LaunchConfiguration("external_target_mask_topic")
    require_external_target_mask = LaunchConfiguration("require_external_target_mask")

    d435_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ghost_mgg_real"), "launch", "d435_realsense.launch.py"]
            )
        ),
        launch_arguments={
            "depth_profile": LaunchConfiguration("depth_profile"),
            "color_profile": LaunchConfiguration("color_profile"),
            "infra_profile": LaunchConfiguration("infra_profile"),
            "enable_infra": "true",
            "enable_pointcloud": "true",
            "enable_align_depth": "true",
            "initial_reset": LaunchConfiguration("initial_reset"),
        }.items(),
        condition=IfCondition(start_camera),
    )

    state_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ghost_mgg_real"),
                    "launch",
                    "m6_mycobot_state_bridge.launch.py",
                ]
            )
        ),
        launch_arguments={
            "robot_host": robot_host,
            "robot_user": robot_user,
            "ssh_command": ssh_command,
        }.items(),
        condition=IfCondition(start_state_bridge),
    )

    moveit_shadow_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ghost_mgg_moveit_config"),
                    "launch",
                    "m6_shadow_move_group.launch.py",
                ]
            )
        ),
        launch_arguments={
            "show_rviz": "false",
            "use_fake_joint_states": use_fake_joint_states,
        }.items(),
        condition=IfCondition(start_moveit_shadow),
    )

    rough_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="m8_rough_camera_to_base_tf",
        output="screen",
        arguments=[
            "--x",
            camera_x,
            "--y",
            camera_y,
            "--z",
            camera_z,
            "--roll",
            camera_roll,
            "--pitch",
            camera_pitch,
            "--yaw",
            camera_yaw,
            "--frame-id",
            "base_link",
            "--child-frame-id",
            "camera_link",
        ],
    )

    transparent_detector = Node(
        package="ghost_mgg_sim",
        executable="m8_transparent_detector_node.py",
        name="m8_transparent_detector_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "color_topic": "/camera/camera/color/image_raw",
                "boxes_topic": "/ghost_mgg/d435/transparent_boxes",
                "min_confidence": 0.25,
                "max_rate_hz": 6.0,
            }
        ],
    )

    live_hypothesis_publisher = Node(
        package="ghost_mgg_sim",
        executable="m4_live_hypothesis_publisher_node.py",
        name="m8_real_live_hypothesis_publisher_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "raw_depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
                "corrupted_depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
                "color_topic": "/camera/camera/color/image_raw",
                "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
                "mask_topic": "/ghost_mgg/d435/target_mask",
                "external_mask_topic": external_target_mask_topic,
                "require_external_mask": ParameterValue(
                    require_external_target_mask, value_type=bool
                ),
                "hypothesis_topic": live_hypothesis_topic,
                "frame_id": "camera_color_optical_frame",
                "base_frame_id": "base_link",
                "table_z_m": ParameterValue(table_z_m, value_type=float),
                "top_k": ParameterValue(top_k, value_type=int),
                "depth_margin_m": ParameterValue(depth_margin_m, value_type=float),
                "min_component_area_px": ParameterValue(min_component_area_px, value_type=int),
                "primitive_height_m": ParameterValue(primitive_height_m, value_type=float),
                "publish_rate_hz": ParameterValue(publish_rate_hz, value_type=float),
                "processing_stride": ParameterValue(processing_stride, value_type=int),
                "target_color_hint": target_color_hint,
                "enable_target_lock": ParameterValue(enable_target_lock, value_type=bool),
                "max_locked_center_distance_px": 60.0,
                "enable_table_foreground_gate": ParameterValue(
                    enable_table_foreground_gate, value_type=bool
                ),
                "foreground_min_height_m": ParameterValue(
                    foreground_min_height_m, value_type=float
                ),
                "foreground_max_height_m": ParameterValue(
                    foreground_max_height_m, value_type=float
                ),
                "workspace_min_x_m": ParameterValue(workspace_min_x_m, value_type=float),
                "workspace_max_x_m": ParameterValue(workspace_max_x_m, value_type=float),
                "workspace_min_y_m": ParameterValue(workspace_min_y_m, value_type=float),
                "workspace_max_y_m": ParameterValue(workspace_max_y_m, value_type=float),
                "stable_foreground_min_observations": ParameterValue(
                    stable_foreground_min_observations, value_type=int
                ),
                "stable_foreground_max_center_jump_m": ParameterValue(
                    stable_foreground_max_center_jump_m, value_type=float
                ),
                "stable_foreground_max_misses": ParameterValue(
                    stable_foreground_max_misses, value_type=int
                ),
                "stable_output_hold_misses": ParameterValue(
                    LaunchConfiguration("stable_output_hold_misses"), value_type=int
                ),
                "min_hypothesis_score": ParameterValue(
                    LaunchConfiguration("min_hypothesis_score"), value_type=float
                ),
                "stable_shape_switch_observations": ParameterValue(
                    stable_shape_switch_observations, value_type=int
                ),
                "stable_smoothing_alpha": ParameterValue(
                    stable_smoothing_alpha, value_type=float
                ),
                "stable_dimension_smoothing_alpha": ParameterValue(
                    stable_dimension_smoothing_alpha, value_type=float
                ),
                "stable_dimension_max_step_ratio": ParameterValue(
                    stable_dimension_max_step_ratio, value_type=float
                ),
            }
        ],
    )

    live_contract_markers = Node(
        package="ghost_mgg_sim",
        executable="hypothesis_markers_node",
        name="m8_real_shadow_markers_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "hypotheses_topic": live_hypothesis_topic,
                "marker_topic": live_marker_topic,
                "hide_executed_hypotheses": False,
                "max_visible_hypotheses": ParameterValue(
                    max_visible_hypotheses, value_type=int
                ),
            }
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="m8_real_shadow_rviz",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("ghost_mgg_real"), "rviz", "m8_real_shadow.rviz"]
            ),
        ],
        condition=IfCondition(show_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument("start_state_bridge", default_value="false"),
            DeclareLaunchArgument("start_moveit_shadow", default_value="false"),
            DeclareLaunchArgument("show_rviz", default_value="true"),
            DeclareLaunchArgument("use_fake_joint_states", default_value="true"),
            DeclareLaunchArgument("robot_host", default_value="10.42.0.169"),
            DeclareLaunchArgument("robot_user", default_value="elephant"),
            DeclareLaunchArgument("ssh_command", default_value="ssh -o BatchMode=yes"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
            DeclareLaunchArgument("color_profile", default_value="640x480x30"),
            DeclareLaunchArgument("infra_profile", default_value="640x480x30"),
            DeclareLaunchArgument("initial_reset", default_value="false"),
            DeclareLaunchArgument("camera_x", default_value="0.00"),
            DeclareLaunchArgument("camera_y", default_value="0.39"),
            DeclareLaunchArgument("camera_z", default_value="0.16"),
            DeclareLaunchArgument("camera_roll", default_value="0.0"),
            DeclareLaunchArgument("camera_pitch", default_value="0.83"),
            DeclareLaunchArgument("camera_yaw", default_value="-1.57"),
            DeclareLaunchArgument(
                "live_hypothesis_topic", default_value="/ghost_mgg/m4_live_hypotheses"
            ),
            DeclareLaunchArgument(
                "live_marker_topic", default_value="/ghost_mgg/m8_real_shadow_markers"
            ),
            DeclareLaunchArgument("table_z_m", default_value="0.0"),
            DeclareLaunchArgument("top_k", default_value="10"),
            DeclareLaunchArgument("max_visible_hypotheses", default_value="10"),
            DeclareLaunchArgument("depth_margin_m", default_value="0.035"),
            DeclareLaunchArgument("min_component_area_px", default_value="80"),
            DeclareLaunchArgument("primitive_height_m", default_value="0.04"),
            DeclareLaunchArgument("publish_rate_hz", default_value="5.0"),
            DeclareLaunchArgument("processing_stride", default_value="2"),
            DeclareLaunchArgument("target_color_hint", default_value="none"),
            DeclareLaunchArgument("enable_target_lock", default_value="false"),
            DeclareLaunchArgument("enable_table_foreground_gate", default_value="true"),
            DeclareLaunchArgument("foreground_min_height_m", default_value="0.003"),
            DeclareLaunchArgument("foreground_max_height_m", default_value="0.200"),
            DeclareLaunchArgument("workspace_min_x_m", default_value="-0.40"),
            DeclareLaunchArgument("workspace_max_x_m", default_value="0.40"),
            DeclareLaunchArgument("workspace_min_y_m", default_value="-0.05"),
            DeclareLaunchArgument("workspace_max_y_m", default_value="0.65"),
            DeclareLaunchArgument("stable_foreground_min_observations", default_value="2"),
            DeclareLaunchArgument("stable_foreground_max_center_jump_m", default_value="0.080"),
            DeclareLaunchArgument("stable_foreground_max_misses", default_value="1"),
            DeclareLaunchArgument("stable_output_hold_misses", default_value="2"),
            DeclareLaunchArgument("min_hypothesis_score", default_value="0.05"),
            DeclareLaunchArgument("stable_shape_switch_observations", default_value="8"),
            DeclareLaunchArgument("stable_smoothing_alpha", default_value="0.45"),
            DeclareLaunchArgument("stable_dimension_smoothing_alpha", default_value="0.12"),
            DeclareLaunchArgument("stable_dimension_max_step_ratio", default_value="0.20"),
            DeclareLaunchArgument(
                "external_target_mask_topic",
                default_value="/ghost_mgg/d435/external_target_mask",
            ),
            DeclareLaunchArgument("require_external_target_mask", default_value="false"),
            d435_launch,
            state_bridge_launch,
            moveit_shadow_launch,
            rough_camera_tf,
            transparent_detector,
            live_hypothesis_publisher,
            live_contract_markers,
            rviz,
        ]
    )
