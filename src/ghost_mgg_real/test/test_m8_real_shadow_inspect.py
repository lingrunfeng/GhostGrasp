from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_PKG = REPO_ROOT / "src" / "ghost_mgg_real"


def test_m8_real_shadow_launch_wires_d435_tf_m8_hypotheses_and_markers():
    launch_path = REAL_PKG / "launch" / "m8_real_shadow_inspect.launch.py"
    source = launch_path.read_text(encoding="utf-8")

    for required in [
        "d435_realsense.launch.py",
        "m6_shadow_move_group.launch.py",
        "m6_mycobot_state_bridge.launch.py",
        "m4_live_hypothesis_publisher_node.py",
        "hypothesis_markers_node",
        "/camera/camera/aligned_depth_to_color/image_raw",
        "/camera/camera/aligned_depth_to_color/camera_info",
        "/camera/camera/color/image_raw",
        "/ghost_mgg/m4_live_hypotheses",
        "/ghost_mgg/m8_real_shadow_markers",
        "camera_color_optical_frame",
        "base_link",
        "table_z_m",
        "enable_target_lock",
        "target_color_hint",
        "processing_stride",
        "enable_table_foreground_gate",
        "foreground_min_height_m",
        "foreground_max_height_m",
        "workspace_min_x_m",
        "workspace_max_y_m",
        "stable_foreground_min_observations",
        "stable_foreground_max_center_jump_m",
        "stable_shape_switch_observations",
        "stable_smoothing_alpha",
        "stable_dimension_smoothing_alpha",
        "stable_dimension_max_step_ratio",
        "m8_real_shadow.rviz",
        'DeclareLaunchArgument("start_state_bridge", default_value="false")',
        'DeclareLaunchArgument("start_moveit_shadow", default_value="false")',
        'DeclareLaunchArgument("use_fake_joint_states", default_value="true")',
    ]:
        assert required in source

    for forbidden in [
        "m7_mycobot_control_server",
        "moveit_sim_execute_server",
        "execute_m4_joint_hypothesis_action",
        "ExecuteGrasp",
        "FollowJointTrajectory",
        "ros2 action send_goal",
    ]:
        assert forbidden not in source


def test_m8_real_shadow_script_uses_site_camera_defaults_without_ssh_gate():
    script_path = REPO_ROOT / "scripts" / "run_m8_real_shadow_inspect.sh"
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "m8_real_shadow_inspect.launch.py",
        'camera_x="${CAMERA_X:-0.00}"',
        'camera_y="${CAMERA_Y:-0.39}"',
        'camera_z="${CAMERA_Z:-0.16}"',
        'camera_roll="${CAMERA_ROLL:-0.0}"',
        'camera_pitch="${CAMERA_PITCH:-0.83}"',
        'camera_yaw="${CAMERA_YAW:--1.57}"',
        'table_z_m="${TABLE_Z_M:-0.0}"',
        'target_color_hint="${TARGET_COLOR_HINT:-none}"',
        'enable_target_lock="${ENABLE_TARGET_LOCK:-false}"',
        'enable_table_foreground_gate="${ENABLE_TABLE_FOREGROUND_GATE:-true}"',
        'foreground_min_height_m="${FOREGROUND_MIN_HEIGHT_M:-0.003}"',
        'foreground_max_height_m="${FOREGROUND_MAX_HEIGHT_M:-0.200}"',
        'workspace_min_x_m="${WORKSPACE_MIN_X_M:--0.40}"',
        'workspace_max_y_m="${WORKSPACE_MAX_Y_M:-0.65}"',
        'stable_foreground_min_observations="${STABLE_FOREGROUND_MIN_OBSERVATIONS:-2}"',
        'stable_foreground_max_center_jump_m="${STABLE_FOREGROUND_MAX_CENTER_JUMP_M:-0.080}"',
        'stable_shape_switch_observations="${STABLE_SHAPE_SWITCH_OBSERVATIONS:-8}"',
        'stable_smoothing_alpha="${STABLE_SMOOTHING_ALPHA:-0.45}"',
        'stable_dimension_smoothing_alpha="${STABLE_DIMENSION_SMOOTHING_ALPHA:-0.12}"',
        'stable_dimension_max_step_ratio="${STABLE_DIMENSION_MAX_STEP_RATIO:-0.20}"',
        'start_state_bridge="${START_STATE_BRIDGE:-false}"',
        'start_moveit_shadow="${START_MOVEIT_SHADOW:-false}"',
        'start_moveit_shadow:="${start_moveit_shadow}"',
        "M8 real D435 shadow-mode",
    ]:
        assert required in source

    for forbidden in [
        "setup_m6_mycobot_ssh_key.sh",
        "ssh -o BatchMode=yes",
        "M7_EXECUTE",
        "grasp_m7_real_once",
    ]:
        assert forbidden not in source


def test_m8_real_shadow_rviz_shows_pointcloud_live_hypotheses_robot_and_images():
    rviz_path = REAL_PKG / "rviz" / "m8_real_shadow.rviz"
    source = rviz_path.read_text(encoding="utf-8")

    for required in [
        "Fixed Frame: base_link",
        "RobotModel",
        "Base Link Axes",
        "Camera Link Axes",
        "/camera/camera/depth/color/points",
        "/camera/camera/color/image_raw",
        "/camera/camera/aligned_depth_to_color/image_raw",
        "/ghost_mgg/m8_real_shadow_markers",
        "/ghost_mgg/d435/target_mask",
    ]:
        assert required in source


def test_m8_real_shadow_assets_are_installed_and_depend_on_sim_marker_tools():
    cmake_source = (REAL_PKG / "CMakeLists.txt").read_text(encoding="utf-8")
    package_source = (REAL_PKG / "package.xml").read_text(encoding="utf-8")

    assert "test_m8_real_shadow_inspect" in cmake_source
    assert "<exec_depend>ghost_mgg_sim</exec_depend>" in package_source
