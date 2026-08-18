from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_PKG = REPO_ROOT / "src" / "ghost_mgg_real"


def test_m6_rough_tf_launch_wires_camera_robot_moveit_and_static_tf():
    launch_path = REAL_PKG / "launch" / "m6_rough_tf_shadow_inspect.launch.py"
    source = launch_path.read_text(encoding="utf-8")

    for required in [
        "d435_realsense.launch.py",
        "m6_mycobot_state_bridge.launch.py",
        "m6_shadow_move_group.launch.py",
        "tf2_ros",
        "static_transform_publisher",
        "base_link",
        "camera_link",
        "camera_x",
        "camera_y",
        "camera_z",
        "camera_roll",
        "camera_pitch",
        "camera_yaw",
        "show_inspect_rviz",
        "start_state_bridge",
        "use_fake_joint_states",
        'DeclareLaunchArgument("start_state_bridge", default_value="true")',
        'DeclareLaunchArgument("use_fake_joint_states", default_value="false")',
        "robot_host",
        "robot_user",
        "ssh_command",
        "BatchMode=yes",
        "false",
        "true",
        "allow_trajectory_execution",
        "m6_rough_tf_shadow.rviz",
        "640x480x30",
    ]:
        assert required in source

    for forbidden in [
        "moveit_sim_execute_server",
        "execute_m4_joint_hypothesis",
        "ExecuteGrasp",
        "FollowJointTrajectory",
        "ros2 action send_goal",
    ]:
        assert forbidden not in source


def test_m6_rough_tf_defaults_match_current_site_calibration():
    launch_path = REAL_PKG / "launch" / "m6_rough_tf_shadow_inspect.launch.py"
    script_path = REPO_ROOT / "scripts" / "run_m6_rough_tf_shadow_inspect.sh"
    doc_path = REPO_ROOT / "docs" / "m6_rough_tf_shadow_mode.md"

    launch_source = launch_path.read_text(encoding="utf-8")
    script_source = script_path.read_text(encoding="utf-8")
    doc_source = doc_path.read_text(encoding="utf-8")

    expected_defaults = {
        "camera_x": "0.00",
        "camera_y": "0.39",
        "camera_z": "0.16",
        "camera_roll": "0.0",
        "camera_pitch": "0.83",
        "camera_yaw": "-1.57",
    }

    for name, value in expected_defaults.items():
        assert f'DeclareLaunchArgument("{name}", default_value="{value}")' in launch_source
        env_name = name.upper()
        assert f"{name}:={value}" in doc_source

    for required in [
        'camera_x="${CAMERA_X:-0.00}"',
        'camera_y="${CAMERA_Y:-0.39}"',
        'camera_z="${CAMERA_Z:-0.16}"',
        'camera_roll="${CAMERA_ROLL:-0.0}"',
        'camera_pitch="${CAMERA_PITCH:-0.83}"',
        'camera_yaw="${CAMERA_YAW:--1.57}"',
        "M6 rough camera_to_base TF",
        "camera_x=${camera_x}",
        "camera_yaw=${camera_yaw}",
    ]:
        assert required in script_source


def test_m6_rough_tf_rviz_shows_robot_tf_camera_pointcloud_in_base_frame():
    rviz_path = REAL_PKG / "rviz" / "m6_rough_tf_shadow.rviz"
    source = rviz_path.read_text(encoding="utf-8")

    for required in [
        "Fixed Frame: base_link",
        "RobotModel",
        "TF",
        "Enabled: false",
        "Camera Link Axes",
        "Reference Frame: camera_link",
        "Base Link Axes",
        "Reference Frame: base_link",
        "D435 Depth Color Points",
        "/camera/camera/depth/color/points",
        "D435 Color Image",
        "/camera/camera/color/image_raw",
    ]:
        assert required in source

    for hidden_frame in [
        "camera_depth_optical_frame:",
        "camera_color_optical_frame:",
        "camera_infra1_optical_frame:",
        "camera_infra2_optical_frame:",
    ]:
        assert hidden_frame not in source


def test_m6_rough_tf_script_and_docs_explain_visual_only_shadow_flow():
    script_path = REPO_ROOT / "scripts" / "run_m6_rough_tf_shadow_inspect.sh"
    doc_path = REPO_ROOT / "docs" / "m6_rough_tf_shadow_mode.md"

    script_source = script_path.read_text(encoding="utf-8")
    doc_source = doc_path.read_text(encoding="utf-8")

    assert "m6_rough_tf_shadow_inspect.launch.py" in script_source
    assert "camera_x:=" in script_source
    assert "camera_pitch:=" in script_source
    assert "setup_m6_mycobot_ssh_key.sh" in script_source
    assert "粗略 TF" in doc_source
    assert "默认连接 myCobot SSH" in doc_source
    assert "不会发送真实运动命令" in doc_source
    assert "base_link -> camera_link" in doc_source


def test_m6_ssh_key_setup_script_creates_key_and_uses_batch_verification():
    script_path = REPO_ROOT / "scripts" / "setup_m6_mycobot_ssh_key.sh"
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "ghost_mgg_m6_mycobot_ed25519",
        "ssh-keygen",
        "ssh-copy-id",
        "BatchMode=yes",
        "10.42.0.169",
        "elephant",
        "GHOST-MGG M6 myCobot",
        "ssh_config",
    ]:
        assert required in source
