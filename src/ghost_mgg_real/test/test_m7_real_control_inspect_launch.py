from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_PKG = REPO_ROOT / "src" / "ghost_mgg_real"


def test_m7_real_control_rviz_flag_does_not_collide_with_nested_moveit_flag():
    launch_source = (
        REAL_PKG / "launch" / "m7_real_control_inspect.launch.py"
    ).read_text(encoding="utf-8")
    script_source = (REPO_ROOT / "scripts" / "run_m7_real_control_inspect.sh").read_text(
        encoding="utf-8"
    )

    assert 'LaunchConfiguration("show_m7_rviz")' in launch_source
    assert 'DeclareLaunchArgument("show_m7_rviz", default_value="true")' in launch_source
    assert '"show_rviz": "false"' in launch_source
    assert "show_m7_rviz:=" in script_source

    assert 'LaunchConfiguration("show_rviz")' not in launch_source
    assert 'DeclareLaunchArgument("show_rviz", default_value="true")' not in launch_source


def test_m7_real_control_server_parameters_use_native_types_not_strings():
    launch_source = (
        REAL_PKG / "launch" / "m7_real_control_inspect.launch.py"
    ).read_text(encoding="utf-8")

    assert '"publish_shadow_gripper_joints": True' in launch_source
    assert '"shadow_gripper_position": 0.15' in launch_source
    assert '"publish_shadow_gripper_joints": "true"' not in launch_source
    assert '"shadow_gripper_position": "0.15"' not in launch_source


def test_m7_refresh_script_exposes_explicit_target_offsets():
    script_source = (REPO_ROOT / "scripts" / "refresh_m7_real_target.sh").read_text(
        encoding="utf-8"
    )

    for required in [
        "M7_TARGET_X_OFFSET_M",
        "M7_TARGET_Y_OFFSET_M",
        "M7_TARGET_Z_OFFSET_M",
        "m7_target_offset_m",
        'target["center_x_m"]',
        'target["center_y_m"]',
        'target["center_z_m"]',
    ]:
        assert required in script_source


def test_m7_grasp_script_supports_extra_descend_without_lowering_pregrasp():
    script_source = (REPO_ROOT / "scripts" / "grasp_m7_real_once.sh").read_text(
        encoding="utf-8"
    )

    assert 'extra_descend = float(os.environ.get("M7_DESCEND_EXTRA_MM", "0"))' in script_source
    assert "grasp_z = base_grasp_z - extra_descend" in script_source
    assert "pregrasp_z = float(os.environ.get(\"M7_PREGRASP_Z_MM\", \"140.0\"))" in script_source


def test_m7_grasp_script_rejects_stale_targets_and_prints_current_mapping():
    script_source = (REPO_ROOT / "scripts" / "grasp_m7_real_once.sh").read_text(
        encoding="utf-8"
    )

    for required in [
        "M7_MAX_TARGET_AGE_SEC",
        "M7_ALLOW_STALE_TARGET",
        "target_age_sec",
        "stale target report",
        "M7_EXECUTE_TARGET_GRASP",
        "# target_center_base_m=",
        "# command_grasp_xyz_mm=",
    ]:
        assert required in script_source


def test_m7_grasp_script_exposes_base_to_robot_axis_sign_calibration():
    script_source = (REPO_ROOT / "scripts" / "grasp_m7_real_once.sh").read_text(
        encoding="utf-8"
    )

    for required in [
        "M7_ROBOT_X_FROM_BASE_Y_SIGN",
        "M7_ROBOT_Y_FROM_BASE_X_SIGN",
        "robot_x_from_base_y_sign",
        "robot_y_from_base_x_sign",
        'os.environ.get("M7_ROBOT_Y_FROM_BASE_X_SIGN", "-1")',
        "# base_to_robot_signs=",
    ]:
        assert required in script_source
