from pathlib import Path


SIM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
BRINGUP_DIR = REPO_ROOT / "src" / "ghost_mgg_bringup"


def test_m4_graspability_marker_node_is_registered():
    cmake_text = (SIM_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_executable(m4_graspability_markers_node" in cmake_text
    assert "src/m4_graspability_markers_node.cpp" in cmake_text
    assert "m4_graspability_markers_node" in cmake_text


def test_m4_graspability_marker_node_reads_dryrun_fields():
    source = (SIM_DIR / "src" / "m4_graspability_markers_node.cpp").read_text(
        encoding="utf-8"
    )

    for required in [
        "/ghost_mgg/m4_graspability_markers",
        "reports/m4_graspability_dryrun/graspability.json",
        "grasp_width_axis",
        "required_gripper_width_m",
        "pregrasp_z_m",
        "source_center_u",
        "source_center_v",
        "failure_reason",
        "M4_graspability",
        "m4_grasp_valid_top_grasp",
        "m4_grasp_invalid_reject",
        "m4_grasp_width_bar",
        "m4_graspability_panel_text",
        "MarkerArray",
    ]:
        assert required in source
    assert "M4 graspability" not in source
    assert 'label << "valid "' not in source
    assert "point(row.grasp_x_m, row.grasp_y_m, row.pregrasp_z_m)" in source
    assert "projected_anchor(" not in source


def test_m4_inspect_launch_starts_graspability_marker_node():
    launch_text = (
        BRINGUP_DIR / "launch" / "m4_offline_ranking_inspect.launch.py"
    ).read_text(encoding="utf-8")

    for required in [
        "DeclareLaunchArgument('graspability_report_path'",
        "m4_graspability_markers_node",
        "'marker_topic': '/ghost_mgg/m4_graspability_markers'",
        "'graspability_report_path': graspability_report_path",
        "'frame_id': 'world'",
    ]:
        assert required in launch_text


def test_rviz_config_displays_m4_graspability_markers():
    rviz_text = (SIM_DIR / "rviz" / "m3_failure_camera_inspect.rviz").read_text(
        encoding="utf-8"
    )

    assert "M4 Graspability Markers" in rviz_text
    assert "/ghost_mgg/m4_graspability_markers" in rviz_text


def test_m4_marker_smoke_checks_graspability_topic():
    smoke_text = (REPO_ROOT / "scripts" / "smoke_m4_offline_ranking_markers.sh").read_text(
        encoding="utf-8"
    )

    for required in [
        "/ghost_mgg/m4_graspability_markers visualization_msgs/msg/MarkerArray",
        "m4_grasp_valid_top_grasp",
        "m4_grasp_invalid_reject",
        "M4_graspability",
    ]:
        assert required in smoke_text
