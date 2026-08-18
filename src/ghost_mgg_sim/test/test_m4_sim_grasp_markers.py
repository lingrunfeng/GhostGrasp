from pathlib import Path


SIM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
BRINGUP_DIR = REPO_ROOT / "src" / "ghost_mgg_bringup"


def test_m4_sim_grasp_target_config_tracks_tabletop_objects():
    config_text = (SIM_DIR / "config" / "m4_sim_grasp_targets.json").read_text(
        encoding="utf-8"
    )

    for required in [
        '"schema_version": "m4_sim_grasp_targets_v1"',
        '"target_id": "red_cube"',
        '"target_id": "blue_cylinder"',
        '"target_id": "green_cylinder"',
        '"target_id": "glass_block"',
        '"center_x_m": 0.070172',
        '"center_y_m": 0.012156',
        '"center_x_m": 0.115',
        '"center_y_m": 0.075',
        '"center_x_m": -0.035',
        '"center_y_m": -0.047344',
        '"center_x_m": 0.002',
        '"center_y_m": 0.1',
        '"top_grasp"',
        '"pregrasp_clearance_m"',
    ]:
        assert required in config_text


def test_m4_sim_grasp_marker_node_is_registered_and_reads_target_config():
    cmake_text = (SIM_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    source = (SIM_DIR / "src" / "m4_sim_grasp_markers_node.cpp").read_text(
        encoding="utf-8"
    )

    assert "add_executable(m4_sim_grasp_markers_node" in cmake_text
    assert "src/m4_sim_grasp_markers_node.cpp" in cmake_text
    assert "m4_sim_grasp_markers_node" in cmake_text
    for required in [
        "/ghost_mgg/m4_sim_grasp_markers",
        "config/m4_sim_grasp_targets.json",
        "m4_sim_bound_top_grasp",
        "m4_sim_bound_width_bar",
        "m4_sim_bound_target_proxy",
        "M4_sim_grasp",
        "target_id",
        "center_x_m",
        "pregrasp_clearance_m",
        "MarkerArray",
    ]:
        assert required in source


def test_m4_inspect_launch_starts_sim_bound_grasp_marker_node():
    launch_text = (
        BRINGUP_DIR / "launch" / "m4_offline_ranking_inspect.launch.py"
    ).read_text(encoding="utf-8")

    for required in [
        "DeclareLaunchArgument",
        "'sim_grasp_targets_path'",
        "m4_sim_grasp_markers_node",
        "'marker_topic': '/ghost_mgg/m4_sim_grasp_markers'",
        "'targets_path': sim_grasp_targets_path",
        "'frame_id': 'world'",
    ]:
        assert required in launch_text


def test_rviz_config_displays_m4_sim_grasp_markers():
    rviz_text = (SIM_DIR / "rviz" / "m3_failure_camera_inspect.rviz").read_text(
        encoding="utf-8"
    )

    assert "M4 Sim Grasp Markers" in rviz_text
    assert "/ghost_mgg/m4_sim_grasp_markers" in rviz_text


def test_m4_marker_smoke_checks_sim_bound_grasp_topic():
    smoke_text = (
        REPO_ROOT / "scripts" / "smoke_m4_offline_ranking_markers.sh"
    ).read_text(encoding="utf-8")

    for required in [
        "/ghost_mgg/m4_sim_grasp_markers visualization_msgs/msg/MarkerArray",
        "m4_sim_bound_top_grasp",
        "m4_sim_bound_target_proxy",
        "red_cube",
        "glass_block",
    ]:
        assert required in smoke_text
