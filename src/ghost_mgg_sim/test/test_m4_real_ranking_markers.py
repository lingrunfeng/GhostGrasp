from pathlib import Path


SIM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
BRINGUP_DIR = REPO_ROOT / "src" / "ghost_mgg_bringup"


def test_m4_real_ranking_marker_node_is_registered():
    cmake_text = (SIM_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_executable(m4_real_ranking_markers_node" in cmake_text
    assert "src/m4_real_ranking_markers_node.cpp" in cmake_text
    assert "m4_real_ranking_markers_node" in cmake_text


def test_m4_real_ranking_marker_node_reads_failure_breakdown():
    source = (SIM_DIR / "src" / "m4_real_ranking_markers_node.cpp").read_text(
        encoding="utf-8"
    )

    for required in [
        "/ghost_mgg/m4_real_ranking_markers",
        "reports/m5_real_d435_ranking/m5_real_ranking.json",
        "failure_inside_hole",
        "failure_inside_table_leakage",
        "failure_outside_hole_penalty",
        "failure_outside_table_leakage_penalty",
        "m4_real_failure_aware_proxy",
        "m4_real_silhouette_only_proxy",
        "M4_real_ranking",
        "MarkerArray",
    ]:
        assert required in source
    assert "M4 real ranking" not in source
    assert '<< " " << row.hypothesis_id' not in source
    assert '<< " T="' not in source


def test_m4_offline_launch_starts_real_ranking_marker_node():
    launch_text = (
        BRINGUP_DIR / "launch" / "m4_offline_ranking_inspect.launch.py"
    ).read_text(encoding="utf-8")

    for required in [
        "DeclareLaunchArgument('real_scene_id'",
        "DeclareLaunchArgument('real_ranking_report_path'",
        "m4_real_ranking_markers_node",
        "'marker_topic': '/ghost_mgg/m4_real_ranking_markers'",
        "'ranking_report_path': real_ranking_report_path",
        "'scene_id': real_scene_id",
    ]:
        assert required in launch_text


def test_rviz_config_displays_m4_real_ranking_markers():
    rviz_text = (SIM_DIR / "rviz" / "m3_failure_camera_inspect.rviz").read_text(
        encoding="utf-8"
    )

    assert "M4 Real Ranking Markers" in rviz_text
    assert "/ghost_mgg/m4_real_ranking_markers" in rviz_text
