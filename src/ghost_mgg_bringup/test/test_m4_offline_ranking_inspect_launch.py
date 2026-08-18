from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_PACKAGE = REPO_ROOT / "src" / "ghost_mgg_sim"


def test_m4_offline_ranking_inspect_launch_starts_m3_scene_and_marker_node():
    launch_path = BRINGUP_DIR / "launch" / "m4_offline_ranking_inspect.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text(encoding="utf-8")

    for required in [
        "DeclareLaunchArgument('scenario_id', default_value='S6')",
        "DeclareLaunchArgument('headless', default_value='true')",
        "DeclareLaunchArgument('show_rviz', default_value='true')",
        "DeclareLaunchArgument('ranking_report_path', default_value='reports/m4_offline_ranking.json')",
        "DeclareLaunchArgument('top_k', default_value='3')",
        "m3_failure_scenario.launch.py",
        "m4_offline_ranking_markers_node",
        "'marker_topic': '/ghost_mgg/hypothesis_markers'",
        "'frame_id': 'd435_depth_optical_frame'",
        "'ranking_report_path': ranking_report_path",
        "'scenario_id': scenario_id",
        "'top_k': ParameterValue(top_k, value_type=int)",
    ]:
        assert required in launch_text


def test_m4_marker_node_is_registered_and_installed():
    cmake_text = (SIM_PACKAGE / "CMakeLists.txt").read_text(encoding="utf-8")
    node_text = (SIM_PACKAGE / "src" / "m4_offline_ranking_markers_node.cpp").read_text(
        encoding="utf-8"
    )

    assert "add_executable(m4_offline_ranking_markers_node" in cmake_text
    assert "m4_offline_ranking_markers_node" in cmake_text
    assert "MarkerArray" in node_text
    assert "m4_failure_aware_proxy" in node_text
    assert "m4_silhouette_only_proxy" in node_text
    assert "d435_depth_optical_frame" in node_text


def test_m4_marker_smoke_launches_inspect_stack_and_checks_marker_topic():
    smoke_path = REPO_ROOT / "scripts" / "smoke_m4_offline_ranking_markers.sh"
    assert smoke_path.exists()

    smoke_text = smoke_path.read_text(encoding="utf-8")

    for required in [
        "m4_offline_ranking_inspect.launch.py",
        "headless:=true",
        "show_rviz:=false",
        "/ghost_mgg/hypothesis_markers visualization_msgs/msg/MarkerArray",
        "/ghost_mgg/m4_real_ranking_markers visualization_msgs/msg/MarkerArray",
        "m4_failure_aware_proxy",
        "m4_silhouette_only_proxy",
        "m4_real_failure_aware_proxy",
        "m4_real_silhouette_only_proxy",
        "M4 offline ranking marker smoke passed",
    ]:
        assert required in smoke_text
