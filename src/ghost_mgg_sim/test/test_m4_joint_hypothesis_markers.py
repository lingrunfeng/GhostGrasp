from pathlib import Path


SIM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
BRINGUP_DIR = REPO_ROOT / "src" / "ghost_mgg_bringup"


def test_joint_hypothesis_marker_node_is_installed_and_reads_reports():
    cmake_text = (SIM_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    marker_node = SIM_DIR / "scripts" / "m4_joint_hypothesis_markers_node.py"
    source = marker_node.read_text(encoding="utf-8")

    assert marker_node.exists()
    assert "scripts/m4_joint_hypothesis_markers_node.py" in cmake_text
    for required in [
        "/ghost_mgg/m4_joint_hypothesis_markers",
        "m4_joint_rank_text",
        "m4_joint_executable_target",
        "m4_joint_summary_text",
        "joint_hypotheses.json",
        "m4_sim_grasp_targets.json",
        "executed_hypotheses_topic",
        "/ghost_mgg/m4_executed_hypotheses",
        "executed_hypothesis_ids",
        "reset_executed_hypotheses",
        "executed_success",
        "DONE",
        "candidate",
        "executable",
        "transient_local",
    ]:
        assert required in source


def test_m4_inspect_launch_and_rviz_show_joint_hypothesis_markers():
    launch_text = (
        BRINGUP_DIR / "launch" / "m4_offline_ranking_inspect.launch.py"
    ).read_text(encoding="utf-8")
    rviz_text = (SIM_DIR / "rviz" / "m3_failure_camera_inspect.rviz").read_text(
        encoding="utf-8"
    )

    for required in [
        "joint_hypothesis_report_path",
        "m4_joint_hypothesis_markers_node.py",
        "'marker_topic': '/ghost_mgg/m4_joint_hypothesis_markers'",
        "'executed_hypotheses_topic': executed_hypotheses_topic",
        "'report_path': joint_hypothesis_report_path",
    ]:
        assert required in launch_text

    assert "M4 Joint Hypothesis Markers" in rviz_text
    assert "/ghost_mgg/m4_joint_hypothesis_markers" in rviz_text


def test_m4_marker_smoke_checks_joint_hypothesis_marker_topic():
    smoke_text = (
        REPO_ROOT / "scripts" / "smoke_m4_offline_ranking_markers.sh"
    ).read_text(encoding="utf-8")

    for required in [
        "run_m4_joint_hypothesis_report.sh",
        "joint_hypothesis_report_path:=",
        "/ghost_mgg/m4_joint_hypothesis_markers visualization_msgs/msg/MarkerArray",
        "m4_joint_rank_text",
        "m4_joint_summary_text",
        "executable",
        "candidate",
    ]:
        assert required in smoke_text
