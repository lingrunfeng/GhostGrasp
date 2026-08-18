from pathlib import Path


SIM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
BRINGUP_DIR = REPO_ROOT / "src" / "ghost_mgg_bringup"


def test_joint_hypothesis_publisher_node_maps_report_to_contract_topic():
    cmake_text = (SIM_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    publisher_node = SIM_DIR / "scripts" / "m4_joint_hypothesis_publisher_node.py"
    assert publisher_node.exists()

    source = publisher_node.read_text(encoding="utf-8")
    assert "scripts/m4_joint_hypothesis_publisher_node.py" in cmake_text
    for required in [
        "GeometryHypothesisArray",
        "GeometryHypothesis",
        "GraspCandidate",
        "ScoreBreakdown",
        "/ghost_mgg/m4_joint_hypotheses",
        "joint_hypotheses.json",
        "m4_sim_grasp_targets.json",
        "TRANSIENT_LOCAL",
        "shape_type",
        "grasp_candidates",
        "validation_state",
        "executable",
    ]:
        assert required in source


def test_hypothesis_marker_node_can_limit_visible_ranked_hypotheses():
    source = (SIM_DIR / "src" / "hypothesis_markers_node.cpp").read_text(
        encoding="utf-8"
    )

    assert "max_visible_hypotheses" in source
    assert "visible_count" in source


def test_hypothesis_marker_node_can_clear_all_after_success():
    source = (SIM_DIR / "src" / "hypothesis_markers_node.cpp").read_text(
        encoding="utf-8"
    )

    assert "hide_all_after_success" in source
    assert "successful_execution_seen_" in source


def test_hypothesis_marker_node_labels_visible_ranked_candidates():
    source = (SIM_DIR / "src" / "hypothesis_markers_node.cpp").read_text(
        encoding="utf-8"
    )

    for required in [
        "make_rank_label_marker",
        "Marker::TEXT_VIEW_FACING",
        "m2_hypothesis_rank_label",
        "stable_hypothesis_label",
        "\"track_\"",
        "\"\\nT=\"",
        "\"/F=\"",
        "make_rank_label_marker(hypotheses, hypothesis, rank, id++, executed)",
    ]:
        assert required in source

    assert '"R" << (rank + 1) << ":"' not in source


def test_m4_inspect_launch_starts_joint_hypothesis_contract_publisher():
    launch_text = (
        BRINGUP_DIR / "launch" / "m4_offline_ranking_inspect.launch.py"
    ).read_text(encoding="utf-8")
    rviz_text = (SIM_DIR / "rviz" / "m3_failure_camera_inspect.rviz").read_text(
        encoding="utf-8"
    )

    for required in [
        "joint_hypothesis_topic",
        "m4_joint_hypothesis_publisher_node.py",
        "'hypothesis_topic': joint_hypothesis_topic",
        "'report_path': joint_hypothesis_report_path",
        "'targets_path': sim_grasp_targets_path",
        "DeclareLaunchArgument('joint_hypothesis_topic'",
        "default_value='/ghost_mgg/m4_joint_hypotheses'",
    ]:
        assert required in launch_text

    for required in [
        "m4_joint_contract_markers_node",
        "executable='hypothesis_markers_node'",
        "'hypotheses_topic': joint_hypothesis_topic",
        "'marker_topic': '/ghost_mgg/m4_joint_contract_markers'",
        "'executed_hypotheses_topic': executed_hypotheses_topic",
        "'hide_executed_hypotheses': True",
    ]:
        assert required in launch_text

    assert "M4 Joint Contract Markers" in rviz_text
    assert "/ghost_mgg/m4_joint_contract_markers" in rviz_text


def test_m4_marker_smoke_checks_joint_hypothesis_contract_topic():
    smoke_text = (
        REPO_ROOT / "scripts" / "smoke_m4_offline_ranking_markers.sh"
    ).read_text(encoding="utf-8")

    for required in [
        "/ghost_mgg/m4_joint_hypotheses",
        "ghost_mgg_interfaces/msg/GeometryHypothesisArray",
        "joint_hypothesis_topic:=",
        "backend_name: ghost_mgg_m4_joint",
        "hypothesis_id: blue_cylinder",
        "grasp_candidates:",
        "validation_state: 1",
        "/ghost_mgg/m4_joint_contract_markers visualization_msgs/msg/MarkerArray",
        "m2_hypothesis_proxy",
        "m2_hypothesis_grasp",
        "m2_hypothesis_approach",
    ]:
        assert required in smoke_text
