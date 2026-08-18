from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m4_moveit_dryrun_launch_publishes_joint_hypotheses():
    launch_text = (
        BRINGUP_DIR / "launch" / "m4_sim_grasp_moveit_dryrun.launch.py"
    ).read_text(encoding="utf-8")

    for required in [
        "LaunchConfiguration('headless')",
        "'headless': headless",
        "DeclareLaunchArgument('headless', default_value='true')",
        "joint_hypothesis_report_path",
        "joint_hypothesis_topic",
        "m4_joint_hypothesis_publisher_node.py",
        "'report_path': joint_hypothesis_report_path",
        "'targets_path': sim_grasp_targets_path",
        "'hypothesis_topic': joint_hypothesis_topic",
        "default_value='reports/m4_joint_hypotheses/joint_hypotheses.json'",
        "default_value='/ghost_mgg/m4_joint_hypotheses'",
    ]:
        assert required in launch_text


def test_m4_moveit_probe_can_plan_from_hypothesis_topic():
    probe_text = (REPO_ROOT / "scripts" / "probe_m4_sim_grasp_moveit.py").read_text(
        encoding="utf-8"
    )

    for required in [
        "GeometryHypothesisArray",
        "GeometryHypothesis",
        "GraspCandidate",
        "--input-mode",
        "--hypotheses-topic",
        "wait_for_hypotheses",
        "hypothesis_to_target",
        "hypotheses_topic",
        "m4_joint_hypotheses",
        "source_input",
    ]:
        assert required in probe_text


def test_m4_joint_hypothesis_moveit_smoke_uses_contract_topic():
    smoke_path = REPO_ROOT / "scripts" / "smoke_m4_joint_hypothesis_moveit_dryrun.sh"
    assert smoke_path.exists()
    smoke_text = smoke_path.read_text(encoding="utf-8")

    for required in [
        "m4_sim_grasp_moveit_dryrun.launch.py",
        "run_m4_joint_hypothesis_report.sh",
        "probe_m4_sim_grasp_moveit.py",
        "--input-mode hypotheses",
        "--hypotheses-topic /ghost_mgg/m4_joint_hypotheses",
        "/ghost_mgg/m4_joint_hypotheses ghost_mgg_interfaces/msg/GeometryHypothesisArray",
        "summary planned=4/4",
        "M4 joint hypothesis MoveIt dry-run smoke passed",
    ]:
        assert required in smoke_text
