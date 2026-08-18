from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m6_shadow_moveit_plan_only_script_is_shadow_only():
    script_path = REPO_ROOT / "scripts" / "run_m6_shadow_moveit_plan_only.sh"
    assert script_path.exists()

    script_text = script_path.read_text(encoding="utf-8")

    for required in [
        "generate_m6_shadow_grasp_target.py",
        "probe_m4_sim_grasp_moveit.py",
        "--planning-frame base_link",
        "moveit_plan_only_shadow_allowlist.json",
        "--allowed-collision-pair base_link:m4_table_collision",
        "--allowed-collision-pair gripper_base:link4",
        "shadow-only",
    ]:
        assert required in script_text

    for forbidden in [
        "ExecuteGrasp",
        "send_angles",
        "FollowJointTrajectory",
        "moveit_sim_execute_server",
    ]:
        assert forbidden not in script_text


def test_m6_shadow_decision_docs_reference_moveit_plan_only():
    docs = (REPO_ROOT / "docs" / "m6_shadow_decision.md").read_text(encoding="utf-8")

    for required in [
        "scripts/run_m6_shadow_moveit_plan_only.sh",
        "moveit_plan_only_shadow_allowlist.json",
        "不会发送真实运动命令",
    ]:
        assert required in docs
