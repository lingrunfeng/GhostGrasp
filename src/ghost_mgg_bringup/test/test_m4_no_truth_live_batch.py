from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m4_no_truth_live_batch_runner_generates_report_contract():
    script_path = REPO_ROOT / "scripts" / "run_m4_no_truth_live_batch_eval.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "m4_no_truth_live_moveit_execute.launch.py",
        "/ghost_mgg/m4_live_hypotheses",
        "m4_live_no_truth_ghost_mgg_v1",
        "probe_m4_sim_grasp_moveit.py",
        "--omit-target-obstacles",
        "execute_m4_joint_hypothesis_action.py",
        "set_model_pose",
        "move_non_target_clutter_out_of_view",
        "reports/m4_no_truth_live_batch",
        "batch.json",
        "batch.csv",
        "index.md",
        "dryrun_planned",
        "execute_status_name",
        "no_truth_audit",
        "awk '/^\\{.*\\}$/ {line=$0} END {print line}'",
        "[[ -z \"${output}\" ]]",
        "last_rejected_hypothesis=",
        "deadline=$((SECONDS + 90))",
        "M4 no-truth live batch eval failed: no accepted hypothesis",
        "M4 no-truth live batch eval passed",
    ]:
        assert required in source


def test_m4_no_truth_live_batch_runner_has_dynamic_cases_and_execute_subset():
    script_path = REPO_ROOT / "scripts" / "run_m4_no_truth_live_batch_eval.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "red_center,0.055,0.012,0.20,dryrun",
        "red_forward,0.055,0.055,0.20,dryrun",
        "red_side,0.035,0.035,0.55,dryrun",
        "red_center_execute,0.055,0.012,0.20,execute",
        "dynamic_displacement_m",
        "execute_success_rate",
        "dryrun_success_rate",
    ]:
        assert required in source
