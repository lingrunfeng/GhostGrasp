from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_refresh_grasp_trial_wrapper_records_refresh_grasp_and_outcome_contract():
    script = REPO_ROOT / "scripts" / "run_m7_refresh_grasp_trial.sh"

    assert script.exists()
    source = script.read_text(encoding="utf-8")
    for required in [
        "scripts/refresh_m7_real_target.sh",
        "scripts/grasp_m7_real_once.sh",
        "M7_TRIAL_DIR",
        "trial_metadata.json",
        "target_snapshot.json",
        "service_call.log",
        "outcome_status",
        "manual_outcome_required",
    ]:
        assert required in source


def test_refresh_grasp_trial_summary_contract():
    script = REPO_ROOT / "scripts" / "summarize_m7_refresh_grasp_trials.py"

    assert script.exists()
    source = script.read_text(encoding="utf-8")
    for required in [
        "success_rate",
        "manual_success",
        "manual_failure",
        "trial_summary.csv",
        "trial_summary.json",
    ]:
        assert required in source


def test_red_box_logged_trial_once_contract():
    script = REPO_ROOT / "scripts" / "run_m7_red_box_logged_trial_once.sh"

    assert script.exists()
    source = script.read_text(encoding="utf-8")
    for required in [
        "scripts/refresh_m7_real_target.sh red_box",
        "scripts/grasp_m7_real_once.sh",
        "scripts/home_m7_real.sh",
        "DX_MM",
        "DY_MM",
        "M7_DESCEND_EXTRA_MM",
        "M7_TARGET_Y_OFFSET_M",
        "TRIAL SUMMARY",
        "operator_outcome",
        "target_snapshot.json",
        "trial_request.json",
        "service_call.log",
        "home.log",
    ]:
        assert required in source
