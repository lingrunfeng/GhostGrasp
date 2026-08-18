from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m4_no_truth_external_mask_dynamic_rerun_eval_script_contract():
    script_path = REPO_ROOT / "scripts" / "run_m4_no_truth_external_mask_dynamic_rerun_eval.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "m4_no_truth_live_moveit_execute.launch.py",
        "target_color_hint:=none",
        "require_external_target_mask:=true",
        "publish_m4_color_target_mask.py",
        "/ghost_mgg/d435/external_target_mask",
        "grasp_m4_live_once.sh",
        "/ghost_mgg/m4_live_hypotheses",
        "set_model_pose",
        "move_non_target_clutter_out_of_view",
        "external_mask_dynamic_rerun_summary.csv",
        "external_mask_dynamic_rerun_summary.json",
        "trial_index,target_x,target_y,selected_x,selected_y,position_error_m,status_name,hypothesis_id,json_path",
        "run_live_grasp_once_with_retry",
        "GHOST_MGG_M4_EXTERNAL_MASK_DYNAMIC_RERUN_GRASP_RETRIES",
        "M4 no-truth external-mask dynamic rerun eval passed",
    ]:
        assert required in source

    for forbidden in [
        "gz model -m",
        "current_targets.json",
        "m4_sim_grasp_targets.json",
    ]:
        assert forbidden not in source
