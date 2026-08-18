from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m4_no_truth_scenario_sweep_runner_contract():
    script_path = REPO_ROOT / "scripts" / "run_m4_no_truth_scenario_sweep.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "m4_no_truth_live_perception.launch.py",
        "/ghost_mgg/m4_live_hypotheses",
        "scenario_ids=(S0 S1 S2 S3 S4 S5 S6 S7)",
        "set_model_pose",
        "move_non_target_clutter_out_of_view",
        "reports/m4_no_truth_scenario_sweep",
        "scenario_sweep.json",
        "scenario_sweep.csv",
        "index.md",
        "hypothesis_count",
        "pose_error_m",
        "no_truth_audit",
        "awk '/^\\{.*\\}$/ {line=$0} END {print line}'",
        "[[ -z \"${output}\" ]]",
        "M4 no-truth scenario sweep failed: no accepted hypothesis",
        "M4 no-truth scenario sweep passed",
    ]:
        assert required in source


def test_m4_no_truth_scenario_sweep_registered_in_cmake():
    cmake_text = (REPO_ROOT / "src" / "ghost_mgg_bringup" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert "test_m4_no_truth_scenario_sweep" in cmake_text
