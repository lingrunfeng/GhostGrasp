from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m4_no_truth_headless_suite_runs_all_no_truth_gates():
    script_path = REPO_ROOT / "scripts" / "run_m4_no_truth_headless_suite.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "run_m4_no_truth_live_batch_eval.sh",
        "run_m4_no_truth_multi_target_eval.sh",
        "run_m4_no_truth_scenario_sweep.sh",
        "smoke_m4_no_truth_live_ranked_fallback.sh",
        "smoke_m4_no_truth_live_dynamic_execute.sh",
        "run_m4_no_truth_gate.sh",
        "reports/m4_no_truth_headless_suite",
        "cleanup_stale_m4_processes",
        "pgrep -af",
        "suite.json",
        "index.md",
        "M4 no-truth headless suite passed",
        "ranked_fallback",
    ]:
        assert required in source


def test_m4_no_truth_headless_suite_registered_in_cmake():
    cmake_text = (REPO_ROOT / "src" / "ghost_mgg_bringup" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert "test_m4_no_truth_headless_suite" in cmake_text
