import csv
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "generate_m4_gate_summary.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m4_gate_summary", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _input_paths(tmp_path: Path, *, weak_gt_pass_count: int = 2) -> dict[str, Path]:
    calibration = _write_json(
        tmp_path / "calibration" / "best_weights.json",
        {
            "schema_version": "m4_real_weight_calibration_best_v1",
            "best_weights": {"visual": 0.75, "failure": 1.0, "depth": 0.0},
            "best_summary": {
                "num_scenes": 2,
                "weak_gt_pass_count": weak_gt_pass_count,
                "failure_gain_checked_count": 1,
                "failure_gain_pass_count": 1,
            },
            "summary": {"num_scenes": 2, "num_weight_sets": 12},
        },
    )
    dashboard = _write_json(
        tmp_path / "dashboard" / "dashboard.json",
        {
            "schema_version": "m4_real_dashboard_v1",
            "num_scenes": 2,
            "weak_gt_pass_count": weak_gt_pass_count,
            "top1_changed_count": 1,
            "shape_changed_count": 1,
            "mean_hole_ratio": 0.4,
            "mean_table_leakage_ratio": 0.2,
            "mean_failure_score_delta": 0.7,
            "rows": [
                {"scene_id": "scene_a", "weak_gt_pass": True},
                {"scene_id": "scene_b", "weak_gt_pass": weak_gt_pass_count == 2},
            ],
        },
    )
    weak_gt = _write_json(
        tmp_path / "weak_gt" / "weak_gt_eval.json",
        {
            "schema_version": "m5_real_weak_gt_eval_v1",
            "num_scenes": 2,
            "weak_gt_pass_count": weak_gt_pass_count,
            "failure_gain_checked_count": 1,
            "failure_gain_pass_count": 1,
            "visual_drop_pass_count": 2,
            "shape_stability_checked_count": 1,
            "shape_stability_pass_count": 1,
            "rows": [
                {"scene_id": "scene_a", "weak_gt_pass": True},
                {"scene_id": "scene_b", "weak_gt_pass": weak_gt_pass_count == 2},
            ],
        },
    )
    ranking = _write_json(
        tmp_path / "ranking" / "m5_real_ranking.json",
        {
            "schema_version": "m5_real_ranking_v1",
            "num_scenes": 2,
            "num_rows": 12,
            "failure_aware_weights": {
                "visual": 0.75,
                "failure": 1.0,
                "depth": 0.0,
                "physical": 0.0,
                "grasp": 0.0,
                "prior": 0.0,
            },
            "rows": [],
        },
    )
    visual_board = _write_json(
        tmp_path / "visual_board" / "manifest.json",
        {
            "schema_version": "m4_visual_ranking_board_manifest_v1",
            "num_scenes": 2,
            "scenes": [
                {"scene_id": "scene_a", "board_path": "scene_a.png", "weak_gt_pass": True},
                {"scene_id": "scene_b", "board_path": "scene_b.png", "weak_gt_pass": True},
            ],
        },
    )
    return {
        "calibration_json": calibration,
        "dashboard_json": dashboard,
        "weak_gt_json": weak_gt,
        "ranking_json": ranking,
        "visual_board_manifest": visual_board,
    }


def test_gate_summary_passes_when_current_reports_are_consistent(tmp_path):
    module = _load_module()
    inputs = _input_paths(tmp_path)

    summary = module.build_gate_summary(**inputs)

    assert summary.overall_status == "pass"
    assert summary.num_scenes == 2
    assert summary.best_weights["failure"] == 1.0
    assert {check.gate_id: check.status for check in summary.checks} == {
        "calibration_weights": "pass",
        "ranking_coverage": "pass",
        "weak_gt": "pass",
        "failure_gain": "pass",
        "visual_boards": "pass",
    }


def test_gate_summary_fails_when_weak_gt_scene_fails(tmp_path):
    module = _load_module()
    inputs = _input_paths(tmp_path, weak_gt_pass_count=1)

    summary = module.build_gate_summary(**inputs)

    assert summary.overall_status == "fail"
    weak_gt_check = next(check for check in summary.checks if check.gate_id == "weak_gt")
    assert weak_gt_check.status == "fail"
    assert "1/2" in weak_gt_check.summary


def test_write_gate_summary_outputs_json_csv_and_markdown(tmp_path):
    module = _load_module()
    summary = module.build_gate_summary(**_input_paths(tmp_path))
    output_dir = tmp_path / "report"

    module.write_gate_summary(summary, output_dir)

    payload = json.loads((output_dir / "gate_summary.json").read_text())
    assert payload["schema_version"] == "m4_gate_summary_v1"
    assert payload["overall_status"] == "pass"
    assert payload["best_weights"]["failure"] == 1.0

    with (output_dir / "gate_checks.csv").open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["gate_id"] == "calibration_weights"
    assert rows[0]["status"] == "pass"

    index = (output_dir / "index.md").read_text()
    assert "M4 Gate Summary" in index
    assert "overall_status: pass" in index
    assert "calibration_weights" in index
