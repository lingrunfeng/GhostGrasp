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
    / "generate_m4_real_weight_calibration.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_m4_real_weight_calibration", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _ranking_rows() -> list[dict[str, object]]:
    return [
        {
            "scene_id": "transparent_scene",
            "target_label": "transparent_jelly_cup",
            "shape_hint": "cup_like",
            "ranker": "silhouette_only",
            "rank": 1,
            "hypothesis_id": "visual_cylinder",
            "shape_type": "cylinder",
            "center_u": 10.0,
            "center_v": 10.0,
            "size_u_px": 12.0,
            "size_v_px": 12.0,
            "visual_score": 0.95,
            "failure_score": -0.05,
            "depth_score": 0.0,
            "total_score": 0.95,
            "validation_state": "accepted",
        },
        {
            "scene_id": "transparent_scene",
            "target_label": "transparent_jelly_cup",
            "shape_hint": "cup_like",
            "ranker": "failure_aware",
            "rank": 1,
            "hypothesis_id": "failure_box",
            "shape_type": "box",
            "center_u": 10.0,
            "center_v": 10.0,
            "size_u_px": 10.0,
            "size_v_px": 10.0,
            "visual_score": 0.88,
            "failure_score": 0.55,
            "depth_score": 0.0,
            "total_score": 3.08,
            "validation_state": "accepted",
        },
        {
            "scene_id": "opaque_scene",
            "target_label": "opaque_box",
            "shape_hint": "box",
            "ranker": "silhouette_only",
            "rank": 1,
            "hypothesis_id": "opaque_box_visual",
            "shape_type": "box",
            "center_u": 20.0,
            "center_v": 20.0,
            "size_u_px": 8.0,
            "size_v_px": 8.0,
            "visual_score": 0.90,
            "failure_score": 0.02,
            "depth_score": 0.1,
            "total_score": 0.90,
            "validation_state": "accepted",
        },
        {
            "scene_id": "opaque_scene",
            "target_label": "opaque_box",
            "shape_hint": "box",
            "ranker": "failure_aware",
            "rank": 1,
            "hypothesis_id": "opaque_box_failure",
            "shape_type": "box",
            "center_u": 20.0,
            "center_v": 20.0,
            "size_u_px": 9.0,
            "size_v_px": 9.0,
            "visual_score": 0.88,
            "failure_score": 0.10,
            "depth_score": 0.1,
            "total_score": 1.38,
            "validation_state": "accepted",
        },
    ]


def _weak_gt_payload() -> dict[str, object]:
    return {
        "schema_version": "m5_real_weak_gt_v1",
        "scenes": [
            {
                "scene_id": "transparent_scene",
                "target_label": "transparent_jelly_cup",
                "material_class": "transparent",
                "object_family": "jelly_cup",
                "acceptable_shapes": ["box", "cylinder"],
                "expected_failure_sensitive": True,
                "require_shape_stability": False,
                "min_failure_score_delta": 0.2,
                "max_visual_score_drop": 0.12,
            },
            {
                "scene_id": "opaque_scene",
                "target_label": "opaque_box",
                "material_class": "opaque",
                "object_family": "box",
                "acceptable_shapes": ["box"],
                "expected_failure_sensitive": False,
                "require_shape_stability": True,
                "min_failure_score_delta": 0.0,
                "max_visual_score_drop": 0.05,
            },
        ],
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    ranking_csv = tmp_path / "ranking.csv"
    weak_gt_json = tmp_path / "weak_gt.json"
    _write_csv(ranking_csv, _ranking_rows())
    weak_gt_json.write_text(json.dumps(_weak_gt_payload()), encoding="utf-8")
    return ranking_csv, weak_gt_json


def test_calibration_selects_failure_sensitive_weights(tmp_path):
    module = _load_module()
    ranking_csv, weak_gt_json = _write_inputs(tmp_path)

    result = module.calibrate_weights(
        ranking_csv=ranking_csv,
        weak_gt_json=weak_gt_json,
        visual_weights=(1.0,),
        failure_weights=(0.0, 4.0),
        depth_weights=(0.0,),
    )

    assert result.summary["num_scenes"] == 2
    assert result.best_summary["weak_gt_pass_count"] == 2
    assert result.best_summary["failure_gain_pass_count"] == 1
    assert result.best_weights["failure"] == 4.0
    assert result.best_top1_by_scene["transparent_scene"].hypothesis_id == "failure_box"
    assert result.best_top1_by_scene["opaque_scene"].shape_type == "box"


def test_write_calibration_report_creates_json_csv_and_index(tmp_path):
    module = _load_module()
    ranking_csv, weak_gt_json = _write_inputs(tmp_path)
    result = module.calibrate_weights(
        ranking_csv=ranking_csv,
        weak_gt_json=weak_gt_json,
        visual_weights=(1.0,),
        failure_weights=(0.0, 4.0),
        depth_weights=(0.0,),
    )

    module.write_calibration_report(result, tmp_path / "report")

    best = json.loads((tmp_path / "report" / "best_weights.json").read_text())
    assert best["schema_version"] == "m4_real_weight_calibration_best_v1"
    assert best["best_weights"]["failure"] == 4.0
    assert (tmp_path / "report" / "calibration_grid.csv").exists()
    assert (tmp_path / "report" / "calibrated_top1.csv").exists()
    assert (tmp_path / "report" / "calibration_grid.json").exists()
    assert (tmp_path / "report" / "calibrated_top1.json").exists()
    index = (tmp_path / "report" / "index.md").read_text()
    assert "M4 Real Weight Calibration" in index
    assert "transparent_scene" in index
