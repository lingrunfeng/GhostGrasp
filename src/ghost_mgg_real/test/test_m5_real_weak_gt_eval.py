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
    / "generate_m5_real_weak_gt_eval.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m5_real_weak_gt_eval", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_weak_gt_rows_checks_shape_failure_gain_and_visual_drop(tmp_path):
    module = _load_module()
    weak_gt_path = tmp_path / "weak_gt.json"
    top1_path = tmp_path / "top1_comparison.csv"
    evidence_path = tmp_path / "summary.csv"

    weak_gt_path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    _write_csv(
        top1_path,
        [
            {
                "scene_id": "transparent_scene",
                "target_label": "transparent_jelly_cup",
                "shape_hint": "cup_like",
                "silhouette_top": "cylinder_s1.00",
                "failure_top": "box_s1.00",
                "silhouette_shape": "cylinder",
                "failure_shape": "box",
                "top1_changed": "True",
                "shape_changed": "True",
                "visual_score_delta": "-0.05",
                "failure_score_delta": "0.73",
            },
            {
                "scene_id": "opaque_scene",
                "target_label": "opaque_box",
                "shape_hint": "box",
                "silhouette_top": "box_s0.95",
                "failure_top": "box_s1.00",
                "silhouette_shape": "box",
                "failure_shape": "box",
                "top1_changed": "True",
                "shape_changed": "False",
                "visual_score_delta": "-0.02",
                "failure_score_delta": "0.48",
            },
        ],
    )
    _write_csv(
        evidence_path,
        [
            {
                "scene_id": "transparent_scene",
                "hole_ratio": "0.59",
                "table_leakage_ratio": "0.19",
                "foreground_ratio": "0.02",
            },
            {
                "scene_id": "opaque_scene",
                "hole_ratio": "0.16",
                "table_leakage_ratio": "0.04",
                "foreground_ratio": "0.10",
            },
        ],
    )

    rows, summary = module.evaluate_weak_gt(
        weak_gt_path=weak_gt_path,
        top1_comparison_csv=top1_path,
        evidence_summary_csv=evidence_path,
    )

    assert summary["num_scenes"] == 2
    assert summary["weak_gt_pass_count"] == 2
    assert summary["failure_gain_checked_count"] == 1
    assert rows[0].weak_gt_pass is True
    assert rows[0].failure_gain_ok is True
    assert rows[0].visual_drop_ok is True
    assert rows[1].shape_stability_ok is True


def test_write_weak_gt_report_creates_json_csv_and_index(tmp_path):
    module = _load_module()
    row = module.WeakGTEvalRow(
        scene_id="scene_a",
        target_label="cup",
        material_class="transparent",
        object_family="jelly_cup",
        silhouette_top="cylinder_s1.00",
        failure_top="box_s1.00",
        silhouette_shape="cylinder",
        failure_shape="box",
        acceptable_shapes="box;cylinder",
        expected_failure_sensitive=True,
        require_shape_stability=False,
        top1_changed=True,
        shape_changed=True,
        failure_score_delta=0.73,
        visual_score_delta=-0.05,
        hole_ratio=0.59,
        table_leakage_ratio=0.19,
        foreground_ratio=0.02,
        acceptable_shape_ok=True,
        failure_gain_ok=True,
        visual_drop_ok=True,
        shape_stability_ok=True,
        weak_gt_pass=True,
    )

    module.write_weak_gt_report(
        rows=[row],
        summary={"schema_version": "m5_real_weak_gt_eval_v1", "num_scenes": 1},
        output_dir=tmp_path,
    )

    assert (tmp_path / "weak_gt_eval.json").exists()
    assert (tmp_path / "weak_gt_eval.csv").exists()
    assert (tmp_path / "index.md").exists()
    payload = json.loads((tmp_path / "weak_gt_eval.json").read_text())
    assert payload["schema_version"] == "m5_real_weak_gt_eval_v1"
    assert payload["rows"][0]["weak_gt_pass"] is True
    index = (tmp_path / "index.md").read_text()
    assert "M5 Real Weak-GT Evaluation" in index
