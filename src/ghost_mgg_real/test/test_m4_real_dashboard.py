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
    / "generate_m4_real_dashboard.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m4_real_dashboard", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_dashboard_rows_joins_evidence_ranking_and_weak_gt(tmp_path):
    module = _load_module()
    evidence_path = tmp_path / "evidence.csv"
    top1_path = tmp_path / "top1.csv"
    weak_gt_path = tmp_path / "weak_gt.csv"

    _write_csv(
        evidence_path,
        [
            {
                "scene_id": "scene_a",
                "target_label": "transparent_jelly_cup",
                "shape_hint": "cup_like",
                "target_pixels": 100,
                "valid_depth_ratio": 0.4,
                "hole_ratio": 0.6,
                "table_leakage_ratio": 0.2,
                "foreground_ratio": 0.03,
            }
        ],
    )
    _write_csv(
        top1_path,
        [
            {
                "scene_id": "scene_a",
                "silhouette_top": "cylinder_s1.00",
                "failure_top": "box_s1.00",
                "silhouette_shape": "cylinder",
                "failure_shape": "box",
                "top1_changed": "True",
                "shape_changed": "True",
                "failure_score_delta": "0.73",
                "visual_score_delta": "-0.05",
            }
        ],
    )
    _write_csv(
        weak_gt_path,
        [
            {
                "scene_id": "scene_a",
                "material_class": "transparent",
                "object_family": "jelly_cup",
                "acceptable_shapes": "box;cylinder",
                "acceptable_shape_ok": "True",
                "failure_gain_ok": "True",
                "visual_drop_ok": "True",
                "shape_stability_ok": "True",
                "weak_gt_pass": "True",
            }
        ],
    )

    rows, summary = module.build_dashboard(
        evidence_summary_csv=evidence_path,
        top1_comparison_csv=top1_path,
        weak_gt_eval_csv=weak_gt_path,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.scene_id == "scene_a"
    assert row.material_class == "transparent"
    assert row.failure_top == "box_s1.00"
    assert row.hole_ratio == 0.6
    assert row.top1_changed is True
    assert row.weak_gt_pass is True
    assert summary["num_scenes"] == 1
    assert summary["weak_gt_pass_count"] == 1
    assert summary["top1_changed_count"] == 1
    assert summary["mean_failure_score_delta"] == 0.73


def test_write_dashboard_outputs_json_csv_and_markdown(tmp_path):
    module = _load_module()
    row = module.DashboardRow(
        scene_id="scene_a",
        target_label="transparent_jelly_cup",
        shape_hint="cup_like",
        material_class="transparent",
        object_family="jelly_cup",
        target_pixels=100,
        valid_depth_ratio=0.4,
        hole_ratio=0.6,
        table_leakage_ratio=0.2,
        foreground_ratio=0.03,
        silhouette_top="cylinder_s1.00",
        failure_top="box_s1.00",
        silhouette_shape="cylinder",
        failure_shape="box",
        top1_changed=True,
        shape_changed=True,
        failure_score_delta=0.73,
        visual_score_delta=-0.05,
        acceptable_shapes="box;cylinder",
        acceptable_shape_ok=True,
        failure_gain_ok=True,
        visual_drop_ok=True,
        shape_stability_ok=True,
        weak_gt_pass=True,
    )
    summary = {
        "schema_version": "m4_real_dashboard_v1",
        "num_scenes": 1,
        "weak_gt_pass_count": 1,
        "top1_changed_count": 1,
        "shape_changed_count": 1,
        "mean_hole_ratio": 0.6,
        "mean_table_leakage_ratio": 0.2,
        "mean_failure_score_delta": 0.73,
        "mean_visual_score_delta": -0.05,
    }

    module.write_dashboard(rows=[row], summary=summary, output_dir=tmp_path)

    assert (tmp_path / "dashboard.json").exists()
    assert (tmp_path / "dashboard.csv").exists()
    assert (tmp_path / "index.md").exists()
    payload = json.loads((tmp_path / "dashboard.json").read_text())
    assert payload["schema_version"] == "m4_real_dashboard_v1"
    assert payload["rows"][0]["scene_id"] == "scene_a"
    index = (tmp_path / "index.md").read_text()
    assert "M4 Real Dashboard" in index
    assert "scene_a" in index
