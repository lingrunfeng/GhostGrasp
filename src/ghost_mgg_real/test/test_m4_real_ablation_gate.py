import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT / "src" / "ghost_mgg_real" / "scripts" / "generate_m4_real_ablation_gate.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m4_real_ablation_gate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rank_scene_ablations_can_show_table_leakage_changes_top1():
    module = _load_module()
    target_mask = np.zeros((20, 20), dtype=bool)
    target_mask[6:14, 6:14] = True
    current_depth = np.zeros((20, 20), dtype=np.uint16) + 1000
    background_depth = np.zeros((20, 20), dtype=np.uint16)
    background_depth[8:12, 8:12] = 1000

    hypotheses = [
        module.PrimitiveHypothesis(
            hypothesis_id="silhouette_bbox",
            shape_type="box",
            center_uv=(9.5, 9.5),
            size_px=(8.0, 8.0),
            depth_m=1.0,
            height_m=0.1,
        ),
        module.PrimitiveHypothesis(
            hypothesis_id="leak_region",
            shape_type="box",
            center_uv=(9.5, 9.5),
            size_px=(4.0, 4.0),
            depth_m=1.0,
            height_m=0.1,
        ),
    ]

    rows = module.rank_scene_ablations(
        scene_id="leak_scene",
        target_label="transparent_jelly_cup",
        shape_hint="cup_like",
        target_mask=target_mask,
        current_depth=current_depth,
        background_depth=background_depth,
        hypotheses=hypotheses,
        top_k=2,
    )

    top_by_ranker = {row["ranker"]: row["hypothesis_id"] for row in rows if row["rank"] == 1}
    assert top_by_ranker["full"] == "leak_region"
    assert top_by_ranker["without_table_leakage"] == "silhouette_bbox"
    assert top_by_ranker["without_failure"] == "silhouette_bbox"
    assert top_by_ranker["silhouette_only"] == "silhouette_bbox"


def test_ablation_gate_excludes_ood_and_summarizes_ranker_losses(tmp_path):
    module = _load_module()
    rows = [
        _row("good_scene", "full", "box_s1.00", "box", 3.0, 0.8),
        _row("good_scene", "silhouette_only", "cylinder_s1.00", "cylinder", 0.9, -0.1),
        _row("good_scene", "without_failure", "cylinder_s1.00", "cylinder", 1.2, 0.8),
        _row("good_scene", "without_table_leakage", "box_s0.95", "box", 2.0, 0.2),
        _row("good_scene", "without_weak_depth", "box_s1.00", "box", 2.8, 0.8),
        _row("questionable_scene", "full", "box_s1.00", "box", 2.8, 0.7),
        _row("questionable_scene", "silhouette_only", "cylinder_s1.00", "cylinder", 0.8, -0.2),
        _row("questionable_scene", "without_failure", "cylinder_s1.00", "cylinder", 1.1, 0.7),
        _row("questionable_scene", "without_table_leakage", "cylinder_s1.00", "cylinder", 1.8, 0.1),
        _row("questionable_scene", "without_weak_depth", "box_s1.00", "box", 2.5, 0.7),
        _row("spoon_scene", "full", "box_s1.00", "box", 2.0, 0.5, target_label="metal_spoon"),
        _row("spoon_scene", "silhouette_only", "box_s0.95", "box", 0.8, -0.1, target_label="metal_spoon"),
    ]
    dashboard_json = _write_json(
        tmp_path / "dashboard.json",
        {
            "cards": [
                {"scene_id": "good_scene", "quality": {"status": "good", "reasons": []}},
                {
                    "scene_id": "questionable_scene",
                    "quality": {"status": "questionable", "reasons": ["multi-object"]},
                },
                {"scene_id": "spoon_scene", "quality": {"status": "ood", "reasons": []}},
            ]
        },
    )

    report = module.build_ablation_gate_report(rows=rows, dashboard_json=dashboard_json)

    assert report["schema_version"] == "m4_real_ablation_gate_v1"
    assert report["overall_status"] == "pass"
    assert report["num_evaluable_scenes"] == 2
    assert report["num_excluded_scenes"] == 1
    assert report["ablation_summary"]["without_failure"]["top1_changed_count"] == 2
    assert report["ablation_summary"]["without_table_leakage"]["top1_changed_count"] == 2
    assert report["ablation_summary"]["without_weak_depth"]["top1_changed_count"] == 0
    assert report["rows"][-1]["gate_decision"] == "excluded_ood"


def test_write_ablation_gate_outputs_json_csv_markdown_and_docs(tmp_path):
    module = _load_module()
    report = {
        "schema_version": "m4_real_ablation_gate_v1",
        "overall_status": "pass",
        "num_scenes": 1,
        "num_evaluable_scenes": 1,
        "num_excluded_scenes": 0,
        "ablation_summary": {
            "without_failure": {
                "top1_changed_count": 1,
                "shape_changed_count": 1,
                "mean_full_total_advantage": 1.0,
            }
        },
        "gate_reasons": ["full beats without_failure on evaluable scenes"],
        "rows": [_row("good_scene", "full", "box_s1.00", "box", 3.0, 0.8)],
    }

    module.write_ablation_gate_report(report, tmp_path / "report")

    payload = json.loads((tmp_path / "report" / "ablation_gate.json").read_text())
    assert payload["schema_version"] == "m4_real_ablation_gate_v1"
    with (tmp_path / "report" / "ablation_gate.csv").open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["scene_id"] == "good_scene"
    assert "M4 Real Ablation Gate" in (tmp_path / "report" / "index.md").read_text()


def test_real_ablation_gate_run_script_contract_and_progress_doc_exist():
    script_path = REPO_ROOT / "scripts" / "run_m4_real_ablation_gate.sh"
    assert script_path.exists()
    script = script_path.read_text(encoding="utf-8")
    for required in [
        "generate_m4_real_ablation_gate.py",
        "run_m4_real_external_mask_visual_dashboard.sh",
        "reports/m4_real_ablation_gate",
        "ablation_gate.json",
        "M4 real ablation gate passed",
    ]:
        assert required in script
    for forbidden in ["gz model", "gz service", "current_targets.json"]:
        assert forbidden not in script

    doc_path = REPO_ROOT / "docs" / "superpowers" / "plans" / "2026-07-04-m4-real-ablation-gate.md"
    assert doc_path.exists()
    doc = doc_path.read_text(encoding="utf-8")
    assert "M4 Real Ablation Gate" in doc
    assert "without_table_leakage" in doc
    assert "run_m4_real_ablation_gate.sh" in doc


def _row(
    scene_id,
    ranker,
    hypothesis_id,
    shape_type,
    total_score,
    failure_score,
    *,
    target_label="transparent_jelly_cup",
):
    return {
        "scene_id": scene_id,
        "target_label": target_label,
        "shape_hint": "cup_like",
        "ranker": ranker,
        "rank": 1,
        "hypothesis_id": hypothesis_id,
        "shape_type": shape_type,
        "center_u": 10.0,
        "center_v": 11.0,
        "size_u_px": 12.0,
        "size_v_px": 13.0,
        "visual_score": 0.8,
        "failure_score": failure_score,
        "depth_score": 0.0,
        "total_score": total_score,
        "validation_state": "accepted",
    }
