import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "src" / "ghost_mgg_real" / "scripts" / "generate_m4_failure_atlas.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m4_failure_atlas", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sample_dashboard(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "m4_real_external_mask_visual_dashboard_v1",
            "cards": [
                _card(
                    "hole_scene",
                    "transparent_jelly_cup",
                    "good",
                    hole=0.62,
                    leakage=0.04,
                    foreground=0.00,
                ),
                _card(
                    "leak_scene",
                    "transparent_jelly_cup",
                    "good",
                    hole=0.31,
                    leakage=0.22,
                    foreground=0.00,
                ),
                _card(
                    "opaque_scene",
                    "opaque_box",
                    "good",
                    hole=0.03,
                    leakage=0.01,
                    foreground=0.24,
                ),
                _card(
                    "spoon_scene",
                    "metal_spoon",
                    "ood",
                    hole=0.14,
                    leakage=0.03,
                    foreground=0.02,
                ),
            ],
        },
    )


def _sample_ablation(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "m4_real_ablation_gate_v1",
            "rows": [
                _ablation("hole_scene", "without_table_leakage", False, 0.10),
                _ablation("leak_scene", "without_table_leakage", True, 0.76),
                _ablation("opaque_scene", "without_table_leakage", False, 0.01),
                _ablation("spoon_scene", "without_table_leakage", True, 0.50, decision="excluded_ood"),
            ],
        },
    )


def test_build_failure_atlas_assigns_categories_and_representatives(tmp_path):
    module = _load_module()
    dashboard = _sample_dashboard(tmp_path / "dashboard.json")
    ablation = _sample_ablation(tmp_path / "ablation_gate.json")

    atlas = module.build_failure_atlas(dashboard_json=dashboard, ablation_gate_json=ablation)

    assert atlas["schema_version"] == "m4_failure_atlas_v1"
    assert atlas["num_scenes"] == 4
    by_scene = {row["scene_id"]: row for row in atlas["scenes"]}
    assert "hole_dominant" in by_scene["hole_scene"]["categories"]
    assert "table_leakage" in by_scene["leak_scene"]["categories"]
    assert "table_leakage_ablation_effect" in by_scene["leak_scene"]["categories"]
    assert "foreground_supported_control" in by_scene["opaque_scene"]["categories"]
    assert "ood_or_primitive_mismatch" in by_scene["spoon_scene"]["categories"]

    reps = atlas["representatives"]
    assert reps["hole_dominant"]["scene_id"] == "hole_scene"
    assert reps["table_leakage"]["scene_id"] == "leak_scene"
    assert reps["table_leakage_ablation_effect"]["scene_id"] == "leak_scene"
    assert reps["ood_or_primitive_mismatch"]["scene_id"] == "spoon_scene"


def test_write_failure_atlas_outputs_visual_html_json_and_markdown(tmp_path):
    module = _load_module()
    atlas = module.build_failure_atlas(
        dashboard_json=_sample_dashboard(tmp_path / "dashboard.json"),
        ablation_gate_json=_sample_ablation(tmp_path / "ablation_gate.json"),
    )

    module.write_failure_atlas(atlas, tmp_path / "atlas")

    payload = json.loads((tmp_path / "atlas" / "failure_atlas.json").read_text())
    assert payload["schema_version"] == "m4_failure_atlas_v1"
    html = (tmp_path / "atlas" / "index.html").read_text(encoding="utf-8")
    assert "M4 Failure Evidence Atlas" in html
    assert "hole_dominant" in html
    assert "table_leakage_ablation_effect" in html
    assert "<img" in html
    markdown = (tmp_path / "atlas" / "index.md").read_text(encoding="utf-8")
    assert "M4 Failure Evidence Atlas" in markdown
    assert "Representative Scenes" in markdown


def test_failure_atlas_run_script_contract_and_doc_exist():
    script_path = REPO_ROOT / "scripts" / "run_m4_failure_atlas.sh"
    assert script_path.exists()
    script = script_path.read_text(encoding="utf-8")
    for required in [
        "run_m4_real_ablation_gate.sh",
        "generate_m4_failure_atlas.py",
        "reports/m4_failure_atlas",
        "failure_atlas.json",
        "index.html",
        "M4 failure atlas passed",
    ]:
        assert required in script
    for forbidden in ["gz model", "gz service", "current_targets.json"]:
        assert forbidden not in script

    doc_path = REPO_ROOT / "docs" / "superpowers" / "plans" / "2026-07-04-m4-failure-atlas.md"
    assert doc_path.exists()
    doc = doc_path.read_text(encoding="utf-8")
    assert "M4 Failure Evidence Atlas" in doc
    assert "hole_dominant" in doc
    assert "run_m4_failure_atlas.sh" in doc


def _card(scene_id, label, quality_status, *, hole, leakage, foreground):
    return {
        "scene_id": scene_id,
        "target_label": label,
        "shape_hint": "box" if "opaque" in label else "cup_like",
        "top_hypothesis_id": "box_s1.00",
        "top_shape_type": "box",
        "top_total_score": 2.0,
        "hole_ratio": hole,
        "table_leakage_ratio": leakage,
        "quality": {
            "status": quality_status,
            "reasons": ["sample reason"],
            "evidence_strength": round(hole + leakage, 6),
        },
        "images": {
            "rgb": f"../m5_real_d435_replay_samples/{scene_id}/color.png",
            "aligned_depth": f"../m5_real_d435_replay_samples/{scene_id}/aligned_depth_viz.png",
            "formal_mask": f"../m5_real_d435_masked_evidence/{scene_id}/formal_mask.png",
            "evidence_overlay": f"../m5_real_d435_masked_evidence/{scene_id}/evidence_overlay.png",
        },
        "hypotheses": [
            {
                "rank": 1,
                "hypothesis_id": "box_s1.00",
                "shape_type": "box",
                "valid_depth_ratio": max(0.0, 1.0 - hole),
                "foreground_ratio": foreground,
                "failure_score": hole + leakage,
                "total_score": 2.0,
            }
        ],
    }


def _ablation(scene_id, ranker, changed, advantage, *, decision="evaluable"):
    return {
        "scene_id": scene_id,
        "ranker": ranker,
        "rank": 1,
        "gate_decision": decision,
        "top1_changed_vs_full": changed,
        "full_total_advantage": advantage,
        "full_failure_advantage": advantage / 2.0,
        "hypothesis_id": "cylinder_s1.00" if changed else "box_s1.00",
        "full_hypothesis_id": "box_s1.00",
    }
