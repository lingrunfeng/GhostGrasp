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
    / "generate_m4_real_external_mask_visual_dashboard.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_m4_real_external_mask_visual_dashboard", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_hypotheses_payload():
    return {
        "schema_version": "m4_real_external_mask_replay_v1",
        "contract": "external_mask_replay_to_live_hypotheses",
        "num_scenes": 1,
        "num_hypotheses": 2,
        "rows": [
            {
                "scene_id": "scene_a",
                "target_label": "transparent_jelly_cup",
                "shape_hint": "cup_like",
                "rank": 1,
                "hypothesis_id": "box_s1.00",
                "shape_type": "box",
                "total_score": 3.1,
                "failure_score": 0.7,
                "visual_score": 0.9,
                "hole_ratio": 0.6,
                "table_leakage_ratio": 0.2,
                "provenance": "m4_real_external_mask_replay:no_truth:external_mask_contract",
            },
            {
                "scene_id": "scene_a",
                "target_label": "transparent_jelly_cup",
                "shape_hint": "cup_like",
                "rank": 2,
                "hypothesis_id": "cylinder_s1.10",
                "shape_type": "cylinder",
                "total_score": 2.4,
                "failure_score": 0.5,
                "visual_score": 0.8,
                "hole_ratio": 0.6,
                "table_leakage_ratio": 0.2,
                "provenance": "m4_real_external_mask_replay:no_truth:external_mask_contract",
            },
        ],
    }


def test_build_scene_cards_groups_hypotheses_and_image_paths(tmp_path):
    module = _load_module()
    payload_path = tmp_path / "m4_real_live_hypotheses.json"
    payload_path.write_text(json.dumps(_sample_hypotheses_payload()), encoding="utf-8")

    cards = module.build_scene_cards(
        hypotheses_json=payload_path,
        replay_samples_dir=Path("reports/m5_real_d435_replay_samples"),
        masked_evidence_dir=Path("reports/m5_real_d435_masked_evidence"),
        output_dir=tmp_path,
    )

    assert len(cards) == 1
    card = cards[0]
    assert card["scene_id"] == "scene_a"
    assert card["top_hypothesis_id"] == "box_s1.00"
    assert card["top_shape_type"] == "box"
    assert card["images"]["rgb"].endswith("scene_a/color.png")
    assert card["images"]["aligned_depth"].endswith("scene_a/aligned_depth_viz.png")
    assert card["images"]["formal_mask"].endswith("scene_a/formal_mask.png")
    assert card["images"]["evidence_overlay"].endswith("scene_a/evidence_overlay.png")
    assert len(card["hypotheses"]) == 2
    assert card["quality"]["status"] == "good"
    assert card["quality"]["top_margin"] == 0.7
    assert card["quality"]["evidence_strength"] == 0.8


def test_assign_scene_verdict_labels_good_questionable_fail_and_ood():
    module = _load_module()
    rows = _sample_hypotheses_payload()["rows"]
    good = {
        "scene_id": "good_scene",
        "target_label": "transparent_jelly_cup",
        "shape_hint": "cup_like",
        "top_total_score": 3.1,
        "hole_ratio": 0.6,
        "table_leakage_ratio": 0.2,
        "hypotheses": rows,
    }
    assert module.assign_scene_verdict(good)["status"] == "good"

    ambiguous = dict(good)
    ambiguous["scene_id"] = "ambiguous_scene"
    ambiguous["target_label"] = "multi_objects"
    assert module.assign_scene_verdict(ambiguous)["status"] == "questionable"

    low_score = dict(good)
    low_score["scene_id"] = "low_score_scene"
    low_score["top_total_score"] = 0.6
    low_score["hypotheses"] = [{**rows[0], "total_score": 0.6}, {**rows[1], "total_score": 0.5}]
    assert module.assign_scene_verdict(low_score)["status"] == "fail"

    spoon = dict(good)
    spoon["scene_id"] = "spoon_scene"
    spoon["target_label"] = "metal_spoon"
    spoon["shape_hint"] = "unknown"
    assert module.assign_scene_verdict(spoon)["status"] == "ood"


def test_write_visual_dashboard_outputs_human_readable_html(tmp_path):
    module = _load_module()
    cards = [
        {
            "scene_id": "scene_a",
            "target_label": "transparent_jelly_cup",
            "shape_hint": "cup_like",
            "top_hypothesis_id": "box_s1.00",
            "top_shape_type": "box",
            "top_total_score": 3.1,
            "hole_ratio": 0.6,
            "table_leakage_ratio": 0.2,
            "images": {
                "rgb": "../m5_real_d435_replay_samples/scene_a/color.png",
                "aligned_depth": "../m5_real_d435_replay_samples/scene_a/aligned_depth_viz.png",
                "formal_mask": "../m5_real_d435_masked_evidence/scene_a/formal_mask.png",
                "evidence_overlay": "../m5_real_d435_masked_evidence/scene_a/evidence_overlay.png",
            },
            "hypotheses": _sample_hypotheses_payload()["rows"],
        }
    ]

    module.write_visual_dashboard(
        cards=cards,
        output_dir=tmp_path,
        source_hypotheses_json=Path("reports/m4_real_external_mask_replay/m4_real_live_hypotheses.json"),
    )

    html_path = tmp_path / "index.html"
    manifest_path = tmp_path / "dashboard.json"
    assert html_path.exists()
    assert manifest_path.exists()
    assert (tmp_path / "gate_summary.json").exists()
    assert (tmp_path / "gate_summary.md").exists()
    html = html_path.read_text(encoding="utf-8")
    assert "M4 Real External-Mask Visual Dashboard" in html
    assert "Quality Verdict" in html
    assert "data-quality=\"good\"" in html
    assert "filterByQuality" in html
    assert "external mask contract" in html
    assert "scene_a" in html
    assert "RGB" in html
    assert "Aligned Depth" in html
    assert "Formal Mask" in html
    assert "Evidence Overlay" in html
    assert "Top-3 Hypotheses" in html
    assert "box_s1.00" in html
    assert "<img" in html
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "m4_real_external_mask_visual_dashboard_v1"
    assert manifest["num_scenes"] == 1
    assert manifest["cards"][0]["quality"]["status"] == "good"

    gate_summary = json.loads((tmp_path / "gate_summary.json").read_text(encoding="utf-8"))
    assert gate_summary["schema_version"] == "m4_real_external_mask_gate_summary_v1"
    assert gate_summary["status_counts"]["good"] == 1
    assert gate_summary["num_scenes"] == 1


def test_visual_dashboard_run_script_contract_is_no_truth():
    script_path = REPO_ROOT / "scripts" / "run_m4_real_external_mask_visual_dashboard.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "run_m4_real_external_mask_replay.sh",
        "generate_m4_real_external_mask_visual_dashboard.py",
        "--hypotheses-json reports/m4_real_external_mask_replay/m4_real_live_hypotheses.json",
        "--output-dir reports/m4_real_external_mask_visual_dashboard",
        "index.html",
        "gate_summary.json",
        "gate_summary.md",
        "M4 real external-mask visual dashboard passed",
    ]:
        assert required in source

    for forbidden in [
        "gz model",
        "gz service",
        "current_targets.json",
        "m4_sim_grasp_targets.json",
    ]:
        assert forbidden not in source
