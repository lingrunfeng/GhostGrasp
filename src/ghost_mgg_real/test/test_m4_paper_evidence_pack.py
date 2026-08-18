import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT / "src" / "ghost_mgg_real" / "scripts" / "generate_m4_paper_evidence_pack.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m4_paper_evidence_pack", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sample_inputs(tmp_path: Path):
    algorithm_gate = _write_json(
        tmp_path / "algorithm_gate.json",
        {
            "schema_version": "m4_real_algorithm_gate_v1",
            "overall_status": "pass",
            "num_scenes": 11,
            "num_evaluable_scenes": 10,
            "num_excluded_scenes": 1,
            "evaluable_top1_changed_count": 10,
            "evaluable_shape_changed_count": 7,
            "mean_evaluable_failure_score_delta": 0.615358,
            "quality_counts": {"good": 8, "questionable": 2, "ood": 1},
        },
    )
    ablation_gate = _write_json(
        tmp_path / "ablation_gate.json",
        {
            "schema_version": "m4_real_ablation_gate_v1",
            "overall_status": "pass",
            "num_scenes": 11,
            "num_evaluable_scenes": 10,
            "num_excluded_scenes": 1,
            "ablation_summary": {
                "without_failure": {
                    "checked_count": 10,
                    "top1_changed_count": 10,
                    "shape_changed_count": 7,
                    "mean_full_total_advantage": 1.927942,
                },
                "without_table_leakage": {
                    "checked_count": 10,
                    "top1_changed_count": 3,
                    "shape_changed_count": 3,
                    "mean_full_total_advantage": 0.741838,
                },
                "without_weak_depth": {
                    "checked_count": 10,
                    "top1_changed_count": 0,
                    "shape_changed_count": 0,
                    "mean_full_total_advantage": 0.018513,
                },
            },
        },
    )
    failure_atlas = _write_json(
        tmp_path / "failure_atlas.json",
        {
            "schema_version": "m4_failure_atlas_v1",
            "num_scenes": 11,
            "category_counts": {
                "hole_dominant": 7,
                "table_leakage": 10,
                "mixed_hole_leakage": 8,
                "table_leakage_ablation_effect": 3,
                "weak_failure": 1,
                "ood_or_primitive_mismatch": 1,
            },
            "representatives": {
                "hole_dominant": {"scene_id": "daylight_glass_cup_001"},
                "table_leakage": {"scene_id": "daylight_jelly_001"},
            },
        },
    )
    return algorithm_gate, ablation_gate, failure_atlas


def test_build_paper_evidence_pack_lists_supported_and_unsupported_claims(tmp_path):
    module = _load_module()
    algorithm_gate, ablation_gate, failure_atlas = _sample_inputs(tmp_path)

    pack = module.build_paper_evidence_pack(
        algorithm_gate_json=algorithm_gate,
        ablation_gate_json=ablation_gate,
        failure_atlas_json=failure_atlas,
    )

    assert pack["schema_version"] == "m4_paper_evidence_pack_v1"
    assert pack["overall_readiness"] == "m4_real_replay_evidence_ready"
    supported_ids = {claim["claim_id"] for claim in pack["supported_claims"]}
    unsupported_ids = {claim["claim_id"] for claim in pack["unsupported_claims"]}
    assert "real_d435_failure_evidence_is_observable" in supported_ids
    assert "failure_evidence_changes_real_replay_ranking" in supported_ids
    assert "table_leakage_has_measurable_ablation_effect" in supported_ids
    assert "real_robot_lift_and_hold_success" in unsupported_ids
    assert "statistically_significant_grasp_improvement" in unsupported_ids
    assert pack["key_metrics"]["evaluable_scenes"] == 10
    assert pack["key_metrics"]["without_table_leakage_top1_changed"] == 3
    assert pack["next_steps"][0]["stage"] == "M5/M6"


def test_write_paper_evidence_pack_outputs_json_markdown_and_html(tmp_path):
    module = _load_module()
    algorithm_gate, ablation_gate, failure_atlas = _sample_inputs(tmp_path)
    pack = module.build_paper_evidence_pack(
        algorithm_gate_json=algorithm_gate,
        ablation_gate_json=ablation_gate,
        failure_atlas_json=failure_atlas,
    )

    module.write_paper_evidence_pack(pack, tmp_path / "pack")

    payload = json.loads((tmp_path / "pack" / "evidence_pack.json").read_text())
    assert payload["schema_version"] == "m4_paper_evidence_pack_v1"
    markdown = (tmp_path / "pack" / "index.md").read_text(encoding="utf-8")
    assert "M4 Paper Evidence Pack" in markdown
    assert "Supported Claims" in markdown
    assert "Unsupported Claims" in markdown
    html = (tmp_path / "pack" / "index.html").read_text(encoding="utf-8")
    assert "M4 Paper Evidence Pack" in html
    assert "real_robot_lift_and_hold_success" in html


def test_paper_evidence_pack_run_script_contract_and_doc_exist():
    script_path = REPO_ROOT / "scripts" / "run_m4_paper_evidence_pack.sh"
    assert script_path.exists()
    script = script_path.read_text(encoding="utf-8")
    for required in [
        "run_m4_failure_atlas.sh",
        "generate_m4_paper_evidence_pack.py",
        "reports/m4_paper_evidence_pack",
        "evidence_pack.json",
        "M4 paper evidence pack passed",
    ]:
        assert required in script
    for forbidden in ["gz model", "gz service", "current_targets.json"]:
        assert forbidden not in script

    doc_path = REPO_ROOT / "docs" / "superpowers" / "plans" / "2026-07-04-m4-paper-evidence-pack.md"
    assert doc_path.exists()
    doc = doc_path.read_text(encoding="utf-8")
    assert "M4 Paper Evidence Pack" in doc
    assert "unsupported claims" in doc
    assert "run_m4_paper_evidence_pack.sh" in doc
