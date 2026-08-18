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
    / "generate_m4_real_algorithm_gate.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m4_real_algorithm_gate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sample_inputs(tmp_path: Path, *, changed: bool = True) -> tuple[Path, Path]:
    ranking_dir = tmp_path / "ranking"
    _write_json(
        ranking_dir / "top1_comparison.json",
        {
            "schema_version": "m5_real_top1_comparison_v1",
            "num_scenes": 3,
            "top1_changed_count": 2 if changed else 0,
            "shape_changed_count": 1 if changed else 0,
            "rows": [
                {
                    "scene_id": "good_scene",
                    "target_label": "transparent_jelly_cup",
                    "shape_hint": "cup_like",
                    "silhouette_top": "cylinder_s1.00",
                    "failure_top": "box_s1.00" if changed else "cylinder_s1.00",
                    "silhouette_shape": "cylinder",
                    "failure_shape": "box" if changed else "cylinder",
                    "top1_changed": changed,
                    "shape_changed": changed,
                    "silhouette_visual_score": 0.91,
                    "failure_visual_score": 0.84,
                    "silhouette_failure_score": -0.10,
                    "failure_failure_score": 0.62,
                    "visual_score_delta": -0.07,
                    "failure_score_delta": 0.72 if changed else 0.0,
                },
                {
                    "scene_id": "questionable_scene",
                    "target_label": "multi_objects",
                    "shape_hint": "unknown",
                    "silhouette_top": "cylinder_s1.00",
                    "failure_top": "box_s1.00" if changed else "cylinder_s1.00",
                    "silhouette_shape": "cylinder",
                    "failure_shape": "box" if changed else "cylinder",
                    "top1_changed": changed,
                    "shape_changed": changed,
                    "silhouette_visual_score": 0.88,
                    "failure_visual_score": 0.79,
                    "silhouette_failure_score": -0.20,
                    "failure_failure_score": 0.50,
                    "visual_score_delta": -0.09,
                    "failure_score_delta": 0.70 if changed else 0.0,
                },
                {
                    "scene_id": "spoon_scene",
                    "target_label": "metal_spoon",
                    "shape_hint": "unknown",
                    "silhouette_top": "box_s0.95",
                    "failure_top": "box_s1.00",
                    "silhouette_shape": "box",
                    "failure_shape": "box",
                    "top1_changed": True,
                    "shape_changed": False,
                    "silhouette_visual_score": 0.75,
                    "failure_visual_score": 0.70,
                    "silhouette_failure_score": -0.40,
                    "failure_failure_score": 0.30,
                    "visual_score_delta": -0.05,
                    "failure_score_delta": 0.70,
                },
            ],
        },
    )
    _write_json(
        ranking_dir / "m5_real_ranking.json",
        {
            "schema_version": "m5_real_ranking_v1",
            "num_scenes": 3,
            "num_rows": 18,
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
    dashboard = _write_json(
        tmp_path / "dashboard" / "dashboard.json",
        {
            "schema_version": "m4_real_external_mask_visual_dashboard_v1",
            "num_scenes": 3,
            "cards": [
                {
                    "scene_id": "good_scene",
                    "quality": {
                        "status": "good",
                        "reasons": ["clear evidence"],
                    },
                },
                {
                    "scene_id": "questionable_scene",
                    "quality": {
                        "status": "questionable",
                        "reasons": ["multi-object scene"],
                    },
                },
                {
                    "scene_id": "spoon_scene",
                    "quality": {
                        "status": "ood",
                        "reasons": ["outside primitive family"],
                    },
                },
            ],
        },
    )
    return ranking_dir, dashboard


def test_algorithm_gate_uses_quality_layers_and_excludes_ood(tmp_path):
    module = _load_module()
    ranking_dir, dashboard = _sample_inputs(tmp_path)

    report = module.build_algorithm_gate_report(
        ranking_dir=ranking_dir,
        dashboard_json=dashboard,
    )

    assert report["schema_version"] == "m4_real_algorithm_gate_v1"
    assert report["overall_status"] == "pass"
    assert report["num_scenes"] == 3
    assert report["num_evaluable_scenes"] == 2
    assert report["num_excluded_scenes"] == 1
    assert report["evaluable_top1_changed_count"] == 2
    assert report["evaluable_shape_changed_count"] == 2
    assert report["mean_evaluable_failure_score_delta"] == 0.71
    assert report["rows"][0]["quality_status"] == "good"
    assert report["rows"][2]["gate_decision"] == "excluded_ood"


def test_algorithm_gate_fails_without_failure_driven_ranking_change(tmp_path):
    module = _load_module()
    ranking_dir, dashboard = _sample_inputs(tmp_path, changed=False)

    report = module.build_algorithm_gate_report(
        ranking_dir=ranking_dir,
        dashboard_json=dashboard,
    )

    assert report["overall_status"] == "review"
    assert report["evaluable_top1_changed_count"] == 0
    assert "failure-aware top-1 did not differ" in report["gate_reasons"][0]


def test_write_algorithm_gate_outputs_json_csv_and_markdown(tmp_path):
    module = _load_module()
    ranking_dir, dashboard = _sample_inputs(tmp_path)
    report = module.build_algorithm_gate_report(ranking_dir=ranking_dir, dashboard_json=dashboard)

    module.write_algorithm_gate_report(report, tmp_path / "out")

    payload = json.loads((tmp_path / "out" / "algorithm_gate.json").read_text())
    assert payload["schema_version"] == "m4_real_algorithm_gate_v1"
    assert payload["overall_status"] == "pass"

    with (tmp_path / "out" / "algorithm_gate.csv").open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["scene_id"] == "good_scene"
    assert rows[0]["quality_status"] == "good"
    assert rows[2]["gate_decision"] == "excluded_ood"

    index = (tmp_path / "out" / "index.md").read_text(encoding="utf-8")
    assert "M4 Real Algorithm Gate" in index
    assert "overall_status: pass" in index
    assert "excluded_ood" in index


def test_algorithm_gate_run_script_contract_is_no_truth():
    script_path = REPO_ROOT / "scripts" / "run_m4_real_algorithm_gate.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "generate_m5_real_ranking_report.py",
        "run_m4_real_external_mask_visual_dashboard.sh",
        "generate_m4_real_algorithm_gate.py",
        "reports/m4_real_algorithm_gate",
        "algorithm_gate.json",
        "algorithm_gate.csv",
        "M4 real algorithm gate passed",
    ]:
        assert required in source

    for forbidden in [
        "gz model",
        "gz service",
        "current_targets.json",
        "m4_sim_grasp_targets.json",
    ]:
        assert forbidden not in source
