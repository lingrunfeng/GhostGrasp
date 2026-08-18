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
    / "generate_m4_graspability_dryrun.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m4_graspability_dryrun", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric_proxy_payload() -> dict:
    return {
        "schema_version": "m4_metric_proxies_v1",
        "num_rows": 2,
        "num_scenes": 1,
        "rows": [
            {
                "scene_id": "scene_a",
                "target_label": "cup",
                "shape_hint": "cup_like",
                "ranker": "failure_aware",
                "rank": 1,
                "hypothesis_id": "target",
                "shape_type": "box",
                "center_x_m": 0.10,
                "center_y_m": 0.05,
                "center_z_m": 0.77,
                "width_m": 0.025,
                "depth_m": 0.025,
                "height_m": 0.04,
                "table_z_m": 0.75,
                "table_depth_m": 1.0,
                "source_center_u": 370.0,
                "source_center_v": 260.0,
                "visual_score": 0.8,
                "failure_score": 0.6,
                "total_score": 1.4,
            },
            {
                "scene_id": "scene_a",
                "target_label": "neighbor",
                "shape_hint": "box",
                "ranker": "failure_aware",
                "rank": 1,
                "hypothesis_id": "neighbor",
                "shape_type": "box",
                "center_x_m": 0.18,
                "center_y_m": 0.05,
                "center_z_m": 0.77,
                "width_m": 0.025,
                "depth_m": 0.025,
                "height_m": 0.04,
                "table_z_m": 0.75,
                "table_depth_m": 1.0,
                "source_center_u": 410.0,
                "source_center_v": 260.0,
                "visual_score": 0.6,
                "failure_score": 0.1,
                "total_score": 0.7,
            },
        ],
    }


def test_score_top_grasp_accepts_reachable_proxy_with_width_margin():
    module = _load_module()
    proxy = _metric_proxy_payload()["rows"][0]

    row = module.score_top_grasp(
        proxy,
        neighbors=[],
        max_gripper_width_m=0.07,
        workspace_radius_m=0.35,
        min_neighbor_clearance_m=0.03,
    )

    assert row.hypothesis_id == "target"
    assert row.valid is True
    assert row.required_gripper_width_m == 0.037
    assert row.gripper_width_margin_m > 0.03
    assert row.source_center_u == 370.0
    assert row.source_center_v == 260.0
    assert row.table_depth_m == 1.0
    assert row.failure_reason == ""
    assert row.score > 0.5


def test_score_top_grasp_uses_shorter_top_grasp_axis_for_rectangular_proxy():
    module = _load_module()
    proxy = dict(_metric_proxy_payload()["rows"][0])
    proxy["width_m"] = 0.025
    proxy["depth_m"] = 0.060

    row = module.score_top_grasp(
        proxy,
        neighbors=[],
        max_gripper_width_m=0.07,
        workspace_radius_m=0.35,
        min_neighbor_clearance_m=0.03,
    )

    assert row.valid is True
    assert row.required_gripper_width_m == 0.037
    assert row.grasp_width_axis == "x"
    assert row.grasp_width_base_m == 0.025


def test_score_top_grasp_rejects_close_neighbor():
    module = _load_module()
    target, neighbor = _metric_proxy_payload()["rows"]
    neighbor = dict(neighbor)
    neighbor["center_x_m"] = 0.115

    row = module.score_top_grasp(
        target,
        neighbors=[neighbor],
        max_gripper_width_m=0.07,
        workspace_radius_m=0.35,
        min_neighbor_clearance_m=0.03,
    )

    assert row.valid is False
    assert row.failure_reason == "neighbor_clearance"
    assert row.nearest_neighbor_clearance_m < 0.03


def test_graspability_report_writes_json_csv_and_index(tmp_path):
    module = _load_module()
    metric_path = tmp_path / "metric_proxies.json"
    metric_path.write_text(json.dumps(_metric_proxy_payload()), encoding="utf-8")
    output_dir = tmp_path / "graspability"

    rows = module.generate_graspability_report(
        metric_proxies_json=metric_path,
        output_dir=output_dir,
        max_gripper_width_m=0.07,
        workspace_radius_m=0.35,
        min_neighbor_clearance_m=0.03,
    )

    assert len(rows) == 2
    payload = json.loads((output_dir / "graspability.json").read_text())
    assert payload["schema_version"] == "m4_graspability_dryrun_v1"
    assert payload["num_rows"] == 2
    assert "valid_count" in payload
    assert (output_dir / "graspability.csv").exists()
    index = (output_dir / "index.md").read_text()
    assert "M4 Graspability Dry-Run" in index
    assert "target" in index


def test_report_does_not_treat_same_target_alternate_ranker_as_neighbor(tmp_path):
    module = _load_module()
    payload = _metric_proxy_payload()
    payload["rows"][1]["target_label"] = "cup"
    payload["rows"][1]["hypothesis_id"] = "target_alternate"
    payload["rows"][1]["center_x_m"] = 0.105
    metric_path = tmp_path / "metric_proxies.json"
    metric_path.write_text(json.dumps(payload), encoding="utf-8")

    rows = module.generate_graspability_report(
        metric_proxies_json=metric_path,
        output_dir=tmp_path / "graspability",
        max_gripper_width_m=0.07,
        workspace_radius_m=0.35,
        min_neighbor_clearance_m=0.03,
    )

    assert all(row.nearest_neighbor_clearance_m == 999.0 for row in rows)
    assert all(row.failure_reason == "" for row in rows)
