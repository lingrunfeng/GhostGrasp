import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "generate_m5_real_ranking_report.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m5_real_ranking_report", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _failure_synthetic_scene(module):
    target_mask = np.zeros((20, 20), dtype=bool)
    target_mask[6:14, 6:14] = True
    current_depth = np.zeros((20, 20), dtype=np.uint16) + 1000
    background_depth = np.zeros((20, 20), dtype=np.uint16)
    background_depth[8:12, 8:12] = 1000
    current_depth[8:12, 8:12] = 0

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
            hypothesis_id="failure_region",
            shape_type="box",
            center_uv=(9.5, 9.5),
            size_px=(4.0, 4.0),
            depth_m=1.0,
            height_m=0.1,
        ),
    ]
    return target_mask, current_depth, background_depth, hypotheses


def test_evidence_maps_from_depth_background_sets_expected_channels():
    module = _load_module()
    target_mask = np.array([[True, True, True, True]], dtype=bool)
    current_depth = np.array([[0, 1000, 850, 1200]], dtype=np.uint16)
    background_depth = np.array([[1000, 1005, 1000, 1000]], dtype=np.uint16)

    evidence = module.evidence_maps_from_depth_background(
        target_mask,
        current_depth,
        background_depth,
        leakage_tolerance_mm=15,
        foreground_margin_mm=50,
    )

    assert evidence.hole[0, 0] == 1.0
    assert evidence.table_leakage[0, 1] == 1.0
    assert evidence.foreground_support[0, 2] == 1.0
    assert evidence.valid[0, 3] == 1.0
    assert evidence.edge.sum() == 0.0
    assert evidence.flying_point.sum() == 0.0


def test_rank_scene_rows_include_two_rankers_and_failure_can_change_top_rank():
    module = _load_module()
    target_mask, current_depth, background_depth, hypotheses = _failure_synthetic_scene(module)

    rows = module.rank_real_scene(
        scene_id="scene_a",
        target_label="transparent_jelly_cup",
        shape_hint="cup_like",
        target_mask=target_mask,
        current_depth=current_depth,
        background_depth=background_depth,
        hypotheses=hypotheses,
        top_k=2,
    )

    assert len(rows) == 4
    silhouette_top = [row for row in rows if row.ranker == "silhouette_only" and row.rank == 1][0]
    failure_top = [row for row in rows if row.ranker == "failure_aware" and row.rank == 1][0]
    assert silhouette_top.hypothesis_id == "silhouette_bbox"
    assert failure_top.hypothesis_id == "failure_region"
    assert failure_top.failure_score > silhouette_top.failure_score
    assert failure_top.failure_inside_hole > silhouette_top.failure_inside_hole
    assert failure_top.failure_inside_table_leakage == 0.0
    assert failure_top.failure_total_check == failure_top.failure_score


def test_calibrated_weights_json_can_disable_failure_gain_and_prior(tmp_path):
    module = _load_module()
    target_mask, current_depth, background_depth, hypotheses = _failure_synthetic_scene(module)
    weights_path = tmp_path / "best_weights.json"
    weights_path.write_text(
        json.dumps(
            {
                "schema_version": "m4_real_weight_calibration_best_v1",
                "best_weights": {"visual": 1.0, "failure": 0.0, "depth": 0.0},
            }
        ),
        encoding="utf-8",
    )

    weights = module.load_failure_aware_weights(weights_path)
    rows = module.rank_real_scene(
        scene_id="scene_a",
        target_label="transparent_jelly_cup",
        shape_hint="cup_like",
        target_mask=target_mask,
        current_depth=current_depth,
        background_depth=background_depth,
        hypotheses=hypotheses,
        top_k=2,
        failure_aware_weights=weights,
    )

    assert weights["prior"] == 0.0
    failure_top = [row for row in rows if row.ranker == "failure_aware" and row.rank == 1][0]
    assert failure_top.hypothesis_id == "silhouette_bbox"


def test_write_ranking_reports_creates_json_csv_and_index(tmp_path):
    module = _load_module()
    row = module.RealRankingRow(
        scene_id="scene_a",
        target_label="cup",
        shape_hint="cup_like",
        ranker="failure_aware",
        rank=1,
        hypothesis_id="box_s1.00",
        shape_type="box",
        center_u=10.0,
        center_v=11.0,
        size_u_px=12.0,
        size_v_px=13.0,
        visual_score=0.5,
        failure_score=0.7,
        failure_inside_hole=0.2,
        failure_inside_table_leakage=0.3,
        failure_boundary_edge=0.0,
        failure_boundary_flying_point=0.0,
        failure_outside_hole_penalty=0.1,
        failure_outside_table_leakage_penalty=0.0,
        failure_total_check=0.7,
        depth_score=0.0,
        total_score=3.3,
        validation_state="accepted",
    )

    module.write_ranking_reports(
        [row],
        tmp_path,
        failure_aware_weights={
            "visual": 0.75,
            "failure": 1.0,
            "depth": 0.0,
            "physical": 0.0,
            "grasp": 0.0,
            "prior": 0.0,
        },
    )

    assert (tmp_path / "m5_real_ranking.json").exists()
    assert (tmp_path / "m5_real_ranking.csv").exists()
    assert (tmp_path / "index.md").exists()
    assert b"\r\n" not in (tmp_path / "m5_real_ranking.csv").read_bytes()
    payload = json.loads((tmp_path / "m5_real_ranking.json").read_text())
    assert payload["schema_version"] == "m5_real_ranking_v1"
    assert payload["num_rows"] == 1
    assert payload["failure_aware_weights"]["failure"] == 1.0
    assert payload["failure_aware_weights"]["prior"] == 0.0
    assert payload["rows"][0]["hypothesis_id"] == "box_s1.00"
    assert payload["rows"][0]["failure_inside_hole"] == 0.2
    assert payload["rows"][0]["failure_total_check"] == payload["rows"][0]["failure_score"]
    csv_text = (tmp_path / "m5_real_ranking.csv").read_text()
    assert "failure_inside_table_leakage" in csv_text
    assert "failure_outside_hole_penalty" in csv_text
    index = (tmp_path / "index.md").read_text()
    assert "failure-aware weights" in index
    assert "Failure Likelihood Breakdown" in index
    assert "failure_inside_hole" in index


def test_write_ranking_reports_creates_top1_comparison(tmp_path):
    module = _load_module()

    rows = [
        module.RealRankingRow(
            scene_id="scene_a",
            target_label="cup",
            shape_hint="cup_like",
            ranker="silhouette_only",
            rank=1,
            hypothesis_id="cylinder_s1.00",
            shape_type="cylinder",
            center_u=10.0,
            center_v=11.0,
            size_u_px=12.0,
            size_v_px=13.0,
            visual_score=0.90,
            failure_score=-0.10,
            failure_inside_hole=0.0,
            failure_inside_table_leakage=0.0,
            failure_boundary_edge=0.0,
            failure_boundary_flying_point=0.0,
            failure_outside_hole_penalty=0.1,
            failure_outside_table_leakage_penalty=0.0,
            failure_total_check=-0.10,
            depth_score=0.0,
            total_score=0.90,
            validation_state="accepted",
        ),
        module.RealRankingRow(
            scene_id="scene_a",
            target_label="cup",
            shape_hint="cup_like",
            ranker="failure_aware",
            rank=1,
            hypothesis_id="box_s1.00",
            shape_type="box",
            center_u=10.0,
            center_v=11.0,
            size_u_px=12.0,
            size_v_px=13.0,
            visual_score=0.84,
            failure_score=0.67,
            failure_inside_hole=0.7,
            failure_inside_table_leakage=0.0,
            failure_boundary_edge=0.0,
            failure_boundary_flying_point=0.0,
            failure_outside_hole_penalty=0.03,
            failure_outside_table_leakage_penalty=0.0,
            failure_total_check=0.67,
            depth_score=0.0,
            total_score=3.50,
            validation_state="accepted",
        ),
    ]

    module.write_ranking_reports(rows, tmp_path)

    summary = json.loads((tmp_path / "top1_comparison.json").read_text())
    assert summary["schema_version"] == "m5_real_top1_comparison_v1"
    assert summary["num_scenes"] == 1
    assert summary["top1_changed_count"] == 1
    assert summary["shape_changed_count"] == 1
    assert summary["rows"][0]["top1_changed"] is True
    assert summary["rows"][0]["failure_score_delta"] == 0.77
    assert (tmp_path / "top1_comparison.csv").exists()
    index = (tmp_path / "index.md").read_text()
    assert "Top-1 Comparison" in index
    assert "failure_score_delta" in index
