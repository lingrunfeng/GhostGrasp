import json
from pathlib import Path

import numpy as np

from ghost_mgg_core_py.evaluation.m3_capture_arrays import build_m3_capture_arrays
from ghost_mgg_core_py.evaluation.m3_dataset import load_m3_capture
from ghost_mgg_core_py.evaluation.m4_ablation_eval import run_ablation_eval
from ghost_mgg_core_py.evaluation.m4_geometry_eval import (
    run_geometry_eval,
    truth_from_sample,
    write_geometry_eval_reports,
)
from ghost_mgg_core_py.evaluation.m4_offline_ranking import run_offline_ranking, write_ranking_reports
from ghost_mgg_core_py.evaluation.m4_minimal_benchmark import run_benchmark, write_reports


def _write_sample(root: Path, scenario_id: str, failure_mode: str, summary: dict, with_arrays=False):
    sample_dir = root / scenario_id
    sample_dir.mkdir()
    metadata = {
        "schema_version": "m4_m3_sample_v1",
        "scenario_id": scenario_id,
        "failure_mode": failure_mode,
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (sample_dir / "evidence_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest_item = {
        "scenario_id": scenario_id,
        "failure_mode": failure_mode,
        "metadata_path": f"{scenario_id}/metadata.json",
        "evidence_summary_path": f"{scenario_id}/evidence_summary.json",
    }
    if with_arrays:
        arrays = build_m3_capture_arrays(scenario_id, summary, metadata)
        np.savez_compressed(sample_dir / "arrays.npz", **arrays)
        manifest_item["arrays_path"] = f"{scenario_id}/arrays.npz"
    return manifest_item


def test_load_m3_capture_reads_manifest_samples(tmp_path):
    manifest = {
        "schema_version": "m4_m3_capture_v1",
        "scenarios": [
            _write_sample(
                tmp_path,
                "S4",
                "table_leakage",
                {
                    "failure_mode": "table_leakage",
                    "total_pixels": 100,
                    "roi_pixels": 10,
                    "valid_depth_ratio": 1.0,
                    "hole_ratio": 0.0,
                    "table_leakage_ratio": 1.0,
                },
            )
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    samples = load_m3_capture(tmp_path)

    assert len(samples) == 1
    assert samples[0].scenario_id == "S4"
    assert samples[0].failure_mode == "table_leakage"
    assert samples[0].ratio("table_leakage_ratio") == 1.0


def test_m4_minimal_benchmark_scores_failure_aware_rows(tmp_path):
    manifest = {
        "schema_version": "m4_m3_capture_v1",
        "scenarios": [
            _write_sample(
                tmp_path,
                "S0",
                "disabled",
                {
                    "failure_mode": "disabled",
                    "total_pixels": 100,
                    "roi_pixels": 10,
                    "valid_depth_ratio": 1.0,
                    "hole_ratio": 0.0,
                    "table_leakage_ratio": 0.0,
                    "edge_ratio": 0.0,
                    "flying_point_ratio": 0.0,
                    "biased_depth_ratio": 0.0,
                },
            ),
            _write_sample(
                tmp_path,
                "S4",
                "table_leakage",
                {
                    "failure_mode": "table_leakage",
                    "total_pixels": 100,
                    "roi_pixels": 10,
                    "valid_depth_ratio": 1.0,
                    "hole_ratio": 0.0,
                    "table_leakage_ratio": 1.0,
                    "edge_ratio": 0.0,
                    "flying_point_ratio": 0.0,
                    "biased_depth_ratio": 0.0,
                },
            ),
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rows = run_benchmark(tmp_path)

    assert len(rows) == 2
    assert rows[0].predicted_failure_family == "none"
    assert rows[1].predicted_failure_family == "table_leakage"
    assert rows[1].failure_aware_simple_score > rows[0].failure_aware_simple_score
    assert rows[1].mask_extrusion_score < rows[0].mask_extrusion_score


def test_m4_minimal_benchmark_writes_csv_and_json(tmp_path):
    manifest = {
        "schema_version": "m4_m3_capture_v1",
        "scenarios": [
            _write_sample(
                tmp_path,
                "S3",
                "hole",
                {
                    "failure_mode": "hole",
                    "total_pixels": 100,
                    "roi_pixels": 10,
                    "valid_depth_ratio": 0.5,
                    "hole_ratio": 0.5,
                    "table_leakage_ratio": 0.0,
                    "edge_ratio": 0.0,
                    "flying_point_ratio": 0.0,
                    "biased_depth_ratio": 0.0,
                },
            )
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = run_benchmark(tmp_path)
    output_csv = tmp_path / "report.csv"
    output_json = tmp_path / "report.json"

    write_reports(rows, output_csv, output_json)

    assert "scenario_id,failure_mode" in output_csv.read_text(encoding="utf-8")
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == "m4_minimal_benchmark_v1"
    assert report["num_samples"] == 1


def test_build_m3_capture_arrays_matches_key_summary_counts():
    summary = {
        "failure_mode": "table_leakage",
        "total_pixels": 307200,
        "roi_pixels": 14946,
        "valid_depth_ratio": 1.0,
        "hole_ratio": 0.0,
        "table_leakage_ratio": 1.0,
    }

    arrays = build_m3_capture_arrays("S4", summary, {"failure_mode": "table_leakage"})

    assert arrays["target_mask"].shape == (480, 640)
    assert int(arrays["target_mask"].sum()) == 14946
    assert int(arrays["evidence_table_leakage"].sum()) == 14946
    assert int(arrays["evidence_hole"].sum()) == 0
    assert np.allclose(arrays["corrupted_depth_m"][arrays["target_mask"]], 1.20)


def test_load_m3_capture_loads_arrays_when_present(tmp_path):
    summary = {
        "failure_mode": "hole",
        "total_pixels": 307200,
        "roi_pixels": 14946,
        "valid_depth_ratio": 0.951348,
        "hole_ratio": 1.0,
        "table_leakage_ratio": 0.0,
        "edge_ratio": 0.0,
        "flying_point_ratio": 0.0,
        "biased_depth_ratio": 0.0,
    }
    manifest = {
        "schema_version": "m4_m3_capture_v1",
        "scenarios": [_write_sample(tmp_path, "S3", "hole", summary, with_arrays=True)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    sample = load_m3_capture(tmp_path)[0]
    arrays = sample.load_arrays()

    assert arrays["target_mask"].dtype == np.bool_
    assert int(arrays["evidence_hole"].sum()) == 14946
    assert np.isnan(arrays["corrupted_depth_m"][arrays["target_mask"]]).all()


def test_m4_offline_ranking_runs_two_rankers(tmp_path):
    summary = {
        "failure_mode": "mixed",
        "total_pixels": 307200,
        "roi_pixels": 14946,
        "valid_depth_ratio": 0.975674,
        "hole_ratio": 0.5,
        "table_leakage_ratio": 0.5,
        "edge_ratio": 0.0,
        "flying_point_ratio": 0.0,
        "biased_depth_ratio": 0.0,
    }
    manifest = {
        "schema_version": "m4_m3_capture_v1",
        "scenarios": [_write_sample(tmp_path, "S6", "mixed", summary, with_arrays=True)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rows = run_offline_ranking(tmp_path, top_k=2)

    assert len(rows) == 4
    assert {row.ranker for row in rows} == {"silhouette_only", "failure_aware"}
    assert all(row.validation_state == "accepted" for row in rows)
    assert all(row.shape_type in {"box", "cylinder"} for row in rows)


def test_m4_offline_ranking_writes_reports(tmp_path):
    summary = {
        "failure_mode": "disabled",
        "total_pixels": 307200,
        "roi_pixels": 14946,
        "valid_depth_ratio": 1.0,
        "hole_ratio": 0.0,
        "table_leakage_ratio": 0.0,
        "edge_ratio": 0.0,
        "flying_point_ratio": 0.0,
        "biased_depth_ratio": 0.0,
    }
    manifest = {
        "schema_version": "m4_m3_capture_v1",
        "scenarios": [_write_sample(tmp_path, "S0", "disabled", summary, with_arrays=True)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = run_offline_ranking(tmp_path, top_k=1)
    output_csv = tmp_path / "ranking.csv"
    output_json = tmp_path / "ranking.json"

    write_ranking_reports(rows, output_csv, output_json)

    assert "scenario_id,failure_mode,ranker" in output_csv.read_text(encoding="utf-8")
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == "m4_offline_ranking_v1"
    assert report["num_rows"] == 2


def test_m4_geometry_truth_uses_target_mask_bbox(tmp_path):
    summary = {
        "failure_mode": "disabled",
        "total_pixels": 307200,
        "roi_pixels": 14946,
        "valid_depth_ratio": 1.0,
        "hole_ratio": 0.0,
        "table_leakage_ratio": 0.0,
        "edge_ratio": 0.0,
        "flying_point_ratio": 0.0,
        "biased_depth_ratio": 0.0,
    }
    manifest = {
        "schema_version": "m4_m3_capture_v1",
        "scenarios": [_write_sample(tmp_path, "S0", "disabled", summary, with_arrays=True)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    sample = load_m3_capture(tmp_path)[0]
    truth = truth_from_sample(sample)

    assert truth.shape_type == "box"
    assert truth.center_u == 320.0
    assert truth.center_v == 277.5
    assert truth.size_u_px == 141.0
    assert truth.size_v_px == 106.0


def test_m4_geometry_eval_reports_top1_metrics(tmp_path):
    summary = {
        "failure_mode": "mixed",
        "total_pixels": 307200,
        "roi_pixels": 14946,
        "valid_depth_ratio": 0.975674,
        "hole_ratio": 0.5,
        "table_leakage_ratio": 0.5,
        "edge_ratio": 0.0,
        "flying_point_ratio": 0.0,
        "biased_depth_ratio": 0.0,
    }
    manifest = {
        "schema_version": "m4_m3_capture_v1",
        "scenarios": [_write_sample(tmp_path, "S6", "mixed", summary, with_arrays=True)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rows = run_geometry_eval(tmp_path, top_k=3)

    assert {row.ranker for row in rows} == {"silhouette_only", "failure_aware"}
    assert all(row.truth_shape_type == "box" for row in rows)
    assert all(row.top1_shape_correct for row in rows)
    assert all(row.topk_contains_exact_proxy for row in rows)
    assert all(row.top1_silhouette_iou == 1.0 for row in rows)


def test_m4_geometry_eval_writes_aggregate_report(tmp_path):
    summary = {
        "failure_mode": "hole",
        "total_pixels": 307200,
        "roi_pixels": 14946,
        "valid_depth_ratio": 0.951348,
        "hole_ratio": 1.0,
        "table_leakage_ratio": 0.0,
        "edge_ratio": 0.0,
        "flying_point_ratio": 0.0,
        "biased_depth_ratio": 0.0,
    }
    manifest = {
        "schema_version": "m4_m3_capture_v1",
        "scenarios": [_write_sample(tmp_path, "S3", "hole", summary, with_arrays=True)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = run_geometry_eval(tmp_path, top_k=2)
    output_csv = tmp_path / "geometry_eval.csv"
    output_json = tmp_path / "geometry_eval.json"

    write_geometry_eval_reports(rows, output_csv, output_json)

    assert "top1_shape_correct" in output_csv.read_text(encoding="utf-8")
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == "m4_geometry_ranking_eval_v1"
    assert report["aggregate_by_ranker"]["failure_aware"]["top1_shape_accuracy"] == 1.0


def test_m4_ablation_eval_runs_expected_rankers(tmp_path):
    summary = {
        "failure_mode": "table_leakage",
        "total_pixels": 307200,
        "roi_pixels": 14946,
        "valid_depth_ratio": 1.0,
        "hole_ratio": 0.0,
        "table_leakage_ratio": 1.0,
        "edge_ratio": 0.0,
        "flying_point_ratio": 0.0,
        "biased_depth_ratio": 0.0,
    }
    manifest = {
        "schema_version": "m4_m3_capture_v1",
        "scenarios": [_write_sample(tmp_path, "S4", "table_leakage", summary, with_arrays=True)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    ranking_rows, eval_rows = run_ablation_eval(tmp_path, top_k=1)

    expected_rankers = {
        "full",
        "silhouette_only",
        "without_failure",
        "without_table_leakage",
        "without_edge_flying",
        "without_weak_depth",
    }
    assert {row.ranker for row in ranking_rows} == expected_rankers
    assert {row.ranker for row in eval_rows} == expected_rankers
    assert len(ranking_rows) == len(expected_rankers)
    assert len(eval_rows) == len(expected_rankers)
