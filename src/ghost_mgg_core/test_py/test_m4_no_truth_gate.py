import json
from pathlib import Path

from ghost_mgg_core_py.evaluation.m4_no_truth_gate import (
    NoTruthSyntheticRow,
    build_no_truth_gate_report,
    summarize_no_truth_gate,
)


def test_m4_no_truth_gate_writes_json_csv_and_index_with_real_rows(tmp_path):
    real_dashboard = tmp_path / "dashboard.json"
    multi_target = tmp_path / "multi_target.json"
    scenario_sweep = tmp_path / "scenario_sweep.json"
    dynamic_execute = tmp_path / "dynamic_execute.json"
    ranked_fallback = tmp_path / "ranked_fallback.json"
    real_dashboard.write_text(
        json.dumps(
            {
                "schema_version": "m4_real_dashboard_v1",
                "num_scenes": 1,
                "rows": [
                    {
                        "scene_id": "daylight_transparent_jelly_cup_001",
                        "target_label": "transparent_jelly_cup",
                        "failure_top": "box_s1.00",
                        "silhouette_top": "cylinder_s1.00",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    multi_target.write_text(
        json.dumps(
            {
                "summary": {
                    "schema_version": "m4_no_truth_multi_target_eval_v1",
                    "target_model_count": 4,
                    "dryrun_success_rate": 1.0,
                    "execute_success_rate": 1.0,
                    "no_truth_audit_pass_rate": 1.0,
                    "pose_accept_rate": 1.0,
                    "shape_family_match_rate": 0.8,
                    "batch_status": "pass",
                },
                "rows": [
                    {
                        "case_id": "glass_block_box",
                        "target_model": "glass_block",
                        "expected_shape": "box",
                        "predicted_shape": "cylinder",
                        "pose_error_m": 0.008,
                        "dryrun_planned": True,
                        "execute_status_name": "SKIPPED",
                        "no_truth_audit": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scenario_sweep.write_text(
        json.dumps(
            {
                "summary": {
                    "schema_version": "m4_no_truth_scenario_sweep_v1",
                    "scenario_count": 5,
                    "hypothesis_success_rate": 1.0,
                    "pose_accept_rate": 1.0,
                    "no_truth_audit_pass_rate": 1.0,
                    "batch_status": "pass",
                },
                "rows": [
                    {
                        "scenario_id": "S4",
                        "target_model": "red_cube",
                        "hypothesis_count": 3,
                        "hypothesis_id": "box_yaw+022_s0.75",
                        "shape_type": "box",
                        "pose_error_m": 0.012,
                        "confidence": 1.0,
                        "no_truth_audit": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dynamic_execute.write_text(
        json.dumps(
            {
                "schema_version": "m4_joint_hypothesis_execute_v1",
                "hypotheses_topic": "/ghost_mgg/m4_live_hypotheses",
                "selected_hypothesis_id": "box_s1.00",
                "status_name": "SUCCEEDED",
                "final_status_name": "SUCCEEDED",
                "executed_event_published": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ranked_fallback.write_text(
        json.dumps(
            {
                "schema_version": "m4_joint_hypothesis_execute_v1",
                "hypotheses_topic": "/ghost_mgg/m4_live_hypotheses",
                "selected_hypothesis_id": "box_s0.90",
                "status_name": "SUCCEEDED",
                "final_status_name": "SUCCEEDED",
                "executed_event_published": True,
                "attempts": [
                    {
                        "attempt_index": 1,
                        "hypothesis_id": "box_s1.00",
                        "attempt_status_name": "FAILED",
                        "failure_reason": "simulated_failure",
                        "simulated": True,
                        "simulate_first_failure": True,
                    },
                    {
                        "attempt_index": 2,
                        "hypothesis_id": "box_s0.90",
                        "attempt_status_name": "SUCCEEDED",
                        "failure_reason": "",
                        "simulated": False,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_no_truth_gate_report(
        output_dir=tmp_path / "gate",
        real_dashboard_json=real_dashboard,
        multi_target_json=multi_target,
        scenario_sweep_json=scenario_sweep,
        dynamic_execute_json=dynamic_execute,
        ranked_fallback_json=ranked_fallback,
    )

    assert report["schema_version"] == "m4_no_truth_gate_v1"
    assert report["synthetic_top1_success"] == 1.0
    assert report["real_replay_processed"] == 1
    assert report["live_multi_target_models"] == 4
    assert report["live_multi_target_dryrun_success_rate"] == 1.0
    assert report["live_multi_target_pose_accept_rate"] == 1.0
    assert report["live_multi_target_shape_match_rate"] == 0.8
    assert report["live_scenario_sweep_count"] == 5
    assert report["live_scenario_sweep_status"] == "pass"
    assert report["live_scenario_hypothesis_success_rate"] == 1.0
    assert report["live_scenario_pose_accept_rate"] == 1.0
    assert report["live_scenario_no_truth_rate"] == 1.0
    assert report["live_dynamic_execute_status"] == "SUCCEEDED"
    assert report["live_dynamic_execute_event_published"] is True
    assert report["live_ranked_fallback_status"] == "SUCCEEDED"
    assert report["live_ranked_fallback_attempt_count"] == 2
    assert report["live_ranked_fallback_first_attempt_simulated"] is True
    assert report["full_vs_silhouette_delta"] > 0.0
    assert report["without_failure_delta"] > 0.0
    assert report["no_truth_audit_pass"] is True
    assert report["gate_status"] == "pass"
    assert report["known_gaps"]
    assert any(row["category"] == "synthetic" for row in report["rows"])
    assert any(row["category"] == "real_replay" for row in report["rows"])
    assert any(row["category"] == "ablation" for row in report["rows"])
    assert any(row["category"] == "no_truth_audit" for row in report["rows"])
    assert any(row["category"] == "live_multi_target" for row in report["rows"])
    assert any(row["category"] == "live_scenario_sweep" for row in report["rows"])
    assert any(row["category"] == "live_dynamic_execute" for row in report["rows"])
    assert any(row["category"] == "live_ranked_fallback" for row in report["rows"])

    assert (tmp_path / "gate" / "gate.json").exists()
    assert "category,scenario_id,ranker" in (tmp_path / "gate" / "gate.csv").read_text(
        encoding="utf-8"
    )
    assert "M4 No-Truth Gate" in (tmp_path / "gate" / "index.md").read_text(encoding="utf-8")


def test_m4_no_truth_gate_fails_when_full_does_not_beat_silhouette():
    rows = [
        NoTruthSyntheticRow(
            scene_id="synthetic_bad",
            ranker="silhouette_only",
            ground_truth_id="box_gt",
            top1_hypothesis_id="box_gt",
            top1_correct=True,
            ground_truth_rank=1,
        ),
        NoTruthSyntheticRow(
            scene_id="synthetic_bad",
            ranker="full",
            ground_truth_id="box_gt",
            top1_hypothesis_id="box_gt",
            top1_correct=True,
            ground_truth_rank=1,
        ),
        NoTruthSyntheticRow(
            scene_id="synthetic_bad",
            ranker="without_failure",
            ground_truth_id="box_gt",
            top1_hypothesis_id="box_gt",
            top1_correct=True,
            ground_truth_rank=1,
        ),
    ]

    summary = summarize_no_truth_gate(
        synthetic_rows=rows,
        real_rows=[],
        no_truth_audit_pass=True,
    )

    assert summary["full_vs_silhouette_delta"] == 0.0
    assert summary["gate_status"] == "fail"


def test_m4_no_truth_gate_fails_when_live_reports_are_missing():
    rows = [
        NoTruthSyntheticRow(
            scene_id="synthetic_good",
            ranker="silhouette_only",
            ground_truth_id="box_gt",
            top1_hypothesis_id="wrong",
            top1_correct=False,
            ground_truth_rank=2,
        ),
        NoTruthSyntheticRow(
            scene_id="synthetic_good",
            ranker="full",
            ground_truth_id="box_gt",
            top1_hypothesis_id="box_gt",
            top1_correct=True,
            ground_truth_rank=1,
        ),
        NoTruthSyntheticRow(
            scene_id="synthetic_good",
            ranker="without_failure",
            ground_truth_id="box_gt",
            top1_hypothesis_id="wrong",
            top1_correct=False,
            ground_truth_rank=2,
        ),
    ]

    summary = summarize_no_truth_gate(
        synthetic_rows=rows,
        real_rows=[],
        no_truth_audit_pass=True,
    )

    assert summary["full_vs_silhouette_delta"] > 0.0
    assert summary["live_multi_target_status"] == "missing"
    assert summary["live_scenario_sweep_status"] == "missing"
    assert summary["live_dynamic_execute_status"] == "missing"
    assert summary["live_ranked_fallback_status"] == "missing"
    assert summary["gate_status"] == "fail"


def test_run_m4_no_truth_gate_script_invokes_module():
    script = Path(__file__).resolve().parents[3] / "scripts" / "run_m4_no_truth_gate.sh"

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "ghost_mgg_core_py.evaluation.m4_no_truth_gate" in text
    assert "reports/m4_no_truth_gate" in text
    assert "reports/m4_no_truth_multi_target/multi_target.json" in text
    assert "reports/m4_no_truth_scenario_sweep/scenario_sweep.json" in text
    assert "reports/m4_no_truth_live_dynamic_execute/result.json" in text
    assert "reports/m4_no_truth_live_ranked_fallback/result.json" in text
