import json
import math

from ghost_mgg_core_py.evaluation.m8_live_tabletop_eval import (
    M8HypothesisGeometry,
    M8LiveHypothesisSnapshot,
    M8MovedObjectSnapshot,
    M8TruthGeometry,
    evaluate_dynamic_tabletop_update,
    evaluate_strict_geometry_snapshot,
    summarize_dynamic_rows,
    write_dynamic_eval_report,
)


def test_evaluator_matches_moved_object_before_and_after_pose():
    before_object = M8MovedObjectSnapshot(
        object_id="red_cube",
        shape_type="box",
        center_xy_m=(0.02, 0.18),
        size_xy_m=(0.04, 0.04),
        yaw_rad=0.1,
    )
    after_object = M8MovedObjectSnapshot(
        object_id="red_cube",
        shape_type="box",
        center_xy_m=(0.10, 0.22),
        size_xy_m=(0.04, 0.04),
        yaw_rad=0.2,
    )
    before_hypothesis = _hypothesis("component_1_box", (0.021, 0.179), (0.041, 0.039), 0.11)
    after_hypothesis = _hypothesis("component_1_box", (0.101, 0.219), (0.041, 0.040), 0.19)

    row = evaluate_dynamic_tabletop_update(
        before_object=before_object,
        after_object=after_object,
        before_hypotheses=[before_hypothesis],
        after_hypotheses=[after_hypothesis],
        update_latency_sec=0.42,
    )

    assert row.object_id == "red_cube"
    assert row.status == "pass"
    assert row.matched_after_hypothesis_id == "component_1_box"
    assert row.after_center_error_m < 0.003
    assert row.update_latency_sec == 0.42


def test_evaluator_reports_size_yaw_drift_and_hypothesis_count():
    row = evaluate_dynamic_tabletop_update(
        before_object=M8MovedObjectSnapshot(
            object_id="yawed_box",
            shape_type="box",
            center_xy_m=(0.0, 0.0),
            size_xy_m=(0.05, 0.03),
            yaw_rad=0.7,
        ),
        after_object=M8MovedObjectSnapshot(
            object_id="yawed_box",
            shape_type="box",
            center_xy_m=(0.04, 0.01),
            size_xy_m=(0.05, 0.03),
            yaw_rad=0.7,
        ),
        before_hypotheses=[_hypothesis("before", (0.0, 0.0), (0.050, 0.030), 0.7)],
        after_hypotheses=[
            _hypothesis("after_a", (0.041, 0.011), (0.052, 0.031), 0.68),
            _hypothesis("after_b", (0.20, 0.20), (0.020, 0.020), 0.0),
        ],
        update_latency_sec=0.30,
    )

    assert row.before_hypothesis_count == 1
    assert row.after_hypothesis_count == 2
    assert row.size_drift_m < 0.004
    assert row.yaw_drift_rad < math.radians(3.0)


def test_evaluator_fails_no_truth_audit_when_provenance_mentions_truth():
    row = evaluate_dynamic_tabletop_update(
        before_object=M8MovedObjectSnapshot(
            object_id="red_cube",
            shape_type="box",
            center_xy_m=(0.0, 0.0),
            size_xy_m=(0.04, 0.04),
            yaw_rad=0.0,
        ),
        after_object=M8MovedObjectSnapshot(
            object_id="red_cube",
            shape_type="box",
            center_xy_m=(0.05, 0.0),
            size_xy_m=(0.04, 0.04),
            yaw_rad=0.0,
        ),
        before_hypotheses=[_hypothesis("before", (0.0, 0.0), (0.04, 0.04), 0.0)],
        after_hypotheses=[
            _hypothesis(
                "after",
                (0.05, 0.0),
                (0.04, 0.04),
                0.0,
                provenance="model_pose=gazebo_truth",
            )
        ],
        update_latency_sec=0.10,
    )

    assert row.no_truth_audit_pass is False
    assert row.status == "fail"


def test_writes_dynamic_eval_report(tmp_path):
    rows = [
        evaluate_dynamic_tabletop_update(
            before_object=M8MovedObjectSnapshot(
                object_id="red_cube",
                shape_type="box",
                center_xy_m=(0.0, 0.0),
                size_xy_m=(0.04, 0.04),
                yaw_rad=0.0,
            ),
            after_object=M8MovedObjectSnapshot(
                object_id="red_cube",
                shape_type="box",
                center_xy_m=(0.05, 0.0),
                size_xy_m=(0.04, 0.04),
                yaw_rad=0.0,
            ),
            before_hypotheses=[_hypothesis("before", (0.0, 0.0), (0.04, 0.04), 0.0)],
            after_hypotheses=[_hypothesis("after", (0.05, 0.0), (0.04, 0.04), 0.0)],
            update_latency_sec=0.10,
        )
    ]

    summary = summarize_dynamic_rows(rows)
    write_dynamic_eval_report(rows, tmp_path)

    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "m8_live_tabletop_headless_eval_v1"
    assert payload["summary"]["gate_status"] == summary["gate_status"] == "pass"
    assert (tmp_path / "index.md").exists()


def test_strict_geometry_eval_accepts_square_yaw_modulo_quarter_turn():
    rows = evaluate_strict_geometry_snapshot(
        truths=[
            M8TruthGeometry(
                object_id="red_cube",
                shape_type="box",
                center_xy_m=(0.035, 0.060),
                size_xy_m=(0.025, 0.025),
                yaw_rad=math.radians(45.0),
            )
        ],
        hypotheses=[
            M8HypothesisGeometry(
                hypothesis_id="component_2_bbox_r1",
                shape_type="box",
                center_xy_m=(0.037, 0.058),
                size_xy_m=(0.024, 0.026),
                yaw_rad=math.radians(-45.0),
                provenance="m4_live_no_truth_ghost_mgg_v1",
            )
        ],
    )

    assert rows[0].status == "pass"
    assert rows[0].center_error_m < 0.004
    assert rows[0].yaw_error_rad < math.radians(1.0)


def test_strict_geometry_eval_ignores_unobservable_square_yaw():
    rows = evaluate_strict_geometry_snapshot(
        truths=[
            M8TruthGeometry(
                object_id="glass_block",
                shape_type="box",
                center_xy_m=(0.0, 0.10),
                size_xy_m=(0.025, 0.025),
                yaw_rad=0.0,
            )
        ],
        hypotheses=[
            M8HypothesisGeometry(
                hypothesis_id="component_3_bbox_r1",
                shape_type="box",
                center_xy_m=(0.002, 0.102),
                size_xy_m=(0.026, 0.026),
                yaw_rad=math.radians(-28.5),
                provenance="m4_live_no_truth_ghost_mgg_v1",
            )
        ],
    )

    assert rows[0].status == "pass"
    assert rows[0].yaw_error_rad == 0.0


def test_strict_geometry_eval_flags_wrong_shape_and_truth_leakage():
    rows = evaluate_strict_geometry_snapshot(
        truths=[
            M8TruthGeometry(
                object_id="blue_cylinder",
                shape_type="cylinder",
                center_xy_m=(0.10, 0.07),
                size_xy_m=(0.025, 0.025),
                yaw_rad=0.0,
            )
        ],
        hypotheses=[
            M8HypothesisGeometry(
                hypothesis_id="component_3_bbox_r1",
                shape_type="box",
                center_xy_m=(0.101, 0.071),
                size_xy_m=(0.025, 0.025),
                yaw_rad=0.0,
                provenance="model_pose=gazebo_truth",
            )
        ],
    )

    assert rows[0].shape_match is False
    assert rows[0].no_truth_audit_pass is False
    assert rows[0].status == "fail"


def test_strict_geometry_eval_prefers_shape_matching_top_k_candidate():
    rows = evaluate_strict_geometry_snapshot(
        truths=[
            M8TruthGeometry(
                object_id="m8_tall_cylinder",
                shape_type="cylinder",
                center_xy_m=(0.020, 0.020),
                size_xy_m=(0.024, 0.024),
                yaw_rad=0.0,
            )
        ],
        hypotheses=[
            M8HypothesisGeometry(
                hypothesis_id="box_s0.90",
                shape_type="box",
                center_xy_m=(0.021, 0.020),
                size_xy_m=(0.024, 0.024),
                yaw_rad=math.radians(45.0),
                provenance="m4_live_no_truth_ghost_mgg_v1",
            ),
            M8HypothesisGeometry(
                hypothesis_id="cylinder_s1.10",
                shape_type="cylinder",
                center_xy_m=(0.022, 0.020),
                size_xy_m=(0.025, 0.025),
                yaw_rad=0.0,
                provenance="m4_live_no_truth_ghost_mgg_v1",
            ),
        ],
    )

    assert rows[0].status == "pass"
    assert rows[0].matched_hypothesis_id == "cylinder_s1.10"
    assert rows[0].shape_match is True


def _hypothesis(
    hypothesis_id,
    center_xy_m,
    size_xy_m,
    yaw_rad,
    provenance="m4_live_no_truth_live_tabletop_core",
):
    return M8LiveHypothesisSnapshot(
        hypothesis_id=hypothesis_id,
        component_id=1,
        shape_type="box",
        center_xy_m=center_xy_m,
        size_xy_m=size_xy_m,
        yaw_rad=yaw_rad,
        provenance=provenance,
    )
