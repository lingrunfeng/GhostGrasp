import numpy as np
from types import SimpleNamespace

from ghost_mgg_core_py.evidence.types import EvidenceMaps
from ghost_mgg_core_py.hypotheses.primitives import PrimitiveHypothesis
from ghost_mgg_core_py.scoring.joint_ranker import rank_hypotheses
from ghost_mgg_core_py.scoring.score_terms import (
    failure_likelihood,
    failure_likelihood_breakdown,
    silhouette_iou,
    score_hypothesis,
)


def _empty_evidence(shape):
    zeros = np.zeros(shape, dtype=float)
    return EvidenceMaps(
        valid=zeros.copy(),
        hole=zeros.copy(),
        table_leakage=zeros.copy(),
        edge=zeros.copy(),
        flying_point=zeros.copy(),
        foreground_support=zeros.copy(),
    )


def test_silhouette_iou_matches_expected_overlap():
    a = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
    b = np.array([[1, 0, 0], [1, 0, 0]], dtype=bool)

    assert silhouette_iou(a, b) == 1 / 3


def test_silhouette_iou_returns_one_for_two_empty_masks():
    empty = np.zeros((2, 2), dtype=bool)

    assert silhouette_iou(empty, empty) == 1.0


def test_joint_ranker_prefers_failure_aligned_hypothesis_over_prior():
    mask = np.zeros((20, 20), dtype=bool)
    mask[6:14, 6:14] = True
    evidence = _empty_evidence(mask.shape)
    evidence.table_leakage[8:12, 8:12] = 1.0
    evidence.hole[8:12, 8:12] = 1.0

    correct = PrimitiveHypothesis(
        hypothesis_id="correct",
        shape_type="box",
        center_uv=(9.5, 9.5),
        size_px=(4.0, 4.0),
        depth_m=1.0,
        height_m=0.1,
        prior_score=0.0,
    )
    distractor = PrimitiveHypothesis(
        hypothesis_id="distractor",
        shape_type="box",
        center_uv=(9.5, 9.5),
        size_px=(8.0, 8.0),
        depth_m=1.0,
        height_m=0.1,
        prior_score=0.10,
    )

    ranked = rank_hypotheses([distractor, correct], mask, evidence, min_total_score=0.0)

    assert ranked[0].hypothesis.hypothesis_id == "correct"
    assert ranked[0].score.failure > ranked[1].score.failure
    assert set(ranked[0].score.as_dict()) == {
        "visual",
        "failure",
        "depth",
        "physical",
        "grasp",
        "prior",
        "total",
    }


def test_ranker_rejects_low_confidence_candidates():
    mask = np.ones((10, 10), dtype=bool)
    candidate = PrimitiveHypothesis(
        hypothesis_id="candidate",
        shape_type="box",
        center_uv=(5.0, 5.0),
        size_px=(2.0, 2.0),
        depth_m=1.0,
        height_m=0.1,
    )

    ranked = rank_hypotheses([candidate], mask, _empty_evidence(mask.shape), min_total_score=10.0)

    assert ranked[0].validation_state == "rejected"


def test_failure_likelihood_penalizes_only_target_outside_region():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    evidence = _empty_evidence(mask.shape)
    evidence.hole[0, 0] = 10.0
    evidence.hole[1, 1] = 1.0
    candidate = PrimitiveHypothesis(
        hypothesis_id="candidate",
        shape_type="box",
        center_uv=(2.5, 2.5),
        size_px=(2.0, 2.0),
        depth_m=1.0,
        height_m=0.1,
    )

    score = score_hypothesis(candidate, mask, evidence, weights={"visual": 0.0, "failure": 1.0})

    assert score.failure < 0.0


def test_failure_likelihood_breakdown_exposes_each_term_and_total():
    target = np.ones((4, 4), dtype=bool)
    silhouette = np.zeros((4, 4), dtype=bool)
    silhouette[0:2, 0:2] = True
    boundary = np.zeros((4, 4), dtype=bool)
    boundary[0, 0] = True
    boundary[0, 1] = True
    rendered = SimpleNamespace(silhouette=silhouette, boundary=boundary)
    evidence = _empty_evidence(target.shape)
    evidence.hole[0, 0] = 4.0
    evidence.table_leakage[1, 1] = 2.0
    evidence.edge[0, 0] = 3.0
    evidence.edge[0, 1] = 5.0
    evidence.flying_point[0, 0] = 1.0
    evidence.flying_point[0, 1] = 3.0
    evidence.hole[3, 3] = 6.0
    evidence.table_leakage[3, 2] = 12.0

    breakdown = failure_likelihood_breakdown(rendered, target, evidence)

    assert breakdown.inside_hole == 1.0
    assert breakdown.inside_table_leakage == 0.5
    assert breakdown.boundary_edge == 4.0
    assert breakdown.boundary_flying_point == 2.0
    assert breakdown.outside_hole_penalty == 0.5
    assert breakdown.outside_table_leakage_penalty == 1.0
    assert breakdown.total == 6.0
    assert failure_likelihood(rendered, target, evidence) == breakdown.total
    assert breakdown.as_dict() == {
        "inside_hole": 1.0,
        "inside_table_leakage": 0.5,
        "boundary_edge": 4.0,
        "boundary_flying_point": 2.0,
        "outside_hole_penalty": 0.5,
        "outside_table_leakage_penalty": 1.0,
        "total": 6.0,
    }


def test_score_hypothesis_rejects_mismatched_evidence_shape():
    mask = np.ones((6, 6), dtype=bool)
    evidence = _empty_evidence((5, 5))
    candidate = PrimitiveHypothesis(
        hypothesis_id="candidate",
        shape_type="box",
        center_uv=(2.5, 2.5),
        size_px=(2.0, 2.0),
        depth_m=1.0,
        height_m=0.1,
    )

    with np.testing.assert_raises(ValueError):
        score_hypothesis(candidate, mask, evidence)


def test_score_hypothesis_rejects_nonfinite_evidence_prior_and_weights():
    mask = np.ones((6, 6), dtype=bool)
    candidate = PrimitiveHypothesis(
        hypothesis_id="candidate",
        shape_type="box",
        center_uv=(2.5, 2.5),
        size_px=(2.0, 2.0),
        depth_m=1.0,
        height_m=0.1,
    )
    evidence = _empty_evidence(mask.shape)
    evidence.hole[2, 2] = np.nan

    with np.testing.assert_raises(ValueError):
        score_hypothesis(candidate, mask, evidence)

    nonfinite_prior = PrimitiveHypothesis(
        hypothesis_id="candidate",
        shape_type="box",
        center_uv=(2.5, 2.5),
        size_px=(2.0, 2.0),
        depth_m=1.0,
        height_m=0.1,
        prior_score=np.nan,
    )
    with np.testing.assert_raises(ValueError):
        score_hypothesis(nonfinite_prior, mask, _empty_evidence(mask.shape))

    with np.testing.assert_raises(ValueError):
        score_hypothesis(candidate, mask, _empty_evidence(mask.shape), weights={"failure": np.nan})


def test_score_hypothesis_rejects_unknown_weight_keys():
    mask = np.ones((6, 6), dtype=bool)
    candidate = PrimitiveHypothesis(
        hypothesis_id="candidate",
        shape_type="box",
        center_uv=(2.5, 2.5),
        size_px=(2.0, 2.0),
        depth_m=1.0,
        height_m=0.1,
    )

    with np.testing.assert_raises(ValueError):
        score_hypothesis(candidate, mask, _empty_evidence(mask.shape), weights={"failuer": 1.0})
