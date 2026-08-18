import numpy as np

from ghost_mgg_core_py.evidence.types import EvidenceMaps
from ghost_mgg_core_py.hypotheses.primitives import PrimitiveHypothesis
from ghost_mgg_core_py.rendering.proxy_renderer import render_proxy
from ghost_mgg_core_py.scoring.score_terms import (
    counterfactual_sensor_likelihood_breakdown,
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


def test_counterfactual_likelihood_rewards_candidate_covering_failure_evidence():
    mask = np.zeros((30, 30), dtype=bool)
    mask[8:22, 8:22] = True
    evidence = _empty_evidence(mask.shape)
    evidence.hole[11:17, 11:17] = 1.0
    evidence.table_leakage[11:17, 11:17] = 1.0

    aligned = PrimitiveHypothesis("aligned", "box", (13.5, 13.5), (7.0, 7.0), 1.0, 0.04)
    distractor = PrimitiveHypothesis("distractor", "box", (18.0, 18.0), (7.0, 7.0), 1.0, 0.04)

    aligned_breakdown = counterfactual_sensor_likelihood_breakdown(
        render_proxy(aligned, mask.shape),
        mask,
        evidence,
    )
    distractor_breakdown = counterfactual_sensor_likelihood_breakdown(
        render_proxy(distractor, mask.shape),
        mask,
        evidence,
    )

    assert aligned_breakdown.failure_inside > distractor_breakdown.failure_inside
    assert aligned_breakdown.leak > distractor_breakdown.leak
    assert aligned_breakdown.total > distractor_breakdown.total


def test_counterfactual_likelihood_penalizes_unexplained_failure_outside_candidate():
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:16, 4:16] = True
    evidence = _empty_evidence(mask.shape)
    evidence.hole[5:15, 5:15] = 1.0

    too_small = PrimitiveHypothesis("small", "box", (9.5, 9.5), (4.0, 4.0), 1.0, 0.04)
    breakdown = counterfactual_sensor_likelihood_breakdown(render_proxy(too_small, mask.shape), mask, evidence)

    assert breakdown.failure_outside < 0.0
    assert breakdown.total < breakdown.failure_inside


def test_counterfactual_likelihood_uses_boundary_edge_flying_and_weak_depth():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    candidate = PrimitiveHypothesis("box", "box", (9.5, 9.5), (10.0, 10.0), 1.0, 0.04)
    rendered = render_proxy(candidate, mask.shape)
    evidence = _empty_evidence(mask.shape)
    evidence.edge[rendered.boundary] = 1.0
    evidence.flying_point[rendered.boundary] = 0.5
    evidence.foreground_support[rendered.silhouette] = 0.75

    breakdown = counterfactual_sensor_likelihood_breakdown(rendered, mask, evidence)

    assert breakdown.failure_boundary > 0.0
    assert breakdown.weak_depth == 0.75
    assert breakdown.total > breakdown.failure_boundary


def test_score_hypothesis_remains_finite_for_degenerate_empty_evidence():
    mask = np.zeros((6, 6), dtype=bool)
    candidate = PrimitiveHypothesis("box", "box", (3.0, 3.0), (2.0, 2.0), 1.0, 0.04)
    evidence = _empty_evidence(mask.shape)

    score = score_hypothesis(candidate, mask, evidence)
    breakdown = counterfactual_sensor_likelihood_breakdown(render_proxy(candidate, mask.shape), mask, evidence)

    assert all(np.isfinite(value) for value in score.as_dict().values())
    assert all(np.isfinite(value) for value in breakdown.as_dict().values())
    assert set(breakdown.as_dict()) == {
        "failure_inside",
        "failure_boundary",
        "failure_outside",
        "leak",
        "weak_depth",
        "total",
    }
