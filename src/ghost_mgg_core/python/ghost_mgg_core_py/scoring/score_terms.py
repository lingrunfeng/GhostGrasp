from dataclasses import dataclass

import numpy as np

from ghost_mgg_core_py.rendering.proxy_renderer import render_proxy


DEFAULT_WEIGHTS = {
    "visual": 1.0,
    "failure": 4.0,
    "depth": 0.5,
    "physical": 0.0,
    "grasp": 0.0,
    "prior": 1.0,
}


@dataclass(frozen=True)
class FailureLikelihoodBreakdown:
    inside_hole: float
    inside_table_leakage: float
    boundary_edge: float
    boundary_flying_point: float
    outside_hole_penalty: float
    outside_table_leakage_penalty: float
    total: float

    def as_dict(self):
        return {
            "inside_hole": self.inside_hole,
            "inside_table_leakage": self.inside_table_leakage,
            "boundary_edge": self.boundary_edge,
            "boundary_flying_point": self.boundary_flying_point,
            "outside_hole_penalty": self.outside_hole_penalty,
            "outside_table_leakage_penalty": self.outside_table_leakage_penalty,
            "total": self.total,
        }


@dataclass(frozen=True)
class CounterfactualSensorLikelihoodBreakdown:
    failure_inside: float
    failure_boundary: float
    failure_outside: float
    leak: float
    weak_depth: float
    total: float

    def as_dict(self):
        return {
            "failure_inside": self.failure_inside,
            "failure_boundary": self.failure_boundary,
            "failure_outside": self.failure_outside,
            "leak": self.leak,
            "weak_depth": self.weak_depth,
            "total": self.total,
        }


@dataclass(frozen=True)
class ScoreBreakdown:
    visual: float
    failure: float
    depth: float
    physical: float
    grasp: float
    prior: float
    total: float

    def as_dict(self):
        return {
            "visual": self.visual,
            "failure": self.failure,
            "depth": self.depth,
            "physical": self.physical,
            "grasp": self.grasp,
            "prior": self.prior,
            "total": self.total,
        }


def silhouette_iou(a, b):
    a_mask = np.asarray(a, dtype=bool)
    b_mask = np.asarray(b, dtype=bool)
    union = np.logical_or(a_mask, b_mask)
    union_count = int(union.sum())
    if union_count == 0:
        return 1.0
    intersection = np.logical_and(a_mask, b_mask)
    return float(intersection.sum()) / float(union_count)


def failure_likelihood(rendered, target_mask, evidence):
    return failure_likelihood_breakdown(rendered, target_mask, evidence).total


def counterfactual_sensor_likelihood_breakdown(rendered, target_mask, evidence):
    silhouette = np.asarray(rendered.silhouette, dtype=bool)
    boundary = np.asarray(rendered.boundary, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    evidence_arrays = _validated_evidence_arrays(evidence, target.shape)
    _validate_mask_shape(silhouette, target.shape, "rendered.silhouette")
    _validate_mask_shape(boundary, target.shape, "rendered.boundary")

    outside = target & ~silhouette
    failure_inside = _masked_mean(evidence_arrays["hole"], silhouette)
    leak = _masked_mean(evidence_arrays["table_leakage"], silhouette)
    failure_boundary = _masked_mean(evidence_arrays["edge"], boundary) + _masked_mean(
        evidence_arrays["flying_point"], boundary
    )
    failure_outside = -(
        _masked_mean(evidence_arrays["hole"], outside)
        + _masked_mean(evidence_arrays["table_leakage"], outside)
    )
    weak_depth = _masked_mean(evidence_arrays["foreground_support"], silhouette)
    total = failure_inside + leak + failure_boundary + failure_outside + weak_depth
    return CounterfactualSensorLikelihoodBreakdown(
        failure_inside=float(failure_inside),
        failure_boundary=float(failure_boundary),
        failure_outside=float(failure_outside),
        leak=float(leak),
        weak_depth=float(weak_depth),
        total=float(total),
    )


def failure_likelihood_breakdown(rendered, target_mask, evidence):
    silhouette = np.asarray(rendered.silhouette, dtype=bool)
    boundary = np.asarray(rendered.boundary, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    evidence_arrays = _validated_evidence_arrays(evidence, target.shape)
    _validate_mask_shape(silhouette, target.shape, "rendered.silhouette")
    _validate_mask_shape(boundary, target.shape, "rendered.boundary")

    outside = target & ~silhouette
    inside_hole = _masked_mean(evidence_arrays["hole"], silhouette)
    inside_table_leakage = _masked_mean(evidence_arrays["table_leakage"], silhouette)
    boundary_edge = _masked_mean(evidence_arrays["edge"], boundary)
    boundary_flying_point = _masked_mean(evidence_arrays["flying_point"], boundary)
    outside_hole_penalty = _masked_mean(evidence_arrays["hole"], outside)
    outside_table_leakage_penalty = _masked_mean(evidence_arrays["table_leakage"], outside)
    total = (
        inside_hole
        + inside_table_leakage
        + boundary_edge
        + boundary_flying_point
        - outside_hole_penalty
        - outside_table_leakage_penalty
    )

    return FailureLikelihoodBreakdown(
        inside_hole=float(inside_hole),
        inside_table_leakage=float(inside_table_leakage),
        boundary_edge=float(boundary_edge),
        boundary_flying_point=float(boundary_flying_point),
        outside_hole_penalty=float(outside_hole_penalty),
        outside_table_leakage_penalty=float(outside_table_leakage_penalty),
        total=float(total),
    )


def score_hypothesis(hypothesis, target_mask, evidence, weights=None):
    target = np.asarray(target_mask, dtype=bool)
    rendered = render_proxy(hypothesis, target.shape)
    evidence_arrays = _validated_evidence_arrays(evidence, target.shape)

    visual = silhouette_iou(rendered.silhouette, target)
    failure = failure_likelihood(rendered, target, evidence)
    depth = _masked_mean(evidence_arrays["foreground_support"], rendered.silhouette)
    physical = 0.0
    grasp = 0.0
    prior = _validate_finite_scalar(hypothesis.prior_score, "prior_score")

    resolved_weights = _resolve_weights(weights)
    total = (
        resolved_weights["visual"] * visual
        + resolved_weights["failure"] * failure
        + resolved_weights["depth"] * depth
        + resolved_weights["physical"] * physical
        + resolved_weights["grasp"] * grasp
        + resolved_weights["prior"] * prior
    )

    return ScoreBreakdown(
        visual=float(visual),
        failure=float(failure),
        depth=float(depth),
        physical=physical,
        grasp=grasp,
        prior=prior,
        total=float(total),
    )


def _masked_mean(values, mask):
    selected = np.asarray(mask, dtype=bool)
    value_array = np.asarray(values, dtype=float)
    _validate_array_finite(value_array, "values")
    _validate_mask_shape(selected, value_array.shape, "mask")
    if not selected.any():
        return 0.0
    return float(value_array[selected].mean())


def _validated_evidence_arrays(evidence, shape):
    arrays = {}
    for name in (
        "valid",
        "hole",
        "table_leakage",
        "edge",
        "flying_point",
        "foreground_support",
    ):
        array = np.asarray(getattr(evidence, name), dtype=float)
        if array.shape != shape:
            raise ValueError(f"evidence.{name} shape {array.shape} does not match target shape {shape}")
        _validate_array_finite(array, f"evidence.{name}")
        arrays[name] = array
    return arrays


def _resolve_weights(weights):
    if weights is None:
        return DEFAULT_WEIGHTS.copy()

    provided = dict(weights)
    unknown = set(provided) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown score weight keys: {sorted(unknown)}")

    resolved = DEFAULT_WEIGHTS | provided
    for name, value in resolved.items():
        resolved[name] = _validate_finite_scalar(value, f"weight {name}")
    return resolved


def _validate_mask_shape(mask, shape, name):
    if np.asarray(mask).shape != shape:
        raise ValueError(f"{name} shape {np.asarray(mask).shape} does not match expected shape {shape}")


def _validate_array_finite(array, name):
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")


def _validate_finite_scalar(value, name):
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric
