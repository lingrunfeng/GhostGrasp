from __future__ import annotations

import numpy as np

from ghost_mgg_core_py.live_tabletop.fitters import fit_component_geometry
from ghost_mgg_core_py.live_tabletop.types import (
    GeometryFit,
    PixelComponent,
    RankedGeometryHypothesis,
    TabletopEvidence,
)

_SHAPE_PRIORS = {
    "box": 0.12,
    "cylinder": 0.10,
    "bbox": 0.08,
}


def score_geometry_fit(
    fit: GeometryFit,
    evidence: TabletopEvidence,
    component: PixelComponent,
    *,
    table_z_m: float,
    table_penetration_tolerance_m: float = 0.005,
) -> RankedGeometryHypothesis | None:
    table_z = float(table_z_m)
    if fit.bottom_z_m < table_z - float(table_penetration_tolerance_m):
        return None

    mask = np.asarray(component.mask, dtype=bool)
    area = max(int(np.count_nonzero(mask)), 1)
    support_ratio = _masked_ratio(evidence.support_mask, mask, area)
    failure_ratio = _masked_ratio(evidence.failure_mask, mask, area)
    coverage_ratio = _masked_ratio(evidence.candidate_mask, mask, area)
    leakage_ratio = _masked_ratio(evidence.table_leakage_mask, mask, area)
    shape_prior = float(_SHAPE_PRIORS.get(fit.shape_type, 0.05))

    score = (
        0.20 * coverage_ratio
        + 0.60 * support_ratio
        + 0.25 * failure_ratio
        - 0.20 * leakage_ratio
        + shape_prior
    )
    score = max(0.0, float(score))

    return RankedGeometryHypothesis(
        fit=fit,
        score=score,
        score_terms={
            "coverage_ratio": float(coverage_ratio),
            "support_ratio": float(support_ratio),
            "failure_ratio": float(failure_ratio),
            "table_leakage_ratio": float(leakage_ratio),
            "shape_prior": float(shape_prior),
        },
    )


def rank_tabletop_components(
    components: list[PixelComponent],
    evidence: TabletopEvidence,
    *,
    points_xy_by_component: dict[int, np.ndarray],
    points_z_by_component: dict[int, np.ndarray],
    table_z_m: float,
    height_prior_m: float,
    pixel_size_m: float = 0.001,
    pixel_origin_xy_m: tuple[float, float] = (0.0, 0.0),
    max_fit_points: int = 256,
) -> list[RankedGeometryHypothesis]:
    ranked: list[RankedGeometryHypothesis] = []
    for component in components:
        points_xy = points_xy_by_component.get(
            component.component_id, np.empty((0, 2), dtype=float)
        )
        points_z = points_z_by_component.get(component.component_id, np.empty((0,), dtype=float))
        points_xy, points_z = _sample_fit_points(points_xy, points_z, max_points=int(max_fit_points))
        fit = fit_component_geometry(
            component,
            points_xy_m=points_xy,
            points_z_m=points_z,
            table_z_m=table_z_m,
            height_prior_m=height_prior_m,
            pixel_size_m=pixel_size_m,
            pixel_origin_xy_m=pixel_origin_xy_m,
        )
        scored = score_geometry_fit(fit, evidence, component, table_z_m=table_z_m)
        if scored is not None:
            ranked.append(scored)

    return sorted(ranked, key=lambda item: (-item.score, item.fit.component_id))


def _sample_fit_points(
    points_xy: np.ndarray,
    points_z: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(points_xy, dtype=float)
    z = np.asarray(points_z, dtype=float)
    if xy.ndim != 2 or xy.shape[0] <= int(max_points) or int(max_points) <= 0:
        return xy, z
    indices = np.linspace(0, xy.shape[0] - 1, int(max_points), dtype=int)
    sampled_xy = xy[indices]
    if z.ndim == 1 and z.shape[0] == xy.shape[0]:
        return sampled_xy, z[indices]
    return sampled_xy, z


def _masked_ratio(values: np.ndarray, mask: np.ndarray, area: int) -> float:
    array = np.asarray(values, dtype=bool)
    if array.shape != mask.shape:
        raise ValueError("evidence masks must match component mask shape")
    return float(np.count_nonzero(array & mask) / float(area))
