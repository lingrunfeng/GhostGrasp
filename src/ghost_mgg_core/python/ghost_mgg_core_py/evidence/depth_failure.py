from __future__ import annotations

import numpy as np

from ghost_mgg_core_py.evidence.table_plane import signed_distance_to_plane
from ghost_mgg_core_py.evidence.types import EvidenceMaps, EvidenceSummary, TablePlane

_DEPTH_AXIS_TABLE_TOLERANCE = 1e-6


def evidence_from_raw_depth(
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    *,
    table_depth_m: float | None = None,
    table_z_m: float | None = None,
    points_base: np.ndarray | None = None,
    min_depth_m: float = 0.10,
    max_depth_m: float = 2.00,
    table_distance_sigma_m: float = 0.015,
    foreground_min_height_m: float = 0.035,
    flying_point_threshold_m: float = 0.12,
    edge_band_px: int = 1,
) -> tuple[EvidenceMaps, EvidenceSummary]:
    depth = np.asarray(depth_m, dtype=float)
    mask = np.asarray(target_mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError("depth_m and target_mask must have matching shapes")
    if table_depth_m is None and (table_z_m is None or points_base is None):
        raise ValueError("table_depth_m or table_z_m with points_base is required")
    min_depth = float(min_depth_m)
    max_depth = float(max_depth_m)
    if not np.isfinite(min_depth) or not np.isfinite(max_depth) or min_depth >= max_depth:
        raise ValueError("min_depth_m and max_depth_m must be finite with min < max")
    sigma = _validate_positive_finite(table_distance_sigma_m, "table_distance_sigma_m")

    valid_depth = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    valid = (mask & valid_depth).astype(float)
    hole = (mask & ~valid_depth).astype(float)
    table_leakage = np.zeros(depth.shape, dtype=float)
    foreground_support = np.zeros(depth.shape, dtype=float)
    edge = np.zeros(depth.shape, dtype=float)
    flying_point = np.zeros(depth.shape, dtype=float)

    if table_z_m is not None and points_base is not None:
        points = np.asarray(points_base, dtype=float)
        if points.shape != depth.shape + (3,):
            raise ValueError("points_base must have shape depth_m.shape + (3,)")
        signed_height = points[..., 2] - float(table_z_m)
    else:
        signed_height = float(table_depth_m) - depth

    masked_valid = mask & valid_depth & np.isfinite(signed_height)
    if np.any(masked_valid):
        table_leakage[masked_valid] = np.exp(
            -(signed_height[masked_valid] ** 2) / (2.0 * sigma**2)
        )
        foreground_support[masked_valid & (signed_height >= float(foreground_min_height_m))] = 1.0
        edge[masked_valid & _mask_boundary_band(mask, int(edge_band_px))] = 1.0
        flying_point[
            masked_valid
            & _flying_point_mask(
                depth,
                valid_depth,
                float(flying_point_threshold_m),
                candidate_mask=masked_valid,
            )
        ] = 1.0

    evidence = EvidenceMaps(
        valid=valid,
        hole=hole,
        table_leakage=table_leakage,
        edge=edge,
        flying_point=flying_point,
        foreground_support=foreground_support,
    )
    return evidence, _summarize_evidence(evidence, mask)


def analyze_depth_failure(
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    table_plane: TablePlane,
    table_distance_sigma_m: float = 0.02,
    foreground_min_height_m: float = 0.04,
) -> tuple[EvidenceMaps, EvidenceSummary]:
    depth = np.asarray(depth_m, dtype=float)
    mask = np.asarray(target_mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError("depth_m and target_mask must have matching shapes")
    sigma = _validate_positive_finite(table_distance_sigma_m, "table_distance_sigma_m")
    table_plane = _validate_depth_axis_table_plane(table_plane)

    valid_depth = np.isfinite(depth) & (depth > 0.0)
    masked_valid = mask & valid_depth

    valid = masked_valid.astype(float)
    hole = (mask & ~valid_depth).astype(float)
    table_leakage = np.zeros(depth.shape, dtype=float)
    foreground_support = np.zeros(depth.shape, dtype=float)
    edge = np.zeros(depth.shape, dtype=float)
    flying_point = np.zeros(depth.shape, dtype=float)

    if np.any(masked_valid):
        points = np.zeros(depth.shape + (3,), dtype=float)
        points[..., 2] = depth
        distances = signed_distance_to_plane(points, table_plane)

        table_leakage[masked_valid] = np.exp(
            -(distances[masked_valid] ** 2) / (2.0 * sigma**2)
        )
        foreground_support[masked_valid & (distances <= -float(foreground_min_height_m))] = 1.0

    evidence = EvidenceMaps(
        valid=valid,
        hole=hole,
        table_leakage=table_leakage,
        edge=edge,
        flying_point=flying_point,
        foreground_support=foreground_support,
    )
    summary = EvidenceSummary(
        valid_depth_ratio=_masked_mean(valid, mask),
        hole_ratio=_masked_mean(hole, mask),
        table_leakage_ratio=_masked_mean(table_leakage, mask),
        edge_support_ratio=_masked_mean(edge, mask),
        flying_point_ratio=_masked_mean(flying_point, mask),
        foreground_support_ratio=_masked_mean(foreground_support, mask),
        failure_spatial_concentration=_masked_max(hole + table_leakage, mask),
    )
    return evidence, summary


def _validate_positive_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric


def _validate_depth_axis_table_plane(table_plane: TablePlane) -> TablePlane:
    normalized = table_plane.normalized()
    normal = np.asarray(normalized.normal, dtype=float)
    if (
        abs(float(normal[0])) > _DEPTH_AXIS_TABLE_TOLERANCE
        or abs(float(normal[1])) > _DEPTH_AXIS_TABLE_TOLERANCE
        or float(normal[2]) <= 0.0
    ):
        raise ValueError(
            "analyze_depth_failure currently supports only table planes aligned with the depth axis"
        )
    return normalized


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


def _masked_max(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.max(values[mask]))


def _summarize_evidence(evidence: EvidenceMaps, mask: np.ndarray) -> EvidenceSummary:
    failure = evidence.hole + evidence.table_leakage + evidence.edge + evidence.flying_point
    return EvidenceSummary(
        valid_depth_ratio=_masked_mean(evidence.valid, mask),
        hole_ratio=_masked_mean(evidence.hole, mask),
        table_leakage_ratio=_masked_mean(evidence.table_leakage, mask),
        edge_support_ratio=_masked_mean(evidence.edge, mask),
        flying_point_ratio=_masked_mean(evidence.flying_point, mask),
        foreground_support_ratio=_masked_mean(evidence.foreground_support, mask),
        failure_spatial_concentration=_masked_max(np.clip(failure, 0.0, 1.0), mask),
    )


def _mask_boundary_band(mask: np.ndarray, band_px: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    band = max(1, int(band_px))
    padded = np.pad(mask_bool, band, mode="constant", constant_values=False)
    eroded = np.ones(mask_bool.shape, dtype=bool)
    size = 2 * band + 1
    for dy in range(size):
        for dx in range(size):
            eroded &= padded[dy : dy + mask_bool.shape[0], dx : dx + mask_bool.shape[1]]
    return mask_bool & ~eroded


def _flying_point_mask(
    depth: np.ndarray,
    valid_depth: np.ndarray,
    threshold_m: float,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    threshold = float(threshold_m)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("flying_point_threshold_m must be positive and finite")
    result = np.zeros(depth.shape, dtype=bool)
    if candidate_mask is None:
        candidates = np.asarray(valid_depth, dtype=bool)
    else:
        candidates = np.asarray(candidate_mask, dtype=bool)
        if candidates.shape != depth.shape:
            raise ValueError("candidate_mask must have the same shape as depth")
        candidates &= np.asarray(valid_depth, dtype=bool)
    rows, cols = depth.shape
    candidate_rows, candidate_cols = np.nonzero(candidates)
    for row, col in zip(candidate_rows, candidate_cols, strict=False):
        r0 = max(0, row - 1)
        r1 = min(rows, row + 2)
        c0 = max(0, col - 1)
        c1 = min(cols, col + 2)
        window_valid = valid_depth[r0:r1, c0:c1]
        window = depth[r0:r1, c0:c1][window_valid]
        if window.size < 4:
            continue
        median = float(np.median(window))
        if abs(float(depth[row, col]) - median) >= threshold:
            result[row, col] = True
    return result
