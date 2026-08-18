from __future__ import annotations

import numpy as np

from ghost_mgg_core_py.live_tabletop.types import TabletopEvidence


def build_tabletop_evidence(
    depth_m: np.ndarray,
    rgb_mask: np.ndarray | None,
    table_mask: np.ndarray,
    table_z_m: float,
    min_height_m: float,
    max_object_height_m: float,
) -> TabletopEvidence:
    depth = np.asarray(depth_m, dtype=float)
    table = np.asarray(table_mask, dtype=bool)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if table.shape != depth.shape:
        raise ValueError("table_mask must match depth_m shape")
    if rgb_mask is None:
        visible = np.zeros(depth.shape, dtype=bool)
    else:
        visible = np.asarray(rgb_mask, dtype=bool)
        if visible.shape != depth.shape:
            raise ValueError("rgb_mask must match depth_m shape")

    table_z = float(table_z_m)
    min_height = float(min_height_m)
    max_height = float(max_object_height_m)
    if not np.isfinite(table_z):
        raise ValueError("table_z_m must be finite")
    if not np.isfinite(min_height) or min_height < 0.0:
        raise ValueError("min_height_m must be finite and non-negative")
    if not np.isfinite(max_height) or max_height <= min_height:
        raise ValueError("max_object_height_m must be finite and greater than min_height_m")

    valid_depth = np.isfinite(depth) & (depth > 0.0)
    height = depth - table_z
    in_table = table
    plausible_height = (height >= min_height) & (height <= max_height)

    foreground = in_table & valid_depth & plausible_height
    table_leakage = in_table & valid_depth & (np.abs(height) < min_height)
    hole = in_table & visible & ~valid_depth
    failure = hole.copy()
    candidate = foreground | failure

    return TabletopEvidence(
        foreground_mask=foreground.astype(bool),
        hole_mask=hole.astype(bool),
        candidate_mask=candidate.astype(bool),
        support_mask=foreground.astype(bool),
        failure_mask=failure.astype(bool),
        table_leakage_mask=table_leakage.astype(bool),
    )
