from __future__ import annotations

import numpy as np

from ghost_mgg_core_py.evidence.types import CameraIntrinsics, TablePlane

_MIN_TRIANGLE_SIN_ANGLE = 1e-6
_MIN_RANK_TOLERANCE_M = 1e-9


def depth_to_points(depth_m: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D array")

    height, width = depth.shape
    u = np.arange(width, dtype=float)
    v = np.arange(height, dtype=float)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(float, copy=False)
    x = (uu - float(intrinsics.cx)) * z / float(intrinsics.fx)
    y = (vv - float(intrinsics.cy)) * z / float(intrinsics.fy)
    return np.stack([x, y, z], axis=-1)


def signed_distance_to_plane(points: np.ndarray, plane: TablePlane) -> np.ndarray:
    point_array = np.asarray(points, dtype=float)
    if point_array.shape[-1] != 3:
        raise ValueError("points must have last dimension of size 3")

    normalized_plane = plane.normalized()
    return point_array @ normalized_plane.normal + normalized_plane.offset


def fit_table_plane_ransac(
    points: np.ndarray,
    threshold_m: float = 0.005,
    max_iterations: int = 100,
    random_seed: int = 0,
) -> tuple[TablePlane, dict[str, float]]:
    point_array = np.asarray(points, dtype=float)
    if point_array.shape[-1] != 3:
        raise ValueError("points must have last dimension of size 3")

    point_array = point_array.reshape(-1, 3)
    finite_points = point_array[np.isfinite(point_array).all(axis=1)]
    if finite_points.shape[0] < 3:
        raise ValueError("at least 3 finite points are required to fit a plane")

    rng = np.random.default_rng(random_seed)
    best_inlier_mask: np.ndarray | None = None
    best_inlier_count = -1
    best_mean_residual = float("inf")

    for _ in range(max_iterations):
        sample = finite_points[rng.choice(finite_points.shape[0], size=3, replace=False)]
        plane = _plane_from_three_points(sample)
        if plane is None:
            continue

        distances = np.abs(signed_distance_to_plane(finite_points, plane))
        inlier_mask = distances <= threshold_m
        inlier_count = int(np.count_nonzero(inlier_mask))
        mean_residual = _mean_or_inf(distances[inlier_mask])

        if (inlier_count, -mean_residual) > (best_inlier_count, -best_mean_residual):
            best_inlier_mask = inlier_mask
            best_inlier_count = inlier_count
            best_mean_residual = mean_residual

    if best_inlier_mask is None or best_inlier_count < 3:
        raise RuntimeError("RANSAC failed to find a non-degenerate table plane")

    refined_plane = _fit_plane_svd(finite_points[best_inlier_mask])

    refined_plane = _orient_positive_z(refined_plane).normalized()
    residuals = np.abs(signed_distance_to_plane(finite_points, refined_plane))
    final_inlier_mask = residuals <= threshold_m
    final_inlier_residuals = residuals[final_inlier_mask]
    inlier_count = int(np.count_nonzero(final_inlier_mask))

    diagnostics = {
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / finite_points.shape[0]),
        "mean_abs_residual_m": float(np.mean(final_inlier_residuals))
        if final_inlier_residuals.size
        else float("nan"),
        "max_abs_residual_m": float(np.max(final_inlier_residuals))
        if final_inlier_residuals.size
        else float("nan"),
    }
    return refined_plane, diagnostics


def _plane_from_three_points(points: np.ndarray) -> TablePlane | None:
    a, b, c = points
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    norm = float(np.linalg.norm(normal))
    edge_norm = float(np.linalg.norm(ab) * np.linalg.norm(ac))
    if edge_norm <= _MIN_RANK_TOLERANCE_M or norm / edge_norm <= _MIN_TRIANGLE_SIN_ANGLE:
        return None

    normal = normal / norm
    offset = -float(np.dot(normal, a))
    return _orient_positive_z(TablePlane(normal=normal, offset=offset))


def _fit_plane_svd(points: np.ndarray) -> TablePlane:
    if points.shape[0] < 3:
        raise RuntimeError("at least three inlier points are required to fit a table plane")
    centroid = np.mean(points, axis=0)
    _, singular_values, vh = np.linalg.svd(points - centroid, full_matrices=False)
    rank_tolerance = max(_MIN_RANK_TOLERANCE_M, float(singular_values[0]) * 1e-6)
    if singular_values[1] <= rank_tolerance:
        raise RuntimeError("table plane inliers are rank deficient")
    normal = vh[-1]
    offset = -float(np.dot(normal, centroid))
    return _orient_positive_z(TablePlane(normal=normal, offset=offset))


def _orient_positive_z(plane: TablePlane) -> TablePlane:
    normal = np.asarray(plane.normal, dtype=float)
    if normal[2] < 0.0:
        return TablePlane(normal=-normal, offset=-float(plane.offset))
    return TablePlane(normal=normal, offset=float(plane.offset))


def _mean_or_inf(values: np.ndarray) -> float:
    if values.size == 0:
        return float("inf")
    return float(np.mean(values))
