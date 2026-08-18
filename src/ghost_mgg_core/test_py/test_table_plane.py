import numpy as np

from ghost_mgg_core_py.evidence.table_plane import (
    depth_to_points,
    fit_table_plane_ransac,
    signed_distance_to_plane,
)
from ghost_mgg_core_py.evidence.types import CameraIntrinsics, TablePlane


def test_depth_to_points_uses_pinhole_intrinsics():
    depth = np.array([[1.0, 2.0]], dtype=np.float32)
    intr = CameraIntrinsics(width=2, height=1, fx=2.0, fy=2.0, cx=0.0, cy=0.0)

    points = depth_to_points(depth, intr)

    assert points.shape == (1, 2, 3)
    np.testing.assert_allclose(points[0, 0], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(points[0, 1], [1.0, 0.0, 2.0], atol=1e-6)


def test_signed_distance_to_plane_is_positive_above_table():
    plane = TablePlane(normal=np.array([0.0, 0.0, 1.0]), offset=-1.0)
    points = np.array([[0.0, 0.0, 1.05], [0.0, 0.0, 0.95]])

    distances = signed_distance_to_plane(points, plane)

    np.testing.assert_allclose(distances, [0.05, -0.05], atol=1e-6)


def test_fit_table_plane_ransac_recovers_synthetic_plane_with_outliers():
    rng = np.random.default_rng(7)
    xy = rng.uniform(-0.2, 0.2, size=(200, 2))
    z = 1.0 + rng.normal(0.0, 0.001, size=(200, 1))
    inliers = np.concatenate([xy, z], axis=1)
    outliers = np.array([[0.5, 0.5, 0.7], [-0.4, 0.3, 1.4]], dtype=float)
    points = np.concatenate([inliers, outliers], axis=0)

    plane, diagnostics = fit_table_plane_ransac(
        points,
        threshold_m=0.005,
        max_iterations=80,
        random_seed=11,
    )

    assert diagnostics["inlier_count"] >= 190
    assert diagnostics["inlier_ratio"] > 0.93
    assert float(plane.normal[2]) > 0.99
    assert abs(float(plane.offset) + 1.0) < 0.01


def test_fit_table_plane_ransac_rejects_collinear_points():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    with np.testing.assert_raises(RuntimeError):
        fit_table_plane_ransac(points, threshold_m=0.005, max_iterations=20, random_seed=5)


def test_fit_table_plane_ransac_rejects_duplicate_points():
    points = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=float), repeats=4, axis=0)

    with np.testing.assert_raises(RuntimeError):
        fit_table_plane_ransac(points, threshold_m=0.005, max_iterations=20, random_seed=5)
