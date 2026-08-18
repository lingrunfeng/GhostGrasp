import numpy as np

from ghost_mgg_core_py.evidence.depth_failure import analyze_depth_failure
from ghost_mgg_core_py.evidence.types import TablePlane


def test_depth_failure_classifies_holes_leakage_and_foreground_support():
    depth = np.array(
        [
            [1.00, 1.00, 1.00, 1.00],
            [1.00, 0.00, 1.00, 0.82],
            [1.00, 1.00, 1.00, 0.80],
            [1.00, 1.00, 1.00, 1.00],
        ],
        dtype=np.float32,
    )
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:4] = True
    table = TablePlane(normal=np.array([0.0, 0.0, 1.0]), offset=-1.0)

    evidence, summary = analyze_depth_failure(
        depth_m=depth,
        target_mask=mask,
        table_plane=table,
        table_distance_sigma_m=0.015,
        foreground_min_height_m=0.05,
    )

    assert evidence.hole[1, 1] > 0.99
    assert evidence.table_leakage[1, 2] > 0.90
    assert evidence.foreground_support[1, 3] > 0.90
    assert summary.hole_ratio == 1 / 6
    assert summary.table_leakage_ratio > 0.30
    assert summary.foreground_support_ratio > 0.30


def test_depth_failure_does_not_count_outside_mask_pixels():
    depth = np.zeros((3, 3), dtype=np.float32)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    table = TablePlane(normal=np.array([0.0, 0.0, 1.0]), offset=-1.0)

    evidence, summary = analyze_depth_failure(depth, mask, table)

    assert evidence.hole[0, 0] == 0.0
    assert evidence.hole[1, 1] == 1.0
    assert summary.hole_ratio == 1.0


def test_depth_failure_foreground_support_means_in_front_of_table_depth():
    depth = np.array([[0.92, 1.00, 1.08]], dtype=np.float32)
    mask = np.ones((1, 3), dtype=bool)
    table = TablePlane(normal=np.array([0.0, 0.0, 1.0]), offset=-1.0)

    evidence, _ = analyze_depth_failure(depth, mask, table, foreground_min_height_m=0.05)

    assert evidence.foreground_support[0, 0] == 1.0
    assert evidence.foreground_support[0, 1] == 0.0
    assert evidence.foreground_support[0, 2] == 0.0


def test_depth_failure_empty_mask_returns_zero_summary():
    depth = np.ones((2, 2), dtype=np.float32)
    mask = np.zeros((2, 2), dtype=bool)
    table = TablePlane(normal=np.array([0.0, 0.0, 1.0]), offset=-1.0)

    evidence, summary = analyze_depth_failure(depth, mask, table)

    assert not evidence.valid.any()
    assert summary.valid_depth_ratio == 0.0
    assert summary.hole_ratio == 0.0
    assert summary.failure_spatial_concentration == 0.0


def test_depth_failure_rejects_invalid_table_sigma():
    depth = np.ones((1, 1), dtype=np.float32)
    mask = np.ones((1, 1), dtype=bool)
    table = TablePlane(normal=np.array([0.0, 0.0, 1.0]), offset=-1.0)

    with np.testing.assert_raises(ValueError):
        analyze_depth_failure(depth, mask, table, table_distance_sigma_m=0.0)


def test_depth_failure_rejects_tilted_plane_without_intrinsics():
    depth = np.ones((1, 1), dtype=np.float32)
    mask = np.ones((1, 1), dtype=bool)
    table = TablePlane(normal=np.array([0.1, 0.0, 1.0]), offset=-1.0)

    with np.testing.assert_raises(ValueError):
        analyze_depth_failure(depth, mask, table)
