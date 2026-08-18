import numpy as np

from ghost_mgg_core_py.evidence.depth_failure import evidence_from_raw_depth, _flying_point_mask


def test_evidence_from_raw_depth_classifies_hole_leak_foreground_edge_and_flying_point():
    depth = np.full((7, 7), 1.20, dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[1:6, 1:6] = True
    depth[2, 2] = 0.0
    depth[2, 3] = np.nan
    depth[3, 3] = 1.00
    depth[3, 4] = 0.92
    depth[4, 4] = 1.34
    depth[1, 3] = 0.93

    evidence, summary = evidence_from_raw_depth(
        depth,
        mask,
        table_depth_m=1.00,
        min_depth_m=0.20,
        max_depth_m=1.50,
    )

    assert evidence.hole[2, 2] == 1.0
    assert evidence.hole[2, 3] == 1.0
    assert evidence.table_leakage[3, 3] > 0.9
    assert evidence.foreground_support[3, 4] == 1.0
    assert evidence.edge[1, 3] > 0.0
    assert evidence.flying_point[4, 4] > 0.0
    assert evidence.hole[0, 0] == 0.0
    assert summary.hole_ratio == 2 / 25
    assert summary.table_leakage_ratio > 0.0
    assert summary.edge_support_ratio > 0.0
    assert summary.flying_point_ratio > 0.0
    assert summary.foreground_support_ratio > 0.0
    assert 0.0 <= summary.failure_spatial_concentration <= 1.0


def test_flying_point_mask_can_be_limited_to_target_roi():
    depth = np.full((5, 5), 1.0, dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    depth[2, 2] = 1.3
    depth[0, 0] = 1.3
    candidate_mask = np.zeros_like(depth, dtype=bool)
    candidate_mask[2, 2] = True

    flying = _flying_point_mask(
        depth,
        valid,
        threshold_m=0.12,
        candidate_mask=candidate_mask,
    )

    assert flying[2, 2]
    assert not flying[0, 0]


def test_evidence_from_raw_depth_uses_base_points_when_table_z_is_available():
    depth = np.full((3, 3), 1.0, dtype=np.float32)
    mask = np.ones_like(depth, dtype=bool)
    points_base = np.zeros(depth.shape + (3,), dtype=np.float32)
    points_base[..., 2] = 0.75
    points_base[1, 1, 2] = 0.79

    evidence, summary = evidence_from_raw_depth(
        depth,
        mask,
        table_z_m=0.75,
        points_base=points_base,
    )

    assert evidence.table_leakage[0, 0] > 0.9
    assert evidence.foreground_support[1, 1] == 1.0
    assert summary.table_leakage_ratio > summary.foreground_support_ratio


def test_evidence_summary_as_dict_contains_finite_ratios():
    depth = np.array([[0.0, 1.0], [0.9, 1.2]], dtype=np.float32)
    mask = np.ones_like(depth, dtype=bool)

    _, summary = evidence_from_raw_depth(depth, mask, table_depth_m=1.0)
    values = summary.as_dict()

    assert set(values) == {
        "valid_depth_ratio",
        "hole_ratio",
        "table_leakage_ratio",
        "edge_support_ratio",
        "flying_point_ratio",
        "foreground_support_ratio",
        "failure_spatial_concentration",
    }
    assert all(np.isfinite(value) for value in values.values())


def test_evidence_from_raw_depth_rejects_missing_table_reference():
    depth = np.ones((2, 2), dtype=np.float32)
    mask = np.ones_like(depth, dtype=bool)

    with np.testing.assert_raises(ValueError):
        evidence_from_raw_depth(depth, mask)
