import numpy as np

from ghost_mgg_core_py.live_tabletop.evidence import build_tabletop_evidence


def test_valid_foreground_points_above_table_become_support():
    depth_z = np.full((5, 6), 0.74, dtype=float)
    depth_z[2:4, 2:4] = 0.81
    table_mask = np.ones(depth_z.shape, dtype=bool)

    evidence = build_tabletop_evidence(
        depth_z,
        rgb_mask=None,
        table_mask=table_mask,
        table_z_m=0.74,
        min_height_m=0.03,
        max_object_height_m=0.20,
    )

    assert int(np.count_nonzero(evidence.foreground_mask)) == 4
    assert int(np.count_nonzero(evidence.support_mask)) == 4
    assert int(np.count_nonzero(evidence.candidate_mask)) == 4
    assert int(np.count_nonzero(evidence.hole_mask)) == 0


def test_depth_holes_inside_visible_mask_are_candidate_failure_evidence():
    depth_z = np.full((5, 6), 0.74, dtype=float)
    depth_z[1:4, 2:5] = np.nan
    rgb_mask = np.zeros(depth_z.shape, dtype=bool)
    rgb_mask[1:4, 2:5] = True
    table_mask = np.ones(depth_z.shape, dtype=bool)

    evidence = build_tabletop_evidence(
        depth_z,
        rgb_mask=rgb_mask,
        table_mask=table_mask,
        table_z_m=0.74,
        min_height_m=0.03,
        max_object_height_m=0.20,
    )

    assert int(np.count_nonzero(evidence.hole_mask)) == 9
    assert int(np.count_nonzero(evidence.failure_mask)) == 9
    assert int(np.count_nonzero(evidence.candidate_mask)) == 9


def test_target_rear_voids_are_retained_as_failure_evidence():
    depth_z = np.full((6, 8), 0.74, dtype=float)
    depth_z[2:5, 2:4] = 0.82
    depth_z[2:5, 4:6] = 0.0
    rgb_mask = np.zeros(depth_z.shape, dtype=bool)
    rgb_mask[2:5, 2:6] = True
    table_mask = np.ones(depth_z.shape, dtype=bool)

    evidence = build_tabletop_evidence(
        depth_z,
        rgb_mask=rgb_mask,
        table_mask=table_mask,
        table_z_m=0.74,
        min_height_m=0.03,
        max_object_height_m=0.20,
    )

    assert int(np.count_nonzero(evidence.support_mask)) == 6
    assert int(np.count_nonzero(evidence.hole_mask)) == 6
    assert int(np.count_nonzero(evidence.failure_mask)) == 6
    assert int(np.count_nonzero(evidence.candidate_mask)) == 12


def test_table_leakage_is_not_counted_as_foreground_support():
    depth_z = np.full((5, 6), 0.74, dtype=float)
    depth_z[2, 2] = 0.82
    depth_z[2, 3] = 0.742
    rgb_mask = np.zeros(depth_z.shape, dtype=bool)
    rgb_mask[2, 2:4] = True
    table_mask = np.ones(depth_z.shape, dtype=bool)

    evidence = build_tabletop_evidence(
        depth_z,
        rgb_mask=rgb_mask,
        table_mask=table_mask,
        table_z_m=0.74,
        min_height_m=0.03,
        max_object_height_m=0.20,
    )

    assert bool(evidence.support_mask[2, 2])
    assert not bool(evidence.support_mask[2, 3])
    assert bool(evidence.table_leakage_mask[2, 3])
    assert bool(evidence.candidate_mask[2, 2])
    assert not bool(evidence.candidate_mask[2, 3])
