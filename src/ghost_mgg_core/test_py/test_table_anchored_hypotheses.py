import inspect

import numpy as np

from ghost_mgg_core_py.evidence.types import EvidenceMaps
from ghost_mgg_core_py.hypotheses.hypothesis_generator import generate_table_anchored_hypotheses


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


def test_table_anchored_hypotheses_are_bottom_supported_for_boxes_and_cylinders():
    mask = np.zeros((20, 30), dtype=bool)
    mask[6:14, 10:22] = True

    hypotheses = generate_table_anchored_hypotheses(
        mask,
        _empty_evidence(mask.shape),
        footprint_center_xy_m=(0.07, 0.012),
        footprint_size_xy_m=(0.026, 0.024),
        table_z_m=0.75,
        height_priors_m=(0.025,),
        shape_types=("box", "cylinder"),
    )

    assert [hyp.shape_type for hyp in hypotheses] == ["box", "cylinder"]
    assert all(hyp.center_xy_m == (0.07, 0.012) for hyp in hypotheses)
    assert all(hyp.bottom_z_m == 0.75 for hyp in hypotheses)
    assert all(hyp.center_z_m == 0.7625 for hyp in hypotheses)
    assert all(hyp.height_m == 0.025 for hyp in hypotheses)
    assert hypotheses[0].size_xy_m == (0.026, 0.026)
    assert hypotheses[1].size_xy_m == (0.026, 0.026)


def test_table_anchored_hypotheses_keep_elongated_box_and_round_cylinder():
    mask = np.ones((10, 20), dtype=bool)

    hypotheses = generate_table_anchored_hypotheses(
        mask,
        _empty_evidence(mask.shape),
        footprint_center_xy_m=(0.0, 0.0),
        footprint_size_xy_m=(0.060, 0.025),
        table_z_m=0.75,
        height_priors_m=(0.03,),
    )

    by_id = {hyp.hypothesis_id: hyp for hyp in hypotheses}
    box = by_id["box_h030"]
    cylinder = by_id["cylinder_h030"]
    assert box.size_xy_m == (0.060, 0.025)
    assert cylinder.size_xy_m == (0.060, 0.060)
    assert box.size_px == (20.0, 10.0)
    assert cylinder.size_px == (20.0, 20.0)


def test_table_anchored_hypotheses_emit_requested_box_yaw_candidates():
    mask = np.ones((10, 20), dtype=bool)

    hypotheses = generate_table_anchored_hypotheses(
        mask,
        _empty_evidence(mask.shape),
        footprint_center_xy_m=(0.0, 0.0),
        footprint_size_xy_m=(0.060, 0.025),
        table_z_m=0.75,
        height_priors_m=(0.03,),
        shape_types=("box", "cylinder"),
        yaw_rads=(0.0, np.deg2rad(45.0)),
    )

    box_yaws = [hyp.yaw_rad for hyp in hypotheses if hyp.shape_type == "box"]
    cylinder_yaws = [hyp.yaw_rad for hyp in hypotheses if hyp.shape_type == "cylinder"]
    assert len(box_yaws) == 2
    assert any(abs(np.rad2deg(yaw) - 45.0) < 1e-6 for yaw in box_yaws)
    assert cylinder_yaws == [0.0]
    assert "box_yaw+045_h030" in {hyp.hypothesis_id for hyp in hypotheses}


def test_table_anchored_hypotheses_reject_invalid_inputs_and_sizes():
    mask = np.ones((3, 3), dtype=bool)

    with np.testing.assert_raises(ValueError):
        generate_table_anchored_hypotheses(
            mask,
            _empty_evidence(mask.shape),
            footprint_center_xy_m=(0.0, 0.0),
            footprint_size_xy_m=(0.0, 0.02),
            table_z_m=0.75,
        )

    with np.testing.assert_raises(ValueError):
        generate_table_anchored_hypotheses(
            mask,
            _empty_evidence(mask.shape),
            footprint_center_xy_m=(0.0, 0.0),
            footprint_size_xy_m=(0.02, 0.02),
            table_z_m=0.75,
            shape_types=("sphere",),
        )


def test_table_anchored_hypothesis_generator_contains_no_gazebo_truth_hooks():
    source = inspect.getsource(generate_table_anchored_hypotheses)

    forbidden = ("gz model", "model://", "red_cube", "blue_cylinder", "material_id", "worlds/")
    assert not any(token in source for token in forbidden)
