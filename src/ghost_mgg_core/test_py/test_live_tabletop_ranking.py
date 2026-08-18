import numpy as np

from ghost_mgg_core_py.live_tabletop.components import extract_components
from ghost_mgg_core_py.live_tabletop.evidence import build_tabletop_evidence
from ghost_mgg_core_py.live_tabletop.ranking import (
    rank_tabletop_components,
    score_geometry_fit,
)
from ghost_mgg_core_py.live_tabletop.types import GeometryFit


def test_every_component_gets_at_least_one_hypothesis():
    candidate = np.zeros((10, 12), dtype=bool)
    candidate[1:4, 1:4] = True
    candidate[6:9, 7:10] = True
    components = extract_components(candidate, min_area_px=2)
    depth_z = np.full(candidate.shape, 0.74, dtype=float)
    depth_z[candidate] = 0.79
    evidence = build_tabletop_evidence(depth_z, None, np.ones(candidate.shape, bool), 0.74, 0.03, 0.20)

    ranked = rank_tabletop_components(
        components,
        evidence,
        points_xy_by_component={},
        points_z_by_component={},
        table_z_m=0.74,
        height_prior_m=0.04,
        pixel_size_m=0.005,
    )

    assert {item.fit.component_id for item in ranked} == {component.component_id for component in components}
    assert all(item.score > 0.0 for item in ranked)


def test_shape_scores_are_independent_for_box_cylinder_and_bbox():
    mask = np.ones((5, 5), dtype=bool)
    component = extract_components(mask)[0]
    evidence = build_tabletop_evidence(
        np.full(mask.shape, 0.80),
        None,
        np.ones(mask.shape, bool),
        0.74,
        0.03,
        0.20,
    )

    fits = [
        GeometryFit(component.component_id, "box", (0.0, 0.0), 0.03, 0.03, 0.04, 0.0, "test", 0.74, 0.76),
        GeometryFit(component.component_id, "cylinder", (0.0, 0.0), 0.03, 0.03, 0.04, 0.0, "test", 0.74, 0.76),
        GeometryFit(component.component_id, "bbox", (0.0, 0.0), 0.03, 0.03, 0.04, 0.0, "test", 0.74, 0.76),
    ]

    scored = [score_geometry_fit(fit, evidence, component, table_z_m=0.74) for fit in fits]

    assert all(item is not None for item in scored)
    scores_by_shape = {item.fit.shape_type: item.score for item in scored}
    assert set(scores_by_shape) == {"box", "cylinder", "bbox"}
    assert scores_by_shape["box"] != scores_by_shape["bbox"]


def test_transparent_failure_evidence_keeps_bbox_hypothesis_rankable():
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:8] = True
    depth_z = np.full(mask.shape, 0.74, dtype=float)
    depth_z[mask] = np.nan
    evidence = build_tabletop_evidence(depth_z, mask, np.ones(mask.shape, bool), 0.74, 0.03, 0.20)
    component = extract_components(evidence.candidate_mask, min_area_px=2)[0]

    ranked = rank_tabletop_components(
        [component],
        evidence,
        points_xy_by_component={},
        points_z_by_component={},
        table_z_m=0.74,
        height_prior_m=0.04,
        pixel_size_m=0.005,
    )

    assert len(ranked) == 1
    assert ranked[0].fit.shape_type == "bbox"
    assert ranked[0].score_terms["failure_ratio"] == 1.0
    assert ranked[0].score > 0.5


def test_residual_foreground_support_outranks_pure_depth_failure_hole():
    candidate = np.zeros((12, 18), dtype=bool)
    candidate[2:8, 2:8] = True
    candidate[3:9, 10:16] = True
    depth_z = np.full(candidate.shape, 0.74, dtype=float)
    depth_z[candidate] = np.nan
    support = np.zeros_like(candidate)
    support[3:9, 10:16] = True
    depth_z[support] = 0.78
    evidence = build_tabletop_evidence(
        depth_z,
        candidate,
        np.ones(candidate.shape, bool),
        0.74,
        0.03,
        0.20,
    )
    components = extract_components(evidence.candidate_mask, min_area_px=2)
    supported_component = next(component for component in components if component.centroid_uv[0] > 9.0)
    supported_pixels = np.argwhere(support)
    points_xy = np.column_stack(
        (
            supported_pixels[:, 1].astype(float) * 0.005,
            supported_pixels[:, 0].astype(float) * 0.005,
        )
    )
    ranked = rank_tabletop_components(
        components,
        evidence,
        points_xy_by_component={supported_component.component_id: points_xy},
        points_z_by_component={
            supported_component.component_id: np.full(points_xy.shape[0], 0.78)
        },
        table_z_m=0.74,
        height_prior_m=0.04,
        pixel_size_m=0.005,
    )

    assert ranked[0].fit.component_id == supported_component.component_id
    assert ranked[0].score_terms["support_ratio"] == 1.0
    assert ranked[0].score > ranked[1].score


def test_table_penetrating_hypotheses_are_rejected():
    mask = np.ones((4, 4), dtype=bool)
    component = extract_components(mask)[0]
    evidence = build_tabletop_evidence(
        np.full(mask.shape, 0.80),
        None,
        np.ones(mask.shape, bool),
        0.74,
        0.03,
        0.20,
    )
    penetrating = GeometryFit(
        component.component_id,
        "box",
        (0.0, 0.0),
        0.03,
        0.03,
        0.04,
        0.0,
        "test",
        0.70,
        0.72,
    )

    assert score_geometry_fit(penetrating, evidence, component, table_z_m=0.74) is None
