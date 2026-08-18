import numpy as np

from ghost_mgg_core_py.hypotheses.primitives import PrimitiveHypothesis
from ghost_mgg_core_py.hypotheses.hypothesis_generator import generate_local_hypotheses
from ghost_mgg_core_py.rendering.proxy_renderer import render_proxy


def test_box_proxy_renderer_returns_silhouette_and_boundary():
    hypothesis = PrimitiveHypothesis(
        hypothesis_id="box_1",
        shape_type="box",
        center_uv=(10.0, 10.0),
        size_px=(8.0, 6.0),
        depth_m=1.0,
        height_m=0.08,
        prior_score=0.5,
    )

    rendered = render_proxy(hypothesis, image_shape=(24, 24), boundary_width_px=1)

    assert rendered.silhouette.dtype == bool
    assert rendered.silhouette.shape == (24, 24)
    assert 35 <= int(rendered.silhouette.sum()) <= 63
    assert int(rendered.boundary.sum()) > 0
    assert rendered.expected_depth.shape == (24, 24)
    assert np.isfinite(rendered.expected_depth[rendered.silhouette]).all()


def test_cylinder_proxy_renderer_is_ellipse_like():
    hypothesis = PrimitiveHypothesis(
        hypothesis_id="cyl_1",
        shape_type="cylinder",
        center_uv=(12.0, 12.0),
        size_px=(10.0, 10.0),
        depth_m=1.0,
        height_m=0.10,
        prior_score=0.5,
    )

    rendered = render_proxy(hypothesis, image_shape=(30, 30), boundary_width_px=2)

    area = int(rendered.silhouette.sum())
    assert 60 <= area <= 95
    assert rendered.silhouette[12, 12]
    assert not rendered.silhouette[0, 0]


def test_box_proxy_renderer_supports_yawed_rectangles():
    yawed = PrimitiveHypothesis(
        hypothesis_id="box_yaw45",
        shape_type="box",
        center_uv=(20.0, 20.0),
        size_px=(18.0, 8.0),
        depth_m=1.0,
        height_m=0.08,
        yaw_rad=np.pi / 4.0,
    )
    axis_aligned = PrimitiveHypothesis(
        hypothesis_id="box_axis",
        shape_type="box",
        center_uv=(20.0, 20.0),
        size_px=(18.0, 8.0),
        depth_m=1.0,
        height_m=0.08,
    )

    yawed_rendered = render_proxy(yawed, image_shape=(42, 42), boundary_width_px=1)
    axis_rendered = render_proxy(axis_aligned, image_shape=(42, 42), boundary_width_px=1)

    assert yawed_rendered.silhouette[14, 14]
    assert yawed_rendered.silhouette[26, 26]
    assert not axis_rendered.silhouette[14, 14]
    assert not axis_rendered.silhouette[26, 26]


def test_generate_local_hypotheses_returns_stable_box_and_cylinder_ids():
    mask = np.zeros((20, 30), dtype=bool)
    mask[6:14, 10:22] = True

    hypotheses = generate_local_hypotheses(
        mask,
        shape_types=("box", "cylinder"),
        scale_factors=(0.8, 1.0),
        depth_m=1.0,
        height_m=0.08,
    )

    ids = [hyp.hypothesis_id for hyp in hypotheses]
    assert ids == ["box_s0.80", "box_s1.00", "cylinder_s0.80", "cylinder_s1.00"]
    assert all(hyp.center_uv == (15.5, 9.5) for hyp in hypotheses)


def test_generate_local_hypotheses_returns_empty_for_empty_mask():
    mask = np.zeros((4, 4), dtype=bool)

    assert generate_local_hypotheses(mask) == []


def test_generate_local_hypotheses_rejects_non_positive_scale():
    mask = np.ones((4, 4), dtype=bool)

    with np.testing.assert_raises(ValueError):
        generate_local_hypotheses(mask, scale_factors=(1.0, 0.0))


def test_generate_local_hypotheses_rejects_unsupported_shape_type():
    mask = np.ones((4, 4), dtype=bool)

    with np.testing.assert_raises(ValueError):
        generate_local_hypotheses(mask, shape_types=("box", "sphere"))


def test_render_proxy_rejects_unsupported_shape():
    hypothesis = PrimitiveHypothesis(
        hypothesis_id="bad_shape",
        shape_type="sphere",
        center_uv=(2.0, 2.0),
        size_px=(2.0, 2.0),
        depth_m=1.0,
        height_m=0.1,
    )

    with np.testing.assert_raises(ValueError):
        render_proxy(hypothesis, image_shape=(8, 8))


def test_render_proxy_rejects_invalid_geometry_and_depth():
    base = dict(
        hypothesis_id="bad",
        shape_type="box",
        center_uv=(2.0, 2.0),
        size_px=(2.0, 2.0),
        depth_m=1.0,
        height_m=0.1,
    )

    invalid_cases = [
        {"size_px": (0.0, 2.0)},
        {"size_px": (-1.0, 2.0)},
        {"depth_m": np.nan},
        {"depth_m": 0.0},
        {"height_m": 0.0},
        {"height_m": 1.2},
    ]
    for override in invalid_cases:
        hypothesis = PrimitiveHypothesis(**(base | override))
        with np.testing.assert_raises(ValueError):
            render_proxy(hypothesis, image_shape=(8, 8))


def test_render_proxy_rejects_invalid_image_shape_and_boundary_width():
    hypothesis = PrimitiveHypothesis(
        hypothesis_id="box_1",
        shape_type="box",
        center_uv=(2.0, 2.0),
        size_px=(2.0, 2.0),
        depth_m=1.0,
        height_m=0.1,
    )

    with np.testing.assert_raises(ValueError):
        render_proxy(hypothesis, image_shape=(8,))
    with np.testing.assert_raises(ValueError):
        render_proxy(hypothesis, image_shape=(8, 0))
    with np.testing.assert_raises(ValueError):
        render_proxy(hypothesis, image_shape=(8, 8), boundary_width_px=0)
