import math

import numpy as np

from ghost_mgg_core_py.live_tabletop.components import extract_components
from ghost_mgg_core_py.live_tabletop.fitters import fit_component_geometry


def _rotated_rectangle_points(width, depth, yaw_rad, center=(0.0, 0.0), steps=7):
    xs = np.linspace(-width / 2.0, width / 2.0, steps)
    ys = np.linspace(-depth / 2.0, depth / 2.0, steps)
    local = np.array([(x, y) for x in xs for y in ys], dtype=float)
    rot = np.array(
        [
            [math.cos(yaw_rad), -math.sin(yaw_rad)],
            [math.sin(yaw_rad), math.cos(yaw_rad)],
        ]
    )
    return local @ rot.T + np.asarray(center, dtype=float)


def test_oriented_box_footprint_prefers_box_with_correct_yaw():
    yaw = math.radians(30.0)
    points_xy = _rotated_rectangle_points(0.060, 0.030, yaw, center=(0.02, -0.01))
    points_z = np.full(points_xy.shape[0], 0.79)
    mask = np.ones((8, 8), dtype=bool)
    component = extract_components(mask)[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "box"
    assert np.allclose(fit.center_xy_m, (0.02, -0.01), atol=1e-4)
    assert np.allclose(sorted((fit.size_x_m, fit.size_y_m)), [0.030, 0.060], atol=2e-3)
    assert abs(abs(fit.yaw_rad) - yaw) < math.radians(2.0)
    assert np.isclose(fit.size_z_m, 0.05, atol=1e-3)


def test_circular_footprint_prefers_cylinder():
    angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    radii = np.linspace(0.0, 0.015, 5)
    points_xy = np.array(
        [(r * math.cos(a), r * math.sin(a)) for r in radii for a in angles],
        dtype=float,
    )
    points_z = np.full(points_xy.shape[0], 0.775)
    rows, cols = np.indices((101, 101))
    circle_mask = (rows - 50) ** 2 + (cols - 50) ** 2 <= 50**2
    component = extract_components(circle_mask)[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "cylinder"
    assert np.isclose(fit.size_x_m, fit.size_y_m, atol=1e-6)
    assert np.isclose(fit.size_x_m, 0.030, atol=2e-3)
    assert fit.yaw_rad == 0.0


def test_slightly_elliptical_visible_footprint_still_prefers_cylinder():
    angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    radii = np.linspace(0.0, 1.0, 5)
    points_xy = np.array(
        [(0.014 * r * math.cos(a), 0.012 * r * math.sin(a)) for r in radii for a in angles],
        dtype=float,
    )
    points_z = np.full(points_xy.shape[0], 0.765)
    rows, cols = np.indices((80, 80))
    ellipse_mask = ((cols - 40) / 15.0) ** 2 + ((rows - 40) / 12.0) ** 2 <= 1.0
    component = extract_components(ellipse_mask)[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "cylinder"
    assert np.isclose(fit.size_x_m, fit.size_y_m, atol=1e-6)


def test_square_component_mask_prevents_box_from_becoming_cylinder_when_visible_points_are_round():
    angles = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    radii = np.linspace(0.0, 0.015, 5)
    points_xy = np.array(
        [(r * math.cos(a), r * math.sin(a)) for r in radii for a in angles],
        dtype=float,
    )
    points_z = np.full(points_xy.shape[0], 0.765)
    component = extract_components(np.ones((18, 18), dtype=bool))[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "box"


def test_low_fill_square_component_mask_still_prevents_box_from_becoming_cylinder():
    angles = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    radii = np.linspace(0.0, 0.015, 5)
    points_xy = np.array(
        [(r * math.cos(a), r * math.sin(a)) for r in radii for a in angles],
        dtype=float,
    )
    points_z = np.full(points_xy.shape[0], 0.765)
    mask = np.zeros((80, 70), dtype=bool)
    mask[8:72, 8:62] = True
    mask[18:42, 16:40] = False
    mask[38:62, 30:54] = False
    component = extract_components(mask)[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "box"


def test_dense_slightly_rectangular_square_component_mask_still_prevents_box_from_becoming_cylinder():
    angles = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    radii = np.linspace(0.0, 0.015, 5)
    points_xy = np.array(
        [(r * math.cos(a), r * math.sin(a)) for r in radii for a in angles],
        dtype=float,
    )
    points_z = np.full(points_xy.shape[0], 0.765)
    mask = np.zeros((80, 70), dtype=bool)
    mask[10:70, 9:61] = True
    mask[10:14, 9:13] = False
    component = extract_components(mask)[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "box"


def test_near_square_box_preserves_footprint_yaw_and_canonicalizes_side_lengths():
    yaw = math.radians(87.0)
    points_xy = _rotated_rectangle_points(0.034, 0.033, yaw, center=(0.02, -0.01))
    points_z = np.full(points_xy.shape[0], 0.765)
    component = extract_components(np.ones((18, 18), dtype=bool))[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "box"
    assert abs(abs(fit.yaw_rad) - yaw) < math.radians(2.0)
    assert np.isclose(fit.size_x_m, fit.size_y_m, atol=1e-9)


def test_rotated_square_box_preserves_observable_footprint_yaw():
    yaw = math.radians(45.0)
    points_xy = _rotated_rectangle_points(0.030, 0.030, yaw, center=(0.02, -0.01))
    points_z = np.full(points_xy.shape[0], 0.765)
    component = extract_components(np.ones((24, 24), dtype=bool))[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "box"
    assert abs(abs(fit.yaw_rad) - yaw) < math.radians(2.0)
    assert np.isclose(fit.size_x_m, fit.size_y_m, atol=1e-9)


def test_near_cube_box_uses_visible_height_to_reject_inflated_xy_edges():
    yaw = math.radians(44.0)
    points_xy = _rotated_rectangle_points(0.038, 0.036, yaw, center=(0.02, -0.01))
    points_z = np.full(points_xy.shape[0], 0.765)
    component = extract_components(np.ones((22, 22), dtype=bool))[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "box"
    assert abs(abs(fit.yaw_rad) - yaw) < math.radians(2.0)
    assert fit.size_x_m <= 0.030
    assert fit.size_y_m <= 0.030


def test_dense_metric_points_use_measured_height_instead_of_height_prior():
    points_xy = _rotated_rectangle_points(0.030, 0.030, 0.0, steps=5)
    points_z = np.full(points_xy.shape[0], 0.765)
    component = extract_components(np.ones((8, 8), dtype=bool))[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert np.isclose(fit.size_z_m, 0.025, atol=1e-6)
    assert np.isclose(fit.center_z_m, 0.7525, atol=1e-6)


def test_irregular_footprint_uses_oriented_bbox():
    horizontal = np.array([(x, 0.0) for x in np.linspace(0.0, 0.06, 12)])
    vertical = np.array([(0.0, y) for y in np.linspace(0.0, 0.05, 10)])
    points_xy = np.vstack([horizontal, vertical])
    points_z = np.full(points_xy.shape[0], 0.81)
    l_mask = np.zeros((10, 10), dtype=bool)
    l_mask[:, 0:2] = True
    l_mask[-2:, :] = True
    component = extract_components(l_mask)[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "bbox"
    assert fit.size_x_m > 0.05
    assert fit.size_y_m > 0.04
    assert fit.size_z_m > 0.06


def test_filled_l_shape_uses_bbox_even_when_bounding_box_has_corners():
    mask = np.zeros((100, 140), dtype=bool)
    mask[36:62, 38:48] = True
    mask[54:64, 38:70] = True
    component = extract_components(mask)[0]
    rows, cols = np.nonzero(component.mask)
    points_xy = np.stack([cols.astype(float), rows.astype(float)], axis=1) * 0.002
    points_z = np.full(points_xy.shape[0], 0.80)

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "bbox"


def test_sparse_transparent_object_uses_mask_footprint_and_height_prior():
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:9, 4:12] = True
    component = extract_components(mask)[0]

    fit = fit_component_geometry(
        component,
        points_xy_m=np.empty((0, 2), dtype=float),
        points_z_m=np.empty((0,), dtype=float),
        table_z_m=0.74,
        height_prior_m=0.04,
        pixel_size_m=0.005,
        pixel_origin_xy_m=(0.0, 0.0),
    )

    assert fit.shape_type == "bbox"
    assert np.isclose(fit.size_x_m, 0.040, atol=1e-6)
    assert np.isclose(fit.size_y_m, 0.030, atol=1e-6)
    assert np.isclose(fit.size_z_m, 0.040, atol=1e-6)
    assert fit.provenance == "mask_footprint_height_prior"


def test_fit_ignores_sparse_flying_pixel_tail():
    from ghost_mgg_core_py.live_tabletop.types import PixelComponent

    points = []
    # solid 27mm circle of points (standing cylinder footprint)
    for ix in range(30):
        for iy in range(30):
            x = -0.0135 + ix * 0.0009
            y = -0.0135 + iy * 0.0009
            if x * x + y * y <= 0.0135 * 0.0135:
                points.append((x, y + 0.22))
    body_n = len(points)
    # sparse flying-pixel tail: ~4% of the points ramping 55mm into the hole
    tail_n = max(8, body_n // 25)
    for i in range(tail_n):
        points.append((0.0, 0.22 - 0.008 - 0.055 * i / tail_n))
    xy = np.asarray(points, dtype=float)
    z = np.full(xy.shape[0], 0.065)
    rows, cols = np.indices((40, 40))
    mask = (rows - 20) ** 2 + (cols - 20) ** 2 <= 15**2
    component = PixelComponent(
        component_id=1,
        area_px=int(mask.sum()),
        bbox_xyxy=(5, 5, 35, 35),
        centroid_uv=(20.0, 20.0),
        mask=mask,
    )

    fit = fit_component_geometry(
        component,
        points_xy_m=xy,
        points_z_m=z,
        table_z_m=0.0,
        height_prior_m=0.04,
    )

    assert max(fit.size_x_m, fit.size_y_m) <= 0.033  # tail must not stretch the footprint
    assert abs(fit.center_xy_m[1] - 0.22) <= 0.006
    assert fit.shape_type == "cylinder"


def test_dense_smeared_circle_mask_still_allows_cylinder():
    from ghost_mgg_core_py.live_tabletop.types import PixelComponent

    angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    radii = np.linspace(0.0, 0.015, 5)
    points_xy = np.array(
        [(r * math.cos(a), r * math.sin(a)) for r in radii for a in angles],
        dtype=float,
    )
    points_z = np.full(points_xy.shape[0], 0.775)
    # circle mask fattened by a smear skirt: bbox fill rises past 0.88 but the
    # bbox corners stay empty — must not be forced into a box
    rows, cols = np.indices((31, 31))
    mask = (rows - 15) ** 2 + (cols - 15) ** 2 <= 17**2
    mask &= np.ones((31, 31), dtype=bool)
    component = PixelComponent(
        component_id=1,
        area_px=int(mask.sum()),
        bbox_xyxy=(0, 0, 30, 30),
        centroid_uv=(15.0, 15.0),
        mask=mask,
    )
    assert mask.sum() / (31 * 31) >= 0.88  # in the dense band

    fit = fit_component_geometry(
        component,
        points_xy_m=points_xy,
        points_z_m=points_z,
        table_z_m=0.74,
        height_prior_m=0.04,
    )

    assert fit.shape_type == "cylinder"
