from __future__ import annotations

import math

import numpy as np

from ghost_mgg_core_py.live_tabletop.types import GeometryFit, PixelComponent

_CORNER_THRESHOLD = 0.38


def fit_component_geometry(
    component: PixelComponent,
    *,
    points_xy_m: np.ndarray,
    points_z_m: np.ndarray,
    table_z_m: float,
    height_prior_m: float,
    pixel_size_m: float = 0.001,
    pixel_origin_xy_m: tuple[float, float] = (0.0, 0.0),
) -> GeometryFit:
    xy = np.asarray(points_xy_m, dtype=float)
    z = np.asarray(points_z_m, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("points_xy_m must have shape (N, 2)")
    if z.ndim != 1 or z.shape[0] != xy.shape[0]:
        raise ValueError("points_z_m must have shape (N,)")
    table_z = _finite_scalar(table_z_m, "table_z_m")
    height_prior = _positive_scalar(height_prior_m, "height_prior_m")

    valid_points = np.all(np.isfinite(xy), axis=1) & np.isfinite(z)
    if int(np.count_nonzero(valid_points)) >= 4:
        return _fit_from_metric_points(
            component,
            xy[valid_points],
            z[valid_points],
            table_z,
            height_prior,
        )

    return _fit_from_mask_footprint(
        component, height_prior, pixel_size_m, pixel_origin_xy_m, table_z
    )


def _robust_local_interval(
    values: np.ndarray,
    *,
    bin_width_m: float = 0.004,
    min_density_ratio: float = 0.05,
    max_tail_mass_fraction: float = 0.08,
) -> tuple[float, float]:
    """Interval after shaving sparse TERMINAL tails off the point distribution.

    Depth-edge flying pixels form sparse tails (a few percent of the points
    smeared over tens of millimetres, typically into a stereo-shadow hole).
    A plain min/max footprint stretches over such a tail. Only terminal bins
    below a small fraction of the peak density are dropped, and never more
    than a small share of the total mass — interior low-density gaps (an IR
    dropout stripe crossing one object, or the gap of a genuinely merged
    two-object component) are left for the component-level logic to handle.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span = hi - lo
    if span <= 4.0 * float(bin_width_m):
        return lo, hi
    bins = max(8, min(96, int(math.ceil(span / float(bin_width_m)))))
    counts, edges = np.histogram(finite, bins=bins)
    peak = float(np.max(counts))
    if peak <= 0.0:
        return lo, hi
    threshold = max(1.0, float(min_density_ratio) * peak)
    tail_budget = float(max_tail_mass_fraction) * float(finite.size)
    start = 0
    dropped_low = 0.0
    while (
        start < bins - 1
        and float(counts[start]) < threshold
        and dropped_low + float(counts[start]) <= tail_budget
    ):
        dropped_low += float(counts[start])
        start += 1
    end = bins - 1
    dropped_high = 0.0
    while (
        end > start
        and float(counts[end]) < threshold
        and dropped_high + float(counts[end]) <= tail_budget
    ):
        dropped_high += float(counts[end])
        end -= 1
    return float(edges[start]), float(edges[end + 1])


def _trim_flying_pixel_tails(
    xy: np.ndarray,
    z: np.ndarray,
    axes: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if xy.shape[0] < 40:
        return xy, z
    local = (xy - center) @ axes
    keep = np.ones(local.shape[0], dtype=bool)
    for axis_index in range(2):
        lo, hi = _robust_local_interval(local[:, axis_index])
        keep &= (local[:, axis_index] >= lo - 1e-9) & (local[:, axis_index] <= hi + 1e-9)
    kept = int(np.count_nonzero(keep))
    # only ever remove a small tail: a merged two-object component must reach
    # the seeded split untouched, not be halved here
    if kept == keep.size or kept < max(16, int(0.75 * keep.size)):
        return xy, z
    return xy[keep], z[keep]


def _fit_from_metric_points(
    component: PixelComponent,
    xy: np.ndarray,
    z: np.ndarray,
    table_z_m: float,
    height_prior_m: float,
) -> GeometryFit:
    center, axes = _minimum_area_rect_axes(xy)
    trimmed_xy, trimmed_z = _trim_flying_pixel_tails(xy, z, axes, center)
    if trimmed_xy.shape[0] != xy.shape[0]:
        xy, z = trimmed_xy, trimmed_z
        center, axes = _minimum_area_rect_axes(xy)
    local = (xy - center) @ axes
    min_xy = np.min(local, axis=0)
    max_xy = np.max(local, axis=0)
    extents = np.maximum(max_xy - min_xy, 1e-6)
    local_center = 0.5 * (min_xy + max_xy)
    world_center = center + local_center @ axes.T
    yaw = _normalize_half_turn(math.atan2(float(axes[1, 0]), float(axes[0, 0])))
    major = float(extents[0])
    minor = float(extents[1])
    if minor > major:
        major, minor = minor, major
        yaw = _normalize_half_turn(yaw + math.pi / 2.0)

    if z.size >= 25:
        height = max(float(np.nanpercentile(z, 98.0) - table_z_m), 0.005)
    else:
        height = max(float(np.nanmax(z) - table_z_m), 0.005)
    aspect = major / max(minor, 1e-6)
    corner_supported = _has_corner_support(local, min_xy, max_xy)
    bbox_fill = _component_bbox_fill_ratio(component)
    bbox_aspect = _component_bbox_aspect(component)
    # a dense mask alone cannot veto a cylinder: a smear skirt can push a
    # circle mask's fill past 0.88, but only a square keeps its bbox corners
    dense_square_mask = (
        bbox_fill >= 0.88
        and bbox_aspect <= 1.20
        and _mask_corner_occupancy(component) >= 0.5
    )
    # low fill alone cannot tell a hole-riddled square from a plain circle
    # (a discrete circle mask fills ~0.74-0.78 of its bbox); a square keeps
    # its bbox corners occupied, a circle leaves them empty
    squareish_low_fill_mask = (
        0.60 <= bbox_fill <= 0.76
        and bbox_aspect <= 1.20
        and _mask_corner_occupancy(component) >= 0.5
    )
    near_square_metric = aspect <= 1.10

    if (
        aspect <= 1.25
        and not corner_supported
        and not dense_square_mask
        and not squareish_low_fill_mask
    ):
        diameter = max(major, minor)
        return GeometryFit(
            component_id=component.component_id,
            shape_type="cylinder",
            center_xy_m=(float(world_center[0]), float(world_center[1])),
            size_x_m=float(diameter),
            size_y_m=float(diameter),
            size_z_m=float(height),
            yaw_rad=0.0,
            provenance="metric_points_circular_footprint",
            bottom_z_m=float(table_z_m),
            center_z_m=float(table_z_m + 0.5 * height),
        )

    shape = (
        "box"
        if dense_square_mask
        or squareish_low_fill_mask
        or (corner_supported and bbox_fill >= 0.65)
        else "bbox"
    )
    if shape == "box" and near_square_metric:
        side = max(major, minor)
        if height / max(side, 1e-9) >= 0.55:
            side = min(side, height * 1.08)
        major = side
        minor = side
    return GeometryFit(
        component_id=component.component_id,
        shape_type=shape,
        center_xy_m=(float(world_center[0]), float(world_center[1])),
        size_x_m=float(major),
        size_y_m=float(minor),
        size_z_m=float(height),
        yaw_rad=float(yaw),
        provenance="metric_points_oriented_footprint",
        bottom_z_m=float(table_z_m),
        center_z_m=float(table_z_m + 0.5 * height),
    )


def _fit_from_mask_footprint(
    component: PixelComponent,
    height_prior_m: float,
    pixel_size_m: float,
    pixel_origin_xy_m: tuple[float, float],
    table_z_m: float,
) -> GeometryFit:
    pixel_size = _positive_scalar(pixel_size_m, "pixel_size_m")
    origin_x = _finite_scalar(pixel_origin_xy_m[0], "pixel_origin_xy_m[0]")
    origin_y = _finite_scalar(pixel_origin_xy_m[1], "pixel_origin_xy_m[1]")
    min_x, min_y, max_x, max_y = component.bbox_xyxy
    width = float(max_x - min_x + 1) * pixel_size
    depth = float(max_y - min_y + 1) * pixel_size
    center_x = origin_x + (float(min_x + max_x + 1) * 0.5) * pixel_size
    center_y = origin_y + (float(min_y + max_y + 1) * 0.5) * pixel_size

    return GeometryFit(
        component_id=component.component_id,
        shape_type="bbox",
        center_xy_m=(center_x, center_y),
        size_x_m=width,
        size_y_m=depth,
        size_z_m=height_prior_m,
        yaw_rad=0.0,
        provenance="mask_footprint_height_prior",
        bottom_z_m=float(table_z_m),
        center_z_m=float(table_z_m + 0.5 * height_prior_m),
    )


def _pca_axes(xy: np.ndarray) -> np.ndarray:
    centered = xy - np.mean(xy, axis=0)
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    if np.linalg.det(axes) < 0.0:
        axes[:, 1] *= -1.0
    return axes


def _minimum_area_rect_axes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hull = _convex_hull(xy)
    if hull.shape[0] < 2:
        return np.mean(xy, axis=0), np.eye(2)

    best_area = math.inf
    best_angle = 0.0
    best_local_center = np.zeros(2, dtype=float)
    best_axes = np.eye(2)

    for index in range(hull.shape[0]):
        edge = hull[(index + 1) % hull.shape[0]] - hull[index]
        if float(np.linalg.norm(edge)) <= 1e-12:
            continue
        angle = math.atan2(float(edge[1]), float(edge[0]))
        angle = _normalize_quarter_turn(angle)
        axes = _axes_from_yaw(angle)
        local = xy @ axes
        min_xy = np.min(local, axis=0)
        max_xy = np.max(local, axis=0)
        extents = np.maximum(max_xy - min_xy, 1e-9)
        area = float(extents[0] * extents[1])
        local_center = 0.5 * (min_xy + max_xy)
        if area < best_area - 1e-12 or (
            abs(area - best_area) <= 1e-12 and abs(angle) < abs(best_angle)
        ):
            best_area = area
            best_angle = angle
            best_local_center = local_center
            best_axes = axes

    world_center = best_local_center @ best_axes.T
    return world_center, best_axes


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(points, dtype=float), axis=0)
    if unique.shape[0] <= 1:
        return unique
    order = np.lexsort((unique[:, 1], unique[:, 0]))
    sorted_points = unique[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _axes_from_yaw(yaw: float) -> np.ndarray:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)


def _normalize_quarter_turn(angle: float) -> float:
    normalized = _normalize_half_turn(angle)
    if normalized <= -math.pi / 4.0:
        normalized += math.pi / 2.0
    if normalized > math.pi / 4.0:
        normalized -= math.pi / 2.0
    return normalized


def _has_corner_support(local_xy: np.ndarray, min_xy: np.ndarray, max_xy: np.ndarray) -> bool:
    """True when the footprint occupies at least three DISTINCT bbox corners.

    Counting corner points globally lets a trimmed circle fake support: after
    tail trimming, rim points near one shrunken edge normalize past the corner
    threshold, all in the same one or two corner zones. A real rectangle puts
    points into (nearly) all four corners.
    """
    span = np.maximum(max_xy - min_xy, 1e-6)
    normalized = (local_xy - min_xy) / span - 0.5
    in_corner = (np.abs(normalized[:, 0]) >= _CORNER_THRESHOLD) & (
        np.abs(normalized[:, 1]) >= _CORNER_THRESHOLD
    )
    if int(np.count_nonzero(in_corner)) < 4:
        return False
    occupied_zones = 0
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            zone = (
                in_corner
                & (np.sign(normalized[:, 0]) == sign_x)
                & (np.sign(normalized[:, 1]) == sign_y)
            )
            if int(np.count_nonzero(zone)) >= 2:
                occupied_zones += 1
    return occupied_zones >= 3


def _mask_corner_occupancy(component: PixelComponent, corner_fraction: float = 0.2) -> float:
    mask = np.asarray(component.mask, dtype=bool)
    min_x, min_y, max_x, max_y = component.bbox_xyxy
    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)
    corner_w = max(1, int(round(corner_fraction * width)))
    corner_h = max(1, int(round(corner_fraction * height)))
    submask = mask[min_y : max_y + 1, min_x : max_x + 1]
    corners = (
        submask[:corner_h, :corner_w],
        submask[:corner_h, width - corner_w :],
        submask[height - corner_h :, :corner_w],
        submask[height - corner_h :, width - corner_w :],
    )
    total = sum(int(c.size) for c in corners)
    occupied = sum(int(np.count_nonzero(c)) for c in corners)
    return float(occupied) / float(max(1, total))


def _component_bbox_fill_ratio(component: PixelComponent) -> float:
    min_x, min_y, max_x, max_y = component.bbox_xyxy
    bbox_area = max(1, int(max_x - min_x + 1) * int(max_y - min_y + 1))
    return float(component.area_px) / float(bbox_area)


def _component_bbox_aspect(component: PixelComponent) -> float:
    min_x, min_y, max_x, max_y = component.bbox_xyxy
    width = max(1, int(max_x - min_x + 1))
    height = max(1, int(max_y - min_y + 1))
    return float(max(width, height)) / float(min(width, height))


def _normalize_half_turn(angle: float) -> float:
    normalized = float(angle)
    while normalized <= -math.pi / 2.0:
        normalized += math.pi
    while normalized > math.pi / 2.0:
        normalized -= math.pi
    return normalized


def _finite_scalar(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _positive_scalar(value: float, name: str) -> float:
    numeric = _finite_scalar(value, name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric
