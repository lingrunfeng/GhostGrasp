from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RenderedProxy:
    silhouette: np.ndarray
    boundary: np.ndarray
    expected_depth: np.ndarray


def render_proxy(hypothesis, image_shape, boundary_width_px=2):
    height, width = _validate_image_shape(image_shape)
    boundary_width = _validate_positive_integer(boundary_width_px, "boundary_width_px")
    yy, xx = np.indices((height, width))

    center_u, center_v = hypothesis.center_uv
    size_u, size_v = _validate_size(hypothesis.size_px)
    yaw_rad = _validate_finite_scalar(getattr(hypothesis, "yaw_rad", 0.0), "yaw_rad")
    expected_surface_depth = _validate_depth_interval(hypothesis.depth_m, hypothesis.height_m)

    if hypothesis.shape_type == "box":
        silhouette = _box_silhouette(xx, yy, center_u, center_v, size_u, size_v, yaw_rad)
    elif hypothesis.shape_type == "cylinder":
        silhouette = _ellipse_silhouette(xx, yy, center_u, center_v, size_u, size_v)
    else:
        raise ValueError(f"Unsupported primitive shape_type: {hypothesis.shape_type}")

    if not silhouette.any():
        raise ValueError("proxy silhouette is empty in the requested image shape")

    boundary = _boundary_band(silhouette, boundary_width)
    expected_depth = np.full((height, width), np.nan, dtype=float)
    expected_depth[silhouette] = expected_surface_depth

    return RenderedProxy(
        silhouette=silhouette.astype(bool, copy=False),
        boundary=boundary,
        expected_depth=expected_depth,
    )


def _box_silhouette(xx, yy, center_u, center_v, size_u, size_v, yaw_rad):
    half_u = float(size_u) / 2.0
    half_v = float(size_v) / 2.0
    du = xx - float(center_u)
    dv = yy - float(center_v)
    cos_yaw = np.cos(float(yaw_rad))
    sin_yaw = np.sin(float(yaw_rad))
    local_u = cos_yaw * du + sin_yaw * dv
    local_v = -sin_yaw * du + cos_yaw * dv
    return (np.abs(local_u) <= half_u) & (np.abs(local_v) <= half_v)


def _ellipse_silhouette(xx, yy, center_u, center_v, size_u, size_v):
    radius_u = float(size_u) / 2.0
    radius_v = float(size_v) / 2.0
    normalized = ((xx - float(center_u)) / radius_u) ** 2 + ((yy - float(center_v)) / radius_v) ** 2
    return normalized <= 1.0


def _boundary_band(silhouette, boundary_width_px):
    dilated = _dilate(silhouette, boundary_width_px)
    eroded = ~_dilate(~silhouette, boundary_width_px)
    return dilated & ~eroded


def _dilate(mask, radius_px):
    radius = int(radius_px)
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask, dtype=bool)

    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            if du * du + dv * dv <= radius * radius:
                v_start = radius + dv
                u_start = radius + du
                dilated |= padded[v_start : v_start + mask.shape[0], u_start : u_start + mask.shape[1]]

    return dilated


def _validate_image_shape(image_shape):
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain height and width")
    height = _validate_positive_integer(image_shape[0], "image_shape height")
    width = _validate_positive_integer(image_shape[1], "image_shape width")
    return height, width


def _validate_positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    numeric = int(value)
    if float(value) != float(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return numeric


def _validate_size(size_px):
    if len(size_px) != 2:
        raise ValueError("size_px must contain width and height")
    return (
        _validate_positive_finite(size_px[0], "size_px width"),
        _validate_positive_finite(size_px[1], "size_px height"),
    )


def _validate_depth_interval(depth_m, height_m):
    depth = _validate_positive_finite(depth_m, "depth_m")
    height = _validate_positive_finite(height_m, "height_m")
    expected_surface_depth = depth - height
    if expected_surface_depth <= 0.0:
        raise ValueError("depth_m - height_m must be positive")
    return expected_surface_depth


def _validate_finite_scalar(value, name):
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _validate_positive_finite(value, name):
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric
