import numpy as np

from ghost_mgg_core_py.evidence.types import EvidenceMaps
from ghost_mgg_core_py.hypotheses.primitives import PrimitiveHypothesis

_SUPPORTED_SHAPE_TYPES = frozenset({"box", "cylinder"})
_SQUARE_ANISOTROPY_THRESHOLD = 1.12


def generate_local_hypotheses(
    target_mask,
    shape_types=("box", "cylinder"),
    scale_factors=(0.8, 1.0, 1.2),
    depth_m=1.0,
    height_m=0.08,
    yaw_rads=(0.0,),
):
    mask = np.asarray(target_mask, dtype=bool)
    if not mask.any():
        return []

    rows, cols = np.nonzero(mask)
    min_u = float(cols.min())
    max_u = float(cols.max())
    min_v = float(rows.min())
    max_v = float(rows.max())

    center_uv = ((min_u + max_u) / 2.0, (min_v + max_v) / 2.0)
    width_px = max_u - min_u + 1.0
    height_px = max_v - min_v + 1.0

    hypotheses = []
    for shape_type in shape_types:
        shape_name = str(shape_type)
        if shape_name not in _SUPPORTED_SHAPE_TYPES:
            raise ValueError(f"unsupported shape_type: {shape_name}")
        yaw_values = tuple(float(yaw) for yaw in yaw_rads) if shape_name == "box" else (0.0,)
        for scale in scale_factors:
            scale_value = _positive_finite(scale, "scale_factors")
            for yaw_rad in yaw_values:
                yaw_suffix = "" if abs(float(yaw_rad)) < 1e-6 else f"_yaw{int(round(np.rad2deg(yaw_rad))):+03d}"
                hypotheses.append(
                    PrimitiveHypothesis(
                        hypothesis_id=f"{shape_type}{yaw_suffix}_s{scale_value:.2f}",
                        shape_type=shape_name,
                        center_uv=center_uv,
                        size_px=(width_px * scale_value, height_px * scale_value),
                        depth_m=float(depth_m),
                        height_m=float(height_m),
                        yaw_rad=float(yaw_rad),
                    )
                )

    return hypotheses


def generate_table_anchored_hypotheses(
    target_mask: np.ndarray,
    evidence: EvidenceMaps,
    *,
    footprint_center_xy_m: tuple[float, float],
    footprint_size_xy_m: tuple[float, float],
    table_z_m: float,
    height_priors_m: tuple[float, ...] = (0.02, 0.03, 0.04, 0.06),
    shape_types: tuple[str, ...] = ("box", "cylinder"),
    yaw_rads: tuple[float, ...] = (0.0,),
) -> list[PrimitiveHypothesis]:
    mask = np.asarray(target_mask, dtype=bool)
    _validate_evidence_shape(evidence, mask.shape)
    if not mask.any():
        return []

    center_x = _finite_scalar(footprint_center_xy_m[0], "footprint_center_xy_m[0]")
    center_y = _finite_scalar(footprint_center_xy_m[1], "footprint_center_xy_m[1]")
    size_x = _positive_finite(footprint_size_xy_m[0], "footprint_size_xy_m[0]")
    size_y = _positive_finite(footprint_size_xy_m[1], "footprint_size_xy_m[1]")
    table_z = _finite_scalar(table_z_m, "table_z_m")

    rows, cols = np.nonzero(mask)
    min_u = float(cols.min())
    max_u = float(cols.max())
    min_v = float(rows.min())
    max_v = float(rows.max())
    center_uv = ((min_u + max_u) / 2.0, (min_v + max_v) / 2.0)
    width_px = max_u - min_u + 1.0
    height_px = max_v - min_v + 1.0

    hypotheses: list[PrimitiveHypothesis] = []
    for shape_type in shape_types:
        shape_name = str(shape_type)
        if shape_name not in _SUPPORTED_SHAPE_TYPES:
            raise ValueError(f"unsupported shape_type: {shape_name}")
        yaw_values = tuple(float(yaw) for yaw in yaw_rads) if shape_name == "box" else (0.0,)
        for height in height_priors_m:
            height_m = _positive_finite(height, "height_priors_m")
            if shape_name == "cylinder":
                metric_x = metric_y = max(size_x, size_y)
                px_x = px_y = max(width_px, height_px)
            else:
                metric_x, metric_y = _stabilize_box_size(size_x, size_y)
                px_x, px_y = _stabilize_box_size(width_px, height_px)
            height_mm = int(round(height_m * 1000.0))
            for yaw_rad in yaw_values:
                yaw_suffix = "" if abs(float(yaw_rad)) < 1e-6 else f"_yaw{int(round(np.rad2deg(yaw_rad))):+04d}"
                hypotheses.append(
                    PrimitiveHypothesis(
                        hypothesis_id=f"{shape_name}{yaw_suffix}_h{height_mm:03d}",
                        shape_type=shape_name,
                        center_uv=center_uv,
                        size_px=(float(px_x), float(px_y)),
                        depth_m=1.0,
                        height_m=float(height_m),
                        prior_score=0.0,
                        yaw_rad=float(yaw_rad),
                        center_xy_m=(float(center_x), float(center_y)),
                        size_xy_m=(float(metric_x), float(metric_y)),
                        bottom_z_m=float(table_z),
                        center_z_m=float(table_z + 0.5 * height_m),
                    )
                )
    return hypotheses


def _positive_finite(value, name):
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} entries must be positive and finite")
    return numeric


def _finite_scalar(value, name):
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _stabilize_box_size(width, depth):
    width_value = _positive_finite(width, "width")
    depth_value = _positive_finite(depth, "depth")
    smaller = min(width_value, depth_value)
    larger = max(width_value, depth_value)
    if larger / smaller < _SQUARE_ANISOTROPY_THRESHOLD:
        return larger, larger
    return width_value, depth_value


def _validate_evidence_shape(evidence: EvidenceMaps, shape: tuple[int, int]) -> None:
    for name, array in evidence.as_dict().items():
        if np.asarray(array).shape != shape:
            raise ValueError(f"evidence.{name} shape {np.asarray(array).shape} does not match target mask {shape}")
