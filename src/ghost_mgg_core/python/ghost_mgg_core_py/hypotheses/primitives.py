from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrimitiveHypothesis:
    hypothesis_id: str
    shape_type: str
    center_uv: tuple[float, float]
    size_px: tuple[float, float]
    depth_m: float
    height_m: float
    prior_score: float = 0.0
    yaw_rad: float = 0.0
    center_xy_m: tuple[float, float] | None = None
    size_xy_m: tuple[float, float] | None = None
    bottom_z_m: float | None = None
    center_z_m: float | None = None
