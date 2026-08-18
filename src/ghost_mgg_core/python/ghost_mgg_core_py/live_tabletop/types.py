from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PixelComponent:
    component_id: int
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_uv: tuple[float, float]
    mask: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class TabletopEvidence:
    foreground_mask: np.ndarray = field(repr=False, compare=False)
    hole_mask: np.ndarray = field(repr=False, compare=False)
    candidate_mask: np.ndarray = field(repr=False, compare=False)
    support_mask: np.ndarray = field(repr=False, compare=False)
    failure_mask: np.ndarray = field(repr=False, compare=False)
    table_leakage_mask: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class GeometryFit:
    component_id: int
    shape_type: str
    center_xy_m: tuple[float, float]
    size_x_m: float
    size_y_m: float
    size_z_m: float
    yaw_rad: float
    provenance: str
    bottom_z_m: float
    center_z_m: float


@dataclass(frozen=True)
class RankedGeometryHypothesis:
    fit: GeometryFit
    score: float
    score_terms: dict[str, float]
