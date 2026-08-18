from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class TablePlane:
    normal: np.ndarray
    offset: float

    def normalized(self) -> "TablePlane":
        normal = np.asarray(self.normal, dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise ValueError("table plane normal must be nonzero")
        return TablePlane(normal=normal / norm, offset=float(self.offset) / norm)


@dataclass(frozen=True)
class EvidenceMaps:
    valid: np.ndarray
    hole: np.ndarray
    table_leakage: np.ndarray
    edge: np.ndarray
    flying_point: np.ndarray
    foreground_support: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "valid": self.valid,
            "hole": self.hole,
            "table_leakage": self.table_leakage,
            "edge": self.edge,
            "flying_point": self.flying_point,
            "foreground_support": self.foreground_support,
        }


@dataclass(frozen=True)
class EvidenceSummary:
    valid_depth_ratio: float
    hole_ratio: float
    table_leakage_ratio: float
    edge_support_ratio: float
    flying_point_ratio: float
    foreground_support_ratio: float
    failure_spatial_concentration: float

    def as_dict(self) -> dict[str, float]:
        return {
            "valid_depth_ratio": self.valid_depth_ratio,
            "hole_ratio": self.hole_ratio,
            "table_leakage_ratio": self.table_leakage_ratio,
            "edge_support_ratio": self.edge_support_ratio,
            "flying_point_ratio": self.flying_point_ratio,
            "foreground_support_ratio": self.foreground_support_ratio,
            "failure_spatial_concentration": self.failure_spatial_concentration,
        }
