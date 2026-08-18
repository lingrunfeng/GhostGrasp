from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ghost_mgg_core_py.evidence.types import EvidenceMaps
from ghost_mgg_core_py.hypotheses.hypothesis_generator import generate_local_hypotheses
from ghost_mgg_core_py.scoring.joint_ranker import RankedHypothesis, rank_hypotheses


GHOST_MGG_V0_WEIGHTS = {
    "visual": 1.0,
    "failure": 4.0,
    "depth": 0.5,
    "physical": 0.0,
    "grasp": 0.0,
    "prior": 1.0,
}

SILHOUETTE_ONLY_WEIGHTS = {
    "visual": 1.0,
    "failure": 0.0,
    "depth": 0.0,
    "physical": 0.0,
    "grasp": 0.0,
    "prior": 0.0,
}

GHOST_MGG_V0_ABLATIONS = {
    "full": {"weights": GHOST_MGG_V0_WEIGHTS, "zero_channels": ()},
    "silhouette_only": {"weights": SILHOUETTE_ONLY_WEIGHTS, "zero_channels": ()},
    "without_failure": {
        "weights": {
            "visual": 1.0,
            "failure": 0.0,
            "depth": 0.5,
            "physical": 0.0,
            "grasp": 0.0,
            "prior": 1.0,
        },
        "zero_channels": (),
    },
    "without_table_leakage": {"weights": GHOST_MGG_V0_WEIGHTS, "zero_channels": ("table_leakage",)},
    "without_edge_flying": {"weights": GHOST_MGG_V0_WEIGHTS, "zero_channels": ("edge", "flying_point")},
    "without_weak_depth": {"weights": GHOST_MGG_V0_WEIGHTS, "zero_channels": ("foreground_support",)},
}


@dataclass(frozen=True)
class GhostMGGV0Config:
    shape_types: tuple[str, ...] = ("box", "cylinder")
    scale_factors: tuple[float, ...] = (0.75, 0.90, 1.00, 1.10, 1.25)
    depth_m: float = 1.20
    height_m: float = 0.08
    top_k: int = 3
    min_total_score: float = -float("inf")


def run_ghost_mgg_v0(
    target_mask,
    evidence: EvidenceMaps,
    *,
    config: GhostMGGV0Config | None = None,
    weights: dict[str, float] | None = None,
    hypotheses=None,
) -> list[RankedHypothesis]:
    resolved_config = config or GhostMGGV0Config()
    mask = np.asarray(target_mask, dtype=bool)
    candidates = list(hypotheses) if hypotheses is not None else generate_local_hypotheses(
        mask,
        shape_types=resolved_config.shape_types,
        scale_factors=resolved_config.scale_factors,
        depth_m=resolved_config.depth_m,
        height_m=resolved_config.height_m,
    )
    ranked = rank_hypotheses(
        candidates,
        mask,
        evidence,
        min_total_score=resolved_config.min_total_score,
        weights=GHOST_MGG_V0_WEIGHTS if weights is None else weights,
    )
    return ranked[: max(1, int(resolved_config.top_k))]


def evidence_from_capture_arrays(arrays, zero_channels: tuple[str, ...] = ()) -> EvidenceMaps:
    values = {
        "valid": arrays["evidence_valid"],
        "hole": arrays["evidence_hole"],
        "table_leakage": arrays["evidence_table_leakage"],
        "edge": arrays["evidence_edge"],
        "flying_point": arrays["evidence_flying_point"],
        "foreground_support": arrays["evidence_foreground_support"],
    }
    for channel in zero_channels:
        values[channel] = np.zeros_like(values[channel], dtype=np.float32)
    return EvidenceMaps(**values)
