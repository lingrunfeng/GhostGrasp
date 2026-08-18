from dataclasses import dataclass

import numpy as np

from ghost_mgg_core_py.evidence.types import EvidenceMaps
from ghost_mgg_core_py.hypotheses.primitives import PrimitiveHypothesis


@dataclass(frozen=True)
class SyntheticScene:
    scene_id: str
    target_mask: np.ndarray
    evidence: EvidenceMaps
    candidates: tuple[PrimitiveHypothesis, ...]
    ground_truth_id: str


def make_failure_ranking_scene(scene_id: str, shift_px: int) -> SyntheticScene:
    image_shape = (32, 32)
    target_mask = np.zeros(image_shape, dtype=bool)
    target_mask[8:24, 8:24] = True

    evidence = _empty_evidence(image_shape)
    offset = int(shift_px)
    core_start = 12 + offset
    core_stop = core_start + 6
    core_center = (float(core_start + core_stop - 1) / 2.0, 15.5)
    evidence.hole[13:19, core_start:core_stop] = 1.0
    evidence.table_leakage[13:19, core_start:core_stop] = 1.0

    candidates = (
        PrimitiveHypothesis(
            hypothesis_id="gt_core",
            shape_type="box",
            center_uv=core_center,
            size_px=(6.0, 6.0),
            depth_m=1.0,
            height_m=0.1,
            prior_score=0.0,
        ),
        PrimitiveHypothesis(
            hypothesis_id="silhouette_prior",
            shape_type="box",
            center_uv=(15.5 + float(offset), 15.5),
            size_px=(16.0, 16.0),
            depth_m=1.0,
            height_m=0.1,
            prior_score=0.2,
        ),
    )

    return SyntheticScene(
        scene_id=scene_id,
        target_mask=target_mask,
        evidence=evidence,
        candidates=candidates,
        ground_truth_id="gt_core",
    )


def _empty_evidence(shape: tuple[int, int]) -> EvidenceMaps:
    zeros = np.zeros(shape, dtype=float)
    return EvidenceMaps(
        valid=np.ones(shape, dtype=float),
        hole=zeros.copy(),
        table_leakage=zeros.copy(),
        edge=zeros.copy(),
        flying_point=zeros.copy(),
        foreground_support=zeros.copy(),
    )
