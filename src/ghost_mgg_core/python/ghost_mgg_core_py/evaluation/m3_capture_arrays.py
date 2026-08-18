from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class M3ArrayConfig:
    width: int = 640
    height: int = 480
    roi_center_u_ratio: float = 0.50
    roi_center_v_ratio: float = 0.58
    roi_width_ratio: float = 0.22
    roi_height_ratio: float = 0.22
    table_depth_m: float = 1.20
    object_surface_depth_m: float = 1.12
    flying_point_offset_m: float = 0.12
    biased_depth_offset_m: float = -0.04
    edge_band_pixels: int = 2
    flying_point_stride: int = 5
    pattern_seed: int = 0


def scenario_array_config(scenario_id: str, metadata: dict[str, Any] | None = None) -> M3ArrayConfig:
    values = dict(metadata or {})
    scenario_defaults = {
        "S1": {"pattern_seed": 1},
        "S2": {"pattern_seed": 2},
        "S3": {"pattern_seed": 3},
        "S4": {"pattern_seed": 4},
        "S5": {
            "pattern_seed": 5,
            "edge_band_pixels": 3,
            "flying_point_offset_m": 0.16,
            "flying_point_stride": 4,
        },
        "S6": {"pattern_seed": 6},
        "S7": {
            "pattern_seed": 7,
            "biased_depth_offset_m": -0.05,
            "flying_point_offset_m": 0.14,
        },
    }
    values = scenario_defaults.get(str(scenario_id), {}) | values
    allowed = {field.name for field in M3ArrayConfig.__dataclass_fields__.values()}
    filtered = {key: values[key] for key in values if key in allowed}
    return M3ArrayConfig(**filtered)


def build_m3_capture_arrays(
    scenario_id: str,
    summary: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    config = scenario_array_config(scenario_id, metadata)
    failure_mode = str(summary.get("failure_mode", metadata.get("failure_mode", "disabled") if metadata else "disabled"))
    target_mask = _make_target_mask(config)
    raw_depth = np.full((config.height, config.width), config.table_depth_m, dtype=np.float32)
    raw_depth[target_mask] = np.float32(config.object_surface_depth_m)
    corrupted = raw_depth.copy()

    hole = np.zeros_like(target_mask, dtype=bool)
    table_leakage = np.zeros_like(target_mask, dtype=bool)
    edge = np.zeros_like(target_mask, dtype=bool)
    flying_point = np.zeros_like(target_mask, dtype=bool)
    biased_depth = np.zeros_like(target_mask, dtype=bool)

    rows, cols = np.nonzero(target_mask)
    for v, u in zip(rows.astype(np.uint32), cols.astype(np.uint32), strict=True):
        if _should_make_hole(failure_mode, int(u), int(v), config.pattern_seed):
            corrupted[v, u] = np.nan
            hole[v, u] = True
        elif _should_make_table_leakage(failure_mode, int(u), int(v), config.pattern_seed):
            corrupted[v, u] = np.float32(config.table_depth_m)
            table_leakage[v, u] = True
        elif failure_mode == "edge_only":
            if _is_edge_band_pixel(target_mask, int(u), int(v), config.edge_band_pixels):
                edge[v, u] = True
            else:
                corrupted[v, u] = np.nan
                hole[v, u] = True
        elif failure_mode == "flying_points":
            if _should_make_flying_point(int(u), int(v), config):
                corrupted[v, u] = np.float32(
                    _finite_or_table_depth(raw_depth[v, u], config) + _flying_sign(int(u), int(v), config) * config.flying_point_offset_m
                )
                flying_point[v, u] = True
            else:
                corrupted[v, u] = np.nan
                hole[v, u] = True
        elif failure_mode == "edge_flying":
            if _is_edge_band_pixel(target_mask, int(u), int(v), config.edge_band_pixels):
                edge[v, u] = True
            elif _should_make_flying_point(int(u), int(v), config):
                corrupted[v, u] = np.float32(
                    _finite_or_table_depth(raw_depth[v, u], config) + _flying_sign(int(u), int(v), config) * config.flying_point_offset_m
                )
                flying_point[v, u] = True
            else:
                corrupted[v, u] = np.nan
                hole[v, u] = True
        elif failure_mode == "biased_patch":
            corrupted[v, u] = np.float32(_finite_or_table_depth(raw_depth[v, u], config) + config.biased_depth_offset_m)
            biased_depth[v, u] = True
        elif failure_mode == "reflective":
            selector = (int(u) + 3 * int(v) + config.pattern_seed) % 5
            if selector == 0:
                corrupted[v, u] = np.nan
                hole[v, u] = True
            elif selector in (1, 2):
                corrupted[v, u] = np.float32(_finite_or_table_depth(raw_depth[v, u], config) + config.biased_depth_offset_m)
                biased_depth[v, u] = True
            elif selector == 3:
                corrupted[v, u] = np.float32(_finite_or_table_depth(raw_depth[v, u], config) + config.flying_point_offset_m)
                flying_point[v, u] = True
            elif _is_edge_band_pixel(target_mask, int(u), int(v), config.edge_band_pixels):
                edge[v, u] = True

    valid = target_mask & np.isfinite(corrupted) & (corrupted > 0.0)
    foreground_support = target_mask & valid & ~(table_leakage | flying_point | biased_depth | edge)
    foreground_support &= corrupted < (config.table_depth_m - 0.04)

    return {
        "target_mask": target_mask,
        "raw_depth_m": raw_depth,
        "corrupted_depth_m": corrupted,
        "evidence_valid": valid.astype(np.float32),
        "evidence_hole": hole.astype(np.float32),
        "evidence_table_leakage": table_leakage.astype(np.float32),
        "evidence_edge": edge.astype(np.float32),
        "evidence_flying_point": flying_point.astype(np.float32),
        "evidence_biased_depth": biased_depth.astype(np.float32),
        "evidence_foreground_support": foreground_support.astype(np.float32),
        "camera_k": np.array(
            [
                [554.0, 0.0, config.width / 2.0],
                [0.0, 554.0, config.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "table_plane_depth_axis": np.array([0.0, 0.0, 1.0, -config.table_depth_m], dtype=np.float32),
    }


def _make_target_mask(config: M3ArrayConfig) -> np.ndarray:
    width_px = _clamped_pixel_count(config.roi_width_ratio, config.width)
    height_px = _clamped_pixel_count(config.roi_height_ratio, config.height)
    center_u = round(np.clip(config.roi_center_u_ratio, 0.0, 1.0) * config.width)
    center_v = round(np.clip(config.roi_center_v_ratio, 0.0, 1.0) * config.height)
    u_min = int(np.clip(center_u - width_px // 2, 0, max(0, config.width - width_px)))
    v_min = int(np.clip(center_v - height_px // 2, 0, max(0, config.height - height_px)))
    mask = np.zeros((config.height, config.width), dtype=bool)
    mask[v_min : v_min + height_px, u_min : u_min + width_px] = True
    return mask


def _clamped_pixel_count(ratio: float, extent: int) -> int:
    return max(1, round(float(np.clip(ratio, 0.0, 1.0)) * int(extent)))


def _should_make_hole(mode: str, u: int, v: int, seed: int) -> bool:
    if mode == "hole":
        return True
    if mode != "mixed":
        return False
    return ((u + v + seed) % 2) == 0


def _should_make_table_leakage(mode: str, u: int, v: int, seed: int) -> bool:
    if mode == "table_leakage":
        return True
    if mode != "mixed":
        return False
    return not _should_make_hole(mode, u, v, seed)


def _is_edge_band_pixel(mask: np.ndarray, u: int, v: int, band: int) -> bool:
    if not mask[v, u]:
        return False
    band = max(1, int(band))
    height, width = mask.shape
    for dv in range(-band, band + 1):
        for du in range(-band, band + 1):
            nu = u + du
            nv = v + dv
            if nu < 0 or nv < 0 or nu >= width or nv >= height:
                return True
            if not mask[nv, nu]:
                return True
    return False


def _should_make_flying_point(u: int, v: int, config: M3ArrayConfig) -> bool:
    stride = max(1, int(config.flying_point_stride))
    hashed = (u * 73856093) ^ (v * 19349663) ^ (int(config.pattern_seed) * 83492791)
    return (hashed % stride) == 0


def _flying_sign(u: int, v: int, config: M3ArrayConfig) -> float:
    return 1.0 if ((u + v + int(config.pattern_seed)) % 2) == 0 else -1.0


def _finite_or_table_depth(depth: float, config: M3ArrayConfig) -> float:
    return float(depth) if np.isfinite(depth) and float(depth) > 0.0 else config.table_depth_m
