#!/usr/bin/env python3
"""Live no-truth GHOST-MGG v1 hypothesis publisher.

This node is intentionally camera-data driven. It does not query Gazebo model
poses and does not read simulation target metadata.
"""

import math
import os
import time as pytime
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from std_msgs.msg import Float32MultiArray
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from ghost_mgg_interfaces.msg import (
    GeometryHypothesis,
    GeometryHypothesisArray,
    GraspCandidate,
    ScoreBreakdown,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CORE_PYTHON = REPO_ROOT / "src" / "ghost_mgg_core" / "python"
if str(CORE_PYTHON) not in sys.path:
    sys.path.insert(0, str(CORE_PYTHON))

from ghost_mgg_core_py.evidence.depth_failure import evidence_from_raw_depth
from ghost_mgg_core_py.evidence.types import EvidenceMaps
from ghost_mgg_core_py.ghost_mgg_v0 import GhostMGGV0Config, run_ghost_mgg_v0
from ghost_mgg_core_py.hypotheses.hypothesis_generator import (
    generate_local_hypotheses,
    generate_table_anchored_hypotheses,
)
from ghost_mgg_core_py.hypotheses.primitives import PrimitiveHypothesis
from ghost_mgg_core_py.live_tabletop import (
    PixelComponent,
    build_tabletop_evidence,
    extract_components,
    group_components_by_foreground_islands as core_group_components_by_foreground_islands,
    rank_tabletop_components,
)
from ghost_mgg_core_py.live_tabletop.grouping import pixel_component_from_mask
from ghost_mgg_core_py.live_tabletop.types import GeometryFit


DEFAULT_RAW_DEPTH_TOPIC = "/ghost_mgg/d435/depth/image_rect_raw"
DEFAULT_CORRUPTED_DEPTH_TOPIC = "/ghost_mgg/d435/depth/m3_corrupted"
DEFAULT_COLOR_TOPIC = "/ghost_mgg/d435/color/image_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/ghost_mgg/d435/depth/camera_info"
DEFAULT_MASK_TOPIC = "/ghost_mgg/d435/target_mask"
DEFAULT_EXTERNAL_MASK_TOPIC = "/ghost_mgg/d435/external_target_mask"
DEFAULT_HYPOTHESIS_TOPIC = "/ghost_mgg/m4_live_hypotheses"


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class TableFootprint:
    center_xy_m: tuple[float, float]
    size_x_m: float
    size_y_m: float
    width_px: float
    height_px: float
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class CenterRefinement:
    suffix: str
    center_xy_m: tuple[float, float]
    center_uv_delta: tuple[float, float] | None


@dataclass(frozen=True)
class StableHypothesisTrack:
    shape_type: int
    center_xy_m: tuple[float, float]
    observations: int
    misses: int = 0
    center_z_m: float = 0.0
    size_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_rad: float = 0.0
    pending_shape_type: int | None = None
    pending_shape_observations: int = 0
    track_id: int = 0
    dimension_hold: bool = False
    shrink_streak: int = 0
    expand_streak: int = 0
    fast_relock: bool = False
    shape_votes: tuple = ()


@dataclass(frozen=True)
class StableComponentTrack:
    center_xy_m: tuple[float, float]
    area_px: int
    observations: int
    misses: int = 0
    track_id: int = 0


def _diag_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.4f}"


def _diag_xy(value: tuple[float, float] | None) -> str:
    if value is None:
        return "none"
    return f"{_diag_float(value[0])}/{_diag_float(value[1])}"


def _diag_footprint(footprint: TableFootprint | None) -> str:
    if footprint is None:
        return "none"
    return (
        f"xy={_diag_xy(footprint.center_xy_m)},"
        f"size={_diag_float(footprint.size_x_m)}/{_diag_float(footprint.size_y_m)},"
        f"yaw={math.degrees(float(footprint.yaw_rad)):.1f}"
    )


def evidence_diagnostics_provenance(
    *,
    fit: GeometryFit,
    support_source: str,
    support_footprint: TableFootprint | None,
    full_mask_footprint: TableFootprint | None,
    fit_points_count: int,
    residual_points_count: int,
) -> str:
    return (
        "evidence_diag:"
        f"support_source={support_source or 'none'};"
        f"residual_n={int(residual_points_count)};"
        f"fit_n={int(fit_points_count)};"
        f"final_xy={_diag_xy(fit.center_xy_m)};"
        f"final_size={_diag_float(fit.size_x_m)}/{_diag_float(fit.size_y_m)}/{_diag_float(fit.size_z_m)};"
        f"final_yaw_deg={math.degrees(float(fit.yaw_rad)):.1f};"
        f"shadow_{_diag_footprint(support_footprint)};"
        f"full_{_diag_footprint(full_mask_footprint)}"
    )


def decode_depth_image(image: Image) -> np.ndarray:
    if image.encoding == "32FC1":
        dtype = np.dtype(">f4" if image.is_bigendian else "<f4")
        depth = np.frombuffer(bytes(image.data), dtype=dtype).reshape(image.height, image.step // 4)
        return depth[:, : image.width].astype(np.float32, copy=True)
    if image.encoding == "16UC1":
        dtype = np.dtype(">u2" if image.is_bigendian else "<u2")
        depth_mm = np.frombuffer(bytes(image.data), dtype=dtype).reshape(image.height, image.step // 2)
        return depth_mm[:, : image.width].astype(np.float32, copy=True) * 0.001
    raise ValueError(f"unsupported depth encoding: {image.encoding}")


def decode_color_image(image: Image) -> np.ndarray:
    channels_by_encoding = {
        "rgb8": (3, (0, 1, 2)),
        "bgr8": (3, (2, 1, 0)),
        "rgba8": (4, (0, 1, 2)),
        "bgra8": (4, (2, 1, 0)),
    }
    if image.encoding not in channels_by_encoding:
        raise ValueError(f"unsupported color encoding: {image.encoding}")
    channel_count, rgb_indices = channels_by_encoding[image.encoding]
    row_stride = image.step
    row_values = np.frombuffer(bytes(image.data), dtype=np.uint8).reshape(image.height, row_stride)
    packed = row_values[:, : image.width * channel_count].reshape(
        image.height,
        image.width,
        channel_count,
    )
    return packed[:, :, rgb_indices].astype(np.uint8, copy=True)


def encode_mask_image(mask: np.ndarray, header: Any) -> Image:
    resolved = np.asarray(mask, dtype=bool)
    image = Image()
    image.header = header
    image.height = int(resolved.shape[0])
    image.width = int(resolved.shape[1])
    image.encoding = "mono8"
    image.is_bigendian = False
    image.step = image.width
    image.data = (resolved.astype(np.uint8) * 255).reshape(-1).tolist()
    return image


def decode_mask_image(image: Image) -> np.ndarray:
    if image.encoding not in ("mono8", "8UC1"):
        raise ValueError(f"unsupported mask encoding: {image.encoding}")
    row_values = np.frombuffer(bytes(image.data), dtype=np.uint8).reshape(image.height, image.step)
    return row_values[:, : image.width] > 0


def resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    target_height, target_width = int(shape[0]), int(shape[1])
    if source.shape == (target_height, target_width):
        return source.copy()
    if source.size == 0 or target_height <= 0 or target_width <= 0:
        return np.zeros((target_height, target_width), dtype=bool)
    row_indices = np.clip(
        np.round(np.linspace(0, source.shape[0] - 1, target_height)).astype(int),
        0,
        source.shape[0] - 1,
    )
    col_indices = np.clip(
        np.round(np.linspace(0, source.shape[1] - 1, target_width)).astype(int),
        0,
        source.shape[1] - 1,
    )
    return source[row_indices[:, None], col_indices[None, :]]


def decimated_camera_model(intrinsics: CameraModel, stride: int) -> CameraModel:
    step = max(1, int(stride))
    if step == 1:
        return intrinsics
    return CameraModel(
        width=int(math.ceil(float(intrinsics.width) / float(step))),
        height=int(math.ceil(float(intrinsics.height) / float(step))),
        fx=float(intrinsics.fx) / float(step),
        fy=float(intrinsics.fy) / float(step),
        cx=float(intrinsics.cx) / float(step),
        cy=float(intrinsics.cy) / float(step),
    )


def decimate_mask_any(mask: np.ndarray, stride: int) -> np.ndarray:
    step = max(1, int(stride))
    source = np.asarray(mask, dtype=bool)
    if step == 1:
        return source
    target_height = int(math.ceil(float(source.shape[0]) / float(step)))
    target_width = int(math.ceil(float(source.shape[1]) / float(step)))
    padded = np.zeros((target_height * step, target_width * step), dtype=bool)
    padded[: source.shape[0], : source.shape[1]] = source
    return padded.reshape(target_height, step, target_width, step).any(axis=(1, 3))


def target_color_mask_from_image(
    rgb_image: np.ndarray,
    *,
    color_hint: str,
    min_area_px: int = 40,
    locked_center_uv: tuple[float, float] | None = None,
    max_locked_center_distance_px: float = 60.0,
) -> np.ndarray:
    hint = str(color_hint).strip().lower()
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or hint in ("", "none", "auto", "depth"):
        return np.zeros(rgb.shape[:2], dtype=bool)

    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    maximum = np.maximum.reduce([red, green, blue])
    minimum = np.minimum.reduce([red, green, blue])
    saturation = maximum - minimum

    if hint == "red":
        raw = (red >= 90) & (red >= green + 45) & (red >= blue + 45) & (saturation >= 45)
    elif hint == "green":
        raw = (green >= 80) & (green >= red + 35) & (green >= blue + 35) & (saturation >= 35)
    elif hint == "blue":
        raw = (blue >= 80) & (blue >= red + 35) & (blue >= green + 35) & (saturation >= 35)
    else:
        return np.zeros(rgb.shape[:2], dtype=bool)

    return target_sized_component(
        raw,
        min_area_px=int(min_area_px),
        locked_center_uv=locked_center_uv,
        max_locked_center_distance_px=max_locked_center_distance_px,
    )


def camera_model_from_info(info: CameraInfo) -> CameraModel:
    return CameraModel(
        width=int(info.width),
        height=int(info.height),
        fx=float(info.k[0]),
        fy=float(info.k[4]),
        cx=float(info.k[2]),
        cy=float(info.k[5]),
    )


def foreground_mask_from_depth(
    depth_m: np.ndarray,
    *,
    min_depth_m: float = 0.15,
    max_depth_m: float = 2.0,
    depth_margin_m: float = 0.035,
    min_component_area_px: int = 40,
    locked_center_uv: tuple[float, float] | None = None,
    max_locked_center_distance_px: float = 60.0,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= float(min_depth_m)) & (depth <= float(max_depth_m))
    if not valid.any():
        return np.zeros(depth.shape, dtype=bool)

    row_table = np.full(depth.shape[0], np.nan, dtype=np.float32)
    for row_index in range(depth.shape[0]):
        row_valid_depth = depth[row_index, valid[row_index]]
        if row_valid_depth.size:
            row_table[row_index] = np.percentile(row_valid_depth, 72.0)
    global_table = float(np.nanpercentile(depth[valid], 72.0))
    row_table = np.where(np.isfinite(row_table), row_table, global_table)
    margin_candidates = [
        float(depth_margin_m),
        float(depth_margin_m) * 0.75,
        float(depth_margin_m) * 0.50,
        0.020,
        0.015,
        0.010,
        0.005,
    ]
    seen: set[float] = set()
    for margin in margin_candidates:
        margin = round(max(0.001, float(margin)), 4)
        if margin in seen:
            continue
        seen.add(margin)
        foreground = valid & (depth < (row_table[:, None] - margin))
        component = target_sized_component(
            foreground,
            min_area_px=int(min_component_area_px),
            locked_center_uv=locked_center_uv,
            max_locked_center_distance_px=max_locked_center_distance_px,
        )
        if component.any():
            return component
    return np.zeros(depth.shape, dtype=bool)


def foreground_components_mask_from_depth(
    depth_m: np.ndarray,
    *,
    min_depth_m: float = 0.15,
    max_depth_m: float = 2.0,
    depth_margin_m: float = 0.035,
    min_component_area_px: int = 40,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= float(min_depth_m)) & (depth <= float(max_depth_m))
    if not valid.any():
        return np.zeros(depth.shape, dtype=bool)

    row_table = np.full(depth.shape[0], np.nan, dtype=np.float32)
    for row_index in range(depth.shape[0]):
        row_valid_depth = depth[row_index, valid[row_index]]
        if row_valid_depth.size:
            row_table[row_index] = np.percentile(row_valid_depth, 72.0)
    global_table = float(np.nanpercentile(depth[valid], 72.0))
    row_table = np.where(np.isfinite(row_table), row_table, global_table)
    margin_candidates = [
        float(depth_margin_m),
        float(depth_margin_m) * 0.75,
        float(depth_margin_m) * 0.50,
        0.020,
        0.015,
        0.010,
        0.005,
    ]
    seen: set[float] = set()
    output = np.zeros(depth.shape, dtype=bool)
    for margin in margin_candidates:
        margin = round(max(0.001, float(margin)), 4)
        if margin in seen:
            continue
        seen.add(margin)
        foreground = valid & (depth < (row_table[:, None] - margin))
        strict_components = target_sized_components(
            foreground,
            min_area_px=int(min_component_area_px),
        )
        output |= strict_components

        broad_components = target_sized_components(
            foreground,
            min_area_px=int(min_component_area_px),
            max_area_fraction=0.20,
            max_bbox_width_fraction=0.85,
            max_bbox_height_fraction=0.85,
        )
        output = add_nonoverlapping_components(
            output,
            broad_components,
            min_area_px=int(min_component_area_px),
            max_overlap_fraction=0.05,
        )
    return output


def target_sized_component(
    mask: np.ndarray,
    *,
    min_area_px: int,
    max_area_fraction: float = 0.055,
    max_bbox_width_fraction: float = 0.42,
    max_bbox_height_fraction: float = 0.42,
    locked_center_uv: tuple[float, float] | None = None,
    max_locked_center_distance_px: float = 60.0,
) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=bool)

    height, width = binary.shape
    max_area_px = max(float(min_area_px), float(max_area_fraction) * float(binary.size))
    max_bbox_width_px = max(1.0, float(max_bbox_width_fraction) * float(width))
    max_bbox_height_px = max(1.0, float(max_bbox_height_fraction) * float(height))
    best_mask: np.ndarray | None = None
    best_score = -float("inf")

    for component in extract_components(binary, min_area_px=int(min_area_px)):
        min_u, min_v, max_u, max_v = component.bbox_xyxy
        area = int(component.area_px)
        if area < int(min_area_px):
            continue
        bbox_width = float(max_u - min_u + 1)
        bbox_height = float(max_v - min_v + 1)
        if area > max_area_px or bbox_width > max_bbox_width_px or bbox_height > max_bbox_height_px:
            continue
        center_u = 0.5 * float(min_u + max_u)
        center_v = 0.5 * float(min_v + max_v)
        center_distance = 0.0
        if locked_center_uv is not None:
            center_distance = math.hypot(
                center_u - float(locked_center_uv[0]),
                center_v - float(locked_center_uv[1]),
            )
            if center_distance > float(max_locked_center_distance_px):
                continue
        bbox_area = max(1.0, bbox_width * bbox_height)
        compactness = float(area) / bbox_area
        score = float(area) * (0.35 + 0.65 * compactness)
        if locked_center_uv is not None:
            score -= 10.0 * center_distance
        if score > best_score:
            best_score = score
            best_mask = component.mask

    output = np.zeros(binary.shape, dtype=bool)
    return output if best_mask is None else np.asarray(best_mask, dtype=bool).copy()


def target_sized_components(
    mask: np.ndarray,
    *,
    min_area_px: int,
    max_area_fraction: float = 0.055,
    max_bbox_width_fraction: float = 0.42,
    max_bbox_height_fraction: float = 0.42,
) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    output = np.zeros(binary.shape, dtype=bool)
    if not binary.any():
        return output

    height, width = binary.shape
    max_area_px = max(float(min_area_px), float(max_area_fraction) * float(binary.size))
    max_bbox_width_px = max(1.0, float(max_bbox_width_fraction) * float(width))
    max_bbox_height_px = max(1.0, float(max_bbox_height_fraction) * float(height))

    for component in extract_components(binary, min_area_px=int(min_area_px)):
        min_u, min_v, max_u, max_v = component.bbox_xyxy
        area = int(component.area_px)
        if area < int(min_area_px):
            continue
        bbox_width = float(max_u - min_u + 1)
        bbox_height = float(max_v - min_v + 1)
        if area > max_area_px or bbox_width > max_bbox_width_px or bbox_height > max_bbox_height_px:
            continue
        output |= component.mask
    return output


def add_nonoverlapping_components(
    base_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    min_area_px: int,
    max_overlap_fraction: float,
) -> np.ndarray:
    output = np.asarray(base_mask, dtype=bool).copy()
    for component in extract_components(candidate_mask, min_area_px=max(1, int(min_area_px))):
        overlap = int(np.count_nonzero(output & component.mask))
        if overlap / float(max(1, component.area_px)) <= float(max_overlap_fraction):
            output |= component.mask
    return output


def largest_component(mask: np.ndarray, *, min_area_px: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=bool)

    components = extract_components(binary, min_area_px=int(min_area_px))
    if not components:
        return np.zeros(binary.shape, dtype=bool)
    return np.asarray(max(components, key=lambda component: component.area_px).mask, dtype=bool).copy()


def table_foreground_supported_mask(
    *,
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    min_height_m: float,
    max_height_m: float,
    min_component_area_px: int,
    workspace_min_x_m: float = -10.0,
    workspace_max_x_m: float = 10.0,
    workspace_min_y_m: float = -10.0,
    workspace_max_y_m: float = 10.0,
) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=bool)

    points_base = base_points_from_depth(
        depth_m=depth_m,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
    )
    z_map = points_base[:, :, 2]
    x_map = points_base[:, :, 0]
    y_map = points_base[:, :, 1]
    height_above_table = z_map - float(table_z_m)
    finite_in_workspace = (
        binary
        & np.isfinite(x_map)
        & np.isfinite(y_map)
        & np.isfinite(z_map)
        & (x_map >= float(workspace_min_x_m))
        & (x_map <= float(workspace_max_x_m))
        & (y_map >= float(workspace_min_y_m))
        & (y_map <= float(workspace_max_y_m))
    )
    supported = (
        finite_in_workspace
        & (height_above_table >= float(min_height_m))
        & (height_above_table <= float(max_height_m))
    )
    # depth noise puts stray pixels a few mm above the table everywhere, so a
    # flat table-warp blob always finds a handful of "supported" pixels; a
    # real object must also rise clear of the noise floor somewhere
    core_height_m = max(float(min_height_m), 0.006)
    core = (
        finite_in_workspace
        & (height_above_table >= core_height_m)
        & (height_above_table <= float(max_height_m))
    )

    output = np.zeros(binary.shape, dtype=bool)
    for component in extract_components(binary, min_area_px=1):
        if int(component.area_px) < int(min_component_area_px):
            continue
        support_count = int(np.count_nonzero(supported & component.mask))
        min_support = max(4, min(24, int(round(float(component.area_px) * 0.02))))
        if support_count < min_support:
            continue
        if int(np.count_nonzero(core & component.mask)) < min_support:
            continue
        output |= component.mask
    return output


def next_stable_track_id(tracks: list[Any]) -> int:
    if not tracks:
        return 1
    return max(0, *(int(getattr(track, "track_id", 0)) for track in tracks)) + 1


def quaternion_yaw(orientation: Quaternion) -> float:
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    w = float(orientation.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def smooth_yaw(previous_yaw: float, candidate_yaw: float, alpha: float) -> float:
    delta = (float(candidate_yaw) - float(previous_yaw) + math.pi) % (2.0 * math.pi) - math.pi
    return float(previous_yaw) + float(alpha) * delta


def shape_label(shape_type: int) -> str:
    if int(shape_type) == int(GeometryHypothesis.SHAPE_BOX):
        return "box"
    if int(shape_type) == int(GeometryHypothesis.SHAPE_CYLINDER):
        return "cylinder"
    if int(shape_type) == int(GeometryHypothesis.SHAPE_CUP_LIKE):
        return "cup"
    return "unknown"


def filter_stable_component_mask(
    mask: np.ndarray,
    tracks: list[StableComponentTrack],
    *,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    min_component_area_px: int,
    min_observations: int,
    max_center_jump_m: float,
    max_misses: int,
    min_area_ratio: float = 0.35,
) -> tuple[np.ndarray, list[StableComponentTrack]]:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        retained_tracks = [
            replace(track, misses=int(track.misses) + 1)
            for track in tracks
            if int(track.misses) + 1 <= int(max_misses)
        ]
        return np.zeros(binary.shape, dtype=bool), retained_tracks

    required_observations = max(1, int(min_observations))
    active_tracks = list(tracks)
    matched_track_indices: set[int] = set()
    next_tracks: list[StableComponentTrack] = []
    stable_mask = np.zeros(binary.shape, dtype=bool)
    next_track_id = next_stable_track_id(active_tracks)

    for component in extract_components(binary, min_area_px=max(1, int(min_component_area_px))):
        table_points = table_points_from_mask(
            mask=component.mask,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=float(table_z_m),
        )
        if table_points.shape[0] < 1:
            continue
        center_xy = np.median(table_points[:, 0:2], axis=0)
        area_px = int(component.area_px)
        best_index: int | None = None
        best_distance = float("inf")
        for track_index, track in enumerate(active_tracks):
            if track_index in matched_track_indices:
                continue
            distance = math.hypot(
                float(center_xy[0]) - float(track.center_xy_m[0]),
                float(center_xy[1]) - float(track.center_xy_m[1]),
            )
            area_ratio = min(float(area_px), float(track.area_px)) / max(
                1.0, max(float(area_px), float(track.area_px))
            )
            if (
                distance <= float(max_center_jump_m)
                and area_ratio >= float(min_area_ratio)
                and distance < best_distance
            ):
                best_index = track_index
                best_distance = distance

        if best_index is None:
            track = StableComponentTrack(
                center_xy_m=(float(center_xy[0]), float(center_xy[1])),
                area_px=area_px,
                observations=1,
                misses=0,
                track_id=next_track_id,
            )
            next_track_id += 1
        else:
            previous = active_tracks[best_index]
            matched_track_indices.add(best_index)
            track = StableComponentTrack(
                center_xy_m=(float(center_xy[0]), float(center_xy[1])),
                area_px=area_px,
                observations=int(previous.observations) + 1,
                misses=0,
                track_id=int(previous.track_id),
            )
        next_tracks.append(track)
        if int(track.observations) >= required_observations:
            stable_mask |= component.mask

    for track_index, track in enumerate(active_tracks):
        if track_index in matched_track_indices:
            continue
        missed = replace(track, misses=int(track.misses) + 1)
        if int(missed.misses) <= int(max_misses):
            next_tracks.append(missed)

    return stable_mask, next_tracks


def hypothesis_center_xy_m(hypothesis: GeometryHypothesis) -> tuple[float, float]:
    pose = hypothesis.pose_base.pose
    return float(pose.position.x), float(pose.position.y)


def hypothesis_center_z_m(hypothesis: GeometryHypothesis) -> float:
    return float(hypothesis.pose_base.pose.position.z)


def hypothesis_size_xyz_m(hypothesis: GeometryHypothesis) -> tuple[float, float, float]:
    return (
        float(hypothesis.dimensions_m.x),
        float(hypothesis.dimensions_m.y),
        float(hypothesis.dimensions_m.z),
    )


def smooth_dimension(
    previous: float,
    candidate: float,
    *,
    alpha: float,
    max_step_ratio: float,
) -> float:
    previous_value = float(previous)
    candidate_value = float(candidate)
    if not math.isfinite(candidate_value) or candidate_value < 0.0:
        return max(0.0, previous_value)
    if not math.isfinite(previous_value) or previous_value <= 0.0:
        return max(0.0, candidate_value)
    bounded_candidate = candidate_value
    step_ratio = float(max_step_ratio)
    if math.isfinite(step_ratio) and step_ratio > 0.0:
        max_step = max(1e-6, abs(previous_value) * step_ratio)
        delta = max(-max_step, min(max_step, candidate_value - previous_value))
        bounded_candidate = previous_value + delta
    return max(0.0, (1.0 - float(alpha)) * previous_value + float(alpha) * bounded_candidate)


def should_hold_dimension_expansion(
    previous_size_xyz_m: tuple[float, float, float],
    candidate_size_xyz_m: tuple[float, float, float],
    *,
    expansion_ratio: float = 1.60,
) -> bool:
    ratio = float(expansion_ratio)
    if not math.isfinite(ratio) or ratio <= 1.0:
        return False
    previous_x = max(1e-9, float(previous_size_xyz_m[0]))
    previous_y = max(1e-9, float(previous_size_xyz_m[1]))
    candidate_x = max(0.0, float(candidate_size_xyz_m[0]))
    candidate_y = max(0.0, float(candidate_size_xyz_m[1]))
    if candidate_x > previous_x * ratio or candidate_y > previous_y * ratio:
        return True
    previous_area = previous_x * previous_y
    candidate_area = candidate_x * candidate_y
    return candidate_area > previous_area * ratio * ratio


TRACK_FAST_RELOCK_SHRINK_RATIO = 0.65
TRACK_FAST_RELOCK_FRAMES = 3
TRACK_EXPAND_ESCAPE_FRAMES = 5
STRONG_SHAPE_SCORE = 0.60
STRONG_SHAPE_SWITCH_OBSERVATIONS = 3
SHAPE_VOTE_WINDOW = 5
ESTABLISHED_TRACK_MISS_BONUS_CAP = 6


def neighbor_track_within_candidate_reach(
    candidate_center_xy: tuple[float, float],
    candidate_size_xyz: tuple[float, float, float],
    track: StableHypothesisTrack,
    other_tracks: list[StableHypothesisTrack],
    *,
    max_center_jump_m: float,
) -> bool:
    candidate_max_dim = max(float(candidate_size_xyz[0]), float(candidate_size_xyz[1]))
    neighbor_radius = max(
        0.040,
        1.35 * candidate_max_dim,
        0.8 * float(max_center_jump_m),
    )
    for other in other_tracks:
        if int(other.track_id) == int(track.track_id):
            continue
        distance = math.hypot(
            float(candidate_center_xy[0]) - float(other.center_xy_m[0]),
            float(candidate_center_xy[1]) - float(other.center_xy_m[1]),
        )
        if distance <= neighbor_radius:
            return True
    return False


def hypothesis_candidate_absorbs_neighbor_track(
    candidate_center_xy: tuple[float, float],
    candidate_size_xyz: tuple[float, float, float],
    track: StableHypothesisTrack,
    other_tracks: list[StableHypothesisTrack],
    *,
    max_center_jump_m: float,
) -> bool:
    if not should_hold_dimension_expansion(track.size_xyz_m, candidate_size_xyz):
        return False
    return neighbor_track_within_candidate_reach(
        candidate_center_xy,
        candidate_size_xyz,
        track,
        other_tracks,
        max_center_jump_m=max_center_jump_m,
    )


def apply_stable_track_to_hypothesis(
    hypothesis: GeometryHypothesis,
    track: StableHypothesisTrack,
) -> GeometryHypothesis:
    hypothesis.shape_type = int(track.shape_type)
    hypothesis.hypothesis_id = f"track_{int(track.track_id)}_{shape_label(int(track.shape_type))}"
    hypothesis.pose_base.pose.position.x = float(track.center_xy_m[0])
    hypothesis.pose_base.pose.position.y = float(track.center_xy_m[1])
    hypothesis.pose_base.pose.position.z = float(track.center_z_m)
    hypothesis.pose_base.pose.orientation = yaw_orientation(float(track.yaw_rad))
    hypothesis.dimensions_m.x = max(0.0, float(track.size_xyz_m[0]))
    hypothesis.dimensions_m.y = max(0.0, float(track.size_xyz_m[1]))
    hypothesis.dimensions_m.z = max(0.0, float(track.size_xyz_m[2]))
    top_z = float(track.center_z_m) + 0.5 * max(0.0, float(track.size_xyz_m[2])) + 0.005
    for grasp in hypothesis.grasp_candidates:
        pregrasp_offset = float(grasp.pregrasp_pose.pose.position.z) - float(
            grasp.grasp_pose.pose.position.z
        )
        if not math.isfinite(pregrasp_offset) or abs(pregrasp_offset) < 1e-6:
            pregrasp_offset = 0.080
        grasp.grasp_pose.pose.position.x = float(track.center_xy_m[0])
        grasp.grasp_pose.pose.position.y = float(track.center_xy_m[1])
        grasp.grasp_pose.pose.position.z = top_z
        grasp.pregrasp_pose.pose.position.x = float(track.center_xy_m[0])
        grasp.pregrasp_pose.pose.position.y = float(track.center_xy_m[1])
        grasp.pregrasp_pose.pose.position.z = top_z + pregrasp_offset
    hypothesis.provenance = (
        f"{hypothesis.provenance};stable_track_id={int(track.track_id)};"
        f"stable_shape={shape_label(int(track.shape_type))}"
    )
    if bool(track.dimension_hold):
        hypothesis.provenance = f"{hypothesis.provenance};stable_dimension_hold"
    if bool(track.fast_relock):
        hypothesis.provenance = f"{hypothesis.provenance};stable_fast_relock"
    return hypothesis


def copy_hypothesis_array_with_hypotheses(
    message: GeometryHypothesisArray,
    hypotheses: list[GeometryHypothesis],
) -> GeometryHypothesisArray:
    filtered = GeometryHypothesisArray()
    filtered.header = message.header
    filtered.trial_id = message.trial_id
    filtered.observation_id = message.observation_id
    filtered.backend_name = message.backend_name
    filtered.hypotheses = list(hypotheses)
    return filtered


MIN_HYPOTHESIS_SCORE = 0.05


def drop_low_evidence_hypotheses(
    message: GeometryHypothesisArray,
    *,
    min_score: float = MIN_HYPOTHESIS_SCORE,
) -> GeometryHypothesisArray:
    """Drop hypotheses whose evidence score is noise-level.

    Zero-score silhouette fits come from flying/near-table pixels (IR-degraded
    table zones, hole rims); their outline geometry is meaningless. Deliberate
    hole-existence proxies are kept: the hole itself is their evidence.
    """
    kept = [
        hypothesis
        for hypothesis in message.hypotheses
        if float(hypothesis.score.total) >= float(min_score)
        or "hole_existence_only" in str(hypothesis.provenance)
    ]
    return copy_hypothesis_array_with_hypotheses(message, kept)



def retire_contained_duplicate_tracks(
    message: GeometryHypothesisArray,
    tracks: list[StableHypothesisTrack],
) -> tuple[GeometryHypothesisArray, list[StableHypothesisTrack]]:
    protected_ids: set[int] = set()
    for hypothesis in message.hypotheses:
        if "hole_existence_only" in str(hypothesis.provenance):
            match = re.match(r"track_(\d+)_", str(hypothesis.hypothesis_id))
            if match:
                protected_ids.add(int(match.group(1)))
    """Drop small tracks living inside a bigger track's footprint.

    Two rigid objects cannot share table area, so a track whose center lies
    inside another track's footprint with at most half its area is a fragment
    duplicate (e.g. a glass bowl's rim shard next to the assembled cylinder).
    """
    def overlap_over_min(a: StableHypothesisTrack, b: StableHypothesisTrack) -> float:
        iw = min(
            float(a.center_xy_m[0]) + 0.5 * float(a.size_xyz_m[0]),
            float(b.center_xy_m[0]) + 0.5 * float(b.size_xyz_m[0]),
        ) - max(
            float(a.center_xy_m[0]) - 0.5 * float(a.size_xyz_m[0]),
            float(b.center_xy_m[0]) - 0.5 * float(b.size_xyz_m[0]),
        )
        ih = min(
            float(a.center_xy_m[1]) + 0.5 * float(a.size_xyz_m[1]),
            float(b.center_xy_m[1]) + 0.5 * float(b.size_xyz_m[1]),
        ) - max(
            float(a.center_xy_m[1]) - 0.5 * float(a.size_xyz_m[1]),
            float(b.center_xy_m[1]) - 0.5 * float(b.size_xyz_m[1]),
        )
        if iw <= 0.0 or ih <= 0.0:
            return 0.0
        area_a = max(1e-9, float(a.size_xyz_m[0]) * float(a.size_xyz_m[1]))
        area_b = max(1e-9, float(b.size_xyz_m[0]) * float(b.size_xyz_m[1]))
        return (iw * ih) / min(area_a, area_b)

    drop_ids: set[int] = set()
    for small in tracks:
        if int(small.track_id) in drop_ids or int(small.track_id) in protected_ids:
            continue
        area_small = max(1e-9, float(small.size_xyz_m[0]) * float(small.size_xyz_m[1]))
        for big in tracks:
            if (
                int(big.track_id) == int(small.track_id)
                or int(big.track_id) in drop_ids
                or int(big.observations) < 3
            ):
                continue
            area_big = float(big.size_xyz_m[0]) * float(big.size_xyz_m[1])
            contained = (
                area_small <= 0.5 * area_big
                and abs(float(small.center_xy_m[0]) - float(big.center_xy_m[0]))
                <= 0.5 * float(big.size_xyz_m[0])
                and abs(float(small.center_xy_m[1]) - float(big.center_xy_m[1]))
                <= 0.5 * float(big.size_xyz_m[1])
            )
            # micro slivers (speckle debris on reflective edges) hug a much
            # bigger neighbor without their center falling inside it
            micro = (
                max(float(small.size_xyz_m[0]), float(small.size_xyz_m[1]))
                <= MICRO_TRACK_MAX_DIM_M
                and area_small <= 0.15 * area_big
                and math.hypot(
                    float(small.center_xy_m[0]) - float(big.center_xy_m[0]),
                    float(small.center_xy_m[1]) - float(big.center_xy_m[1]),
                )
                <= 0.5 * max(float(big.size_xyz_m[0]), float(big.size_xyz_m[1]))
                + MICRO_TRACK_ABSORB_MARGIN_M
            )
            # near-equal duplicates: heavy overlap, comparable size -> the
            # younger track is the redundant one
            duplicate = (
                overlap_over_min(small, big) >= 0.60
                and area_small <= 2.0 * area_big
                and int(small.observations) < int(big.observations)
            )
            if contained or duplicate or micro:
                drop_ids.add(int(small.track_id))
                break
    if not drop_ids:
        return message, tracks
    kept_tracks = [t for t in tracks if int(t.track_id) not in drop_ids]
    kept_hyps = [
        h
        for h in message.hypotheses
        if not any(f"track_{d}_" in str(h.hypothesis_id) for d in drop_ids)
    ]
    return copy_hypothesis_array_with_hypotheses(message, kept_hyps), kept_tracks


def held_hypothesis_from_track(
    track: StableHypothesisTrack,
    frame_id: str,
) -> GeometryHypothesis:
    hypothesis = GeometryHypothesis()
    hypothesis.hypothesis_id = f"track_{int(track.track_id)}_{shape_label(int(track.shape_type))}"
    hypothesis.shape_type = int(track.shape_type)
    hypothesis.pose_base = make_pose(
        str(frame_id),
        float(track.center_xy_m[0]),
        float(track.center_xy_m[1]),
        float(track.center_z_m),
        yaw_orientation(float(track.yaw_rad)),
    )
    hypothesis.dimensions_m.x = max(0.0, float(track.size_xyz_m[0]))
    hypothesis.dimensions_m.y = max(0.0, float(track.size_xyz_m[1]))
    hypothesis.dimensions_m.z = max(0.0, float(track.size_xyz_m[2]))
    hypothesis.validation_state = GeometryHypothesis.VALIDATION_VALID
    hypothesis.provenance = (
        f"stable_output_hold;stable_track_id={int(track.track_id)};"
        f"stable_shape={shape_label(int(track.shape_type))};"
        f"stable_misses={int(track.misses)}"
    )
    return hypothesis


def filter_stable_hypothesis_array(
    message: GeometryHypothesisArray,
    tracks: list[StableHypothesisTrack],
    *,
    min_observations: int,
    max_center_jump_m: float,
    max_misses: int,
    shape_switch_observations: int = 3,
    smoothing_alpha: float = 0.45,
    dimension_smoothing_alpha: float | None = None,
    dimension_max_step_ratio: float = 0.0,
    output_hold_misses: int = 0,
) -> tuple[GeometryHypothesisArray, list[StableHypothesisTrack]]:
    required_observations = max(1, int(min_observations))
    required_shape_switch = max(1, int(shape_switch_observations))
    alpha = max(0.0, min(1.0, float(smoothing_alpha)))
    dimension_alpha = alpha if dimension_smoothing_alpha is None else float(
        dimension_smoothing_alpha
    )
    dimension_alpha = max(0.0, min(1.0, dimension_alpha))
    dimension_step_ratio = max(0.0, float(dimension_max_step_ratio))
    active_tracks = list(tracks)
    matched_track_indices: set[int] = set()
    retired_track_indices: set[int] = set()
    next_tracks: list[StableHypothesisTrack] = []
    stable_hypotheses: list[GeometryHypothesis] = []
    next_track_id = next_stable_track_id(active_tracks)

    for hypothesis in message.hypotheses:
        center_xy = hypothesis_center_xy_m(hypothesis)
        center_z = hypothesis_center_z_m(hypothesis)
        size_xyz = hypothesis_size_xyz_m(hypothesis)
        candidate_shape_type = int(hypothesis.shape_type)
        candidate_yaw = quaternion_yaw(hypothesis.pose_base.pose.orientation)
        best_index: int | None = None
        best_distance = float("inf")
        for track_index, track in enumerate(active_tracks):
            if track_index in matched_track_indices:
                continue
            distance = math.hypot(
                center_xy[0] - float(track.center_xy_m[0]),
                center_xy[1] - float(track.center_xy_m[1]),
            )
            if distance > float(max_center_jump_m) or distance >= best_distance:
                continue
            if hypothesis_candidate_absorbs_neighbor_track(
                center_xy,
                size_xyz,
                track,
                active_tracks,
                max_center_jump_m=max_center_jump_m,
            ):
                # A candidate that would balloon this track while another track
                # sits inside its footprint reach is merged evidence; let it
                # match the larger owner instead of stealing this one.
                continue
            best_distance = distance
            best_index = track_index

        if best_index is None:
            ring_certified = "transparent_ring_cylinder" in str(hypothesis.provenance)
            covered_established = 0
            covered_indices: list[int] = []
            for track_index, track in enumerate(active_tracks):
                if int(track.observations) < 3:
                    continue
                if (
                    abs(center_xy[0] - float(track.center_xy_m[0]))
                    <= 0.5 * float(size_xyz[0]) + 0.010
                    and abs(center_xy[1] - float(track.center_xy_m[1]))
                    <= 0.5 * float(size_xyz[1]) + 0.010
                ):
                    covered_established += 1
                    covered_indices.append(track_index)
            if covered_established >= 2 and not ring_certified:
                # a candidate whose footprint blankets two established objects
                # is merged evidence; it must never become a track of its own.
                # Exception: a ring-certified transparent cylinder legitimately
                # replaces its own two rim-arc tracks.
                continue
            if ring_certified:
                # the rim-arc tracks are this same object: retire them now
                # instead of letting them starve next to the new cylinder
                retired_track_indices.update(covered_indices)
            track = StableHypothesisTrack(
                shape_type=candidate_shape_type,
                center_xy_m=(float(center_xy[0]), float(center_xy[1])),
                observations=1,
                misses=0,
                center_z_m=float(center_z),
                size_xyz_m=size_xyz,
                yaw_rad=float(candidate_yaw),
                pending_shape_type=None,
                pending_shape_observations=0,
                track_id=next_track_id,
            )
            next_track_id += 1
        else:
            previous = active_tracks[best_index]
            matched_track_indices.add(best_index)
            shape_type = int(previous.shape_type)
            candidate_score = float(hypothesis.score.total)
            # sliding-window majority vote: a shape that wins most of the last
            # SHAPE_VOTE_WINDOW frames flips the track even if interrupted by
            # the occasional contrary frame (consecutive counting never
            # converged on a cylinder classified box every third frame)
            shape_votes = (tuple(previous.shape_votes) + (candidate_shape_type,))[
                -SHAPE_VOTE_WINDOW:
            ]
            weak_required = max(2, min(4, required_shape_switch))
            required_votes = (
                min(weak_required, STRONG_SHAPE_SWITCH_OBSERVATIONS)
                if candidate_score >= STRONG_SHAPE_SCORE
                else weak_required
            )
            vote_count = shape_votes.count(candidate_shape_type)
            pending_shape_type: int | None = None
            pending_shape_observations = 0
            if candidate_shape_type != shape_type:
                if vote_count >= required_votes:
                    shape_type = candidate_shape_type
                    shape_votes = tuple(
                        vote for vote in shape_votes if vote == candidate_shape_type
                    )
                else:
                    pending_shape_type = candidate_shape_type
                    pending_shape_observations = vote_count

            previous_size = previous.size_xyz_m
            candidate_max_dim = max(float(size_xyz[0]), float(size_xyz[1]))
            previous_max_dim = max(
                float(previous_size[0]), float(previous_size[1]), 1e-9
            )
            if candidate_max_dim <= TRACK_FAST_RELOCK_SHRINK_RATIO * previous_max_dim:
                shrink_streak = int(previous.shrink_streak) + 1
            else:
                # decay instead of reset: intermittent merged frames must not
                # erase accumulated much-smaller evidence
                shrink_streak = max(0, int(previous.shrink_streak) - 1)
            dimension_hold = should_hold_dimension_expansion(previous_size, size_xyz)
            expand_streak = int(previous.expand_streak) + 1 if dimension_hold else 0
            fast_relock = shrink_streak >= TRACK_FAST_RELOCK_FRAMES
            if (
                not fast_relock
                and dimension_hold
                and expand_streak >= TRACK_EXPAND_ESCAPE_FRAMES
                and not neighbor_track_within_candidate_reach(
                    center_xy,
                    size_xyz,
                    previous,
                    active_tracks,
                    max_center_jump_m=max_center_jump_m,
                )
            ):
                # the "merged evidence" suspicion did not pan out: the larger
                # size persists and no neighbor explains it — accept reality
                fast_relock = True
                dimension_hold = False
            smoothed_center_xy = (
                (1.0 - alpha) * float(previous.center_xy_m[0]) + alpha * float(center_xy[0]),
                (1.0 - alpha) * float(previous.center_xy_m[1]) + alpha * float(center_xy[1]),
            )
            if fast_relock:
                # consistent contradicting evidence owns the geometry: snap to
                # the measurement instead of unwinding a stale/poisoned track
                # state over tens of seconds
                smoothed_center_xy = (float(center_xy[0]), float(center_xy[1]))
                smoothed_size = size_xyz
                smoothed_yaw = float(candidate_yaw)
                shape_type = candidate_shape_type
                pending_shape_type = None
                pending_shape_observations = 0
                shape_votes = (candidate_shape_type,)
                shrink_streak = 0
                expand_streak = 0
                dimension_hold = False
            elif dimension_hold:
                # Sudden footprint ballooning is merged/hole-contaminated
                # evidence, not the object growing: keep the track's size/yaw.
                smoothed_size = previous_size
                smoothed_yaw = float(previous.yaw_rad)
            else:
                smoothed_size = (
                    smooth_dimension(
                        previous_size[0],
                        size_xyz[0],
                        alpha=dimension_alpha,
                        max_step_ratio=dimension_step_ratio,
                    ),
                    smooth_dimension(
                        previous_size[1],
                        size_xyz[1],
                        alpha=dimension_alpha,
                        max_step_ratio=dimension_step_ratio,
                    ),
                    smooth_dimension(
                        previous_size[2],
                        size_xyz[2],
                        alpha=dimension_alpha,
                        max_step_ratio=dimension_step_ratio,
                    ),
                )
                smoothed_yaw = smooth_yaw(float(previous.yaw_rad), float(candidate_yaw), alpha)
            track = StableHypothesisTrack(
                shape_type=shape_type,
                center_xy_m=smoothed_center_xy,
                observations=int(previous.observations) + 1,
                misses=0,
                center_z_m=(1.0 - alpha) * float(previous.center_z_m) + alpha * float(center_z),
                size_xyz_m=smoothed_size,
                yaw_rad=smoothed_yaw,
                pending_shape_type=pending_shape_type,
                pending_shape_observations=pending_shape_observations,
                track_id=int(previous.track_id),
                dimension_hold=dimension_hold,
                shrink_streak=shrink_streak,
                expand_streak=expand_streak,
                fast_relock=fast_relock,
                shape_votes=shape_votes,
            )
        next_tracks.append(track)
        if int(track.observations) >= required_observations:
            hypothesis = apply_stable_track_to_hypothesis(hypothesis, track)
            hypothesis.provenance = (
                f"{hypothesis.provenance};stable_observations={track.observations};"
                f"stable_pending_shape={shape_label(int(track.pending_shape_type)) if track.pending_shape_type is not None else 'none'};"
                f"stable_pending_shape_observations={int(track.pending_shape_observations)}"
            )
            stable_hypotheses.append(hypothesis)

    def _tracks_overlap_ratio(a: StableHypothesisTrack, b: StableHypothesisTrack) -> float:
        iw = min(
            float(a.center_xy_m[0]) + 0.5 * float(a.size_xyz_m[0]),
            float(b.center_xy_m[0]) + 0.5 * float(b.size_xyz_m[0]),
        ) - max(
            float(a.center_xy_m[0]) - 0.5 * float(a.size_xyz_m[0]),
            float(b.center_xy_m[0]) - 0.5 * float(b.size_xyz_m[0]),
        )
        ih = min(
            float(a.center_xy_m[1]) + 0.5 * float(a.size_xyz_m[1]),
            float(b.center_xy_m[1]) + 0.5 * float(b.size_xyz_m[1]),
        ) - max(
            float(a.center_xy_m[1]) - 0.5 * float(a.size_xyz_m[1]),
            float(b.center_xy_m[1]) - 0.5 * float(b.size_xyz_m[1]),
        )
        if iw <= 0.0 or ih <= 0.0:
            return 0.0
        area_a = max(1e-9, float(a.size_xyz_m[0]) * float(a.size_xyz_m[1]))
        area_b = max(1e-9, float(b.size_xyz_m[0]) * float(b.size_xyz_m[1]))
        return (iw * ih) / min(area_a, area_b)

    matched_tracks_updated = [
        track for track in next_tracks if int(track.misses) == 0
    ]
    for track_index, track in enumerate(active_tracks):
        if track_index in matched_track_indices or track_index in retired_track_indices:
            continue
        # two objects cannot share table area: an unmatched track whose
        # footprint lies mostly inside a freshly-updated track is a stale
        # duplicate (fragment ping-pong) and retires instead of starving
        if any(
            int(keeper.observations) >= 3
            and _tracks_overlap_ratio(track, keeper) >= 0.5
            for keeper in matched_tracks_updated
        ):
            continue
        missed = StableHypothesisTrack(
            shape_type=int(track.shape_type),
            center_xy_m=track.center_xy_m,
            observations=int(track.observations),
            misses=int(track.misses) + 1,
            center_z_m=float(track.center_z_m),
            size_xyz_m=track.size_xyz_m,
            yaw_rad=float(track.yaw_rad),
            pending_shape_type=track.pending_shape_type,
            pending_shape_observations=int(track.pending_shape_observations),
            track_id=int(track.track_id),
            shape_votes=tuple(track.shape_votes),
        )
        # a long-established object earns a longer miss budget: intermittent
        # detections (small clutter at the workspace edge) keep their identity
        # instead of dying and being reborn under a new id every gap
        miss_budget = int(max_misses) + min(
            ESTABLISHED_TRACK_MISS_BONUS_CAP, int(track.observations) // 5
        )
        if int(missed.misses) <= miss_budget:
            next_tracks.append(missed)
            if (
                int(output_hold_misses) > 0
                and int(missed.misses) <= int(output_hold_misses)
                and int(missed.observations) >= max(3, required_observations)
            ):
                # An established object that briefly loses evidence (typically
                # swallowed by a depth hole) stays on display at its last
                # geometry instead of blinking.
                stable_hypotheses.append(
                    held_hypothesis_from_track(missed, message.header.frame_id)
                )

    return copy_hypothesis_array_with_hypotheses(message, stable_hypotheses), next_tracks


def evidence_from_depth_and_mask(depth_m: np.ndarray, target_mask: np.ndarray) -> EvidenceMaps:
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(target_mask, dtype=bool)
    valid = np.isfinite(depth) & (depth > 0.0)
    hole = mask & ~valid
    foreground = mask & valid
    zeros = np.zeros(mask.shape, dtype=np.float32)
    return EvidenceMaps(
        valid=(valid & mask).astype(np.float32),
        hole=hole.astype(np.float32),
        table_leakage=zeros.copy(),
        edge=zeros.copy(),
        flying_point=zeros.copy(),
        foreground_support=foreground.astype(np.float32),
    )


def metric_pose_from_mask(mask: np.ndarray, depth_m: np.ndarray, intrinsics: CameraModel) -> tuple[float, float, float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return 0.0, 0.0, 0.0
    center_u = float(cols.mean())
    center_v = float(rows.mean())
    mask_depth = np.asarray(depth_m, dtype=np.float32)[mask]
    valid_depth = mask_depth[np.isfinite(mask_depth) & (mask_depth > 0.0)]
    depth = float(np.median(valid_depth)) if valid_depth.size else 1.0
    x = ((center_u - float(intrinsics.cx)) / float(intrinsics.fx)) * depth
    y = ((center_v - float(intrinsics.cy)) / float(intrinsics.fy)) * depth
    return x, y, depth


def mask_center_pixel(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return 0.0, 0.0
    return float(cols.mean()), float(rows.mean())


def mask_bbox_pixel_edges(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return (
        float(cols.min()) - 0.5,
        float(cols.max()) + 0.5,
        float(rows.min()) - 0.5,
        float(rows.max()) + 0.5,
    )


def mask_rectangular_support(mask: np.ndarray) -> tuple[float, float]:
    binary = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(binary)
    if rows.size == 0:
        return 0.0, 0.0
    min_u = int(cols.min())
    max_u = int(cols.max())
    min_v = int(rows.min())
    max_v = int(rows.max())
    width = max_u - min_u + 1
    height = max_v - min_v + 1
    bbox = binary[min_v : max_v + 1, min_u : max_u + 1]
    fill_ratio = float(bbox.mean()) if bbox.size else 0.0
    corner_size = max(2, int(round(min(width, height) * 0.30)))
    corner_windows = (
        bbox[:corner_size, :corner_size],
        bbox[:corner_size, -corner_size:],
        bbox[-corner_size:, :corner_size],
        bbox[-corner_size:, -corner_size:],
    )
    corner_support = float(np.mean([window.mean() for window in corner_windows if window.size]))
    return clamp(fill_ratio, 0.0, 1.0), clamp(corner_support, 0.0, 1.0)


def expanded_bbox_mask(mask: np.ndarray, *, radius_px: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(binary)
    output = np.zeros(binary.shape, dtype=bool)
    if rows.size == 0:
        return output
    radius = max(0, int(radius_px))
    min_row = max(0, int(rows.min()) - radius)
    max_row = min(binary.shape[0], int(rows.max()) + radius + 1)
    min_col = max(0, int(cols.min()) - radius)
    max_col = min(binary.shape[1], int(cols.max()) + radius + 1)
    output[min_row:max_row, min_col:max_col] = True
    return output


def normalize_yaw_half_turn(yaw_rad: float) -> float:
    yaw = float(yaw_rad)
    while yaw <= -0.5 * math.pi:
        yaw += math.pi
    while yaw > 0.5 * math.pi:
        yaw -= math.pi
    return yaw


def canonicalize_square_yaw(yaw_rad: float) -> float:
    yaw = normalize_yaw_half_turn(yaw_rad)
    quarter_turn = 0.5 * math.pi
    while yaw <= -0.25 * math.pi:
        yaw += quarter_turn
    while yaw > 0.25 * math.pi:
        yaw -= quarter_turn
    return yaw


def stabilize_weak_square_yaw(yaw_rad: float, *, axis_snap_deg: float = 18.0) -> tuple[float, bool]:
    yaw = canonicalize_square_yaw(yaw_rad)
    if abs(float(yaw)) <= math.radians(float(axis_snap_deg)):
        return 0.0, True
    return yaw, False


def principal_yaw_from_mask(mask: np.ndarray, *, min_anisotropy: float = 1.70) -> float | None:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size < 8:
        return None
    points = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    values, vectors = np.linalg.eigh(covariance)
    if float(values[1]) <= 1e-9:
        return None
    anisotropy = math.sqrt(max(float(values[1]), 1e-9) / max(float(values[0]), 1e-9))
    if anisotropy < float(min_anisotropy):
        return None
    axis = vectors[:, int(np.argmax(values))]
    return normalize_yaw_half_turn(math.atan2(float(axis[1]), float(axis[0])))


def oriented_mask_size(mask: np.ndarray, yaw_rad: float) -> tuple[float, float]:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return 1.0, 1.0
    center_u, center_v = mask_center_pixel(mask)
    du = cols.astype(np.float64) - float(center_u)
    dv = rows.astype(np.float64) - float(center_v)
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    local_u = cos_yaw * du + sin_yaw * dv
    local_v = -sin_yaw * du + cos_yaw * dv
    return (
        max(1.0, float(local_u.max() - local_u.min() + 1.0)),
        max(1.0, float(local_v.max() - local_v.min() + 1.0)),
    )


def generate_live_hypotheses_with_priors(mask: np.ndarray, config: GhostMGGV0Config):
    candidates = generate_local_hypotheses(
        mask,
        shape_types=config.shape_types,
        scale_factors=config.scale_factors,
        depth_m=config.depth_m,
        height_m=config.height_m,
    )
    if not candidates:
        return []
    yaw_rad = principal_yaw_from_mask(mask)
    if yaw_rad is not None and abs(yaw_rad) > math.radians(7.5) and "box" in config.shape_types:
        center_uv = mask_center_pixel(mask)
        oriented_width_px, oriented_height_px = oriented_mask_size(mask, yaw_rad)
        for scale in config.scale_factors:
            scale_value = float(scale)
            candidates.append(
                PrimitiveHypothesis(
                    hypothesis_id=f"box_yaw{int(round(math.degrees(yaw_rad))):+03d}_s{scale_value:.2f}",
                    shape_type="box",
                    center_uv=center_uv,
                    size_px=(oriented_width_px * scale_value, oriented_height_px * scale_value),
                    depth_m=float(config.depth_m),
                    height_m=float(config.height_m),
                    yaw_rad=float(yaw_rad),
                )
            )
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    bbox_width = max(1.0, float(cols.max() - cols.min() + 1))
    bbox_height = max(1.0, float(rows.max() - rows.min() + 1))
    fill_ratio, corner_support = mask_rectangular_support(mask)
    circular_fill_affinity = clamp(1.0 - abs(fill_ratio - 0.785) / 0.22, 0.0, 1.0)
    non_circular_visible_blob = clamp((0.72 - fill_ratio) / 0.22, 0.0, 1.0)
    adjusted = []
    for candidate in candidates:
        scale_x = float(candidate.size_px[0]) / bbox_width
        scale_y = float(candidate.size_px[1]) / bbox_height
        scale_error = abs(0.5 * (scale_x + scale_y) - 1.0)
        if candidate.shape_type == "box":
            shape_prior = 0.35 * corner_support + 0.15 * fill_ratio + 0.12 * non_circular_visible_blob
            if abs(float(getattr(candidate, "yaw_rad", 0.0))) > math.radians(7.5):
                shape_prior += 0.12
        elif candidate.shape_type == "cylinder":
            shape_prior = (
                0.30 * (1.0 - corner_support) * circular_fill_affinity
                + 0.05 * fill_ratio
                - 0.16 * non_circular_visible_blob
            )
        else:
            shape_prior = 0.0
        prior = shape_prior - 0.45 * scale_error
        adjusted.append(replace(candidate, prior_score=float(prior)))
    return adjusted


def generate_live_table_anchored_hypotheses(
    *,
    mask: np.ndarray,
    evidence: EvidenceMaps,
    metric_footprint: TableFootprint,
    table_z_m: float,
    config: GhostMGGV0Config,
    intrinsics: CameraModel | None = None,
    camera_to_base_transform: TransformStamped | None = None,
    min_size_xy_m: float = 0.005,
) -> list[PrimitiveHypothesis]:
    fill_ratio, corner_support = mask_rectangular_support(mask)
    foreground_ratio = float(np.mean(evidence.foreground_support[np.asarray(mask, dtype=bool)]))
    visible_foreground = clamp((foreground_ratio - 0.02) / 0.25, 0.0, 1.0)
    circular_fill_affinity = clamp(1.0 - abs(fill_ratio - 0.785) / 0.22, 0.0, 1.0)
    non_circular_visible_blob = visible_foreground * clamp((0.72 - fill_ratio) / 0.22, 0.0, 1.0)
    footprint_aspect = 1.0
    if metric_footprint.size_x_m > 0.0 and metric_footprint.size_y_m > 0.0:
        footprint_aspect = max(metric_footprint.size_x_m, metric_footprint.size_y_m) / max(
            min(metric_footprint.size_x_m, metric_footprint.size_y_m),
            1e-9,
        )
    near_square_foreground = visible_foreground * clamp((1.35 - footprint_aspect) / 0.35, 0.0, 1.0)
    square_visible_evidence = near_square_foreground * clamp(
        (0.88 - circular_fill_affinity) / 0.35,
        0.0,
        1.0,
    )
    if foreground_ratio >= 0.02:
        yaw_candidates = (float(metric_footprint.yaw_rad),)
        center_refinements = (
            CenterRefinement(
                suffix="",
                center_xy_m=metric_footprint.center_xy_m,
                center_uv_delta=None,
            ),
        )
    else:
        yaw_candidates = table_anchored_yaw_candidates(metric_footprint.yaw_rad)
        center_refinements = table_center_refinement_candidates(
            metric_footprint=metric_footprint,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=float(table_z_m),
        )
    include_yaw_in_ids = len(yaw_candidates) > 1
    base_candidates = generate_table_anchored_hypotheses(
        mask,
        evidence,
        footprint_center_xy_m=metric_footprint.center_xy_m,
        footprint_size_xy_m=(metric_footprint.size_x_m, metric_footprint.size_y_m),
        table_z_m=float(table_z_m),
        height_priors_m=(float(config.height_m),),
        shape_types=tuple(config.shape_types),
        yaw_rads=yaw_candidates,
    )
    if not base_candidates:
        return []

    scale_penalty_weight = 0.45 + 4.0 * clamp(foreground_ratio, 0.0, 1.0) * (
        1.0 - square_visible_evidence
    )
    candidates: list[PrimitiveHypothesis] = []
    for candidate in base_candidates:
        for refinement in center_refinements:
            center_uv = candidate.center_uv
            if refinement.center_uv_delta is not None:
                center_uv = (
                    float(candidate.center_uv[0]) + float(refinement.center_uv_delta[0]),
                    float(candidate.center_uv[1]) + float(refinement.center_uv_delta[1]),
                )
            for scale in config.scale_factors:
                scale_value = float(scale)
                size_px = (
                    max(1.0, float(candidate.size_px[0]) * scale_value),
                    max(1.0, float(candidate.size_px[1]) * scale_value),
                )
                size_xy_m = candidate.size_xy_m
                if foreground_ratio < 0.02 and size_xy_m is not None:
                    size_xy_m = (
                        max(float(min_size_xy_m), float(size_xy_m[0]) * scale_value),
                        max(float(min_size_xy_m), float(size_xy_m[1]) * scale_value),
                    )
                scale_error = abs(scale_value - 1.0)
                expansion_error = max(0.0, scale_value - 1.0)
                if candidate.shape_type == "box":
                    shape_prior = (
                        0.35 * corner_support
                        + 0.15 * fill_ratio
                        + 0.14 * non_circular_visible_blob
                        + 0.05 * near_square_foreground
                        + 0.06 * visible_foreground * corner_support
                        + 0.65 * square_visible_evidence
                    )
                    yaw_rad = float(candidate.yaw_rad)
                else:
                    shape_prior = (
                        0.30 * (1.0 - corner_support) * circular_fill_affinity
                        + 0.05 * fill_ratio
                        - 0.20 * non_circular_visible_blob
                        - 0.06 * visible_foreground * corner_support
                        - 0.25 * square_visible_evidence
                    )
                    yaw_rad = 0.0
                yaw_suffix = ""
                if include_yaw_in_ids and abs(yaw_rad) >= 1e-6:
                    yaw_suffix = f"_yaw{int(round(math.degrees(yaw_rad))):+04d}"
                candidates.append(
                    replace(
                        candidate,
                        hypothesis_id=f"{candidate.shape_type}{yaw_suffix}{refinement.suffix}_s{scale_value:.2f}",
                        center_uv=center_uv,
                        size_px=size_px,
                        yaw_rad=yaw_rad,
                        center_xy_m=refinement.center_xy_m,
                        size_xy_m=size_xy_m,
                        prior_score=float(
                            shape_prior
                            - scale_penalty_weight * scale_error
                            - 10.0 * visible_foreground * expansion_error
                        ),
                    )
                )
    return candidates


def table_center_refinement_candidates(
    *,
    metric_footprint: TableFootprint,
    intrinsics: CameraModel | None,
    camera_to_base_transform: TransformStamped | None,
    table_z_m: float,
) -> tuple[CenterRefinement, ...]:
    center_x, center_y = metric_footprint.center_xy_m
    smaller_size = min(float(metric_footprint.size_x_m), float(metric_footprint.size_y_m))
    step_m = clamp(0.15 * smaller_size, 0.0025, 0.0045)
    offset_values = (
        (0.0, 0.0),
        (step_m, 0.0),
        (-step_m, 0.0),
        (0.0, step_m),
        (0.0, -step_m),
        (step_m, step_m),
        (step_m, -step_m),
        (-step_m, step_m),
        (-step_m, -step_m),
    )
    candidates: list[CenterRefinement] = []
    for dx_m, dy_m in offset_values:
        refined_xy = (float(center_x + dx_m), float(center_y + dy_m))
        refined_uv_delta = None
        if intrinsics is not None and camera_to_base_transform is not None:
            base_uv = project_table_xy_to_pixel(
                center_xy_m=metric_footprint.center_xy_m,
                intrinsics=intrinsics,
                camera_to_base_transform=camera_to_base_transform,
                table_z_m=float(table_z_m),
            )
            refined_uv = project_table_xy_to_pixel(
                center_xy_m=refined_xy,
                intrinsics=intrinsics,
                camera_to_base_transform=camera_to_base_transform,
                table_z_m=float(table_z_m),
            )
            if base_uv is None or refined_uv is None:
                continue
            refined_uv_delta = (
                float(refined_uv[0]) - float(base_uv[0]),
                float(refined_uv[1]) - float(base_uv[1]),
            )
        suffix = ""
        if abs(dx_m) >= 1e-9 or abs(dy_m) >= 1e-9:
            suffix = f"_dx{int(round(dx_m * 1000.0)):+04d}_dy{int(round(dy_m * 1000.0)):+04d}"
        candidates.append(
            CenterRefinement(
                suffix=suffix,
                center_xy_m=refined_xy,
                center_uv_delta=refined_uv_delta,
            )
        )
    return tuple(candidates) if candidates else (
        CenterRefinement(suffix="", center_xy_m=metric_footprint.center_xy_m, center_uv_delta=None),
    )


def project_table_xy_to_pixel(
    *,
    center_xy_m: tuple[float, float],
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
) -> tuple[float, float] | None:
    point_base = np.array([float(center_xy_m[0]), float(center_xy_m[1]), float(table_z_m)], dtype=np.float64)
    rotation = quaternion_to_rotation_matrix(camera_to_base_transform.transform.rotation)
    translation = transform_translation_vector(camera_to_base_transform)
    point_camera = (point_base - translation) @ rotation
    if not np.all(np.isfinite(point_camera)) or float(point_camera[2]) <= 0.0:
        return None
    u = float(intrinsics.fx) * float(point_camera[0]) / float(point_camera[2]) + float(intrinsics.cx)
    v = float(intrinsics.fy) * float(point_camera[1]) / float(point_camera[2]) + float(intrinsics.cy)
    if not math.isfinite(u) or not math.isfinite(v):
        return None
    return (u, v)


def table_anchored_yaw_candidates(base_yaw_rad: float) -> tuple[float, ...]:
    base = normalize_yaw_half_turn(float(base_yaw_rad))
    offsets_deg = (0.0, -45.0, 45.0, -22.5, 22.5, -67.5, 67.5, 90.0)
    values: list[float] = []
    for offset_deg in offsets_deg:
        candidate = normalize_yaw_half_turn(base + math.radians(offset_deg))
        if not any(abs(normalize_yaw_half_turn(candidate - existing)) < math.radians(0.5) for existing in values):
            values.append(candidate)
    return tuple(values)


def metric_dimensions_from_hypothesis(hypothesis, intrinsics: CameraModel) -> tuple[float, float, float]:
    width_px, height_px = hypothesis.size_px
    depth = float(hypothesis.depth_m)
    width_m = max(0.005, float(width_px) / float(intrinsics.fx) * depth)
    depth_m = max(0.005, float(height_px) / float(intrinsics.fy) * depth)
    height_m = max(0.005, float(hypothesis.height_m))
    return width_m, depth_m, height_m


def clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def quaternion_to_rotation_matrix(quaternion: Quaternion) -> np.ndarray:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return np.eye(3, dtype=np.float64)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_translation_vector(transform: TransformStamped) -> np.ndarray:
    return np.array(
        [
            float(transform.transform.translation.x),
            float(transform.transform.translation.y),
            float(transform.transform.translation.z),
        ],
        dtype=np.float64,
    )


def transform_point(point: np.ndarray, transform: TransformStamped) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(transform.transform.rotation)
    return rotation @ np.asarray(point, dtype=np.float64) + transform_translation_vector(transform)


def transform_direction(direction: np.ndarray, transform: TransformStamped) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(transform.transform.rotation)
    return rotation @ np.asarray(direction, dtype=np.float64)


def table_anchored_base_center(
    *,
    mask: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    height_m: float,
) -> np.ndarray | None:
    center_u, center_v = mask_center_pixel(mask)
    ray_camera = np.array(
        [
            (center_u - float(intrinsics.cx)) / float(intrinsics.fx),
            (center_v - float(intrinsics.cy)) / float(intrinsics.fy),
            1.0,
        ],
        dtype=np.float64,
    )
    origin_base = transform_translation_vector(camera_to_base_transform)
    ray_base = transform_direction(ray_camera, camera_to_base_transform)
    if abs(float(ray_base[2])) < 1e-9:
        return None
    distance_scale = (float(table_z_m) - float(origin_base[2])) / float(ray_base[2])
    if distance_scale <= 0.0:
        return None
    table_point = origin_base + distance_scale * ray_base
    return np.array(
        [
            float(table_point[0]),
            float(table_point[1]),
            float(table_z_m) + 0.5 * float(height_m),
        ],
        dtype=np.float64,
    )


def table_intersection_for_pixel(
    *,
    u: float,
    v: float,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
) -> np.ndarray | None:
    ray_camera = np.array(
        [
            (float(u) - float(intrinsics.cx)) / float(intrinsics.fx),
            (float(v) - float(intrinsics.cy)) / float(intrinsics.fy),
            1.0,
        ],
        dtype=np.float64,
    )
    origin_base = transform_translation_vector(camera_to_base_transform)
    ray_base = transform_direction(ray_camera, camera_to_base_transform)
    if abs(float(ray_base[2])) < 1e-9:
        return None
    distance_scale = (float(table_z_m) - float(origin_base[2])) / float(ray_base[2])
    if distance_scale <= 0.0:
        return None
    return origin_base + distance_scale * ray_base


def robust_oriented_bounds_xy(
    points_xy: np.ndarray,
    *,
    robust_percentile: float = 5.0,
    angle_step_deg: float = 3.0,
    max_points: int = 256,
) -> tuple[np.ndarray, float, float, float]:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("points_xy must be a non-empty Nx2 array")
    if points.shape[0] <= 768:
        return _exhaustive_robust_oriented_bounds_xy(
            points,
            robust_percentile=robust_percentile,
            angle_step_deg=min(float(angle_step_deg), 0.5),
        )
    if points.shape[0] > int(max_points):
        indices = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=int)
        points = points[indices]
    percentile = float(robust_percentile)
    local_step_rad = math.radians(float(angle_step_deg))
    if local_step_rad <= 0.0 or not math.isfinite(local_step_rad):
        raise ValueError("angle_step_deg must be positive and finite")

    candidate_yaws = {0.0, math.radians(45.0), math.radians(-45.0)}
    if points.shape[0] >= 3:
        centered = points - np.mean(points, axis=0, keepdims=True)
        covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        principal = math.atan2(float(axis[1]), float(axis[0]))
        candidate_yaws.update(
            {
                principal,
                principal + 0.5 * math.pi,
                principal - 0.5 * math.pi,
            }
        )
    expanded_yaws: set[float] = set()
    for seed in candidate_yaws:
        for offset in (-1, 0, 1):
            expanded_yaws.add(normalize_yaw_half_turn(float(seed) + float(offset) * local_step_rad))

    best = None
    for yaw_rad in sorted(expanded_yaws):
        cos_yaw = math.cos(float(yaw_rad))
        sin_yaw = math.sin(float(yaw_rad))
        local_x = cos_yaw * points[:, 0] + sin_yaw * points[:, 1]
        local_y = -sin_yaw * points[:, 0] + cos_yaw * points[:, 1]
        lower_x, upper_x = np.percentile(local_x, [percentile, 100.0 - percentile])
        lower_y, upper_y = np.percentile(local_y, [percentile, 100.0 - percentile])
        width = max(0.0, float(upper_x - lower_x))
        depth = max(0.0, float(upper_y - lower_y))
        area = width * depth
        if best is None or area < best[0]:
            best = (area, float(yaw_rad), float(lower_x), float(upper_x), float(lower_y), float(upper_y))

    _, yaw_rad, lower_x, upper_x, lower_y, upper_y = best
    width = float(upper_x - lower_x)
    depth = float(upper_y - lower_y)
    if width < depth:
        yaw_rad += 0.5 * math.pi
    yaw_rad = normalize_yaw_half_turn(yaw_rad)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    local_x = cos_yaw * points[:, 0] + sin_yaw * points[:, 1]
    local_y = -sin_yaw * points[:, 0] + cos_yaw * points[:, 1]
    lower_x, upper_x = np.percentile(local_x, [percentile, 100.0 - percentile])
    lower_y, upper_y = np.percentile(local_y, [percentile, 100.0 - percentile])
    width = max(0.0, float(upper_x - lower_x))
    depth = max(0.0, float(upper_y - lower_y))
    center_local_x = 0.5 * (float(lower_x) + float(upper_x))
    center_local_y = 0.5 * (float(lower_y) + float(upper_y))
    center = np.array(
        [
            cos_yaw * center_local_x - sin_yaw * center_local_y,
            sin_yaw * center_local_x + cos_yaw * center_local_y,
        ],
        dtype=np.float64,
    )
    return center, float(width), float(depth), float(yaw_rad)


def _exhaustive_robust_oriented_bounds_xy(
    points_xy: np.ndarray,
    *,
    robust_percentile: float,
    angle_step_deg: float,
) -> tuple[np.ndarray, float, float, float]:
    points = np.asarray(points_xy, dtype=np.float64)
    percentile = float(robust_percentile)
    step_rad = math.radians(float(angle_step_deg))
    best = None
    for yaw_rad in np.arange(-0.5 * math.pi, 0.5 * math.pi, step_rad):
        cos_yaw = math.cos(float(yaw_rad))
        sin_yaw = math.sin(float(yaw_rad))
        local_x = cos_yaw * points[:, 0] + sin_yaw * points[:, 1]
        local_y = -sin_yaw * points[:, 0] + cos_yaw * points[:, 1]
        lower_x, upper_x = np.percentile(local_x, [percentile, 100.0 - percentile])
        lower_y, upper_y = np.percentile(local_y, [percentile, 100.0 - percentile])
        width = max(0.0, float(upper_x - lower_x))
        depth = max(0.0, float(upper_y - lower_y))
        area = width * depth
        if best is None or area < best[0]:
            best = (area, float(yaw_rad), float(lower_x), float(upper_x), float(lower_y), float(upper_y))

    _, yaw_rad, lower_x, upper_x, lower_y, upper_y = best
    width = float(upper_x - lower_x)
    depth = float(upper_y - lower_y)
    if width < depth:
        yaw_rad += 0.5 * math.pi
    yaw_rad = normalize_yaw_half_turn(yaw_rad)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    local_x = cos_yaw * points[:, 0] + sin_yaw * points[:, 1]
    local_y = -sin_yaw * points[:, 0] + cos_yaw * points[:, 1]
    lower_x, upper_x = np.percentile(local_x, [percentile, 100.0 - percentile])
    lower_y, upper_y = np.percentile(local_y, [percentile, 100.0 - percentile])
    width = max(0.0, float(upper_x - lower_x))
    depth = max(0.0, float(upper_y - lower_y))
    center_local_x = 0.5 * (float(lower_x) + float(upper_x))
    center_local_y = 0.5 * (float(lower_y) + float(upper_y))
    center = np.array(
        [
            cos_yaw * center_local_x - sin_yaw * center_local_y,
            sin_yaw * center_local_x + cos_yaw * center_local_y,
        ],
        dtype=np.float64,
    )
    return center, float(width), float(depth), float(yaw_rad)


def table_footprint_from_mask(
    *,
    mask: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    min_size_m: float = 0.005,
) -> TableFootprint | None:
    bbox = mask_bbox_pixel_edges(mask)
    if bbox is None:
        return None
    pixel_edges = contact_band_pixel_edges(mask, bbox) or bbox
    return table_footprint_from_pixel_edges(
        pixel_edges=pixel_edges,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
        table_z_m=table_z_m,
        min_size_m=min_size_m,
    )


def table_footprint_from_pixel_edges(
    *,
    pixel_edges: tuple[float, float, float, float],
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    min_size_m: float = 0.005,
) -> TableFootprint | None:
    left_u, right_u, top_v, bottom_v = pixel_edges
    center_u = 0.5 * (left_u + right_u)
    center_v = 0.5 * (top_v + bottom_v)
    corners = [
        table_intersection_for_pixel(
            u=u,
            v=v,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=table_z_m,
        )
        for u, v in (
            (left_u, top_v),
            (right_u, top_v),
            (right_u, bottom_v),
            (left_u, bottom_v),
        )
    ]
    center = table_intersection_for_pixel(
        u=center_u,
        v=center_v,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
        table_z_m=table_z_m,
    )
    if center is None or any(corner is None for corner in corners):
        return None
    width_px = max(1.0, float(right_u - left_u))
    height_px = max(1.0, float(bottom_v - top_v))
    horizontal_edge = (
        table_intersection_for_pixel(
            u=center_u - 0.5 * width_px,
            v=center_v,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=table_z_m,
        ),
        table_intersection_for_pixel(
            u=center_u + 0.5 * width_px,
            v=center_v,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=table_z_m,
        ),
    )
    vertical_edge = (
        table_intersection_for_pixel(
            u=center_u,
            v=center_v - 0.5 * height_px,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=table_z_m,
        ),
        table_intersection_for_pixel(
            u=center_u,
            v=center_v + 0.5 * height_px,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=table_z_m,
        ),
    )
    if any(point is None for point in (*horizontal_edge, *vertical_edge)):
        corner_array = np.asarray(corners, dtype=np.float64)
        size_x = float(corner_array[:, 0].max() - corner_array[:, 0].min())
        size_y = float(corner_array[:, 1].max() - corner_array[:, 1].min())
    else:
        size_x = float(np.linalg.norm(horizontal_edge[1] - horizontal_edge[0]))
        size_y = float(np.linalg.norm(vertical_edge[1] - vertical_edge[0]))
    size_x = max(float(min_size_m), size_x)
    size_y = max(float(min_size_m), size_y)
    return TableFootprint(
        center_xy_m=(float(center[0]), float(center[1])),
        size_x_m=size_x,
        size_y_m=size_y,
        width_px=width_px,
        height_px=height_px,
    )


def contact_band_pixel_edges(
    mask: np.ndarray,
    full_bbox: tuple[float, float, float, float],
    *,
    band_fraction: float = 0.35,
    max_width_fraction: float = 0.78,
    width_margin: float = 1.15,
    width_percentile: float = 25.0,
    smooth_tip_width_percentile: float = 75.0,
    neck_jump_ratio: float = 2.5,
    neck_min_width_px: float = 6.0,
) -> tuple[float, float, float, float] | None:
    binary = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(binary)
    if rows.size == 0:
        return None
    full_left, full_right, full_top, full_bottom = full_bbox
    full_width = max(1.0, float(full_right - full_left))
    full_height = max(1.0, float(full_bottom - full_top))
    band_height = max(2, int(round(full_height * float(band_fraction))))
    band_start = int(max(0, math.floor(float(full_bottom) + 0.5) - band_height))
    band_rows = rows >= band_start
    if not np.any(band_rows):
        return None
    row_records = []
    selected_rows = rows[band_rows]
    for row in range(int(selected_rows.min()), int(selected_rows.max()) + 1):
        row_cols = cols[rows == row]
        if row_cols.size:
            row_width = float(row_cols.max() - row_cols.min() + 1)
            row_center = 0.5 * float(row_cols.min() + row_cols.max())
            row_records.append((row_width, row_center, float(row)))
    if not row_records:
        return None
    widths = np.asarray([record[0] for record in row_records], dtype=np.float64)
    centers_u = np.asarray([record[1] for record in row_records], dtype=np.float64)
    centers_v = np.asarray([record[2] for record in row_records], dtype=np.float64)
    bottom_to_top_widths = widths[::-1]
    neck_end = None
    for index in range(1, len(bottom_to_top_widths)):
        previous_width = float(bottom_to_top_widths[index - 1])
        current_width = float(bottom_to_top_widths[index])
        if (
            previous_width >= float(neck_min_width_px)
            and current_width / max(previous_width, 1.0) > float(neck_jump_ratio)
        ):
            neck_end = index
            break
    if neck_end is not None and neck_end >= 2:
        raw_width = float(np.median(bottom_to_top_widths[:neck_end]))
    else:
        raw_width = float(np.percentile(widths, float(smooth_tip_width_percentile)))
    if raw_width * float(width_margin) >= full_width * float(max_width_fraction):
        raw_width = float(np.percentile(widths, float(width_percentile)))
    representative_width = raw_width * float(width_margin)
    contact_width = max(1.0, representative_width)
    if contact_width >= full_width * float(max_width_fraction):
        return None

    contact_center_u = float(np.median(centers_u))
    contact_center_v = float(np.median(centers_v))
    contact_height = max(1.0, min(contact_width, full_height))
    half_width = 0.5 * contact_width
    half_height = 0.5 * contact_height
    return (
        contact_center_u - half_width,
        contact_center_u + half_width,
        contact_center_v - half_height,
        contact_center_v + half_height,
    )


def shadow_support_pixel_edges(
    mask: np.ndarray,
    full_bbox: tuple[float, float, float, float],
    *,
    elongated_ratio: float = 1.35,
    band_fraction: float = 0.35,
    width_margin: float = 1.08,
) -> tuple[float, float, float, float] | None:
    contact_rect = contact_band_pixel_edges(mask, full_bbox)
    if contact_rect is not None:
        return contact_rect

    binary = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(binary)
    if rows.size == 0:
        return None
    full_left, full_right, full_top, full_bottom = full_bbox
    full_width = max(1.0, float(full_right - full_left))
    full_height = max(1.0, float(full_bottom - full_top))
    if full_height < full_width * float(elongated_ratio):
        return None

    band_height = max(2, int(round(full_height * float(band_fraction))))
    band_start = int(max(0, math.floor(float(full_bottom) + 0.5) - band_height))
    band_rows = rows >= band_start
    if not np.any(band_rows):
        return None

    row_records = []
    selected_rows = rows[band_rows]
    for row in range(int(selected_rows.min()), int(selected_rows.max()) + 1):
        row_cols = cols[rows == row]
        if row_cols.size:
            row_width = float(row_cols.max() - row_cols.min() + 1)
            row_center = 0.5 * float(row_cols.min() + row_cols.max())
            row_records.append((row_width, row_center, float(row)))
    if not row_records:
        return None

    widths = np.asarray([record[0] for record in row_records], dtype=np.float64)
    centers_u = np.asarray([record[1] for record in row_records], dtype=np.float64)
    centers_v = np.asarray([record[2] for record in row_records], dtype=np.float64)
    support_width = max(1.0, min(full_width, float(np.percentile(widths, 75.0)) * float(width_margin)))
    support_height = max(1.0, min(full_height, support_width))
    if support_height >= full_height * 0.85:
        return None

    center_u = float(np.median(centers_u))
    center_v = float(np.median(centers_v))
    half_width = 0.5 * support_width
    half_height = 0.5 * support_height
    return (
        center_u - half_width,
        center_u + half_width,
        center_v - half_height,
        center_v + half_height,
    )


def shadow_support_footprint_from_mask(
    *,
    mask: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    min_size_m: float = 0.005,
) -> TableFootprint | None:
    bbox = mask_bbox_pixel_edges(mask)
    if bbox is None:
        return None
    support_edges = shadow_support_pixel_edges(mask, bbox)
    if support_edges is None:
        return None
    return table_footprint_from_pixel_edges(
        pixel_edges=support_edges,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
        table_z_m=table_z_m,
        min_size_m=min_size_m,
    )


def base_points_from_mask_depth(
    *,
    mask: np.ndarray,
    raw_depth_m: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    depth = np.asarray(raw_depth_m, dtype=np.float32)
    valid_depth = depth[rows, cols]
    valid = np.isfinite(valid_depth) & (valid_depth > 0.0)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)
    rows = rows[valid]
    cols = cols[valid]
    valid_depth = valid_depth[valid].astype(np.float64)
    x = (cols.astype(np.float64) - float(intrinsics.cx)) * valid_depth / float(intrinsics.fx)
    y = (rows.astype(np.float64) - float(intrinsics.cy)) * valid_depth / float(intrinsics.fy)
    points_camera = np.stack([x, y, valid_depth], axis=1)
    rotation = quaternion_to_rotation_matrix(camera_to_base_transform.transform.rotation)
    translation = transform_translation_vector(camera_to_base_transform)
    return points_camera @ rotation.T + translation


def base_points_from_depth(
    *,
    depth_m: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    rows, cols = np.indices(depth.shape)
    valid_depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, np.nan).astype(np.float64)
    x = (cols.astype(np.float64) - float(intrinsics.cx)) * valid_depth / float(intrinsics.fx)
    y = (rows.astype(np.float64) - float(intrinsics.cy)) * valid_depth / float(intrinsics.fy)
    points_camera = np.stack([x, y, valid_depth], axis=-1)
    rotation = quaternion_to_rotation_matrix(camera_to_base_transform.transform.rotation)
    translation = transform_translation_vector(camera_to_base_transform)
    flat_base = points_camera.reshape(-1, 3) @ rotation.T + translation
    return flat_base.reshape(depth.shape + (3,))


def foreground_footprint_from_depth(
    *,
    mask: np.ndarray,
    raw_depth_m: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    min_foreground_height_m: float = 0.006,
    min_points: int = 16,
    min_mask_fraction: float = 0.08,
    robust_percentile: float = 5.0,
    margin_m: float = 0.002,
    min_size_m: float = 0.005,
    square_anisotropy_threshold: float = 1.12,
) -> TableFootprint | None:
    bbox = mask_bbox_pixel_edges(mask)
    if bbox is None:
        return None
    points_base = base_points_from_mask_depth(
        mask=mask,
        raw_depth_m=raw_depth_m,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
    )
    if points_base.size == 0:
        return None
    foreground = points_base[:, 2] > float(table_z_m) + float(min_foreground_height_m)
    foreground_count = int(foreground.sum())
    mask_count = max(1, int(np.asarray(mask, dtype=bool).sum()))
    if foreground_count < int(min_points) or foreground_count / mask_count < float(min_mask_fraction):
        return None
    foreground_points = points_base[foreground]
    center, width, depth, yaw_rad = robust_oriented_bounds_xy(
        foreground_points[:, 0:2],
        robust_percentile=float(robust_percentile),
    )
    left_u, right_u, top_v, bottom_v = bbox
    size_x = max(float(min_size_m), float(width) + float(margin_m))
    size_y = max(float(min_size_m), float(depth) + float(margin_m))
    smaller = min(size_x, size_y)
    larger = max(size_x, size_y)
    if smaller > 0.0 and larger / smaller < float(square_anisotropy_threshold):
        size_x = larger
        size_y = larger
        yaw_rad = canonicalize_square_yaw(yaw_rad)
    return TableFootprint(
        center_xy_m=(float(center[0]), float(center[1])),
        size_x_m=size_x,
        size_y_m=size_y,
        width_px=max(1.0, float(right_u - left_u)),
        height_px=max(1.0, float(bottom_v - top_v)),
        yaw_rad=float(yaw_rad),
    )


def visible_height_from_depth(
    *,
    mask: np.ndarray,
    raw_depth_m: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    fallback_height_m: float,
    min_height_m: float = 0.010,
    max_height_m: float = 0.120,
) -> float:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return float(fallback_height_m)
    depth = np.asarray(raw_depth_m, dtype=np.float32)
    valid_depth = depth[rows, cols]
    valid = np.isfinite(valid_depth) & (valid_depth > 0.0)
    if not valid.any():
        return float(fallback_height_m)
    points_base = base_points_from_mask_depth(
        mask=mask,
        raw_depth_m=raw_depth_m,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
    )
    visible_top_z = float(np.percentile(points_base[:, 2], 90.0))
    estimated_height = visible_top_z - float(table_z_m)
    if not math.isfinite(estimated_height) or estimated_height <= 0.0:
        return float(fallback_height_m)
    return clamp(estimated_height, min_height_m, max_height_m)


def fallback_height_from_footprint(
    footprint: TableFootprint,
    *,
    fallback_height_m: float,
    min_height_m: float = 0.015,
) -> float:
    footprint_height = min(float(footprint.size_x_m), float(footprint.size_y_m))
    if not math.isfinite(footprint_height) or footprint_height <= 0.0:
        return float(fallback_height_m)
    return clamp(footprint_height, min_height_m, float(fallback_height_m))


def refine_shadow_expanded_fit(
    fit: GeometryFit,
    support_footprint: TableFootprint | None,
    *,
    min_area_reduction: float = 0.12,
) -> GeometryFit:
    if support_footprint is None or fit.shape_type not in {"box", "cylinder"}:
        return fit
    fit_area = max(1e-9, float(fit.size_x_m) * float(fit.size_y_m))
    support_area = max(1e-9, float(support_footprint.size_x_m) * float(support_footprint.size_y_m))
    if support_area >= fit_area * (1.0 - float(min_area_reduction)):
        return fit

    support_x = max(0.005, float(support_footprint.size_x_m))
    support_y = max(0.005, float(support_footprint.size_y_m))
    if fit.shape_type == "cylinder":
        diameter = max(support_x, support_y)
        size_x = diameter
        size_y = diameter
        yaw_rad = 0.0
    else:
        side = max(support_x, support_y)
        fit_aspect = max(float(fit.size_x_m), float(fit.size_y_m)) / max(
            min(float(fit.size_x_m), float(fit.size_y_m)),
            1e-9,
        )
        if fit_aspect >= 1.35:
            size_x = side
            size_y = side
        else:
            size_x = support_x
            size_y = support_y
        yaw_rad = float(fit.yaw_rad)

    return replace(
        fit,
        center_xy_m=(
            float(support_footprint.center_xy_m[0]),
            float(support_footprint.center_xy_m[1]),
        ),
        size_x_m=float(size_x),
        size_y_m=float(size_y),
        yaw_rad=yaw_rad,
        provenance=f"{fit.provenance};support=shadow_compact",
    )


def coerce_projected_shadow_fit_to_unknown_bbox(
    fit: GeometryFit,
    support_source: str,
    support_footprint: TableFootprint | None,
    component: PixelComponent,
    full_mask_footprint: TableFootprint | None = None,
) -> GeometryFit:
    if str(support_source) not in {"mask_full", "shadow_trimmed"}:
        return fit

    if support_footprint is None:
        return replace(
            fit,
            shape_type="bbox",
            provenance=f"{fit.provenance};shadow_shape=unknown",
        )

    support_x = max(0.005, float(support_footprint.size_x_m))
    support_y = max(0.005, float(support_footprint.size_y_m))
    support_aspect = max(support_x, support_y) / max(min(support_x, support_y), 1e-9)
    support_pixel_width = max(1.0, float(support_footprint.width_px))
    support_pixel_height = max(1.0, float(support_footprint.height_px))
    support_pixel_aspect = max(support_pixel_width, support_pixel_height) / max(
        min(support_pixel_width, support_pixel_height),
        1e-9,
    )
    component_fill = component_bbox_fill_ratio(component)
    _, component_corner_support = mask_rectangular_support(component.mask)
    component_aspect = component_bbox_aspect(component)
    shape_diag = (
        f";shape_diag:fill={component_fill:.3f},corner={component_corner_support:.3f},"
        f"component_aspect={component_aspect:.3f},support_aspect={support_aspect:.3f},"
        f"support_pixel_aspect={support_pixel_aspect:.3f}"
    )
    support_yaw = float(support_footprint.yaw_rad)
    provenance_suffix = ";shadow_shape=unknown"
    full_mask_squareish_yaw_hint = False
    full_mask_yaw = 0.0
    if full_mask_footprint is not None:
        full_x = max(1e-9, float(full_mask_footprint.size_x_m))
        full_y = max(1e-9, float(full_mask_footprint.size_y_m))
        full_aspect = max(full_x, full_y) / max(min(full_x, full_y), 1e-9)
        full_area = full_x * full_y
        compact_area = max(1e-9, support_x * support_y)
        full_mask_squareish_yaw_hint = (
            support_aspect <= 1.45
            and full_aspect <= 2.25
            and full_area >= compact_area * 1.10
        )
        full_mask_yaw = float(full_mask_footprint.yaw_rad)
    round_compact_shadow = (
        0.66 <= component_fill <= 0.86
        and component_corner_support <= 0.58
        and component_aspect <= 1.30
        and support_aspect <= 1.18
        and support_pixel_aspect <= 1.35
        and not (
            full_mask_squareish_yaw_hint
            and component_fill < 0.74
            and component_aspect <= 1.15
            and support_pixel_aspect <= 1.10
        )
    )
    round_supported_component = (
        0.78 <= component_fill <= 0.88
        and 0.44 <= component_corner_support <= 0.64
        and component_aspect <= 1.32
        and support_aspect <= 1.35
        and support_pixel_aspect <= 1.35
    )
    if (
        0.74 <= component_fill <= 0.86
        and component_aspect <= 1.25
        and support_aspect <= 1.35
    ) or round_compact_shadow or round_supported_component:
        diameter = max(support_x, support_y)
        return replace(
            fit,
            shape_type="cylinder",
            center_xy_m=(
                float(support_footprint.center_xy_m[0]),
                float(support_footprint.center_xy_m[1]),
            ),
            size_x_m=float(diameter),
            size_y_m=float(diameter),
            yaw_rad=0.0,
            provenance=f"{fit.provenance};shadow_shape=mask_round_cylinder{shape_diag}",
        )

    squareish_component = (
        (
            component_aspect <= 1.35
            and (component_fill >= 0.80 or (component_fill >= 0.70 and support_pixel_aspect <= 1.35))
        )
        or (
            full_mask_squareish_yaw_hint
            and component_fill >= 0.70
            and support_aspect <= 1.35
            and support_pixel_aspect <= 1.35
        )
    )
    support_min_side = min(support_x, support_y)
    cube_like_compact_support = (
        full_mask_squareish_yaw_hint
        and component_fill >= 0.55
        and support_aspect <= 1.45
        and abs(support_min_side - float(fit.size_z_m)) <= max(0.0035, 0.18 * max(0.005, float(fit.size_z_m)))
    )
    if support_aspect <= 1.45 and (squareish_component or cube_like_compact_support):
        side = max(support_x, support_y)
        support_x = side
        support_y = side
        if full_mask_squareish_yaw_hint:
            support_yaw = full_mask_yaw
            provenance_suffix += ";full_mask_yaw"
        support_yaw, snapped = stabilize_weak_square_yaw(support_yaw)
        if snapped:
            provenance_suffix += ";weak_square_yaw_axis_snap"
    elif full_mask_squareish_yaw_hint:
        support_yaw, snapped = stabilize_weak_square_yaw(full_mask_yaw)
        provenance_suffix += ";full_mask_yaw"
        if snapped:
            provenance_suffix += ";weak_square_yaw_axis_snap"

    return replace(
        fit,
        shape_type="bbox",
        center_xy_m=(
            float(support_footprint.center_xy_m[0]),
            float(support_footprint.center_xy_m[1]),
        ),
        size_x_m=max(0.005, float(support_x)),
        size_y_m=max(0.005, float(support_y)),
        yaw_rad=float(support_yaw),
        provenance=f"{fit.provenance}{provenance_suffix}{shape_diag}",
    )


def expand_sparse_foreground_fit_with_support(
    fit: GeometryFit,
    support_source: str,
    support_footprint: TableFootprint | None,
    *,
    min_area_gain: float = 1.35,
) -> GeometryFit:
    if str(support_source) not in {"foreground", "nearby_foreground"}:
        return fit
    if support_footprint is None:
        return fit

    fit_area = max(1e-9, float(fit.size_x_m) * float(fit.size_y_m))
    support_x = max(0.005, float(support_footprint.size_x_m))
    support_y = max(0.005, float(support_footprint.size_y_m))
    support_area = support_x * support_y
    if support_area < fit_area * float(min_area_gain):
        return fit

    return replace(
        fit,
        provenance=f"{fit.provenance};shadow_support_suppressed_for_tight_fit",
    )


def align_sparse_foreground_box_to_visible_edges(
    fit: GeometryFit,
    support_source: str,
    support_footprint: TableFootprint | None,
    points_xy_m: np.ndarray,
    *,
    min_points: int = 6,
    max_squareish_aspect: float = 1.45,
    max_boundary_error_fraction: float = 0.16,
) -> GeometryFit:
    if str(support_source) != "foreground" or fit.shape_type not in {"box", "bbox"}:
        return fit

    points = np.asarray(points_xy_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return fit
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.shape[0] < int(min_points):
        return fit

    try:
        _, visible_width, visible_depth, visible_yaw = robust_oriented_bounds_xy(
            points,
            robust_percentile=0.0,
            angle_step_deg=0.5,
        )
    except ValueError:
        return fit

    visible_minor = max(1e-9, min(float(visible_width), float(visible_depth)))
    visible_aspect = max(float(visible_width), float(visible_depth)) / visible_minor
    support_aspect = 1.0
    support_center = np.asarray(fit.center_xy_m, dtype=np.float64)
    support_yaw = float(fit.yaw_rad)
    support_size = max(float(fit.size_x_m), float(fit.size_y_m))
    if support_footprint is not None:
        support_x = max(1e-9, float(support_footprint.size_x_m))
        support_y = max(1e-9, float(support_footprint.size_y_m))
        support_aspect = max(support_x, support_y) / max(min(support_x, support_y), 1e-9)
        support_center = np.asarray(support_footprint.center_xy_m, dtype=np.float64)
        support_yaw = float(support_footprint.yaw_rad)
        support_size = max(support_size, min(support_x, support_y))

    # Square-completion must be justified by the measured points alone; the
    # shadow/hole support may position the completion but never argue an
    # elongated visible cloud into a square.
    if visible_aspect > float(max_squareish_aspect):
        return fit

    side = max(
        0.005,
        float(visible_width),
        float(visible_depth),
        float(fit.size_z_m),
        min(float(fit.size_x_m), float(fit.size_y_m)),
    )
    if math.isfinite(support_size) and support_size > 0.0:
        side = min(side, max(side, float(support_size)))
    if min(float(visible_width), float(visible_depth)) < 0.45 * side:
        return fit

    yaw_candidates = [
        float(fit.yaw_rad),
        float(visible_yaw),
        float(support_yaw),
        float(visible_yaw + 0.5 * math.pi),
        float(support_yaw + 0.5 * math.pi),
    ]
    unique_yaws: list[float] = []
    for yaw in yaw_candidates:
        yaw = normalize_yaw_half_turn(yaw)
        if all(abs(normalize_yaw_half_turn(yaw - existing)) > math.radians(1.0) for existing in unique_yaws):
            unique_yaws.append(yaw)

    best: tuple[float, np.ndarray, float] | None = None
    for yaw in unique_yaws:
        center, boundary_error = _visible_edge_aligned_center(points, yaw, side, support_center)
        support_distance = float(np.linalg.norm(center - support_center))
        score = boundary_error + 0.15 * support_distance
        if best is None or score < best[0]:
            best = (score, center, yaw)

    if best is None:
        return fit

    _, center, yaw = best
    if best[0] > float(max_boundary_error_fraction) * side:
        return fit
    yaw_reference = float(support_yaw if support_footprint is not None else visible_yaw)
    yaw = _canonicalize_square_yaw_near(float(yaw), yaw_reference)
    original_center = np.asarray(fit.center_xy_m, dtype=np.float64)
    if float(np.linalg.norm(center - original_center)) > max(0.030, 2.0 * side):
        return fit

    return replace(
        fit,
        center_xy_m=(float(center[0]), float(center[1])),
        size_x_m=float(side),
        size_y_m=float(side),
        yaw_rad=float(normalize_yaw_half_turn(yaw)),
        provenance=f"{fit.provenance};foreground_edge_aligned",
    )


def _canonicalize_square_yaw_near(yaw_rad: float, reference_yaw_rad: float) -> float:
    candidates = [
        normalize_yaw_half_turn(float(yaw_rad)),
        normalize_yaw_half_turn(float(yaw_rad) + 0.5 * math.pi),
        normalize_yaw_half_turn(float(yaw_rad) - 0.5 * math.pi),
    ]
    reference = float(reference_yaw_rad)
    return float(
        min(
            candidates,
            key=lambda candidate: abs(normalize_yaw_half_turn(float(candidate) - reference)),
        )
    )


def _visible_edge_aligned_center(
    points_xy_m: np.ndarray,
    yaw_rad: float,
    side_m: float,
    support_center_xy_m: np.ndarray,
) -> tuple[np.ndarray, float]:
    yaw = float(yaw_rad)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    points = np.asarray(points_xy_m, dtype=np.float64)
    local_x = cos_yaw * points[:, 0] + sin_yaw * points[:, 1]
    local_y = -sin_yaw * points[:, 0] + cos_yaw * points[:, 1]
    support = np.asarray(support_center_xy_m, dtype=np.float64)
    support_x = cos_yaw * support[0] + sin_yaw * support[1]
    support_y = -sin_yaw * support[0] + cos_yaw * support[1]
    side = max(0.005, float(side_m))
    center_x = _visible_edge_axis_center(local_x, side, float(support_x))
    center_y = _visible_edge_axis_center(local_y, side, float(support_y))
    boundary_x = np.minimum(np.abs(local_x - (center_x - 0.5 * side)), np.abs(local_x - (center_x + 0.5 * side)))
    boundary_y = np.minimum(np.abs(local_y - (center_y - 0.5 * side)), np.abs(local_y - (center_y + 0.5 * side)))
    boundary_error = float(np.mean(np.minimum(boundary_x, boundary_y)))
    center = np.array(
        [
            cos_yaw * center_x - sin_yaw * center_y,
            sin_yaw * center_x + cos_yaw * center_y,
        ],
        dtype=np.float64,
    )
    return center, boundary_error


def _visible_edge_axis_center(values: np.ndarray, size_m: float, support_center_value: float) -> float:
    axis_values = np.asarray(values, dtype=np.float64)
    lower, upper = np.percentile(axis_values, [2.0, 98.0])
    span = float(upper - lower)
    size = max(0.005, float(size_m))
    if span >= 0.55 * size:
        return 0.5 * float(lower + upper)
    boundary = float(np.median(axis_values))
    candidates = (boundary - 0.5 * size, boundary + 0.5 * size)
    return float(min(candidates, key=lambda candidate: abs(candidate - float(support_center_value))))


HOLE_EXISTENCE_PROXY_SIZE_M = 0.030
HOLE_ADJACENCY_MIN_FRACTION = 0.20


def component_hole_adjacency_fraction(
    component_mask: np.ndarray,
    depth_m: np.ndarray,
    *,
    radius_px: int = 6,
) -> float:
    mask = np.asarray(component_mask, dtype=bool)
    if not mask.any():
        return 0.0
    depth = np.asarray(depth_m, dtype=np.float32)
    hole = ~np.isfinite(depth) | (depth <= 0.0)
    neighborhood = expanded_bbox_mask(mask, radius_px=int(radius_px)) & ~mask
    neighborhood_px = int(np.count_nonzero(neighborhood))
    if neighborhood_px == 0:
        return 0.0
    return float(np.count_nonzero(hole & neighborhood)) / float(neighborhood_px)


def coerce_hole_adjacent_fit_to_existence_proxy(
    fit,
    points_xy_m: np.ndarray,
    *,
    proxy_size_m: float = HOLE_EXISTENCE_PROXY_SIZE_M,
):
    center_x = float(fit.center_xy_m[0])
    center_y = float(fit.center_xy_m[1])
    points = np.asarray(points_xy_m, dtype=np.float64)
    if points.ndim == 2 and points.shape[1] == 2 and points.shape[0] > 0:
        finite = np.all(np.isfinite(points), axis=1)
        if finite.any():
            center_x = float(np.median(points[finite, 0]))
            center_y = float(np.median(points[finite, 1]))
    size = max(0.005, float(proxy_size_m))
    return replace(
        fit,
        shape_type="bbox",
        center_xy_m=(center_x, center_y),
        size_x_m=size,
        size_y_m=size,
        yaw_rad=0.0,
        provenance=f"{fit.provenance};hole_existence_only",
    )


def min_foreground_points_for_component(component_area_px: int) -> int:
    return max(6, min(24, int(round(float(component_area_px) * 0.02))))


def has_compact_foreground_residue(
    foreground: np.ndarray,
    *,
    min_points: int = 8,
    min_fill_ratio: float = 0.35,
    max_aspect: float = 4.0,
) -> bool:
    rows, cols = np.nonzero(np.asarray(foreground, dtype=bool))
    if rows.size < int(min_points):
        return False
    width = int(cols.max() - cols.min() + 1)
    height = int(rows.max() - rows.min() + 1)
    if width <= 0 or height <= 0:
        return False
    aspect = float(max(width, height)) / float(max(1, min(width, height)))
    fill_ratio = float(rows.size) / float(max(1, width * height))
    return aspect <= float(max_aspect) and fill_ratio >= float(min_fill_ratio)


def component_bbox_fill_ratio(component: PixelComponent) -> float:
    min_x, min_y, max_x, max_y = component.bbox_xyxy
    bbox_area = max(1, int(max_x - min_x + 1) * int(max_y - min_y + 1))
    return float(component.area_px) / float(bbox_area)


def component_bbox_aspect(component: PixelComponent) -> float:
    min_x, min_y, max_x, max_y = component.bbox_xyxy
    width = max(1, int(max_x - min_x + 1))
    height = max(1, int(max_y - min_y + 1))
    return float(max(width, height)) / float(min(width, height))


def top_down_orientation() -> Quaternion:
    orientation = Quaternion()
    orientation.x = -math.sqrt(0.5)
    orientation.y = 0.0
    orientation.z = 0.0
    orientation.w = math.sqrt(0.5)
    return orientation


def yaw_orientation(yaw_rad: float) -> Quaternion:
    orientation = Quaternion()
    half_yaw = 0.5 * float(yaw_rad)
    orientation.z = math.sin(half_yaw)
    orientation.w = math.cos(half_yaw)
    return orientation


def make_pose(frame_id: str, x: float, y: float, z: float, orientation: Quaternion | None = None) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = float(z)
    pose.pose.orientation = orientation or Quaternion(w=1.0)
    return pose


def shape_constant(shape_type: str) -> int:
    if shape_type == "box":
        return GeometryHypothesis.SHAPE_BOX
    if shape_type == "bbox":
        return GeometryHypothesis.SHAPE_BOX
    if shape_type == "cylinder":
        return GeometryHypothesis.SHAPE_CYLINDER
    return GeometryHypothesis.SHAPE_UNKNOWN


def make_score(ranked) -> ScoreBreakdown:
    score = ScoreBreakdown()
    score.visual = float(ranked.score.visual)
    score.failure = float(ranked.score.failure)
    score.depth = float(ranked.score.depth)
    score.physical = float(ranked.score.physical)
    score.grasp = float(ranked.score.grasp)
    score.prior = float(ranked.score.prior)
    score.total = float(ranked.score.total)
    return score


def make_live_tabletop_score(ranked) -> ScoreBreakdown:
    terms = dict(ranked.score_terms)
    score = ScoreBreakdown()
    score.visual = float(terms.get("coverage_ratio", 0.0))
    score.failure = float(terms.get("failure_ratio", 0.0))
    score.depth = float(terms.get("support_ratio", 0.0))
    score.physical = max(0.0, 1.0 - float(terms.get("table_leakage_ratio", 0.0)))
    score.grasp = float(terms.get("shape_prior", 0.0))
    score.prior = float(terms.get("shape_prior", 0.0))
    score.total = float(ranked.score)
    return score


def build_live_hypothesis_array(
    *,
    raw_depth_m: np.ndarray,
    corrupted_depth_m: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: CameraModel,
    frame_id: str = "d435_depth_optical_frame",
    base_frame_id: str = "world",
    camera_to_base_transform: TransformStamped | None = None,
    table_z_m: float | None = None,
    top_k: int = 3,
    primitive_height_m: float = 0.04,
    min_tabletop_proxy_size_m: float = 0.005,
    min_tabletop_proxy_height_m: float = 0.005,
    split_seed_centers_xy: list[tuple[float, float]] | None = None,
    split_seed_half_extents_m: list[float] | None = None,
) -> GeometryHypothesisArray:
    mask = np.asarray(target_mask, dtype=bool)
    use_base_frame = camera_to_base_transform is not None and table_z_m is not None
    if not mask.any():
        return empty_live_hypothesis_array(
            frame_id=frame_id,
            base_frame_id=base_frame_id,
            use_base_frame=use_base_frame,
        )

    components = extract_components(mask, min_area_px=1)
    split_component_ids: set[int] = set()
    component_grouping_reasons: dict[int, str] = {}
    if use_base_frame:
        points_base = base_points_from_depth(
            depth_m=corrupted_depth_m,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
        )
        seeded_masks: list[np.ndarray] = []
        if split_seed_centers_xy:
            components, seeded_ids = split_components_by_recent_track_seeds(
                components,
                points_base=points_base,
                z_map=points_base[:, :, 2],
                table_z_m=float(table_z_m),
                seed_centers_xy=list(split_seed_centers_xy),
                seed_half_extents_m=(
                    list(split_seed_half_extents_m)
                    if split_seed_half_extents_m is not None
                    else None
                ),
            )
            seeded_masks = [
                np.asarray(component.mask, dtype=bool)
                for component in components
                if int(component.component_id) in seeded_ids
            ]
        (
            components,
            split_component_ids,
            component_grouping_reasons,
        ) = split_components_by_foreground_islands(
            components,
            z_map=points_base[:, :, 2],
            table_z_m=float(table_z_m),
        )
        for component in components:
            component_mask = np.asarray(component.mask, dtype=bool)
            area = max(1, int(np.count_nonzero(component_mask)))
            for seeded_mask in seeded_masks:
                if int(np.count_nonzero(component_mask & seeded_mask)) >= 0.9 * area:
                    component_grouping_reasons[int(component.component_id)] = (
                        "recent_track_seeded_split"
                    )
                    split_component_ids.add(int(component.component_id))
                    break
        hole_pixels = ~np.isfinite(corrupted_depth_m) | (corrupted_depth_m <= 0.0)
        components, ring_fits = merge_transparent_ring_components(
            components,
            points_base=points_base,
            z_map=points_base[:, :, 2],
            hole_mask=hole_pixels,
            table_z_m=float(table_z_m),
        )
        for ring_component_id in ring_fits:
            component_grouping_reasons[int(ring_component_id)] = "transparent_ring_merge"
    else:
        ring_fits = {}
    has_masked_holes = bool(np.any(mask & (~np.isfinite(corrupted_depth_m) | (corrupted_depth_m <= 0.0))))
    if use_base_frame and (len(components) > 1 or has_masked_holes or ring_fits):
        return build_live_tabletop_component_hypothesis_array(
            raw_depth_m=raw_depth_m,
            corrupted_depth_m=corrupted_depth_m,
            target_mask=mask,
            components=components,
            intrinsics=intrinsics,
            frame_id=frame_id,
            base_frame_id=base_frame_id,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=float(table_z_m),
            primitive_height_m=primitive_height_m,
            min_tabletop_proxy_height_m=min_tabletop_proxy_height_m,
            split_component_ids=split_component_ids,
            component_grouping_reasons=component_grouping_reasons,
            ring_fits=ring_fits,
        )

    if len(components) > 1:
        message = empty_live_hypothesis_array(
            frame_id=frame_id,
            base_frame_id=base_frame_id,
            use_base_frame=use_base_frame,
        )
        for component in components:
            component_message = _build_live_hypothesis_array_single_target(
                raw_depth_m=raw_depth_m,
                corrupted_depth_m=corrupted_depth_m,
                target_mask=component.mask,
                intrinsics=intrinsics,
                frame_id=frame_id,
                base_frame_id=base_frame_id,
                camera_to_base_transform=camera_to_base_transform,
                table_z_m=table_z_m,
                top_k=top_k,
                primitive_height_m=primitive_height_m,
                min_tabletop_proxy_size_m=min_tabletop_proxy_size_m,
                min_tabletop_proxy_height_m=min_tabletop_proxy_height_m,
            )
            for hypothesis in component_message.hypotheses[: max(1, int(top_k))]:
                hypothesis.hypothesis_id = f"component_{component.component_id}_{hypothesis.hypothesis_id}"
                hypothesis.provenance = (
                    f"{hypothesis.provenance};component_id={component.component_id};"
                    f"component_area_px={component.area_px};component_path=legacy_split"
                )
                message.hypotheses.append(hypothesis)
        return message

    return _build_live_hypothesis_array_single_target(
        raw_depth_m=raw_depth_m,
        corrupted_depth_m=corrupted_depth_m,
        target_mask=mask,
        intrinsics=intrinsics,
        frame_id=frame_id,
        base_frame_id=base_frame_id,
        camera_to_base_transform=camera_to_base_transform,
        table_z_m=table_z_m,
        top_k=top_k,
        primitive_height_m=primitive_height_m,
        min_tabletop_proxy_size_m=min_tabletop_proxy_size_m,
        min_tabletop_proxy_height_m=min_tabletop_proxy_height_m,
    )


def empty_live_hypothesis_array(
    *,
    frame_id: str,
    base_frame_id: str,
    use_base_frame: bool,
) -> GeometryHypothesisArray:
    message = GeometryHypothesisArray()
    message.header.frame_id = str(base_frame_id) if use_base_frame else frame_id
    message.trial_id = "m4_live"
    message.observation_id = "m4_live_hypotheses"
    message.backend_name = "ghost_mgg_m4_live"
    return message


SEED_SPLIT_MIN_CLUSTER_PX = 80
SEED_SPLIT_MIN_SEED_SEPARATION_M = 0.040
SEED_SPLIT_MIN_CLUSTER_SEPARATION_M = 0.030
SEED_SPLIT_SEED_REACH_M = 0.025
GHOST_SEED_TTL_PUBLISHES = 25
GHOST_SEED_LIVE_SUPPRESSION_M = 0.030

TRANSPARENT_HOLE_MIN_AREA_PX = 40
TRANSPARENT_HOLE_MAX_DIM_M = 0.25
TRANSPARENT_HOLE_PROXY_HEIGHT_M = 0.030
TRANSPARENT_HOLE_EXPLAIN_DISTANCE_M = 0.150
TRANSPARENT_HOLE_EXPLAIN_CONE_DEG = 25.0  # base half-angle; widened by occluder size
TRANSPARENT_SPECK_MIN_AREA_PX = 6
TRANSPARENT_SPECK_MIN_HEIGHT_M = 0.008
TRANSPARENT_BELOW_TABLE_M = 0.005
TRANSPARENT_BELOW_TABLE_MAX_M = 0.050  # refraction bends mm..cm, not the floor drop-off
TRANSPARENT_BELOW_MIN_AREA_PX = 25
TRANSPARENT_HOLE_RING_TABLE_FRACTION = 0.60
TRANSPARENT_NEAR_OPAQUE_MARGIN_M = 0.030
TRANSPARENT_PROXY_SUPPRESS_MARGIN_M = 0.025
SPARSE_ISLAND_MIN_GEOMETRY_PX = 80

TRANSPARENT_SPECK_MAX_AREA_PX = 200
TRANSPARENT_HOLE_FRONT_BAND_QUANTILE = 75.0

RGB_SALIENCY_BLUR_PX = 7
RGB_SALIENCY_GRAD_MIN = 6.0
RGB_SALIENCY_GRAD_RATIO = 3.0
RGB_SALIENCY_BRIGHT_MIN = 10.0
RGB_SALIENCY_BRIGHT_RATIO = 4.0
RGB_SALIENCY_MIN_AREA_PX = 12
RGB_SALIENCY_LINK_PX = 2

FIELD_CELL_M = 0.004
FIELD_TAU_S = 2.5
# measured 3D points on a transparent object are themselves displaced by
# refraction; the RGB base-contact band is the only unbiased position cue,
# so it carries the position weight while depth cues carry existence
FIELD_WEIGHT_DIRECT = 1.5
FIELD_WEIGHT_HOLE_EDGE = 1.5
FIELD_WEIGHT_RGB = 3.0
FIELD_BIRTH_MASS = 2.5
FIELD_SUSTAIN_MASS = 1.0
FIELD_DIRECT_REGION_MIN = 8.0
FIELD_REGION_TOTAL_MASS_MIN = 40.0
FIELD_FOOTPRINT_MIN_MAX_DIM_M = 0.025
FIELD_ANCHOR_FOOTPRINT_MIN_DIM_M = 0.025
FIELD_YAW_AMBIGUOUS_ASPECT = 1.30
REFRACTION_ARTIFACT_MAX_DIM_M = 0.065
# a core only counts as viable (blocking evidence-rescue merge) when it
# clears the footprint floor with margin; borderline cores flicker across
# the gate, which IS the shattered-evidence regime the merge exists for
FIELD_UNION_CORE_SPAN_M = 1.5 * 0.025
# orientation hysteresis: a blob's min-area angle sweep is nearly flat, so
# the argmin random-walks with fringe noise; only rotate the footprint when
# the new orientation is DECISIVELY tighter than the remembered one
FIELD_YAW_KEEP_AREA_FRACTION = 1.15
# refraction artifacts are SPARSE (rearbag: area 30-416px, coverage
# 0.49-0.57); a dense component is a real object no matter where it sits
REFRACTION_ARTIFACT_MAX_AREA_PX = 600
REFRACTION_ARTIFACT_MAX_COVERAGE = 0.75
# a hole overlapping a detector instance box this much is claimed by the
# transparent instance: shadow-cone attribution to occluders is overridden
TRANSPARENT_HOLE_INSTANCE_OVERLAP = 0.5
# an occluder's own displaced stereo hole lands up to ~4cm around its
# footprint (image-space hole displacement projected to the table); any
# transparency vote inside that clearance is pollution from the occluder
TRANSPARENT_VOTE_OCCLUDER_CLEARANCE_M = 0.04
# below this color-image mean, stateless RGB saliency is sensor noise (the
# RGB sensor faintly sees the IR projector dots in darkness) — instance
# boxes (IR detector) remain trusted, anonymous saliency does not
DARK_LUMINANCE_MEAN = 35.0
# --- ablation switches (offline experiments ONLY; env-gated, default off) ---
ABLATION_NO_INSTANCE = bool(int(os.environ.get("MGG_ABL_NO_INSTANCE", "0")))
ABLATION_NO_HOLE = bool(int(os.environ.get("MGG_ABL_NO_HOLE", "0")))
ABLATION_NO_ANCHOR = bool(int(os.environ.get("MGG_ABL_NO_ANCHOR", "0")))
ABLATION_NO_MERGE = bool(int(os.environ.get("MGG_ABL_NO_MERGE", "0")))
REFRACTION_ARTIFACT_REACH_M = 0.12
FIELD_BARE_TAU_S = 0.4
FIELD_SEED_REACH_M = 0.06
MICRO_TRACK_MAX_DIM_M = 0.015
MICRO_TRACK_ABSORB_MARGIN_M = 0.020
FIELD_FOOTPRINT_PEAK_FRACTION = 0.35
FIELD_ANCHOR_ENTER_MASS = 6.0
FIELD_ANCHOR_STAY_MASS = 2.5
RGB_FRONT_BAND_QUANTILE = 60.0
DETECTOR_BOX_BOTTOM_BAND_FRACTION = 0.35
DETECTOR_BOX_BODY_WEIGHT = 0.75
DETECTOR_BOX_TALL_ASPECT = 1.2
DETECTOR_BOX_FLAT_ASPECT = 0.9
DETECTOR_BOXES_FRESH_S = 1.2
DEFAULT_TRANSPARENT_BOXES_TOPIC = "/ghost_mgg/d435/transparent_boxes"


def rgb_luma(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    return 0.299 * values[..., 0] + 0.587 * values[..., 1] + 0.114 * values[..., 2]


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    window = 2 * int(radius) + 1
    for axis in (0, 1):
        pad = [(radius, radius) if a == axis else (0, 0) for a in range(2)]
        padded = np.pad(result, pad, mode="edge")
        cumsum = np.cumsum(padded, axis=axis, dtype=np.float64)
        zero_shape = [1 if a == axis else cumsum.shape[a] for a in range(2)]
        cumsum = np.concatenate(
            [np.zeros(zero_shape, dtype=np.float64), cumsum], axis=axis
        )
        take_hi = tuple(
            slice(window, None) if a == axis else slice(None) for a in range(2)
        )
        take_lo = tuple(
            slice(0, -window) if a == axis else slice(None) for a in range(2)
        )
        result = ((cumsum[take_hi] - cumsum[take_lo]) / float(window)).astype(
            np.float32
        )
    return result


def _luma_gradient_magnitude(luma: np.ndarray) -> np.ndarray:
    values = np.asarray(luma, dtype=np.float32)
    grad = np.zeros_like(values)
    grad[:, 1:] += np.abs(np.diff(values, axis=1))
    grad[1:, :] += np.abs(np.diff(values, axis=0))
    return grad


def rgb_transparency_saliency_mask(rgb: np.ndarray) -> np.ndarray:
    """Stateless per-frame RGB saliency: outline edges and specular glints.

    No background model, no temporal state — nothing to corrupt and no
    rate-dependent dynamics. On a locally smooth table a transparent object
    is visible as fine-scale gradient energy (refraction outline, caustics)
    and small bright highlights. Thresholds adapt to the frame's own noise
    floor (median statistics), not to a fixed scene.
    """
    luma = rgb_luma(np.asarray(rgb, dtype=np.float32))
    gradient = _luma_gradient_magnitude(luma)
    gradient_floor = max(
        RGB_SALIENCY_GRAD_MIN,
        RGB_SALIENCY_GRAD_RATIO * float(np.median(gradient)),
    )
    highlight = luma - _box_blur(luma, RGB_SALIENCY_BLUR_PX)
    highlight_floor = max(
        RGB_SALIENCY_BRIGHT_MIN,
        RGB_SALIENCY_BRIGHT_RATIO * float(np.median(np.abs(highlight))),
    )
    return (gradient > gradient_floor) | (highlight > highlight_floor)


def transparent_boxes_weight_mask(
    boxes_xyxy: np.ndarray,
    full_shape: tuple[int, int],
    strided_shape: tuple[int, int],
    saliency_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-pixel vote weights from detector boxes on the strided grid.

    A detector box says WHERE an instance is, not exactly what it covers —
    boxes go loose on out-of-distribution poses. When stateless saliency is
    available, the votes are refined to (box AND saliency): the outline and
    glints inside the box are the object, the box slack is not. The base
    band (full weight — the one part whose table ray-cast is not displaced
    behind a tall object) is then taken from the SALIENCY SUPPORT's row
    span, so a loose box cannot drag the anchor. Without enough saliency
    support the geometric box band is the fallback. Band fraction adapts to
    the (support) aspect: tall = standing (narrow band), flat = lying
    (whole silhouette IS the footprint); a frame-cropped box top forces
    standing.
    """
    weights = np.zeros(strided_shape, dtype=np.float32)
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return weights
    support_full = None
    if saliency_mask is not None:
        support_full = np.asarray(saliency_mask, dtype=bool)
        for _ in range(2):
            support_full = _binary_dilate_1px(support_full)
    scale_r = strided_shape[0] / float(max(1, full_shape[0]))
    scale_c = strided_shape[1] / float(max(1, full_shape[1]))
    for x0, y0, x1, y1 in boxes[:, :4]:
        r0 = int(max(0, min(strided_shape[0] - 1, math.floor(y0 * scale_r))))
        r1 = int(max(0, min(strided_shape[0], math.ceil(y1 * scale_r))))
        c0 = int(max(0, min(strided_shape[1] - 1, math.floor(x0 * scale_c))))
        c1 = int(max(0, min(strided_shape[1], math.ceil(x1 * scale_c))))
        if r1 <= r0 or c1 <= c0:
            continue
        # SHRINK a loose box to the saliency support's bounding box (the
        # detector says where the instance is; the outline says how big it
        # actually is), then vote the full rectangle as usual so the anchor
        # keeps its vote density. Shrinking only applies when the support
        # is substantial and reaches the base region (glint-only support
        # would hang the box in mid-air).
        if support_full is not None:
            box_support = np.zeros(strided_shape, dtype=bool)
            box_support[r0:r1, c0:c1] = support_full[r0:r1, c0:c1]
            if int(np.count_nonzero(box_support)) >= 20:
                support_rows = np.nonzero(box_support.any(axis=1))[0]
                base_reach = r1 - max(2, int(round(0.2 * (r1 - r0))))
                if int(support_rows[-1]) >= base_reach:
                    support_cols = np.nonzero(box_support.any(axis=0))[0]
                    r0 = max(r0, int(support_rows[0]) - 1)
                    r1 = min(r1, int(support_rows[-1]) + 2)
                    c0 = max(c0, int(support_cols[0]) - 1)
                    c1 = min(c1, int(support_cols[-1]) + 2)
        width_px = max(1, c1 - c0)
        height_px = max(1, r1 - r0)
        aspect = height_px / width_px
        if y0 <= 2.0:
            # the box top is cropped by the frame: the missing rows ARE the
            # object's height, so this is a standing object regardless of
            # the (distorted) visible aspect
            band_fraction = DETECTOR_BOX_BOTTOM_BAND_FRACTION
        elif aspect >= DETECTOR_BOX_TALL_ASPECT:
            band_fraction = DETECTOR_BOX_BOTTOM_BAND_FRACTION
        elif aspect <= DETECTOR_BOX_FLAT_ASPECT:
            band_fraction = 1.0
        else:
            span = DETECTOR_BOX_TALL_ASPECT - DETECTOR_BOX_FLAT_ASPECT
            blend = (DETECTOR_BOX_TALL_ASPECT - aspect) / span
            band_fraction = DETECTOR_BOX_BOTTOM_BAND_FRACTION + blend * (
                1.0 - DETECTOR_BOX_BOTTOM_BAND_FRACTION
            )
        band_top_r = int(max(r0, math.floor(r1 - band_fraction * (r1 - r0))))
        weights[r0:r1, c0:c1] = np.maximum(
            weights[r0:r1, c0:c1], DETECTOR_BOX_BODY_WEIGHT
        )
        weights[band_top_r:r1, c0:c1] = FIELD_WEIGHT_RGB
    return weights


def sample_rgb_to_shape(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(rgb)
    if values.shape[:2] == tuple(shape):
        return values
    rows = np.clip(
        np.round(np.linspace(0, values.shape[0] - 1, shape[0])).astype(int),
        0,
        values.shape[0] - 1,
    )
    cols = np.clip(
        np.round(np.linspace(0, values.shape[1] - 1, shape[1])).astype(int),
        0,
        values.shape[1] - 1,
    )
    return values[rows][:, cols]


def _footprint_overlaps_occluder(
    footprint: TableFootprint,
    occluders_xyhalf_sizes: list[tuple[float, float, float, float]],
    min_fraction: float = 0.30,
) -> bool:
    """Axis-aligned overlap test between a proxy footprint and surviving
    real hypotheses: a big object's hole-derived proxy center can sit
    several cm behind the real's center (outside the center-distance
    suppression margin) while the rectangles clearly cover the same object.
    Overlap >= min_fraction of the SMALLER rectangle = same object."""
    fp_x, fp_y = float(footprint.center_xy_m[0]), float(footprint.center_xy_m[1])
    fp_hx = 0.5 * float(footprint.size_x_m)
    fp_hy = 0.5 * float(footprint.size_y_m)
    fp_area = max(1e-9, 4.0 * fp_hx * fp_hy)
    for ox, oy, ohx, ohy in occluders_xyhalf_sizes:
        ix = max(0.0, min(fp_x + fp_hx, ox + ohx) - max(fp_x - fp_hx, ox - ohx))
        iy = max(0.0, min(fp_y + fp_hy, oy + ohy) - max(fp_y - fp_hy, oy - ohy))
        inter = ix * iy
        smaller = min(fp_area, max(1e-9, 4.0 * ohx * ohy))
        if inter / smaller >= min_fraction:
            return True
    return False


def _near_any_occluder(
    center_x: float,
    center_y: float,
    occluders_xy_half: list[tuple[float, float, float]],
    margin_m: float,
) -> bool:
    for occluder_x, occluder_y, occluder_half in occluders_xy_half:
        if math.hypot(occluder_x - center_x, occluder_y - center_y) <= float(
            occluder_half
        ) + float(margin_m):
            return True
    return False


def table_plane_self_check(
    depth_m: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    workspace: tuple[float, float, float, float],
) -> str | None:
    """One-shot startup diagnostic: fit the observed table plane in base
    coordinates and compare against the configured table height. Returns a
    human-readable warning string when the rig looks mis-calibrated (bumped
    camera, wrong TABLE_Z_M / CAMERA_PITCH), else None. Log-only helper —
    it never changes pipeline behavior."""
    points = base_points_from_depth(
        depth_m=depth_m,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
    )
    x_map, y_map, z_map = points[:, :, 0], points[:, :, 1], points[:, :, 2]
    near_table = (
        np.isfinite(z_map)
        & (x_map >= workspace[0])
        & (x_map <= workspace[1])
        & (y_map >= workspace[2])
        & (y_map <= workspace[3])
        & (np.abs(z_map - table_z_m) < 0.06)
    )
    if int(np.count_nonzero(near_table)) < 5000:
        return "table self-check: not enough table pixels to fit a plane"
    xs, ys, zs = x_map[near_table], y_map[near_table], z_map[near_table]
    basis = np.column_stack([xs, ys, np.ones(xs.shape[0])])
    coef, *_ = np.linalg.lstsq(basis, zs, rcond=None)
    slope_a, slope_b, intercept = (float(v) for v in coef)
    tilt_deg = math.degrees(math.atan(math.hypot(slope_a, slope_b)))
    center_z = slope_b * 0.5 * (workspace[2] + workspace[3]) + intercept
    offset_mm = (center_z - table_z_m) * 1000.0
    if abs(offset_mm) <= 4.0 and tilt_deg <= 1.0:
        return None
    problems = []
    if abs(offset_mm) > 4.0:
        problems.append(
            f"height offset {offset_mm:+.1f}mm -> suggest TABLE_Z_M={center_z:.4f}"
        )
    if tilt_deg > 1.0:
        problems.append(
            f"residual tilt {tilt_deg:.2f} deg -> camera PITCH/ROLL likely off"
            " (was the camera bumped?)"
        )
    return "TABLE PLANE SELF-CHECK: " + "; ".join(problems)


def dense_occluders_xy_half(
    message: GeometryHypothesisArray,
    exclude_seeds_xy: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float, float]]:
    """Occluders CONFIRMED by dense evidence (same density fingerprint the
    refraction filter trusts). Sparse flickering boxes (refraction debris,
    hand-motion transients) must not clear field mass or block votes —
    only the per-frame attribution gates may use those.

    An otherwise-dense hypothesis sitting AT an instance seed is the detected
    transparent object's own partial return (dark scenes: the stronger dot
    contrast resolves patches of the shell itself) — clearing field mass
    there would erase the instance's own evidence, so it is excluded."""
    confirmed: list[tuple[float, float, float]] = []
    for hypothesis in message.hypotheses:
        provenance = str(hypothesis.provenance)
        if "hole_existence_only" in provenance:
            continue
        area_match = re.search(r"component_area_px=(\d+)", provenance)
        coverage_match = re.search(r"coverage=([\d.]+)", provenance)
        if (
            area_match is None
            or coverage_match is None
            or int(area_match.group(1)) < REFRACTION_ARTIFACT_MAX_AREA_PX
            or float(coverage_match.group(1)) < REFRACTION_ARTIFACT_MAX_COVERAGE
        ):
            continue
        center_x = float(hypothesis.pose_base.pose.position.x)
        center_y = float(hypothesis.pose_base.pose.position.y)
        half = 0.5 * max(
            float(hypothesis.dimensions_m.x), float(hypothesis.dimensions_m.y)
        )
        if any(
            math.hypot(center_x - seed_x, center_y - seed_y) <= half + 0.04
            for seed_x, seed_y in (exclude_seeds_xy or [])
        ):
            continue
        confirmed.append((center_x, center_y, half))
    return confirmed


def transparent_evidence_votes(
    *,
    depth_m: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    occluders_xy_half: list[tuple[float, float, float]],
    workspace: tuple[float, float, float, float],
    min_area_px: int = TRANSPARENT_HOLE_MIN_AREA_PX,
    rgb_saliency_mask: np.ndarray | None = None,
    rgb_instance_evidence: bool = False,
    confirmed_occluders_xy_half: list[tuple[float, float, float]] | None = None,
    instance_seeds_xy: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ABLATION_NO_INSTANCE:
        rgb_instance_evidence = False
        instance_seeds_xy = None
    """Per-frame transparency evidence as weighted table-plane votes.

    Every cue casts (x, y, weight) votes for the evidence accumulator:
    - sparse raised specks / below-table refraction patches: measured 3D
      points ON the object — highest weight;
    - table holes with nothing on their camera side: only the CAMERA-SIDE
      EDGE band votes (the one part of a displaced stereo shadow anchored
      to the object base); the hole body argues existence via mass only;
    - RGB saliency over table/hole pixels: outline and glint votes at the
      object, lower weight (optical imprints can be displaced by the lamp).
    Gates (image-border truncation, occluder adjacency, ring embeddedness,
    shadow-cone explanation) filter votes; classification mistakes cost one
    frame of votes, never a track. Returns (anchor_xyw, hole_xyw, rgb_xyw,
    bare_xy — a subsample of pixels where plain table is OBSERVED, used to
    fast-forget stale field mass):
    anchors are measured points ON the object, hole votes argue existence,
    RGB votes are the camera-side (base-contact) band of each saliency
    component — the one part of a silhouette whose table ray-cast is not
    displaced behind a tall object.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    hole = ~np.isfinite(depth) | (depth <= 0.0)
    camera_xy = (
        float(camera_to_base_transform.transform.translation.x),
        float(camera_to_base_transform.transform.translation.y),
    )
    points_base = base_points_from_depth(
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
    )
    x_map = points_base[:, :, 0]
    y_map = points_base[:, :, 1]
    z_map = points_base[:, :, 2]
    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
        & np.isfinite(x_map)
        & np.isfinite(y_map)
        & np.isfinite(z_map)
    )
    in_workspace = (
        valid
        & (x_map >= workspace[0])
        & (x_map <= workspace[1])
        & (y_map >= workspace[2])
        & (y_map <= workspace[3])
    )
    height = z_map - float(table_z_m)
    border_ring = np.zeros_like(hole)
    border_ring[:3, :] = True
    border_ring[-3:, :] = True
    border_ring[:, :3] = True
    border_ring[:, -3:] = True
    # a standing object's TOP legitimately leaves the frame; only a base
    # band cut by the bottom/left/right border is an untrustworthy footprint
    base_border = np.zeros_like(hole)
    base_border[-3:, :] = True
    base_border[:, :3] = True
    base_border[:, -3:] = True

    anchor_votes: list[np.ndarray] = []
    hole_votes: list[np.ndarray] = []
    rgb_votes: list[np.ndarray] = []

    def _outside_occluders(points_xy: np.ndarray) -> np.ndarray:
        # transparency evidence cannot live INSIDE a confirmed opaque
        # object: votes landing there (e.g. a detector-box base band
        # optically overlapped by a nearer object) are pollution
        keep = np.ones(points_xy.shape[0], dtype=bool)
        for occluder_x, occluder_y, occluder_half in (
            confirmed_occluders_xy_half or []
        ):
            keep &= (
                np.hypot(
                    points_xy[:, 0] - float(occluder_x),
                    points_xy[:, 1] - float(occluder_y),
                )
                > float(occluder_half) + TRANSPARENT_VOTE_OCCLUDER_CLEARANCE_M
            )
        return keep

    def _append(bucket: list[np.ndarray], points_xy: np.ndarray, weight: float) -> None:
        points = np.asarray(points_xy, dtype=np.float64)
        points = points[np.all(np.isfinite(points), axis=1)]
        if points.shape[0]:
            bucket.append(
                np.column_stack([points, np.full(points.shape[0], float(weight))])
            )

    # --- below-table patches: light bent through something transparent ---
    below = (
        in_workspace
        & (height < -float(TRANSPARENT_BELOW_TABLE_M))
        & (height > -float(TRANSPARENT_BELOW_TABLE_MAX_M))
    )
    below_evidence = np.zeros_like(hole)
    for component in extract_components(below, min_area_px=int(TRANSPARENT_BELOW_MIN_AREA_PX)):
        component_mask = np.asarray(component.mask, dtype=bool)
        if bool(np.any(component_mask & border_ring)):
            # truncated by the FOV: out-of-frame structure, not tabletop
            continue
        center_x = float(np.median(x_map[component_mask]))
        center_y = float(np.median(y_map[component_mask]))
        if _near_any_occluder(
            center_x, center_y, occluders_xy_half, TRANSPARENT_NEAR_OPAQUE_MARGIN_M
        ):
            # multipath at an opaque object's edge, not refraction
            continue
        below_evidence |= component_mask
        _append(
            anchor_votes,
            np.column_stack([x_map[component_mask], y_map[component_mask]]),
            FIELD_WEIGHT_DIRECT,
        )

    # --- sparse raised specks: specular returns off the transparent rim ---
    speck = (
        in_workspace
        & (height > float(TRANSPARENT_SPECK_MIN_HEIGHT_M))
        & (height < 0.2)
    )
    speck_evidence = np.zeros_like(hole)
    for component in extract_components(speck, min_area_px=int(TRANSPARENT_SPECK_MIN_AREA_PX)):
        if int(component.area_px) > int(TRANSPARENT_SPECK_MAX_AREA_PX):
            # far too solid for glints: an unmodeled opaque object, owned
            # by the regular pipeline
            continue
        component_mask = np.asarray(component.mask, dtype=bool)
        if bool(np.any(component_mask & border_ring)):
            # truncated by the FOV: out-of-frame structure, not tabletop
            continue
        center_x = float(np.median(x_map[component_mask]))
        center_y = float(np.median(y_map[component_mask]))
        if _near_any_occluder(
            center_x, center_y, occluders_xy_half, TRANSPARENT_NEAR_OPAQUE_MARGIN_M
        ):
            # flying pixels hug opaque silhouettes
            continue
        speck_evidence |= component_mask
        _append(
            anchor_votes,
            np.column_stack([x_map[component_mask], y_map[component_mask]]),
            FIELD_WEIGHT_DIRECT,
        )

    near_table = valid & (np.abs(height) < 0.006)

    # --- RGB saliency: outline/glints where the table should be ---
    # the imprint may sit exactly inside the object's own stereo hole (no
    # valid depth there), so hole pixels qualify too; their position comes
    # from the table-plane ray-cast like any hole pixel
    rgb_evidence = np.zeros_like(hole)
    if rgb_saliency_mask is not None:
        rgb_values = np.asarray(rgb_saliency_mask)
        if rgb_values.dtype == bool:
            rgb_weights = rgb_values.astype(np.float32) * FIELD_WEIGHT_RGB
        else:
            rgb_weights = rgb_values.astype(np.float32)
        rgb_candidate = (rgb_weights > 0.0) & ((near_table & in_workspace) | hole)
        linked = rgb_candidate
        for _ in range(int(RGB_SALIENCY_LINK_PX)):
            linked = _binary_dilate_1px(linked)
        for component in extract_components(linked, min_area_px=1):
            component_mask = np.asarray(component.mask, dtype=bool) & rgb_candidate
            if int(np.count_nonzero(component_mask)) < int(RGB_SALIENCY_MIN_AREA_PX):
                continue
            if rgb_instance_evidence:
                base_part = component_mask & (rgb_weights >= FIELD_WEIGHT_RGB - 1e-3)
                if bool(np.any(base_part & base_border)):
                    # the base contact band itself is cut by the FOV
                    continue
            elif bool(np.any(component_mask & border_ring)):
                # truncated by the FOV: out-of-frame structure, not tabletop
                continue
            measured_part = component_mask & near_table
            hole_part = component_mask & ~valid
            projected = np.empty((0, 3))
            if np.any(hole_part):
                projected = table_points_from_mask(
                    mask=hole_part,
                    intrinsics=intrinsics,
                    camera_to_base_transform=camera_to_base_transform,
                    table_z_m=float(table_z_m),
                )
            if np.any(measured_part):
                center_x = float(np.median(x_map[measured_part]))
                center_y = float(np.median(y_map[measured_part]))
            elif projected.shape[0] >= 4:
                center_x = float(np.median(projected[:, 0]))
                center_y = float(np.median(projected[:, 1]))
            else:
                continue
            if not (
                workspace[0] <= center_x <= workspace[1]
                and workspace[2] <= center_y <= workspace[3]
            ):
                continue
            if not rgb_instance_evidence and _near_any_occluder(
                center_x, center_y, occluders_xy_half, TRANSPARENT_NEAR_OPAQUE_MARGIN_M
            ):
                # anonymous saliency near an opaque object is its own
                # outline/penumbra; a detector instance box is not
                continue
            rgb_evidence |= component_mask
            if rgb_instance_evidence:
                # the weight mask already encodes base-band emphasis
                if np.any(measured_part):
                    points = np.column_stack(
                        [
                            x_map[measured_part],
                            y_map[measured_part],
                            rgb_weights[measured_part],
                        ]
                    )
                    points = points[np.all(np.isfinite(points), axis=1)]
                    if points.shape[0]:
                        points = points[_outside_occluders(points[:, 0:2])]
                    if points.shape[0]:
                        rgb_votes.append(points)
                for part_weight in (FIELD_WEIGHT_RGB, DETECTOR_BOX_BODY_WEIGHT):
                    part = hole_part & np.isclose(rgb_weights, part_weight)
                    if not np.any(part):
                        continue
                    part_projected = table_points_from_mask(
                        mask=part,
                        intrinsics=intrinsics,
                        camera_to_base_transform=camera_to_base_transform,
                        table_z_m=float(table_z_m),
                    )
                    if part_projected.shape[0]:
                        part_points = part_projected[:, 0:2]
                        part_points = part_points[_outside_occluders(part_points)]
                        if part_points.shape[0]:
                            _append(rgb_votes, part_points, part_weight)
                continue
            component_points: list[np.ndarray] = []
            if np.any(measured_part):
                component_points.append(
                    np.column_stack([x_map[measured_part], y_map[measured_part]])
                )
            if projected.shape[0]:
                component_points.append(projected[:, 0:2])
            merged = np.vstack(component_points)
            merged = merged[np.all(np.isfinite(merged), axis=1)]
            if merged.shape[0] < 4:
                continue
            # a tall object's silhouette ray-casts BEHIND its base; only the
            # camera-side (base contact) band is positionally trustworthy
            toward = np.asarray(
                [camera_xy[0] - center_x, camera_xy[1] - center_y], dtype=np.float64
            )
            norm = float(np.linalg.norm(toward))
            if norm > 1e-6:
                toward /= norm
                projections = (
                    (merged[:, 0] - center_x) * toward[0]
                    + (merged[:, 1] - center_y) * toward[1]
                )
                band = merged[
                    projections >= np.percentile(projections, RGB_FRONT_BAND_QUANTILE)
                ]
                if band.shape[0] >= 4:
                    merged = band
            _append(rgb_votes, merged, FIELD_WEIGHT_RGB)

    # a fresh detector box is affirmative instance evidence: holes it covers
    # belong to the transparent object even if an occluder also shadows there
    instance_region = None
    if rgb_instance_evidence and rgb_saliency_mask is not None:
        instance_values = np.asarray(rgb_saliency_mask)
        if instance_values.dtype == bool:
            instance_region = instance_values
        else:
            instance_region = instance_values.astype(np.float32) > 0.0

    # --- hole components: nothing on the camera side may explain them ---
    for hole_component in extract_components(hole, min_area_px=int(min_area_px)):
        component_mask = np.asarray(hole_component.mask, dtype=bool)
        border_fraction = float(
            np.count_nonzero(component_mask & border_ring)
        ) / float(max(1, hole_component.area_px))
        if border_fraction > 0.15:
            # a depth-FOV band hugs the image border; a real object's hole
            # merely touches it
            continue
        grown = component_mask
        for _ in range(3):
            grown = _binary_dilate_1px(grown)
        ring = grown & ~component_mask
        ring_supported = ring & (
            near_table | below_evidence | speck_evidence | rgb_evidence
        )
        ring_fraction = float(np.count_nonzero(ring_supported)) / float(
            max(1, np.count_nonzero(ring))
        )
        if ring_fraction < TRANSPARENT_HOLE_RING_TABLE_FRACTION:
            # a tabletop object's hole is embedded in the table; a hole
            # bordering the void past the table edge is a FOV artifact
            continue
        table_points = table_points_from_mask(
            mask=hole_component.mask,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=float(table_z_m),
        )
        if table_points.shape[0] < 12:
            continue
        center_x, center_y = (
            float(np.median(table_points[:, 0])),
            float(np.median(table_points[:, 1])),
        )
        if not (
            workspace[0] <= center_x <= workspace[1]
            and workspace[2] <= center_y <= workspace[3]
        ):
            continue
        toward_camera = np.asarray(
            [camera_xy[0] - center_x, camera_xy[1] - center_y], dtype=np.float64
        )
        camera_norm = float(np.linalg.norm(toward_camera))
        if camera_norm < 1e-6:
            continue
        toward_camera /= camera_norm
        span_x = float(
            np.percentile(table_points[:, 0], 98) - np.percentile(table_points[:, 0], 2)
        )
        span_y = float(
            np.percentile(table_points[:, 1], 98) - np.percentile(table_points[:, 1], 2)
        )
        half_diag = 0.5 * math.hypot(span_x, span_y)
        instance_claimed = False
        if instance_region is not None:
            overlap = float(
                np.count_nonzero(component_mask & instance_region)
            ) / float(max(1, hole_component.area_px))
            instance_claimed = overlap >= TRANSPARENT_HOLE_INSTANCE_OVERLAP

        def _occluder_owns_instance(occluder_x, occluder_y, occluder_half) -> bool:
            # the detector box belongs to THIS occluder (e.g. a specular
            # cylinder the detector also fires on): its own displaced hole
            # is its own depth noise, not a hidden transparent object
            for seed_x, seed_y in instance_seeds_xy or []:
                if (
                    math.hypot(occluder_x - seed_x, occluder_y - seed_y)
                    <= float(occluder_half) + 0.04
                ):
                    return True
            return False
        explained = False
        for occluder_x, occluder_y, occluder_half in occluders_xy_half:
            offset = np.asarray(
                [occluder_x - center_x, occluder_y - center_y], dtype=np.float64
            )
            distance = float(np.linalg.norm(offset))
            if distance <= float(occluder_half) + 0.025:
                # the hole sits inside an opaque object's own footprint
                # (e.g. a see-through patch inside a glass bowl's circle)
                explained = True
                break
            if instance_claimed and not _occluder_owns_instance(
                occluder_x, occluder_y, occluder_half
            ):
                # the detector box claims this hole for a transparent
                # instance: a FOREIGN occluder's shadow-cone attribution
                # does not apply (the box outranks the shadow ambiguity)
                continue
            if distance < 1e-6 or distance > half_diag + TRANSPARENT_HOLE_EXPLAIN_DISTANCE_M:
                continue
            cosine = max(-1.0, min(1.0, float(offset @ toward_camera) / distance))
            angle_deg = math.degrees(math.acos(cosine))
            # a shadow is a wedge, not a ray: widen the cone by the angular
            # half-width the occluder subtends at this distance
            allowed_deg = TRANSPARENT_HOLE_EXPLAIN_CONE_DEG + math.degrees(
                math.atan2(float(occluder_half), distance)
            )
            if angle_deg <= allowed_deg:
                explained = True
                break
        if explained:
            continue
        # only the camera-side edge band is anchored to the object base;
        # the rest of the hole is its displaced stereo shadow
        projections = (
            (table_points[:, 0] - center_x) * toward_camera[0]
            + (table_points[:, 1] - center_y) * toward_camera[1]
        )
        band = table_points[
            projections >= np.percentile(projections, TRANSPARENT_HOLE_FRONT_BAND_QUANTILE)
        ]
        if band.shape[0] >= 4:
            band_points = band[:, 0:2]
            band_points = band_points[_outside_occluders(band_points)]
            if band_points.shape[0] >= 4:
                _append(hole_votes, band_points, FIELD_WEIGHT_HOLE_EDGE)

    if ABLATION_NO_ANCHOR:
        anchor_votes = []
    if ABLATION_NO_HOLE:
        hole_votes = []
    anchor = np.vstack(anchor_votes) if anchor_votes else np.empty((0, 3))
    hole_out = np.vstack(hole_votes) if hole_votes else np.empty((0, 3))
    rgb = np.vstack(rgb_votes) if rgb_votes else np.empty((0, 3))
    observed_bare = (
        near_table
        & in_workspace
        & ~hole
        & ~below_evidence
        & ~speck_evidence
        & ~rgb_evidence
    )
    bare_rows, bare_columns = np.nonzero(observed_bare[::2, ::2])
    bare_xy = np.column_stack(
        [
            x_map[::2, ::2][bare_rows, bare_columns],
            y_map[::2, ::2][bare_rows, bare_columns],
        ]
    )
    return anchor, hole_out, rgb, bare_xy


class TransparentEvidenceField:
    """Time-integrated transparency evidence on a table-plane grid.

    Votes accumulate with weight x dt and decay with a TIME constant, so
    the steady-state mass is publish-rate independent and replay matches
    the live node. Footprints are read out with hysteresis (high birth
    mass, low sustain mass) which replaces per-frame streak matching:
    single-frame evidence fluctuations only ripple the field.
    """

    MAX_SPAN_M = 2.0  # unset workspaces default to +-10m; cap the grid

    def __init__(
        self,
        workspace: tuple[float, float, float, float],
        cell_m: float = FIELD_CELL_M,
        tau_s: float = FIELD_TAU_S,
    ) -> None:
        self.cell_m = float(cell_m)
        self.tau_s = float(tau_s)
        span_x = min(float(workspace[1] - workspace[0]), self.MAX_SPAN_M)
        span_y = min(float(workspace[3] - workspace[2]), self.MAX_SPAN_M)
        self.x0 = 0.5 * float(workspace[0] + workspace[1]) - 0.5 * span_x
        self.y0 = 0.5 * float(workspace[2] + workspace[3]) - 0.5 * span_y
        columns = max(2, int(math.ceil(span_x / self.cell_m)))
        rows = max(2, int(math.ceil(span_y / self.cell_m)))
        self.anchor_mass = np.zeros((rows, columns), dtype=np.float32)
        self.hole_mass = np.zeros((rows, columns), dtype=np.float32)
        self.rgb_mass = np.zeros((rows, columns), dtype=np.float32)
        self.active = np.zeros((rows, columns), dtype=bool)
        self.anchor_active = np.zeros((rows, columns), dtype=bool)
        self.yaw_grid = np.zeros((rows, columns), dtype=np.float32)
        self.yaw_valid = np.zeros((rows, columns), dtype=bool)

    def _deposit(self, field: np.ndarray, votes_xyw: np.ndarray, dt_s: float) -> None:
        votes = np.asarray(votes_xyw, dtype=np.float64)
        if votes.ndim != 2 or votes.shape[0] == 0:
            return
        columns = np.floor((votes[:, 0] - self.x0) / self.cell_m).astype(int)
        rows = np.floor((votes[:, 1] - self.y0) / self.cell_m).astype(int)
        keep = (
            (rows >= 0)
            & (rows < field.shape[0])
            & (columns >= 0)
            & (columns < field.shape[1])
            & np.isfinite(votes[:, 2])
        )
        np.add.at(
            field,
            (rows[keep], columns[keep]),
            (votes[keep, 2] * float(dt_s)).astype(np.float32),
        )

    def _cells_of(self, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_xy, dtype=np.float64)
        columns = np.floor((points[:, 0] - self.x0) / self.cell_m).astype(int)
        rows = np.floor((points[:, 1] - self.y0) / self.cell_m).astype(int)
        keep = (
            (rows >= 0)
            & (rows < self.anchor_mass.shape[0])
            & (columns >= 0)
            & (columns < self.anchor_mass.shape[1])
        )
        return rows[keep], columns[keep]

    def update(
        self,
        anchor_xyw: np.ndarray,
        hole_xyw: np.ndarray,
        rgb_xyw: np.ndarray,
        dt_s: float,
        bare_xy: np.ndarray | None = None,
        occluders_xy_half: list[tuple[float, float, float]] | None = None,
    ) -> None:
        dt_s = float(min(0.5, max(0.005, dt_s)))
        decay = math.exp(-dt_s / self.tau_s)
        self.anchor_mass *= decay
        self.hole_mass *= decay
        self.rgb_mass *= decay
        fast = math.exp(-dt_s / FIELD_BARE_TAU_S) / max(decay, 1e-9)
        if bare_xy is not None and np.asarray(bare_xy).size:
            # we can SEE plain table in these cells right now: transparency
            # mass there is stale and forgets on a much faster clock
            rows, columns = self._cells_of(np.asarray(bare_xy, dtype=np.float64))
            if rows.size:
                self.anchor_mass[rows, columns] *= fast
                self.hole_mass[rows, columns] *= fast
                self.rgb_mass[rows, columns] *= fast
        if occluders_xy_half:
            # a CONFIRMED opaque object stands in these cells: transparency
            # mass there is pollution (e.g. a placement burst voted through
            # the occluder's displaced stereo hole) — same fast clock as bare
            columns_grid = self.x0 + (
                np.arange(self.anchor_mass.shape[1], dtype=np.float64) + 0.5
            ) * self.cell_m
            rows_grid = self.y0 + (
                np.arange(self.anchor_mass.shape[0], dtype=np.float64) + 0.5
            ) * self.cell_m
            for occluder_x, occluder_y, occluder_half in occluders_xy_half:
                reach = float(occluder_half) + TRANSPARENT_VOTE_OCCLUDER_CLEARANCE_M
                column_delta = columns_grid - float(occluder_x)
                row_delta = rows_grid - float(occluder_y)
                if (
                    float(np.min(np.abs(column_delta))) > reach
                    or float(np.min(np.abs(row_delta))) > reach
                ):
                    continue
                zone = (
                    row_delta[:, None] ** 2 + column_delta[None, :] ** 2
                ) <= reach * reach
                self.anchor_mass[zone] *= fast
                self.hole_mass[zone] *= fast
                self.rgb_mass[zone] *= fast
        self._deposit(self.anchor_mass, anchor_xyw, dt_s)
        self._deposit(self.hole_mass, hole_xyw, dt_s)
        self._deposit(self.rgb_mass, rgb_xyw, dt_s)

    def extract(
        self,
        camera_xy: tuple[float, float],
        instance_seeds_xy: list[tuple[float, float]] | None = None,
    ) -> list[TableFootprint]:
        if ABLATION_NO_INSTANCE:
            instance_seeds_xy = None
        anchor = _box_blur(self.anchor_mass + self.rgb_mass, 1)
        depth = _box_blur(self.anchor_mass + self.hole_mass, 1)
        total = _box_blur(self.anchor_mass + self.hole_mass + self.rgb_mass, 1)
        sustain = total > FIELD_SUSTAIN_MASS
        if not np.any(sustain):
            self.active[:] = False
            self.anchor_active[:] = False
            return []
        results: list[TableFootprint] = []
        new_active = np.zeros_like(sustain)
        new_anchor_active = np.zeros_like(sustain)
        new_yaw_grid = np.zeros_like(self.yaw_grid)
        new_yaw_valid = np.zeros_like(self.yaw_valid)
        region_masks: list[np.ndarray] = []
        region_seeds: list[tuple[float, float] | None] = []
        region_unions: list[bool] = []
        for region in extract_components(sustain, min_area_px=4):
            whole_mask = np.asarray(region.mask, dtype=bool)
            # detector boxes are INSTANCE evidence: a connected mass region
            # covering two seeds is two objects bridged by junk evidence —
            # split cells by nearest seed (splitting is always 乌龙-safe)
            split = False
            if instance_seeds_xy and len(instance_seeds_xy) >= 2:
                rows, columns = np.nonzero(whole_mask)
                cell_x = self.x0 + (columns.astype(np.float64) + 0.5) * self.cell_m
                cell_y = self.y0 + (rows.astype(np.float64) + 0.5) * self.cell_m
                near_seeds = [
                    (sx, sy)
                    for sx, sy in instance_seeds_xy
                    if float(
                        np.min(np.hypot(cell_x - sx, cell_y - sy))
                    ) <= FIELD_SEED_REACH_M
                ]
                if len(near_seeds) >= 2:
                    distances = np.stack(
                        [np.hypot(cell_x - sx, cell_y - sy) for sx, sy in near_seeds]
                    )
                    assignment = np.argmin(distances, axis=0)
                    for seed_index in range(len(near_seeds)):
                        part = np.zeros_like(whole_mask)
                        keep = assignment == seed_index
                        part[rows[keep], columns[keep]] = True
                        if int(np.count_nonzero(part)) >= 4:
                            region_masks.append(part)
                            region_seeds.append(near_seeds[seed_index])
                            region_unions.append(False)
                    split = True
            if not split:
                region_masks.append(whole_mask)
                region_seeds.append(None)
                region_unions.append(False)
        # the dual of splitting: one detector box is INSTANCE evidence of
        # one-ness, so mass fragments that all sit within seed reach of the
        # SAME single seed are one object whose evidence happens to be
        # shattered (far/oblique holes fragment below the pinpoint floor).
        # Fragments near two seeds stay untouched (ambiguous = no merge).
        if instance_seeds_xy:
            # ADOPTION: a region whose cells reach exactly one instance seed
            # belongs to that instance — this both enables same-seed fragment
            # rescue below and puts the region under the seed-consistency
            # gate (its footprint may not land >reach from its own seed;
            # kills silhouette-shadow smears behind detected objects)
            for index, (mask, seed) in enumerate(zip(region_masks, region_seeds)):
                if seed is not None:
                    continue
                rows, columns = np.nonzero(mask)
                cell_x = self.x0 + (columns.astype(np.float64) + 0.5) * self.cell_m
                cell_y = self.y0 + (rows.astype(np.float64) + 0.5) * self.cell_m
                near = [
                    (sx, sy)
                    for sx, sy in instance_seeds_xy
                    if float(
                        np.min(np.hypot(cell_x - sx, cell_y - sy))
                    ) <= FIELD_SEED_REACH_M
                ]
                if len(near) == 1:
                    region_seeds[index] = near[0]
            seed_groups: dict[tuple[float, float], list[int]] = {}
            for index, seed in enumerate(region_seeds):
                if seed is not None:
                    seed_groups.setdefault(
                        (float(seed[0]), float(seed[1])), []
                    ).append(index)
            merged_away: set[int] = set()
            for seed_key, member_indices in seed_groups.items():
                if ABLATION_NO_MERGE:
                    break
                if len(member_indices) < 2:
                    continue
                # rescue-only: merging exists to save SHATTERED evidence.
                # If any member already spans a viable footprint on its own,
                # the object has a healthy core — leave the whole group
                # alone so proven single-region behavior is untouched.
                viable_core = False
                for member in member_indices:
                    member_mask = region_masks[member]
                    member_peak = float(total[member_mask].max())
                    selected = member_mask & (
                        total
                        >= max(
                            FIELD_SUSTAIN_MASS,
                            FIELD_FOOTPRINT_PEAK_FRACTION * member_peak,
                        )
                    )
                    rows, columns = np.nonzero(selected)
                    if rows.size == 0:
                        continue
                    span = max(
                        (rows.max() - rows.min() + 1) * self.cell_m,
                        (columns.max() - columns.min() + 1) * self.cell_m,
                    )
                    if span >= FIELD_UNION_CORE_SPAN_M:
                        viable_core = True
                        break
                if viable_core:
                    continue
                union = np.zeros_like(region_masks[member_indices[0]])
                for member in member_indices:
                    union |= region_masks[member]
                    merged_away.add(member)
                region_masks.append(union)
                region_seeds.append(seed_key)
                region_unions.append(True)
            if merged_away:
                region_masks = [
                    m for i, m in enumerate(region_masks) if i not in merged_away
                ]
                region_seeds = [
                    s for i, s in enumerate(region_seeds) if i not in merged_away
                ]
                region_unions = [
                    u for i, u in enumerate(region_unions) if i not in merged_away
                ]
        for region_mask, region_seed, region_union in zip(
            region_masks, region_seeds, region_unions
        ):
            peak = float(total[region_mask].max())
            was_active = bool(np.any(region_mask & self.active))
            if peak < FIELD_BIRTH_MASS and not was_active:
                continue
            if float(depth[region_mask].sum()) < FIELD_DIRECT_REGION_MIN:
                # RGB mass alone never births an object: depth proves
                # existence, RGB anchors position and extent
                continue
            if float(total[region_mask].sum()) < FIELD_REGION_TOTAL_MASS_MIN:
                # a pinpoint of persistent evidence (sensor artifact, table
                # mark) never reaches the mass a real object's evidence
                # sustains — the field equivalent of a per-frame area floor
                continue
            # position/extent come from ON-object evidence (specks,
            # refraction, RGB base band) whenever there is enough of it;
            # the hole mass only argues existence. The mode is sticky so a
            # borderline anchor mass cannot flip the footprint around.
            anchor_sum = float(anchor[region_mask].sum())
            anchor_was = bool(np.any(region_mask & self.anchor_active))
            anchor_bar = FIELD_ANCHOR_STAY_MASS if anchor_was else FIELD_ANCHOR_ENTER_MASS
            anchor_mode = anchor_sum >= anchor_bar

            def _footprint_from(
                shape_field: np.ndarray, min_dim_m: float
            ) -> TableFootprint | None:
                shape_peak = float(shape_field[region_mask].max())
                selection_floor = (
                    FIELD_SUSTAIN_MASS
                    if region_union
                    else max(
                        FIELD_SUSTAIN_MASS,
                        FIELD_FOOTPRINT_PEAK_FRACTION * shape_peak,
                    )
                )
                selected = region_mask & (shape_field >= selection_floor)
                rows, columns = np.nonzero(selected)
                if rows.size < 4:
                    return None
                points = np.column_stack(
                    [
                        self.x0 + (columns.astype(np.float64) + 0.5) * self.cell_m,
                        self.y0 + (rows.astype(np.float64) + 0.5) * self.cell_m,
                    ]
                )
                candidate = table_footprint_from_points(points)
                if candidate is None:
                    return None
                largest = max(float(candidate.size_x_m), float(candidate.size_y_m))
                smallest = max(1e-6, min(float(candidate.size_x_m), float(candidate.size_y_m)))
                if largest < min_dim_m:
                    # a pinpoint (sensor artifact, table mark) never spreads
                    # like a real object's evidence does
                    return None
                if largest > TRANSPARENT_HOLE_MAX_DIM_M:
                    return None
                if largest / smallest >= FIELD_YAW_AMBIGUOUS_ASPECT:
                    prev_cells = region_mask & self.yaw_valid
                    if np.any(prev_cells):
                        prev_yaw = float(np.median(self.yaw_grid[prev_cells]))
                        cos_p = math.cos(prev_yaw)
                        sin_p = math.sin(prev_yaw)
                        local_x = cos_p * points[:, 0] + sin_p * points[:, 1]
                        local_y = -sin_p * points[:, 0] + cos_p * points[:, 1]
                        lower_x, upper_x = np.percentile(local_x, [5.0, 95.0])
                        lower_y, upper_y = np.percentile(local_y, [5.0, 95.0])
                        prev_width = max(0.005, float(upper_x - lower_x))
                        prev_depth = max(0.005, float(upper_y - lower_y))
                        new_area = float(candidate.size_x_m) * float(candidate.size_y_m)
                        if prev_width * prev_depth <= new_area * FIELD_YAW_KEEP_AREA_FRACTION:
                            mid_x = 0.5 * (float(lower_x) + float(upper_x))
                            mid_y = 0.5 * (float(lower_y) + float(upper_y))
                            candidate = TableFootprint(
                                center_xy_m=(
                                    cos_p * mid_x - sin_p * mid_y,
                                    sin_p * mid_x + cos_p * mid_y,
                                ),
                                size_x_m=prev_width,
                                size_y_m=prev_depth,
                                width_px=candidate.width_px,
                                height_px=candidate.height_px,
                                yaw_rad=prev_yaw,
                            )
                if largest / smallest < FIELD_YAW_AMBIGUOUS_ASPECT:
                    # a near-square footprint has no meaningful orientation:
                    # the oriented-bounds yaw is pure noise, so publish it
                    # axis-aligned instead of spinning every frame
                    span_x = float(np.percentile(points[:, 0], 98) - np.percentile(points[:, 0], 2))
                    span_y = float(np.percentile(points[:, 1], 98) - np.percentile(points[:, 1], 2))
                    candidate = TableFootprint(
                        center_xy_m=candidate.center_xy_m,
                        size_x_m=max(min_dim_m, span_x),
                        size_y_m=max(min_dim_m, span_y),
                        width_px=candidate.width_px,
                        height_px=candidate.height_px,
                        yaw_rad=0.0,
                    )
                return candidate

            # a small object's anchor band is legitimately narrow, so its
            # floor is lower; the pinpoint-artifact floor guards the
            # hole-dominated (total) mode where those artifacts live
            footprint = (
                _footprint_from(anchor, FIELD_ANCHOR_FOOTPRINT_MIN_DIM_M)
                if anchor_mode
                else None
            )
            if footprint is None:
                # anchor evidence too small/sparse to shape a footprint:
                # fall back to everything (holes included)
                anchor_mode = False
                footprint = _footprint_from(total, FIELD_FOOTPRINT_MIN_MAX_DIM_M)
            if footprint is None:
                continue
            if region_seed is not None and (
                math.hypot(
                    float(footprint.center_xy_m[0]) - region_seed[0],
                    float(footprint.center_xy_m[1]) - region_seed[1],
                )
                > FIELD_SEED_REACH_M
            ):
                # a partition whose footprint lands nowhere near its own
                # seed is the junk BRIDGE between two instances, not an
                # object
                continue
            results.append(footprint)
            new_yaw_grid[region_mask] = float(footprint.yaw_rad)
            new_yaw_valid[region_mask] = True
            new_active |= region_mask
            if anchor_mode or anchor_was:
                # LATCH: once a region anchored on measured evidence it
                # stays in anchor mode until the region itself dies —
                # mode flapping alternates compact/wide footprints
                new_anchor_active |= region_mask
        self.active = new_active
        self.anchor_active = new_anchor_active
        self.yaw_grid = new_yaw_grid
        self.yaw_valid = new_yaw_valid
        return _drop_shadowed_transparent_footprints(
            results, camera_xy, instance_seeds_xy=instance_seeds_xy
        )


def consolidate_instance_partial_returns(
    message: GeometryHypothesisArray,
    transparent_footprints: list[TableFootprint],
    instance_seeds_xy: list[tuple[float, float]] | None,
) -> GeometryHypothesisArray:
    """A transparent object with partial specular returns (glass rim/body,
    stronger in darkness) floods the REAL pipeline with small dense
    fragments. When a detector-box seed CONFIRMS the transparent instance
    and an active transparent footprint owns that seed, real hypotheses
    that are small relative to the footprint and sit inside it are that
    object's partial returns: the proxy already represents the object, so
    the fragments are dropped. A real hypothesis comparable in size to the
    footprint (e.g. the specular metal cylinder's single body fit) is NOT
    consolidated — it stays, and publish-time suppression keeps preferring
    it over the proxy as before."""
    if not transparent_footprints or not instance_seeds_xy:
        return message

    owned = []
    for footprint in transparent_footprints:
        near = [
            (sx, sy)
            for sx, sy in instance_seeds_xy
            if math.hypot(
                float(footprint.center_xy_m[0]) - sx,
                float(footprint.center_xy_m[1]) - sy,
            )
            <= FIELD_SEED_REACH_M
        ]
        if len(near) == 1:
            owned.append(footprint)
    if not owned:
        return message

    kept = []
    for hypothesis in message.hypotheses:
        if "hole_existence_only" in str(hypothesis.provenance):
            kept.append(hypothesis)
            continue
        center_x = float(hypothesis.pose_base.pose.position.x)
        center_y = float(hypothesis.pose_base.pose.position.y)
        max_dim = max(
            float(hypothesis.dimensions_m.x), float(hypothesis.dimensions_m.y)
        )
        consolidated = False
        for footprint in owned:
            fp_max = max(float(footprint.size_x_m), float(footprint.size_y_m))
            if max_dim >= 0.5 * fp_max:
                continue
            half_x = 0.5 * float(footprint.size_x_m) + 0.01
            half_y = 0.5 * float(footprint.size_y_m) + 0.01
            dx = center_x - float(footprint.center_xy_m[0])
            dy = center_y - float(footprint.center_xy_m[1])
            yaw = float(footprint.yaw_rad)
            local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
            local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
            if abs(local_x) <= half_x and abs(local_y) <= half_y:
                consolidated = True
                break
        if not consolidated:
            kept.append(hypothesis)
    if len(kept) == len(message.hypotheses):
        return message
    return copy_hypothesis_array_with_hypotheses(message, kept)


def drop_refraction_artifacts_behind_transparent(
    message: GeometryHypothesisArray,
    transparent_footprints: list[TableFootprint],
    camera_xy: tuple[float, float],
) -> GeometryHypothesisArray:
    """Small real hypotheses in the shadow wedge right behind a confirmed
    transparent footprint are refraction focus debris (light bent through
    the body lands behind it as sparse speckles that fit into small boxes
    and churn tracks). Bounded so a genuine object survives: only small
    (<=65mm), only close behind (<=12cm), only inside the shadow cone."""
    if not transparent_footprints:
        return message
    kept = []
    for hypothesis in message.hypotheses:
        if "hole_existence_only" in str(hypothesis.provenance):
            kept.append(hypothesis)
            continue
        max_dim = max(
            float(hypothesis.dimensions_m.x), float(hypothesis.dimensions_m.y)
        )
        if max_dim > REFRACTION_ARTIFACT_MAX_DIM_M:
            kept.append(hypothesis)
            continue
        provenance = str(hypothesis.provenance)
        area_match = re.search(r"component_area_px=(\d+)", provenance)
        coverage_match = re.search(r"coverage=([\d.]+)", provenance)
        if (
            area_match is not None
            and coverage_match is not None
            and int(area_match.group(1)) >= REFRACTION_ARTIFACT_MAX_AREA_PX
            and float(coverage_match.group(1)) >= REFRACTION_ARTIFACT_MAX_COVERAGE
        ):
            # dense, well-covered component: a real object regardless of
            # where it sits (refraction debris is sparse speckle)
            kept.append(hypothesis)
            continue
        center_x = float(hypothesis.pose_base.pose.position.x)
        center_y = float(hypothesis.pose_base.pose.position.y)
        toward_camera = np.asarray(
            [camera_xy[0] - center_x, camera_xy[1] - center_y], dtype=np.float64
        )
        camera_norm = float(np.linalg.norm(toward_camera))
        if camera_norm < 1e-6:
            kept.append(hypothesis)
            continue
        toward_camera /= camera_norm
        artifact = False
        for footprint in transparent_footprints:
            offset = np.asarray(
                [
                    footprint.center_xy_m[0] - center_x,
                    footprint.center_xy_m[1] - center_y,
                ],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(offset))
            if distance < 1e-6 or distance > REFRACTION_ARTIFACT_REACH_M:
                continue
            footprint_half = 0.5 * math.hypot(
                float(footprint.size_x_m), float(footprint.size_y_m)
            )
            cosine = max(-1.0, min(1.0, float(offset @ toward_camera) / distance))
            angle_deg = math.degrees(math.acos(cosine))
            allowed_deg = TRANSPARENT_HOLE_EXPLAIN_CONE_DEG + math.degrees(
                math.atan2(footprint_half, distance)
            )
            if angle_deg <= allowed_deg:
                artifact = True
                break
        if not artifact:
            kept.append(hypothesis)
    if len(kept) == len(message.hypotheses):
        return message
    return copy_hypothesis_array_with_hypotheses(message, kept)


def _drop_shadowed_transparent_footprints(
    footprints: list[TableFootprint],
    camera_xy: tuple[float, float],
    instance_seeds_xy: list[tuple[float, float]] | None = None,
) -> list[TableFootprint]:
    """A transparent object displaces its own stereo shadow away from the
    camera; leftover hole fragments behind an accepted footprint are that
    same object's shadow, not a second object. Keep the front footprint,
    drop candidates sitting in its shadow wedge.

    A candidate that OWNS a detector-box seed (exactly one seed within
    reach — near-two is a bridging blob, not ownership) is a confirmed
    separate instance: several transparent objects legitimately stand in
    each other's shadow wedges, and the wedge rule must not eat them."""
    if len(footprints) < 2:
        return footprints

    def _owns_seed(footprint: TableFootprint) -> bool:
        near = [
            (sx, sy)
            for sx, sy in instance_seeds_xy or []
            if math.hypot(
                float(footprint.center_xy_m[0]) - sx,
                float(footprint.center_xy_m[1]) - sy,
            )
            <= FIELD_SEED_REACH_M
        ]
        return len(near) == 1

    kept: list[TableFootprint] = []
    for candidate in footprints:
        if _owns_seed(candidate):
            kept.append(candidate)
            continue
        candidate_x, candidate_y = candidate.center_xy_m
        toward_camera = np.asarray(
            [camera_xy[0] - candidate_x, camera_xy[1] - candidate_y],
            dtype=np.float64,
        )
        camera_norm = float(np.linalg.norm(toward_camera))
        if camera_norm < 1e-6:
            kept.append(candidate)
            continue
        toward_camera /= camera_norm
        shadowed = False
        for other in footprints:
            if other is candidate:
                continue
            other_half = 0.5 * math.hypot(
                float(other.size_x_m), float(other.size_y_m)
            )
            offset = np.asarray(
                [other.center_xy_m[0] - candidate_x, other.center_xy_m[1] - candidate_y],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(offset))
            if distance < 1e-6 or distance > TRANSPARENT_HOLE_EXPLAIN_DISTANCE_M:
                continue
            cosine = max(-1.0, min(1.0, float(offset @ toward_camera) / distance))
            angle_deg = math.degrees(math.acos(cosine))
            allowed_deg = TRANSPARENT_HOLE_EXPLAIN_CONE_DEG + math.degrees(
                math.atan2(other_half, distance)
            )
            if angle_deg <= allowed_deg:
                shadowed = True
                break
        if not shadowed:
            kept.append(candidate)
    return kept


def transparent_hole_hypothesis(
    footprint: TableFootprint,
    *,
    base_frame_id: str,
    table_z_m: float,
    index: int,
    proxy_height_m: float = TRANSPARENT_HOLE_PROXY_HEIGHT_M,
) -> GeometryHypothesis:
    hypothesis = GeometryHypothesis()
    hypothesis.hypothesis_id = f"transparent_hole_{index}"
    hypothesis.shape_type = GeometryHypothesis.SHAPE_BOX
    height = max(0.01, float(proxy_height_m))
    hypothesis.pose_base = make_pose(
        str(base_frame_id),
        float(footprint.center_xy_m[0]),
        float(footprint.center_xy_m[1]),
        float(table_z_m) + 0.5 * height,
        yaw_orientation(float(footprint.yaw_rad)),
    )
    hypothesis.dimensions_m.x = max(0.01, float(footprint.size_x_m))
    hypothesis.dimensions_m.y = max(0.01, float(footprint.size_y_m))
    hypothesis.dimensions_m.z = height
    hypothesis.confidence = 0.2
    hypothesis.uncertainty = 0.8
    hypothesis.validation_state = GeometryHypothesis.VALIDATION_VALID
    hypothesis.provenance = (
        "transparent_hole_existence;hole_existence_only;"
        "support=unexplained_table_hole"
    )
    return hypothesis



HYPOTHESIS_FUSE_MIN_OVERLAP = 0.40


def fuse_overlapping_hypotheses(
    message: GeometryHypothesisArray,
    *,
    min_overlap_ratio: float = HYPOTHESIS_FUSE_MIN_OVERLAP,
) -> GeometryHypothesisArray:
    """Fuse hypotheses whose table footprints substantially overlap.

    Two distinct rigid objects cannot share table area: >=40% overlap of the
    smaller footprint means the two hypotheses are fragments of ONE object
    (e.g. the dome patch and the rim crescent of an upside-down glass bowl).
    Deliberately separated candidates (seeded splits) are never fused back.
    """

    def rect(h) -> tuple[float, float, float, float]:
        cx = float(h.pose_base.pose.position.x)
        cy = float(h.pose_base.pose.position.y)
        hx = 0.5 * float(h.dimensions_m.x)
        hy = 0.5 * float(h.dimensions_m.y)
        return cx - hx, cy - hy, cx + hx, cy + hy

    def overlap_ratio(a, b) -> float:
        ax0, ay0, ax1, ay1 = rect(a)
        bx0, by0, bx1, by1 = rect(b)
        iw = min(ax1, bx1) - max(ax0, bx0)
        ih = min(ay1, by1) - max(ay0, by0)
        if iw <= 0.0 or ih <= 0.0:
            return 0.0
        area_a = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
        area_b = max(1e-9, (bx1 - bx0) * (by1 - by0))
        return (iw * ih) / min(area_a, area_b)

    hypotheses = list(message.hypotheses)
    changed = True
    while changed:
        changed = False
        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                a, b = hypotheses[i], hypotheses[j]
                if "hole_existence_only" in str(a.provenance) or (
                    "hole_existence_only" in str(b.provenance)
                ):
                    # existence proxies are evidence markers, not fragments:
                    # a real neighbor must never swallow them (nor vice versa)
                    continue
                required = float(min_overlap_ratio)
                if "recent_track_seeded_split" in str(a.provenance) or (
                    "recent_track_seeded_split" in str(b.provenance)
                ):
                    # deliberate splits produce disjoint halves; only a
                    # degenerate split overlaps this heavily
                    required = 0.65
                if overlap_ratio(a, b) < required:
                    continue
                keeper, other = (
                    (a, b) if float(a.score.total) >= float(b.score.total) else (b, a)
                )
                ax0, ay0, ax1, ay1 = rect(a)
                bx0, by0, bx1, by1 = rect(b)
                ux0, uy0 = min(ax0, bx0), min(ay0, by0)
                ux1, uy1 = max(ax1, bx1), max(ay1, by1)
                keeper.pose_base.pose.position.x = 0.5 * (ux0 + ux1)
                keeper.pose_base.pose.position.y = 0.5 * (uy0 + uy1)
                keeper.dimensions_m.x = ux1 - ux0
                keeper.dimensions_m.y = uy1 - uy0
                keeper.dimensions_m.z = max(
                    float(a.dimensions_m.z), float(b.dimensions_m.z)
                )
                bottom = min(
                    float(a.pose_base.pose.position.z) - 0.5 * float(a.dimensions_m.z),
                    float(b.pose_base.pose.position.z) - 0.5 * float(b.dimensions_m.z),
                )
                keeper.pose_base.pose.position.z = bottom + 0.5 * float(
                    keeper.dimensions_m.z
                )
                if "overlap_fused" not in str(keeper.provenance):
                    keeper.provenance = f"{keeper.provenance};overlap_fused"
                hypotheses.remove(other)
                changed = True
                break
            if changed:
                break
    return copy_hypothesis_array_with_hypotheses(message, hypotheses)


RING_MERGE_MIN_MEMBER_PX = 24
RING_MERGE_MIN_GROUP_PX = 400
RING_MERGE_MAX_GAP_M = 0.055
RING_MERGE_MIN_RADIUS_M = 0.015
RING_MERGE_MAX_RADIUS_M = 0.090
RING_MERGE_MAX_MED_RESIDUAL_M = 0.004
RING_MERGE_MAX_P90_RESIDUAL_M = 0.009
RING_MERGE_MIN_ARC_SPAN_DEG = 50.0
RING_MERGE_MIN_UNION_SPAN_DEG = 160.0
RING_MERGE_MAX_ARC_OVERLAP_DEG = 30.0
RING_MERGE_MAX_INTERIOR_ELEVATED = 0.05
RING_MERGE_MIN_INTERIOR_SEE_THROUGH = 0.80


def _taubin_circle_fit(points_xy: np.ndarray) -> tuple[float, float, float] | None:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 6:
        return None
    mean_x = float(np.mean(points[:, 0]))
    mean_y = float(np.mean(points[:, 1]))
    u = points[:, 0] - mean_x
    v = points[:, 1] - mean_y
    suu = float(np.sum(u * u))
    svv = float(np.sum(v * v))
    suv = float(np.sum(u * v))
    suuu = float(np.sum(u**3))
    svvv = float(np.sum(v**3))
    suvv = float(np.sum(u * v * v))
    svuu = float(np.sum(v * u * u))
    matrix = np.array([[suu, suv], [suv, svv]], dtype=np.float64)
    rhs = 0.5 * np.array([suuu + suvv, svvv + svuu], dtype=np.float64)
    try:
        local_center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    center_x = float(local_center[0] + mean_x)
    center_y = float(local_center[1] + mean_y)
    radius = float(
        np.sqrt(np.mean((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2))
    )
    if not (math.isfinite(center_x) and math.isfinite(center_y) and math.isfinite(radius)):
        return None
    return center_x, center_y, radius


def _angle_bins_deg(points_xy: np.ndarray, center_xy: tuple[float, float]) -> set[int]:
    angles = np.degrees(
        np.arctan2(
            np.asarray(points_xy)[:, 1] - center_xy[1],
            np.asarray(points_xy)[:, 0] - center_xy[0],
        )
    )
    return {int((angle + 360.0) % 360.0) // 10 for angle in angles}


def merge_transparent_ring_components(
    components: list[PixelComponent],
    *,
    points_base: np.ndarray,
    z_map: np.ndarray,
    hole_mask: np.ndarray,
    table_z_m: float,
) -> tuple[list[PixelComponent], dict[int, tuple[float, float, float, float]]]:
    """Assemble fragments of ONE round transparent object into one component.

    A glass bowl fragments differently by pose: upright it returns two rim
    arcs; upside-down it returns a dome patch, a rim crescent and small edge
    fragments. What all poses share: the CONVEX HULL of the fragment union is
    a circle. Two genuinely distinct nearby objects produce a capsule- or
    rectangle-shaped hull whose circle residuals are large, so they are never
    glued together (a square hull already misses by ~15 percent of R).
    """
    z_values = np.asarray(z_map, dtype=float)
    finite_z = np.isfinite(z_values)
    foreground_z_min = float(table_z_m) + 0.006

    members: list[tuple[int, np.ndarray, np.ndarray]] = []
    for index, component in enumerate(components):
        component_mask = np.asarray(component.mask, dtype=bool)
        foreground = component_mask & finite_z & (z_values > foreground_z_min)
        if int(np.count_nonzero(foreground)) < RING_MERGE_MIN_MEMBER_PX:
            continue
        pts = points_base[foreground]
        members.append((index, pts[:, 0:2], pts[:, 2]))

    ring_fits: dict[int, tuple[float, float, float, float]] = {}
    if len(members) < 2:
        return components, ring_fits

    # neighbourhood chaining: fragments within RING_MERGE_MAX_GAP_M of each
    # other belong to one candidate group
    parent = list(range(len(members)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bounds = []
    for _, xy, _z in members:
        bounds.append(
            (
                float(np.min(xy[:, 0])),
                float(np.min(xy[:, 1])),
                float(np.max(xy[:, 0])),
                float(np.max(xy[:, 1])),
            )
        )
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            ax0, ay0, ax1, ay1 = bounds[a]
            bx0, by0, bx1, by1 = bounds[b]
            gap_x = max(0.0, max(ax0, bx0) - min(ax1, bx1))
            gap_y = max(0.0, max(ay0, by0) - min(ay1, by1))
            if math.hypot(gap_x, gap_y) <= RING_MERGE_MAX_GAP_M:
                parent[find(a)] = find(b)

    groups: dict[int, list[int]] = {}
    for a in range(len(members)):
        groups.setdefault(find(a), []).append(a)

    def hull_circle(candidate: list[int]) -> tuple[float, float, float, float] | None:
        xy = np.vstack([members[m][1] for m in candidate])
        zs = np.concatenate([members[m][2] for m in candidate])
        if xy.shape[0] < RING_MERGE_MIN_GROUP_PX:
            return None
        hull = _points_convex_hull(xy)
        if hull.shape[0] < 6:
            return None
        fit = _taubin_circle_fit(hull)
        if fit is None:
            return None
        center_x, center_y, radius = fit
        if not (RING_MERGE_MIN_RADIUS_M <= radius <= RING_MERGE_MAX_RADIUS_M):
            return None
        residual = np.abs(np.hypot(hull[:, 0] - center_x, hull[:, 1] - center_y) - radius)
        if (
            float(np.median(residual)) > RING_MERGE_MAX_MED_RESIDUAL_M
            or float(np.percentile(residual, 90)) > RING_MERGE_MAX_P90_RESIDUAL_M
        ):
            return None
        if len(_angle_bins_deg(hull, (center_x, center_y))) * 10 < RING_MERGE_MIN_UNION_SPAN_DEG:
            return None
        outside = np.hypot(xy[:, 0] - center_x, xy[:, 1] - center_y) > radius + 0.006
        if int(np.count_nonzero(outside)) > 0.02 * xy.shape[0]:
            return None
        height = max(float(np.percentile(zs, 95)) - float(table_z_m), 0.01)
        return center_x, center_y, radius, height

    merged_indices: set[int] = set()
    assemblies: list[tuple[list[int], tuple[float, float, float, float]]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        ring = hull_circle(group)
        if ring is None and len(group) > 2:
            # degenerate group (e.g. a neighbouring opaque object chained in):
            # fall back to the best co-circular pair
            best = None
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    pair_ring = hull_circle([group[a], group[b]])
                    if pair_ring is not None and (best is None or True):
                        best = ([group[a], group[b]], pair_ring)
                        break
                if best is not None:
                    break
            if best is not None:
                assemblies.append(best)
                merged_indices.update(best[0])
            continue
        if ring is not None:
            assemblies.append((group, ring))
            merged_indices.update(group)

    if not assemblies:
        return components, ring_fits

    consumed = {members[m][0] for group, _ in assemblies for m in group}
    output: list[PixelComponent] = []
    next_id = 1
    for index, component in enumerate(components):
        if index in consumed:
            continue
        rebuilt = pixel_component_from_mask(next_id, np.asarray(component.mask, dtype=bool))
        if rebuilt is not None:
            output.append(rebuilt)
            next_id += 1
    for group, ring in assemblies:
        union = np.zeros_like(np.asarray(components[members[group[0]][0]].mask, dtype=bool))
        for m in group:
            union |= np.asarray(components[members[m][0]].mask, dtype=bool)
        rebuilt = pixel_component_from_mask(next_id, union)
        if rebuilt is not None:
            output.append(rebuilt)
            ring_fits[next_id] = ring
            next_id += 1
    return output, ring_fits


def _points_convex_hull(points_xy: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(np.asarray(points_xy, dtype=float), 4), axis=0)
    if unique.shape[0] <= 3:
        return unique
    order = np.lexsort((unique[:, 1], unique[:, 0]))
    pts = unique[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)



def _binary_dilate_1px(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return out


def update_ghost_seed_pool(
    pool: list[dict],
    previous_tracks: list[StableHypothesisTrack],
    surviving_tracks: list[StableHypothesisTrack],
) -> list[dict]:
    """Remember freshly dead tracks as split seeds for a while.

    An object that flickers between merged and separate evidence (e.g. a flat
    plank bridged to a neighbor by a third object's stereo shadow) keeps
    killing its track before it can serve as a split seed. The ghost pool
    keeps the memory alive; ghosts only ever ENABLE seeded splits, which stay
    gated by cluster size, separation and the density valley.
    """
    surviving_ids = {int(track.track_id) for track in surviving_tracks}
    for track in previous_tracks:
        if int(track.track_id) in surviving_ids or int(track.observations) < 1:
            continue
        pool.append(
            {
                "center": (float(track.center_xy_m[0]), float(track.center_xy_m[1])),
                "half_extent": 0.5
                * max(float(track.size_xyz_m[0]), float(track.size_xyz_m[1])),
                "ttl": GHOST_SEED_TTL_PUBLISHES,
            }
        )
    fresh: list[dict] = []
    for ghost in pool:
        ghost["ttl"] = int(ghost["ttl"]) - 1
        if int(ghost["ttl"]) <= 0:
            continue
        # only a COMMENSURATE live track counts as this object re-tracked; a
        # big merged-blob track hovering nearby must not eat the ghost
        near_live = any(
            math.hypot(
                float(ghost["center"][0]) - float(track.center_xy_m[0]),
                float(ghost["center"][1]) - float(track.center_xy_m[1]),
            )
            <= GHOST_SEED_LIVE_SUPPRESSION_M
            and 0.5 * max(float(track.size_xyz_m[0]), float(track.size_xyz_m[1]))
            <= 2.0 * max(0.005, float(ghost["half_extent"]))
            for track in surviving_tracks
        )
        if not near_live:
            fresh.append(ghost)
    return fresh


def _clusters_separated_by_density_valley(
    xs: np.ndarray,
    ys: np.ndarray,
    center_a: tuple[float, float],
    center_b: tuple[float, float],
    *,
    bins: int = 9,
    max_valley_ratio: float = 0.55,
) -> bool:
    axis = np.asarray([center_b[0] - center_a[0], center_b[1] - center_a[1]], dtype=np.float64)
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return False
    axis /= length
    positions = (np.asarray(xs) - center_a[0]) * axis[0] + (np.asarray(ys) - center_a[1]) * axis[1]
    inside = (positions >= 0.0) & (positions <= length)
    if int(np.count_nonzero(inside)) < 2 * int(bins):
        return False
    counts, _ = np.histogram(positions[inside], bins=int(bins), range=(0.0, length))
    edge = max(1, int(bins) // 3)
    peak = float(min(counts[:edge].max(), counts[-edge:].max()))
    valley = float(counts[edge:-edge].min()) if int(bins) > 2 * edge else float(counts.min())
    if peak <= 0.0:
        return False
    return valley <= float(max_valley_ratio) * peak


def split_components_by_recent_track_seeds(
    components: list[PixelComponent],
    *,
    points_base: np.ndarray,
    z_map: np.ndarray,
    table_z_m: float,
    seed_centers_xy: list[tuple[float, float]],
    seed_half_extents_m: list[float] | None = None,
    min_cluster_px: int = SEED_SPLIT_MIN_CLUSTER_PX,
) -> tuple[list[PixelComponent], set[int]]:
    """Split hole-merged components using recently tracked object centers.

    When two close objects fuse into one depth component (the rear one inside
    the front one's stereo shadow), no per-frame geometric cue separates them
    reliably — but the tracker remembers that two objects were just there.
    Foreground pixels are assigned to the nearest recent track seed; the split
    is accepted only when both clusters are substantial, their centers stay
    clearly apart, and a density valley separates them, so a lone object
    (which never had two established tracks inside its footprint) is never
    split.

    A fast hand-carried move can leave the moved object's track (and thus its
    seed) far behind. When only ONE seed reaches the merged component, the
    seed's own track size carves out its halo cluster and the remainder stands
    on its own — still gated by the density valley, so a lone object whose far
    side reappears (dense, no valley) is never carved.
    """
    if len(seed_centers_xy) < 2:
        return list(components), set()
    z_values = np.asarray(z_map, dtype=float)
    finite_z = np.isfinite(z_values)
    foreground_z_min = float(table_z_m) + 0.006

    output: list[PixelComponent] = []
    seeded_ids: set[int] = set()
    next_id = 1

    def emit(mask: np.ndarray, seeded: bool) -> None:
        nonlocal next_id
        component = pixel_component_from_mask(next_id, mask)
        if component is None:
            return
        output.append(component)
        if seeded:
            seeded_ids.add(int(component.component_id))
        next_id += 1

    def required_cluster_px(seed_index: int | None, xs: np.ndarray, ys: np.ndarray) -> int:
        # a remembered small object only needs a cluster commensurate with the
        # footprint its track promises, not the generic minimum
        if seed_index is None or seed_half_extents_m is None:
            return int(min_cluster_px)
        half_extent = float(seed_half_extents_m[seed_index])
        if half_extent <= 0.0:
            return int(min_cluster_px)
        span_x = max(1e-3, float(np.percentile(xs, 98) - np.percentile(xs, 2)))
        span_y = max(1e-3, float(np.percentile(ys, 98) - np.percentile(ys, 2)))
        px_per_m2 = float(xs.size) / (span_x * span_y)
        expected_px = px_per_m2 * (2.0 * half_extent) * (2.0 * half_extent)
        # a seeded side is vouched for by track memory and still has to pass
        # the separation and valley gates, so its pixel bar caps lower
        return int(max(24, min(64, int(min_cluster_px), round(0.35 * expected_px))))

    def masks_with_leftovers(
        component_mask: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        assign_a: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # give the component's non-foreground pixels (low smear, silhouette —
        # the connective tissue) to whichever cluster reaches them first, so
        # each emitted mask stays as connected as the original was
        mask_a = np.zeros(component_mask.shape, dtype=bool)
        mask_a[rows[assign_a], cols[assign_a]] = True
        mask_b = np.zeros(component_mask.shape, dtype=bool)
        mask_b[rows[~assign_a], cols[~assign_a]] = True
        leftovers = component_mask & ~mask_a & ~mask_b
        for _ in range(8):
            if not leftovers.any():
                break
            grown_a = _binary_dilate_1px(mask_a) & leftovers
            mask_a |= grown_a
            leftovers &= ~grown_a
            grown_b = _binary_dilate_1px(mask_b) & leftovers
            mask_b |= grown_b
            leftovers &= ~grown_b
        return mask_a, mask_b

    def try_split(
        component_mask: np.ndarray, allow_one_seed_carve: bool
    ) -> tuple[np.ndarray, np.ndarray] | None:
        foreground = component_mask & finite_z & (z_values > foreground_z_min)
        if int(np.count_nonzero(foreground)) < 2 * int(min_cluster_px):
            return None
        rows, cols = np.nonzero(foreground)
        xs = points_base[rows, cols, 0]
        ys = points_base[rows, cols, 1]
        lo_x = float(np.percentile(xs, 2)) - SEED_SPLIT_SEED_REACH_M
        hi_x = float(np.percentile(xs, 98)) + SEED_SPLIT_SEED_REACH_M
        lo_y = float(np.percentile(ys, 2)) - SEED_SPLIT_SEED_REACH_M
        hi_y = float(np.percentile(ys, 98)) + SEED_SPLIT_SEED_REACH_M
        reachable = [
            (float(seed[0]), float(seed[1]), index)
            for index, seed in enumerate(seed_centers_xy)
            if lo_x <= float(seed[0]) <= hi_x and lo_y <= float(seed[1]) <= hi_y
        ]
        best_pair = None
        for i in range(len(reachable)):
            for j in range(i + 1, len(reachable)):
                separation = math.hypot(
                    reachable[i][0] - reachable[j][0],
                    reachable[i][1] - reachable[j][1],
                )
                if separation >= SEED_SPLIT_MIN_SEED_SEPARATION_M and (
                    best_pair is None or separation > best_pair[0]
                ):
                    best_pair = (separation, reachable[i], reachable[j])
        if (
            best_pair is None
            and allow_one_seed_carve
            and len(reachable) == 1
            and seed_half_extents_m is not None
        ):
            # single reachable seed: carve out its own halo, the remainder is
            # the (possibly just-moved) other object
            seed = reachable[0]
            half_extent = float(seed_half_extents_m[seed[2]])
            halo_radius = max(0.030, half_extent + 0.015)
            distance_seed = (xs - seed[0]) ** 2 + (ys - seed[1]) ** 2
            assign_a = distance_seed <= halo_radius * halo_radius
            if bool(assign_a.any()) and bool((~assign_a).any()):
                # refine with the remainder's core as a second anchor so the
                # seed keeps its own sparse outskirts (e.g. a flat plank's
                # dimly measured surface) instead of shedding them as shards
                anchor_b = (
                    float(np.median(xs[~assign_a])),
                    float(np.median(ys[~assign_a])),
                )
                distance_b = (xs - anchor_b[0]) ** 2 + (ys - anchor_b[1]) ** 2
                assign_a = distance_seed <= distance_b
            count_a = int(np.count_nonzero(assign_a))
            count_b = int(assign_a.size - count_a)
            accept = (
                count_a >= required_cluster_px(seed[2], xs, ys)
                and count_b >= int(min_cluster_px)
            )
            if accept:
                center_a = (float(np.median(xs[assign_a])), float(np.median(ys[assign_a])))
                center_b = (float(np.median(xs[~assign_a])), float(np.median(ys[~assign_a])))
                accept = (
                    math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
                    >= SEED_SPLIT_MIN_CLUSTER_SEPARATION_M
                ) and _clusters_separated_by_density_valley(xs, ys, center_a, center_b)
            if accept:
                return masks_with_leftovers(component_mask, rows, cols, assign_a)
        if best_pair is None:
            return None
        _, seed_a, seed_b = best_pair
        distance_a = (xs - seed_a[0]) ** 2 + (ys - seed_a[1]) ** 2
        distance_b = (xs - seed_b[0]) ** 2 + (ys - seed_b[1]) ** 2
        assign_a = distance_a <= distance_b
        count_a = int(np.count_nonzero(assign_a))
        count_b = int(assign_a.size - count_a)
        accept = (
            count_a >= required_cluster_px(seed_a[2], xs, ys)
            and count_b >= required_cluster_px(seed_b[2], xs, ys)
        )
        if accept:
            center_a = (float(np.median(xs[assign_a])), float(np.median(ys[assign_a])))
            center_b = (float(np.median(xs[~assign_a])), float(np.median(ys[~assign_a])))
            accept = (
                math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
                >= SEED_SPLIT_MIN_CLUSTER_SEPARATION_M
            )
        if accept:
            # bridge-valley test: a truly merged component thins out between the
            # two clusters (the hole-rim smear bridge), while a lone solid object
            # cut by the seed bisector stays dense across the cut — never split it.
            accept = _clusters_separated_by_density_valley(xs, ys, center_a, center_b)
        if not accept:
            return None
        return masks_with_leftovers(component_mask, rows, cols, assign_a)

    # a merged blob can hold MORE than two objects (e.g. cylinder + plank +
    # clutter under one stereo shadow): split children keep splitting until
    # no gate fires any more
    items: list[tuple[np.ndarray, bool]] = [
        (np.asarray(component.mask, dtype=bool), False) for component in components
    ]
    for _ in range(3):
        split_any = False
        next_items: list[tuple[np.ndarray, bool]] = []
        for item_mask, item_seeded in items:
            # a child of a previous split is already believed to be one
            # object: only the pair path (two remembered objects inside) may
            # split it further
            result = try_split(item_mask, allow_one_seed_carve=not item_seeded)
            if result is None:
                next_items.append((item_mask, item_seeded))
            else:
                next_items.append((result[0], True))
                next_items.append((result[1], True))
                split_any = True
        items = next_items
        if not split_any:
            break

    for item_mask, item_seeded in items:
        emit(item_mask, item_seeded)
    return output, seeded_ids


def split_components_by_foreground_islands(
    components: list[PixelComponent],
    *,
    z_map: np.ndarray,
    table_z_m: float,
    min_foreground_height_m: float = 0.006,
    min_island_area_px: int = 12,
    min_foreground_fraction: float = 0.35,
) -> tuple[list[PixelComponent], set[int], dict[int, str]]:
    result = core_group_components_by_foreground_islands(
        components,
        z_map=z_map,
        table_z_m=float(table_z_m),
        min_foreground_height_m=float(min_foreground_height_m),
        min_island_area_px=int(min_island_area_px),
        min_foreground_fraction=float(min_foreground_fraction),
    )
    return result.components, result.split_component_ids, result.reason_by_component_id


def table_points_from_mask(
    *,
    mask: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
) -> np.ndarray:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    ray_camera = np.column_stack(
        [
            (cols.astype(np.float64) - float(intrinsics.cx)) / float(intrinsics.fx),
            (rows.astype(np.float64) - float(intrinsics.cy)) / float(intrinsics.fy),
            np.ones(rows.shape[0], dtype=np.float64),
        ]
    )
    rotation = quaternion_to_rotation_matrix(camera_to_base_transform.transform.rotation)
    origin_base = transform_translation_vector(camera_to_base_transform)
    ray_base = ray_camera @ rotation.T
    denominator = ray_base[:, 2]
    valid = np.isfinite(denominator) & (np.abs(denominator) >= 1e-9)
    scale = np.empty(rows.shape[0], dtype=np.float64)
    scale.fill(np.nan)
    scale[valid] = (float(table_z_m) - float(origin_base[2])) / denominator[valid]
    valid &= np.isfinite(scale) & (scale > 0.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    points = origin_base[None, :] + ray_base[valid] * scale[valid, None]
    return points[np.all(np.isfinite(points), axis=1)]


def oriented_shadow_support_footprint_from_mask(
    *,
    mask: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    min_size_m: float = 0.005,
) -> TableFootprint | None:
    bbox = mask_bbox_pixel_edges(mask)
    if bbox is None:
        return None
    support_edges = shadow_support_pixel_edges(mask, bbox)
    rect_footprint = shadow_support_footprint_from_mask(
        mask=mask,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
        table_z_m=table_z_m,
        min_size_m=min_size_m,
    )
    if support_edges is None:
        return None

    binary = np.asarray(mask, dtype=bool)
    rows, cols = np.indices(binary.shape)
    full_left, full_right, full_top, full_bottom = bbox
    full_height = max(1.0, float(full_bottom - full_top))
    band_height = max(2, int(round(full_height * 0.35)))
    band_start = int(max(0, math.floor(float(full_bottom) + 0.5) - band_height))
    support_mask = binary & (rows >= band_start)
    if int(np.count_nonzero(support_mask)) < 4:
        support_mask = np.zeros(binary.shape, dtype=bool)
        left_u, right_u, top_v, bottom_v = support_edges
        support_mask = (
            binary
            & (cols.astype(np.float64) >= float(left_u))
            & (cols.astype(np.float64) <= float(right_u))
            & (rows.astype(np.float64) >= float(top_v))
            & (rows.astype(np.float64) <= float(bottom_v))
        )
    if int(np.count_nonzero(support_mask)) < 4:
        return shadow_support_footprint_from_mask(
            mask=mask,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=table_z_m,
            min_size_m=min_size_m,
        )

    table_points = table_points_from_mask(
        mask=support_mask,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
        table_z_m=table_z_m,
    )
    footprint = table_footprint_from_points(table_points, min_size_m=min_size_m)
    if footprint is not None and rect_footprint is not None:
        size_x = max(float(footprint.size_x_m), float(rect_footprint.size_x_m))
        size_y = max(float(footprint.size_y_m), float(rect_footprint.size_y_m))
        yaw_rad = float(footprint.yaw_rad)
        smaller = min(size_x, size_y)
        larger = max(size_x, size_y)
        if smaller > 0.0 and larger / smaller <= 1.20:
            yaw_rad = canonicalize_square_yaw(yaw_rad)
        return replace(
            footprint,
            size_x_m=float(size_x),
            size_y_m=float(size_y),
            yaw_rad=float(yaw_rad),
        )
    return footprint or shadow_support_footprint_from_mask(
        mask=mask,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
        table_z_m=table_z_m,
        min_size_m=min_size_m,
    )


def shadow_trimmed_table_points_from_mask(
    *,
    mask: np.ndarray,
    intrinsics: CameraModel,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    min_tail_ratio: float = 1.20,
    depth_to_width_ratio: float = 1.15,
    min_support_depth_m: float = 0.006,
) -> tuple[np.ndarray, str]:
    table_points = table_points_from_mask(
        mask=mask,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
        table_z_m=table_z_m,
    )
    if table_points.shape[0] < 4:
        return table_points, "mask_full"

    xy = table_points[:, 0:2]
    camera_xy = transform_translation_vector(camera_to_base_transform)[0:2]
    center_xy = np.median(xy, axis=0)
    axis = center_xy - camera_xy
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        return table_points, "mask_full"
    axis = axis / axis_norm
    cross_axis = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    s_values = xy @ axis
    t_values = xy @ cross_axis
    s_low, s_high = np.percentile(s_values, [5.0, 95.0])
    t_low, t_high = np.percentile(t_values, [5.0, 95.0])
    s_span = max(0.0, float(s_high - s_low))
    t_span = max(0.0, float(t_high - t_low))
    if s_span <= max(float(min_support_depth_m), t_span * float(min_tail_ratio)):
        return table_points, "mask_full"

    support_depth = max(float(min_support_depth_m), t_span * float(depth_to_width_ratio))
    support_depth = min(s_span, support_depth)
    keep_limit = float(s_low) + support_depth
    keep = s_values <= keep_limit
    if int(np.count_nonzero(keep)) < 4:
        return table_points, "mask_full"
    return table_points[keep], "shadow_trimmed"


def fallback_height_from_table_points(
    table_points: np.ndarray,
    *,
    fallback_height_m: float,
    min_height_m: float = 0.015,
) -> float:
    points = np.asarray(table_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 2:
        return float(fallback_height_m)
    _, width, depth, _ = robust_oriented_bounds_xy(points[:, 0:2])
    footprint_height = min(float(width), float(depth))
    if not math.isfinite(footprint_height) or footprint_height <= 0.0:
        return float(fallback_height_m)
    return clamp(footprint_height, min_height_m, float(fallback_height_m))


def table_footprint_from_points(
    table_points: np.ndarray,
    *,
    min_size_m: float = 0.005,
) -> TableFootprint | None:
    points = np.asarray(table_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 2:
        return None
    center, width, depth, yaw_rad = robust_oriented_bounds_xy(points[:, 0:2])
    return TableFootprint(
        center_xy_m=(float(center[0]), float(center[1])),
        size_x_m=max(float(min_size_m), float(width)),
        size_y_m=max(float(min_size_m), float(depth)),
        width_px=1.0,
        height_px=1.0,
        yaw_rad=float(yaw_rad),
    )


def build_live_tabletop_component_hypothesis_array(
    *,
    raw_depth_m: np.ndarray,
    corrupted_depth_m: np.ndarray,
    target_mask: np.ndarray,
    components,
    intrinsics: CameraModel,
    frame_id: str,
    base_frame_id: str,
    camera_to_base_transform: TransformStamped,
    table_z_m: float,
    primitive_height_m: float,
    min_tabletop_proxy_height_m: float,
    split_component_ids: set[int] | None = None,
    component_grouping_reasons: dict[int, str] | None = None,
    ring_fits: dict[int, tuple[float, float, float, float]] | None = None,
) -> GeometryHypothesisArray:
    message = empty_live_hypothesis_array(
        frame_id=frame_id,
        base_frame_id=base_frame_id,
        use_base_frame=True,
    )
    mask = np.asarray(target_mask, dtype=bool)
    split_component_ids = split_component_ids or set()
    component_grouping_reasons = component_grouping_reasons or {}
    ring_fits = ring_fits or {}
    # a scattered speckle field (e.g. specular returns off a transparent rim)
    # can sprawl through the full-res area gate yet decimate into tiny
    # islands: those are existence evidence, not fittable surfaces
    components = [
        component
        for component in components
        if not (
            component_grouping_reasons.get(int(component.component_id))
            in (
                "separate_sparse_foreground_islands",
                "hole_only_bridge_between_foreground_islands",
            )
            and int(component.area_px) < SPARSE_ISLAND_MIN_GEOMETRY_PX
        )
    ]
    points_base = base_points_from_depth(
        depth_m=corrupted_depth_m,
        intrinsics=intrinsics,
        camera_to_base_transform=camera_to_base_transform,
    )
    z_map = points_base[:, :, 2]
    evidence = build_tabletop_evidence(
        z_map,
        rgb_mask=mask,
        table_mask=np.isfinite(z_map) | mask,
        table_z_m=float(table_z_m),
        min_height_m=0.006,
        max_object_height_m=max(0.12, float(primitive_height_m) * 3.0),
    )

    points_xy_by_component: dict[int, np.ndarray] = {}
    points_z_by_component: dict[int, np.ndarray] = {}
    shadow_support_by_component: dict[int, TableFootprint] = {}
    full_shadow_support_by_component: dict[int, TableFootprint] = {}
    support_source_by_component: dict[int, str] = {}
    residual_points_count_by_component: dict[int, int] = {}
    foreground_z_min = float(table_z_m) + 0.006
    component_union_mask = np.zeros_like(mask, dtype=bool)
    for component in components:
        component_union_mask |= np.asarray(component.mask, dtype=bool)
    for component in components:
        component_mask = np.asarray(component.mask, dtype=bool)
        full_shadow_points = table_points_from_mask(
            mask=component_mask,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=float(table_z_m),
        )
        full_shadow_footprint = table_footprint_from_points(full_shadow_points)
        if full_shadow_footprint is not None:
            full_shadow_support_by_component[component.component_id] = full_shadow_footprint
        axis_aligned_support_footprint = shadow_support_footprint_from_mask(
            mask=component_mask,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=float(table_z_m),
        )
        oriented_support_footprint = oriented_shadow_support_footprint_from_mask(
            mask=component_mask,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=float(table_z_m),
        )
        component_fill = component_bbox_fill_ratio(component)
        component_aspect = component_bbox_aspect(component)
        round_silhouette = 0.74 <= component_fill <= 0.86 and component_aspect <= 1.25
        support_footprint = (
            axis_aligned_support_footprint
            if round_silhouette
            else (oriented_support_footprint or axis_aligned_support_footprint)
        )
        if support_footprint is not None:
            shadow_support_by_component[component.component_id] = support_footprint
        foreground = component_mask & np.isfinite(z_map) & (z_map > foreground_z_min)
        residual_points_count_by_component[component.component_id] = int(np.count_nonzero(foreground))
        min_foreground_points = min_foreground_points_for_component(component.area_px)
        if int(np.count_nonzero(foreground)) >= min_foreground_points or has_compact_foreground_residue(
            foreground
        ):
            component_points = points_base[foreground]
            points_xy_by_component[component.component_id] = component_points[:, 0:2]
            points_z_by_component[component.component_id] = component_points[:, 2]
            support_source_by_component[component.component_id] = "foreground"
            continue

        nearby_region = expanded_bbox_mask(component_mask, radius_px=8)
        other_component_mask = component_union_mask & ~component_mask
        nearby_foreground = (
            nearby_region
            & ~component_mask
            & ~other_component_mask
            & np.isfinite(z_map)
            & (z_map > foreground_z_min)
        )
        residual_points_count_by_component[component.component_id] += int(
            np.count_nonzero(nearby_foreground)
        )
        min_nearby_foreground_points = max(4, min(20, int(round(float(component.area_px) * 0.02))))
        if int(np.count_nonzero(nearby_foreground)) >= min_nearby_foreground_points:
            component_points = points_base[nearby_foreground]
            points_xy_by_component[component.component_id] = component_points[:, 0:2]
            points_z_by_component[component.component_id] = component_points[:, 2]
            support_source_by_component[component.component_id] = "nearby_foreground"
            continue

        fallback_height = max(float(min_tabletop_proxy_height_m), float(primitive_height_m))
        if support_footprint is not None:
            fallback_height = fallback_height_from_footprint(
                support_footprint,
                fallback_height_m=fallback_height,
            )

        table_points, support_source = shadow_trimmed_table_points_from_mask(
            mask=component_mask,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=float(table_z_m),
        )
        fallback_height = fallback_height_from_table_points(
            table_points,
            fallback_height_m=fallback_height,
        )
        if support_source == "shadow_trimmed":
            trimmed_footprint = table_footprint_from_points(table_points)
            if support_footprint is None and trimmed_footprint is not None:
                shadow_support_by_component[component.component_id] = trimmed_footprint
        if (
            component_hole_adjacency_fraction(component_mask, corrupted_depth_m)
            >= HOLE_ADJACENCY_MIN_FRACTION
        ):
            # Depth-hole rim silhouettes only prove that something occupies this
            # spot; their outline must not become the proxy's footprint.
            support_source = "hole_evidence"
        support_source_by_component[component.component_id] = support_source
        points_xy_by_component[component.component_id] = table_points[:, 0:2]
        points_z_by_component[component.component_id] = np.full(
            table_points.shape[0],
            float(table_z_m) + float(fallback_height),
            dtype=np.float64,
        )

    ranked_items = rank_tabletop_components(
        list(components),
        evidence,
        points_xy_by_component=points_xy_by_component,
        points_z_by_component=points_z_by_component,
        table_z_m=float(table_z_m),
        height_prior_m=max(float(min_tabletop_proxy_height_m), float(primitive_height_m)),
    )
    for rank, ranked in enumerate(ranked_items, start=1):
        fit = ranked.fit
        support_source = support_source_by_component.get(int(fit.component_id), "")
        component = next(
            item for item in components if int(item.component_id) == int(fit.component_id)
        )
        if int(fit.component_id) in ring_fits:
            ring_x, ring_y, ring_radius, ring_height = ring_fits[int(fit.component_id)]
            fit = replace(
                fit,
                shape_type="cylinder",
                center_xy_m=(float(ring_x), float(ring_y)),
                size_x_m=2.0 * float(ring_radius),
                size_y_m=2.0 * float(ring_radius),
                size_z_m=float(ring_height),
                yaw_rad=0.0,
                center_z_m=float(table_z_m) + 0.5 * float(ring_height),
                provenance=f"{fit.provenance};transparent_ring_cylinder",
            )
        elif support_source == "hole_evidence":
            fit = coerce_hole_adjacent_fit_to_existence_proxy(
                fit,
                points_xy_by_component.get(
                    int(fit.component_id), np.empty((0, 2), dtype=np.float64)
                ),
            )
        else:
            if support_source not in {"foreground", "nearby_foreground"}:
                fit = refine_shadow_expanded_fit(
                    fit,
                    shadow_support_by_component.get(int(fit.component_id)),
                )
            fit = expand_sparse_foreground_fit_with_support(
                fit,
                support_source,
                shadow_support_by_component.get(int(fit.component_id)),
            )
            fit = align_sparse_foreground_box_to_visible_edges(
                fit,
                support_source,
                shadow_support_by_component.get(int(fit.component_id)),
                points_xy_by_component.get(int(fit.component_id), np.empty((0, 2), dtype=np.float64)),
            )
            fit = coerce_projected_shadow_fit_to_unknown_bbox(
                fit,
                support_source,
                shadow_support_by_component.get(int(fit.component_id)),
                component,
                full_shadow_support_by_component.get(int(fit.component_id)),
            )
        if support_source and f"support={support_source}" not in str(fit.provenance):
            fit = replace(fit, provenance=f"{fit.provenance};support={support_source}")
        fit_points = points_xy_by_component.get(
            int(fit.component_id),
            np.empty((0, 2), dtype=np.float64),
        )
        evidence_diag = evidence_diagnostics_provenance(
            fit=fit,
            support_source=support_source,
            support_footprint=shadow_support_by_component.get(int(fit.component_id)),
            full_mask_footprint=full_shadow_support_by_component.get(int(fit.component_id)),
            fit_points_count=int(fit_points.shape[0]),
            residual_points_count=residual_points_count_by_component.get(int(fit.component_id), 0),
        )
        center_x, center_y, center_z = metric_pose_from_mask(component.mask, raw_depth_m, intrinsics)
        camera_center_z = center_z - 0.5 * float(fit.size_z_m)
        msg = GeometryHypothesis()
        msg.hypothesis_id = f"component_{fit.component_id}_{fit.shape_type}_r{rank}"
        msg.shape_type = shape_constant(fit.shape_type)
        msg.pose_camera = make_pose(
            frame_id,
            center_x,
            center_y,
            camera_center_z,
            yaw_orientation(float(fit.yaw_rad)),
        )
        msg.pose_base = make_pose(
            str(base_frame_id),
            float(fit.center_xy_m[0]),
            float(fit.center_xy_m[1]),
            float(fit.center_z_m),
            yaw_orientation(float(fit.yaw_rad)),
        )
        msg.dimensions_m.x = max(0.005, float(fit.size_x_m))
        msg.dimensions_m.y = max(0.005, float(fit.size_y_m))
        msg.dimensions_m.z = max(0.005, float(fit.size_z_m))
        msg.score = make_live_tabletop_score(ranked)
        msg.confidence = max(0.0, min(1.0, float(ranked.score)))
        msg.uncertainty = 1.0 - msg.confidence
        msg.provenance = (
            f"m4_live_no_truth_ghost_mgg_v1;live_tabletop_core:rank={rank};"
            f"component_id={fit.component_id};component_area_px={component.area_px};"
            f"fit={fit.shape_type};fit_source={fit.provenance};"
            f"coverage={ranked.score_terms.get('coverage_ratio', 0.0):.3f};"
            f"support={ranked.score_terms.get('support_ratio', 0.0):.3f};"
            f"failure={ranked.score_terms.get('failure_ratio', 0.0):.3f};"
            f"leak={ranked.score_terms.get('table_leakage_ratio', 0.0):.3f};"
            f"{evidence_diag}"
        )
        grouping_reason = component_grouping_reasons.get(int(fit.component_id))
        if grouping_reason:
            msg.provenance = f"{msg.provenance};component_grouping={grouping_reason}"
        if int(fit.component_id) in split_component_ids:
            msg.provenance = f"{msg.provenance};component_split=foreground_islands"
            if grouping_reason:
                msg.provenance = f"{msg.provenance};component_split_reason={grouping_reason}"
        msg.validation_state = GeometryHypothesis.VALIDATION_VALID

        grasp_z = float(fit.center_z_m) + 0.5 * float(fit.size_z_m) + 0.005
        pregrasp_z = grasp_z + 0.080
        grasp = GraspCandidate()
        grasp.grasp_id = f"{msg.hypothesis_id}_top"
        grasp.grasp_pose = make_pose(
            str(base_frame_id),
            float(fit.center_xy_m[0]),
            float(fit.center_xy_m[1]),
            grasp_z,
            top_down_orientation(),
        )
        grasp.pregrasp_pose = make_pose(
            str(base_frame_id),
            float(fit.center_xy_m[0]),
            float(fit.center_xy_m[1]),
            pregrasp_z,
            top_down_orientation(),
        )
        grasp.approach_vector.z = -1.0
        grasp.gripper_width_m = min(float(fit.size_x_m), float(fit.size_y_m)) + 0.012
        grasp.grasp_type = GraspCandidate.GRASP_TYPE_TOP
        grasp.score = msg.score.total
        grasp.validation_state = GraspCandidate.VALIDATION_VALID
        msg.grasp_candidates = [grasp]
        message.hypotheses.append(msg)
    return message


def _build_live_hypothesis_array_single_target(
    *,
    raw_depth_m: np.ndarray,
    corrupted_depth_m: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: CameraModel,
    frame_id: str = "d435_depth_optical_frame",
    base_frame_id: str = "world",
    camera_to_base_transform: TransformStamped | None = None,
    table_z_m: float | None = None,
    top_k: int = 3,
    primitive_height_m: float = 0.04,
    min_tabletop_proxy_size_m: float = 0.005,
    min_tabletop_proxy_height_m: float = 0.005,
) -> GeometryHypothesisArray:
    mask = np.asarray(target_mask, dtype=bool)
    message = GeometryHypothesisArray()
    use_base_frame = camera_to_base_transform is not None and table_z_m is not None
    message.header.frame_id = str(base_frame_id) if use_base_frame else frame_id
    message.trial_id = "m4_live"
    message.observation_id = "m4_live_hypotheses"
    message.backend_name = "ghost_mgg_m4_live"
    if not mask.any():
        return message

    _, _, object_depth = metric_pose_from_mask(mask, raw_depth_m, intrinsics)
    ranking_height_m = float(primitive_height_m)
    if use_base_frame:
        ranking_height_m = visible_height_from_depth(
            mask=mask,
            raw_depth_m=corrupted_depth_m,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=float(table_z_m),
            fallback_height_m=float(primitive_height_m),
        )
        ranking_height_m = max(float(min_tabletop_proxy_height_m), float(ranking_height_m))
    metric_footprint = None
    if use_base_frame:
        points_base = base_points_from_depth(
            depth_m=corrupted_depth_m,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
        )
        evidence, evidence_summary = evidence_from_raw_depth(
            corrupted_depth_m,
            mask,
            table_z_m=float(table_z_m),
            points_base=points_base,
            table_distance_sigma_m=0.008,
            foreground_min_height_m=0.006,
        )
    else:
        evidence, evidence_summary = evidence_from_raw_depth(
            corrupted_depth_m,
            mask,
            table_depth_m=float(object_depth),
            foreground_min_height_m=0.006,
        )
    if use_base_frame:
        foreground_footprint = foreground_footprint_from_depth(
            mask=mask,
            raw_depth_m=corrupted_depth_m,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=float(table_z_m),
            min_size_m=float(min_tabletop_proxy_size_m),
        )
        mask_footprint = table_footprint_from_mask(
            mask=mask,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=float(table_z_m),
            min_size_m=float(min_tabletop_proxy_size_m),
        )
        weak_foreground = (
            float(evidence_summary.foreground_support_ratio) < 0.12
            or float(evidence_summary.hole_ratio) > 0.50
        )
        if weak_foreground:
            metric_footprint = mask_footprint or foreground_footprint
        else:
            metric_footprint = foreground_footprint or mask_footprint
        if (
            metric_footprint is not None
            and weak_foreground
        ):
            ranking_height_m = fallback_height_from_footprint(
                metric_footprint,
                fallback_height_m=float(primitive_height_m),
            )
    config = GhostMGGV0Config(
        depth_m=object_depth,
        height_m=ranking_height_m,
        top_k=max(1, int(top_k)),
    )
    if use_base_frame and metric_footprint is not None:
        live_hypotheses = generate_live_table_anchored_hypotheses(
            mask=mask,
            evidence=evidence,
            metric_footprint=metric_footprint,
            table_z_m=float(table_z_m),
            config=config,
            intrinsics=intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            min_size_xy_m=float(min_tabletop_proxy_size_m),
        )
    else:
        live_hypotheses = generate_live_hypotheses_with_priors(mask, config)
    ranked_items = run_ghost_mgg_v0(
        mask,
        evidence,
        config=config,
        hypotheses=live_hypotheses,
    )
    evidence_summary_values = evidence_summary.as_dict()

    for rank, ranked in enumerate(ranked_items, start=1):
        hypothesis = ranked.hypothesis
        center_x, center_y, center_z = metric_pose_from_mask(mask, raw_depth_m, intrinsics)
        width_m, depth_m, height_m = metric_dimensions_from_hypothesis(hypothesis, intrinsics)
        if hypothesis.size_xy_m is not None:
            width_m = max(0.005, float(hypothesis.size_xy_m[0]))
            depth_m = max(0.005, float(hypothesis.size_xy_m[1]))
        if hypothesis.center_z_m is not None and hypothesis.bottom_z_m is not None:
            height_m = max(0.005, float(hypothesis.center_z_m) - float(hypothesis.bottom_z_m)) * 2.0
        camera_center_z = center_z - 0.5 * height_m
        base_center = None
        if use_base_frame:
            base_center = table_anchored_base_center(
                mask=mask,
                intrinsics=intrinsics,
                camera_to_base_transform=camera_to_base_transform,
                table_z_m=float(table_z_m),
                height_m=height_m,
            )
            if hypothesis.center_xy_m is not None and hypothesis.center_z_m is not None:
                base_center = np.array(
                    [
                        float(hypothesis.center_xy_m[0]),
                        float(hypothesis.center_xy_m[1]),
                        float(hypothesis.center_z_m),
                    ],
                    dtype=np.float64,
                )
            elif metric_footprint is not None:
                base_center = np.array(
                    [
                        float(metric_footprint.center_xy_m[0]),
                        float(metric_footprint.center_xy_m[1]),
                        float(table_z_m) + 0.5 * float(height_m),
                    ],
                    dtype=np.float64,
                )
        if base_center is None:
            base_center = np.array([center_x, center_y, camera_center_z], dtype=np.float64)
            resolved_base_frame = frame_id
        else:
            resolved_base_frame = str(base_frame_id)

        msg = GeometryHypothesis()
        msg.hypothesis_id = str(hypothesis.hypothesis_id)
        msg.shape_type = shape_constant(hypothesis.shape_type)
        camera_proxy_orientation = yaw_orientation(float(getattr(hypothesis, "yaw_rad", 0.0)))
        base_yaw_rad = float(getattr(hypothesis, "yaw_rad", 0.0))
        base_proxy_orientation = yaw_orientation(base_yaw_rad)
        msg.pose_camera = make_pose(frame_id, center_x, center_y, camera_center_z, camera_proxy_orientation)
        msg.pose_base = make_pose(
            resolved_base_frame,
            float(base_center[0]),
            float(base_center[1]),
            float(base_center[2]),
            base_proxy_orientation,
        )
        msg.dimensions_m.x = width_m
        msg.dimensions_m.y = depth_m
        msg.dimensions_m.z = height_m
        msg.score = make_score(ranked)
        msg.confidence = max(0.0, min(1.0, float(ranked.score.total)))
        msg.uncertainty = 1.0 - msg.confidence
        msg.provenance = (
            f"m4_live_no_truth_ghost_mgg_v1:rank={rank};"
            f"hole={evidence_summary_values['hole_ratio']:.3f};"
            f"leak={evidence_summary_values['table_leakage_ratio']:.3f};"
            f"fg={evidence_summary_values['foreground_support_ratio']:.3f}"
        )
        msg.validation_state = GeometryHypothesis.VALIDATION_VALID

        grasp_x = float(base_center[0])
        grasp_y = float(base_center[1])
        grasp_z = float(base_center[2]) + 0.5 * height_m + 0.005
        pregrasp_z = grasp_z + 0.080
        grasp = GraspCandidate()
        grasp.grasp_id = f"{msg.hypothesis_id}_top"
        grasp.grasp_pose = make_pose(
            resolved_base_frame,
            grasp_x,
            grasp_y,
            grasp_z,
            top_down_orientation(),
        )
        grasp.pregrasp_pose = make_pose(
            resolved_base_frame,
            grasp_x,
            grasp_y,
            pregrasp_z,
            top_down_orientation(),
        )
        grasp.approach_vector.z = -1.0
        grasp.gripper_width_m = min(width_m, depth_m) + 0.012
        grasp.grasp_type = GraspCandidate.GRASP_TYPE_TOP
        grasp.score = msg.score.total
        grasp.validation_state = GraspCandidate.VALIDATION_VALID
        msg.grasp_candidates = [grasp]
        message.hypotheses.append(msg)

    return message


class M4LiveHypothesisPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("m4_live_hypothesis_publisher_node")
        self.raw_depth_topic = self.declare_parameter("raw_depth_topic", DEFAULT_RAW_DEPTH_TOPIC).value
        self.corrupted_depth_topic = self.declare_parameter(
            "corrupted_depth_topic", DEFAULT_CORRUPTED_DEPTH_TOPIC
        ).value
        self.color_topic = self.declare_parameter("color_topic", DEFAULT_COLOR_TOPIC).value
        self.camera_info_topic = self.declare_parameter(
            "camera_info_topic", DEFAULT_CAMERA_INFO_TOPIC
        ).value
        self.mask_topic = self.declare_parameter("mask_topic", DEFAULT_MASK_TOPIC).value
        self.external_mask_topic = self.declare_parameter(
            "external_mask_topic", DEFAULT_EXTERNAL_MASK_TOPIC
        ).value
        self.hypothesis_topic = self.declare_parameter(
            "hypothesis_topic", DEFAULT_HYPOTHESIS_TOPIC
        ).value
        self.frame_id = self.declare_parameter("frame_id", "d435_depth_optical_frame").value
        self.base_frame_id = self.declare_parameter("base_frame_id", "world").value
        self.table_z_m = float(self.declare_parameter("table_z_m", 0.7400).value)
        self.depth_margin_m = float(self.declare_parameter("depth_margin_m", 0.035).value)
        self.min_component_area_px = int(self.declare_parameter("min_component_area_px", 40).value)
        self.primitive_height_m = float(self.declare_parameter("primitive_height_m", 0.04).value)
        self.publish_rate_hz = max(
            0.1,
            float(self.declare_parameter("publish_rate_hz", 5.0).value),
        )
        self.processing_stride = max(1, int(self.declare_parameter("processing_stride", 2).value))
        self.top_k = int(self.declare_parameter("top_k", 3).value)
        self.min_hypothesis_score = float(
            self.declare_parameter("min_hypothesis_score", MIN_HYPOTHESIS_SCORE).value
        )
        self.stable_output_hold_misses = int(
            self.declare_parameter("stable_output_hold_misses", 2).value
        )
        self.recent_ghost_seeds: list[dict] = []
        self.enable_rgb_transparency_evidence = bool(
            self.declare_parameter("enable_rgb_transparency_evidence", True).value
        )
        self.transparent_boxes_topic = str(
            self.declare_parameter(
                "transparent_boxes_topic", DEFAULT_TRANSPARENT_BOXES_TOPIC
            ).value
        )
        self.last_transparent_update_s: float | None = None
        self.latest_transparent_boxes: np.ndarray | None = None
        self.latest_transparent_boxes_monotonic = 0.0
        self.target_color_hint = str(self.declare_parameter("target_color_hint", "").value)
        self.require_external_mask = bool(self.declare_parameter("require_external_mask", False).value)
        self.enable_target_lock = bool(self.declare_parameter("enable_target_lock", True).value)
        self.max_locked_center_distance_px = float(
            self.declare_parameter("max_locked_center_distance_px", 60.0).value
        )
        self.enable_table_foreground_gate = bool(
            self.declare_parameter("enable_table_foreground_gate", False).value
        )
        self.foreground_min_height_m = float(
            self.declare_parameter("foreground_min_height_m", 0.006).value
        )
        self.foreground_max_height_m = float(
            self.declare_parameter("foreground_max_height_m", 0.25).value
        )
        self.workspace_min_x_m = float(self.declare_parameter("workspace_min_x_m", -10.0).value)
        self.workspace_max_x_m = float(self.declare_parameter("workspace_max_x_m", 10.0).value)
        self.workspace_min_y_m = float(self.declare_parameter("workspace_min_y_m", -10.0).value)
        self.workspace_max_y_m = float(self.declare_parameter("workspace_max_y_m", 10.0).value)
        self.transparent_field = TransparentEvidenceField(
            (
                self.workspace_min_x_m,
                self.workspace_max_x_m,
                self.workspace_min_y_m,
                self.workspace_max_y_m,
            )
        )
        self.stable_foreground_min_observations = max(
            1, int(self.declare_parameter("stable_foreground_min_observations", 1).value)
        )
        self.stable_foreground_max_center_jump_m = float(
            self.declare_parameter("stable_foreground_max_center_jump_m", 0.05).value
        )
        self.stable_foreground_max_misses = max(
            0, int(self.declare_parameter("stable_foreground_max_misses", 1).value)
        )
        self.stable_shape_switch_observations = max(
            1, int(self.declare_parameter("stable_shape_switch_observations", 3).value)
        )
        self.stable_smoothing_alpha = float(
            self.declare_parameter("stable_smoothing_alpha", 0.45).value
        )
        self.stable_dimension_smoothing_alpha = float(
            self.declare_parameter("stable_dimension_smoothing_alpha", 0.45).value
        )
        self.stable_dimension_max_step_ratio = float(
            self.declare_parameter("stable_dimension_max_step_ratio", 0.0).value
        )

        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        contract_qos = QoSProfile(depth=1)
        contract_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        contract_qos.reliability = ReliabilityPolicy.RELIABLE
        mask_qos = QoSProfile(depth=1)
        mask_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        mask_qos.reliability = ReliabilityPolicy.RELIABLE

        self.mask_pub = self.create_publisher(Image, self.mask_topic, qos)
        self.hypothesis_pub = self.create_publisher(
            GeometryHypothesisArray, self.hypothesis_topic, contract_qos
        )
        self.raw_depth_sub = self.create_subscription(
            Image, self.raw_depth_topic, self.handle_raw_depth, qos
        )
        self.corrupted_depth_sub = self.create_subscription(
            Image, self.corrupted_depth_topic, self.handle_corrupted_depth, qos
        )
        self.color_sub = self.create_subscription(Image, self.color_topic, self.handle_color, qos)
        self.transparent_boxes_sub = None
        if str(self.transparent_boxes_topic):
            self.transparent_boxes_sub = self.create_subscription(
                Float32MultiArray,
                str(self.transparent_boxes_topic),
                self.handle_transparent_boxes,
                qos,
            )
        self.external_mask_sub = None
        if str(self.external_mask_topic):
            self.external_mask_sub = self.create_subscription(
                Image,
                str(self.external_mask_topic),
                self.handle_external_mask,
                mask_qos,
            )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.handle_camera_info, qos
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.latest_raw_depth: np.ndarray | None = None
        self.latest_raw_header = None
        self.latest_target_mask: np.ndarray | None = None
        self.latest_corrupted_depth: np.ndarray | None = None
        self.latest_corrupted_generation = 0
        self.last_published_corrupted_generation = -1
        self.latest_color_rgb: np.ndarray | None = None
        self.latest_external_mask: np.ndarray | None = None
        self.latest_camera_model: CameraModel | None = None
        self.locked_target_center_uv: tuple[float, float] | None = None
        self.stable_component_tracks: list[StableComponentTrack] = []
        self.stable_hypothesis_tracks: list[StableHypothesisTrack] = []
        self.publish_timer = self.create_timer(1.0 / self.publish_rate_hz, self.try_publish)

    def handle_camera_info(self, msg: CameraInfo) -> None:
        self.latest_camera_model = camera_model_from_info(msg)

    def handle_raw_depth(self, msg: Image) -> None:
        try:
            self.latest_raw_depth = decode_depth_image(msg)
            self.latest_raw_header = msg.header
        except ValueError as error:
            self.get_logger().warn(str(error))
            return
        self.publish_target_mask_from_raw_depth()

    def handle_color(self, msg: Image) -> None:
        try:
            self.latest_color_rgb = decode_color_image(msg)
        except ValueError as error:
            self.get_logger().warn(str(error))
            return
        self.publish_target_mask_from_raw_depth()

    def handle_transparent_boxes(self, msg: Float32MultiArray) -> None:
        values = np.asarray(msg.data, dtype=np.float64)
        if values.size % 5 == 0 and values.size > 0:
            # empty detections do NOT clear the cache: a borderline pose
            # flickers around the confidence threshold, and the freshness
            # window is what expires stale boxes
            self.latest_transparent_boxes = values.reshape(-1, 5)
            self.latest_transparent_boxes_monotonic = pytime.monotonic()

    def handle_external_mask(self, msg: Image) -> None:
        try:
            self.latest_external_mask = decode_mask_image(msg)
        except ValueError as error:
            self.get_logger().warn(str(error))
            return
        self.publish_target_mask_from_raw_depth()

    def handle_corrupted_depth(self, msg: Image) -> None:
        try:
            self.latest_corrupted_depth = decode_depth_image(msg)
        except ValueError as error:
            self.get_logger().warn(str(error))
            return
        self.latest_corrupted_generation += 1
        self.try_publish()

    def publish_target_mask_from_raw_depth(self) -> None:
        if self.latest_raw_depth is None or self.latest_raw_header is None:
            return
        color_hint = str(self.target_color_hint).strip().lower()
        if self.latest_external_mask is not None and self.latest_external_mask.any():
            self.latest_target_mask = resize_mask_nearest(
                self.latest_external_mask,
                self.latest_raw_depth.shape,
            )
        elif self.require_external_mask:
            self.latest_target_mask = np.zeros(self.latest_raw_depth.shape, dtype=bool)
        elif color_hint and color_hint not in ("none", "auto", "depth"):
            if self.latest_color_rgb is None:
                return
            color_mask = target_color_mask_from_image(
                self.latest_color_rgb,
                color_hint=color_hint,
                min_area_px=self.min_component_area_px,
                locked_center_uv=None,
                max_locked_center_distance_px=self.max_locked_center_distance_px,
            )
            self.latest_target_mask = resize_mask_nearest(color_mask, self.latest_raw_depth.shape)
        else:
            if self.enable_target_lock:
                self.latest_target_mask = foreground_mask_from_depth(
                    self.latest_raw_depth,
                    depth_margin_m=self.depth_margin_m,
                    min_component_area_px=self.min_component_area_px,
                    locked_center_uv=self.locked_target_center_uv,
                    max_locked_center_distance_px=self.max_locked_center_distance_px,
                )
            else:
                self.latest_target_mask = foreground_components_mask_from_depth(
                    self.latest_raw_depth,
                    depth_margin_m=self.depth_margin_m,
                    min_component_area_px=self.min_component_area_px,
                )
        if self.enable_target_lock and self.latest_target_mask.any():
            self.locked_target_center_uv = mask_center_pixel(self.latest_target_mask)
        self.mask_pub.publish(encode_mask_image(self.latest_target_mask, self.latest_raw_header))

    def try_publish(self) -> None:
        if (
            self.latest_raw_depth is None
            or self.latest_corrupted_depth is None
            or self.latest_camera_model is None
            or self.latest_raw_header is None
            or self.latest_target_mask is None
        ):
            return
        if self.latest_corrupted_generation == self.last_published_corrupted_generation:
            return
        camera_to_base_transform = self.lookup_camera_to_base_transform()
        if camera_to_base_transform is None:
            return
        raw_depth = self.latest_raw_depth
        corrupted_depth = self.latest_corrupted_depth
        target_mask = self.latest_target_mask
        camera_model = self.latest_camera_model
        if not getattr(self, "_table_self_check_done", False):
            self._table_self_check_done = True
            try:
                warning = table_plane_self_check(
                    raw_depth,
                    camera_model,
                    camera_to_base_transform,
                    self.table_z_m,
                    (
                        self.workspace_min_x_m,
                        self.workspace_max_x_m,
                        self.workspace_min_y_m,
                        self.workspace_max_y_m,
                    ),
                )
            except Exception as error:  # noqa: BLE001 - diagnostic only
                warning = f"table self-check failed: {error}"
            if warning is None:
                self.get_logger().info("table plane self-check: OK")
            else:
                self.get_logger().warn(warning)
        if self.enable_table_foreground_gate:
            target_mask = table_foreground_supported_mask(
                mask=target_mask,
                depth_m=raw_depth,
                intrinsics=camera_model,
                camera_to_base_transform=camera_to_base_transform,
                table_z_m=self.table_z_m,
                min_height_m=self.foreground_min_height_m,
                max_height_m=self.foreground_max_height_m,
                min_component_area_px=self.min_component_area_px,
                workspace_min_x_m=self.workspace_min_x_m,
                workspace_max_x_m=self.workspace_max_x_m,
                workspace_min_y_m=self.workspace_min_y_m,
                workspace_max_y_m=self.workspace_max_y_m,
            )
        if self.stable_foreground_min_observations > 1:
            target_mask, self.stable_component_tracks = filter_stable_component_mask(
                target_mask,
                self.stable_component_tracks,
                intrinsics=camera_model,
                camera_to_base_transform=camera_to_base_transform,
                table_z_m=self.table_z_m,
                min_component_area_px=self.min_component_area_px,
                min_observations=self.stable_foreground_min_observations,
                max_center_jump_m=self.stable_foreground_max_center_jump_m,
                max_misses=self.stable_foreground_max_misses,
            )
        if self.processing_stride > 1:
            stride = int(self.processing_stride)
            raw_depth = raw_depth[::stride, ::stride]
            corrupted_depth = corrupted_depth[::stride, ::stride]
            target_mask = decimate_mask_any(target_mask, stride)
            camera_model = decimated_camera_model(camera_model, stride)
        established_tracks = [
            track
            for track in self.stable_hypothesis_tracks
            if int(track.observations) >= 2
        ]
        split_seed_centers_xy = [
            (float(track.center_xy_m[0]), float(track.center_xy_m[1]))
            for track in established_tracks
        ]
        split_seed_half_extents_m = [
            0.5 * max(float(track.size_xyz_m[0]), float(track.size_xyz_m[1]))
            for track in established_tracks
        ]
        for ghost in self.recent_ghost_seeds:
            split_seed_centers_xy.append(ghost["center"])
            split_seed_half_extents_m.append(float(ghost["half_extent"]))
        rgb_saliency = None
        rgb_instance = False
        if self.enable_rgb_transparency_evidence:
            boxes = self.latest_transparent_boxes
            boxes_fresh = (
                boxes is not None
                and boxes.shape[0] > 0
                and pytime.monotonic() - self.latest_transparent_boxes_monotonic
                <= DETECTOR_BOXES_FRESH_S
            )
            if self.latest_color_rgb is not None:
                dark_scene = (
                    float(self.latest_color_rgb[::8, ::8].mean())
                    < DARK_LUMINANCE_MEAN
                )
                color_strided = sample_rgb_to_shape(
                    self.latest_color_rgb, corrupted_depth.shape
                )
                stateless_saliency = (
                    None if dark_scene
                    else rgb_transparency_saliency_mask(color_strided)
                )
                if boxes_fresh:
                    # detector says WHERE the instance is; saliency inside
                    # the box says exactly WHAT it covers (loose-box immune;
                    # in darkness there is no usable saliency, so the box
                    # is used unshrunk)
                    rgb_saliency = transparent_boxes_weight_mask(
                        boxes,
                        self.latest_color_rgb.shape[:2],
                        corrupted_depth.shape,
                        saliency_mask=stateless_saliency,
                    )
                    rgb_instance = True
                else:
                    rgb_saliency = stateless_saliency
        message = build_live_hypothesis_array(
            raw_depth_m=raw_depth,
            corrupted_depth_m=corrupted_depth,
            target_mask=target_mask,
            intrinsics=camera_model,
            frame_id=self.frame_id,
            base_frame_id=self.base_frame_id,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=self.table_z_m,
            top_k=self.top_k,
            primitive_height_m=self.primitive_height_m,
            split_seed_centers_xy=split_seed_centers_xy,
            split_seed_half_extents_m=split_seed_half_extents_m,
        )
        occluders = [
            (
                float(h.pose_base.pose.position.x),
                float(h.pose_base.pose.position.y),
                0.5 * max(float(h.dimensions_m.x), float(h.dimensions_m.y)),
            )
            for h in message.hypotheses
            if "hole_existence_only" not in str(h.provenance)
        ]
        instance_seeds: list[tuple[float, float]] = []
        if rgb_instance and self.latest_transparent_boxes is not None:
            boxes = self.latest_transparent_boxes
            seed_mask = np.zeros(corrupted_depth.shape, dtype=bool)
            scale_r = corrupted_depth.shape[0] / float(
                max(1, self.latest_color_rgb.shape[0])
            )
            scale_c = corrupted_depth.shape[1] / float(
                max(1, self.latest_color_rgb.shape[1])
            )
            for x0, y0, x1, y1 in boxes[:, :4]:
                v = (y1 - 0.15 * max(1.0, y1 - y0)) * scale_r
                u = 0.5 * (x0 + x1) * scale_c
                r = int(max(0, min(corrupted_depth.shape[0] - 1, round(v))))
                c = int(max(0, min(corrupted_depth.shape[1] - 1, round(u))))
                seed_mask[r, c] = True
            seed_points = table_points_from_mask(
                mask=seed_mask,
                intrinsics=camera_model,
                camera_to_base_transform=camera_to_base_transform,
                table_z_m=self.table_z_m,
            )
            instance_seeds = [
                (float(p[0]), float(p[1])) for p in seed_points
            ]
        confirmed_occluders = dense_occluders_xy_half(
            message, exclude_seeds_xy=instance_seeds
        )
        anchor_votes, hole_votes, rgb_votes, bare_xy = transparent_evidence_votes(
            depth_m=corrupted_depth,
            intrinsics=camera_model,
            camera_to_base_transform=camera_to_base_transform,
            table_z_m=self.table_z_m,
            occluders_xy_half=occluders,
            confirmed_occluders_xy_half=confirmed_occluders,
            workspace=(
                self.workspace_min_x_m,
                self.workspace_max_x_m,
                self.workspace_min_y_m,
                self.workspace_max_y_m,
            ),
            rgb_saliency_mask=rgb_saliency,
            rgb_instance_evidence=rgb_instance,
            instance_seeds_xy=instance_seeds,
        )
        now_s = float(self.get_clock().now().nanoseconds) / 1e9
        if self.last_transparent_update_s is None:
            update_dt_s = 0.08
        else:
            update_dt_s = now_s - self.last_transparent_update_s
        self.last_transparent_update_s = now_s
        self.transparent_field.update(
            anchor_votes,
            hole_votes,
            rgb_votes,
            update_dt_s,
            bare_xy=bare_xy,
            occluders_xy_half=confirmed_occluders,
        )
        transparent_footprints = self.transparent_field.extract(
            (
                float(camera_to_base_transform.transform.translation.x),
                float(camera_to_base_transform.transform.translation.y),
            ),
            instance_seeds_xy=instance_seeds,
        )
        message = drop_refraction_artifacts_behind_transparent(
            message,
            transparent_footprints,
            (
                float(camera_to_base_transform.transform.translation.x),
                float(camera_to_base_transform.transform.translation.y),
            ),
        )
        message = consolidate_instance_partial_returns(
            message, transparent_footprints, instance_seeds
        )
        message = fuse_overlapping_hypotheses(message)
        # suppression must see the POST-consolidation reality: fragments the
        # instance consolidation just removed cannot both suppress the proxy
        # and vanish themselves (that leaves the object with no output)
        surviving_occluders = [
            (
                float(h.pose_base.pose.position.x),
                float(h.pose_base.pose.position.y),
                0.5 * max(float(h.dimensions_m.x), float(h.dimensions_m.y)),
            )
            for h in message.hypotheses
            if "hole_existence_only" not in str(h.provenance)
        ]
        surviving_rects = [
            (
                float(h.pose_base.pose.position.x),
                float(h.pose_base.pose.position.y),
                0.5 * float(h.dimensions_m.x),
                0.5 * float(h.dimensions_m.y),
            )
            for h in message.hypotheses
            if "hole_existence_only" not in str(h.provenance)
        ]
        for index, footprint in enumerate(transparent_footprints):
            if _footprint_overlaps_occluder(footprint, surviving_rects):
                # a surviving real already covers this object's extent
                continue
            if _near_any_occluder(
                float(footprint.center_xy_m[0]),
                float(footprint.center_xy_m[1]),
                surviving_occluders,
                TRANSPARENT_PROXY_SUPPRESS_MARGIN_M,
            ):
                # a real hypothesis owns this spot right now; the field
                # keeps its mass but the existence proxy yields the frame
                continue
            message.hypotheses.append(
                transparent_hole_hypothesis(
                    footprint,
                    base_frame_id=self.base_frame_id,
                    table_z_m=self.table_z_m,
                    index=index,
                )
            )
        message = drop_low_evidence_hypotheses(
            message,
            min_score=self.min_hypothesis_score,
        )
        previous_tracks = list(self.stable_hypothesis_tracks)
        message, self.stable_hypothesis_tracks = filter_stable_hypothesis_array(
            message,
            self.stable_hypothesis_tracks,
            min_observations=1,
            max_center_jump_m=self.stable_foreground_max_center_jump_m,
            max_misses=max(
                self.stable_foreground_max_misses,
                self.stable_output_hold_misses,
            ),
            shape_switch_observations=self.stable_shape_switch_observations,
            smoothing_alpha=self.stable_smoothing_alpha,
            dimension_smoothing_alpha=self.stable_dimension_smoothing_alpha,
            dimension_max_step_ratio=self.stable_dimension_max_step_ratio,
            output_hold_misses=self.stable_output_hold_misses,
        )
        message, self.stable_hypothesis_tracks = retire_contained_duplicate_tracks(
            message, self.stable_hypothesis_tracks
        )
        self.recent_ghost_seeds = update_ghost_seed_pool(
            self.recent_ghost_seeds,
            previous_tracks,
            self.stable_hypothesis_tracks,
        )
        message.header.stamp = self.get_clock().now().to_msg()
        # final workspace guard: NOTHING outside the configured workspace is
        # published, whichever pipeline produced it (the transparent field is
        # bounded upstream, but real-pipeline fits from structures at the
        # workspace fringe, e.g. the arm's acrylic sheet edge, were not)
        in_workspace = [
            h for h in message.hypotheses
            if self.workspace_min_x_m <= float(h.pose_base.pose.position.x) <= self.workspace_max_x_m
            and self.workspace_min_y_m <= float(h.pose_base.pose.position.y) <= self.workspace_max_y_m
        ]
        if len(in_workspace) != len(message.hypotheses):
            message = copy_hypothesis_array_with_hypotheses(message, in_workspace)
        self.hypothesis_pub.publish(message)
        self.last_published_corrupted_generation = self.latest_corrupted_generation

    def lookup_camera_to_base_transform(self) -> TransformStamped | None:
        if not self.base_frame_id:
            return None
        try:
            return self.tf_buffer.lookup_transform(
                str(self.base_frame_id),
                str(self.frame_id),
                rclpy.time.Time(),
            )
        except Exception as error:
            self.get_logger().warn(
                f"Waiting for TF {self.base_frame_id} <- {self.frame_id}: {error}",
                throttle_duration_sec=5.0,
            )
            return None


def main() -> None:
    rclpy.init()
    node = M4LiveHypothesisPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
