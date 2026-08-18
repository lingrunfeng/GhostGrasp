#!/usr/bin/env python3
"""Generate a MoveIt dry-run target from one M6 shadow decision.

This script converts a masked aligned-depth observation into a top-grasp target
in the MoveIt world/base frame. It does not command robot motion.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return base_dir / path


def _read_mask(mask_path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {mask_path}")
    if mask.shape != expected_shape:
        raise ValueError(f"mask/depth shape mismatch: {mask.shape} vs {expected_shape}")
    return mask > 0


def project_mask_depth_to_camera_points(
    *,
    mask: np.ndarray,
    depth_mm: np.ndarray,
    camera_info: dict[str, Any],
) -> np.ndarray:
    fx = float(camera_info["k"][0])
    fy = float(camera_info["k"][4])
    cx = float(camera_info["k"][2])
    cy = float(camera_info["k"][5])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("camera_info fx/fy must be nonzero")
    valid = mask.astype(bool) & (depth_mm > 0)
    if not valid.any():
        raise ValueError("target mask contains no valid depth")
    rows, cols = np.nonzero(valid)
    z = depth_mm[rows, cols].astype(np.float64) * 0.001
    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    return np.column_stack([x, y, z])


def _quaternion_to_rotation_matrix(q: dict[str, Any]) -> np.ndarray:
    x = float(q["x"])
    y = float(q["y"])
    z = float(q["z"])
    w = float(q["w"])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("zero-length quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(points_camera: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    rotation = _quaternion_to_rotation_matrix(transform["rotation_quat"])
    t = transform["translation"]
    translation = np.array([float(t["x"]), float(t["y"]), float(t["z"])], dtype=np.float64)
    return points_camera @ rotation.T + translation


def _round(value: float) -> float:
    return round(float(value), 6)


def _shape_from_hint(shape_hint: str) -> str:
    return "cylinder" if shape_hint in {"cylinder", "cup_like"} else "box"


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    binary = np.asarray(mask, dtype=bool)
    output = np.zeros(binary.shape, dtype=bool)
    if not binary.any():
        return output, 0
    labels, stats = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=4)[1:3]
    if stats.shape[0] <= 1:
        return output, 0
    largest_label = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    output = labels == largest_label
    return output, int(stats[largest_label, cv2.CC_STAT_AREA])


def refine_mask_with_depth_support(
    *,
    mask: np.ndarray,
    depth_mm: np.ndarray,
    foreground_depth_window_mm: float = 80.0,
    min_supported_pixels: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    binary = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_mm)
    if binary.shape != depth.shape:
        raise ValueError(f"mask/depth shape mismatch: {binary.shape} vs {depth.shape}")

    valid = binary & (depth > 0)
    original_pixels = int(binary.sum())
    diagnostics: dict[str, Any] = {
        "schema_version": "m7_mask_refinement_v1",
        "original_pixels": original_pixels,
        "valid_depth_pixels": int(valid.sum()),
        "foreground_depth_mm": None,
        "foreground_depth_window_mm": float(foreground_depth_window_mm),
        "removed_far_depth_pixels": 0,
        "refined_pixels": 0,
        "fallback_used": False,
        "fallback_reason": "",
    }
    if original_pixels == 0:
        return np.zeros(binary.shape, dtype=bool), diagnostics

    min_pixels = int(max(1, min_supported_pixels))
    if int(valid.sum()) < min_pixels:
        largest, area = _largest_component(binary)
        diagnostics.update(
            {
                "refined_pixels": int(area),
                "fallback_used": True,
                "fallback_reason": "insufficient_valid_depth_support",
            }
        )
        return largest, diagnostics

    valid_depths = depth[valid].astype(np.float64)
    foreground_depth = float(np.percentile(valid_depths, 20.0))
    max_supported_depth = foreground_depth + max(float(foreground_depth_window_mm), 0.0)
    supported = valid & (depth.astype(np.float64) <= max_supported_depth)
    diagnostics["foreground_depth_mm"] = round(foreground_depth, 3)
    diagnostics["removed_far_depth_pixels"] = int((valid & ~supported).sum())

    if int(supported.sum()) < min_pixels:
        largest, area = _largest_component(binary)
        diagnostics.update(
            {
                "refined_pixels": int(area),
                "fallback_used": True,
                "fallback_reason": "depth_supported_component_too_small",
            }
        )
        return largest, diagnostics

    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(supported.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    largest, area = _largest_component(closed)
    diagnostics["refined_pixels"] = int(area)
    return largest, diagnostics


def _normalize_yaw_pi(yaw_rad: float) -> float:
    yaw = float(yaw_rad)
    while yaw <= -0.5 * math.pi:
        yaw += math.pi
    while yaw > 0.5 * math.pi:
        yaw -= math.pi
    return yaw


def estimate_oriented_footprint(
    points_xy_m: np.ndarray,
    *,
    square_anisotropy_threshold: float = 1.12,
    min_size_m: float = 0.008,
) -> dict[str, Any]:
    points = np.asarray(points_xy_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("points_xy_m must be an Nx2 array")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_xy_m must contain only finite values")

    center = points.mean(axis=0)
    if points.shape[0] < 3:
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        size = np.maximum(max_xy - min_xy, min_size_m)
        return {
            "schema_version": "m7_oriented_footprint_v1",
            "estimator": "axis_aligned_fallback",
            "center_xy_m": [float(center[0]), float(center[1])],
            "size_x_m": float(max(size[0], min_size_m)),
            "size_y_m": float(max(size[1], min_size_m)),
            "yaw_rad": 0.0,
            "anisotropy": float(max(size) / max(min(size), 1e-9)),
        }

    centered = points - center
    covariance = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    minor = np.array([-major[1], major[0]], dtype=np.float64)
    yaw = _normalize_yaw_pi(math.atan2(float(major[1]), float(major[0])))
    projections_major = centered @ major
    projections_minor = centered @ minor
    size_x = float(projections_major.max() - projections_major.min())
    size_y = float(projections_minor.max() - projections_minor.min())
    size_x = max(size_x, float(min_size_m))
    size_y = max(size_y, float(min_size_m))
    larger = max(size_x, size_y)
    smaller = min(size_x, size_y)
    anisotropy = larger / max(smaller, 1e-9)
    if anisotropy < float(square_anisotropy_threshold):
        size_x = size_y = larger
        yaw = 0.0
        estimator = "square_canonical"
    else:
        estimator = "pca_obb"
        if size_y > size_x:
            size_x, size_y = size_y, size_x
            yaw = _normalize_yaw_pi(yaw + 0.5 * math.pi)
    return {
        "schema_version": "m7_oriented_footprint_v1",
        "estimator": estimator,
        "center_xy_m": [float(center[0]), float(center[1])],
        "size_x_m": float(size_x),
        "size_y_m": float(size_y),
        "yaw_rad": float(yaw),
        "anisotropy": float(anisotropy),
    }


def _mask_bbox_aspect(mask: np.ndarray) -> float | None:
    binary = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(binary)
    if rows.size == 0:
        return None
    width = float(cols.max() - cols.min() + 1)
    height = float(rows.max() - rows.min() + 1)
    smaller = min(width, height)
    if smaller <= 0.0:
        return None
    return max(width, height) / smaller


def canonicalize_square_box_footprint_from_mask(
    footprint: dict[str, Any],
    *,
    refined_mask: np.ndarray | None,
    shape_type: str,
    square_mask_aspect_threshold: float = 1.35,
    depth_anisotropy_threshold: float = 1.45,
) -> dict[str, Any]:
    if str(shape_type) != "box" or refined_mask is None:
        return footprint
    mask_aspect = _mask_bbox_aspect(refined_mask)
    if mask_aspect is None or mask_aspect > float(square_mask_aspect_threshold):
        return footprint
    size_x = float(footprint["size_x_m"])
    size_y = float(footprint["size_y_m"])
    smaller = min(size_x, size_y)
    larger = max(size_x, size_y)
    if smaller <= 0.0 or larger / smaller < float(depth_anisotropy_threshold):
        return footprint
    canonical = dict(footprint)
    canonical["size_x_m"] = float(smaller)
    canonical["size_y_m"] = float(smaller)
    canonical["yaw_rad"] = 0.0
    canonical["anisotropy"] = 1.0
    canonical["estimator"] = f"{footprint.get('estimator', 'unknown')}_mask_square_canonical"
    canonical["mask_bbox_aspect"] = float(mask_aspect)
    canonical["canonicalization_reason"] = "square_mask_overrides_elongated_depth_obb"
    return canonical


def generate_grasp_yaw_candidates(
    *,
    shape_type: str,
    yaw_rad: float,
    size_x_m: float,
    size_y_m: float,
) -> list[dict[str, Any]]:
    shape = str(shape_type)
    yaw = _normalize_yaw_pi(float(yaw_rad))
    size_x = float(size_x_m)
    size_y = float(size_y_m)
    if shape == "cylinder":
        width = max(size_x, size_y)
        return [
            {
                "label": "cylinder_top_grasp",
                "yaw_rad": 0.0,
                "required_gripper_width_m": _round(width),
                "score": 1.0,
            }
        ]
    short_axis_yaw = _normalize_yaw_pi(yaw + 0.5 * math.pi)
    candidates = [
        {
            "label": "box_short_axis",
            "yaw_rad": _round(short_axis_yaw),
            "required_gripper_width_m": _round(min(size_x, size_y)),
            "score": 1.0,
        },
        {
            "label": "box_long_axis",
            "yaw_rad": _round(yaw),
            "required_gripper_width_m": _round(max(size_x, size_y)),
            "score": 0.75,
        },
    ]
    return candidates


def rank_geometry_hypotheses_for_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    shape_type = str(row["shape_type"])
    yaw = float(row.get("yaw_rad", 0.0))
    if shape_type == "cylinder":
        diameter = 2.0 * float(row["radius_m"])
        primary = {
            "rank": 1,
            "shape_type": "cylinder",
            "score": 1.0,
            "yaw_rad": 0.0,
            "radius_m": _round(float(row["radius_m"])),
            "height_m": _round(float(row["height_m"])),
            "reason": "shape_hint_primary",
        }
        alternate = {
            "rank": 2,
            "shape_type": "box",
            "score": 0.65,
            "yaw_rad": 0.0,
            "size_x_m": _round(diameter),
            "size_y_m": _round(diameter),
            "size_z_m": _round(float(row["height_m"])),
            "reason": "conservative_box_fallback",
        }
        return [primary, alternate]

    size_x = float(row["size_x_m"])
    size_y = float(row["size_y_m"])
    radius = 0.5 * max(size_x, size_y)
    primary = {
        "rank": 1,
        "shape_type": "box",
        "score": 1.0,
        "yaw_rad": _round(yaw),
        "size_x_m": _round(size_x),
        "size_y_m": _round(size_y),
        "size_z_m": _round(float(row["size_z_m"])),
        "reason": "shape_hint_primary",
    }
    alternate = {
        "rank": 2,
        "shape_type": "cylinder",
        "score": 0.55,
        "yaw_rad": 0.0,
        "radius_m": _round(radius),
        "height_m": _round(float(row["size_z_m"])),
        "reason": "round_proxy_fallback",
    }
    return [primary, alternate]


def build_target_row_from_points(
    *,
    observation_id: str,
    shape_hint: str,
    points_base: np.ndarray,
    object_height_m: float,
    pregrasp_clearance_m: float,
    refined_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if points_base.size == 0:
        raise ValueError("cannot build target from empty point set")
    shape_type = _shape_from_hint(shape_hint)
    footprint = estimate_oriented_footprint(points_base[:, :2])
    footprint = canonicalize_square_box_footprint_from_mask(
        footprint,
        refined_mask=refined_mask,
        shape_type=shape_type,
    )
    center_xy = np.asarray(footprint["center_xy_m"], dtype=np.float64)
    top_z = float(np.percentile(points_base[:, 2], 95.0))
    height = max(float(object_height_m), 0.005)
    center_z = top_z - 0.5 * height

    delta_xy = points_base[:, :2] - center_xy
    robust_radius = float(np.percentile(np.linalg.norm(delta_xy, axis=1), 90.0))
    size_x = max(float(footprint["size_x_m"]), 0.008)
    size_y = max(float(footprint["size_y_m"]), 0.008)
    row: dict[str, Any] = {
        "target_id": observation_id,
        "shape_type": shape_type,
        "center_x_m": _round(center_xy[0]),
        "center_y_m": _round(center_xy[1]),
        "center_z_m": _round(center_z),
        "yaw_rad": _round(footprint["yaw_rad"] if shape_type == "box" else 0.0),
        "pregrasp_clearance_m": float(pregrasp_clearance_m),
        "grasp_type": "top_grasp",
        "footprint": {
            **footprint,
            "center_xy_m": [_round(footprint["center_xy_m"][0]), _round(footprint["center_xy_m"][1])],
            "size_x_m": _round(footprint["size_x_m"]),
            "size_y_m": _round(footprint["size_y_m"]),
            "yaw_rad": _round(footprint["yaw_rad"]),
        },
        "valid": True,
        "failure_reason": "",
    }
    if shape_type == "cylinder":
        radius = max(robust_radius, 0.004)
        row.update(
            {
                "radius_m": _round(radius),
                "height_m": _round(height),
                "required_gripper_width_m": _round(2.0 * radius),
            }
        )
    else:
        row.update(
            {
                "size_x_m": _round(size_x),
                "size_y_m": _round(size_y),
                "size_z_m": _round(height),
                "required_gripper_width_m": _round(min(size_x, size_y)),
            }
        )
    row["grasp_yaw_candidates"] = generate_grasp_yaw_candidates(
        shape_type=shape_type,
        yaw_rad=float(row["yaw_rad"]),
        size_x_m=float(row.get("size_x_m", row.get("radius_m", 0.02) * 2.0)),
        size_y_m=float(row.get("size_y_m", row.get("radius_m", 0.02) * 2.0)),
    )
    row["ranked_geometry_hypotheses"] = rank_geometry_hypotheses_for_row(row)
    return row


def estimate_table_top_z(
    *,
    mask: np.ndarray,
    depth_mm: np.ndarray,
    camera_info: dict[str, Any],
    transform: dict[str, Any],
) -> float:
    outside = ~mask.astype(bool)
    valid_outside = outside & (depth_mm > 0)
    if not valid_outside.any():
        return 0.0
    points_camera = project_mask_depth_to_camera_points(
        mask=valid_outside,
        depth_mm=depth_mm,
        camera_info=camera_info,
    )
    points_base = transform_points(points_camera, transform)
    return float(np.median(points_base[:, 2]))


def _snapshot_dir_from_decision(decision: dict[str, Any], base_dir: Path) -> tuple[Path, dict[str, Any]]:
    observation_path = _resolve_path(str(decision["shadow_observation_path"]), base_dir)
    observation = _read_json(observation_path)
    snapshot_dir = _resolve_path(str(observation["snapshot"]["dir"]), observation_path.parent)
    return snapshot_dir, observation


def lookup_live_transform(
    *,
    parent_frame: str,
    child_frame: str,
    timeout_sec: float,
) -> dict[str, Any]:
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.time import Time
    import tf2_ros

    if not rclpy.ok():
        rclpy.init(args=None)
        owns_context = True
    else:
        owns_context = False
    node = Node("m6_shadow_grasp_target_tf_lookup")
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer, node)
    try:
        deadline = time.time() + float(timeout_sec)
        last_error: Exception | None = None
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                transform = buffer.lookup_transform(
                    parent_frame,
                    child_frame,
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
                t = transform.transform.translation
                q = transform.transform.rotation
                return {
                    "parent_frame": parent_frame,
                    "child_frame": child_frame,
                    "translation": {"x": float(t.x), "y": float(t.y), "z": float(t.z)},
                    "rotation_quat": {
                        "x": float(q.x),
                        "y": float(q.y),
                        "z": float(q.z),
                        "w": float(q.w),
                    },
                }
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"timed out waiting for TF {parent_frame}->{child_frame}: {last_error}")
    finally:
        node.destroy_node()
        if owns_context and rclpy.ok():
            rclpy.shutdown()


def _render_index(report: dict[str, Any]) -> str:
    target = report["target"]
    lines = [
        "# M6 Shadow Grasp Target",
        "",
        f"- observation_id: `{report['observation_id']}`",
        f"- safety_mode: `{report['safety_mode']}`",
        f"- motion_authorized: `{str(report['motion_authorized']).lower()}`",
        f"- target_id: `{target['target_id']}`",
        f"- shape_type: `{target['shape_type']}`",
        f"- center_xyz_m: `{target['center_x_m']}, {target['center_y_m']}, {target['center_z_m']}`",
        f"- table_top_z_m: `{report['table_top_z_m']:.6f}`",
        f"- targets_path: `{report['targets_path']}`",
        "",
        "This target is for MoveIt shadow planning only and does not authorize real motion.",
        "",
    ]
    return "\n".join(lines)


def generate_shadow_grasp_target(
    *,
    shadow_decision_path: Path,
    output_dir: Path,
    camera_to_base_transform: dict[str, Any],
    object_height_m: float = 0.025,
    pregrasp_clearance_m: float = 0.095,
) -> dict[str, Any]:
    shadow_decision_path = Path(shadow_decision_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = _read_json(shadow_decision_path)
    if not decision.get("ready_for_shadow_planning"):
        raise ValueError("shadow decision is not ready for shadow planning")
    snapshot_dir, _observation = _snapshot_dir_from_decision(decision, shadow_decision_path.parent)
    depth_mm = np.load(snapshot_dir / "aligned_depth_raw.npy")
    camera_info = _read_json(snapshot_dir / "aligned_depth_camera_info.json")
    mask_path = _resolve_path(str(decision["mask_path"]), shadow_decision_path.parent)
    mask = _read_mask(mask_path, depth_mm.shape)
    refined_mask, mask_refinement = refine_mask_with_depth_support(
        mask=mask,
        depth_mm=depth_mm,
    )
    refined_mask_path = output_dir / "refined_target_mask.png"
    cv2.imwrite(str(refined_mask_path), refined_mask.astype(np.uint8) * 255)
    points_camera = project_mask_depth_to_camera_points(
        mask=refined_mask,
        depth_mm=depth_mm,
        camera_info=camera_info,
    )
    points_base = transform_points(points_camera, camera_to_base_transform)
    target = build_target_row_from_points(
        observation_id=str(decision["observation_id"]),
        shape_hint=str(decision.get("shape_hint", "unknown")),
        points_base=points_base,
        object_height_m=object_height_m,
        pregrasp_clearance_m=pregrasp_clearance_m,
        refined_mask=refined_mask,
    )
    table_top_z_m = estimate_table_top_z(
        mask=mask,
        depth_mm=depth_mm,
        camera_info=camera_info,
        transform=camera_to_base_transform,
    )
    targets_payload = {
        "schema_version": "m4_sim_grasp_targets_v1",
        "rows": [target],
    }
    targets_path = output_dir / "m6_shadow_grasp_targets.json"
    _write_json(targets_path, targets_payload)
    report = {
        "schema_version": "m6_shadow_grasp_target_v1",
        "generated_at_utc": _utc_now(),
        "observation_id": str(decision["observation_id"]),
        "safety_mode": "shadow_only_no_motion",
        "motion_authorized": False,
        "shadow_decision_path": str(shadow_decision_path),
        "targets_path": str(targets_path),
        "camera_frame": str(camera_info.get("frame_id", "")),
        "camera_to_base_transform": camera_to_base_transform,
        "table_top_z_m": table_top_z_m,
        "mask_refinement": mask_refinement | {"refined_mask_path": str(refined_mask_path)},
        "target": target,
    }
    _write_json(output_dir / "m6_shadow_grasp_target.json", report)
    (output_dir / "index.md").write_text(_render_index(report), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-frame", default="world")
    parser.add_argument("--child-frame")
    parser.add_argument("--tf-timeout-sec", type=float, default=6.0)
    parser.add_argument("--object-height-m", type=float, default=0.025)
    parser.add_argument("--pregrasp-clearance-m", type=float, default=0.095)
    parser.add_argument("--transform-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    decision = _read_json(args.shadow_decision)
    snapshot_dir, _observation = _snapshot_dir_from_decision(decision, args.shadow_decision.parent)
    camera_info = _read_json(snapshot_dir / "aligned_depth_camera_info.json")
    child_frame = args.child_frame or str(camera_info.get("frame_id", "camera_color_optical_frame"))
    if args.transform_json:
        transform = _read_json(args.transform_json)
    else:
        transform = lookup_live_transform(
            parent_frame=args.parent_frame,
            child_frame=child_frame,
            timeout_sec=args.tf_timeout_sec,
        )
    report = generate_shadow_grasp_target(
        shadow_decision_path=args.shadow_decision,
        output_dir=args.output_dir,
        camera_to_base_transform=transform,
        object_height_m=args.object_height_m,
        pregrasp_clearance_m=args.pregrasp_clearance_m,
    )
    print(
        "M6 shadow grasp target: "
        f"{report['target']['target_id']} -> {report['targets_path']}"
    )


if __name__ == "__main__":
    main()
