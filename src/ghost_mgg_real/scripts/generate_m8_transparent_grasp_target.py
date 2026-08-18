#!/usr/bin/env python3
"""Bridge: live transparent proxy hypothesis -> m6_shadow_grasp_target.json.

The mask/depth-based m6 shadow chain cannot see a transparent object (empty
color mask, no valid depth under the mask). This bridge listens to the live
hypothesis stream, selects the transparent existence proxy (or any
hypothesis via --select any), averages its pose over a short window and
writes a grasp-target JSON compatible with grasp_m7_real_once.sh /
m7_real_grasp_marker_node.py. Downstream, the executor consumes only
center_x_m / center_y_m (+ taught z from env), which is exactly what the
proxy provides with mm-level replay-validated accuracy.

Usage:
  generate_m8_transparent_grasp_target.py --out <target.json> \
      [--select proxy|any] [--hint-x X --hint-y Y] [--window-s 4.0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ghost_mgg_interfaces.msg import GeometryHypothesisArray


def quaternion_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--topic", default="/ghost_mgg/m4_live_hypotheses")
    parser.add_argument("--select", choices=["proxy", "any", "auto"], default="auto")
    parser.add_argument("--hint-x", type=float, default=None)
    parser.add_argument("--hint-y", type=float, default=None)
    parser.add_argument("--window-s", type=float, default=4.0)
    parser.add_argument("--observation-id", default=os.environ.get("M7_OBS", "m7_real_current"))
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("m8_transparent_grasp_target_bridge")
    samples: list[tuple[float, float, float, float, float, float, float]] = []

    def on_message(message: GeometryHypothesisArray) -> None:
        proxies = [
            h for h in message.hypotheses
            if "hole_existence_only" in str(h.provenance)
        ]
        if args.select == "proxy":
            candidates = proxies
        elif args.select == "any":
            candidates = list(message.hypotheses)
        else:  # auto: prefer the proxy, fall back to real hypotheses (an
            # object with partial returns is represented by a real fit)
            candidates = proxies or list(message.hypotheses)
        # a graspable object is never an extreme thin strip; those are table
        # artifacts (tape lines, edges). Drop them when alternatives exist.
        compact = [
            h for h in candidates
            if max(h.dimensions_m.x, h.dimensions_m.y)
            / max(1e-6, min(h.dimensions_m.x, h.dimensions_m.y)) <= 6.0
        ]
        if compact:
            candidates = compact
        if not candidates:
            return
        if args.hint_x is not None and args.hint_y is not None:
            best = min(
                candidates,
                key=lambda h: math.hypot(
                    h.pose_base.pose.position.x - args.hint_x,
                    h.pose_base.pose.position.y - args.hint_y,
                ),
            )
        else:
            best = max(
                candidates,
                key=lambda h: float(h.dimensions_m.x) * float(h.dimensions_m.y),
            )
        samples.append(
            (
                float(best.pose_base.pose.position.x),
                float(best.pose_base.pose.position.y),
                float(best.pose_base.pose.position.z),
                float(best.dimensions_m.x),
                float(best.dimensions_m.y),
                float(best.dimensions_m.z),
                quaternion_yaw(best.pose_base.pose.orientation),
            )
        )

    qos = QoSProfile(depth=10)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    node.create_subscription(GeometryHypothesisArray, args.topic, on_message, qos)
    deadline = time.monotonic() + args.window_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()

    if len(samples) < 2:
        raise SystemExit(
            f"no usable hypothesis observed on {args.topic} within "
            f"{args.window_s}s (got {len(samples)} samples; is the M8 "
            f"launch running and the object stable in RViz?)"
        )

    arr = np.asarray(samples)
    center_x, center_y, center_z = (float(np.median(arr[:, i])) for i in range(3))
    size_x, size_y, size_z = (float(np.median(arr[:, i])) for i in range(3, 6))
    yaw = float(np.median(arr[:, 6]))
    position_spread_mm = float(
        1000.0 * max(arr[:, 0].max() - arr[:, 0].min(), arr[:, 1].max() - arr[:, 1].min())
    )

    short_axis, long_axis = sorted([size_x, size_y])
    yaw_candidates = [
        {
            "label": "box_short_axis",
            "required_gripper_width_m": round(short_axis, 6),
            "score": 1.0,
            "yaw_rad": round(yaw if size_x >= size_y else yaw + 0.5 * math.pi, 6),
        },
        {
            "label": "box_long_axis",
            "required_gripper_width_m": round(long_axis, 6),
            "score": 0.75,
            "yaw_rad": round(yaw + 0.5 * math.pi if size_x >= size_y else yaw, 6),
        },
    ]
    payload = {
        "schema_version": "m6_shadow_grasp_target_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "observation_id": args.observation_id,
        "source": "m8_transparent_proxy_bridge",
        "selection": {
            "mode": args.select,
            "samples": len(samples),
            "window_s": args.window_s,
            "position_spread_mm": round(position_spread_mm, 1),
        },
        "motion_authorized": False,
        "safety_mode": "shadow_only_no_motion",
        "table_top_z_m": round(center_z - 0.5 * size_z, 6),
        "target": {
            "center_x_m": round(center_x, 6),
            "center_y_m": round(center_y, 6),
            "center_z_m": round(center_z, 6),
            "failure_reason": "",
            "footprint": {
                "center_xy_m": [round(center_x, 6), round(center_y, 6)],
                "estimator": "m8_transparent_proxy",
                "schema_version": "m7_oriented_footprint_v1",
                "size_x_m": round(size_x, 6),
                "size_y_m": round(size_y, 6),
                "yaw_rad": round(yaw, 6),
                "anisotropy": round(max(size_x, size_y) / max(1e-6, min(size_x, size_y)), 4),
            },
            "grasp_type": "top_grasp",
            "grasp_yaw_candidates": yaw_candidates,
            "height_m": round(size_z, 6),
            "size_z_m": round(size_z, 6),
            "m7_target_offset_m": {"x": 0.0, "y": 0.0, "z": 0.0},
            "pregrasp_clearance_m": 0.095,
            "shape": "box",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(
        f"bridge target: ({center_x:.3f},{center_y:.3f}) {size_x*1000:.0f}x{size_y*1000:.0f}mm "
        f"yaw={yaw:+.2f} samples={len(samples)} spread={position_spread_mm:.1f}mm -> {args.out}"
    )
    if position_spread_mm > 15.0:
        print("WARNING: proxy position spread >15mm during the window — scene not settled?")


if __name__ == "__main__":
    main()
