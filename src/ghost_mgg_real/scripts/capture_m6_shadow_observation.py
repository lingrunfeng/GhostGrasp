#!/usr/bin/env python3
"""Capture one M6 shadow observation from live D435 topics and robot state.

This script is read-only. It records the current live observation, one
/joint_states sample, and the current base_link -> camera_link transform.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from capture_m5_5_real_online_snapshot import capture_live_topic_snapshot  # noqa: E402


ARM_JOINT_NAMES = (
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_link6",
    "link6_to_link6_flange",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _time_to_dict(stamp: Any) -> dict[str, int]:
    return {
        "sec": int(getattr(stamp, "sec", 0)),
        "nanosec": int(getattr(stamp, "nanosec", 0)),
    }


def joint_state_to_dict(msg: Any) -> dict[str, Any]:
    return {
        "stamp": _time_to_dict(msg.header.stamp),
        "stamp_sec": int(msg.header.stamp.sec),
        "stamp_nanosec": int(msg.header.stamp.nanosec),
        "frame_id": str(msg.header.frame_id),
        "name": list(msg.name),
        "position": [float(value) for value in msg.position],
        "velocity": [float(value) for value in msg.velocity],
        "effort": [float(value) for value in msg.effort],
    }


def transform_to_dict(transform: Any, *, parent_frame: str, child_frame: str) -> dict[str, Any]:
    t = transform.transform.translation
    q = transform.transform.rotation
    return {
        "parent_frame": parent_frame,
        "child_frame": child_frame,
        "stamp": _time_to_dict(transform.header.stamp),
        "translation": {
            "x": float(t.x),
            "y": float(t.y),
            "z": float(t.z),
        },
        "rotation_quat": {
            "x": float(q.x),
            "y": float(q.y),
            "z": float(q.z),
            "w": float(q.w),
        },
    }


def wait_for_joint_state(*, timeout_sec: float) -> dict[str, Any]:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class JointStateOnce(Node):
        def __init__(self) -> None:
            super().__init__("m6_shadow_observation_joint_state_once")
            self.sample: Any | None = None
            self.subscription = self.create_subscription(
                JointState,
                "/joint_states",
                self._callback,
                10,
            )

        def _callback(self, msg: Any) -> None:
            if self.sample is None:
                self.sample = msg

    rclpy.init(args=None)
    node = JointStateOnce()
    try:
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline and node.sample is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.sample is None:
            raise RuntimeError("timed out waiting for /joint_states")
        return joint_state_to_dict(node.sample)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def wait_for_transform(
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

    rclpy.init(args=None)
    node = Node("m6_shadow_observation_tf_once")
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
                return transform_to_dict(
                    transform,
                    parent_frame=parent_frame,
                    child_frame=child_frame,
                )
            except Exception as exc:  # tf2 exception types vary by distro.
                last_error = exc
        raise RuntimeError(
            f"timed out waiting for TF {parent_frame} -> {child_frame}: {last_error}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def write_shadow_observation_report(
    *,
    observation_id: str,
    output_dir: Path,
    snapshot_dir: Path,
    snapshot_manifest: dict[str, Any],
    joint_state: dict[str, Any],
    camera_to_base: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_joint_set = set(ARM_JOINT_NAMES)
    joint_names = set(joint_state.get("name", []))
    gate_checks = {
        "has_snapshot": bool(snapshot_manifest.get("copied_files")),
        "has_real_arm_joints": arm_joint_set.issubset(joint_names),
        "has_camera_to_base_tf": bool(
            camera_to_base.get("parent_frame") == "base_link"
            and camera_to_base.get("child_frame") == "camera_link"
        ),
        "has_aligned_depth_raw": "aligned_depth_raw.npy"
        in snapshot_manifest.get("copied_files", []),
    }
    report = {
        "schema_version": "m6_shadow_observation_v1",
        "generated_at_utc": _utc_now(),
        "observation_id": observation_id,
        "safety_mode": "shadow_only_no_motion",
        "snapshot": {
            "dir": str(snapshot_dir),
            "manifest": snapshot_manifest,
        },
        "joint_state": joint_state,
        "camera_to_base": camera_to_base,
        "gate_checks": gate_checks,
        "next_steps": [
            "review color/depth/pointcloud in RViz",
            "annotate one target mask",
            "generate masked evidence and backend selection",
            "run MoveIt plan-only before any real execution",
        ],
    }
    _write_json(output_dir / "m6_shadow_observation.json", report)
    (output_dir / "index.md").write_text(_render_index(report), encoding="utf-8")
    return report


def _render_index(report: dict[str, Any]) -> str:
    checks = report["gate_checks"]
    tf = report["camera_to_base"]["translation"]
    lines = [
        "# M6 Shadow Observation",
        "",
        f"- observation_id: `{report['observation_id']}`",
        f"- safety_mode: `{report['safety_mode']}`",
        f"- snapshot_dir: `{report['snapshot']['dir']}`",
        "",
        "## Gate Checks",
        "",
        "| check | status |",
        "|---|---|",
    ]
    for name, status in checks.items():
        lines.append(f"| {name} | {'pass' if status else 'fail'} |")
    lines.extend(
        [
            "",
            "## Current Rough TF",
            "",
            f"- base_link -> camera_link translation: x={tf['x']:.3f}, y={tf['y']:.3f}, z={tf['z']:.3f}",
            "",
            "This report is shadow-only and does not authorize real motion.",
            "",
        ]
    )
    return "\n".join(lines)


def capture_shadow_observation(
    *,
    observation_id: str,
    output_root: Path,
    parent_frame: str,
    child_frame: str,
    timeout_sec: float,
) -> dict[str, Any]:
    output_dir = Path(output_root) / observation_id
    snapshot_dir = output_dir / "snapshot"
    snapshot_manifest = capture_live_topic_snapshot(
        output_dir=snapshot_dir,
        observation_id=observation_id,
        timeout_sec=timeout_sec,
    )
    joint_state = wait_for_joint_state(timeout_sec=timeout_sec)
    camera_to_base = wait_for_transform(
        parent_frame=parent_frame,
        child_frame=child_frame,
        timeout_sec=timeout_sec,
    )
    return write_shadow_observation_report(
        observation_id=observation_id,
        output_dir=output_dir,
        snapshot_dir=snapshot_dir,
        snapshot_manifest=snapshot_manifest,
        joint_state=joint_state,
        camera_to_base=camera_to_base,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-id", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/m6_shadow_observations"),
    )
    parser.add_argument("--parent-frame", default="base_link")
    parser.add_argument("--child-frame", default="camera_link")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    observation_id = args.observation_id or "m6_shadow_" + datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    report = capture_shadow_observation(
        observation_id=observation_id,
        output_root=args.output_root,
        parent_frame=args.parent_frame,
        child_frame=args.child_frame,
        timeout_sec=args.timeout_sec,
    )
    print(
        "M6 shadow observation captured: "
        f"{report['observation_id']} -> {Path(args.output_root) / report['observation_id']}"
    )


if __name__ == "__main__":
    main()
