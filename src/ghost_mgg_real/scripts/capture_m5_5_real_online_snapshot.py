#!/usr/bin/env python3
"""Capture a M5.5 observation snapshot from live topics or an offline sample."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_m5_replay_samples import (  # noqa: E402
    TOPIC_OUTPUTS,
    decode_image_msg,
    write_scene_sample_outputs,
)


SNAPSHOT_FILES = (
    "metadata.json",
    "color.png",
    "depth_viz.png",
    "aligned_depth_viz.png",
    "infra1.png",
    "infra2.png",
    "aligned_depth_camera_info.json",
)

CAMERA_INFO_TOPICS = {
    "/camera/camera/aligned_depth_to_color/camera_info": "aligned_depth_camera_info.json",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def capture_offline_sample_snapshot(
    *,
    offline_sample_dir: Path,
    output_dir: Path,
    observation_id: str,
) -> dict[str, Any]:
    offline_sample_dir = Path(offline_sample_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    missing_files: list[str] = []
    for filename in SNAPSHOT_FILES:
        source = offline_sample_dir / filename
        if not source.exists():
            missing_files.append(filename)
            continue
        shutil.copy2(source, output_dir / filename)
        copied_files.append(filename)

    manifest = {
        "schema_version": "m5_5_real_online_snapshot_v1",
        "generated_at_utc": _utc_now(),
        "observation_id": observation_id,
        "source": "offline_replay_sample",
        "source_dir": str(offline_sample_dir),
        "copied_files": copied_files,
        "missing_files": missing_files,
        "metadata_path": "metadata.json" if "metadata.json" in copied_files else None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_live_snapshot_outputs(
    *,
    observation_id: str,
    frames: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_parent = output_dir.parent / f".{output_dir.name}_tmp"
    if temp_parent.exists():
        shutil.rmtree(temp_parent)
    temp_parent.mkdir(parents=True)
    scene_output = write_scene_sample_outputs(observation_id, frames, temp_parent)
    scene_dir = temp_parent / observation_id

    copied_files: list[str] = []
    for path in sorted(scene_dir.iterdir()):
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)
            copied_files.append(path.name)
    shutil.rmtree(temp_parent)
    raw_files = _write_raw_depth_arrays(frames, output_dir)
    copied_files.extend(raw_files)
    camera_info_files = _write_camera_info_files(frames, output_dir)
    copied_files.extend(camera_info_files)

    manifest = {
        "schema_version": "m5_5_real_online_snapshot_v1",
        "generated_at_utc": _utc_now(),
        "observation_id": observation_id,
        "source": "live_ros_topics",
        "copied_files": copied_files,
        "missing_topics": scene_output.get("missing_topics", []),
        "metadata_path": "metadata.json",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_raw_depth_arrays(frames: dict[str, Any], output_dir: Path) -> list[str]:
    topic_to_filename = {
        "/camera/camera/depth/image_rect_raw": "depth_raw.npy",
        "/camera/camera/aligned_depth_to_color/image_raw": "aligned_depth_raw.npy",
    }
    written: list[str] = []
    for topic, filename in topic_to_filename.items():
        msg = frames.get(topic)
        if msg is None:
            continue
        if str(msg.encoding) != "16UC1":
            continue
        image = decode_image_msg(msg)
        npy_path = output_dir / filename
        import numpy as np

        np.save(npy_path, image)
        written.append(filename)
    return written


def _camera_info_to_dict(msg: Any) -> dict[str, Any]:
    header = getattr(msg, "header", None)
    return {
        "schema_version": "camera_info_snapshot_v1",
        "frame_id": str(getattr(header, "frame_id", "")),
        "width": int(getattr(msg, "width")),
        "height": int(getattr(msg, "height")),
        "distortion_model": str(getattr(msg, "distortion_model", "")),
        "d": [float(value) for value in getattr(msg, "d", [])],
        "k": [float(value) for value in getattr(msg, "k", [])],
        "r": [float(value) for value in getattr(msg, "r", [])],
        "p": [float(value) for value in getattr(msg, "p", [])],
    }


def _write_camera_info_files(frames: dict[str, Any], output_dir: Path) -> list[str]:
    written: list[str] = []
    for topic, filename in CAMERA_INFO_TOPICS.items():
        msg = frames.get(topic)
        if msg is None:
            continue
        path = output_dir / filename
        path.write_text(
            json.dumps(_camera_info_to_dict(msg), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(filename)
    return written


def capture_live_topic_snapshot(
    *,
    output_dir: Path,
    observation_id: str,
    timeout_sec: float,
) -> dict[str, Any]:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image

    class SnapshotNode(Node):
        def __init__(self) -> None:
            super().__init__("m5_5_real_online_snapshot")
            self.frames: dict[str, Any] = {}
            self.subscriptions_ = [
                self.create_subscription(
                    Image,
                    topic,
                    self._callback_factory(topic),
                    10,
                )
                for topic in TOPIC_OUTPUTS
            ]
            self.subscriptions_.extend(
                self.create_subscription(
                    CameraInfo,
                    topic,
                    self._callback_factory(topic),
                    10,
                )
                for topic in CAMERA_INFO_TOPICS
            )

        def _callback_factory(self, topic: str):
            def callback(msg: Any) -> None:
                self.frames.setdefault(topic, msg)

            return callback

    rclpy.init(args=None)
    node = SnapshotNode()
    try:
        deadline = time.time() + float(timeout_sec)
        required_topics = set(TOPIC_OUTPUTS) | set(CAMERA_INFO_TOPICS)
        while time.time() < deadline and not required_topics.issubset(node.frames):
            rclpy.spin_once(node, timeout_sec=0.1)
        missing = sorted(topic for topic in required_topics if topic not in node.frames)
        if missing:
            raise RuntimeError(f"timed out waiting for image topics: {missing}")
        return write_live_snapshot_outputs(
            observation_id=observation_id,
            frames=node.frames,
            output_dir=output_dir,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-sample-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation-id", type=str, default="live_000001")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Capture one frame from live ROS image topics.",
    )
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.live:
        manifest = capture_live_topic_snapshot(
            output_dir=args.output_dir,
            observation_id=args.observation_id,
            timeout_sec=args.timeout_sec,
        )
    else:
        if args.offline_sample_dir is None:
            raise SystemExit("--offline-sample-dir is required unless --live is set")
        manifest = capture_offline_sample_snapshot(
            offline_sample_dir=args.offline_sample_dir,
            output_dir=args.output_dir,
            observation_id=args.observation_id,
        )
    print(
        "M5.5 snapshot: "
        f"{manifest['observation_id']} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
