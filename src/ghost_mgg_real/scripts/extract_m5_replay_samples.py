#!/usr/bin/env python3
"""Extract replay-check PNG samples from M5 real D435 rosbag2 captures."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TOPIC_OUTPUTS = {
    "/camera/camera/color/image_raw": ("color", "color.png"),
    "/camera/camera/depth/image_rect_raw": ("depth", "depth_viz.png"),
    "/camera/camera/aligned_depth_to_color/image_raw": (
        "aligned_depth",
        "aligned_depth_viz.png",
    ),
    "/camera/camera/infra1/image_rect_raw": ("infra1", "infra1.png"),
    "/camera/camera/infra2/image_rect_raw": ("infra2", "infra2.png"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def decode_image_msg(msg: Any) -> np.ndarray:
    encoding = str(msg.encoding)
    height = int(msg.height)
    width = int(msg.width)
    if encoding == "16UC1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width)
    if encoding == "mono8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
    if encoding in {"rgb8", "bgr8"}:
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
    raise ValueError(f"unsupported image encoding: {encoding}")


def _depth_to_viz(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    if not valid.any():
        return np.zeros(depth.shape, dtype=np.uint8)
    valid_values = depth[valid].astype(np.float32)
    low = float(np.percentile(valid_values, 2.0))
    high = float(np.percentile(valid_values, 98.0))
    if high <= low:
        high = low + 1.0
    clipped = np.clip(depth.astype(np.float32), low, high)
    viz = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)
    viz[~valid] = 0
    return viz


def _image_metrics(image: np.ndarray, encoding: str) -> dict[str, float]:
    if encoding == "16UC1":
        valid = image > 0
        metrics = {"valid_ratio": float(valid.mean())}
        if valid.any():
            valid_m = image[valid].astype(np.float64) / 1000.0
            metrics.update(
                {
                    "mean_valid_depth_m": float(valid_m.mean()),
                    "min_valid_depth_m": float(valid_m.min()),
                    "max_valid_depth_m": float(valid_m.max()),
                }
            )
        return metrics
    return {
        "mean_intensity": float(image.mean()),
        "nonzero_ratio": float((image > 0).mean()),
        "saturated_ratio": float((image >= 250).mean()),
    }


def _json_safe_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    safe: dict[str, float | None] = {}
    for key, value in metrics.items():
        safe[key] = None if isinstance(value, float) and math.isnan(value) else value
    return safe


def write_image_png(output_path: Path, image: np.ndarray, encoding: str) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if encoding == "16UC1":
        writable = _depth_to_viz(image)
    elif encoding == "rgb8":
        writable = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        writable = image

    if not cv2.imwrite(str(output_path), writable):
        raise RuntimeError(f"failed to write image: {output_path}")

    metadata: dict[str, Any] = {
        "encoding": encoding,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "output_path": output_path.name,
    }
    metadata.update(_json_safe_metrics(_image_metrics(image, encoding)))
    return metadata


def write_scene_sample_outputs(
    scene_id: str,
    frames: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    scene_dir = output_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    frame_metadata: dict[str, dict[str, Any]] = {}
    missing_topics: list[str] = []

    for topic, (output_key, filename) in TOPIC_OUTPUTS.items():
        msg = frames.get(topic)
        if msg is None:
            missing_topics.append(topic)
            continue
        image = decode_image_msg(msg)
        image_metadata = write_image_png(scene_dir / filename, image, str(msg.encoding))
        outputs[output_key] = f"{scene_id}/{filename}"
        frame_metadata[topic] = image_metadata

    metadata = {
        "schema_version": "m5_replay_sample_scene_v1",
        "scene_id": scene_id,
        "generated_at_utc": _utc_now(),
        "outputs": outputs,
        "missing_topics": missing_topics,
        "frames": frame_metadata,
    }
    (scene_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "scene_id": scene_id,
        "outputs": outputs,
        "metadata_path": f"{scene_id}/metadata.json",
        "missing_topics": missing_topics,
    }


def read_first_image_frames(scene_dir: Path) -> dict[str, Any]:
    from rclpy.serialization import deserialize_message
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message

    storage_options = rosbag2_py.StorageOptions(uri=str(scene_dir), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    image_type = get_message("sensor_msgs/msg/Image")
    frames: dict[str, Any] = {}

    while reader.has_next():
        topic, data, _timestamp = reader.read_next()
        if topic not in TOPIC_OUTPUTS or topic in frames:
            continue
        if topic_types.get(topic) != "sensor_msgs/msg/Image":
            continue
        frames[topic] = deserialize_message(data, image_type)
        if len(frames) == len(TOPIC_OUTPUTS):
            break

    return frames


def extract_replay_samples(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    scene_dirs = sorted(path.parent for path in data_dir.glob("*/metadata.yaml"))
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = []
    for scene_dir in scene_dirs:
        frames = read_first_image_frames(scene_dir)
        scenes.append(write_scene_sample_outputs(scene_dir.name, frames, output_dir))

    manifest = {
        "schema_version": "m5_replay_samples_manifest_v1",
        "generated_at_utc": _utc_now(),
        "source_data_dir": str(data_dir),
        "num_scenes": len(scenes),
        "scenes": scenes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_index_markdown(manifest, output_dir)
    return manifest


def write_index_markdown(manifest: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# M5 Replay Samples",
        "",
        f"- Scenes: {manifest.get('num_scenes', len(manifest.get('scenes', [])))}",
        "",
        "Each row links the first replay-check frame extracted from a real D435 bag.",
        "",
        "| scene_id | color | depth | aligned depth | infra1 | infra2 |",
        "|---|---|---|---|---|---|",
    ]
    for scene in manifest.get("scenes", []):
        outputs = scene.get("outputs", {})
        lines.append(
            "| "
            f"{scene['scene_id']} | "
            f"{_md_link(outputs.get('color'))} | "
            f"{_md_link(outputs.get('depth'))} | "
            f"{_md_link(outputs.get('aligned_depth'))} | "
            f"{_md_link(outputs.get('infra1'))} | "
            f"{_md_link(outputs.get('infra2'))} |"
        )
    lines.append("")
    (output_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _md_link(path: str | None) -> str:
    if not path:
        return ""
    return f"[{Path(path).name}]({path})"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/real_d435_m5"),
        help="Directory containing M5 rosbag2 scene directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m5_real_d435_replay_samples"),
        help="Output directory for replay PNG samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = extract_replay_samples(args.data_dir, args.output_dir)
    print(f"Wrote replay samples for {manifest['num_scenes']} scenes to {args.output_dir}")


if __name__ == "__main__":
    main()
