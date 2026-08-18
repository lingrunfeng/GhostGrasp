#!/usr/bin/env python3
"""Build M5 real-D435 manifests and lightweight frame statistics from rosbag2 data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REQUIRED_TOPICS = (
    "/camera/camera/color/image_raw",
    "/camera/camera/color/camera_info",
    "/camera/camera/depth/image_rect_raw",
    "/camera/camera/depth/camera_info",
    "/camera/camera/aligned_depth_to_color/image_raw",
    "/camera/camera/aligned_depth_to_color/camera_info",
    "/camera/camera/infra1/image_rect_raw",
    "/camera/camera/infra1/camera_info",
    "/camera/camera/infra2/image_rect_raw",
    "/camera/camera/infra2/camera_info",
    "/camera/camera/extrinsics/depth_to_color",
    "/camera/camera/extrinsics/depth_to_infra1",
    "/camera/camera/extrinsics/depth_to_infra2",
    "/tf_static",
)

IMAGE_TOPICS = (
    "/camera/camera/color/image_raw",
    "/camera/camera/depth/image_rect_raw",
    "/camera/camera/aligned_depth_to_color/image_raw",
    "/camera/camera/infra1/image_rect_raw",
    "/camera/camera/infra2/image_rect_raw",
)

SCENE_NOTES = {
    "empty_table_001": "Empty-table baseline for table plane and background depth statistics.",
    "daylight_opaque_box_001": "Opaque single-object baseline with reliable depth expected.",
    "daylight_transparent_jelly_cup_001": "Transparent jelly cup; field observation: severe point-cloud loss and dark depth shadow.",
    "daylight_transparent_jelly_cup_yaw45_001": "Same jelly cup at changed yaw/pose for failure-pattern sensitivity.",
    "daylight_glass_cup_001": "Glass cup, cup-like/OOD transparent object.",
    "daylight_frosted_plastic_bowl_001": "Frosted translucent bowl, cup-like semi-transparent object.",
    "daylight_reflective_object_001": "Metal cup / reflective object sample.",
    "daylight_metal_spoon_001": "Metal spoon reflective thin-object sample.",
    "daylight_multi_objects_001": "Multi-object scene; observed failure severity: jelly cup > frosted bowl > spoon.",
    "lowlight_transparent_jelly_cup_001": "Low-light transparent jelly cup with IR/depth still usable.",
    "dark_ir_transparent_jelly_cup_001": "Near-dark scene; RGB unusable, IR visible, point cloud present.",
    "daylight_transparent_jelly_cup_offcenter_001": "Jelly cup placed in RGB upper-right / off-center view.",
    "daylight_transparent_jelly_cup_visible_points_001": "Same jelly cup in a pose with more visible point returns.",
}


@dataclass(frozen=True)
class BagSummary:
    scene_id: str
    duration_sec: float
    message_count: int
    storage_identifier: str
    bag_size_bytes: int
    topic_counts: dict[str, int]
    missing_required_topics: list[str]
    note: str

    @property
    def is_complete(self) -> bool:
        return not self.missing_required_topics


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bag_size_bytes(scene_dir: Path) -> int:
    return sum(path.stat().st_size for path in scene_dir.glob("*.mcap"))


def parse_bag_metadata(scene_dir: Path) -> BagSummary:
    metadata_path = scene_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing rosbag metadata: {metadata_path}")

    raw = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    info = raw["rosbag2_bagfile_information"]
    duration_ns = int(info["duration"]["nanoseconds"])
    topic_counts = {
        item["topic_metadata"]["name"]: int(item["message_count"])
        for item in info.get("topics_with_message_count", [])
    }
    missing = [topic for topic in REQUIRED_TOPICS if topic_counts.get(topic, 0) <= 0]

    return BagSummary(
        scene_id=scene_dir.name,
        duration_sec=duration_ns / 1_000_000_000.0,
        message_count=int(info["message_count"]),
        storage_identifier=str(info["storage_identifier"]),
        bag_size_bytes=_bag_size_bytes(scene_dir),
        topic_counts=topic_counts,
        missing_required_topics=missing,
        note=SCENE_NOTES.get(scene_dir.name, ""),
    )


def discover_bag_summaries(data_dir: Path) -> list[BagSummary]:
    scene_dirs = sorted(path.parent for path in data_dir.glob("*/metadata.yaml"))
    return [parse_bag_metadata(scene_dir) for scene_dir in scene_dirs]


def image_metrics_from_msg(msg: Any) -> dict[str, float]:
    encoding = str(msg.encoding)
    height = int(msg.height)
    width = int(msg.width)

    if encoding == "16UC1":
        image = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width)
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
        else:
            metrics.update(
                {
                    "mean_valid_depth_m": math.nan,
                    "min_valid_depth_m": math.nan,
                    "max_valid_depth_m": math.nan,
                }
            )
        return metrics

    if encoding in {"mono8", "rgb8", "bgr8"}:
        channels = 1 if encoding == "mono8" else 3
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, channels)
        return {
            "mean_intensity": float(image.mean()),
            "nonzero_ratio": float((image > 0).mean()),
            "saturated_ratio": float((image >= 250).mean()),
        }

    return {}


def _mean_metric(records: list[dict[str, float]], key: str) -> float:
    values = [record[key] for record in records if key in record and not math.isnan(record[key])]
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def sample_image_stats(scene_dir: Path, max_frames_per_topic: int) -> dict[str, dict[str, float]]:
    if max_frames_per_topic <= 0:
        return {}

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

    topic_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    image_type = get_message("sensor_msgs/msg/Image")
    records: dict[str, list[dict[str, float]]] = {topic: [] for topic in IMAGE_TOPICS}

    while reader.has_next():
        topic, data, _timestamp = reader.read_next()
        if topic not in records or len(records[topic]) >= max_frames_per_topic:
            continue
        if topic_types.get(topic) != "sensor_msgs/msg/Image":
            continue
        msg = deserialize_message(data, image_type)
        metrics = image_metrics_from_msg(msg)
        if metrics:
            records[topic].append(metrics)
        if all(len(records[topic]) >= max_frames_per_topic for topic in records):
            break

    stats: dict[str, dict[str, float]] = {}
    for topic, topic_records in records.items():
        if not topic_records:
            continue
        keys = sorted({key for record in topic_records for key in record})
        stats[topic] = {"frames_sampled": float(len(topic_records))}
        for key in keys:
            stats[topic][key] = _mean_metric(topic_records, key)
    return stats


def sample_all_image_stats(
    summaries: list[BagSummary],
    data_dir: Path,
    max_frames_per_topic: int,
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        summary.scene_id: sample_image_stats(data_dir / summary.scene_id, max_frames_per_topic)
        for summary in summaries
    }


def _summary_to_json(summary: BagSummary) -> dict[str, Any]:
    record = asdict(summary)
    record["required_topics_complete"] = summary.is_complete
    return record


def write_reports(
    summaries: list[BagSummary],
    image_stats_by_scene: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "m5_real_d435_manifest_v1",
        "generated_at_utc": _utc_now(),
        "num_bags": len(summaries),
        "required_topics": list(REQUIRED_TOPICS),
        "bags": [_summary_to_json(summary) for summary in summaries],
        "image_stats": image_stats_by_scene,
    }
    (output_dir / "m5_real_d435_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "m5_real_d435_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "scene_id",
            "duration_sec",
            "message_count",
            "bag_size_bytes",
            "storage_identifier",
            "required_topics_complete",
            "missing_required_topics",
            "note",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "scene_id": summary.scene_id,
                    "duration_sec": f"{summary.duration_sec:.3f}",
                    "message_count": summary.message_count,
                    "bag_size_bytes": summary.bag_size_bytes,
                    "storage_identifier": summary.storage_identifier,
                    "required_topics_complete": summary.is_complete,
                    "missing_required_topics": ";".join(summary.missing_required_topics),
                    "note": summary.note,
                }
            )

    with (output_dir / "m5_real_d435_frame_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "scene_id",
            "topic",
            "frames_sampled",
            "valid_ratio",
            "mean_valid_depth_m",
            "min_valid_depth_m",
            "max_valid_depth_m",
            "mean_intensity",
            "nonzero_ratio",
            "saturated_ratio",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for scene_id, topic_stats in sorted(image_stats_by_scene.items()):
            for topic, stats in sorted(topic_stats.items()):
                row = {"scene_id": scene_id, "topic": topic}
                row.update({key: _format_float(value) for key, value in stats.items()})
                writer.writerow(row)

    (output_dir / "m5_real_d435_summary.md").write_text(
        _build_markdown_summary(summaries, image_stats_by_scene),
        encoding="utf-8",
    )


def _format_float(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return value


def _build_markdown_summary(
    summaries: list[BagSummary],
    image_stats_by_scene: dict[str, dict[str, dict[str, float]]],
) -> str:
    complete_count = sum(1 for summary in summaries if summary.is_complete)
    total_size_gib = sum(summary.bag_size_bytes for summary in summaries) / (1024**3)
    lines = [
        "# M5 Real D435 Collection Summary",
        "",
        f"- Bags: {len(summaries)}",
        f"- Required topics complete: {complete_count}/{len(summaries)}",
        f"- Total MCAP size: {total_size_gib:.2f} GiB",
        "",
        "## Bag Manifest",
        "",
        "| scene_id | duration_s | messages | required topics complete | note |",
        "|---|---:|---:|---|---|",
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"{summary.scene_id} | "
            f"{summary.duration_sec:.2f} | "
            f"{summary.message_count} | "
            f"{'yes' if summary.is_complete else 'no'} | "
            f"{summary.note} |"
        )

    lines.extend(["", "## Sampled Image Statistics", ""])
    for summary in summaries:
        scene_stats = image_stats_by_scene.get(summary.scene_id, {})
        if not scene_stats:
            continue
        lines.append(f"### {summary.scene_id}")
        lines.append("")
        lines.append("| topic | frames | primary metric |")
        lines.append("|---|---:|---|")
        for topic, stats in sorted(scene_stats.items()):
            primary = _primary_metric(stats)
            lines.append(
                f"| {topic} | {int(stats.get('frames_sampled', 0))} | {primary} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Field Notes",
            "",
            "- Jelly cup pose produced both severe hole/shadow cases and a visible-point counterpart.",
            "- Multi-object scene observed failure severity: jelly cup > frosted plastic bowl > spoon.",
            "- Near-dark scene preserved IR/depth while RGB was unusable.",
            "",
        ]
    )
    return "\n".join(lines)


def _primary_metric(stats: dict[str, float]) -> str:
    if "valid_ratio" in stats:
        depth = stats.get("mean_valid_depth_m", math.nan)
        depth_text = "" if math.isnan(depth) else f", mean_depth={depth:.3f}m"
        return f"valid_ratio={stats['valid_ratio']:.3f}{depth_text}"
    if "mean_intensity" in stats:
        return f"mean_intensity={stats['mean_intensity']:.1f}, nonzero={stats.get('nonzero_ratio', math.nan):.3f}"
    return "n/a"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/real_d435_m5"),
        help="Directory containing one rosbag2 directory per scene.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for M5 manifest and statistics reports.",
    )
    parser.add_argument(
        "--max-frames-per-topic",
        type=int,
        default=12,
        help="Number of image frames to sample per image topic in each bag.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summaries = discover_bag_summaries(args.data_dir)
    image_stats = sample_all_image_stats(
        summaries,
        args.data_dir,
        max_frames_per_topic=args.max_frames_per_topic,
    )
    write_reports(summaries, image_stats, args.output_dir)
    print(f"Wrote M5 D435 reports for {len(summaries)} bags to {args.output_dir}")


if __name__ == "__main__":
    main()
