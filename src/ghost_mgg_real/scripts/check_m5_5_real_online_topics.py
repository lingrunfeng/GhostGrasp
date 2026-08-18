#!/usr/bin/env python3
"""Check the live D435 topic contract required before M5.5/M6 shadow mode."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


REQUIRED_D435_TOPICS: dict[str, str] = {
    "/camera/camera/color/image_raw": "sensor_msgs/msg/Image",
    "/camera/camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/camera/camera/depth/image_rect_raw": "sensor_msgs/msg/Image",
    "/camera/camera/depth/camera_info": "sensor_msgs/msg/CameraInfo",
    "/camera/camera/aligned_depth_to_color/image_raw": "sensor_msgs/msg/Image",
    "/camera/camera/aligned_depth_to_color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/camera/camera/infra1/image_rect_raw": "sensor_msgs/msg/Image",
    "/camera/camera/infra1/camera_info": "sensor_msgs/msg/CameraInfo",
    "/camera/camera/infra2/image_rect_raw": "sensor_msgs/msg/Image",
    "/camera/camera/infra2/camera_info": "sensor_msgs/msg/CameraInfo",
    "/camera/camera/depth/color/points": "sensor_msgs/msg/PointCloud2",
    "/tf_static": "tf2_msgs/msg/TFMessage",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def required_d435_topics() -> dict[str, str]:
    return dict(REQUIRED_D435_TOPICS)


def build_topic_check_report(
    observed_topics: Mapping[str, str],
    now_sec: float,
) -> dict:
    required = required_d435_topics()
    missing_topics = sorted(topic for topic in required if topic not in observed_topics)
    present_topics = sorted(topic for topic in required if topic in observed_topics)
    type_mismatches = {
        topic: {"expected": expected_type, "observed": observed_topics[topic]}
        for topic, expected_type in required.items()
        if topic in observed_topics and observed_topics[topic] != expected_type
    }

    overall_status = "pass" if not missing_topics and not type_mismatches else "fail"
    return {
        "schema_version": "m5_5_real_online_topic_check_v1",
        "generated_at_utc": _utc_now(),
        "timestamp_sec": float(now_sec),
        "overall_status": overall_status,
        "required_topics": required,
        "present_topics": present_topics,
        "missing_topics": missing_topics,
        "type_mismatches": type_mismatches,
    }


def read_ros2_topic_types() -> dict[str, str]:
    completed = subprocess.run(
        ["ros2", "topic", "list", "--show-types"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    observed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped or " [" not in stripped or not stripped.endswith("]"):
            continue
        topic, type_part = stripped.rsplit(" [", 1)
        observed[topic] = type_part[:-1]
    return observed


def run_topic_check(output_path: Path) -> dict:
    report = build_topic_check_report(
        observed_topics=read_ros2_topic_types(),
        now_sec=time.time(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/m5_5_real_online_bridge/topic_check.json"),
        help="JSON output path for the topic check report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_topic_check(args.output)
    print(f"M5.5 topic check: {report['overall_status']} -> {args.output}")
    if report["overall_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
