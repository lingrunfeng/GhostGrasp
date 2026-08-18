#!/usr/bin/env python3
"""Generate conservative ObservationQuality from a D435 replay sample."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_m5_5_backend_selector import select_backend  # noqa: E402


COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
ALIGNED_DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
INFRA1_TOPIC = "/camera/camera/infra1/image_rect_raw"
INFRA2_TOPIC = "/camera/camera/infra2/image_rect_raw"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _frame(metadata: Mapping[str, Any], topic: str) -> Mapping[str, Any]:
    return metadata.get("frames", {}).get(topic, {})


def _float(value: Any, default: float = 0.0) -> float:
    return float(default if value is None else value)


def _image_ok(frame: Mapping[str, Any], *, min_nonzero: float = 0.05) -> bool:
    return bool(frame) and _float(frame.get("nonzero_ratio", 0.0)) >= min_nonzero


def _depth_ok(frame: Mapping[str, Any], *, min_valid: float = 0.30) -> bool:
    return bool(frame) and _float(frame.get("valid_ratio", 0.0)) >= min_valid


def build_observation_quality(
    *,
    observation_id: str,
    sample_metadata: Mapping[str, Any],
    topic_report: Mapping[str, Any] | None = None,
    target_summary: Mapping[str, Any] | None = None,
    planning_requested: bool = False,
) -> dict[str, Any]:
    color = _frame(sample_metadata, COLOR_TOPIC)
    depth = _frame(sample_metadata, DEPTH_TOPIC)
    aligned_depth = _frame(sample_metadata, ALIGNED_DEPTH_TOPIC)
    infra1 = _frame(sample_metadata, INFRA1_TOPIC)
    infra2 = _frame(sample_metadata, INFRA2_TOPIC)

    rgb_ok = _image_ok(color)
    infra1_ok = _image_ok(infra1)
    infra2_ok = _image_ok(infra2)
    ir_ok = infra1_ok and infra2_ok
    depth_ok = _depth_ok(depth)
    aligned_depth_ok = _depth_ok(aligned_depth)
    pointcloud_ok = not topic_report or (
        topic_report.get("overall_status") == "pass"
        and "/camera/camera/depth/color/points"
        not in topic_report.get("missing_topics", [])
    )

    target_pixels = int(_float((target_summary or {}).get("target_pixels", 0), 0.0))
    mask_ok = bool(target_summary) and target_pixels > 0
    target_valid_depth_ratio = _float((target_summary or {}).get("valid_depth_ratio", 0.0))
    target_hole_ratio = _float((target_summary or {}).get("hole_ratio", 0.0))
    target_table_leakage_ratio = _float(
        (target_summary or {}).get("table_leakage_ratio", 0.0)
    )
    quality: dict[str, Any] = {
        "schema_version": "m5_5_observation_quality_v1",
        "observation_id": observation_id,
        "scene_id": sample_metadata.get("scene_id", observation_id),
        "generated_at_utc": _utc_now(),
        "timestamp_sec": time.time(),
        "planning_requested": bool(planning_requested),
        "rgb_ok": rgb_ok,
        "ir_ok": ir_ok,
        "depth_ok": depth_ok,
        "aligned_depth_ok": aligned_depth_ok,
        "pointcloud_ok": pointcloud_ok,
        "mask_ok": mask_ok,
        "table_ok": depth_ok or aligned_depth_ok,
        "tf_ok": bool(topic_report and topic_report.get("overall_status") == "pass"),
        "stale": False,
        "target_valid_depth_ratio": target_valid_depth_ratio,
        "target_hole_ratio": target_hole_ratio,
        "target_table_leakage_ratio": target_table_leakage_ratio,
        "depth_failure_detected": False,
        "stream_metrics": {
            "depth_valid_ratio": depth.get("valid_ratio"),
            "aligned_depth_valid_ratio": aligned_depth.get("valid_ratio"),
            "rgb_mean_intensity": color.get("mean_intensity"),
            "infra1_mean_intensity": infra1.get("mean_intensity"),
            "infra2_mean_intensity": infra2.get("mean_intensity"),
        },
    }
    selection = select_backend(quality, planning_requested=planning_requested)
    quality["recommended_backend"] = selection["recommended_backend"]
    quality["reject_reasons"] = selection["reject_reasons"]
    quality["depth_failure_reasons"] = selection["depth_failure_reasons"]
    quality["depth_failure_detected"] = bool(selection["depth_failure_reasons"])
    quality["backend_thresholds"] = selection["thresholds"]
    return quality


def run_quality_generation(
    *,
    sample_metadata_path: Path,
    output_path: Path,
    topic_report_path: Path | None = None,
    target_summary_path: Path | None = None,
    observation_id: str | None = None,
    planning_requested: bool = False,
) -> dict[str, Any]:
    sample_metadata = json.loads(sample_metadata_path.read_text())
    topic_report = (
        json.loads(topic_report_path.read_text())
        if topic_report_path and topic_report_path.exists()
        else None
    )
    target_summary = (
        json.loads(target_summary_path.read_text())
        if target_summary_path and target_summary_path.exists()
        else None
    )
    quality = build_observation_quality(
        observation_id=observation_id or str(sample_metadata.get("scene_id", "observation")),
        sample_metadata=sample_metadata,
        topic_report=topic_report,
        target_summary=target_summary,
        planning_requested=planning_requested,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
    return quality


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-metadata", type=Path, required=True)
    parser.add_argument("--topic-report", type=Path)
    parser.add_argument("--target-summary", type=Path)
    parser.add_argument("--observation-id", type=str)
    parser.add_argument("--planning-requested", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/m5_5_real_online_bridge/observation_quality.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    quality = run_quality_generation(
        sample_metadata_path=args.sample_metadata,
        topic_report_path=args.topic_report,
        target_summary_path=args.target_summary,
        output_path=args.output,
        observation_id=args.observation_id,
        planning_requested=args.planning_requested,
    )
    print(f"M5.5 quality: {quality['recommended_backend']} -> {args.output}")


if __name__ == "__main__":
    main()
