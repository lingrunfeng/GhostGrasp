#!/usr/bin/env python3
"""Select the conservative M5.5 backend from an ObservationQuality JSON."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_bool(quality: Mapping[str, Any], key: str, default: bool = False) -> bool:
    return bool(quality.get(key, default))


def _as_float(quality: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = quality.get(key, default)
    return float(default if value is None else value)


def select_backend(
    quality: Mapping[str, Any],
    *,
    planning_requested: bool = False,
) -> dict[str, Any]:
    reject_reasons: list[str] = []
    depth_failure_reasons: list[str] = []
    caution_reasons: list[str] = []

    stale = _as_bool(quality, "stale")
    rgb_ok = _as_bool(quality, "rgb_ok")
    ir_ok = _as_bool(quality, "ir_ok")
    depth_ok = _as_bool(quality, "depth_ok")
    mask_ok = _as_bool(quality, "mask_ok")
    table_ok = _as_bool(quality, "table_ok")
    tf_ok = _as_bool(quality, "tf_ok")
    target_valid_depth_ratio = _as_float(quality, "target_valid_depth_ratio")
    target_hole_ratio = _as_float(quality, "target_hole_ratio")
    target_table_leakage_ratio = _as_float(quality, "target_table_leakage_ratio")
    contact_shadow_depth_usable = (
        rgb_ok
        and depth_ok
        and target_valid_depth_ratio >= 0.75
        and target_hole_ratio < 0.25
        and 0.08 <= target_table_leakage_ratio < 0.20
    )

    if stale:
        reject_reasons.append("stale_observation")
    if not mask_ok:
        reject_reasons.append("target_mask_missing")
    if not table_ok:
        reject_reasons.append("table_unreliable")
    if planning_requested and not tf_ok:
        reject_reasons.append("camera_to_base_tf_missing")

    if reject_reasons:
        backend = "abstain"
    else:
        if not contact_shadow_depth_usable:
            if target_hole_ratio >= 0.25:
                depth_failure_reasons.append("target_hole_ratio_high")
            if target_table_leakage_ratio >= 0.08:
                depth_failure_reasons.append("target_table_leakage_ratio_high")

        if depth_failure_reasons:
            backend = "ghost_mgg"
        elif contact_shadow_depth_usable:
            backend = "normal_rgbd"
            caution_reasons.append("contact_shadow_leakage_but_depth_usable")
        elif (
            rgb_ok
            and depth_ok
            and target_valid_depth_ratio >= 0.65
            and target_hole_ratio < 0.20
            and target_table_leakage_ratio < 0.08
        ):
            backend = "normal_rgbd"
        elif (
            rgb_ok
            and depth_ok
            and target_valid_depth_ratio >= 0.70
            and target_hole_ratio < 0.25
            and target_table_leakage_ratio < 0.05
        ):
            backend = "normal_rgbd"
            caution_reasons.append("borderline_hole_ratio_but_depth_usable")
        elif ir_ok and depth_ok and not rgb_ok:
            backend = "ir_depth"
        else:
            backend = "manual_review"

    return {
        "schema_version": "m5_5_backend_selection_v1",
        "generated_at_utc": _utc_now(),
        "recommended_backend": backend,
        "reject_reasons": reject_reasons,
        "depth_failure_reasons": depth_failure_reasons,
        "caution_reasons": caution_reasons,
        "thresholds": {
            "normal_rgbd_min_target_valid_depth_ratio": 0.65,
            "normal_rgbd_max_target_hole_ratio": 0.20,
            "normal_rgbd_borderline_min_target_valid_depth_ratio": 0.70,
            "normal_rgbd_borderline_max_target_hole_ratio": 0.25,
            "normal_rgbd_borderline_max_target_table_leakage_ratio": 0.05,
            "normal_rgbd_contact_shadow_min_target_valid_depth_ratio": 0.75,
            "normal_rgbd_contact_shadow_max_target_hole_ratio": 0.25,
            "normal_rgbd_contact_shadow_max_target_table_leakage_ratio": 0.20,
            "ghost_mgg_min_target_hole_ratio": 0.25,
            "ghost_mgg_min_target_table_leakage_ratio": 0.08,
        },
    }


def run_selector(quality_path: Path, output_path: Path) -> dict[str, Any]:
    quality = json.loads(quality_path.read_text())
    selection = select_backend(
        quality,
        planning_requested=bool(quality.get("planning_requested", False)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return selection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True, help="ObservationQuality JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/m5_5_real_online_bridge/backend_selection.json"),
        help="Backend selection JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selection = run_selector(args.quality, args.output)
    print(f"M5.5 backend: {selection['recommended_backend']} -> {args.output}")


if __name__ == "__main__":
    main()
