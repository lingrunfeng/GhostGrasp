#!/usr/bin/env python3
"""Generate synthetic M7 MoveIt targets using an official-style top grasp.

This script writes MoveIt-compatible target rows only. It does not start ROS,
connect to myCobot, or command hardware motion.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SEED_TARGET_SPECS = [
    "seed_green_center:0.005:0.236:cylinder:0.020:0.025:0.0",
    "seed_green_left:-0.030:0.225:cylinder:0.018:0.025:0.0",
    "seed_box_right:0.040:0.245:box:0.035:0.025:0.035:0.35",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _round(value: float) -> float:
    return round(float(value), 6)


def official_top_grasp_template() -> dict[str, Any]:
    return {
        "template_id": "elephant_official_top_grasp_v1",
        "source": "Elephant Robotics myCobot gripper demo pattern",
        "scope": "template_only_not_a_real_target",
        "approach_axis_base": [0.0, 0.0, -1.0],
        "pregrasp_clearance_m": 0.07,
        "retreat_lift_m": 0.07,
        "moveit_execution_allowed": False,
        "hardware_motion_allowed": False,
        "official_reference_pose_mm_deg": [-177.5, 1.91, 173.49],
        "notes": [
            "Official examples use a taught top-grasp-like pose and a z+70 mm pregrasp.",
            "The raw rx/ry/rz values are retained as calibration evidence, not injected directly into MoveIt.",
            "MoveIt receives a gripper_tcp top-down pose constraint through probe_m4_sim_grasp_moveit.py.",
        ],
    }


def _parse_float(value: str, *, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _validate_target_common(target_id: str, x_m: float, y_m: float, yaw_rad: float) -> None:
    if not target_id:
        raise ValueError("target_id must be non-empty")
    for field_name, value in (("x_m", x_m), ("y_m", y_m), ("yaw_rad", yaw_rad)):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")


def parse_seed_target_spec(
    spec: str,
    *,
    table_top_z_m: float,
    pregrasp_clearance_m: float,
) -> dict[str, Any]:
    """Parse one target spec.

    Formats:
      cylinder: id:x_m:y_m:cylinder:radius_m:height_m:yaw_rad
      box:      id:x_m:y_m:box:size_x_m:size_y_m:size_z_m:yaw_rad
    """

    parts = [part.strip() for part in str(spec).split(":")]
    if len(parts) < 7:
        raise ValueError(f"target spec has too few fields: {spec}")
    target_id = parts[0]
    x_m = _parse_float(parts[1], field_name="x_m")
    y_m = _parse_float(parts[2], field_name="y_m")
    shape_type = parts[3]
    _validate_target_common(target_id, x_m, y_m, 0.0)

    common = {
        "target_id": target_id,
        "center_x_m": _round(x_m),
        "center_y_m": _round(y_m),
        "pregrasp_clearance_m": _round(pregrasp_clearance_m),
        "grasp_type": "top_grasp",
        "valid": True,
        "failure_reason": "",
        "synthetic_source": "manual_seed_not_real_perception",
    }

    if shape_type == "cylinder":
        if len(parts) != 7:
            raise ValueError(f"cylinder spec must have 7 fields: {spec}")
        radius_m = _parse_float(parts[4], field_name="radius_m")
        height_m = _parse_float(parts[5], field_name="height_m")
        yaw_rad = _parse_float(parts[6], field_name="yaw_rad")
        if radius_m <= 0.0 or height_m <= 0.0:
            raise ValueError("cylinder radius and height must be positive")
        _validate_target_common(target_id, x_m, y_m, yaw_rad)
        return {
            **common,
            "shape_type": "cylinder",
            "center_z_m": _round(float(table_top_z_m) + 0.5 * height_m),
            "yaw_rad": _round(yaw_rad),
            "radius_m": _round(radius_m),
            "height_m": _round(height_m),
            "required_gripper_width_m": _round(2.0 * radius_m),
        }

    if shape_type == "box":
        if len(parts) != 8:
            raise ValueError(f"box spec must have 8 fields: {spec}")
        size_x_m = _parse_float(parts[4], field_name="size_x_m")
        size_y_m = _parse_float(parts[5], field_name="size_y_m")
        size_z_m = _parse_float(parts[6], field_name="size_z_m")
        yaw_rad = _parse_float(parts[7], field_name="yaw_rad")
        if size_x_m <= 0.0 or size_y_m <= 0.0 or size_z_m <= 0.0:
            raise ValueError("box dimensions must be positive")
        _validate_target_common(target_id, x_m, y_m, yaw_rad)
        return {
            **common,
            "shape_type": "box",
            "center_z_m": _round(float(table_top_z_m) + 0.5 * size_z_m),
            "yaw_rad": _round(yaw_rad),
            "size_x_m": _round(size_x_m),
            "size_y_m": _round(size_y_m),
            "size_z_m": _round(size_z_m),
            "required_gripper_width_m": _round(min(size_x_m, size_y_m)),
        }

    raise ValueError(f"unsupported shape_type in target spec: {shape_type}")


def _render_index(report: dict[str, Any]) -> str:
    lines = [
        "# M7 Official-Style Top-Grasp MoveIt Seeds",
        "",
        f"- schema_version: `{report['schema_version']}`",
        f"- safety_mode: `{report['safety_mode']}`",
        f"- motion_authorized: `{str(report['motion_authorized']).lower()}`",
        f"- table_top_z_m: `{report['table_top_z_m']}`",
        f"- targets_path: `{report['targets_path']}`",
        "",
        "## Template",
        "",
        f"- template_id: `{report['top_grasp_template']['template_id']}`",
        f"- approach_axis_base: `{report['top_grasp_template']['approach_axis_base']}`",
        f"- pregrasp_clearance_m: `{report['top_grasp_template']['pregrasp_clearance_m']}`",
        f"- retreat_lift_m: `{report['top_grasp_template']['retreat_lift_m']}`",
        "",
        "## Targets",
        "",
    ]
    for row in report["targets"]:
        lines.append(
            "- `{target_id}` {shape_type} center=({x}, {y}, {z}) yaw={yaw}".format(
                target_id=row["target_id"],
                shape_type=row["shape_type"],
                x=row["center_x_m"],
                y=row["center_y_m"],
                z=row["center_z_m"],
                yaw=row["yaw_rad"],
            )
        )
    lines.extend(
        [
            "",
            "These targets are synthetic seed coordinates for MoveIt plan-only testing.",
            "They are not real perception outputs and do not authorize hardware motion.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_official_top_grasp_seed_targets(
    *,
    output_dir: Path,
    table_top_z_m: float,
    seed_target_specs: list[str],
) -> dict[str, Any]:
    template = official_top_grasp_template()
    targets = [
        parse_seed_target_spec(
            spec,
            table_top_z_m=table_top_z_m,
            pregrasp_clearance_m=float(template["pregrasp_clearance_m"]),
        )
        for spec in seed_target_specs
    ]
    output_dir = Path(output_dir)
    targets_path = output_dir / "m7_official_top_grasp_seed_targets.json"
    targets_payload = {
        "schema_version": "m4_sim_grasp_targets_v1",
        "rows": targets,
        "source": "m7_official_top_grasp_seed_targets",
        "safety_mode": "moveit_plan_only_no_hardware_motion",
    }
    _write_json(targets_path, targets_payload)
    report = {
        "schema_version": "m7_official_top_grasp_seed_targets_v1",
        "generated_at_utc": _utc_now(),
        "safety_mode": "moveit_plan_only_no_hardware_motion",
        "motion_authorized": False,
        "table_top_z_m": _round(table_top_z_m),
        "targets_path": str(targets_path),
        "top_grasp_template": template,
        "targets": targets,
        "next_steps": [
            "Run MoveIt plan-only against these synthetic targets.",
            "Inspect pregrasp and grasp planning before any hardware execution adapter is enabled.",
            "Replace synthetic x/y with Ghost-MGG perception output only after the template is calibrated.",
        ],
    }
    _write_json(output_dir / "m7_official_top_grasp_seed_report.json", report)
    (output_dir / "index.md").write_text(_render_index(report), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--table-top-z-m", type=float, default=-0.0127)
    parser.add_argument(
        "--seed-target",
        action="append",
        dest="seed_targets",
        default=[],
        help=(
            "Synthetic target spec. Cylinder: id:x:y:cylinder:radius:height:yaw. "
            "Box: id:x:y:box:size_x:size_y:size_z:yaw."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = generate_official_top_grasp_seed_targets(
        output_dir=args.output_dir,
        table_top_z_m=float(args.table_top_z_m),
        seed_target_specs=list(args.seed_targets or DEFAULT_SEED_TARGET_SPECS),
    )
    print(f"M7 official-style top-grasp seeds: {report['targets_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
