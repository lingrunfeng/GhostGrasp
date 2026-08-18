#!/usr/bin/env python3
"""Generate the M7.2 green-cylinder grasp safety gate report.

This gate is intentionally non-actuating. It checks that the current live
green-cylinder shadow decision and MoveIt plan are plausible for a first real
grasp, and blocks if the implied joint motion is too large.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_OPERATOR_PHRASE = (
    "确认进入 M7.2，允许真实低速抓取绿色圆柱；目标和桌面已清空，"
    "机械臂周围安全，我已准备好断电/急停。"
)
STANDARD_TOP_GRASP_HOME_JOINT_DEG = [0.0, 30.0, -70.0, 0.0, 0.0, 0.0]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_six(values: Any, *, field_name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError(f"{field_name} must contain six numeric values")
    parsed = [float(value) for value in values]
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _pregrasp_target_angles_deg(moveit_plan: dict[str, Any]) -> list[float] | None:
    rows = moveit_plan.get("rows", [])
    if not rows:
        return None
    attempts = rows[0].get("attempts", [])
    if not attempts:
        return None
    positions_rad = attempts[0].get("final_joint_positions")
    if positions_rad is None:
        return None
    return [math.degrees(value) for value in _finite_six(positions_rad, field_name="final_joint_positions")]


def _delta_summary(from_deg: list[float], to_deg: list[float]) -> dict[str, Any]:
    deltas = [to_deg[i] - from_deg[i] for i in range(6)]
    return {
        "from_angles_deg": [round(value, 6) for value in from_deg],
        "to_angles_deg": [round(value, 6) for value in to_deg],
        "delta_deg": [round(value, 6) for value in deltas],
        "max_abs_delta_deg": round(max(abs(value) for value in deltas), 6),
    }


def _parse_angles_csv(value: str) -> list[float]:
    return _finite_six([float(part.strip()) for part in value.split(",")], field_name="angles_csv")


def generate_gate(
    *,
    decision_path: Path,
    target_path: Path,
    moveit_plan_path: Path,
    current_state_path: Path,
    output_dir: Path,
    execute_requested: bool,
    operator_phrase: str,
    max_joint_delta_deg: float = 60.0,
    max_home_to_pregrasp_delta_deg: float = 70.0,
    standard_home_angles_deg: list[float] | None = None,
) -> dict[str, Any]:
    decision = _read_json(decision_path)
    target_report = _read_json(target_path)
    moveit_plan = _read_json(moveit_plan_path)
    current_state = _read_json(current_state_path)

    blockers: list[str] = []
    warnings: list[str] = []

    if execute_requested and operator_phrase != REQUIRED_OPERATOR_PHRASE:
        blockers.append("operator_phrase_mismatch")
    if decision.get("recommended_backend") != "normal_rgbd":
        blockers.append("backend_not_normal_rgbd")
    if decision.get("target_label") != "green_cylinder":
        blockers.append("target_label_not_green_cylinder")
    if not decision.get("ready_for_shadow_planning"):
        blockers.append("shadow_decision_not_ready")

    target = target_report.get("target", {})
    if target.get("shape_type") != "cylinder":
        blockers.append("target_shape_not_cylinder")
    if not target.get("valid"):
        blockers.append("target_invalid")

    summary = moveit_plan.get("summary", {})
    if summary.get("all_planned") is not True:
        blockers.append("moveit_plan_not_all_planned")
    rows = moveit_plan.get("rows", [])
    if not rows or rows[0].get("planned") is not True:
        blockers.append("top_row_not_planned")
    elif rows[0].get("descent_clearance", {}).get("status") != "ok":
        blockers.append("descent_clearance_not_ok")

    current_angles = _finite_six(current_state.get("angles_deg"), field_name="angles_deg")
    standard_home_angles = _finite_six(
        STANDARD_TOP_GRASP_HOME_JOINT_DEG
        if standard_home_angles_deg is None
        else standard_home_angles_deg,
        field_name="standard_home_angles_deg",
    )
    target_angles = _pregrasp_target_angles_deg(moveit_plan)
    joint_delta: dict[str, Any] = {"available": False}
    if target_angles is None:
        blockers.append("pregrasp_joint_target_missing")
    else:
        current_to_pregrasp = _delta_summary(current_angles, target_angles)
        current_to_home = _delta_summary(current_angles, standard_home_angles)
        home_to_pregrasp = _delta_summary(standard_home_angles, target_angles)
        joint_delta = {
            "available": True,
            "current_angles_deg": [round(value, 6) for value in current_angles],
            "standard_home_angles_deg": [round(value, 6) for value in standard_home_angles],
            "pregrasp_target_angles_deg": [round(value, 6) for value in target_angles],
            "delta_deg": current_to_pregrasp["delta_deg"],
            "max_abs_delta_deg": current_to_pregrasp["max_abs_delta_deg"],
            "current_to_pregrasp": {
                **current_to_pregrasp,
                "limit_deg": float(max_joint_delta_deg),
            },
            "current_to_home": {
                **current_to_home,
                "limit_deg": float(max_joint_delta_deg),
            },
            "home_to_pregrasp": {
                **home_to_pregrasp,
                "limit_deg": float(max_home_to_pregrasp_delta_deg),
            },
        }
        if current_to_pregrasp["max_abs_delta_deg"] > float(max_joint_delta_deg):
            blockers.append("pregrasp_joint_delta_too_large")
        if current_to_home["max_abs_delta_deg"] > float(max_joint_delta_deg):
            blockers.append("current_to_home_joint_delta_too_large")
        if home_to_pregrasp["max_abs_delta_deg"] > float(max_home_to_pregrasp_delta_deg):
            blockers.append("home_to_pregrasp_joint_delta_too_large")
        if any(
            blocker in blockers
            for blocker in (
                "pregrasp_joint_delta_too_large",
                "current_to_home_joint_delta_too_large",
                "home_to_pregrasp_joint_delta_too_large",
            )
        ):
            warnings.append(
                "The current first-grasp plan implies a large raw joint move. "
                "Do not execute until joint mapping, start posture, and target frame are rechecked."
            )

    if blockers:
        overall_status = "blocked"
    else:
        overall_status = "ready_for_target_based_top_grasp_execute"

    report = {
        "schema_version": "m7_green_cylinder_grasp_gate_v2",
        "generated_at_utc": _utc_now(),
        "stage": "M7.2e",
        "overall_status": overall_status,
        "motion_authorized": False,
        "execute_requested": bool(execute_requested),
        "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
        "standard_top_grasp_home_joint_deg": [round(value, 6) for value in standard_home_angles],
        "decision_path": str(decision_path),
        "target_path": str(target_path),
        "moveit_plan_path": str(moveit_plan_path),
        "current_state_path": str(current_state_path),
        "target": target,
        "recommended_backend": decision.get("recommended_backend"),
        "depth_failure_reasons": list(decision.get("depth_failure_reasons", [])),
        "caution_reasons": list(decision.get("caution_reasons", [])),
        "joint_delta": joint_delta,
        "max_joint_delta_deg": float(max_joint_delta_deg),
        "max_home_to_pregrasp_delta_deg": float(max_home_to_pregrasp_delta_deg),
        "blockers": blockers,
        "warnings": warnings,
        "next_steps": [
            "If blocked by joint delta, do not send the MoveIt pregrasp angles to hardware.",
            "Recheck raw myCobot angle to MoveIt joint mapping and start posture before M7.2 execution.",
            "Only after this gate is clear should a separate real execute adapter send motion.",
        ],
    }
    output_dir = Path(output_dir)
    _write_json(output_dir / "m7_green_cylinder_grasp_gate.json", report)
    _write_index(output_dir / "index.md", report)
    return report


def _write_index(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# M7 Green Cylinder Grasp Gate",
        "",
        f"- overall_status: `{report['overall_status']}`",
        f"- motion_authorized: `{str(report['motion_authorized']).lower()}`",
        f"- execute_requested: `{str(report['execute_requested']).lower()}`",
        f"- recommended_backend: `{report['recommended_backend']}`",
        f"- standard_top_grasp_home_joint_deg: `{report['standard_top_grasp_home_joint_deg']}`",
        "",
        "## Target",
        "",
        f"- shape_type: `{report['target'].get('shape_type')}`",
        f"- center_xyz_m: `{report['target'].get('center_x_m')}, {report['target'].get('center_y_m')}, {report['target'].get('center_z_m')}`",
        f"- required_gripper_width_m: `{report['target'].get('required_gripper_width_m')}`",
        "",
        "## Joint Delta",
        "",
    ]
    delta = report["joint_delta"]
    if delta.get("available"):
        lines.extend(
            [
                f"- max_abs_delta_deg: `{delta['max_abs_delta_deg']}`",
                f"- limit_deg: `{report['max_joint_delta_deg']}`",
                f"- current_angles_deg: `{delta['current_angles_deg']}`",
                f"- standard_home_angles_deg: `{delta['standard_home_angles_deg']}`",
                f"- pregrasp_target_angles_deg: `{delta['pregrasp_target_angles_deg']}`",
                f"- delta_deg: `{delta['delta_deg']}`",
                f"- current_to_home_max_abs_delta_deg: `{delta['current_to_home']['max_abs_delta_deg']}`",
                f"- home_to_pregrasp_max_abs_delta_deg: `{delta['home_to_pregrasp']['max_abs_delta_deg']}`",
            ]
        )
    else:
        lines.append("- unavailable")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "This gate does not send real motion commands.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--moveit-plan", type=Path, required=True)
    parser.add_argument("--current-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m7_green_cylinder_grasp_gate"))
    parser.add_argument("--execute-requested", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--max-joint-delta-deg", type=float, default=60.0)
    parser.add_argument("--max-home-to-pregrasp-delta-deg", type=float, default=70.0)
    parser.add_argument(
        "--standard-home-angles-deg",
        default=",".join(str(value) for value in STANDARD_TOP_GRASP_HOME_JOINT_DEG),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = generate_gate(
        decision_path=args.decision,
        target_path=args.target,
        moveit_plan_path=args.moveit_plan,
        current_state_path=args.current_state,
        output_dir=args.output_dir,
        execute_requested=args.execute_requested,
        operator_phrase=args.operator_phrase,
        max_joint_delta_deg=args.max_joint_delta_deg,
        max_home_to_pregrasp_delta_deg=args.max_home_to_pregrasp_delta_deg,
        standard_home_angles_deg=_parse_angles_csv(args.standard_home_angles_deg),
    )
    print(f"M7 green cylinder grasp gate: {report['overall_status']} -> {args.output_dir}")


if __name__ == "__main__":
    main()
