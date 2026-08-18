#!/usr/bin/env python3
"""Generate the M7.1 real-motion safety preflight report.

This script never commands robot motion. It only checks that M6 shadow evidence
is frozen and records the operator-controlled gates required before a separate
M7.1 low-amplitude empty-motion test may be run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_OPERATOR_PHRASE = (
    "确认进入 M7.1，允许真实低幅度空载运动；机械臂周围已清空，我已准备好断电/急停。"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate(gate_id: str, status: str, detail: str) -> dict[str, str]:
    return {"gate_id": gate_id, "status": status, "detail": detail}


def generate_m7_safety_preflight(
    *,
    m6_freeze_path: Path,
    output_dir: Path,
    operator_authorized: bool = False,
    required_operator_phrase: str = REQUIRED_OPERATOR_PHRASE,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    freeze = _read_json(m6_freeze_path)
    m6_ok = (
        freeze.get("overall_status") == "pass"
        and freeze.get("motion_authorized") is False
        and int(freeze.get("scene_count", 0)) >= 2
        and int(freeze.get("backend_counts", {}).get("normal_rgbd", 0)) >= 1
        and int(freeze.get("backend_counts", {}).get("ghost_mgg", 0)) >= 1
    )
    limits = {
        "max_joint_delta_deg": 2.0,
        "max_speed": 5,
        "allowed_joint_scope": "single distal wrist joint only",
        "gripper_motion_allowed": False,
        "object_contact_allowed": False,
        "moveit_execution_allowed": False,
    }
    gates = [
        _gate(
            "m6_shadow_freeze",
            "pass" if m6_ok else "fail",
            "M6 includes normal_rgbd and ghost_mgg real shadow evidence, with no motion authorization.",
        ),
        _gate(
            "safety_limits_defined",
            "pass",
            "M7.1 is limited to empty low-amplitude motion, no grasping, no object contact.",
        ),
        _gate(
            "operator_authorization",
            "pass" if operator_authorized else "blocked",
            "Operator must explicitly confirm the required phrase immediately before any real motion.",
        ),
    ]
    blockers = []
    if not m6_ok:
        blockers.append("m6_shadow_freeze_not_passed")
    if not operator_authorized:
        blockers.append("operator_authorization_missing")
    overall = "blocked" if blockers else "ready_for_operator_controlled_m7_1"
    report = {
        "schema_version": "m7_safety_preflight_v1",
        "generated_at_utc": _utc_now(),
        "overall_status": overall,
        "motion_authorized": False,
        "m6_freeze_path": str(m6_freeze_path),
        "required_operator_phrase": required_operator_phrase,
        "limits": limits,
        "gates": gates,
        "blockers": blockers,
        "next_steps": [
            "Do not run real motion from this report.",
            "Before M7.1, reconnect D435 and myCobot, rerun M6 shadow readiness if anything moved.",
            "Run a separate low-amplitude motion script only after the operator confirms the required phrase.",
            "Stop after M7.1; do not grasp objects until M7.2 is explicitly opened.",
        ],
    }
    _write_json(output_dir / "m7_safety_preflight.json", report)
    _write_index(output_dir / "index.md", report)
    return report


def _write_index(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# M7 Safety Preflight",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- motion_authorized: `{str(report['motion_authorized']).lower()}`",
        f"- required_operator_phrase: `{report['required_operator_phrase']}`",
        "",
        "## Limits",
        "",
    ]
    for key, value in report["limits"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Gates", "", "| gate | status | detail |", "|---|---|---|"])
    for gate in report["gates"]:
        lines.append(f"| {gate['gate_id']} | {gate['status']} | {gate['detail']} |")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- {blocker}")
    else:
        lines.append("- None for preflight, but this report still does not command motion.")
    lines.extend(["", "## Next Steps", ""])
    for step in report["next_steps"]:
        lines.append(f"- {step}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m6-freeze",
        type=Path,
        default=Path("reports/m6_shadow_freeze/m6_shadow_freeze.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m7_safety_preflight"))
    parser.add_argument("--operator-authorized", action="store_true")
    parser.add_argument("--required-operator-phrase", default=REQUIRED_OPERATOR_PHRASE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = generate_m7_safety_preflight(
        m6_freeze_path=args.m6_freeze,
        output_dir=args.output_dir,
        operator_authorized=args.operator_authorized,
        required_operator_phrase=args.required_operator_phrase,
    )
    print(f"M7 safety preflight: {report['overall_status']} -> {args.output_dir}")


if __name__ == "__main__":
    main()
