#!/usr/bin/env python3
"""Generate the M6 shadow-mode readiness report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate(gate_id: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {
        "gate_id": gate_id,
        "status": status,
        "evidence": evidence,
        "detail": detail,
    }


def _live_backend_switch_gate(live_report: dict[str, Any]) -> dict[str, str]:
    counts = live_report.get("backend_counts", {})
    ghost_count = int(counts.get("ghost_mgg", 0))
    normal_count = int(counts.get("normal_rgbd", 0))
    status = "pass" if ghost_count >= 1 and normal_count >= 1 else "fail"
    return _gate(
        "m5_5_live_backend_switch",
        status,
        "reports/m5_5_live_smoke_report/live_smoke_report.json",
        f"ghost_mgg={ghost_count}, normal_rgbd={normal_count}",
    )


def _topic_contract_gate(topic_check: dict[str, Any]) -> dict[str, str]:
    status = "pass" if topic_check.get("overall_status") == "pass" else "fail"
    missing = topic_check.get("missing_topics", [])
    return _gate(
        "d435_topic_contract",
        status,
        "reports/m5_5_real_online_bridge/topic_check.json",
        "all required D435 topics present" if not missing else f"missing={missing}",
    )


def generate_m6_shadow_readiness(
    *,
    live_smoke_report_path: Path,
    topic_check_path: Path,
    output_dir: Path,
    real_tf_checked: bool = False,
    moveit_shadow_checked: bool = False,
    mycobot_state_bridge_checked: bool = False,
    real_state_moveit_shadow_checked: bool = False,
    real_tf_evidence_path: str = "reports/m6_camera_to_base_tf_check/tf_check.json",
    moveit_shadow_evidence_path: str = "reports/m6_moveit_shadow_plan_only/planning_probe.log",
    real_state_moveit_shadow_evidence_path: str = "reports/m6_real_state_moveit_shadow_plan_only_shadow_gripper/planning_status.txt",
) -> dict[str, Any]:
    live_report = _read_json(live_smoke_report_path)
    topic_check = _read_json(topic_check_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf_gate = _gate(
        "real_camera_to_base_tf",
        "pass" if real_tf_checked else "blocked",
        (
            real_tf_evidence_path
            if real_tf_checked
            else "pending M6 live TF check"
        ),
        (
            "Verified camera optical frame to robot base TF."
            if real_tf_checked
            else "Need verified camera optical frame to robot base TF before planning."
        ),
    )
    moveit_gate = _gate(
        "moveit_shadow_planning",
        "pass" if moveit_shadow_checked else "blocked",
        (
            moveit_shadow_evidence_path
            if moveit_shadow_checked
            else "pending M6 MoveIt dry-run"
        ),
        (
            "Plan-only MoveIt request succeeded with execution disabled."
            if moveit_shadow_checked
            else "Need plan-only MoveIt check with real observation; no robot motion."
        ),
    )
    mycobot_state_gate = _gate(
        "mycobot_state_bridge",
        "pass" if mycobot_state_bridge_checked else "blocked",
        (
            "reports/m6_mycobot_state_bridge_smoke_shadow_gripper/joint_state_once.yaml"
            if mycobot_state_bridge_checked
            else "pending M6 myCobot state bridge smoke"
        ),
        (
            "Real myCobot angles are published locally as complete /joint_states."
            if mycobot_state_bridge_checked
            else "Need read-only real myCobot joint-state bridge check."
        ),
    )
    real_state_moveit_gate = _gate(
        "real_state_moveit_shadow",
        "pass" if real_state_moveit_shadow_checked else "blocked",
        (
            real_state_moveit_shadow_evidence_path
            if real_state_moveit_shadow_checked
            else "pending M6 real-state MoveIt shadow check"
        ),
        (
            "MoveIt consumed real /joint_states with execution disabled; planning may pass or safely block on current-state collision."
            if real_state_moveit_shadow_checked
            else "Need MoveIt shadow check using real myCobot state and no execution."
        ),
    )

    gates = [
        _live_backend_switch_gate(live_report),
        _topic_contract_gate(topic_check),
        mycobot_state_gate,
        real_state_moveit_gate,
        tf_gate,
        moveit_gate,
    ]
    if any(gate["status"] == "fail" for gate in gates):
        overall = "fail"
    elif any(gate["status"] == "blocked" for gate in gates):
        overall = "blocked"
    else:
        overall = "pass"

    next_steps = []
    if _live_backend_switch_gate(live_report)["status"] != "pass":
        next_steps.append("Start D435 stable 640x480x30 inspect launch.")
        next_steps.append("Run M5.5 real online smoke.")
    if _topic_contract_gate(topic_check)["status"] != "pass":
        next_steps.append("Verify D435 topic contract.")
    if not real_tf_checked:
        next_steps.append("Verify camera-to-base TF.")
    if not moveit_shadow_checked:
        next_steps.append("Run MoveIt shadow planning with execution disabled.")
    if not mycobot_state_bridge_checked:
        next_steps.append("Run myCobot state bridge smoke.")
    if not real_state_moveit_shadow_checked:
        next_steps.append("Run MoveIt shadow with real myCobot joint states.")

    report = {
        "schema_version": "m6_shadow_readiness_v1",
        "generated_at_utc": _utc_now(),
        "overall_status": overall,
        "live_smoke_report_path": str(live_smoke_report_path),
        "topic_check_path": str(topic_check_path),
        "gates": gates,
        "next_required_live_steps": next_steps,
    }
    _write_json(output_dir / "m6_shadow_readiness.json", report)
    _write_index(output_dir / "index.md", report)
    return report


def _write_index(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# M6 Shadow Readiness",
        "",
        f"- Overall status: {report['overall_status']}",
        "",
        "| gate | status | detail |",
        "|---|---|---|",
    ]
    for gate in report["gates"]:
        lines.append(
            "| "
            f"{gate['gate_id']} | "
            f"{gate['status']} | "
            f"{gate['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Next Live Steps",
            "",
        ]
    )
    for step in report["next_required_live_steps"]:
        lines.append(f"- {step}")
    if not report["next_required_live_steps"]:
        lines.append("- None; ready for next M6 gate.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-smoke-report",
        type=Path,
        default=Path("reports/m5_5_live_smoke_report/live_smoke_report.json"),
    )
    parser.add_argument(
        "--topic-check",
        type=Path,
        default=Path("reports/m5_5_real_online_bridge/topic_check.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m6_shadow_readiness"),
    )
    parser.add_argument("--real-tf-checked", action="store_true")
    parser.add_argument("--moveit-shadow-checked", action="store_true")
    parser.add_argument("--mycobot-state-bridge-checked", action="store_true")
    parser.add_argument("--real-state-moveit-shadow-checked", action="store_true")
    parser.add_argument(
        "--real-tf-evidence-path",
        default="reports/m6_camera_to_base_tf_check/tf_check.json",
    )
    parser.add_argument(
        "--moveit-shadow-evidence-path",
        default="reports/m6_moveit_shadow_plan_only/planning_probe.log",
    )
    parser.add_argument(
        "--real-state-moveit-shadow-evidence-path",
        default="reports/m6_real_state_moveit_shadow_plan_only_shadow_gripper/planning_status.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = generate_m6_shadow_readiness(
        live_smoke_report_path=args.live_smoke_report,
        topic_check_path=args.topic_check,
        output_dir=args.output_dir,
        real_tf_checked=args.real_tf_checked,
        moveit_shadow_checked=args.moveit_shadow_checked,
        mycobot_state_bridge_checked=args.mycobot_state_bridge_checked,
        real_state_moveit_shadow_checked=args.real_state_moveit_shadow_checked,
        real_tf_evidence_path=args.real_tf_evidence_path,
        moveit_shadow_evidence_path=args.moveit_shadow_evidence_path,
        real_state_moveit_shadow_evidence_path=args.real_state_moveit_shadow_evidence_path,
    )
    print(f"M6 shadow readiness: {report['overall_status']} -> {args.output_dir}")


if __name__ == "__main__":
    main()
