#!/usr/bin/env python3
"""Run or dry-run an M7.2 current-pregrasp green-cylinder grasp.

This path is for the first physical grasp only when the operator has manually
placed the real arm in a suitable pregrasp pose. It deliberately avoids sending
the large MoveIt pregrasp joint solution to hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_OPERATOR_PHRASE = (
    "确认进入 M7.2，允许真实低速抓取绿色圆柱；目标和桌面已清空，"
    "机械臂周围安全，我已准备好断电/急停。"
)

DEFAULT_DESCEND_MM = 25.0
DEFAULT_LIFT_MM = 35.0
DEFAULT_SPEED = 5
DEFAULT_GRIPPER_SPEED = 20
DEFAULT_MIN_Z_MM = 120.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_list(values: Any, *, length: int, field_name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != int(length):
        raise ValueError(f"{field_name} must contain {length} numeric values")
    parsed = [float(value) for value in values]
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def validate_motion_limits(
    *,
    descend_mm: float,
    lift_mm: float,
    speed: int,
    gripper_speed: int,
    min_z_mm: float,
) -> None:
    if float(descend_mm) <= 0.0 or float(descend_mm) > DEFAULT_DESCEND_MM:
        raise ValueError("M7.2 descend_mm must be in (0, 25]")
    if float(lift_mm) <= 0.0 or float(lift_mm) > DEFAULT_LIFT_MM:
        raise ValueError("M7.2 lift_mm must be in (0, 35]")
    if int(speed) <= 0 or int(speed) > DEFAULT_SPEED:
        raise ValueError("M7.2 Cartesian speed must be in 1..5")
    if int(gripper_speed) <= 0 or int(gripper_speed) > DEFAULT_GRIPPER_SPEED:
        raise ValueError("M7.2 gripper speed must be in 1..20")
    if float(min_z_mm) < 80.0:
        raise ValueError("M7.2 min_z_mm must be at least 80 mm")


def build_remote_current_pregrasp_script(
    *,
    serial_port: str,
    baud: int,
    descend_mm: float,
    lift_mm: float,
    speed: int,
    gripper_speed: int,
    settle_sec: float,
    gripper_settle_sec: float,
    min_z_mm: float,
) -> str:
    validate_motion_limits(
        descend_mm=descend_mm,
        lift_mm=lift_mm,
        speed=speed,
        gripper_speed=gripper_speed,
        min_z_mm=min_z_mm,
    )
    return f"""
import json
import math
import time

from pymycobot.mycobot280 import MyCobot280

mc = MyCobot280({serial_port!r}, {int(baud)})
descend_mm = {float(descend_mm)!r}
lift_mm = {float(lift_mm)!r}
speed = {int(speed)}
gripper_speed = {int(gripper_speed)}
settle_sec = {float(settle_sec)!r}
gripper_settle_sec = {float(gripper_settle_sec)!r}
min_z_mm = {float(min_z_mm)!r}

def finite_list(values, expected_len, name):
    if not isinstance(values, list) or len(values) != expected_len:
        raise RuntimeError(f"bad {{name}} readback: {{values!r}}")
    parsed = [float(value) for value in values]
    if not all(math.isfinite(value) for value in parsed):
        raise RuntimeError(f"non-finite {{name}} readback: {{values!r}}")
    return parsed

def read_angles():
    return finite_list(mc.get_angles(), 6, "angles")

def read_coords():
    return finite_list(mc.get_coords(), 6, "coords")

start_angles = read_angles()
start_coords = read_coords()
gripper_before = mc.get_gripper_value() if hasattr(mc, "get_gripper_value") else None

mc.set_gripper_state(0, gripper_speed)
time.sleep(gripper_settle_sec)
after_open_gripper = mc.get_gripper_value() if hasattr(mc, "get_gripper_value") else None

grasp_coords = list(start_coords)
grasp_coords[2] = grasp_coords[2] - descend_mm
if grasp_coords[2] < min_z_mm:
    raise RuntimeError(f"blocked: planned grasp z {{grasp_coords[2]}} below min_z_mm {{min_z_mm}}")

mc.send_coords(grasp_coords, speed, 1)
time.sleep(settle_sec)
after_descent_coords = read_coords()

mc.set_gripper_state(1, gripper_speed)
time.sleep(gripper_settle_sec)
after_close_gripper = mc.get_gripper_value() if hasattr(mc, "get_gripper_value") else None

lift_coords = list(grasp_coords)
lift_coords[2] = lift_coords[2] + lift_mm
mc.send_coords(lift_coords, speed, 1)
time.sleep(settle_sec)

final_coords = read_coords()
final_angles = read_angles()
gripper_final = mc.get_gripper_value() if hasattr(mc, "get_gripper_value") else None

print(json.dumps({{
    "start_angles_deg": start_angles,
    "start_coords_mm_deg": start_coords,
    "grasp_coords_mm_deg": grasp_coords,
    "after_descent_coords_mm_deg": after_descent_coords,
    "lift_coords_mm_deg": lift_coords,
    "final_coords_mm_deg": final_coords,
    "final_angles_deg": final_angles,
    "gripper_before": gripper_before,
    "after_open_gripper": after_open_gripper,
    "after_close_gripper": after_close_gripper,
    "gripper_final": gripper_final,
    "descend_mm": descend_mm,
    "lift_mm": lift_mm,
    "speed": speed,
    "gripper_speed": gripper_speed,
}}, sort_keys=True), flush=True)
""".strip()


def build_ssh_command(
    *,
    host: str,
    user: str,
    remote_script: str,
    connect_timeout_s: int,
    ssh_command: str,
) -> list[str]:
    remote_command = "python3 -u - <<'GHOST_MGG_M7_CURRENT_PREGRASP_PY'\n"
    remote_command += remote_script
    remote_command += "\nGHOST_MGG_M7_CURRENT_PREGRASP_PY"
    command = shlex.split(ssh_command)
    command.extend(
        [
            "-o",
            f"ConnectTimeout={int(connect_timeout_s)}",
            "-o",
            "ServerAliveInterval=2",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"{user}@{host}",
            remote_command,
        ]
    )
    return command


def _parse_remote_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if text:
            payload = json.loads(text)
            payload["start_angles_deg"] = _finite_list(
                payload["start_angles_deg"], length=6, field_name="start_angles_deg"
            )
            payload["start_coords_mm_deg"] = _finite_list(
                payload["start_coords_mm_deg"], length=6, field_name="start_coords_mm_deg"
            )
            payload["final_coords_mm_deg"] = _finite_list(
                payload["final_coords_mm_deg"], length=6, field_name="final_coords_mm_deg"
            )
            payload["final_angles_deg"] = _finite_list(
                payload["final_angles_deg"], length=6, field_name="final_angles_deg"
            )
            return payload
    raise ValueError("remote current-pregrasp script produced no JSON")


def _execute_remote(
    *,
    host: str,
    user: str,
    serial_port: str,
    baud: int,
    descend_mm: float,
    lift_mm: float,
    speed: int,
    gripper_speed: int,
    settle_sec: float,
    gripper_settle_sec: float,
    min_z_mm: float,
    connect_timeout_s: int,
    ssh_command: str,
) -> dict[str, Any]:
    remote_script = build_remote_current_pregrasp_script(
        serial_port=serial_port,
        baud=baud,
        descend_mm=descend_mm,
        lift_mm=lift_mm,
        speed=speed,
        gripper_speed=gripper_speed,
        settle_sec=settle_sec,
        gripper_settle_sec=gripper_settle_sec,
        min_z_mm=min_z_mm,
    )
    command = build_ssh_command(
        host=host,
        user=user,
        remote_script=remote_script,
        connect_timeout_s=connect_timeout_s,
        ssh_command=ssh_command,
    )
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(30.0, 2.0 * float(settle_sec) + 2.0 * float(gripper_settle_sec) + 20.0),
    )
    return _parse_remote_result(completed.stdout)


def generate_current_pregrasp_grasp_report(
    *,
    output_dir: Path,
    execute: bool,
    operator_phrase: str,
    current_pregrasp_confirmed: bool,
    descend_mm: float = DEFAULT_DESCEND_MM,
    lift_mm: float = DEFAULT_LIFT_MM,
    speed: int = DEFAULT_SPEED,
    gripper_speed: int = DEFAULT_GRIPPER_SPEED,
    settle_sec: float = 2.0,
    gripper_settle_sec: float = 1.0,
    min_z_mm: float = DEFAULT_MIN_Z_MM,
    host: str = "10.42.0.169",
    user: str = "elephant",
    serial_port: str = "/dev/ttyAMA0",
    baud: int = 1000000,
    connect_timeout_s: int = 5,
    ssh_command: str = "ssh",
) -> dict[str, Any]:
    validate_motion_limits(
        descend_mm=descend_mm,
        lift_mm=lift_mm,
        speed=speed,
        gripper_speed=gripper_speed,
        min_z_mm=min_z_mm,
    )
    blockers: list[str] = []
    if execute and operator_phrase != REQUIRED_OPERATOR_PHRASE:
        blockers.append("operator_phrase_mismatch")
    if execute and not current_pregrasp_confirmed:
        blockers.append("current_pregrasp_not_confirmed")

    commanded_motion = {
        "mode": "current_pregrasp_micro_grasp",
        "descend_mm": float(descend_mm),
        "lift_mm": float(lift_mm),
        "speed": int(speed),
        "gripper_speed": int(gripper_speed),
        "settle_sec": float(settle_sec),
        "gripper_settle_sec": float(gripper_settle_sec),
        "min_z_mm": float(min_z_mm),
        "object": "green_cylinder",
    }
    remote_result: dict[str, Any] = {"available": False}
    motion_authorized = False

    if not execute:
        overall_status = "dry_run_only"
    elif blockers:
        overall_status = "blocked"
    else:
        motion_authorized = True
        remote_result = {
            "available": True,
            **_execute_remote(
                host=host,
                user=user,
                serial_port=serial_port,
                baud=baud,
                descend_mm=descend_mm,
                lift_mm=lift_mm,
                speed=speed,
                gripper_speed=gripper_speed,
                settle_sec=settle_sec,
                gripper_settle_sec=gripper_settle_sec,
                min_z_mm=min_z_mm,
                connect_timeout_s=connect_timeout_s,
                ssh_command=ssh_command,
            ),
        }
        overall_status = "executed_current_pregrasp_lift"

    report = {
        "schema_version": "m7_current_pregrasp_grasp_v1",
        "generated_at_utc": _utc_now(),
        "stage": "M7.2a",
        "overall_status": overall_status,
        "motion_authorized": motion_authorized,
        "execute_requested": bool(execute),
        "current_pregrasp_confirmed": bool(current_pregrasp_confirmed),
        "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
        "commanded_motion": commanded_motion,
        "remote_result": remote_result,
        "blockers": blockers,
        "next_steps": [
            "If executed, visually verify whether the green cylinder is retained after lift.",
            "If not retained, stop and adjust pregrasp pose or gripper timing before retry.",
            "Do not use this current-pregrasp shortcut for transparent targets.",
        ],
    }
    output_dir = Path(output_dir)
    _write_json(output_dir / "m7_current_pregrasp_grasp.json", report)
    _write_index(output_dir / "index.md", report)
    return report


def _write_index(path: Path, report: dict[str, Any]) -> None:
    motion = report["commanded_motion"]
    lines = [
        "# M7 Current-Pregrasp Grasp",
        "",
        f"- overall_status: `{report['overall_status']}`",
        f"- motion_authorized: `{str(report['motion_authorized']).lower()}`",
        f"- stage: `{report['stage']}`",
        f"- current_pregrasp_confirmed: `{str(report['current_pregrasp_confirmed']).lower()}`",
        "",
        "## Commanded Motion",
        "",
        f"- object: `{motion['object']}`",
        f"- descend_mm: `{motion['descend_mm']}`",
        f"- lift_mm: `{motion['lift_mm']}`",
        f"- speed: `{motion['speed']}`",
        f"- gripper_speed: `{motion['gripper_speed']}`",
        f"- min_z_mm: `{motion['min_z_mm']}`",
        "",
        "## Remote Result",
        "",
    ]
    remote = report["remote_result"]
    if remote.get("available"):
        lines.extend(
            [
                f"- start_coords_mm_deg: `{remote['start_coords_mm_deg']}`",
                f"- grasp_coords_mm_deg: `{remote['grasp_coords_mm_deg']}`",
                f"- final_coords_mm_deg: `{remote['final_coords_mm_deg']}`",
                f"- gripper_before: `{remote.get('gripper_before')}`",
                f"- after_close_gripper: `{remote.get('after_close_gripper')}`",
                f"- gripper_final: `{remote.get('gripper_final')}`",
            ]
        )
    else:
        lines.append("- no remote result; dry-run or blocked")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "This report is only valid for the manually confirmed green-cylinder pregrasp pose.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m7_current_pregrasp_grasp"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--current-pregrasp-confirmed", action="store_true")
    parser.add_argument("--descend-mm", type=float, default=DEFAULT_DESCEND_MM)
    parser.add_argument("--lift-mm", type=float, default=DEFAULT_LIFT_MM)
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    parser.add_argument("--gripper-speed", type=int, default=DEFAULT_GRIPPER_SPEED)
    parser.add_argument("--settle-sec", type=float, default=2.0)
    parser.add_argument("--gripper-settle-sec", type=float, default=1.0)
    parser.add_argument("--min-z-mm", type=float, default=DEFAULT_MIN_Z_MM)
    parser.add_argument("--host", default="10.42.0.169")
    parser.add_argument("--user", default="elephant")
    parser.add_argument("--serial-port", default="/dev/ttyAMA0")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--connect-timeout-s", type=int, default=5)
    parser.add_argument("--ssh-command", default="ssh")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = generate_current_pregrasp_grasp_report(
        output_dir=args.output_dir,
        execute=args.execute,
        operator_phrase=args.operator_phrase,
        current_pregrasp_confirmed=args.current_pregrasp_confirmed,
        descend_mm=args.descend_mm,
        lift_mm=args.lift_mm,
        speed=args.speed,
        gripper_speed=args.gripper_speed,
        settle_sec=args.settle_sec,
        gripper_settle_sec=args.gripper_settle_sec,
        min_z_mm=args.min_z_mm,
        host=args.host,
        user=args.user,
        serial_port=args.serial_port,
        baud=args.baud,
        connect_timeout_s=args.connect_timeout_s,
        ssh_command=args.ssh_command,
    )
    print(f"M7 current-pregrasp grasp: {report['overall_status']} -> {args.output_dir}")


if __name__ == "__main__":
    main()
