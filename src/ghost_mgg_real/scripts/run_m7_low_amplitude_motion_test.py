#!/usr/bin/env python3
"""Prepare or run the M7.1 low-amplitude empty-motion test.

Default behavior is dry-run only. Real motion requires both ``--execute`` and
the exact operator phrase recorded in the M7 safety preflight report.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_JOINT_NUMBER = 6
DEFAULT_DELTA_DEG = 2.0
DEFAULT_SPEED = 5
DEFAULT_RETURN_TOLERANCE_DEG = 1.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_angle_list(values: Any) -> list[float]:
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError("expected six joint angles")
    parsed = [float(value) for value in values]
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError("joint angles must be finite")
    return parsed


def validate_motion_limits(*, joint_number: int, delta_deg: float, speed: int) -> None:
    if int(joint_number) != DEFAULT_JOINT_NUMBER:
        raise ValueError("M7.1 only allows joint 6 low-amplitude motion")
    if abs(float(delta_deg)) > DEFAULT_DELTA_DEG:
        raise ValueError("M7.1 joint delta exceeds 2.0 deg")
    if int(speed) > DEFAULT_SPEED or int(speed) <= 0:
        raise ValueError("M7.1 speed must be in 1..5")


def build_target_angles(
    start_angles: list[float],
    *,
    joint_number: int = DEFAULT_JOINT_NUMBER,
    delta_deg: float = DEFAULT_DELTA_DEG,
) -> list[float]:
    start = _finite_angle_list(start_angles)
    validate_motion_limits(joint_number=joint_number, delta_deg=delta_deg, speed=DEFAULT_SPEED)
    target = list(start)
    target[int(joint_number) - 1] = round(target[int(joint_number) - 1] + float(delta_deg), 6)
    return target


def build_readback_summary(
    *,
    start_angles: list[float],
    target_angles: list[float],
    after_target_angles: list[float],
    final_angles: list[float],
    return_tolerance_deg: float = DEFAULT_RETURN_TOLERANCE_DEG,
) -> dict[str, Any]:
    start = _finite_angle_list(start_angles)
    target = _finite_angle_list(target_angles)
    after = _finite_angle_list(after_target_angles)
    final = _finite_angle_list(final_angles)
    deltas_final = [round(final[i] - start[i], 6) for i in range(6)]
    max_abs_delta = max(abs(value) for value in deltas_final)
    return {
        "available": True,
        "start_angles": start,
        "target_angles": target,
        "after_target_angles": after,
        "final_angles": final,
        "after_target_j6_delta_from_start_deg": round(after[5] - start[5], 6),
        "final_j6_delta_from_start_deg": round(final[5] - start[5], 6),
        "final_max_abs_delta_from_start_deg": round(max_abs_delta, 6),
        "returned_within_tolerance": max_abs_delta <= float(return_tolerance_deg),
        "return_tolerance_deg": float(return_tolerance_deg),
    }


def build_remote_motion_script(
    *,
    serial_port: str,
    baud: int,
    joint_number: int,
    delta_deg: float,
    speed: int,
    settle_sec: float,
) -> str:
    validate_motion_limits(joint_number=joint_number, delta_deg=delta_deg, speed=speed)
    return f"""
import json
import sys
import time

from pymycobot.mycobot280 import MyCobot280

mc = MyCobot280({serial_port!r}, {int(baud)})
joint_index = {int(joint_number) - 1}
delta_deg = {float(delta_deg)!r}
speed = {int(speed)}
settle_sec = {float(settle_sec)!r}

def read_angles():
    values = mc.get_angles()
    if not isinstance(values, list) or len(values) != 6:
        raise RuntimeError(f"bad angle readback: {{values!r}}")
    return [float(value) for value in values]

start_angles = read_angles()
target_angles = list(start_angles)
target_angles[joint_index] = target_angles[joint_index] + delta_deg
mc.send_angles(target_angles, speed)
time.sleep(settle_sec)
after_target_angles = read_angles()
mc.send_angles(start_angles, speed)
time.sleep(settle_sec)
final_angles = read_angles()
print(json.dumps({{
    "start_angles": start_angles,
    "target_angles": target_angles,
    "after_target_angles": after_target_angles,
    "final_angles": final_angles,
    "speed": speed,
    "delta_deg": delta_deg,
}}), flush=True)
""".strip()


def build_ssh_command(
    *,
    host: str,
    user: str,
    remote_script: str,
    connect_timeout_s: int,
    ssh_command: str,
) -> list[str]:
    remote_command = "python3 -u - <<'GHOST_MGG_M7_REMOTE_PY'\n"
    remote_command += remote_script
    remote_command += "\nGHOST_MGG_M7_REMOTE_PY"
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
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        return {
            "start_angles": _finite_angle_list(payload["start_angles"]),
            "target_angles": _finite_angle_list(payload["target_angles"]),
            "after_target_angles": _finite_angle_list(payload["after_target_angles"]),
            "final_angles": _finite_angle_list(payload["final_angles"]),
        }
    raise ValueError("remote motion script produced no JSON")


def _execute_remote_motion(
    *,
    host: str,
    user: str,
    serial_port: str,
    baud: int,
    joint_number: int,
    delta_deg: float,
    speed: int,
    settle_sec: float,
    connect_timeout_s: int,
    ssh_command: str,
) -> dict[str, Any]:
    remote_script = build_remote_motion_script(
        serial_port=serial_port,
        baud=baud,
        joint_number=joint_number,
        delta_deg=delta_deg,
        speed=speed,
        settle_sec=settle_sec,
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
        timeout=max(20.0, 2.0 * float(settle_sec) + 15.0),
    )
    return _parse_remote_result(completed.stdout)


def _preflight_ready(preflight: dict[str, Any]) -> bool:
    return (
        preflight.get("overall_status") == "ready_for_operator_controlled_m7_1"
        and preflight.get("motion_authorized") is False
        and not preflight.get("blockers")
    )


def generate_m7_low_amplitude_motion_test(
    *,
    preflight_path: Path,
    output_dir: Path,
    execute: bool = False,
    operator_phrase: str = "",
    joint_number: int = DEFAULT_JOINT_NUMBER,
    delta_deg: float = DEFAULT_DELTA_DEG,
    speed: int = DEFAULT_SPEED,
    settle_sec: float = 1.0,
    return_tolerance_deg: float = DEFAULT_RETURN_TOLERANCE_DEG,
    host: str = "10.42.0.169",
    user: str = "elephant",
    serial_port: str = "/dev/ttyAMA0",
    baud: int = 1000000,
    connect_timeout_s: int = 5,
    ssh_command: str = "ssh",
) -> dict[str, Any]:
    preflight_path = Path(preflight_path)
    output_dir = Path(output_dir)
    preflight = _read_json(preflight_path)
    validate_motion_limits(joint_number=joint_number, delta_deg=delta_deg, speed=speed)
    required_phrase = str(preflight.get("required_operator_phrase", ""))
    blockers: list[str] = []

    if execute and not _preflight_ready(preflight):
        blockers.append("preflight_not_ready")
    if execute and operator_phrase != required_phrase:
        blockers.append("operator_phrase_mismatch")

    commanded_motion = {
        "joint_number": int(joint_number),
        "delta_deg": float(delta_deg),
        "speed": int(speed),
        "settle_sec": float(settle_sec),
        "return_tolerance_deg": float(return_tolerance_deg),
        "scope": "empty low-amplitude distal wrist motion only",
    }
    readback: dict[str, Any] = {"available": False}
    motion_authorized = False

    if not execute:
        overall_status = "dry_run_only"
    elif blockers:
        overall_status = "blocked"
    else:
        motion_authorized = True
        remote = _execute_remote_motion(
            host=host,
            user=user,
            serial_port=serial_port,
            baud=baud,
            joint_number=joint_number,
            delta_deg=delta_deg,
            speed=speed,
            settle_sec=settle_sec,
            connect_timeout_s=connect_timeout_s,
            ssh_command=ssh_command,
        )
        readback = build_readback_summary(
            start_angles=remote["start_angles"],
            target_angles=remote["target_angles"],
            after_target_angles=remote["after_target_angles"],
            final_angles=remote["final_angles"],
            return_tolerance_deg=return_tolerance_deg,
        )
        overall_status = "executed_returned" if readback["returned_within_tolerance"] else "executed_return_warning"

    report = {
        "schema_version": "m7_low_amplitude_motion_test_v1",
        "generated_at_utc": _utc_now(),
        "stage": "M7.1",
        "overall_status": overall_status,
        "motion_authorized": motion_authorized,
        "preflight_path": str(preflight_path),
        "required_operator_phrase": required_phrase,
        "commanded_motion": commanded_motion,
        "readback": readback,
        "blockers": blockers,
        "next_steps": [
            "If this is a dry run, do not infer real robot behavior.",
            "If executed and return tolerance passed, review logs before opening M7.2.",
            "Do not run object grasping from this script.",
        ],
    }
    _write_json(output_dir / "m7_low_amplitude_motion_test.json", report)
    _write_index(output_dir / "index.md", report)
    return report


def _write_index(path: Path, report: dict[str, Any]) -> None:
    motion = report["commanded_motion"]
    lines = [
        "# M7 Low-Amplitude Motion Test",
        "",
        f"- overall_status: `{report['overall_status']}`",
        f"- motion_authorized: `{str(report['motion_authorized']).lower()}`",
        f"- stage: `{report['stage']}`",
        "",
        "## Commanded Motion",
        "",
        f"- joint_number: `{motion['joint_number']}`",
        f"- delta_deg: `{motion['delta_deg']}`",
        f"- speed: `{motion['speed']}`",
        f"- scope: `{motion['scope']}`",
        "",
        "## Readback",
        "",
    ]
    readback = report["readback"]
    if readback.get("available"):
        lines.extend(
            [
                f"- after_target_j6_delta_from_start_deg: `{readback['after_target_j6_delta_from_start_deg']}`",
                f"- final_max_abs_delta_from_start_deg: `{readback['final_max_abs_delta_from_start_deg']}`",
                f"- returned_within_tolerance: `{str(readback['returned_within_tolerance']).lower()}`",
            ]
        )
    else:
        lines.append("- no readback; dry-run or blocked")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "This report does not authorize M7.2 grasping.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path("reports/m7_safety_preflight/m7_safety_preflight.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m7_low_amplitude_motion_test"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--joint-number", type=int, default=DEFAULT_JOINT_NUMBER)
    parser.add_argument("--delta-deg", type=float, default=DEFAULT_DELTA_DEG)
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--return-tolerance-deg", type=float, default=DEFAULT_RETURN_TOLERANCE_DEG)
    parser.add_argument("--host", default="10.42.0.169")
    parser.add_argument("--user", default="elephant")
    parser.add_argument("--serial-port", default="/dev/ttyAMA0")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--connect-timeout-s", type=int, default=5)
    parser.add_argument("--ssh-command", default="ssh")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = generate_m7_low_amplitude_motion_test(
        preflight_path=args.preflight,
        output_dir=args.output_dir,
        execute=args.execute,
        operator_phrase=args.operator_phrase,
        joint_number=args.joint_number,
        delta_deg=args.delta_deg,
        speed=args.speed,
        settle_sec=args.settle_sec,
        return_tolerance_deg=args.return_tolerance_deg,
        host=args.host,
        user=args.user,
        serial_port=args.serial_port,
        baud=args.baud,
        connect_timeout_s=args.connect_timeout_s,
        ssh_command=args.ssh_command,
    )
    print(f"M7 low-amplitude motion test: {report['overall_status']} -> {args.output_dir}")


if __name__ == "__main__":
    main()
