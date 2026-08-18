#!/usr/bin/env python3
"""Run or dry-run the M7.2b target-based green-cylinder executor.

This executor consumes the non-actuating M7 green-cylinder gate and the M6
MoveIt plan-only output. It only sends real commands when the gate is ready,
the operator has explicitly confirmed, and no local SSH shadow bridge is
holding the myCobot serial port.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REQUIRED_OPERATOR_PHRASE = (
    "确认进入 M7.2，允许真实低速抓取绿色圆柱；目标和桌面已清空，"
    "机械臂周围安全，我已准备好断电/急停。"
)

STANDARD_TOP_GRASP_HOME_JOINT_DEG = [0.0, 30.0, -70.0, 0.0, 0.0, 0.0]
READY_GATE_STATUSES = {
    "ready_for_separate_real_execute",
    "ready_for_target_based_top_grasp_execute",
}

DEFAULT_SPEED = 3
DEFAULT_MAX_SPEED = 5
DEFAULT_GRIPPER_SPEED = 20
DEFAULT_LIFT_MM = 35.0
DEFAULT_MAX_DESCEND_MM = 110.0
DEFAULT_MIN_Z_MM = 120.0


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


def _finite_coords(values: Any, *, field_name: str) -> list[float]:
    return _finite_six(values, field_name=field_name)


def _top_plan_row(moveit_plan: dict[str, Any]) -> dict[str, Any]:
    rows = moveit_plan.get("rows", [])
    if not rows:
        raise ValueError("MoveIt plan contains no rows")
    if not isinstance(rows[0], dict):
        raise ValueError("MoveIt top row must be an object")
    return rows[0]


def pregrasp_angles_from_gate(gate: dict[str, Any]) -> list[float]:
    joint_delta = gate.get("joint_delta", {})
    return _finite_six(
        joint_delta.get("pregrasp_target_angles_deg"),
        field_name="pregrasp_target_angles_deg",
    )


def descend_mm_from_moveit_plan(moveit_plan: dict[str, Any]) -> float:
    row = _top_plan_row(moveit_plan)
    pregrasp_z_m = float(row["pregrasp_z_m"])
    grasp_z_m = float(row["grasp_z_m"])
    descend_mm = (pregrasp_z_m - grasp_z_m) * 1000.0
    if not math.isfinite(descend_mm) or descend_mm <= 0.0:
        raise ValueError("MoveIt plan must imply a positive vertical descent")
    return round(descend_mm, 6)


def moveit_plan_ready(moveit_plan: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if moveit_plan.get("summary", {}).get("all_planned") is not True:
        blockers.append("moveit_plan_not_all_planned")
    row = _top_plan_row(moveit_plan)
    if row.get("planned") is not True:
        blockers.append("top_row_not_planned")
    if row.get("descent_clearance", {}).get("status") != "ok":
        blockers.append("descent_clearance_not_ok")
    attempts = row.get("attempts", [])
    if not attempts or not attempts[0].get("final_joint_positions"):
        blockers.append("pregrasp_joint_positions_missing")
    return not blockers, blockers


def find_blocking_processes(process_table: str) -> list[str]:
    blockers = []
    for line in str(process_table).splitlines():
        if "m6_ssh_joint_state_bridge" in line:
            blockers.append("local_shadow_bridge_running")
        elif "GHOST_MGG_REMOTE_PY" in line and "get_angles" in line:
            blockers.append("local_shadow_bridge_running")
    return sorted(set(blockers))


def read_process_table() -> str:
    completed = subprocess.run(
        ["ps", "-eo", "pid,cmd"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def validate_motion_limits(
    *,
    descend_mm: float,
    lift_mm: float,
    speed: int,
    gripper_speed: int,
    max_descend_mm: float,
    min_z_mm: float,
) -> None:
    if float(descend_mm) <= 0.0 or float(descend_mm) > float(max_descend_mm):
        raise ValueError("M7.2b descend_mm exceeds the configured target-based limit")
    if float(lift_mm) <= 0.0 or float(lift_mm) > DEFAULT_LIFT_MM:
        raise ValueError("M7.2b lift_mm must be in (0, 35]")
    if int(speed) <= 0 or int(speed) > DEFAULT_MAX_SPEED:
        raise ValueError("M7.2b speed must be in 1..5")
    if int(gripper_speed) <= 0 or int(gripper_speed) > DEFAULT_GRIPPER_SPEED:
        raise ValueError("M7.2b gripper speed must be in 1..20")
    if float(min_z_mm) < 80.0:
        raise ValueError("M7.2b min_z_mm must be at least 80 mm")


def build_remote_target_based_script(
    *,
    serial_port: str,
    baud: int,
    pregrasp_angles_deg: list[float],
    descend_mm: float,
    lift_mm: float,
    speed: int,
    gripper_speed: int,
    pregrasp_settle_sec: float,
    settle_sec: float,
    gripper_settle_sec: float,
    min_z_mm: float,
    home_angles_deg: list[float] | None = None,
    home_settle_sec: float = 3.0,
) -> str:
    pregrasp_angles = _finite_six(pregrasp_angles_deg, field_name="pregrasp_angles_deg")
    home_angles = _finite_six(
        STANDARD_TOP_GRASP_HOME_JOINT_DEG if home_angles_deg is None else home_angles_deg,
        field_name="home_angles_deg",
    )
    validate_motion_limits(
        descend_mm=descend_mm,
        lift_mm=lift_mm,
        speed=speed,
        gripper_speed=gripper_speed,
        max_descend_mm=DEFAULT_MAX_DESCEND_MM,
        min_z_mm=min_z_mm,
    )
    return f"""
import json
import math
import time

from pymycobot.mycobot280 import MyCobot280

mc = MyCobot280({serial_port!r}, {int(baud)})
home_angles_deg = {home_angles!r}
pregrasp_angles_deg = {pregrasp_angles!r}
descend_mm = {float(descend_mm)!r}
lift_mm = {float(lift_mm)!r}
speed = {int(speed)}
gripper_speed = {int(gripper_speed)}
home_settle_sec = {float(home_settle_sec)!r}
pregrasp_settle_sec = {float(pregrasp_settle_sec)!r}
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

mc.send_angles(home_angles_deg, speed)
time.sleep(home_settle_sec)
after_home_angles = read_angles()
after_home_coords = read_coords()

mc.send_angles(pregrasp_angles_deg, speed)
time.sleep(pregrasp_settle_sec)
after_pregrasp_angles = read_angles()
after_pregrasp_coords = read_coords()

grasp_coords = list(after_pregrasp_coords)
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
    "home_angles_deg": home_angles_deg,
    "after_home_angles_deg": after_home_angles,
    "after_home_coords_mm_deg": after_home_coords,
    "after_pregrasp_angles_deg": after_pregrasp_angles,
    "after_pregrasp_coords_mm_deg": after_pregrasp_coords,
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
    remote_command = "python3 -u - <<'GHOST_MGG_M7_TARGET_BASED_PY'\n"
    remote_command += remote_script
    remote_command += "\nGHOST_MGG_M7_TARGET_BASED_PY"
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
        if not text:
            continue
        payload = json.loads(text)
        for key in (
            "start_angles_deg",
            "after_pregrasp_angles_deg",
            "after_home_angles_deg",
            "final_angles_deg",
            "start_coords_mm_deg",
            "after_home_coords_mm_deg",
            "after_pregrasp_coords_mm_deg",
            "final_coords_mm_deg",
        ):
            payload[key] = _finite_six(payload[key], field_name=key)
        return payload
    raise ValueError("remote target-based script produced no JSON")


def _execute_remote(
    *,
    host: str,
    user: str,
    serial_port: str,
    baud: int,
    pregrasp_angles_deg: list[float],
    descend_mm: float,
    lift_mm: float,
    speed: int,
    gripper_speed: int,
    home_angles_deg: list[float],
    pregrasp_settle_sec: float,
    home_settle_sec: float,
    settle_sec: float,
    gripper_settle_sec: float,
    min_z_mm: float,
    connect_timeout_s: int,
    ssh_command: str,
) -> dict[str, Any]:
    remote_script = build_remote_target_based_script(
        serial_port=serial_port,
        baud=baud,
        home_angles_deg=home_angles_deg,
        pregrasp_angles_deg=pregrasp_angles_deg,
        descend_mm=descend_mm,
        lift_mm=lift_mm,
        speed=speed,
        gripper_speed=gripper_speed,
        home_settle_sec=home_settle_sec,
        pregrasp_settle_sec=pregrasp_settle_sec,
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
        timeout=max(
            45.0,
            float(pregrasp_settle_sec)
            + 2.0 * float(settle_sec)
            + 2.0 * float(gripper_settle_sec)
            + 25.0,
        ),
    )
    return _parse_remote_result(completed.stdout)


def generate_target_based_execution_report(
    *,
    gate_path: Path,
    moveit_plan_path: Path,
    output_dir: Path,
    execute: bool,
    operator_phrase: str,
    target_based_confirmed: bool,
    process_table: str | None = None,
    max_descend_mm: float = DEFAULT_MAX_DESCEND_MM,
    lift_mm: float = DEFAULT_LIFT_MM,
    speed: int = DEFAULT_SPEED,
    gripper_speed: int = DEFAULT_GRIPPER_SPEED,
    home_angles_deg: list[float] | None = None,
    home_settle_sec: float = 3.0,
    pregrasp_settle_sec: float = 3.0,
    settle_sec: float = 2.5,
    gripper_settle_sec: float = 1.0,
    min_z_mm: float = DEFAULT_MIN_Z_MM,
    host: str = "10.42.0.169",
    user: str = "elephant",
    serial_port: str = "/dev/ttyAMA0",
    baud: int = 1000000,
    connect_timeout_s: int = 5,
    ssh_command: str = "ssh",
    execute_remote_fn: Callable[..., dict[str, Any]] = _execute_remote,
) -> dict[str, Any]:
    gate_path = Path(gate_path)
    moveit_plan_path = Path(moveit_plan_path)
    gate = _read_json(gate_path)
    moveit_plan = _read_json(moveit_plan_path)
    home_angles_deg = _finite_six(
        gate.get("standard_top_grasp_home_joint_deg", home_angles_deg or STANDARD_TOP_GRASP_HOME_JOINT_DEG),
        field_name="home_angles_deg",
    )
    pregrasp_angles_deg = pregrasp_angles_from_gate(gate)
    descend_mm = descend_mm_from_moveit_plan(moveit_plan)
    validate_motion_limits(
        descend_mm=descend_mm,
        lift_mm=lift_mm,
        speed=speed,
        gripper_speed=gripper_speed,
        max_descend_mm=max_descend_mm,
        min_z_mm=min_z_mm,
    )

    blockers: list[str] = []
    warnings: list[str] = []
    if gate.get("overall_status") not in READY_GATE_STATUSES or gate.get("blockers"):
        blockers.append("gate_not_ready")
        blockers.extend(str(blocker) for blocker in gate.get("blockers", []))
    _, moveit_blockers = moveit_plan_ready(moveit_plan)
    blockers.extend(moveit_blockers)
    if descend_mm > float(max_descend_mm):
        blockers.append("target_descent_exceeds_limit")
    if execute and operator_phrase != REQUIRED_OPERATOR_PHRASE:
        blockers.append("operator_phrase_mismatch")
    if execute and not target_based_confirmed:
        blockers.append("target_based_execution_not_confirmed")
    if execute:
        table = read_process_table() if process_table is None else process_table
        blockers.extend(find_blocking_processes(table))

    blockers = sorted(set(blockers))
    commanded_motion = {
        "mode": "target_based_top_grasp_home_first",
        "home_angles_deg": [round(float(value), 6) for value in home_angles_deg],
        "pregrasp_angles_deg": [round(float(value), 6) for value in pregrasp_angles_deg],
        "descend_mm": float(descend_mm),
        "lift_mm": float(lift_mm),
        "speed": int(speed),
        "gripper_speed": int(gripper_speed),
        "home_settle_sec": float(home_settle_sec),
        "pregrasp_settle_sec": float(pregrasp_settle_sec),
        "settle_sec": float(settle_sec),
        "gripper_settle_sec": float(gripper_settle_sec),
        "min_z_mm": float(min_z_mm),
    }

    remote_result: dict[str, Any] = {"available": False}
    motion_authorized = False
    if not execute:
        overall_status = "dry_run_ready" if not blockers else "dry_run_blocked"
    elif blockers:
        overall_status = "blocked"
    else:
        motion_authorized = True
        remote_result = {
            "available": True,
            **execute_remote_fn(
                host=host,
                user=user,
                serial_port=serial_port,
                baud=baud,
                home_angles_deg=home_angles_deg,
                pregrasp_angles_deg=pregrasp_angles_deg,
                descend_mm=descend_mm,
                lift_mm=lift_mm,
                speed=speed,
                gripper_speed=gripper_speed,
                home_settle_sec=home_settle_sec,
                pregrasp_settle_sec=pregrasp_settle_sec,
                settle_sec=settle_sec,
                gripper_settle_sec=gripper_settle_sec,
                min_z_mm=min_z_mm,
                connect_timeout_s=connect_timeout_s,
                ssh_command=ssh_command,
            ),
        }
        overall_status = "executed_target_based_lift"

    report = {
        "schema_version": "m7_target_based_executor_v2",
        "generated_at_utc": _utc_now(),
        "stage": "M7.2e",
        "overall_status": overall_status,
        "motion_authorized": motion_authorized,
        "execute_requested": bool(execute),
        "target_based_confirmed": bool(target_based_confirmed),
        "gate_path": str(gate_path),
        "moveit_plan_path": str(moveit_plan_path),
        "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
        "commanded_motion": commanded_motion,
        "gate_summary": {
            "overall_status": gate.get("overall_status"),
            "joint_delta": gate.get("joint_delta", {}),
        },
        "remote_result": remote_result,
        "blockers": blockers,
        "warnings": warnings,
        "next_steps": [
            "If blocked, do not send real motion. Fix the listed gate or process issue first.",
            "If executed, visually verify lift-and-hold before treating the attempt as success.",
            "Do not run this while M6 shadow bridge is connected to the myCobot serial port.",
        ],
    }
    output_dir = Path(output_dir)
    _write_json(output_dir / "m7_target_based_executor.json", report)
    (output_dir / "index.md").write_text(_render_index(report), encoding="utf-8")
    return report


def _render_index(report: dict[str, Any]) -> str:
    motion = report["commanded_motion"]
    lines = [
        "# M7 Target-Based Executor",
        "",
        f"- overall_status: `{report['overall_status']}`",
        f"- motion_authorized: `{str(report['motion_authorized']).lower()}`",
        f"- execute_requested: `{str(report['execute_requested']).lower()}`",
        f"- target_based_confirmed: `{str(report['target_based_confirmed']).lower()}`",
        "",
        "## Commanded Motion",
        "",
        f"- mode: `{motion['mode']}`",
        f"- home_angles_deg: `{motion['home_angles_deg']}`",
        f"- pregrasp_angles_deg: `{motion['pregrasp_angles_deg']}`",
        f"- descend_mm: `{motion['descend_mm']}`",
        f"- lift_mm: `{motion['lift_mm']}`",
        f"- speed: `{motion['speed']}`",
        f"- gripper_speed: `{motion['gripper_speed']}`",
        "",
        "## Blockers",
        "",
    ]
    if report["blockers"]:
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "## Remote Result", ""])
    remote = report["remote_result"]
    if remote.get("available"):
        for key in (
            "start_angles_deg",
            "after_home_angles_deg",
            "after_home_coords_mm_deg",
            "after_pregrasp_angles_deg",
            "after_pregrasp_coords_mm_deg",
            "after_descent_coords_mm_deg",
            "final_coords_mm_deg",
            "gripper_before",
            "after_close_gripper",
            "gripper_final",
        ):
            lines.append(f"- {key}: `{remote.get(key)}`")
    else:
        lines.append("- no remote motion result; dry-run or blocked")
    lines.extend(["", "This report controls whether M7.2e target-based real execution was allowed.", ""])
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--moveit-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m7_target_based_executor"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    parser.add_argument("--target-based-confirmed", action="store_true")
    parser.add_argument("--max-descend-mm", type=float, default=DEFAULT_MAX_DESCEND_MM)
    parser.add_argument("--lift-mm", type=float, default=DEFAULT_LIFT_MM)
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    parser.add_argument("--gripper-speed", type=int, default=DEFAULT_GRIPPER_SPEED)
    parser.add_argument("--home-angles-deg", default=",".join(str(value) for value in STANDARD_TOP_GRASP_HOME_JOINT_DEG))
    parser.add_argument("--home-settle-sec", type=float, default=3.0)
    parser.add_argument("--pregrasp-settle-sec", type=float, default=3.0)
    parser.add_argument("--settle-sec", type=float, default=2.5)
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
    home_angles_deg = [float(value) for value in args.home_angles_deg.split(",")]
    report = generate_target_based_execution_report(
        gate_path=args.gate,
        moveit_plan_path=args.moveit_plan,
        output_dir=args.output_dir,
        execute=args.execute,
        operator_phrase=args.operator_phrase,
        target_based_confirmed=args.target_based_confirmed,
        process_table=None,
        max_descend_mm=args.max_descend_mm,
        lift_mm=args.lift_mm,
        speed=args.speed,
        gripper_speed=args.gripper_speed,
        home_angles_deg=home_angles_deg,
        home_settle_sec=args.home_settle_sec,
        pregrasp_settle_sec=args.pregrasp_settle_sec,
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
    print(f"M7 target-based executor: {report['overall_status']} -> {args.output_dir}")


if __name__ == "__main__":
    main()
