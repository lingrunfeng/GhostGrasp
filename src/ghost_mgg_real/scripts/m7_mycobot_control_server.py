#!/usr/bin/env python3
"""M7 real myCobot280Pi control server.

This node is the single local owner of the remote myCobot serial session.  It
keeps one SSH subprocess alive, publishes /joint_states from that subprocess,
and exposes ROS services for explicitly authorized low-speed motions.
"""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import threading
import time
import uuid
from typing import Any, Iterable


REQUIRED_OPERATOR_PHRASE = (
    "确认进入 M7.2，允许真实低速抓取绿色圆柱；目标和桌面已清空，"
    "机械臂周围安全，我已准备好断电/急停。"
)

STANDARD_TOP_GRASP_HOME_JOINT_DEG = [0.0, 30.0, -70.0, 0.0, 0.0, 0.0]
GHOST_MGG_JOINT_NAMES = [
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_link6",
    "link6_to_link6_flange",
]
SHADOW_GRIPPER_JOINT_NAMES = [
    "gripper_controller",
    "gripper_base_to_gripper_left2",
    "gripper_left3_to_gripper_left1",
    "gripper_base_to_gripper_right3",
    "gripper_base_to_gripper_right2",
    "gripper_right3_to_gripper_right1",
]


def _finite_list(values: Any, *, length: int, field_name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != int(length):
        raise ValueError(f"{field_name} must contain {length} numeric values")
    parsed = [float(value) for value in values]
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def degrees_to_ghost_joint_state(angles_deg: Iterable[float]) -> tuple[list[str], list[float]]:
    angles = list(angles_deg)
    if len(angles) != 6:
        raise ValueError(f"expected 6 angles, got {len(angles)}")
    return list(GHOST_MGG_JOINT_NAMES), [math.radians(float(angle)) for angle in angles]


def append_shadow_gripper_state(
    names: list[str],
    positions: list[float],
    *,
    gripper_position: float,
) -> tuple[list[str], list[float]]:
    value = float(gripper_position)
    shadow_positions = [value, value, -value, -value, -value, value]
    return names + list(SHADOW_GRIPPER_JOINT_NAMES), positions + shadow_positions


def parse_remote_control_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    message_type = payload.get("type")
    if message_type == "state":
        payload["angles"] = _finite_list(payload.get("angles"), length=6, field_name="angles")
        coords = payload.get("coords")
        if isinstance(coords, list) and len(coords) == 6:
            payload["coords"] = _finite_list(coords, length=6, field_name="coords")
        return payload
    if message_type == "response" and isinstance(payload.get("id"), str):
        return payload
    return None


def validate_speed(speed: float, *, max_speed: float, field_name: str) -> float:
    value = float(speed)
    if not math.isfinite(value) or value <= 0.0 or value > float(max_speed):
        raise ValueError(f"{field_name} must be in (0, {max_speed}]")
    return value


def build_home_command(
    *,
    request_id: str,
    execute: bool,
    operator_phrase: str,
    target_angles_deg: list[float],
    speed: float,
    max_delta_deg: float,
    staged_home: bool = False,
    open_gripper: bool,
    gripper_speed: float,
) -> dict[str, Any]:
    return {
        "id": str(request_id),
        "command": "go_home",
        "execute": bool(execute),
        "operator_phrase": str(operator_phrase),
        "target_angles_deg": _finite_list(target_angles_deg, length=6, field_name="target_angles_deg"),
        "speed": validate_speed(speed, max_speed=5.0, field_name="speed"),
        "max_delta_deg": float(max_delta_deg),
        "staged_home": bool(staged_home),
        "open_gripper": bool(open_gripper),
        "gripper_speed": validate_speed(
            gripper_speed, max_speed=20.0, field_name="gripper_speed"
        ),
    }


def build_guarded_taught_grasp_command(
    *,
    request_id: str,
    execute: bool,
    operator_phrase: str,
    pregrasp_coords_mm_deg: list[float],
    grasp_coords_mm_deg: list[float],
    lift_coords_mm_deg: list[float],
    speed: float,
    gripper_speed: float,
    min_z_mm: float,
) -> dict[str, Any]:
    pregrasp = _finite_list(
        pregrasp_coords_mm_deg, length=6, field_name="pregrasp_coords_mm_deg"
    )
    grasp = _finite_list(grasp_coords_mm_deg, length=6, field_name="grasp_coords_mm_deg")
    lift = _finite_list(lift_coords_mm_deg, length=6, field_name="lift_coords_mm_deg")
    min_z = float(min_z_mm)
    if grasp[2] < min_z:
        raise ValueError("grasp z is below min_z_mm")
    return {
        "id": str(request_id),
        "command": "guarded_taught_grasp",
        "execute": bool(execute),
        "operator_phrase": str(operator_phrase),
        "pregrasp_coords_mm_deg": pregrasp,
        "grasp_coords_mm_deg": grasp,
        "lift_coords_mm_deg": lift,
        "speed": validate_speed(speed, max_speed=5.0, field_name="speed"),
        "gripper_speed": validate_speed(
            gripper_speed, max_speed=20.0, field_name="gripper_speed"
        ),
        "min_z_mm": min_z,
    }


def build_remote_control_script(serial_port: str, baud: int, sample_hz: float) -> str:
    period_s = 1.0 / max(float(sample_hz), 0.1)
    required_phrase = REQUIRED_OPERATOR_PHRASE
    return f"""
import json
import math
import select
import sys
import time

from pymycobot.mycobot280 import MyCobot280

mc = MyCobot280({serial_port!r}, {int(baud)})
period_s = {period_s!r}
required_operator_phrase = {required_phrase!r}

def finite(values, length, name):
    if not isinstance(values, list) or len(values) != length:
        raise RuntimeError(f"bad {{name}}: {{values!r}}")
    parsed = [float(value) for value in values]
    if not all(math.isfinite(value) for value in parsed):
        raise RuntimeError(f"non-finite {{name}}: {{values!r}}")
    return parsed

def read_state():
    angles = finite(mc.get_angles(), 6, "angles")
    try:
        coords = finite(mc.get_coords(), 6, "coords")
    except Exception:
        coords = None
    try:
        gripper = mc.get_gripper_value()
    except Exception:
        gripper = None
    return {{"angles": angles, "coords": coords, "gripper": gripper}}

def build_joint_stages(start, target, max_delta):
    max_step = max(float(max_delta), 1.0)
    delta = [target[i] - start[i] for i in range(6)]
    max_abs_delta = max(abs(value) for value in delta)
    stage_count = max(1, int(math.ceil(max_abs_delta / max_step)))
    stages = []
    for stage_index in range(1, stage_count + 1):
        ratio = float(stage_index) / float(stage_count)
        stages.append([start[i] + delta[i] * ratio for i in range(6)])
    return stages

def emit(payload):
    print(json.dumps(payload, sort_keys=True), flush=True)

def respond(request_id, ok, status, message="", result=None):
    emit({{
        "type": "response",
        "id": request_id,
        "ok": bool(ok),
        "status": status,
        "message": message,
        "result": result or {{}},
    }})

def handle_go_home(command):
    request_id = str(command.get("id", ""))
    target = finite(command.get("target_angles_deg"), 6, "target_angles_deg")
    speed = int(float(command.get("speed", 3)))
    gripper_speed = int(float(command.get("gripper_speed", 20)))
    max_delta = float(command.get("max_delta_deg", 90.0))
    staged_home = bool(command.get("staged_home", False))
    start = read_state()
    delta = [target[i] - start["angles"][i] for i in range(6)]
    max_abs_delta = max(abs(value) for value in delta)
    stages = build_joint_stages(start["angles"], target, max_delta) if staged_home else [target]
    result = {{
        "start": start,
        "target_angles_deg": target,
        "delta_deg": delta,
        "max_abs_delta_deg": max_abs_delta,
        "speed": speed,
        "staged_home": staged_home,
        "stage_count": len(stages),
        "stage_targets_deg": stages,
        "open_gripper": bool(command.get("open_gripper", False)),
        "gripper_speed": gripper_speed,
    }}
    if command.get("operator_phrase", "") != required_operator_phrase:
        respond(request_id, False, "operator_phrase_mismatch", result=result)
        return
    if max_abs_delta > max_delta and not staged_home:
        respond(request_id, False, "joint_delta_too_large", result=result)
        return
    if not bool(command.get("execute", False)):
        status = "dry_run_staged_home_ready" if staged_home else "dry_run_home_ready"
        respond(request_id, True, status, result=result)
        return
    if bool(command.get("open_gripper", False)):
        mc.set_gripper_state(0, gripper_speed)
        time.sleep(1.0)
        result["after_open_gripper"] = read_state()
    stage_results = []
    for stage in stages:
        mc.send_angles(stage, speed)
        time.sleep(4.0)
        stage_results.append(read_state())
    result["stage_results"] = stage_results
    result["after"] = read_state()
    status = "executed_staged_home" if staged_home else "executed_home"
    respond(request_id, True, status, result=result)

def handle_guarded_taught_grasp(command):
    request_id = str(command.get("id", ""))
    pregrasp = finite(command.get("pregrasp_coords_mm_deg"), 6, "pregrasp_coords_mm_deg")
    grasp = finite(command.get("grasp_coords_mm_deg"), 6, "grasp_coords_mm_deg")
    lift = finite(command.get("lift_coords_mm_deg"), 6, "lift_coords_mm_deg")
    speed = int(float(command.get("speed", 3)))
    gripper_speed = int(float(command.get("gripper_speed", 20)))
    min_z = float(command.get("min_z_mm", 120.0))
    start = read_state()
    result = {{
        "start": start,
        "pregrasp_coords_mm_deg": pregrasp,
        "grasp_coords_mm_deg": grasp,
        "lift_coords_mm_deg": lift,
        "speed": speed,
        "gripper_speed": gripper_speed,
        "min_z_mm": min_z,
    }}
    if command.get("operator_phrase", "") != required_operator_phrase:
        respond(request_id, False, "operator_phrase_mismatch", result=result)
        return
    if grasp[2] < min_z:
        respond(request_id, False, "grasp_z_below_min_z", result=result)
        return
    if not bool(command.get("execute", False)):
        respond(request_id, True, "dry_run_grasp_ready", result=result)
        return
    mc.set_gripper_state(0, gripper_speed)
    time.sleep(1.0)
    result["after_open"] = read_state()
    mc.send_coords(pregrasp, speed, 1)
    time.sleep(4.0)
    result["after_pregrasp"] = read_state()
    mc.send_coords(grasp, speed, 1)
    time.sleep(2.5)
    result["after_descent"] = read_state()
    mc.set_gripper_state(1, gripper_speed)
    time.sleep(1.2)
    result["after_close"] = read_state()
    mc.send_coords(lift, speed, 1)
    time.sleep(3.0)
    result["after_lift"] = read_state()
    respond(request_id, True, "executed_guarded_taught_grasp", result=result)

def handle_command(command):
    name = command.get("command")
    if name == "go_home":
        handle_go_home(command)
    elif name == "guarded_taught_grasp":
        handle_guarded_taught_grasp(command)
    else:
        respond(str(command.get("id", "")), False, "unknown_command")

next_state_time = 0.0
while True:
    now = time.time()
    timeout = max(0.0, min(period_s, next_state_time - now))
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if readable:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            handle_command(json.loads(line))
        except Exception as exc:
            request_id = ""
            try:
                request_id = str(json.loads(line).get("id", ""))
            except Exception:
                pass
            respond(request_id, False, "remote_exception", str(exc))
    if time.time() >= next_state_time:
        try:
            state = read_state()
            emit({{"type": "state", **state}})
        except Exception as exc:
            emit({{"type": "state_error", "error": str(exc)}})
        next_state_time = time.time() + period_s
""".strip()


def build_ssh_command(
    *,
    host: str,
    user: str,
    remote_script: str,
    connect_timeout_s: int,
    ssh_command: str = "ssh",
) -> list[str]:
    # Use -c instead of a heredoc so the remote Python process keeps stdin
    # available for JSON command requests after the script starts.
    remote_command = "python3 -u -c " + shlex.quote(remote_script)
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


class RemoteMyCobotSession:
    def __init__(
        self,
        *,
        command: list[str],
        state_callback,
        response_timeout_s: float = 45.0,
    ) -> None:
        self.command = command
        self.state_callback = state_callback
        self.response_timeout_s = float(response_timeout_s)
        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.condition = threading.Condition()
        self.responses: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self.reader_thread = threading.Thread(target=self._read_stdout_loop, daemon=True)
        self.reader_thread.start()

    def _read_stdout_loop(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            payload = parse_remote_control_line(line)
            if payload is None:
                continue
            if payload["type"] == "state":
                self.state_callback(payload)
                continue
            with self.condition:
                self.responses[payload["id"]] = payload
                self.condition.notify_all()

    def request(self, command: dict[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("remote myCobot control session is not running")
        request_id = str(command["id"])
        self.process.stdin.write(json.dumps(command, sort_keys=True) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + float(timeout_s or self.response_timeout_s)
        with self.condition:
            while request_id not in self.responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"timed out waiting for myCobot response {request_id}")
                self.condition.wait(timeout=remaining)
            return self.responses.pop(request_id)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()


def safe_rclpy_shutdown(rclpy_module) -> None:
    try:
        rclpy_module.shutdown()
    except Exception as exc:
        text = str(exc)
        if "rcl_shutdown already called" in text or "context is zero initialized" in text:
            return
        raise


class M7MyCobotControlServer:
    def __init__(self) -> None:
        import rclpy
        from ghost_mgg_interfaces.srv import MyCobotGoHome, MyCobotGuardedTaughtGrasp
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        class ControlNode(Node):
            def __init__(self) -> None:
                super().__init__("m7_mycobot_control_server")
                self.declare_parameter("robot_host", "10.42.0.169")
                self.declare_parameter("robot_user", "elephant")
                self.declare_parameter("serial_port", "/dev/ttyAMA0")
                self.declare_parameter("baud", 1000000)
                self.declare_parameter("sample_hz", 10.0)
                self.declare_parameter("connect_timeout_s", 5)
                self.declare_parameter("ssh_command", "ssh")
                self.declare_parameter("frame_id", "")
                self.declare_parameter("publish_shadow_gripper_joints", True)
                self.declare_parameter("shadow_gripper_position", 0.15)
                self._joint_state_type = JointState
                self.publisher = self.create_publisher(JointState, "/joint_states", 10)
                remote_script = build_remote_control_script(
                    serial_port=self.get_string("serial_port"),
                    baud=self.get_int("baud"),
                    sample_hz=self.get_double("sample_hz"),
                )
                command = build_ssh_command(
                    host=self.get_string("robot_host"),
                    user=self.get_string("robot_user"),
                    remote_script=remote_script,
                    connect_timeout_s=self.get_int("connect_timeout_s"),
                    ssh_command=self.get_string("ssh_command"),
                )
                self.session = RemoteMyCobotSession(
                    command=command,
                    state_callback=self.publish_state,
                    response_timeout_s=60.0,
                )
                self.session.start()
                self.create_service(
                    MyCobotGoHome,
                    "/ghost_mgg/mycobot/go_home",
                    self.handle_go_home,
                )
                self.create_service(
                    MyCobotGuardedTaughtGrasp,
                    "/ghost_mgg/mycobot/guarded_taught_grasp",
                    self.handle_guarded_taught_grasp,
                )
                self.get_logger().info("M7 myCobot control server is up")

            def get_string(self, name: str) -> str:
                return str(self.get_parameter(name).value)

            def get_int(self, name: str) -> int:
                return int(self.get_parameter(name).value)

            def get_double(self, name: str) -> float:
                return float(self.get_parameter(name).value)

            def get_bool(self, name: str) -> bool:
                value = self.get_parameter(name).value
                if isinstance(value, str):
                    return value.lower() in {"1", "true", "yes", "on"}
                return bool(value)

            def publish_state(self, state: dict[str, Any]) -> None:
                names, positions = degrees_to_ghost_joint_state(state["angles"])
                if self.get_bool("publish_shadow_gripper_joints"):
                    names, positions = append_shadow_gripper_state(
                        names,
                        positions,
                        gripper_position=self.get_double("shadow_gripper_position"),
                    )
                msg = self._joint_state_type()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.get_string("frame_id")
                msg.name = names
                msg.position = positions
                msg.velocity = [0.0] * len(names)
                msg.effort = []
                self.publisher.publish(msg)

            def handle_go_home(self, request, response):
                request_id = str(uuid.uuid4())
                target = (
                    list(request.target_angles_deg)
                    if len(request.target_angles_deg) == 6
                    else list(STANDARD_TOP_GRASP_HOME_JOINT_DEG)
                )
                try:
                    command = build_home_command(
                        request_id=request_id,
                        execute=bool(request.execute),
                        operator_phrase=str(request.operator_phrase),
                        target_angles_deg=target,
                        speed=float(request.speed or 3.0),
                        max_delta_deg=float(request.max_delta_deg or 90.0),
                        staged_home=bool(request.staged_home),
                        open_gripper=bool(request.open_gripper),
                        gripper_speed=float(request.gripper_speed or 20.0),
                    )
                    result = self.session.request(command, timeout_s=45.0)
                    response.success = bool(result.get("ok"))
                    response.status = str(result.get("status", "unknown"))
                    response.message = str(result.get("message", ""))
                    response.result_json = json.dumps(result.get("result", {}), sort_keys=True)
                except Exception as exc:
                    response.success = False
                    response.status = "local_exception"
                    response.message = str(exc)
                    response.result_json = "{}"
                return response

            def handle_guarded_taught_grasp(self, request, response):
                request_id = str(uuid.uuid4())
                try:
                    command = build_guarded_taught_grasp_command(
                        request_id=request_id,
                        execute=bool(request.execute),
                        operator_phrase=str(request.operator_phrase),
                        pregrasp_coords_mm_deg=list(request.pregrasp_coords_mm_deg),
                        grasp_coords_mm_deg=list(request.grasp_coords_mm_deg),
                        lift_coords_mm_deg=list(request.lift_coords_mm_deg),
                        speed=float(request.speed or 3.0),
                        gripper_speed=float(request.gripper_speed or 20.0),
                        min_z_mm=float(request.min_z_mm or 120.0),
                    )
                    result = self.session.request(command, timeout_s=60.0)
                    response.success = bool(result.get("ok"))
                    response.status = str(result.get("status", "unknown"))
                    response.message = str(result.get("message", ""))
                    response.result_json = json.dumps(result.get("result", {}), sort_keys=True)
                except Exception as exc:
                    response.success = False
                    response.status = "local_exception"
                    response.message = str(exc)
                    response.result_json = "{}"
                return response

            def destroy_node(self) -> bool:
                self.session.stop()
                return super().destroy_node()

        self.rclpy = rclpy
        self.node = ControlNode()

    def spin(self) -> None:
        self.rclpy.spin(self.node)

    def shutdown(self) -> None:
        self.node.destroy_node()
        safe_rclpy_shutdown(self.rclpy)


def spin_server_until_shutdown(server) -> None:
    try:
        server.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.shutdown()
        except KeyboardInterrupt:
            pass


def main() -> None:
    import rclpy

    rclpy.init()
    server = M7MyCobotControlServer()
    spin_server_until_shutdown(server)


if __name__ == "__main__":
    main()
