#!/usr/bin/env python3
"""M6 read-only myCobot280Pi joint-state bridge.

This node is intentionally state-only. It reads real joint angles through SSH and
publishes local /joint_states for MoveIt shadow planning. It must not expose or
call any real motion command.
"""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import threading
from typing import Iterable


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


def parse_remote_angle_line(line: str) -> list[float] | None:
    """Parse one remote stdout line into six joint angles in degrees."""
    text = line.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    values = payload.get("angles") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or len(values) != 6:
        return None

    parsed: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        parsed.append(float(value))
    return parsed


def degrees_to_ghost_joint_state(angles_deg: Iterable[float]) -> tuple[list[str], list[float]]:
    """Map six myCobot degree values into Ghost-MGG MoveIt joint-state fields."""
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
    """Append an open shadow gripper state for MoveIt completeness."""
    value = float(gripper_position)
    shadow_positions = [value, value, -value, -value, -value, value]
    return names + list(SHADOW_GRIPPER_JOINT_NAMES), positions + shadow_positions


def build_remote_reader_script(serial_port: str, baud: int, sample_hz: float) -> str:
    period_s = 1.0 / max(float(sample_hz), 0.1)
    return f"""
import json
import sys
import time

from pymycobot.mycobot280 import MyCobot280

mc = MyCobot280({serial_port!r}, {int(baud)})
period_s = {period_s!r}

while True:
    try:
        angles = mc.get_angles()
        if isinstance(angles, list) and len(angles) == 6:
            print(json.dumps({{"angles": [float(value) for value in angles]}}), flush=True)
    except Exception as exc:
        print(json.dumps({{"error": str(exc)}}), file=sys.stderr, flush=True)
    time.sleep(period_s)
""".strip()


def build_ssh_command(
    *,
    host: str,
    user: str,
    remote_script: str,
    connect_timeout_s: int,
    ssh_command: str = "ssh",
) -> list[str]:
    """Build an SSH command without embedding credentials."""
    remote_command = "python3 -u - <<'GHOST_MGG_REMOTE_PY'\n"
    remote_command += remote_script
    remote_command += "\nGHOST_MGG_REMOTE_PY"
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


def safe_rclpy_shutdown(rclpy_module) -> None:
    """Shutdown rclpy without failing if ROS already handled SIGINT shutdown."""
    try:
        rclpy_module.shutdown()
    except Exception as exc:
        text = str(exc)
        if "rcl_shutdown already called" in text or "context is zero initialized" in text:
            return
        raise


class M6SshJointStateBridge:
    def __init__(self) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        class BridgeNode(Node):
            def __init__(self) -> None:
                super().__init__("m6_ssh_joint_state_bridge")
                self.declare_parameter("robot_host", "10.42.0.169")
                self.declare_parameter("robot_user", "elephant")
                self.declare_parameter("serial_port", "/dev/ttyAMA0")
                self.declare_parameter("baud", 1000000)
                self.declare_parameter("sample_hz", 10.0)
                self.declare_parameter("connect_timeout_s", 5)
                self.declare_parameter("frame_id", "")
                self.declare_parameter("ssh_command", "ssh")
                self.declare_parameter("publish_shadow_gripper_joints", True)
                self.declare_parameter("shadow_gripper_position", 0.15)

                self._joint_state_type = JointState
                self.publisher = self.create_publisher(JointState, "/joint_states", 10)
                self.process: subprocess.Popen[str] | None = None
                self.reader_thread: threading.Thread | None = None
                self.start_remote_reader()

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

            def start_remote_reader(self) -> None:
                remote_script = build_remote_reader_script(
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
                self.get_logger().info(
                    "starting read-only myCobot joint bridge via SSH to "
                    f"{self.get_string('robot_user')}@{self.get_string('robot_host')}"
                )
                self.process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=None,
                    text=True,
                    bufsize=1,
                )
                self.reader_thread = threading.Thread(target=self.read_stdout_loop, daemon=True)
                self.reader_thread.start()

            def read_stdout_loop(self) -> None:
                assert self.process is not None
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    angles = parse_remote_angle_line(line)
                    if angles is None:
                        continue
                    names, positions = degrees_to_ghost_joint_state(angles)
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

            def destroy_node(self) -> bool:
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                return super().destroy_node()

        self.rclpy = rclpy
        self.node = BridgeNode()

    def spin(self) -> None:
        self.rclpy.spin(self.node)

    def shutdown(self) -> None:
        self.node.destroy_node()
        safe_rclpy_shutdown(self.rclpy)


def main() -> None:
    import rclpy

    rclpy.init()
    bridge = M6SshJointStateBridge()
    spin_bridge_until_shutdown(bridge)


def spin_bridge_until_shutdown(bridge) -> None:
    try:
        bridge.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            bridge.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
