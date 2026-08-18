#!/usr/bin/env python3
"""M8 live hypothesis update-rate benchmark.

This script is a benchmark harness, not part of the live perception algorithm.
It may command Gazebo model poses and compare live hypotheses against commanded
poses to measure how quickly the no-truth live pipeline reacts.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import rclpy
from ghost_mgg_interfaces.msg import GeometryHypothesisArray


DEFAULT_POSES = (
    (0.035, 0.060, math.radians(45.0)),
    (0.070172, 0.012156, 0.0),
    (0.035, 0.060, math.radians(45.0)),
)
DEFAULT_INITIAL_POSE = (0.070172, 0.012156, 0.0)


@dataclass(frozen=True)
class PoseTarget:
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class UpdateTrialResult:
    trial_index: int
    target_xy_m: tuple[float, float]
    target_yaw_rad: float
    success: bool
    latency_sec: float | None
    first_changed_hypothesis: str | None
    first_changed_center_xy_m: tuple[float, float] | None
    center_error_m: float | None
    message_count_after_command: int


def parse_pose_list(value: str) -> list[PoseTarget]:
    poses: list[PoseTarget] = []
    for raw_item in str(value).split(";"):
        item = raw_item.strip()
        if not item:
            continue
        parts = [float(part.strip()) for part in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"expected x,y,yaw_rad pose item, got {raw_item!r}")
        poses.append(PoseTarget(x_m=parts[0], y_m=parts[1], yaw_rad=parts[2]))
    if not poses:
        raise ValueError("at least one pose is required")
    return poses


def parse_pose(value: str) -> PoseTarget:
    poses = parse_pose_list(value)
    if len(poses) != 1:
        raise ValueError(f"expected exactly one pose, got {len(poses)}")
    return poses[0]


def set_model_pose(
    *,
    world_name: str,
    model_name: str,
    target: PoseTarget,
    target_z_m: float,
    timeout_ms: int,
) -> None:
    qz = math.sin(0.5 * float(target.yaw_rad))
    qw = math.cos(0.5 * float(target.yaw_rad))
    request = (
        f'name: "{model_name}" '
        f"position {{ x: {target.x_m} y: {target.y_m} z: {target_z_m} }} "
        f"orientation {{ z: {qz} w: {qw} }}"
    )
    subprocess.run(
        [
            "gz",
            "service",
            "-s",
            f"/world/{world_name}/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            str(int(timeout_ms)),
            "--req",
            request,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def hypothesis_center_xy(hypothesis) -> tuple[float, float]:
    return (
        float(hypothesis.pose_base.pose.position.x),
        float(hypothesis.pose_base.pose.position.y),
    )


def distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def message_stamp_ns(msg: GeometryHypothesisArray) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


class UpdateRateProbe:
    def __init__(self, topic: str) -> None:
        rclpy.init()
        self.node = rclpy.create_node("m8_update_rate_benchmark")
        self.messages: list[tuple[float, GeometryHypothesisArray]] = []
        self.subscription = self.node.create_subscription(
            GeometryHypothesisArray,
            topic,
            self._handle_message,
            10,
        )

    def _handle_message(self, msg: GeometryHypothesisArray) -> None:
        self.messages.append((time.monotonic(), msg))
        if len(self.messages) > 200:
            self.messages = self.messages[-200:]

    def spin_until_message(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.messages:
                return True
        return False

    def wait_for_target(
        self,
        *,
        target: PoseTarget,
        command_time: float,
        timeout_sec: float,
        center_threshold_m: float,
        min_header_stamp_ns: int | None = None,
        start_message_index: int = 0,
    ) -> tuple[bool, float | None, str | None, tuple[float, float] | None, float | None, int]:
        target_xy = (float(target.x_m), float(target.y_m))
        seen_after_command = 0
        processed_messages = max(0, int(start_message_index))
        nearest_hypothesis_id: str | None = None
        nearest_center_xy: tuple[float, float] | None = None
        nearest_error: float | None = None
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            for stamp, msg in list(self.messages)[processed_messages:]:
                if stamp < command_time:
                    continue
                if min_header_stamp_ns is not None and message_stamp_ns(msg) <= int(
                    min_header_stamp_ns
                ):
                    continue
                seen_after_command += 1
                for hypothesis in msg.hypotheses:
                    center = hypothesis_center_xy(hypothesis)
                    error = distance_xy(center, target_xy)
                    if nearest_error is None or error < nearest_error:
                        nearest_hypothesis_id = str(hypothesis.hypothesis_id)
                        nearest_center_xy = center
                        nearest_error = error
                    if error <= float(center_threshold_m):
                        return (
                            True,
                            max(0.0, stamp - command_time),
                            str(hypothesis.hypothesis_id),
                            center,
                            error,
                            seen_after_command,
                        )
            processed_messages = len(self.messages)
        return False, None, nearest_hypothesis_id, nearest_center_xy, nearest_error, seen_after_command

    def drain(self, duration_sec: float = 0.25) -> None:
        deadline = time.monotonic() + float(duration_sec)
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def close(self) -> None:
        self.node.destroy_node()
        rclpy.shutdown()

    def latest_header_stamp_ns(self) -> int | None:
        if not self.messages:
            return None
        return message_stamp_ns(self.messages[-1][1])


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return float("inf")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * float(percent) / 100.0
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_report(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = payload["summary"]
    lines = [
        "# M8 Update-Rate Benchmark",
        "",
        f"- gate_status: `{summary['gate_status']}`",
        f"- success_count: `{summary['success_count']}/{summary['trial_count']}`",
        f"- median_latency_sec: `{summary['median_latency_sec']:.3f}`",
        f"- p95_latency_sec: `{summary['p95_latency_sec']:.3f}`",
        f"- max_latency_sec: `{summary['max_latency_sec']:.3f}`",
        "",
        "## Trials",
        "",
    ]
    for trial in payload["trials"]:
        lines.append(
            "- trial {trial_index}: {status} latency={latency} center_error_m={error} "
            "hypothesis={hypothesis}".format(
                trial_index=trial["trial_index"],
                status="pass" if trial["success"] else "fail",
                latency=(
                    f"{float(trial['latency_sec']):.3f}"
                    if trial["latency_sec"] is not None
                    else "None"
                ),
                error=(
                    f"{float(trial['center_error_m']):.4f}"
                    if trial["center_error_m"] is not None
                    else "None"
                ),
                hypothesis=trial["first_changed_hypothesis"],
            )
        )
    lines.append("")
    (output_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_pose_text = ";".join(f"{x},{y},{yaw}" for x, y, yaw in DEFAULT_POSES)
    default_initial_pose_text = ",".join(str(value) for value in DEFAULT_INITIAL_POSE)
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/ghost_mgg/m4_live_hypotheses")
    parser.add_argument("--world-name", default="ghost_mgg_m2_visual")
    parser.add_argument("--model", default="red_cube")
    parser.add_argument("--target-z-m", type=float, default=0.7525)
    parser.add_argument("--initial-pose", default=default_initial_pose_text)
    parser.add_argument("--poses", default=default_pose_text)
    parser.add_argument("--center-threshold-m", type=float, default=0.012)
    parser.add_argument("--warmup-timeout-sec", type=float, default=20.0)
    parser.add_argument("--prepare-timeout-sec", type=float, default=10.0)
    parser.add_argument("--trial-timeout-sec", type=float, default=3.0)
    parser.add_argument("--gz-timeout-ms", type=int, default=4000)
    parser.add_argument("--median-threshold-sec", type=float, default=0.5)
    parser.add_argument("--p95-threshold-sec", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m8_update_rate_benchmark"))
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    poses = parse_pose_list(str(args.poses))
    initial_pose = parse_pose(str(args.initial_pose))
    probe = UpdateRateProbe(str(args.topic))
    try:
        if not probe.spin_until_message(float(args.warmup_timeout_sec)):
            print(f"no hypotheses received on {args.topic}", file=sys.stderr)
            return 2
        probe.drain(0.3)
        prepare_start_index = len(probe.messages)
        set_model_pose(
            world_name=str(args.world_name),
            model_name=str(args.model),
            target=initial_pose,
            target_z_m=float(args.target_z_m),
            timeout_ms=int(args.gz_timeout_ms),
        )
        prepare_command_time = time.monotonic()
        prepared, _, prepared_hypothesis, prepared_center, prepared_error, _ = probe.wait_for_target(
            target=initial_pose,
            command_time=prepare_command_time,
            timeout_sec=float(args.prepare_timeout_sec),
            center_threshold_m=float(args.center_threshold_m),
            start_message_index=prepare_start_index,
        )
        if not prepared:
            print(
                "initial pose was not observed before benchmark: "
                f"nearest={prepared_hypothesis} center={prepared_center} error={prepared_error}",
                file=sys.stderr,
            )
            return 2
        trials: list[UpdateTrialResult] = []
        for index, target in enumerate(poses, start=1):
            probe.drain(0.3)
            start_message_index = len(probe.messages)
            set_model_pose(
                world_name=str(args.world_name),
                model_name=str(args.model),
                target=target,
                target_z_m=float(args.target_z_m),
                timeout_ms=int(args.gz_timeout_ms),
            )
            command_time = time.monotonic()
            result = probe.wait_for_target(
                target=target,
                command_time=command_time,
                timeout_sec=float(args.trial_timeout_sec),
                center_threshold_m=float(args.center_threshold_m),
                start_message_index=start_message_index,
            )
            success, latency, hypothesis_id, center_xy, center_error, message_count = result
            trials.append(
                UpdateTrialResult(
                    trial_index=index,
                    target_xy_m=(float(target.x_m), float(target.y_m)),
                    target_yaw_rad=float(target.yaw_rad),
                    success=bool(success),
                    latency_sec=latency,
                    first_changed_hypothesis=hypothesis_id,
                    first_changed_center_xy_m=center_xy,
                    center_error_m=center_error,
                    message_count_after_command=int(message_count),
                )
            )
    finally:
        probe.close()

    latencies = [float(trial.latency_sec) for trial in trials if trial.success and trial.latency_sec is not None]
    success_count = sum(1 for trial in trials if trial.success)
    median_latency = statistics.median(latencies) if latencies else float("inf")
    p95_latency = percentile(latencies, 95.0)
    max_latency = max(latencies) if latencies else float("inf")
    gate_status = (
        "pass"
        if success_count == len(trials)
        and median_latency <= float(args.median_threshold_sec)
        and p95_latency <= float(args.p95_threshold_sec)
        else "fail"
    )
    payload = {
        "summary": {
            "gate_status": gate_status,
            "trial_count": len(trials),
            "success_count": success_count,
            "median_latency_sec": median_latency,
            "p95_latency_sec": p95_latency,
            "max_latency_sec": max_latency,
            "center_threshold_m": float(args.center_threshold_m),
            "median_threshold_sec": float(args.median_threshold_sec),
            "p95_threshold_sec": float(args.p95_threshold_sec),
        },
        "trials": [asdict(trial) for trial in trials],
    }
    write_report(Path(args.output_dir), payload)
    print(
        "M8 update-rate benchmark: {gate_status} "
        "success={success_count}/{trial_count} median={median:.3f}s p95={p95:.3f}s "
        "-> {output_dir}".format(
            gate_status=gate_status,
            success_count=success_count,
            trial_count=len(trials),
            median=median_latency,
            p95=p95_latency,
            output_dir=args.output_dir,
        )
    )
    return 1 if args.fail_on_gate and gate_status != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
