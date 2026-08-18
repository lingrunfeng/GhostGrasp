#!/usr/bin/env python3
"""Strict sim-only M8 geometry benchmark.

This script is intentionally outside the live geometry path. It reads Gazebo
truth only to score the current /ghost_mgg/m4_live_hypotheses output.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import rclpy
from ghost_mgg_interfaces.msg import GeometryHypothesis, GeometryHypothesisArray


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CORE_PYTHON = REPO_ROOT / "src" / "ghost_mgg_core" / "python"
if str(CORE_PYTHON) not in sys.path:
    sys.path.insert(0, str(CORE_PYTHON))

from ghost_mgg_core_py.evaluation.m8_live_tabletop_eval import (
    M8HypothesisGeometry,
    M8TruthGeometry,
    evaluate_strict_geometry_snapshot,
    summarize_strict_geometry_rows,
)


DEFAULT_STRICT_MODELS = (
    "red_cube",
    "glass_block",
    "blue_cylinder",
    "green_cylinder",
)

MODEL_DIMS_M = {
    "red_cube": (0.025, 0.025),
    "glass_block": (0.025, 0.025),
    "blue_cylinder": (0.025, 0.025),
    "green_cylinder": (0.025, 0.025),
    "m8_small_cube": (0.018, 0.018),
    "m8_medium_cube": (0.030, 0.030),
    "m8_large_cube": (0.045, 0.045),
    "m8_rect_box_2x1": (0.050, 0.025),
    "m8_rect_box_3x1": (0.070, 0.024),
    "m8_short_cylinder": (0.028, 0.028),
    "m8_tall_cylinder": (0.024, 0.024),
    "m8_tri_prism": (0.036, 0.031),
    "m8_hex_prism": (0.036, 0.031),
}

MODEL_SHAPES = {
    "red_cube": "box",
    "glass_block": "box",
    "blue_cylinder": "cylinder",
    "green_cylinder": "cylinder",
    "m8_small_cube": "box",
    "m8_medium_cube": "box",
    "m8_large_cube": "box",
    "m8_rect_box_2x1": "box",
    "m8_rect_box_3x1": "box",
    "m8_short_cylinder": "cylinder",
    "m8_tall_cylinder": "cylinder",
    "m8_tri_prism": "box",
    "m8_hex_prism": "box",
}

MESSAGE_SHAPES = {
    GeometryHypothesis.SHAPE_UNKNOWN: "unknown",
    GeometryHypothesis.SHAPE_BOX: "box",
    GeometryHypothesis.SHAPE_CYLINDER: "cylinder",
    GeometryHypothesis.SHAPE_CUP_LIKE: "cup",
}


def normalize_half_turn(angle_rad: float) -> float:
    value = float(angle_rad)
    while value <= -0.5 * math.pi:
        value += math.pi
    while value > 0.5 * math.pi:
        value -= math.pi
    return value


def yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z)),
    )


def parse_gz_pose(model_name: str, output: str) -> M8TruthGeometry | None:
    match = re.search(
        r"Pose \[ XYZ \(m\) \] \[ RPY \(rad\) \]:\s*"
        r"\[([^\]]+)\]\s*\[([^\]]+)\]",
        output,
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    xyz = tuple(float(value) for value in match.group(1).split())
    rpy = tuple(float(value) for value in match.group(2).split())
    if len(xyz) != 3 or len(rpy) != 3:
        return None
    if model_name not in MODEL_DIMS_M:
        return None
    return M8TruthGeometry(
        object_id=model_name,
        shape_type=MODEL_SHAPES[model_name],
        center_xy_m=(float(xyz[0]), float(xyz[1])),
        size_xy_m=MODEL_DIMS_M[model_name],
        yaw_rad=normalize_half_turn(float(rpy[2])),
    )


def query_truth(models: tuple[str, ...], timeout_sec: float) -> list[M8TruthGeometry]:
    truth: list[M8TruthGeometry] = []
    for model_name in models:
        completed = subprocess.run(
            ["gz", "model", "-m", model_name, "-p"],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_sec),
        )
        if completed.returncode != 0:
            continue
        parsed = parse_gz_pose(model_name, completed.stdout)
        if parsed is not None:
            truth.append(parsed)
    return truth


def receive_hypotheses(
    topic: str,
    timeout_sec: float,
    *,
    sample_window_sec: float,
) -> list[M8HypothesisGeometry]:
    rows: list[M8HypothesisGeometry] = []
    rclpy.init()
    node = rclpy.create_node("m8_strict_geometry_benchmark")
    first_message_time = [None]

    def callback(msg: GeometryHypothesisArray) -> None:
        rows.clear()
        for hypothesis in msg.hypotheses:
            pose = hypothesis.pose_base.pose
            dims = hypothesis.dimensions_m
            rows.append(
                M8HypothesisGeometry(
                    hypothesis_id=str(hypothesis.hypothesis_id),
                    shape_type=MESSAGE_SHAPES.get(
                        int(hypothesis.shape_type),
                        str(int(hypothesis.shape_type)),
                    ),
                    center_xy_m=(float(pose.position.x), float(pose.position.y)),
                    size_xy_m=(float(dims.x), float(dims.y)),
                    yaw_rad=normalize_half_turn(yaw_from_quaternion(pose.orientation)),
                    provenance=str(hypothesis.provenance),
                    score_total=float(hypothesis.score.total),
                    score_visual=float(hypothesis.score.visual),
                    score_failure=float(hypothesis.score.failure),
                    score_depth=float(hypothesis.score.depth),
                    score_prior=float(hypothesis.score.prior),
                )
            )
        if first_message_time[0] is None:
            first_message_time[0] = time.monotonic()

    node.create_subscription(GeometryHypothesisArray, topic, callback, 10)
    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if first_message_time[0] is not None:
            if time.monotonic() - float(first_message_time[0]) >= float(sample_window_sec):
                break
    node.destroy_node()
    rclpy.shutdown()
    return rows


def write_report(
    *,
    output_dir: Path,
    truths: list[M8TruthGeometry],
    hypotheses: list[M8HypothesisGeometry],
) -> dict:
    rows = evaluate_strict_geometry_snapshot(truths=truths, hypotheses=hypotheses)
    summary = summarize_strict_geometry_rows(rows)
    payload = {
        "schema_version": "m8_strict_geometry_benchmark_v1",
        "summary": summary,
        "truths": [asdict(item) for item in truths],
        "hypotheses": [asdict(item) for item in hypotheses],
        "rows": [asdict(item) for item in rows],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.md").write_text(markdown_report(payload), encoding="utf-8")
    return payload


def markdown_report(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# M8 Strict Geometry Benchmark",
        "",
        "- Gazebo truth is used only by this benchmark, never by the live algorithm.",
        f"- gate_status: `{summary['gate_status']}`",
        f"- pass_rate: `{summary['pass_rate']:.3f}`",
        f"- max_center_error_m: `{summary['max_center_error_m']:.4f}`",
        f"- max_size_error_m: `{summary['max_size_error_m']:.4f}`",
        f"- max_yaw_error_rad: `{summary['max_yaw_error_rad']:.4f}`",
        "",
        "| object | status | match | shape | center mm | size mm | yaw deg | no-truth |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {object_id} | {status} | {matched_hypothesis_id} | "
            "{truth_shape_type}/{hypothesis_shape_type} | {center_mm:.1f} | "
            "{size_mm:.1f} | {yaw_deg:.1f} | {no_truth} |".format(
                object_id=row["object_id"],
                status=row["status"],
                matched_hypothesis_id=row["matched_hypothesis_id"] or "-",
                truth_shape_type=row["truth_shape_type"],
                hypothesis_shape_type=row["hypothesis_shape_type"] or "-",
                center_mm=1000.0 * float(row["center_error_m"]),
                size_mm=1000.0 * float(row["size_error_m"]),
                yaw_deg=math.degrees(float(row["yaw_error_rad"])),
                no_truth="pass" if row["no_truth_audit_pass"] else "fail",
            )
        )
    lines.extend(["", "## Hypothesis Provenance", ""])
    for hypothesis in payload["hypotheses"]:
        lines.append(
            "- {hypothesis_id}: total={total:.3f} visual={visual:.3f} "
            "failure={failure:.3f} depth={depth:.3f} prior={prior:.3f} `{provenance}`".format(
                hypothesis_id=hypothesis["hypothesis_id"],
                total=float(hypothesis.get("score_total", 0.0)),
                visual=float(hypothesis.get("score_visual", 0.0)),
                failure=float(hypothesis.get("score_failure", 0.0)),
                depth=float(hypothesis.get("score_depth", 0.0)),
                prior=float(hypothesis.get("score_prior", 0.0)),
                provenance=hypothesis["provenance"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict M8 live geometry benchmark.")
    parser.add_argument("--topic", default="/ghost_mgg/m4_live_hypotheses")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--gz-timeout-sec", type=float, default=5.0)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_STRICT_MODELS))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m8_geometry_benchmark_strict"))
    parser.add_argument("--sample-window-sec", type=float, default=1.0)
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hypotheses = receive_hypotheses(
        str(args.topic),
        float(args.timeout_sec),
        sample_window_sec=float(args.sample_window_sec),
    )
    truths = query_truth(tuple(str(model) for model in args.models), float(args.gz_timeout_sec))
    if not hypotheses:
        print(f"no hypotheses received on {args.topic}", file=sys.stderr)
        return 2
    if not truths:
        print("no Gazebo truth models were available", file=sys.stderr)
        return 2
    payload = write_report(output_dir=args.output_dir, truths=truths, hypotheses=hypotheses)
    status = payload["summary"]["gate_status"]
    print(f"M8 strict geometry benchmark: {status} -> {args.output_dir}")
    for row in payload["rows"]:
        print(
            "{object_id}: {status} match={matched_hypothesis_id} "
            "shape={truth_shape_type}/{hypothesis_shape_type} "
            "center_mm={center_mm:.1f} size_mm={size_mm:.1f} yaw_deg={yaw_deg:.1f}".format(
                object_id=row["object_id"],
                status=row["status"],
                matched_hypothesis_id=row["matched_hypothesis_id"] or "-",
                truth_shape_type=row["truth_shape_type"],
                hypothesis_shape_type=row["hypothesis_shape_type"] or "-",
                center_mm=1000.0 * float(row["center_error_m"]),
                size_mm=1000.0 * float(row["size_error_m"]),
                yaw_deg=math.degrees(float(row["yaw_error_rad"])),
            )
        )
    if bool(args.fail_on_gate) and status != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
