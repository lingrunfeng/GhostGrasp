#!/usr/bin/env python3
"""Publish real/M4 perception-ranked hypotheses as the common geometry contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3
from ghost_mgg_interfaces.msg import (
    GeometryHypothesis,
    GeometryHypothesisArray,
    GraspCandidate,
    ScoreBreakdown,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


DEFAULT_JOINT_REPORT_PATH = "reports/m4_joint_hypotheses/joint_hypotheses.json"
DEFAULT_METRIC_PROXY_REPORT_PATH = "reports/m4_metric_proxies/metric_proxies.json"
DEFAULT_GRASPABILITY_REPORT_PATH = "reports/m4_graspability_dryrun/graspability.json"
DEFAULT_HYPOTHESIS_TOPIC = "/ghost_mgg/m4_perception_hypotheses"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clamp01(value: Any, fallback: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(numeric):
        return fallback
    return min(1.0, max(0.0, numeric))


def shape_constant(shape_type: str) -> int:
    if shape_type == "box":
        return GeometryHypothesis.SHAPE_BOX
    if shape_type == "cylinder":
        return GeometryHypothesis.SHAPE_CYLINDER
    if shape_type == "cup_like":
        return GeometryHypothesis.SHAPE_CUP_LIKE
    return GeometryHypothesis.SHAPE_UNKNOWN


def top_down_orientation(yaw_rad: float = 0.0) -> Quaternion:
    half_yaw = 0.5 * yaw_rad
    yaw_sin = math.sin(half_yaw)
    yaw_cos = math.cos(half_yaw)
    roll_sin = -math.sqrt(0.5)
    roll_cos = math.sqrt(0.5)
    orientation = Quaternion()
    orientation.x = yaw_cos * roll_sin
    orientation.y = yaw_sin * roll_sin
    orientation.z = yaw_sin * roll_cos
    orientation.w = yaw_cos * roll_cos
    return orientation


def make_pose(
    frame_id: str,
    x: float,
    y: float,
    z: float,
    orientation: Quaternion | None = None,
) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = float(z)
    pose.pose.orientation = orientation or Quaternion(w=1.0)
    return pose


def make_dimensions(metric_row: dict[str, Any]) -> Vector3:
    dimensions = Vector3()
    dimensions.x = float(metric_row.get("width_m", 0.0))
    dimensions.y = float(metric_row.get("depth_m", 0.0))
    dimensions.z = float(metric_row.get("height_m", 0.0))
    return dimensions


def make_score(joint_row: dict[str, Any], grasp_row: dict[str, Any]) -> ScoreBreakdown:
    score = ScoreBreakdown()
    score.visual = float(joint_row.get("visual_score") or 0.0)
    score.failure = float(joint_row.get("failure_score") or 0.0)
    score.depth = 1.0 if bool(grasp_row.get("valid")) else 0.0
    score.physical = score.depth
    score.grasp = float(joint_row.get("grasp_score") or grasp_row.get("score") or 0.0)
    score.prior = 0.50
    score.total = float(joint_row.get("joint_score") or 0.0)
    return score


def validation_state(joint_row: dict[str, Any], grasp_row: dict[str, Any]) -> int:
    decision = str(joint_row.get("decision", ""))
    if decision in {"candidate", "executable"} and bool(grasp_row.get("valid", True)):
        return GeometryHypothesis.VALIDATION_VALID
    if decision == "reject" or not bool(grasp_row.get("valid", True)):
        return GeometryHypothesis.VALIDATION_REJECTED
    return GeometryHypothesis.VALIDATION_UNKNOWN


def grasp_validation_state(joint_row: dict[str, Any], grasp_row: dict[str, Any]) -> int:
    if validation_state(joint_row, grasp_row) == GeometryHypothesis.VALIDATION_VALID:
        return GraspCandidate.VALIDATION_VALID
    if validation_state(joint_row, grasp_row) == GeometryHypothesis.VALIDATION_REJECTED:
        return GraspCandidate.VALIDATION_REJECTED
    return GraspCandidate.VALIDATION_UNKNOWN


def make_grasp_candidate(
    joint_row: dict[str, Any],
    grasp_row: dict[str, Any],
    frame_id: str,
) -> GraspCandidate:
    orientation = top_down_orientation()
    grasp = GraspCandidate()
    grasp.grasp_id = str(grasp_row.get("grasp_id") or f"{joint_row['hypothesis_id']}_top")
    grasp.grasp_pose = make_pose(
        frame_id,
        float(grasp_row.get("grasp_x_m", 0.0)),
        float(grasp_row.get("grasp_y_m", 0.0)),
        float(grasp_row.get("grasp_z_m", 0.0)),
        orientation,
    )
    grasp.pregrasp_pose = make_pose(
        frame_id,
        float(grasp_row.get("grasp_x_m", 0.0)),
        float(grasp_row.get("grasp_y_m", 0.0)),
        float(grasp_row.get("pregrasp_z_m", grasp_row.get("grasp_z_m", 0.0))),
        orientation,
    )
    grasp.approach_vector.x = float(grasp_row.get("approach_x", 0.0))
    grasp.approach_vector.y = float(grasp_row.get("approach_y", 0.0))
    grasp.approach_vector.z = float(grasp_row.get("approach_z", -1.0))
    grasp.gripper_width_m = float(grasp_row.get("required_gripper_width_m", 0.0))
    grasp.grasp_type = GraspCandidate.GRASP_TYPE_TOP
    grasp.score = float(grasp_row.get("score") or joint_row.get("joint_score") or 0.0)
    grasp.validation_state = grasp_validation_state(joint_row, grasp_row)
    grasp.failure_reason = str(grasp_row.get("failure_reason") or joint_row.get("failure_reason") or "")
    return grasp


def make_hypothesis(
    joint_row: dict[str, Any],
    metric_row: dict[str, Any],
    grasp_row: dict[str, Any],
    frame_id: str,
) -> GeometryHypothesis:
    hypothesis = GeometryHypothesis()
    hypothesis.hypothesis_id = str(joint_row["hypothesis_id"])
    hypothesis.shape_type = shape_constant(str(joint_row.get("shape_type") or metric_row.get("shape_type")))
    hypothesis.pose_camera = make_pose(
        frame_id,
        float(metric_row["center_x_m"]),
        float(metric_row["center_y_m"]),
        float(metric_row["center_z_m"]),
    )
    hypothesis.pose_base = make_pose(
        frame_id,
        float(metric_row["center_x_m"]),
        float(metric_row["center_y_m"]),
        float(metric_row["center_z_m"]),
    )
    hypothesis.dimensions_m = make_dimensions(metric_row)
    hypothesis.score = make_score(joint_row, grasp_row)
    hypothesis.confidence = clamp01(joint_row.get("joint_score"), 0.0)
    hypothesis.uncertainty = 1.0 - hypothesis.confidence
    hypothesis.grasp_candidates = [make_grasp_candidate(joint_row, grasp_row, frame_id)]
    scene_id = str(joint_row.get("target_or_scene_id", "unknown"))
    hypothesis.provenance = (
        f"m4_perception:{scene_id}:{joint_row.get('ranker', 'unknown')}:"
        f"rank={joint_row.get('joint_rank', 0)}"
    )
    hypothesis.validation_state = validation_state(joint_row, grasp_row)
    hypothesis.failure_reason = str(grasp_row.get("failure_reason") or joint_row.get("failure_reason") or "")
    return hypothesis


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    indexed = {}
    for row in rows:
        key = (
            str(row.get("scene_id")),
            str(row.get("ranker")),
            str(row.get("hypothesis_id")),
        )
        indexed[key] = row
    return indexed


def build_perception_hypotheses(
    joint_report: dict[str, Any],
    metric_report: dict[str, Any],
    graspability_report: dict[str, Any],
    *,
    scene_id: str,
    frame_id: str = "world",
    include_rejected: bool = False,
) -> list[GeometryHypothesis]:
    metric_rows = _index_rows(list(metric_report.get("rows", [])))
    grasp_rows = _index_rows(list(graspability_report.get("rows", [])))
    selected_rows = [
        row
        for row in joint_report.get("rows", [])
        if row.get("source_type") == "real_graspability"
        and str(row.get("target_or_scene_id")) == str(scene_id)
        and (include_rejected or str(row.get("decision")) != "reject")
    ]
    selected_rows.sort(key=lambda row: int(row.get("joint_rank") or 9999))

    hypotheses = []
    for joint_row in selected_rows:
        key = (
            str(joint_row.get("target_or_scene_id")),
            str(joint_row.get("ranker")),
            str(joint_row.get("hypothesis_id")),
        )
        metric_row = metric_rows.get(key)
        grasp_row = grasp_rows.get(key)
        if metric_row is None or grasp_row is None:
            continue
        hypotheses.append(make_hypothesis(joint_row, metric_row, grasp_row, frame_id))
    return hypotheses


def build_hypothesis_array(
    joint_report: dict[str, Any],
    metric_report: dict[str, Any],
    graspability_report: dict[str, Any],
    *,
    scene_id: str,
    frame_id: str = "world",
    trial_id: str = "m4_perception",
    observation_id: str = "m4_perception_hypotheses",
    backend_name: str = "ghost_mgg_m4_perception",
    include_rejected: bool = False,
) -> GeometryHypothesisArray:
    message = GeometryHypothesisArray()
    message.header.frame_id = frame_id
    message.trial_id = trial_id
    message.observation_id = observation_id
    message.backend_name = backend_name
    message.hypotheses = build_perception_hypotheses(
        joint_report,
        metric_report,
        graspability_report,
        scene_id=scene_id,
        frame_id=frame_id,
        include_rejected=include_rejected,
    )
    return message


class M4PerceptionHypothesisPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("m4_perception_hypothesis_publisher_node")
        self.joint_report_path = self.declare_parameter(
            "joint_report_path", DEFAULT_JOINT_REPORT_PATH
        ).value
        self.metric_proxy_report_path = self.declare_parameter(
            "metric_proxy_report_path", DEFAULT_METRIC_PROXY_REPORT_PATH
        ).value
        self.graspability_report_path = self.declare_parameter(
            "graspability_report_path", DEFAULT_GRASPABILITY_REPORT_PATH
        ).value
        self.hypothesis_topic = self.declare_parameter(
            "hypothesis_topic", DEFAULT_HYPOTHESIS_TOPIC
        ).value
        self.scene_id = self.declare_parameter(
            "scene_id", "daylight_transparent_jelly_cup_001"
        ).value
        self.frame_id = self.declare_parameter("frame_id", "world").value
        self.trial_id = self.declare_parameter("trial_id", "m4_perception").value
        self.observation_id = self.declare_parameter(
            "observation_id", "m4_perception_hypotheses"
        ).value
        self.backend_name = self.declare_parameter(
            "backend_name", "ghost_mgg_m4_perception"
        ).value
        self.include_rejected = bool(self.declare_parameter("include_rejected", False).value)

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.publisher = self.create_publisher(
            GeometryHypothesisArray, self.hypothesis_topic, qos
        )
        self.timer = self.create_timer(0.5, self.publish_hypotheses)
        self.reported_success = False
        self.reported_missing_inputs = False

    def publish_hypotheses(self) -> None:
        try:
            message = build_hypothesis_array(
                load_json(self.joint_report_path),
                load_json(self.metric_proxy_report_path),
                load_json(self.graspability_report_path),
                scene_id=self.scene_id,
                frame_id=self.frame_id,
                trial_id=self.trial_id,
                observation_id=self.observation_id,
                backend_name=self.backend_name,
                include_rejected=self.include_rejected,
            )
        except Exception as error:
            if not self.reported_missing_inputs:
                self.get_logger().warn(f"Cannot read M4 perception inputs yet: {error}")
                self.reported_missing_inputs = True
            return

        message.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(message)
        if not self.reported_success:
            self.get_logger().info(
                "Published "
                f"{len(message.hypotheses)} M4 perception hypotheses for {self.scene_id} "
                f"to {self.hypothesis_topic}"
            )
            self.reported_success = True


def main() -> None:
    rclpy.init()
    node = M4PerceptionHypothesisPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
