#!/usr/bin/env python3
import json
import math
from pathlib import Path

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


DEFAULT_REPORT_PATH = "reports/m4_joint_hypotheses/joint_hypotheses.json"
DEFAULT_TARGETS_PATH = "config/m4_sim_grasp_targets.json"
DEFAULT_HYPOTHESIS_TOPIC = "/ghost_mgg/m4_joint_hypotheses"
DEFAULT_TOP_GRASP_OFFSET_M = 0.005


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_targets(path: str | Path) -> dict[str, dict]:
    data = load_json(path)
    return {row["target_id"]: row for row in data.get("rows", [])}


def shape_constant(shape_type: str) -> int:
    if shape_type == "box":
        return GeometryHypothesis.SHAPE_BOX
    if shape_type == "cylinder":
        return GeometryHypothesis.SHAPE_CYLINDER
    if shape_type == "cup_like":
        return GeometryHypothesis.SHAPE_CUP_LIKE
    return GeometryHypothesis.SHAPE_UNKNOWN


def dimensions_for_target(target: dict) -> Vector3:
    dimensions = Vector3()
    if target.get("shape_type") == "cylinder":
        diameter = 2.0 * float(target.get("radius_m", 0.0))
        dimensions.x = diameter
        dimensions.y = diameter
        dimensions.z = float(target.get("height_m", 0.0))
        return dimensions

    dimensions.x = float(target.get("size_x_m", 0.0))
    dimensions.y = float(target.get("size_y_m", 0.0))
    dimensions.z = float(target.get("size_z_m", 0.0))
    return dimensions


def top_down_orientation(yaw_rad: float) -> Quaternion:
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


def make_pose(frame_id: str, x: float, y: float, z: float, orientation: Quaternion) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation = orientation
    return pose


def clamp01(value: float | None, fallback: float = 0.0) -> float:
    if value is None:
        value = fallback
    if not math.isfinite(float(value)):
        return fallback
    return max(0.0, min(1.0, float(value)))


def validation_for_decision(decision: str) -> int:
    if decision == "executable":
        return GeometryHypothesis.VALIDATION_VALID
    if decision == "reject":
        return GeometryHypothesis.VALIDATION_REJECTED
    return GeometryHypothesis.VALIDATION_UNKNOWN


def make_score(row: dict) -> ScoreBreakdown:
    decision = row.get("decision", "")
    score = ScoreBreakdown()
    score.visual = float(row.get("visual_score") or 0.0)
    score.failure = 1.0 if decision == "executable" else 0.0
    score.depth = 1.0 if row.get("descent_clearance_status") == "ok" else 0.0
    score.physical = score.depth
    score.grasp = float(row.get("grasp_score") or (1.0 if decision == "executable" else 0.0))
    score.prior = 0.50
    score.total = float(row.get("joint_score") or 0.0)
    return score


def make_grasp_candidate(row: dict, target: dict, frame_id: str) -> GraspCandidate:
    dimensions = dimensions_for_target(target)
    x = float(target["center_x_m"])
    y = float(target["center_y_m"])
    center_z = float(target["center_z_m"])
    yaw = float(target.get("yaw_rad", 0.0))
    grasp_z = center_z + 0.5 * dimensions.z + DEFAULT_TOP_GRASP_OFFSET_M
    pregrasp_z = grasp_z + float(target.get("pregrasp_clearance_m", 0.09))
    orientation = top_down_orientation(yaw)

    grasp = GraspCandidate()
    grasp.grasp_id = f"{row['hypothesis_id']}_top"
    grasp.grasp_pose = make_pose(frame_id, x, y, grasp_z, orientation)
    grasp.pregrasp_pose = make_pose(frame_id, x, y, pregrasp_z, orientation)
    grasp.approach_vector.z = -1.0
    grasp.gripper_width_m = float(target.get("required_gripper_width_m", dimensions.x))
    grasp.grasp_type = GraspCandidate.GRASP_TYPE_TOP
    grasp.score = float(row.get("joint_score") or 0.0)
    if row.get("decision") == "executable":
        grasp.validation_state = GraspCandidate.VALIDATION_VALID
    else:
        grasp.validation_state = GraspCandidate.VALIDATION_REJECTED
    grasp.failure_reason = row.get("failure_reason", "")
    return grasp


def make_hypothesis(row: dict, target: dict, frame_id: str) -> GeometryHypothesis:
    dimensions = dimensions_for_target(target)
    center_orientation = Quaternion()
    center_orientation.w = 1.0
    x = float(target["center_x_m"])
    y = float(target["center_y_m"])
    z = float(target["center_z_m"])

    hypothesis = GeometryHypothesis()
    hypothesis.hypothesis_id = str(row["hypothesis_id"])
    hypothesis.shape_type = shape_constant(str(row.get("shape_type", "")))
    hypothesis.pose_camera = make_pose(frame_id, x, y, z, center_orientation)
    hypothesis.pose_base = make_pose(frame_id, x, y, z, center_orientation)
    hypothesis.dimensions_m = dimensions
    hypothesis.score = make_score(row)
    hypothesis.confidence = clamp01(row.get("joint_score"), 0.0)
    hypothesis.uncertainty = 1.0 - hypothesis.confidence
    hypothesis.grasp_candidates = [make_grasp_candidate(row, target, frame_id)]
    hypothesis.provenance = (
        f"m4_joint_report:{row.get('source_type', 'unknown')}:"
        f"{row.get('ranker', 'unknown')}:rank={row.get('joint_rank', 0)}"
    )
    hypothesis.validation_state = validation_for_decision(str(row.get("decision", "")))
    hypothesis.failure_reason = str(row.get("failure_reason", ""))
    return hypothesis


def report_to_hypotheses(report: dict, targets: dict[str, dict], frame_id: str) -> list[GeometryHypothesis]:
    hypotheses: list[GeometryHypothesis] = []
    rows = sorted(
        report.get("rows", []),
        key=lambda row: (str(row.get("rank_group", "")), int(row.get("joint_rank") or 9999)),
    )
    for row in rows:
        if row.get("source_type") != "sim_moveit":
            continue
        target = targets.get(row.get("target_or_scene_id"))
        if target is None:
            continue
        hypotheses.append(make_hypothesis(row, target, frame_id))
    return hypotheses


class M4JointHypothesisPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("m4_joint_hypothesis_publisher_node")
        self.report_path = self.declare_parameter("report_path", DEFAULT_REPORT_PATH).value
        self.targets_path = self.declare_parameter("targets_path", DEFAULT_TARGETS_PATH).value
        self.hypothesis_topic = self.declare_parameter(
            "hypothesis_topic", DEFAULT_HYPOTHESIS_TOPIC).value
        self.frame_id = self.declare_parameter("frame_id", "world").value
        self.trial_id = self.declare_parameter("trial_id", "m4_tabletop").value
        self.observation_id = self.declare_parameter("observation_id", "m4_joint_hypotheses").value
        self.backend_name = self.declare_parameter("backend_name", "ghost_mgg_m4_joint").value

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.publisher = self.create_publisher(
            GeometryHypothesisArray, self.hypothesis_topic, qos)
        self.timer = self.create_timer(0.5, self.publish_hypotheses)
        self.reported_success = False
        self.reported_missing_inputs = False

    def publish_hypotheses(self) -> None:
        try:
            report = load_json(self.report_path)
            targets = load_targets(self.targets_path)
        except Exception as error:
            if not self.reported_missing_inputs:
                self.get_logger().warn(f"Cannot read M4 joint hypothesis inputs yet: {error}")
                self.reported_missing_inputs = True
            return

        message = GeometryHypothesisArray()
        message.header.frame_id = self.frame_id
        message.header.stamp = self.get_clock().now().to_msg()
        message.trial_id = self.trial_id
        message.observation_id = self.observation_id
        message.backend_name = self.backend_name
        message.hypotheses = report_to_hypotheses(report, targets, self.frame_id)
        self.publisher.publish(message)
        if not self.reported_success:
            self.get_logger().info(
                f"Published {len(message.hypotheses)} M4 joint hypotheses to {self.hypothesis_topic}")
            self.reported_success = True


def main() -> None:
    rclpy.init()
    node = M4JointHypothesisPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
