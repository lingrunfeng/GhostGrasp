#!/usr/bin/env python3
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


DEFAULT_REPORT_PATH = "reports/m4_joint_hypotheses/joint_hypotheses.json"
DEFAULT_TARGETS_PATH = "config/m4_sim_grasp_targets.json"
DEFAULT_MARKER_TOPIC = "/ghost_mgg/m4_joint_hypothesis_markers"
DEFAULT_EXECUTED_TOPIC = "/ghost_mgg/m4_executed_hypotheses"


def delete_all_marker(frame_id: str, stamp) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.action = Marker.DELETEALL
    return marker


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_target_positions(path: str | Path) -> dict[str, dict]:
    data = load_json(path)
    return {row["target_id"]: row for row in data.get("rows", [])}


def set_color(marker: Marker, decision: str, alpha: float = 0.90) -> None:
    marker.color.a = alpha
    if decision == "executable":
        marker.color.r = 0.05
        marker.color.g = 0.95
        marker.color.b = 0.25
    elif decision == "candidate":
        marker.color.r = 0.20
        marker.color.g = 0.62
        marker.color.b = 1.00
    else:
        marker.color.r = 1.00
        marker.color.g = 0.22
        marker.color.b = 0.10


def set_executed_color(marker: Marker, alpha: float = 0.22) -> None:
    marker.color.a = alpha
    marker.color.r = 0.55
    marker.color.g = 0.55
    marker.color.b = 0.55


def make_executable_target_marker(
    row: dict,
    target: dict,
    frame_id: str,
    stamp,
    marker_id: int,
    executed: bool = False,
) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "m4_joint_executable_target"
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.type = Marker.SPHERE
    marker.pose.position.x = float(target["center_x_m"])
    marker.pose.position.y = float(target["center_y_m"])
    marker.pose.position.z = float(target["center_z_m"]) + 0.040
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.040
    marker.scale.y = 0.040
    marker.scale.z = 0.040
    if executed:
        set_executed_color(marker, 0.18)
    else:
        set_color(marker, row["decision"], 0.82)
    return marker


def make_rank_text_marker(
    row: dict,
    target: dict,
    frame_id: str,
    stamp,
    marker_id: int,
    executed: bool = False,
) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "m4_joint_rank_text"
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.type = Marker.TEXT_VIEW_FACING
    marker.pose.position.x = float(target["center_x_m"])
    marker.pose.position.y = float(target["center_y_m"])
    marker.pose.position.z = float(target["center_z_m"]) + 0.082
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.018
    if executed:
        marker.color.r = 0.62
        marker.color.g = 0.62
        marker.color.b = 0.62
        marker.color.a = 0.78
        marker.text = f"DONE:{row['target_or_scene_id']}"
    else:
        marker.color.r = 0.95
        marker.color.g = 0.98
        marker.color.b = 1.00
        marker.color.a = 0.96
        marker.text = (
            f"J{row['joint_rank']}:{row['target_or_scene_id']}\n"
            f"{row['decision']} S={float(row['joint_score']):.2f}"
        )
    return marker


def make_summary_text_marker(report: dict, frame_id: str, stamp, marker_id: int) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "m4_joint_summary_text"
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.type = Marker.TEXT_VIEW_FACING
    marker.pose.position.x = -0.18
    marker.pose.position.y = -0.23
    marker.pose.position.z = 1.13
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.019
    marker.color.r = 0.94
    marker.color.g = 0.98
    marker.color.b = 1.00
    marker.color.a = 0.96
    summary = report.get("summary", {})
    executable = sum(1 for row in report.get("rows", []) if row.get("decision") == "executable")
    candidate = sum(1 for row in report.get("rows", []) if row.get("decision") == "candidate")
    marker.text = (
        "M4_joint_hypotheses\n"
        f"rows={summary.get('total_rows', 0)} executable={executable}\n"
        f"candidate={candidate} reject={summary.get('total_rows', 0) - executable - candidate}"
    )
    return marker


def executed_hypothesis_id_from_message(message: str) -> str:
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return message.strip()
    if not data.get("executed_success", data.get("status_name") == "SUCCEEDED"):
        return ""
    return str(data.get("hypothesis_id", "")).strip()


def reset_executed_hypotheses_requested(message: str) -> bool:
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return False
    return bool(data.get("reset_executed_hypotheses", False))


class M4JointHypothesisMarkersNode(Node):
    def __init__(self) -> None:
        super().__init__("m4_joint_hypothesis_markers_node")
        self.report_path = self.declare_parameter("report_path", DEFAULT_REPORT_PATH).value
        self.targets_path = self.declare_parameter("targets_path", DEFAULT_TARGETS_PATH).value
        self.marker_topic = self.declare_parameter("marker_topic", DEFAULT_MARKER_TOPIC).value
        self.executed_hypotheses_topic = self.declare_parameter(
            "executed_hypotheses_topic", DEFAULT_EXECUTED_TOPIC).value
        self.frame_id = self.declare_parameter("frame_id", "world").value
        self.executed_hypothesis_ids: set[str] = set()
        # Keep transient_local behavior consistent with RViz MarkerArray displays.
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.publisher = self.create_publisher(MarkerArray, self.marker_topic, qos)
        self.executed_subscription = self.create_subscription(
            String,
            self.executed_hypotheses_topic,
            self.handle_executed_hypothesis,
            qos,
        )
        self.timer = self.create_timer(0.5, self.publish_markers)
        self.reported_success = False
        self.reported_missing_report = False

    def handle_executed_hypothesis(self, message: String) -> None:
        if reset_executed_hypotheses_requested(message.data):
            self.executed_hypothesis_ids.clear()
            return
        hypothesis_id = executed_hypothesis_id_from_message(message.data)
        if hypothesis_id:
            self.executed_hypothesis_ids.add(hypothesis_id)

    def publish_markers(self) -> None:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        markers.markers.append(delete_all_marker(self.frame_id, stamp))
        try:
            report = load_json(self.report_path)
            targets = load_target_positions(self.targets_path)
        except Exception as error:
            if not self.reported_missing_report:
                self.get_logger().warn(f"Cannot read M4 joint hypothesis inputs yet: {error}")
                self.reported_missing_report = True
            self.publisher.publish(markers)
            return

        marker_id = 1
        for row in report.get("rows", []):
            if row.get("source_type") != "sim_moveit":
                continue
            target = targets.get(row.get("target_or_scene_id"))
            if target is None:
                continue
            executed = str(row.get("hypothesis_id", "")) in self.executed_hypothesis_ids
            markers.markers.append(
                make_executable_target_marker(row, target, self.frame_id, stamp, marker_id, executed)
            )
            marker_id += 1
            markers.markers.append(
                make_rank_text_marker(row, target, self.frame_id, stamp, marker_id, executed)
            )
            marker_id += 1
        markers.markers.append(make_summary_text_marker(report, self.frame_id, stamp, marker_id))
        self.publisher.publish(markers)
        if not self.reported_success:
            self.get_logger().info(f"Published M4 joint hypothesis markers from {self.report_path}")
            self.reported_success = True


def main() -> None:
    rclpy.init()
    node = M4JointHypothesisMarkersNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
