#!/usr/bin/env python3
import json
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


DEFAULT_REPORT_PATH = "reports/m4_sim_moveit_dryrun/plan_results.json"
DEFAULT_MARKER_TOPIC = "/ghost_mgg/m4_sim_moveit_plan_markers"
COLORS = [
    (0.10, 0.95, 0.25),
    (0.10, 0.55, 1.00),
    (1.00, 0.34, 0.18),
    (0.95, 0.90, 0.18),
]


def point_from_dict(data: dict) -> Point:
    point = Point()
    point.x = float(data["x"])
    point.y = float(data["y"])
    point.z = float(data["z"])
    return point


def delete_all_marker(frame_id: str, stamp) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.action = Marker.DELETEALL
    return marker


def set_color(marker: Marker, color: tuple[float, float, float], alpha: float) -> None:
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = alpha


def first_planned_attempt(row: dict) -> dict | None:
    for attempt in row.get("attempts", []):
        if attempt.get("planned") and attempt.get("path_points_world"):
            return attempt
    return None


def make_path_marker(row: dict, attempt: dict, frame_id: str, stamp, marker_id: int, color) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "m4_sim_moveit_plan_path"
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.type = Marker.LINE_STRIP
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.006
    set_color(marker, color, 0.92)
    marker.points = [point_from_dict(item) for item in attempt.get("path_points_world", [])]
    return marker


def make_goal_marker(row: dict, attempt: dict, frame_id: str, stamp, marker_id: int, color) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "m4_sim_moveit_plan_goal"
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.type = Marker.SPHERE
    marker.pose.orientation.w = 1.0
    path_points = attempt.get("path_points_world", [])
    if path_points:
        marker.pose.position = point_from_dict(path_points[-1])
    marker.scale.x = 0.026
    marker.scale.y = 0.026
    marker.scale.z = 0.026
    set_color(marker, color, 0.95)
    return marker


def clearance_ok(row: dict) -> bool:
    return row.get("descent_clearance", {}).get("status") == "ok"


def make_descent_marker(row: dict, frame_id: str, stamp, marker_id: int, color) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "m4_sim_moveit_descent_path"
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.type = Marker.LINE_STRIP
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.004
    alpha = 0.95 if clearance_ok(row) else 0.55
    set_color(marker, color, alpha)
    marker.points = [point_from_dict(item) for item in row.get("descent_points_world", [])]
    return marker


def make_descent_goal_marker(row: dict, frame_id: str, stamp, marker_id: int, color) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "m4_sim_moveit_descent_goal"
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.type = Marker.SPHERE
    marker.pose.orientation.w = 1.0
    descent_points = row.get("descent_points_world", [])
    if descent_points:
        marker.pose.position = point_from_dict(descent_points[-1])
    marker.scale.x = 0.016
    marker.scale.y = 0.016
    marker.scale.z = 0.016
    if clearance_ok(row):
        set_color(marker, color, 0.96)
    else:
        set_color(marker, (1.0, 0.08, 0.04), 0.96)
    return marker


def make_text_marker(report: dict, rows: list[dict], frame_id: str, stamp, marker_id: int) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "m4_sim_moveit_plan_text"
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.type = Marker.TEXT_VIEW_FACING
    marker.pose.position.x = -0.14
    marker.pose.position.y = -0.20
    marker.pose.position.z = 1.08
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.020
    marker.color.r = 0.93
    marker.color.g = 0.98
    marker.color.b = 1.0
    marker.color.a = 0.96

    summary = report.get("summary", {})
    planned = int(summary.get("planned", 0))
    total = int(summary.get("total", len(rows)))
    clearance_status = "ok" if rows and all(clearance_ok(row) for row in rows) else "check"
    clearance_line = "clearance=ok" if clearance_status == "ok" else "clearance=check"
    lines = [f"M4_MoveIt_plan", f"planned={planned}/{total}", clearance_line]
    for row in rows:
        status = "ok" if row.get("planned") else "fail"
        lines.append(f"{row.get('target_id', 'unknown')}:{status}")
    marker.text = "\n".join(lines)
    return marker


class M4SimMoveItPlanMarkersNode(Node):
    def __init__(self) -> None:
        super().__init__("m4_sim_moveit_plan_markers_node")
        self.report_path = self.declare_parameter("report_path", DEFAULT_REPORT_PATH).value
        self.marker_topic = self.declare_parameter("marker_topic", DEFAULT_MARKER_TOPIC).value
        self.frame_id = self.declare_parameter("frame_id", "world").value
        # Keep transient_local marker behavior consistent with RViz MarkerArray displays.
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.publisher = self.create_publisher(MarkerArray, self.marker_topic, qos)
        self.timer = self.create_timer(0.5, self.publish_markers)
        self.reported_success = False
        self.reported_missing_report = False

    def publish_markers(self) -> None:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        markers.markers.append(delete_all_marker(self.frame_id, stamp))

        try:
            report = json.loads(Path(self.report_path).read_text(encoding="utf-8"))
        except Exception as error:
            if not self.reported_missing_report:
                self.get_logger().warn(f"Cannot read M4 sim MoveIt plan report yet: {error}")
                self.reported_missing_report = True
            self.publisher.publish(markers)
            return

        rows = report.get("rows", [])
        marker_id = 1
        for index, row in enumerate(rows):
            attempt = first_planned_attempt(row)
            if attempt is None:
                continue
            color = COLORS[index % len(COLORS)]
            markers.markers.append(
                make_path_marker(row, attempt, self.frame_id, stamp, marker_id, color)
            )
            marker_id += 1
            markers.markers.append(
                make_goal_marker(row, attempt, self.frame_id, stamp, marker_id, color)
            )
            marker_id += 1
            markers.markers.append(
                make_descent_marker(row, self.frame_id, stamp, marker_id, color)
            )
            marker_id += 1
            markers.markers.append(
                make_descent_goal_marker(row, self.frame_id, stamp, marker_id, color)
            )
            marker_id += 1
        markers.markers.append(make_text_marker(report, rows, self.frame_id, stamp, marker_id))
        self.publisher.publish(markers)

        if not self.reported_success:
            self.get_logger().info(
                f"Published M4 sim MoveIt plan markers from {self.report_path}"
            )
            self.reported_success = True


def main() -> None:
    rclpy.init()
    node = M4SimMoveItPlanMarkersNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
