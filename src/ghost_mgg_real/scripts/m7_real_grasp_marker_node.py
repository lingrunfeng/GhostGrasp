#!/usr/bin/env python3
"""Publish RViz markers for the current real M7 grasp target report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def yaw_to_quaternion_z(yaw_rad: float) -> dict[str, float]:
    half = 0.5 * float(yaw_rad)
    return {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)}


def target_from_report(path: Path) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    target = payload.get("target", payload)
    if not isinstance(target, dict) or not target.get("valid", True):
        return None
    return target


class M7RealGraspMarkerNode:
    def __init__(self) -> None:
        import rclpy
        from rclpy.node import Node
        from visualization_msgs.msg import MarkerArray

        class MarkerNode(Node):
            def __init__(self) -> None:
                super().__init__("m7_real_grasp_marker_node")
                self.declare_parameter(
                    "target_path",
                    "reports/m6_shadow_grasp_targets/m7_real_current/m6_shadow_grasp_target.json",
                )
                self.declare_parameter("frame_id", "base_link")
                self.publisher = self.create_publisher(
                    MarkerArray, "/ghost_mgg/m7_real_grasp_markers", 10
                )
                self.create_timer(0.5, self.publish_markers)

            def publish_markers(self) -> None:
                target = target_from_report(Path(str(self.get_parameter("target_path").value)))
                markers = self.make_marker_array(target)
                self.publisher.publish(markers)

            def make_marker_array(self, target: dict[str, Any] | None):
                from geometry_msgs.msg import Point
                from visualization_msgs.msg import Marker, MarkerArray

                frame_id = str(self.get_parameter("frame_id").value)
                stamp = self.get_clock().now().to_msg()
                markers = MarkerArray()

                delete_all = Marker()
                delete_all.header.frame_id = frame_id
                delete_all.header.stamp = stamp
                delete_all.action = Marker.DELETEALL
                markers.markers.append(delete_all)
                if target is None:
                    return markers

                x = float(target.get("center_x_m", 0.0))
                y = float(target.get("center_y_m", 0.0))
                z = float(target.get("center_z_m", 0.0))
                height = float(target.get("height_m", target.get("size_z_m", 0.03)))
                grasp_z = z + 0.5 * height + 0.005
                pregrasp_z = grasp_z + float(target.get("pregrasp_clearance_m", 0.095))

                body = Marker()
                body.header.frame_id = frame_id
                body.header.stamp = stamp
                body.ns = "m7_real_target"
                body.id = 1
                body.action = Marker.ADD
                body.pose.position.x = x
                body.pose.position.y = y
                body.pose.position.z = z
                quat = yaw_to_quaternion_z(float(target.get("yaw_rad", 0.0)))
                body.pose.orientation.x = quat["x"]
                body.pose.orientation.y = quat["y"]
                body.pose.orientation.z = quat["z"]
                body.pose.orientation.w = quat["w"]
                if target.get("shape_type") == "cylinder":
                    radius = float(target.get("radius_m", 0.02))
                    body.type = Marker.CYLINDER
                    body.scale.x = 2.0 * radius
                    body.scale.y = 2.0 * radius
                    body.scale.z = height
                else:
                    body.type = Marker.CUBE
                    body.scale.x = float(target.get("size_x_m", target.get("required_gripper_width_m", 0.04)))
                    body.scale.y = float(target.get("size_y_m", target.get("required_gripper_width_m", 0.04)))
                    body.scale.z = height
                body.color.r = 0.1
                body.color.g = 0.9
                body.color.b = 0.1
                body.color.a = 0.35
                markers.markers.append(body)

                grasp = Marker()
                grasp.header.frame_id = frame_id
                grasp.header.stamp = stamp
                grasp.ns = "m7_real_target"
                grasp.id = 2
                grasp.action = Marker.ADD
                grasp.type = Marker.SPHERE
                grasp.pose.position.x = x
                grasp.pose.position.y = y
                grasp.pose.position.z = grasp_z
                grasp.pose.orientation.w = 1.0
                grasp.scale.x = 0.018
                grasp.scale.y = 0.018
                grasp.scale.z = 0.018
                grasp.color.r = 1.0
                grasp.color.g = 0.85
                grasp.color.b = 0.0
                grasp.color.a = 0.95
                markers.markers.append(grasp)

                line = Marker()
                line.header.frame_id = frame_id
                line.header.stamp = stamp
                line.ns = "m7_real_target"
                line.id = 3
                line.action = Marker.ADD
                line.type = Marker.LINE_STRIP
                line.scale.x = 0.006
                start = Point()
                start.x = x
                start.y = y
                start.z = pregrasp_z
                end = Point()
                end.x = x
                end.y = y
                end.z = grasp_z
                line.points = [start, end]
                line.color.r = 1.0
                line.color.g = 1.0
                line.color.b = 1.0
                line.color.a = 0.95
                markers.markers.append(line)
                return markers

        self.rclpy = rclpy
        self.node = MarkerNode()

    def spin(self) -> None:
        self.rclpy.spin(self.node)

    def shutdown(self) -> None:
        self.node.destroy_node()
        try:
            self.rclpy.shutdown()
        except Exception:
            pass


def main() -> None:
    import rclpy

    rclpy.init()
    node = M7RealGraspMarkerNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
