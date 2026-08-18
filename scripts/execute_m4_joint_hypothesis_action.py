#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from ghost_mgg_interfaces.action import ExecuteGrasp
from ghost_mgg_interfaces.msg import GeometryHypothesis, GeometryHypothesisArray
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


DEFAULT_HYPOTHESES_TOPIC = "/ghost_mgg/m4_joint_hypotheses"
DEFAULT_EXECUTE_ACTION = "/grasp_executors/moveit_sim/execute"
DEFAULT_EXECUTED_TOPIC = "/ghost_mgg/m4_executed_hypotheses"
DEFAULT_OUTPUT_JSON = Path("reports/m4_joint_hypothesis_execute/result.json")


def status_name(status: int) -> str:
    names = {
        1: "SUCCEEDED",
        2: "FAILED",
        3: "TIMEOUT",
        4: "CANCELED",
    }
    return names.get(int(status), "UNKNOWN")


def hypothesis_center_xyz(hypothesis: GeometryHypothesis) -> tuple[float, float, float]:
    position = hypothesis.pose_base.pose.position
    return (float(position.x), float(position.y), float(position.z))


def hypothesis_center_distance_m(
    left: GeometryHypothesis,
    right: GeometryHypothesis,
) -> float:
    lx, ly, lz = hypothesis_center_xyz(left)
    rx, ry, rz = hypothesis_center_xyz(right)
    return ((lx - rx) ** 2 + (ly - ry) ** 2 + (lz - rz) ** 2) ** 0.5


def hypothesis_xy_distance_m(hypothesis: GeometryHypothesis, target_x: float, target_y: float) -> float:
    x, y, _ = hypothesis_center_xyz(hypothesis)
    return ((x - target_x) ** 2 + (y - target_y) ** 2) ** 0.5


def stable_hypothesis_match(
    left: GeometryHypothesis,
    right: GeometryHypothesis,
    center_tolerance_m: float,
) -> bool:
    return (
        left.hypothesis_id == right.hypothesis_id
        and left.shape_type == right.shape_type
        and hypothesis_center_distance_m(left, right) <= center_tolerance_m
    )


def selected_hypothesis_summary(hypothesis: GeometryHypothesis) -> dict:
    px, py, pz = hypothesis_center_xyz(hypothesis)
    return {
        "selected_shape_type": hypothesis.shape_type,
        "selected_pose_base": {
            "frame_id": hypothesis.pose_base.header.frame_id,
            "x": px,
            "y": py,
            "z": pz,
        },
        "selected_dimensions": {
            "x": float(hypothesis.dimensions_m.x),
            "y": float(hypothesis.dimensions_m.y),
            "z": float(hypothesis.dimensions_m.z),
        },
    }


def wait_for_hypotheses(
    node,
    topic: str,
    timeout_sec: float,
    stable_count: int = 1,
    stable_center_tolerance_m: float = 0.02,
    target_id: str = "",
    target_x: float | None = None,
    target_y: float | None = None,
    target_tolerance_m: float = 0.05,
) -> GeometryHypothesisArray | None:
    received: dict[str, object] = {}
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    qos.reliability = ReliabilityPolicy.RELIABLE
    subscription = node.create_subscription(
        GeometryHypothesisArray,
        topic,
        lambda message: received.update({"message": message, "sequence": object()}),
        qos,
    )
    deadline = time.monotonic() + timeout_sec
    required_stable_count = max(1, int(stable_count))
    last_sequence = None
    last_selected: GeometryHypothesis | None = None
    current_stable_count = 0
    try:
        while time.monotonic() < deadline:
            if "message" in received and received.get("sequence") is not last_sequence:
                last_sequence = received.get("sequence")
                message = received["message"]
                assert isinstance(message, GeometryHypothesisArray)
                selected = select_valid_hypothesis(
                    message,
                    target_id,
                    target_x=target_x,
                    target_y=target_y,
                    target_tolerance_m=target_tolerance_m,
                )
                if selected is None:
                    current_stable_count = 0
                    last_selected = None
                    rclpy.spin_once(node, timeout_sec=0.05)
                    continue
                if required_stable_count <= 1:
                    return message
                if last_selected is not None and stable_hypothesis_match(
                    last_selected,
                    selected,
                    stable_center_tolerance_m,
                ):
                    current_stable_count += 1
                else:
                    current_stable_count = 1
                last_selected = selected
                if current_stable_count >= required_stable_count:
                    return message
            rclpy.spin_once(node, timeout_sec=0.1)
        return None
    finally:
        node.destroy_subscription(subscription)


def select_valid_hypothesis(
    hypotheses: GeometryHypothesisArray,
    target_id: str = "",
    target_x: float | None = None,
    target_y: float | None = None,
    target_tolerance_m: float = 0.05,
) -> GeometryHypothesis | None:
    candidates = select_valid_hypotheses(
        hypotheses,
        target_id,
        target_x=target_x,
        target_y=target_y,
        target_tolerance_m=target_tolerance_m,
    )
    return candidates[0] if candidates else None


def select_valid_hypotheses(
    hypotheses: GeometryHypothesisArray,
    target_id: str = "",
    target_x: float | None = None,
    target_y: float | None = None,
    target_tolerance_m: float = 0.05,
) -> list[GeometryHypothesis]:
    candidates: list[GeometryHypothesis] = []
    for hypothesis in hypotheses.hypotheses:
        if target_id and hypothesis.hypothesis_id != target_id:
            continue
        if hypothesis.validation_state == GeometryHypothesis.VALIDATION_VALID:
            candidates.append(hypothesis)
    if target_x is not None and target_y is not None:
        candidates = [
            hypothesis
            for hypothesis in candidates
            if hypothesis_xy_distance_m(hypothesis, target_x, target_y) <= target_tolerance_m
        ]
        candidates.sort(key=lambda hypothesis: hypothesis_xy_distance_m(hypothesis, target_x, target_y))
    return candidates


def async_send_goal(node, action_client: ActionClient, goal: ExecuteGrasp.Goal, timeout_sec: float = 5.0):
    send_future = action_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future, timeout_sec=timeout_sec)
    if not send_future.done():
        return None
    return send_future.result()


def async_get_result(node, goal_handle, timeout_sec: float):
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_sec)
    if not result_future.done():
        return None
    return result_future.result()


def execute_hypothesis(
    node,
    action_client: ActionClient,
    hypothesis: GeometryHypothesis,
    trial_id: str,
    max_runtime_sec: float,
    result_timeout_sec: float,
):
    goal = ExecuteGrasp.Goal()
    goal.trial_id = trial_id
    goal.hypothesis = hypothesis
    goal.max_runtime_sec = max_runtime_sec
    goal_handle = async_send_goal(node, action_client, goal)
    if goal_handle is None:
        return {
            "action_result_code": 0,
            "status": 3,
            "attempt_status_name": "TIMEOUT",
            "failure_reason": "execute goal send timed out",
            "runtime_sec": 0.0,
        }
    if not goal_handle.accepted:
        return {
            "action_result_code": 0,
            "status": 2,
            "attempt_status_name": "FAILED",
            "failure_reason": "execute goal rejected",
            "runtime_sec": 0.0,
        }

    wrapped_result = async_get_result(node, goal_handle, result_timeout_sec)
    if wrapped_result is None:
        return {
            "action_result_code": 0,
            "status": 3,
            "attempt_status_name": "TIMEOUT",
            "failure_reason": "execute result timed out",
            "runtime_sec": 0.0,
        }

    result = wrapped_result.result
    return {
        "action_result_code": int(wrapped_result.status),
        "status": int(result.status),
        "attempt_status_name": status_name(result.status),
        "failure_reason": result.failure_reason,
        "runtime_sec": float(result.runtime_sec),
    }


def publish_execution_event(
    node,
    topic: str,
    hypothesis_id: str,
    status_name_value: str,
    trial_id: str,
) -> None:
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    qos.reliability = ReliabilityPolicy.RELIABLE
    publisher = node.create_publisher(String, topic, qos)
    message = String()
    message.data = json.dumps(
        {
            "schema_version": "m4_executed_hypothesis_v1",
            "hypothesis_id": hypothesis_id,
            "status_name": status_name_value,
            "executed_success": status_name_value == "SUCCEEDED",
            "trial_id": trial_id,
        },
        separators=(",", ":"),
    )
    for _ in range(3):
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.05)
    node.destroy_publisher(publisher)


def reset_execution_events(node, topic: str, trial_id: str) -> None:
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    qos.reliability = ReliabilityPolicy.RELIABLE
    publisher = node.create_publisher(String, topic, qos)
    message = String()
    message.data = json.dumps(
        {
            "schema_version": "m4_executed_hypotheses_reset_v1",
            "reset_executed_hypotheses": True,
            "trial_id": trial_id,
        },
        separators=(",", ":"),
    )
    for _ in range(3):
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.05)
    node.destroy_publisher(publisher)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypotheses-topic", default=DEFAULT_HYPOTHESES_TOPIC)
    parser.add_argument("--execute-action", default=DEFAULT_EXECUTE_ACTION)
    parser.add_argument("--executed-topic", default=DEFAULT_EXECUTED_TOPIC)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--trial-id", default="m4_joint_hypothesis_execute")
    parser.add_argument("--target-id", default="")
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-tolerance-m", type=float, default=0.05)
    parser.add_argument("--hypotheses-timeout-sec", type=float, default=15.0)
    parser.add_argument("--stable-hypotheses-count", type=int, default=1)
    parser.add_argument("--stable-center-tolerance-m", type=float, default=0.02)
    parser.add_argument("--action-server-timeout-sec", type=float, default=25.0)
    parser.add_argument("--result-timeout-sec", type=float, default=90.0)
    parser.add_argument("--max-runtime-sec", type=float, default=80.0)
    parser.add_argument("--fallback-until-success", action="store_true")
    parser.add_argument("--simulate-failure-id", action="append", default=[])
    parser.add_argument("--simulate-first-failure", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--fallback-delay-sec", type=float, default=0.75)
    parser.add_argument("--no-publish-executed-event", action="store_true")
    parser.add_argument("--reset-executed-before-run", action="store_true")
    return parser.parse_known_args()[0]


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("ghost_mgg_m4_joint_hypothesis_execute_client")
    action_client = ActionClient(node, ExecuteGrasp, args.execute_action)
    try:
        if args.reset_executed_before_run:
            reset_execution_events(node, args.executed_topic, args.trial_id)
        hypotheses = wait_for_hypotheses(
            node,
            args.hypotheses_topic,
            args.hypotheses_timeout_sec,
            stable_count=args.stable_hypotheses_count,
            stable_center_tolerance_m=args.stable_center_tolerance_m,
            target_id=args.target_id,
            target_x=args.target_x,
            target_y=args.target_y,
            target_tolerance_m=args.target_tolerance_m,
        )
        if hypotheses is None:
            print(f"no hypotheses received on {args.hypotheses_topic}", file=sys.stderr)
            return 2
        candidates = select_valid_hypotheses(
            hypotheses,
            args.target_id,
            target_x=args.target_x,
            target_y=args.target_y,
            target_tolerance_m=args.target_tolerance_m,
        )
        if not candidates:
            print("no valid hypothesis selected", file=sys.stderr)
            return 3
        max_attempts = max(1, int(args.max_attempts))
        if not args.fallback_until_success:
            max_attempts = 1
        candidates = candidates[:max_attempts]
        if not action_client.wait_for_server(timeout_sec=args.action_server_timeout_sec):
            print(f"execute action unavailable: {args.execute_action}", file=sys.stderr)
            return 4

        attempts = []
        final_attempt = None
        selected = candidates[0]
        simulated_failure_ids = set(args.simulate_failure_id)
        for index, candidate in enumerate(candidates, start=1):
            selected = candidate
            simulate_first_failure = args.simulate_first_failure and index == 1
            if simulate_first_failure or candidate.hypothesis_id in simulated_failure_ids:
                attempt = {
                    "attempt_index": index,
                    "hypothesis_id": candidate.hypothesis_id,
                    "action_result_code": 0,
                    "status": 2,
                    "attempt_status_name": "FAILED",
                    "failure_reason": "simulated_failure",
                    "runtime_sec": 0.0,
                    "simulated": True,
                    "simulate_first_failure": bool(simulate_first_failure),
                }
            else:
                attempt = execute_hypothesis(
                    node,
                    action_client,
                    candidate,
                    args.trial_id,
                    args.max_runtime_sec,
                    args.result_timeout_sec,
                )
                attempt.update(
                    {
                        "attempt_index": index,
                        "hypothesis_id": candidate.hypothesis_id,
                        "simulated": False,
                        "simulate_first_failure": False,
                    }
                )
            attempts.append(attempt)
            final_attempt = attempt
            if int(attempt["status"]) == 1:
                break
            if not args.fallback_until_success:
                break
            if index < len(candidates):
                deadline = time.monotonic() + max(0.0, float(args.fallback_delay_sec))
                while time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)

        if final_attempt is None:
            print("no hypothesis execution attempted", file=sys.stderr)
            return 8
        output = {
            "schema_version": "m4_joint_hypothesis_execute_v1",
            "hypotheses_topic": args.hypotheses_topic,
            "execute_action": args.execute_action,
            "executed_topic": args.executed_topic,
            "trial_id": args.trial_id,
            "selected_hypothesis_id": selected.hypothesis_id,
            **selected_hypothesis_summary(selected),
            "target_xy": (
                {"x": args.target_x, "y": args.target_y}
                if args.target_x is not None and args.target_y is not None
                else None
            ),
            "target_error_m": (
                hypothesis_xy_distance_m(selected, args.target_x, args.target_y)
                if args.target_x is not None and args.target_y is not None
                else None
            ),
            "attempts": attempts,
            "action_result_code": int(final_attempt["action_result_code"]),
            "status": int(final_attempt["status"]),
            "status_name": status_name(final_attempt["status"]),
            "final_status_name": status_name(final_attempt["status"]),
            "runtime_sec": float(final_attempt["runtime_sec"]),
            "failure_reason": final_attempt["failure_reason"],
        }
        if int(final_attempt["status"]) == 1 and not args.no_publish_executed_event:
            publish_execution_event(
                node,
                args.executed_topic,
                str(final_attempt["hypothesis_id"]),
                output["status_name"],
                args.trial_id,
            )
            output["executed_event_published"] = True
        else:
            output["executed_event_published"] = False
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(
            f"selected_hypothesis_id={selected.hypothesis_id} "
            f"status_name={output['status_name']} output={args.output_json}"
        )
        return 0 if int(final_attempt["status"]) == 1 else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
