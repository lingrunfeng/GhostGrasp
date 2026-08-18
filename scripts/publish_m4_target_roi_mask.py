#!/usr/bin/env python3
import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


DEFAULT_TOPIC = "/ghost_mgg/d435/external_target_mask"


def make_roi_mask(width: int, height: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    u_min, v_min, u_max, v_max = bbox
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    u0 = max(0, min(width, int(u_min)))
    u1 = max(0, min(width, int(u_max)))
    v0 = max(0, min(height, int(v_min)))
    v1 = max(0, min(height, int(v_max)))
    if u1 <= u0 or v1 <= v0:
        raise ValueError("bbox must have positive clipped area")
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[v0:v1, u0:u1] = 255
    return mask


def mask_to_msg(mask: np.ndarray, frame_id: str) -> Image:
    resolved = np.asarray(mask, dtype=np.uint8)
    msg = Image()
    msg.header.frame_id = frame_id
    msg.height = int(resolved.shape[0])
    msg.width = int(resolved.shape[1])
    msg.encoding = "mono8"
    msg.is_bigendian = False
    msg.step = int(resolved.shape[1])
    msg.data = resolved.reshape(-1).tolist()
    return msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a transient-local mono8 target ROI mask for M4 live GHOST-MGG.",
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("U_MIN", "V_MIN", "U_MAX", "V_MAX"), required=True)
    parser.add_argument("--frame-id", default="d435_depth_optical_frame")
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        mask = make_roi_mask(args.width, args.height, tuple(args.bbox))
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    rclpy.init()
    node = rclpy.create_node("ghost_mgg_publish_m4_target_roi_mask")
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    qos.reliability = ReliabilityPolicy.RELIABLE
    publisher = node.create_publisher(Image, args.topic, qos)
    msg = mask_to_msg(mask, args.frame_id)
    period_sec = 1.0 / max(0.1, float(args.rate_hz))
    deadline = time.monotonic() + max(0.0, float(args.duration_sec))
    publish_count = 0
    try:
        while time.monotonic() <= deadline or publish_count == 0:
            msg.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.02)
            publish_count += 1
            time.sleep(period_sec)
        print(
            f"published_roi_mask topic={args.topic} bbox={','.join(map(str, args.bbox))} "
            f"size={args.width}x{args.height} count={publish_count}"
        )
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
