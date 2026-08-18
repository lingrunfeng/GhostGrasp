#!/usr/bin/env python3
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


DEFAULT_COLOR_TOPIC = "/ghost_mgg/d435/color/image_raw"
DEFAULT_MASK_TOPIC = "/ghost_mgg/d435/external_target_mask"


def decode_color_image(image: Image) -> np.ndarray:
    encodings = {
        "rgb8": (3, (0, 1, 2)),
        "bgr8": (3, (2, 1, 0)),
        "rgba8": (4, (0, 1, 2)),
        "bgra8": (4, (2, 1, 0)),
    }
    if image.encoding not in encodings:
        raise ValueError(f"unsupported color encoding: {image.encoding}")
    channel_count, rgb_indices = encodings[image.encoding]
    rows = np.frombuffer(bytes(image.data), dtype=np.uint8).reshape(image.height, image.step)
    packed = rows[:, : image.width * channel_count].reshape(image.height, image.width, channel_count)
    return packed[:, :, rgb_indices].astype(np.uint8, copy=True)


def target_sized_component(mask: np.ndarray, *, min_area_px: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    output = np.zeros(binary.shape, dtype=bool)
    if not binary.any():
        return output

    visited = np.zeros(binary.shape, dtype=bool)
    height, width = binary.shape
    max_area_px = max(float(min_area_px), 0.055 * float(binary.size))
    max_bbox_width_px = max(1.0, 0.42 * float(width))
    max_bbox_height_px = max(1.0, 0.42 * float(height))
    best_pixels: list[tuple[int, int]] = []
    best_score = -float("inf")

    for start_v, start_u in zip(*np.nonzero(binary)):
        if visited[start_v, start_u]:
            continue
        stack = [(int(start_v), int(start_u))]
        visited[start_v, start_u] = True
        pixels = []
        min_v = max_v = int(start_v)
        min_u = max_u = int(start_u)
        while stack:
            v, u = stack.pop()
            pixels.append((v, u))
            min_v = min(min_v, v)
            max_v = max(max_v, v)
            min_u = min(min_u, u)
            max_u = max(max_u, u)
            for nv, nu in ((v - 1, u), (v + 1, u), (v, u - 1), (v, u + 1)):
                if 0 <= nv < height and 0 <= nu < width and binary[nv, nu] and not visited[nv, nu]:
                    visited[nv, nu] = True
                    stack.append((nv, nu))

        area = len(pixels)
        bbox_width = float(max_u - min_u + 1)
        bbox_height = float(max_v - min_v + 1)
        if (
            area < int(min_area_px)
            or area > max_area_px
            or bbox_width > max_bbox_width_px
            or bbox_height > max_bbox_height_px
        ):
            continue
        compactness = float(area) / max(1.0, bbox_width * bbox_height)
        score = float(area) * (0.35 + 0.65 * compactness)
        if score > best_score:
            best_score = score
            best_pixels = pixels

    for v, u in best_pixels:
        output[v, u] = True
    return output


def color_mask(rgb_image: np.ndarray, *, color_hint: str, min_area_px: int) -> np.ndarray:
    hint = str(color_hint).strip().lower()
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    maximum = np.maximum.reduce([red, green, blue])
    minimum = np.minimum.reduce([red, green, blue])
    saturation = maximum - minimum
    if hint == "red":
        raw = (red >= 90) & (red >= green + 45) & (red >= blue + 45) & (saturation >= 45)
    elif hint == "green":
        raw = (green >= 80) & (green >= red + 35) & (green >= blue + 35) & (saturation >= 35)
    elif hint == "blue":
        raw = (blue >= 80) & (blue >= red + 35) & (blue >= green + 35) & (saturation >= 35)
    else:
        raw = np.zeros(rgb.shape[:2], dtype=bool)
    return target_sized_component(raw, min_area_px=int(min_area_px))


def mask_to_msg(mask: np.ndarray, header) -> Image:
    resolved = np.asarray(mask, dtype=bool)
    msg = Image()
    msg.header = header
    msg.height = int(resolved.shape[0])
    msg.width = int(resolved.shape[1])
    msg.encoding = "mono8"
    msg.is_bigendian = False
    msg.step = int(resolved.shape[1])
    msg.data = (resolved.astype(np.uint8) * 255).reshape(-1).tolist()
    return msg


class ColorMaskAdapter(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("m4_external_color_mask_adapter")
        self.args = args
        self.latest_msg: Image | None = None
        sub_qos = QoSProfile(depth=5)
        sub_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        pub_qos = QoSProfile(depth=1)
        pub_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        pub_qos.reliability = ReliabilityPolicy.RELIABLE
        self.publisher = self.create_publisher(Image, args.mask_topic, pub_qos)
        self.subscription = self.create_subscription(Image, args.color_topic, self.handle_image, sub_qos)
        self.timer = self.create_timer(1.0 / max(0.1, float(args.rate_hz)), self.republish_latest)

    def handle_image(self, msg: Image) -> None:
        try:
            rgb = decode_color_image(msg)
            mask = color_mask(rgb, color_hint=self.args.color_hint, min_area_px=self.args.min_area_px)
        except ValueError as error:
            self.get_logger().warn(str(error))
            return
        self.latest_msg = mask_to_msg(mask, msg.header)
        self.publisher.publish(self.latest_msg)

    def republish_latest(self) -> None:
        if self.latest_msg is not None:
            self.latest_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(self.latest_msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish an external mono8 target mask from M4 RGB color.")
    parser.add_argument("--color-topic", default=DEFAULT_COLOR_TOPIC)
    parser.add_argument("--mask-topic", default=DEFAULT_MASK_TOPIC)
    parser.add_argument("--color-hint", default="red", choices=("red", "green", "blue"))
    parser.add_argument("--min-area-px", type=int, default=40)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = ColorMaskAdapter(args)
    deadline = None
    if float(args.duration_sec) > 0.0:
        deadline = time.monotonic() + float(args.duration_sec)
    try:
        while rclpy.ok() and (deadline is None or time.monotonic() < deadline):
            rclpy.spin_once(node, timeout_sec=0.1)
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
