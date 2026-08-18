#!/usr/bin/env python3
"""Transparent-object detector: RGB in, pixel boxes out.

Runs a fine-tuned nano detector (YOLO11n, CPU-friendly) on the color stream
at a bounded rate and publishes [x0, y0, x1, y1, conf] rows on a
Float32MultiArray. The main hypothesis node treats the boxes as POSITION
evidence for its transparent evidence field; existence still requires depth
oddity there, so a detector hallucination on an empty table publishes
nothing. If the model or ultralytics is missing the node idles quietly and
the pipeline falls back to classical RGB saliency."""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

DEFAULT_COLOR_TOPIC = "/camera/camera/color/image_raw"
DEFAULT_BOXES_TOPIC = "/ghost_mgg/d435/transparent_boxes"
DEFAULT_DEBUG_IMAGE_TOPIC = "/ghost_mgg/d435/transparent_debug_image"
DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/Leorover-team8/Ghost-MGG/models/transparent_yolo11n.pt"
)
DEFAULT_IR_MODEL_PATH = os.path.expanduser(
    "~/Leorover-team8/Ghost-MGG/models/transparent_ir_yolo11n.pt"
)
DEFAULT_IR_TOPIC = "/camera/camera/infra1/image_rect_raw"
# below this color-image mean the RGB stream is too dark to trust; switch to
# the active-IR channel (projector dots are illumination-invariant)
DEFAULT_DARK_LUMINANCE = 35.0
# infra1 pixel -> color pixel, table-plane affine calibrated from the rig's
# recorded tf_static + camera infos (residual <4px over the workspace):
#   u_color = a*u_ir + b*v_ir + c ; v_color = d*u_ir + e*v_ir + f
DEFAULT_IR_TO_COLOR_AFFINE = [1.5958, 0.1346, -182.0, -0.0407, 1.5916, -114.0]


def decode_color_image(image: Image) -> np.ndarray:
    channels_by_encoding = {
        "rgb8": (3, (0, 1, 2)),
        "bgr8": (3, (2, 1, 0)),
        "rgba8": (4, (0, 1, 2)),
        "bgra8": (4, (2, 1, 0)),
    }
    if image.encoding not in channels_by_encoding:
        raise ValueError(f"unsupported color encoding: {image.encoding}")
    channel_count, rgb_indices = channels_by_encoding[image.encoding]
    row_values = np.frombuffer(bytes(image.data), dtype=np.uint8).reshape(
        image.height, image.step
    )
    packed = row_values[:, : image.width * channel_count].reshape(
        image.height, image.width, channel_count
    )
    return packed[:, :, rgb_indices].astype(np.uint8, copy=True)


class TransparentDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("m8_transparent_detector_node")
        self.color_topic = str(
            self.declare_parameter("color_topic", DEFAULT_COLOR_TOPIC).value
        )
        self.boxes_topic = str(
            self.declare_parameter("boxes_topic", DEFAULT_BOXES_TOPIC).value
        )
        self.model_path = str(
            self.declare_parameter("model_path", DEFAULT_MODEL_PATH).value
        )
        self.min_confidence = float(
            self.declare_parameter("min_confidence", 0.25).value
        )
        self.max_rate_hz = float(self.declare_parameter("max_rate_hz", 6.0).value)
        self.ir_topic = str(self.declare_parameter("ir_topic", DEFAULT_IR_TOPIC).value)
        self.ir_model_path = str(
            self.declare_parameter("ir_model_path", DEFAULT_IR_MODEL_PATH).value
        )
        self.dark_luminance = float(
            self.declare_parameter("dark_luminance", DEFAULT_DARK_LUMINANCE).value
        )
        # IR model runs at a higher floor: dot-blur texture generalizes to
        # border rig structures at low confidence (right-edge strip class)
        self.ir_min_confidence = float(
            self.declare_parameter("ir_min_confidence", 0.45).value
        )
        affine = self.declare_parameter(
            "ir_to_color_affine", DEFAULT_IR_TO_COLOR_AFFINE
        ).value
        self.ir_to_color_affine = [float(v) for v in affine]
        self.debug_image_topic = str(
            self.declare_parameter("debug_image_topic", DEFAULT_DEBUG_IMAGE_TOPIC).value
        )
        self.model = None
        if os.path.isfile(self.model_path):
            try:
                from ultralytics import YOLO

                self.model = YOLO(self.model_path)
                self.get_logger().info(f"loaded detector: {self.model_path}")
            except Exception as error:  # noqa: BLE001 - degrade, don't die
                self.get_logger().warn(f"detector unavailable: {error}")
        else:
            self.get_logger().warn(
                f"no model at {self.model_path}; transparent boxes disabled"
            )
        self.ir_model = None
        if os.path.isfile(self.ir_model_path):
            try:
                from ultralytics import YOLO

                self.ir_model = YOLO(self.ir_model_path)
                self.get_logger().info(f"loaded IR detector: {self.ir_model_path}")
            except Exception as error:  # noqa: BLE001 - degrade, don't die
                self.get_logger().warn(f"IR detector unavailable: {error}")
        else:
            self.get_logger().info(
                f"no IR model at {self.ir_model_path}; dark scenes fall back to RGB"
            )
        qos = QoSProfile(depth=2)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.boxes_pub = self.create_publisher(Float32MultiArray, self.boxes_topic, qos)
        self.debug_image_pub = None
        if self.debug_image_topic:
            # RELIABLE so RViz's Image display works with its default QoS
            debug_qos = QoSProfile(depth=1)
            debug_qos.reliability = ReliabilityPolicy.RELIABLE
            self.debug_image_pub = self.create_publisher(
                Image, self.debug_image_topic, debug_qos
            )
        self.color_sub = self.create_subscription(
            Image, self.color_topic, self.handle_color, qos
        )
        self.latest_ir: Image | None = None
        self.latest_ir_monotonic = 0.0
        if self.ir_model is not None:
            self.ir_sub = self.create_subscription(
                Image, self.ir_topic, self.handle_ir, qos
            )
        self.last_inference_monotonic = 0.0

    def handle_ir(self, msg: Image) -> None:
        self.latest_ir = msg
        self.latest_ir_monotonic = time.monotonic()

    def map_ir_box_to_color(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> tuple[float, float, float, float]:
        a, b, c, d, e, f = self.ir_to_color_affine
        us = [a * u + b * v + c for u, v in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        vs = [d * u + e * v + f for u, v in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        return (
            max(0.0, min(us)),
            max(0.0, min(vs)),
            min(639.0, max(us)),
            min(479.0, max(vs)),
        )

    def handle_color(self, msg: Image) -> None:
        if self.model is None:
            return
        now = time.monotonic()
        if now - self.last_inference_monotonic < 1.0 / max(0.1, self.max_rate_hz):
            return
        self.last_inference_monotonic = now
        try:
            rgb = decode_color_image(msg)
        except ValueError as error:
            self.get_logger().warn(str(error))
            return
        luminance = float(rgb[::8, ::8].mean())
        use_ir = (
            luminance < self.dark_luminance
            and self.ir_model is not None
            and self.latest_ir is not None
            and now - self.latest_ir_monotonic < 0.7
        )
        rows: list[float] = []
        count = 0
        ir_gray = None
        ir_rows: list[float] = []
        if use_ir:
            ir_msg = self.latest_ir
            ir_gray = np.frombuffer(bytes(ir_msg.data), dtype=np.uint8).reshape(
                ir_msg.height, ir_msg.step
            )[:, : ir_msg.width]
            ir_bgr = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
            result = self.ir_model.predict(
                ir_bgr, conf=self.ir_min_confidence, verbose=False
            )[0]
            for box in result.boxes:
                ix0, iy0, ix1, iy1 = (float(v) for v in box.xyxy[0])
                ir_rows.extend([ix0, iy0, ix1, iy1, float(box.conf)])
                x0, y0, x1, y1 = self.map_ir_box_to_color(ix0, iy0, ix1, iy1)
                rows.extend([x0, y0, x1, y1, float(box.conf)])
                count += 1
        else:
            result = self.model.predict(
                rgb[:, :, ::-1], conf=self.min_confidence, verbose=False
            )[0]
            for box in result.boxes:
                x0, y0, x1, y1 = (float(v) for v in box.xyxy[0])
                rows.extend([x0, y0, x1, y1, float(box.conf)])
                count += 1
        message = Float32MultiArray()
        message.layout.dim = [
            MultiArrayDimension(label="boxes", size=count, stride=5 * count),
            MultiArrayDimension(label="xyxyc", size=5, stride=5),
        ]
        message.data = rows
        self.boxes_pub.publish(message)
        if self.debug_image_pub is not None:
            if ir_gray is not None:
                annotated = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2RGB)
                draw_rows = ir_rows
                label = "IR transparent"
            else:
                annotated = np.ascontiguousarray(rgb)
                draw_rows = rows
                label = "transparent"
            for index in range(count):
                x0, y0, x1, y1, conf = draw_rows[index * 5 : index * 5 + 5]
                p0 = (int(x0), int(y0))
                p1 = (int(x1), int(y1))
                cv2.rectangle(annotated, p0, p1, (0, 255, 90), 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (int(x0), max(14, int(y0) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 90),
                    1,
                )
            debug = Image()
            debug.header = msg.header
            debug.height = int(annotated.shape[0])
            debug.width = int(annotated.shape[1])
            debug.encoding = "rgb8"
            debug.is_bigendian = False
            debug.step = int(annotated.shape[1] * 3)
            debug.data = annotated.tobytes()
            self.debug_image_pub.publish(debug)


def main() -> None:
    rclpy.init()
    node = TransparentDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
