#!/usr/bin/env python3
"""Create a binary target mask directly from one image.

Interactive mode is intentionally simple: click polygon points, press `s` to
save, `u`/right-click to undo, and `q`/Esc to quit without saving.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_display_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _normalize_polygon(polygon: list[tuple[int, int]] | list[list[int]]) -> np.ndarray:
    points = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
    if points.shape[0] < 3:
        raise ValueError("polygon must contain at least 3 points")
    return points


def _overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    color = np.array([0, 255, 255], dtype=np.float32)
    selected = mask > 0
    overlay[selected] = 0.45 * overlay[selected] + 0.55 * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_polygon_mask(
    *,
    image_path: Path,
    output_path: Path,
    polygon: list[tuple[int, int]] | list[list[int]],
) -> dict[str, Any]:
    image_path = Path(image_path)
    output_path = Path(output_path)
    image = _read_display_image(image_path)
    points = _normalize_polygon(polygon)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), mask):
        raise RuntimeError(f"failed to write mask: {output_path}")
    overlay_path = output_path.with_name(output_path.stem + "_overlay.png")
    if not cv2.imwrite(str(overlay_path), _overlay_mask(image, mask)):
        raise RuntimeError(f"failed to write overlay: {overlay_path}")

    summary = {
        "schema_version": "direct_mask_annotation_v1",
        "generated_at_utc": _utc_now(),
        "image_path": str(image_path),
        "mask_path": str(output_path),
        "overlay_path": str(overlay_path),
        "polygon": points.tolist(),
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "mask_pixels": int((mask > 0).sum()),
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


@dataclass
class _InteractiveState:
    image: np.ndarray
    points: list[tuple[int, int]] = field(default_factory=list)

    def add(self, x: int, y: int) -> None:
        self.points.append((int(x), int(y)))

    def undo(self) -> None:
        if self.points:
            self.points.pop()

    def reset(self) -> None:
        self.points.clear()

    def draw(self) -> np.ndarray:
        canvas = self.image.copy()
        for index, point in enumerate(self.points):
            cv2.circle(canvas, point, 4, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                str(index + 1),
                (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        for start, end in zip(self.points, self.points[1:]):
            cv2.line(canvas, start, end, (0, 255, 255), 2)
        if len(self.points) >= 3:
            cv2.line(canvas, self.points[-1], self.points[0], (0, 200, 255), 1)
        cv2.putText(
            canvas,
            "Left click: add | Right/u: undo | r: reset | s: save | q/Esc: quit",
            (10, max(20, canvas.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas


def _mouse_callback(event: int, x: int, y: int, _flags: int, state: _InteractiveState) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        state.add(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.undo()


def run_interactive(image_path: Path, output_path: Path) -> dict[str, Any] | None:
    state = _InteractiveState(image=_read_display_image(image_path))
    window = f"Direct mask: {Path(image_path).name}"
    print("Controls:")
    print("  left click : add polygon point")
    print("  right/u    : undo last point")
    print("  r          : reset polygon")
    print("  s          : save mask")
    print("  q or Esc   : quit without saving")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _mouse_callback, state)
    try:
        while True:
            cv2.imshow(window, state.draw())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                print("Quit without saving.")
                return None
            if key == ord("u"):
                state.undo()
            elif key == ord("r"):
                state.reset()
            elif key == ord("s"):
                if len(state.points) < 3:
                    print("Need at least 3 points before saving.")
                    continue
                summary = save_polygon_mask(
                    image_path=image_path,
                    output_path=output_path,
                    polygon=state.points,
                )
                print(f"Saved mask: {summary['mask_path']}")
                print(f"Saved overlay: {summary['overlay_path']}")
                return summary
    finally:
        cv2.destroyWindow(window)


def _parse_polygon(values: list[str]) -> list[tuple[int, int]]:
    polygon: list[tuple[int, int]] = []
    for value in values:
        if "," not in value:
            raise ValueError(f"polygon point must be x,y: {value}")
        x_text, y_text = value.split(",", 1)
        polygon.append((int(round(float(x_text))), int(round(float(y_text)))))
    return polygon


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--polygon",
        nargs="*",
        help="Optional non-interactive polygon points as x,y x,y x,y ...",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.polygon:
        summary = save_polygon_mask(
            image_path=args.image,
            output_path=args.output,
            polygon=_parse_polygon(args.polygon),
        )
        print(f"Direct mask saved: {summary['mask_path']}")
    else:
        run_interactive(args.image, args.output)


if __name__ == "__main__":
    main()
