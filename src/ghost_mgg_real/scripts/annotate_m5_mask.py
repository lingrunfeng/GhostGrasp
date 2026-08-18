#!/usr/bin/env python3
"""Click polygon vertices to complete one M5 real-D435 mask annotation task."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rasterize_m5_mask_annotations import rasterize_annotation_task


IMAGE_KEYS = ("color", "depth", "aligned_depth", "infra1", "infra2", "seed_mask")


def task_path_for_scene(annotations_root: Path, scene_id: str) -> Path:
    return Path(annotations_root) / "tasks" / f"{scene_id}.json"


def load_task(task_path: Path) -> dict[str, Any]:
    return json.loads(Path(task_path).read_text(encoding="utf-8"))


def _write_task(task_path: Path, task: dict[str, Any]) -> None:
    Path(task_path).write_text(
        json.dumps(task, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalize_polygon(polygon: list[tuple[int, int]] | list[list[int]]) -> list[list[int]]:
    normalized = [[int(round(x)), int(round(y))] for x, y in polygon]
    if len(normalized) < 3:
        raise ValueError("polygon must contain at least 3 points")
    return normalized


def save_completed_polygon(
    task_path: Path,
    annotations_root: Path,
    polygon: list[tuple[int, int]] | list[list[int]],
) -> dict[str, Any]:
    task_path = Path(task_path)
    task = load_task(task_path)
    task["status"] = "complete"
    task["polygons"] = [_normalize_polygon(polygon)]
    _write_task(task_path, task)
    return rasterize_annotation_task(task_path, annotations_root)


def _resolve_image_path(path_value: str, task_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    task_relative = Path(task_path).parent / path
    if task_relative.exists():
        return task_relative
    return cwd_path


def image_path_for_task(task: dict[str, Any], task_path: Path, image_key: str) -> Path:
    image_paths = task.get("image_paths", {})
    if image_key not in image_paths:
        available = ", ".join(sorted(image_paths)) or "none"
        raise KeyError(f"task has no image key '{image_key}', available: {available}")
    return _resolve_image_path(str(image_paths[image_key]), task_path)


def _read_display_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _initial_points(task: dict[str, Any]) -> list[tuple[int, int]]:
    polygons = task.get("polygons", [])
    if not polygons:
        return []
    return [(int(x), int(y)) for x, y in polygons[0]]


@dataclass
class AnnotatorState:
    image: np.ndarray
    points: list[tuple[int, int]] = field(default_factory=list)

    def draw(self) -> np.ndarray:
        canvas = self.image.copy()
        for index, point in enumerate(self.points):
            cv2.circle(canvas, point, 4, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                str(index + 1),
                (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
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

    def add(self, x: int, y: int) -> None:
        self.points.append((int(x), int(y)))

    def undo(self) -> None:
        if self.points:
            self.points.pop()

    def reset(self) -> None:
        self.points.clear()


def _mouse_callback(event: int, x: int, y: int, _flags: int, state: AnnotatorState) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        state.add(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.undo()


def run_interactive_annotation(
    task_path: Path,
    annotations_root: Path,
    image_key: str,
) -> dict[str, Any] | None:
    task = load_task(task_path)
    image_path = image_path_for_task(task, task_path, image_key)
    state = AnnotatorState(
        image=_read_display_image(image_path),
        points=_initial_points(task),
    )
    window = f"M5 mask annotation: {task.get('scene_id', Path(task_path).stem)} [{image_key}]"

    print("Controls:")
    print("  left click : add polygon point")
    print("  right/u    : undo last point")
    print("  r          : reset polygon")
    print("  s          : save task and rasterize mask")
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
                summary = save_completed_polygon(task_path, annotations_root, state.points)
                print(f"Saved mask: {summary.get('mask_path')}")
                print(f"Saved overlay: {summary.get('overlay_path')}")
                return summary
    finally:
        cv2.destroyWindow(window)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("annotations/m5_real_d435_masks"),
    )
    parser.add_argument("--image-key", choices=IMAGE_KEYS, default="color")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    task_path = task_path_for_scene(args.annotations_root, args.scene_id)
    if not task_path.exists():
        raise FileNotFoundError(f"missing annotation task: {task_path}")
    run_interactive_annotation(task_path, args.annotations_root, args.image_key)


if __name__ == "__main__":
    main()
