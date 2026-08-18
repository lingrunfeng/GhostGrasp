#!/usr/bin/env python3
"""Rasterize completed M5 real-D435 polygon annotations into binary masks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MANIFEST_SCHEMA_VERSION = "m5_mask_raster_manifest_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _resolve_path(path_value: str, task_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    task_relative = task_path.parent / path
    if task_relative.exists():
        return task_relative
    return cwd_path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rasterize_polygons(
    polygons: list[list[list[int | float]]],
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        points = np.asarray(polygon, dtype=np.float32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask > 0


def _write_overlay(path: Path, image_bgr: np.ndarray, mask: np.ndarray) -> None:
    overlay = image_bgr.astype(np.float32).copy()
    green = np.array([0, 210, 40], dtype=np.float32)
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * green
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    output = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.drawContours(output, contours, -1, (0, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"failed to write overlay: {path}")


def _skip_summary(scene_id: str, reason: str, task_path: Path) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "status": "skipped",
        "reason": reason,
        "task_path": _portable_path(task_path),
    }


def rasterize_annotation_task(
    task_path: Path,
    annotations_root: Path,
) -> dict[str, Any]:
    task_path = Path(task_path)
    annotations_root = Path(annotations_root)
    task = _read_json(task_path)
    scene_id = str(task.get("scene_id", task_path.stem))
    polygons = task.get("polygons", [])

    if task.get("status") != "complete":
        return _skip_summary(scene_id, "task status is not complete", task_path)
    if not polygons:
        return _skip_summary(scene_id, "no polygons", task_path)

    color_path_value = task.get("image_paths", {}).get("color")
    if not color_path_value:
        return _skip_summary(scene_id, "missing color image path", task_path)

    color_path = _resolve_path(str(color_path_value), task_path)
    image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image is None:
        return _skip_summary(scene_id, f"failed to read color image: {color_path}", task_path)

    height, width = image.shape[:2]
    mask = rasterize_polygons(polygons, height=height, width=width)
    scene_dir = annotations_root / "masks" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    mask_path = scene_dir / "target_mask.png"
    overlay_path = scene_dir / "annotation_overlay.png"
    summary_path = scene_dir / "mask_summary.json"

    if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"failed to write mask: {mask_path}")
    _write_overlay(overlay_path, image, mask)

    summary = {
        "schema_version": "m5_mask_summary_v1",
        "scene_id": scene_id,
        "status": "rasterized",
        "generated_at_utc": _utc_now(),
        "task_path": _portable_path(task_path),
        "color_path": _portable_path(color_path),
        "mask_path": _portable_path(mask_path),
        "overlay_path": _portable_path(overlay_path),
        "height": int(height),
        "width": int(width),
        "num_polygons": int(len(polygons)),
        "mask_pixels": int(mask.sum()),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def rasterize_annotations(annotations_root: Path) -> dict[str, Any]:
    annotations_root = Path(annotations_root)
    tasks_dir = annotations_root / "tasks"
    mask_root = annotations_root / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        rasterize_annotation_task(task_path, annotations_root)
        for task_path in sorted(tasks_dir.glob("*.json"))
    ]
    rasterized = [item for item in summaries if item["status"] == "rasterized"]
    skipped = [item for item in summaries if item["status"] == "skipped"]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "annotations_root": _portable_path(annotations_root),
        "num_tasks": len(summaries),
        "num_rasterized": len(rasterized),
        "num_skipped": len(skipped),
        "tasks": summaries,
    }
    (mask_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("annotations/m5_real_d435_masks"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = rasterize_annotations(args.annotations_root)
    print(
        f"Wrote mask manifest: {args.annotations_root / 'masks' / 'manifest.json'} "
        f"({manifest['num_rasterized']} rasterized, {manifest['num_skipped']} skipped)"
    )


if __name__ == "__main__":
    main()
