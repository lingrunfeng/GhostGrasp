#!/usr/bin/env python3
"""Generate coarse M5 real-D435 target masks and failure evidence previews."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_m5_replay_samples import decode_image_msg, read_first_image_frames


COLOR_TOPIC = "/camera/camera/color/image_raw"
ALIGNED_DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rgb_difference_mask(
    current_rgb: np.ndarray,
    background_rgb: np.ndarray,
    rgb_threshold: int,
) -> np.ndarray:
    diff = np.abs(current_rgb.astype(np.int16) - background_rgb.astype(np.int16))
    gray_diff = diff.max(axis=2)
    return gray_diff > rgb_threshold


def _depth_difference_mask(
    current_depth: np.ndarray,
    background_depth: np.ndarray,
    depth_threshold_mm: int,
) -> np.ndarray:
    valid = (current_depth > 0) & (background_depth > 0)
    diff = np.abs(current_depth.astype(np.int32) - background_depth.astype(np.int32))
    return valid & (diff > depth_threshold_mm)


def build_target_mask(
    current_rgb: np.ndarray,
    background_rgb: np.ndarray,
    current_depth: np.ndarray,
    background_depth: np.ndarray,
    rgb_threshold: int = 22,
    depth_threshold_mm: int = 35,
    morph_kernel: int = 5,
) -> np.ndarray:
    rgb_mask = _rgb_difference_mask(current_rgb, background_rgb, rgb_threshold)
    depth_mask = _depth_difference_mask(current_depth, background_depth, depth_threshold_mm)
    mask = rgb_mask | depth_mask
    if morph_kernel > 1:
        kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
        mask_u8 = (mask.astype(np.uint8) * 255)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask = mask_u8 > 0
    return mask.astype(bool)


def compute_evidence_channels(
    target_mask: np.ndarray,
    current_depth: np.ndarray,
    background_depth: np.ndarray,
    leakage_tolerance_mm: int = 18,
    foreground_margin_mm: int = 45,
) -> dict[str, np.ndarray]:
    current_valid = current_depth > 0
    background_valid = background_depth > 0
    comparable = target_mask & current_valid & background_valid
    delta = current_depth.astype(np.int32) - background_depth.astype(np.int32)

    hole = target_mask & ~current_valid
    table_leakage = comparable & (np.abs(delta) <= leakage_tolerance_mm)
    foreground = comparable & (delta < -foreground_margin_mm)
    background_shift = comparable & (delta > foreground_margin_mm)

    return {
        "hole": hole,
        "table_leakage": table_leakage,
        "foreground": foreground,
        "background_shift": background_shift,
    }


def _ratio(mask: np.ndarray, target_pixels: int) -> float:
    if target_pixels <= 0:
        return 0.0
    return float(mask.sum() / target_pixels)


def _write_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def _write_overlay(path: Path, rgb: np.ndarray, evidence: dict[str, np.ndarray]) -> None:
    overlay = rgb.astype(np.float32).copy()
    colors = {
        "hole": np.array([255, 0, 0], dtype=np.float32),
        "table_leakage": np.array([0, 80, 255], dtype=np.float32),
        "foreground": np.array([0, 255, 0], dtype=np.float32),
        "background_shift": np.array([255, 220, 0], dtype=np.float32),
    }
    for channel, color in colors.items():
        mask = evidence[channel]
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * color
    bgr = cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def write_evidence_outputs(
    scene_id: str,
    rgb: np.ndarray,
    target_mask: np.ndarray,
    evidence: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    scene_dir = output_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    _write_mask_png(scene_dir / "target_mask.png", target_mask)
    _write_overlay(scene_dir / "evidence_overlay.png", rgb, evidence)

    target_pixels = int(target_mask.sum())
    summary = {
        "schema_version": "m5_real_evidence_preview_scene_v1",
        "scene_id": scene_id,
        "generated_at_utc": _utc_now(),
        "method": "coarse_background_difference_preview",
        "target_pixels": target_pixels,
        "hole_ratio": _ratio(evidence["hole"], target_pixels),
        "table_leakage_ratio": _ratio(evidence["table_leakage"], target_pixels),
        "foreground_ratio": _ratio(evidence["foreground"], target_pixels),
        "background_shift_ratio": _ratio(evidence["background_shift"], target_pixels),
        "outputs": {
            "target_mask": f"{scene_id}/target_mask.png",
            "evidence_overlay": f"{scene_id}/evidence_overlay.png",
        },
    }
    (scene_dir / "evidence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _frames_to_rgb_and_depth(frames: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rgb = decode_image_msg(frames[COLOR_TOPIC])
    depth = decode_image_msg(frames[ALIGNED_DEPTH_TOPIC])
    return rgb, depth


def generate_evidence_previews(
    data_dir: Path,
    output_dir: Path,
    background_scene_id: str = "empty_table_001",
) -> dict[str, Any]:
    background_frames = read_first_image_frames(data_dir / background_scene_id)
    background_rgb, background_depth = _frames_to_rgb_and_depth(background_frames)

    scene_dirs = sorted(path.parent for path in data_dir.glob("*/metadata.yaml"))
    summaries: list[dict[str, Any]] = []
    for scene_dir in scene_dirs:
        frames = read_first_image_frames(scene_dir)
        rgb, depth = _frames_to_rgb_and_depth(frames)
        target_mask = build_target_mask(rgb, background_rgb, depth, background_depth)
        evidence = compute_evidence_channels(target_mask, depth, background_depth)
        summaries.append(write_evidence_outputs(scene_dir.name, rgb, target_mask, evidence, output_dir))

    manifest = {
        "schema_version": "m5_real_evidence_preview_manifest_v1",
        "generated_at_utc": _utc_now(),
        "source_data_dir": str(data_dir),
        "background_scene_id": background_scene_id,
        "num_scenes": len(summaries),
        "scenes": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv_summary(summaries, output_dir / "summary.csv")
    write_index_markdown(manifest, output_dir)
    return manifest


def write_csv_summary(summaries: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "scene_id",
            "target_pixels",
            "hole_ratio",
            "table_leakage_ratio",
            "foreground_ratio",
            "background_shift_ratio",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary[key] for key in fieldnames})


def write_index_markdown(manifest: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# M5 Real Evidence Preview",
        "",
        "These masks are coarse background-difference previews, not final paper labels.",
        "",
        "| scene_id | target_pixels | hole | leakage | foreground | overlay | mask |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for scene in manifest["scenes"]:
        outputs = scene["outputs"]
        lines.append(
            "| "
            f"{scene['scene_id']} | "
            f"{scene['target_pixels']} | "
            f"{scene['hole_ratio']:.3f} | "
            f"{scene['table_leakage_ratio']:.3f} | "
            f"{scene['foreground_ratio']:.3f} | "
            f"[overlay]({outputs['evidence_overlay']}) | "
            f"[mask]({outputs['target_mask']}) |"
        )
    lines.append("")
    (output_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/real_d435_m5"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m5_real_d435_evidence_preview"),
    )
    parser.add_argument("--background-scene-id", default="empty_table_001")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = generate_evidence_previews(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        background_scene_id=args.background_scene_id,
    )
    print(f"Wrote coarse evidence previews for {manifest['num_scenes']} scenes to {args.output_dir}")


if __name__ == "__main__":
    main()
