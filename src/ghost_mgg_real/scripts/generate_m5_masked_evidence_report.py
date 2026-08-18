#!/usr/bin/env python3
"""Generate M5 real-D435 failure evidence reports inside completed target masks."""

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
SCHEMA_VERSION = "m5_masked_evidence_scene_v1"
MANIFEST_SCHEMA_VERSION = "m5_masked_evidence_manifest_v1"
EVIDENCE_CHANNELS = (
    "hole",
    "table_leakage",
    "foreground",
    "background_shift",
    "unexplained_valid",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return base_dir / path


def _ratio(mask: np.ndarray, target_pixels: int) -> float:
    if target_pixels <= 0:
        return 0.0
    return float(mask.sum() / target_pixels)


def compute_masked_evidence_summary(
    scene_id: str,
    target_mask: np.ndarray,
    current_depth: np.ndarray,
    background_depth: np.ndarray,
    leakage_tolerance_mm: int = 18,
    foreground_margin_mm: int = 45,
    target_label: str | None = None,
    shape_hint: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target_mask = target_mask.astype(bool)
    current_valid = (current_depth > 0) & target_mask
    background_valid = (background_depth > 0) & target_mask
    comparable = current_valid & background_valid
    delta = current_depth.astype(np.int32) - background_depth.astype(np.int32)

    evidence = {
        "hole": target_mask & ~current_valid,
        "table_leakage": comparable & (np.abs(delta) <= leakage_tolerance_mm),
        "foreground": comparable & (delta < -foreground_margin_mm),
        "background_shift": comparable & (delta > foreground_margin_mm),
    }
    explained = (
        evidence["hole"]
        | evidence["table_leakage"]
        | evidence["foreground"]
        | evidence["background_shift"]
    )
    evidence["unexplained_valid"] = target_mask & current_valid & ~explained

    target_pixels = int(target_mask.sum())
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "target_label": target_label,
        "shape_hint": shape_hint,
        "generated_at_utc": _utc_now(),
        "method": "formal_mask_depth_background_comparison",
        "target_pixels": target_pixels,
        "valid_depth_ratio": _ratio(current_valid, target_pixels),
        "comparable_depth_ratio": _ratio(comparable, target_pixels),
        "hole_ratio": _ratio(evidence["hole"], target_pixels),
        "table_leakage_ratio": _ratio(evidence["table_leakage"], target_pixels),
        "foreground_ratio": _ratio(evidence["foreground"], target_pixels),
        "background_shift_ratio": _ratio(evidence["background_shift"], target_pixels),
        "unexplained_valid_ratio": _ratio(evidence["unexplained_valid"], target_pixels),
        "leakage_tolerance_mm": int(leakage_tolerance_mm),
        "foreground_margin_mm": int(foreground_margin_mm),
    }
    return summary, evidence


def load_completed_mask_records(annotations_root: Path) -> list[dict[str, Any]]:
    annotations_root = Path(annotations_root)
    manifest_path = annotations_root / "masks" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for item in manifest.get("tasks", []):
        if item.get("status") != "rasterized":
            continue
        task_path = _resolve_path(str(item["task_path"]), manifest_path.parent)
        mask_path = _resolve_path(str(item["mask_path"]), manifest_path.parent)
        task = json.loads(task_path.read_text(encoding="utf-8"))
        records.append(
            {
                "scene_id": str(item["scene_id"]),
                "task_path": task_path,
                "mask_path": mask_path,
                "target_label": task.get("target_label"),
                "shape_hint": task.get("shape_hint"),
            }
        )
    return records


def _read_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {mask_path}")
    return mask > 0


def _write_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"failed to write mask: {path}")


def _write_overlay(path: Path, rgb: np.ndarray, evidence: dict[str, np.ndarray]) -> None:
    overlay = rgb.astype(np.float32).copy()
    colors = {
        "hole": np.array([255, 0, 0], dtype=np.float32),
        "table_leakage": np.array([0, 80, 255], dtype=np.float32),
        "foreground": np.array([0, 255, 0], dtype=np.float32),
        "background_shift": np.array([255, 220, 0], dtype=np.float32),
        "unexplained_valid": np.array([255, 0, 255], dtype=np.float32),
    }
    for channel, color in colors.items():
        mask = evidence[channel]
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * color
    bgr = cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"failed to write overlay: {path}")


def write_masked_evidence_outputs(
    scene_id: str,
    rgb: np.ndarray,
    target_mask: np.ndarray,
    evidence: dict[str, np.ndarray],
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    scene_dir = output_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    formal_mask_path = scene_dir / "formal_mask.png"
    overlay_path = scene_dir / "evidence_overlay.png"
    summary_path = scene_dir / "evidence_summary.json"

    _write_mask_png(formal_mask_path, target_mask)
    _write_overlay(overlay_path, rgb, evidence)

    output_summary = dict(summary)
    output_summary["outputs"] = {
        "formal_mask": f"{scene_id}/formal_mask.png",
        "evidence_overlay": f"{scene_id}/evidence_overlay.png",
        "evidence_summary": f"{scene_id}/evidence_summary.json",
    }
    summary_path.write_text(
        json.dumps(output_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_summary


def _frames_to_rgb_and_depth(frames: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rgb = decode_image_msg(frames[COLOR_TOPIC])
    depth = decode_image_msg(frames[ALIGNED_DEPTH_TOPIC])
    return rgb, depth


def generate_masked_evidence_report(
    data_dir: Path,
    annotations_root: Path,
    output_dir: Path,
    background_scene_id: str = "empty_table_001",
    leakage_tolerance_mm: int = 18,
    foreground_margin_mm: int = 45,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    background_frames = read_first_image_frames(data_dir / background_scene_id)
    _background_rgb, background_depth = _frames_to_rgb_and_depth(background_frames)

    summaries: list[dict[str, Any]] = []
    for record in load_completed_mask_records(annotations_root):
        scene_id = record["scene_id"]
        frames = read_first_image_frames(data_dir / scene_id)
        rgb, depth = _frames_to_rgb_and_depth(frames)
        target_mask = _read_mask(record["mask_path"])
        if target_mask.shape != depth.shape:
            raise ValueError(
                f"mask/depth shape mismatch for {scene_id}: {target_mask.shape} vs {depth.shape}"
            )
        summary, evidence = compute_masked_evidence_summary(
            scene_id=scene_id,
            target_mask=target_mask,
            current_depth=depth,
            background_depth=background_depth,
            leakage_tolerance_mm=leakage_tolerance_mm,
            foreground_margin_mm=foreground_margin_mm,
            target_label=record.get("target_label"),
            shape_hint=record.get("shape_hint"),
        )
        summaries.append(
            write_masked_evidence_outputs(
                scene_id=scene_id,
                rgb=rgb,
                target_mask=target_mask,
                evidence=evidence,
                summary=summary,
                output_dir=output_dir,
            )
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "source_data_dir": str(data_dir),
        "annotations_root": str(annotations_root),
        "background_scene_id": background_scene_id,
        "num_scenes": len(summaries),
        "scenes": summaries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv_summary(summaries, output_dir / "summary.csv")
    write_index_markdown(manifest, output_dir)
    return manifest


def write_csv_summary(summaries: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "scene_id",
        "target_label",
        "shape_hint",
        "target_pixels",
        "valid_depth_ratio",
        "comparable_depth_ratio",
        "hole_ratio",
        "table_leakage_ratio",
        "foreground_ratio",
        "background_shift_ratio",
        "unexplained_valid_ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})


def write_index_markdown(manifest: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# M5 Formal-Mask Evidence Report",
        "",
        "Evidence is computed only inside completed human target masks.",
        "",
        "| scene_id | label | pixels | valid | hole | leakage | foreground | overlay | mask |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for scene in manifest["scenes"]:
        outputs = scene["outputs"]
        lines.append(
            "| "
            f"{scene['scene_id']} | "
            f"{scene.get('target_label') or ''} | "
            f"{scene['target_pixels']} | "
            f"{scene['valid_depth_ratio']:.3f} | "
            f"{scene['hole_ratio']:.3f} | "
            f"{scene['table_leakage_ratio']:.3f} | "
            f"{scene['foreground_ratio']:.3f} | "
            f"[overlay]({outputs['evidence_overlay']}) | "
            f"[mask]({outputs['formal_mask']}) |"
        )
    lines.append("")
    (output_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/real_d435_m5"))
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("annotations/m5_real_d435_masks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m5_real_d435_masked_evidence"),
    )
    parser.add_argument("--background-scene-id", default="empty_table_001")
    parser.add_argument("--leakage-tolerance-mm", type=int, default=18)
    parser.add_argument("--foreground-margin-mm", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = generate_masked_evidence_report(
        data_dir=args.data_dir,
        annotations_root=args.annotations_root,
        output_dir=args.output_dir,
        background_scene_id=args.background_scene_id,
        leakage_tolerance_mm=args.leakage_tolerance_mm,
        foreground_margin_mm=args.foreground_margin_mm,
    )
    print(f"Wrote formal-mask evidence report for {manifest['num_scenes']} scenes to {args.output_dir}")


if __name__ == "__main__":
    main()
