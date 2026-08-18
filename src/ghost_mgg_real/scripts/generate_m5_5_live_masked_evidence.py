#!/usr/bin/env python3
"""Generate masked target evidence for one live M5.5 snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_m5_masked_evidence_report import (  # noqa: E402
    compute_masked_evidence_summary,
    write_masked_evidence_outputs,
)


def _read_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {mask_path}")
    return mask > 0


def _read_rgb(snapshot_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    color_path = snapshot_dir / "color.png"
    bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def estimate_table_depth_image(depth: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    valid_outside = (depth > 0) & ~target_mask.astype(bool)
    if not valid_outside.any():
        raise ValueError("cannot estimate table depth: no valid outside-mask depth")
    table_depth_mm = int(round(float(np.median(depth[valid_outside]))))
    return np.full(depth.shape, table_depth_mm, dtype=np.uint16)


def generate_live_masked_evidence(
    *,
    scene_id: str,
    snapshot_dir: Path,
    mask_path: Path,
    output_dir: Path,
    target_label: str = "target",
    shape_hint: str = "unknown",
    leakage_tolerance_mm: int = 18,
    foreground_margin_mm: int = 45,
) -> dict[str, Any]:
    snapshot_dir = Path(snapshot_dir)
    mask_path = Path(mask_path)
    depth_path = snapshot_dir / "aligned_depth_raw.npy"
    if not depth_path.exists():
        raise FileNotFoundError(
            f"missing raw aligned depth: {depth_path}; recapture snapshot with live mode"
        )
    depth = np.load(depth_path)
    target_mask = _read_mask(mask_path)
    if target_mask.shape != depth.shape:
        raise ValueError(
            f"mask/depth shape mismatch: {target_mask.shape} vs {depth.shape}"
        )
    table_depth = estimate_table_depth_image(depth, target_mask)
    summary, evidence = compute_masked_evidence_summary(
        scene_id=scene_id,
        target_mask=target_mask,
        current_depth=depth,
        background_depth=table_depth,
        leakage_tolerance_mm=leakage_tolerance_mm,
        foreground_margin_mm=foreground_margin_mm,
        target_label=target_label,
        shape_hint=shape_hint,
    )
    summary["method"] = "live_mask_estimated_table_depth"
    summary["estimated_table_depth_mm"] = int(table_depth[0, 0])
    rgb = _read_rgb(snapshot_dir, target_mask.shape)
    return write_masked_evidence_outputs(
        scene_id=scene_id,
        rgb=rgb,
        target_mask=target_mask,
        evidence=evidence,
        summary=summary,
        output_dir=Path(output_dir),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--mask-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-label", default="target")
    parser.add_argument("--shape-hint", default="unknown")
    parser.add_argument("--leakage-tolerance-mm", type=int, default=18)
    parser.add_argument("--foreground-margin-mm", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = generate_live_masked_evidence(
        scene_id=args.scene_id,
        snapshot_dir=args.snapshot_dir,
        mask_path=args.mask_path,
        output_dir=args.output_dir,
        target_label=args.target_label,
        shape_hint=args.shape_hint,
        leakage_tolerance_mm=args.leakage_tolerance_mm,
        foreground_margin_mm=args.foreground_margin_mm,
    )
    print(
        "M5.5 live masked evidence: "
        f"{summary['scene_id']} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
