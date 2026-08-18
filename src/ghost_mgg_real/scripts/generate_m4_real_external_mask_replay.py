#!/usr/bin/env python3
"""Generate M4 live-hypothesis-style rows from real D435 bags plus external masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CORE_PYTHON = REPO_ROOT / "src" / "ghost_mgg_core" / "python"
for path in (SCRIPT_DIR, CORE_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from extract_m5_replay_samples import read_first_image_frames
from generate_m5_masked_evidence_report import (
    compute_masked_evidence_summary,
    load_completed_mask_records,
)
from generate_m5_real_ranking_report import (
    _frames_to_rgb_and_depth,
    _read_mask,
    load_failure_aware_weights,
    rank_real_scene,
)
from ghost_mgg_core_py.ghost_mgg_v0 import GHOST_MGG_V0_WEIGHTS


SCHEMA_VERSION = "m4_real_external_mask_replay_v1"
CONTRACT_NAME = "external_mask_replay_to_live_hypotheses"
PROVENANCE = "m4_real_external_mask_replay:no_truth:external_mask_contract:formal_mask_depth_background"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rank_external_mask_scene(
    *,
    scene_id: str,
    target_label: str | None,
    shape_hint: str | None,
    target_mask: np.ndarray,
    current_depth: np.ndarray,
    background_depth: np.ndarray,
    top_k: int = 3,
    failure_aware_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    summary, _evidence = compute_masked_evidence_summary(
        scene_id=scene_id,
        target_mask=target_mask,
        current_depth=current_depth,
        background_depth=background_depth,
        target_label=target_label,
        shape_hint=shape_hint,
    )
    ranked_rows = rank_real_scene(
        scene_id=scene_id,
        target_label=target_label,
        shape_hint=shape_hint,
        target_mask=target_mask,
        current_depth=current_depth,
        background_depth=background_depth,
        top_k=top_k,
        failure_aware_weights=failure_aware_weights,
    )
    live_rows: list[dict[str, Any]] = []
    for row in ranked_rows:
        if row.ranker != "failure_aware":
            continue
        live_rows.append(
            {
                "scene_id": row.scene_id,
                "target_label": row.target_label,
                "shape_hint": row.shape_hint,
                "ranker": row.ranker,
                "rank": int(row.rank),
                "hypothesis_id": row.hypothesis_id,
                "shape_type": row.shape_type,
                "pose_image": {
                    "center_u": float(row.center_u),
                    "center_v": float(row.center_v),
                },
                "center_u": float(row.center_u),
                "center_v": float(row.center_v),
                "dimensions_px": {
                    "width": float(row.size_u_px),
                    "height": float(row.size_v_px),
                },
                "size_u_px": float(row.size_u_px),
                "size_v_px": float(row.size_v_px),
                "score": {
                    "visual": float(row.visual_score),
                    "failure": float(row.failure_score),
                    "depth": float(row.depth_score),
                    "total": float(row.total_score),
                },
                "visual_score": float(row.visual_score),
                "failure_score": float(row.failure_score),
                "depth_score": float(row.depth_score),
                "total_score": float(row.total_score),
                "failure_terms": {
                    "inside_hole": float(row.failure_inside_hole),
                    "inside_table_leakage": float(row.failure_inside_table_leakage),
                    "boundary_edge": float(row.failure_boundary_edge),
                    "boundary_flying_point": float(row.failure_boundary_flying_point),
                    "outside_hole_penalty": float(row.failure_outside_hole_penalty),
                    "outside_table_leakage_penalty": float(
                        row.failure_outside_table_leakage_penalty
                    ),
                    "total_check": float(row.failure_total_check),
                },
                "failure_inside_hole": float(row.failure_inside_hole),
                "failure_inside_table_leakage": float(row.failure_inside_table_leakage),
                "failure_boundary_edge": float(row.failure_boundary_edge),
                "failure_boundary_flying_point": float(row.failure_boundary_flying_point),
                "failure_outside_hole_penalty": float(row.failure_outside_hole_penalty),
                "failure_outside_table_leakage_penalty": float(
                    row.failure_outside_table_leakage_penalty
                ),
                "failure_total_check": float(row.failure_total_check),
                "mask_pixels": int(summary["target_pixels"]),
                "valid_depth_ratio": float(summary["valid_depth_ratio"]),
                "hole_ratio": float(summary["hole_ratio"]),
                "table_leakage_ratio": float(summary["table_leakage_ratio"]),
                "foreground_ratio": float(summary["foreground_ratio"]),
                "validation_state": row.validation_state,
                "provenance": PROVENANCE,
            }
        )
    return live_rows[: max(1, int(top_k))]


def run_real_external_mask_replay(
    *,
    data_dir: Path,
    annotations_root: Path,
    output_dir: Path,
    background_scene_id: str = "empty_table_001",
    top_k: int = 3,
    failure_aware_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    background_frames = read_first_image_frames(data_dir / background_scene_id)
    _background_rgb, background_depth = _frames_to_rgb_and_depth(background_frames)

    rows: list[dict[str, Any]] = []
    for record in load_completed_mask_records(annotations_root):
        scene_id = str(record["scene_id"])
        frames = read_first_image_frames(data_dir / scene_id)
        _rgb, depth = _frames_to_rgb_and_depth(frames)
        target_mask = _read_mask(record["mask_path"])
        if target_mask.shape != depth.shape:
            raise ValueError(
                f"mask/depth shape mismatch for {scene_id}: {target_mask.shape} vs {depth.shape}"
            )
        rows.extend(
            rank_external_mask_scene(
                scene_id=scene_id,
                target_label=record.get("target_label"),
                shape_hint=record.get("shape_hint"),
                target_mask=target_mask,
                current_depth=depth,
                background_depth=background_depth,
                top_k=top_k,
                failure_aware_weights=failure_aware_weights,
            )
        )

    write_external_mask_replay_reports(
        rows,
        output_dir,
        source_data_dir=data_dir,
        annotations_root=annotations_root,
        background_scene_id=background_scene_id,
        top_k=top_k,
        failure_aware_weights=GHOST_MGG_V0_WEIGHTS
        if failure_aware_weights is None
        else failure_aware_weights,
    )
    return rows


def write_external_mask_replay_reports(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    source_data_dir: Path,
    annotations_root: Path,
    background_scene_id: str,
    top_k: int,
    failure_aware_weights: dict[str, float] | None = None,
) -> None:
    if not rows:
        raise ValueError("no external-mask replay hypotheses to write")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "generated_at_utc": _utc_now(),
        "source_data_dir": str(source_data_dir),
        "annotations_root": str(annotations_root),
        "background_scene_id": background_scene_id,
        "top_k": int(top_k),
        "num_scenes": len({row["scene_id"] for row in rows}),
        "num_hypotheses": len(rows),
        "failure_aware_weights": _stable_weights_payload(failure_aware_weights),
        "rows": rows,
    }
    json_path = output_dir / "m4_real_live_hypotheses.json"
    csv_path = output_dir / "m4_real_live_hypotheses.csv"
    index_path = output_dir / "index.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(rows, csv_path)
    _write_index_markdown(payload, index_path)


def _write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "scene_id",
        "target_label",
        "shape_hint",
        "ranker",
        "rank",
        "hypothesis_id",
        "shape_type",
        "center_u",
        "center_v",
        "size_u_px",
        "size_v_px",
        "visual_score",
        "failure_score",
        "depth_score",
        "total_score",
        "failure_inside_hole",
        "failure_inside_table_leakage",
        "failure_outside_hole_penalty",
        "failure_outside_table_leakage_penalty",
        "mask_pixels",
        "valid_depth_ratio",
        "hole_ratio",
        "table_leakage_ratio",
        "foreground_ratio",
        "validation_state",
        "provenance",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_index_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# M4 Real External-Mask Replay",
        "",
        f"- Contract: `{payload['contract']}`",
        f"- Scenes: {payload['num_scenes']}",
        f"- Hypotheses: {payload['num_hypotheses']}",
        "",
        "This report replays real D435 observations through the external mask contract.",
        "It uses completed formal masks and does not query Gazebo or robot target truth.",
        "",
        "| scene_id | rank | hypothesis | shape | total | failure | hole | leakage | provenance |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['scene_id']} | "
            f"{row['rank']} | "
            f"{row['hypothesis_id']} | "
            f"{row['shape_type']} | "
            f"{row['total_score']:.3f} | "
            f"{row['failure_score']:.3f} | "
            f"{row['hole_ratio']:.3f} | "
            f"{row['table_leakage_ratio']:.3f} | "
            f"{row['provenance']} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _stable_weights_payload(weights: dict[str, float] | None) -> dict[str, float]:
    source = GHOST_MGG_V0_WEIGHTS if weights is None else weights
    return {
        name: float(source.get(name, 0.0))
        for name in ("visual", "failure", "depth", "physical", "grasp", "prior")
    }


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
        default=Path("reports/m4_real_external_mask_replay"),
    )
    parser.add_argument("--background-scene-id", default="empty_table_001")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--failure-aware-weights-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = run_real_external_mask_replay(
        data_dir=args.data_dir,
        annotations_root=args.annotations_root,
        output_dir=args.output_dir,
        background_scene_id=args.background_scene_id,
        top_k=args.top_k,
        failure_aware_weights=load_failure_aware_weights(args.failure_aware_weights_json),
    )
    print(
        f"Wrote {len(rows)} M4 real external-mask replay hypotheses to "
        f"{args.output_dir / 'm4_real_live_hypotheses.json'}"
    )


if __name__ == "__main__":
    main()
