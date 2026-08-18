#!/usr/bin/env python3
"""Rank primitive hypotheses on real M5 D435 masks with silhouette/failure rankers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CORE_PYTHON = REPO_ROOT / "src" / "ghost_mgg_core" / "python"
for path in (SCRIPT_DIR, CORE_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from extract_m5_replay_samples import decode_image_msg, read_first_image_frames
from generate_m5_masked_evidence_report import load_completed_mask_records
from ghost_mgg_core_py.evidence.types import EvidenceMaps
from ghost_mgg_core_py.ghost_mgg_v0 import (
    GHOST_MGG_V0_WEIGHTS,
    SILHOUETTE_ONLY_WEIGHTS,
    GhostMGGV0Config,
    run_ghost_mgg_v0,
)
from ghost_mgg_core_py.hypotheses.hypothesis_generator import generate_local_hypotheses
from ghost_mgg_core_py.hypotheses.primitives import PrimitiveHypothesis
from ghost_mgg_core_py.rendering.proxy_renderer import render_proxy
from ghost_mgg_core_py.scoring.score_terms import failure_likelihood_breakdown


COLOR_TOPIC = "/camera/camera/color/image_raw"
ALIGNED_DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"


@dataclass(frozen=True)
class RealRankingRow:
    scene_id: str
    target_label: str | None
    shape_hint: str | None
    ranker: str
    rank: int
    hypothesis_id: str
    shape_type: str
    center_u: float
    center_v: float
    size_u_px: float
    size_v_px: float
    visual_score: float
    failure_score: float
    failure_inside_hole: float
    failure_inside_table_leakage: float
    failure_boundary_edge: float
    failure_boundary_flying_point: float
    failure_outside_hole_penalty: float
    failure_outside_table_leakage_penalty: float
    failure_total_check: float
    depth_score: float
    total_score: float
    validation_state: str


@dataclass(frozen=True)
class Top1ComparisonRow:
    scene_id: str
    target_label: str | None
    shape_hint: str | None
    silhouette_top: str
    failure_top: str
    silhouette_shape: str
    failure_shape: str
    top1_changed: bool
    shape_changed: bool
    silhouette_visual_score: float
    failure_visual_score: float
    silhouette_failure_score: float
    failure_failure_score: float
    visual_score_delta: float
    failure_score_delta: float


def evidence_maps_from_depth_background(
    target_mask: np.ndarray,
    current_depth: np.ndarray,
    background_depth: np.ndarray,
    leakage_tolerance_mm: int = 18,
    foreground_margin_mm: int = 45,
) -> EvidenceMaps:
    target = np.asarray(target_mask, dtype=bool)
    current_valid = (current_depth > 0) & target
    background_valid = (background_depth > 0) & target
    comparable = current_valid & background_valid
    delta = current_depth.astype(np.int32) - background_depth.astype(np.int32)

    hole = target & ~current_valid
    table_leakage = comparable & (np.abs(delta) <= leakage_tolerance_mm)
    foreground = comparable & (delta < -foreground_margin_mm)
    zeros = np.zeros(target.shape, dtype=np.float32)

    return EvidenceMaps(
        valid=current_valid.astype(np.float32),
        hole=hole.astype(np.float32),
        table_leakage=table_leakage.astype(np.float32),
        edge=zeros.copy(),
        flying_point=zeros.copy(),
        foreground_support=foreground.astype(np.float32),
    )


def load_failure_aware_weights(weights_json: Path | None) -> dict[str, float]:
    if weights_json is None:
        return dict(GHOST_MGG_V0_WEIGHTS)

    payload = json.loads(Path(weights_json).read_text(encoding="utf-8"))
    raw_weights = payload.get("best_weights", payload)
    required = {"visual", "failure", "depth"}
    missing = required - set(raw_weights)
    if missing:
        raise ValueError(f"weights JSON missing keys: {sorted(missing)}")

    return {
        "visual": float(raw_weights["visual"]),
        "failure": float(raw_weights["failure"]),
        "depth": float(raw_weights["depth"]),
        "physical": 0.0,
        "grasp": 0.0,
        "prior": 0.0,
    }


def _ranking_rows(
    *,
    scene_id: str,
    target_label: str | None,
    shape_hint: str | None,
    ranker: str,
    ranked,
    target_mask: np.ndarray,
    evidence: EvidenceMaps,
) -> list[RealRankingRow]:
    rows = []
    target = np.asarray(target_mask, dtype=bool)
    for rank_index, item in enumerate(ranked, start=1):
        hypothesis = item.hypothesis
        score = item.score
        rendered = render_proxy(hypothesis, target.shape)
        failure_terms = failure_likelihood_breakdown(rendered, target, evidence)
        rows.append(
            RealRankingRow(
                scene_id=scene_id,
                target_label=target_label,
                shape_hint=shape_hint,
                ranker=ranker,
                rank=rank_index,
                hypothesis_id=str(hypothesis.hypothesis_id),
                shape_type=str(hypothesis.shape_type),
                center_u=float(hypothesis.center_uv[0]),
                center_v=float(hypothesis.center_uv[1]),
                size_u_px=float(hypothesis.size_px[0]),
                size_v_px=float(hypothesis.size_px[1]),
                visual_score=float(score.visual),
                failure_score=float(score.failure),
                failure_inside_hole=float(failure_terms.inside_hole),
                failure_inside_table_leakage=float(failure_terms.inside_table_leakage),
                failure_boundary_edge=float(failure_terms.boundary_edge),
                failure_boundary_flying_point=float(failure_terms.boundary_flying_point),
                failure_outside_hole_penalty=float(failure_terms.outside_hole_penalty),
                failure_outside_table_leakage_penalty=float(
                    failure_terms.outside_table_leakage_penalty
                ),
                failure_total_check=float(failure_terms.total),
                depth_score=float(score.depth),
                total_score=float(score.total),
                validation_state=item.validation_state,
            )
        )
    return rows


def rank_real_scene(
    scene_id: str,
    target_label: str | None,
    shape_hint: str | None,
    target_mask: np.ndarray,
    current_depth: np.ndarray,
    background_depth: np.ndarray,
    *,
    hypotheses: list[PrimitiveHypothesis] | None = None,
    top_k: int = 3,
    leakage_tolerance_mm: int = 18,
    foreground_margin_mm: int = 45,
    failure_aware_weights: dict[str, float] | None = None,
) -> list[RealRankingRow]:
    target = np.asarray(target_mask, dtype=bool)
    evidence = evidence_maps_from_depth_background(
        target,
        current_depth,
        background_depth,
        leakage_tolerance_mm=leakage_tolerance_mm,
        foreground_margin_mm=foreground_margin_mm,
    )
    candidates = hypotheses or generate_local_hypotheses(
        target,
        shape_types=("box", "cylinder"),
        scale_factors=(0.65, 0.80, 0.95, 1.00, 1.10, 1.25),
        depth_m=1.0,
        height_m=0.08,
    )
    config = GhostMGGV0Config(top_k=top_k)
    silhouette = run_ghost_mgg_v0(
        target,
        evidence,
        config=config,
        weights=SILHOUETTE_ONLY_WEIGHTS,
        hypotheses=candidates,
    )
    failure_aware = run_ghost_mgg_v0(
        target,
        evidence,
        config=config,
        weights=GHOST_MGG_V0_WEIGHTS if failure_aware_weights is None else failure_aware_weights,
        hypotheses=candidates,
    )
    return (
        _ranking_rows(
            scene_id=scene_id,
            target_label=target_label,
            shape_hint=shape_hint,
            ranker="silhouette_only",
            ranked=silhouette,
            target_mask=target,
            evidence=evidence,
        )
        + _ranking_rows(
            scene_id=scene_id,
            target_label=target_label,
            shape_hint=shape_hint,
            ranker="failure_aware",
            ranked=failure_aware,
            target_mask=target,
            evidence=evidence,
        )
    )


def _read_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {mask_path}")
    return mask > 0


def _frames_to_rgb_and_depth(frames: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rgb = decode_image_msg(frames[COLOR_TOPIC])
    depth = decode_image_msg(frames[ALIGNED_DEPTH_TOPIC])
    return rgb, depth


def run_real_ranking_report(
    data_dir: Path,
    annotations_root: Path,
    output_dir: Path,
    *,
    background_scene_id: str = "empty_table_001",
    top_k: int = 3,
    failure_aware_weights: dict[str, float] | None = None,
) -> list[RealRankingRow]:
    data_dir = Path(data_dir)
    background_frames = read_first_image_frames(data_dir / background_scene_id)
    _background_rgb, background_depth = _frames_to_rgb_and_depth(background_frames)

    rows: list[RealRankingRow] = []
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
            rank_real_scene(
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

    write_ranking_reports(
        rows,
        output_dir,
        failure_aware_weights=GHOST_MGG_V0_WEIGHTS
        if failure_aware_weights is None
        else failure_aware_weights,
    )
    return rows


def write_ranking_reports(
    rows: list[RealRankingRow],
    output_dir: str | Path,
    *,
    failure_aware_weights: dict[str, float] | None = None,
) -> None:
    if not rows:
        raise ValueError("no ranking rows to write")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]

    json_path = output_dir / "m5_real_ranking.json"
    csv_path = output_dir / "m5_real_ranking.csv"
    index_path = output_dir / "index.md"
    comparison_rows = summarize_top1_comparison(rows)

    json_path.write_text(
        json.dumps(
            {
                "schema_version": "m5_real_ranking_v1",
                "num_rows": len(row_dicts),
                "num_scenes": len({row.scene_id for row in rows}),
                "failure_aware_weights": _stable_weights_payload(failure_aware_weights),
                "rows": row_dicts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(row_dicts[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(row_dicts)
    write_top1_comparison_reports(comparison_rows, output_dir)
    write_index_markdown(
        rows,
        comparison_rows,
        index_path,
        failure_aware_weights=failure_aware_weights,
    )


def summarize_top1_comparison(rows: list[RealRankingRow]) -> list[Top1ComparisonRow]:
    by_scene: dict[str, dict[str, RealRankingRow]] = {}
    for row in rows:
        if row.rank != 1 or row.ranker not in {"silhouette_only", "failure_aware"}:
            continue
        by_scene.setdefault(row.scene_id, {})[row.ranker] = row

    comparison_rows: list[Top1ComparisonRow] = []
    for scene_id in sorted(by_scene):
        scene_rows = by_scene[scene_id]
        silhouette = scene_rows.get("silhouette_only")
        failure = scene_rows.get("failure_aware")
        if silhouette is None or failure is None:
            continue
        comparison_rows.append(
            Top1ComparisonRow(
                scene_id=scene_id,
                target_label=failure.target_label or silhouette.target_label,
                shape_hint=failure.shape_hint or silhouette.shape_hint,
                silhouette_top=silhouette.hypothesis_id,
                failure_top=failure.hypothesis_id,
                silhouette_shape=silhouette.shape_type,
                failure_shape=failure.shape_type,
                top1_changed=silhouette.hypothesis_id != failure.hypothesis_id,
                shape_changed=silhouette.shape_type != failure.shape_type,
                silhouette_visual_score=round(silhouette.visual_score, 6),
                failure_visual_score=round(failure.visual_score, 6),
                silhouette_failure_score=round(silhouette.failure_score, 6),
                failure_failure_score=round(failure.failure_score, 6),
                visual_score_delta=round(failure.visual_score - silhouette.visual_score, 6),
                failure_score_delta=round(failure.failure_score - silhouette.failure_score, 6),
            )
        )
    return comparison_rows


def write_top1_comparison_reports(
    comparison_rows: list[Top1ComparisonRow], output_dir: Path
) -> None:
    row_dicts = [asdict(row) for row in comparison_rows]
    top1_json_path = output_dir / "top1_comparison.json"
    top1_csv_path = output_dir / "top1_comparison.csv"
    failure_deltas = [row.failure_score_delta for row in comparison_rows]
    visual_deltas = [row.visual_score_delta for row in comparison_rows]
    payload = {
        "schema_version": "m5_real_top1_comparison_v1",
        "num_scenes": len(comparison_rows),
        "top1_changed_count": sum(1 for row in comparison_rows if row.top1_changed),
        "shape_changed_count": sum(1 for row in comparison_rows if row.shape_changed),
        "mean_failure_score_delta": round(float(np.mean(failure_deltas)), 6)
        if failure_deltas
        else None,
        "mean_visual_score_delta": round(float(np.mean(visual_deltas)), 6)
        if visual_deltas
        else None,
        "rows": row_dicts,
    }
    top1_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if row_dicts:
        with top1_csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(row_dicts[0].keys()),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(row_dicts)
    else:
        top1_csv_path.write_text("", encoding="utf-8")


def write_index_markdown(
    rows: list[RealRankingRow],
    comparison_rows: list[Top1ComparisonRow],
    output_path: Path,
    *,
    failure_aware_weights: dict[str, float] | None = None,
) -> None:
    weights = _stable_weights_payload(failure_aware_weights)
    lines = [
        "# M5 Real Ranking Report",
        "",
        "## Failure-Aware Weights",
        "",
        "| term | weight |",
        "|---|---:|",
    ]
    for name, value in weights.items():
        lines.append(f"| {name} | {value:.3f} |")
    lines.extend(
        [
            "",
            "These are the failure-aware weights used by the `failure_aware` ranker.",
            "",
            "## Failure Likelihood Breakdown",
            "",
            "Each candidate row includes failure-score terms whose signed sum equals `failure_score`.",
            "",
            "| field | meaning |",
            "|---|---|",
            "| failure_inside_hole | mean hole evidence inside the rendered hypothesis |",
            "| failure_inside_table_leakage | mean table-leakage evidence inside the rendered hypothesis |",
            "| failure_boundary_edge | mean edge evidence in the rendered boundary band |",
            "| failure_boundary_flying_point | mean flying-point evidence in the rendered boundary band |",
            "| failure_outside_hole_penalty | hole evidence inside the target mask but outside the hypothesis |",
            "| failure_outside_table_leakage_penalty | table-leakage evidence inside the target mask but outside the hypothesis |",
            "| failure_total_check | recomputed signed total; should equal `failure_score` |",
            "",
        ]
    )
    lines.extend(
        [
        "## Top-1 Comparison",
        "",
        "| scene_id | silhouette_top | failure_top | changed | shape_changed | failure_score_delta | visual_score_delta |",
        "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            "| "
            f"{row.scene_id} | "
            f"{row.silhouette_top} | "
            f"{row.failure_top} | "
            f"{int(row.top1_changed)} | "
            f"{int(row.shape_changed)} | "
            f"{row.failure_score_delta:.3f} | "
            f"{row.visual_score_delta:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Rows",
            "",
            "| scene_id | ranker | rank | hypothesis | shape | total | failure | visual |",
            "|---|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.scene_id} | "
            f"{row.ranker} | "
            f"{row.rank} | "
            f"{row.hypothesis_id} | "
            f"{row.shape_type} | "
            f"{row.total_score:.3f} | "
            f"{row.failure_score:.3f} | "
            f"{row.visual_score:.3f} |"
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
        default=Path("reports/m5_real_d435_ranking"),
    )
    parser.add_argument("--background-scene-id", default="empty_table_001")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--failure-aware-weights-json",
        type=Path,
        default=None,
        help="Optional M4.1 best_weights.json used for the failure_aware ranker.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = run_real_ranking_report(
        data_dir=args.data_dir,
        annotations_root=args.annotations_root,
        output_dir=args.output_dir,
        background_scene_id=args.background_scene_id,
        top_k=args.top_k,
        failure_aware_weights=load_failure_aware_weights(args.failure_aware_weights_json),
    )
    print(f"Wrote {len(rows)} M5 real ranking rows to {args.output_dir}")


if __name__ == "__main__":
    main()
