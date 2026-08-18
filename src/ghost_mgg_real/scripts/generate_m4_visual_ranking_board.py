#!/usr/bin/env python3
"""Generate per-scene visual boards for real M4 ranking diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PANEL_W = 320
PANEL_H = 240
TEXT_PANEL_H = 240


@dataclass(frozen=True)
class TopHypothesis:
    scene_id: str
    ranker: str
    hypothesis_id: str
    shape_type: str
    center_u: float
    center_v: float
    size_u_px: float
    size_v_px: float
    visual_score: float
    failure_score: float
    total_score: float


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def _read_top_hypotheses(path: Path) -> dict[str, dict[str, TopHypothesis]]:
    by_scene: dict[str, dict[str, TopHypothesis]] = {}
    for row in _read_csv(path):
        if int(row["rank"]) != 1 or row["ranker"] not in {
            "silhouette_only",
            "failure_aware",
        }:
            continue
        hypothesis = TopHypothesis(
            scene_id=row["scene_id"],
            ranker=row["ranker"],
            hypothesis_id=row["hypothesis_id"],
            shape_type=row["shape_type"],
            center_u=float(row["center_u"]),
            center_v=float(row["center_v"]),
            size_u_px=float(row["size_u_px"]),
            size_v_px=float(row["size_v_px"]),
            visual_score=float(row["visual_score"]),
            failure_score=float(row["failure_score"]),
            total_score=float(row["total_score"]),
        )
        by_scene.setdefault(hypothesis.scene_id, {})[hypothesis.ranker] = hypothesis
    return by_scene


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _fit_panel(image: np.ndarray, width: int = PANEL_W, height: int = PANEL_H) -> np.ndarray:
    image = _ensure_bgr(image)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def _add_title(panel: np.ndarray, title: str) -> np.ndarray:
    rendered = panel.copy()
    cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(
        rendered,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return rendered


def draw_hypothesis_overlay(
    image: np.ndarray,
    hypothesis: TopHypothesis,
    *,
    color: tuple[int, int, int],
) -> np.ndarray:
    rendered = _ensure_bgr(image).copy()
    cx = int(round(hypothesis.center_u))
    cy = int(round(hypothesis.center_v))
    half_w = max(1, int(round(hypothesis.size_u_px / 2.0)))
    half_h = max(1, int(round(hypothesis.size_v_px / 2.0)))
    thickness = max(2, min(rendered.shape[:2]) // 120)

    if hypothesis.shape_type == "cylinder":
        cv2.ellipse(rendered, (cx, cy), (half_w, half_h), 0, 0, 360, color, thickness)
    else:
        cv2.rectangle(
            rendered,
            (cx - half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            color,
            thickness,
        )
    cv2.circle(rendered, (cx, cy), max(2, thickness + 1), color, -1)
    label = f"{hypothesis.hypothesis_id} T={hypothesis.total_score:.2f}"
    y = max(18, cy - half_h - 8)
    cv2.putText(
        rendered,
        label,
        (max(4, cx - half_w), y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return rendered


def _mask_panel(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    rgb_bgr = _ensure_bgr(rgb)
    mask_bool = mask > 0
    overlay = rgb_bgr.copy()
    overlay[mask_bool] = (0.55 * overlay[mask_bool] + np.array([0, 180, 255]) * 0.45).astype(
        np.uint8
    )
    contours, _hierarchy = cv2.findContours(
        mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def _summary_panel(
    *,
    dashboard_row: dict[str, str],
    silhouette: TopHypothesis,
    failure: TopHypothesis,
    width: int = PANEL_W,
    height: int = TEXT_PANEL_H,
) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    lines = [
        str(dashboard_row["scene_id"]),
        f"material: {dashboard_row['material_class']} / {dashboard_row['object_family']}",
        f"hole={float(dashboard_row['hole_ratio']):.3f}  leak={float(dashboard_row['table_leakage_ratio']):.3f}",
        f"sil: {silhouette.hypothesis_id}",
        f"  V={silhouette.visual_score:.2f} F={silhouette.failure_score:.2f}",
        f"fail: {failure.hypothesis_id}",
        f"  V={failure.visual_score:.2f} F={failure.failure_score:.2f}",
        f"dF={float(dashboard_row['failure_score_delta']):.3f} dV={float(dashboard_row['visual_score_delta']):.3f}",
        f"weakGT={int(_bool(dashboard_row['weak_gt_pass']))} changed={int(_bool(dashboard_row['top1_changed']))}",
    ]
    y = 28
    for index, line in enumerate(lines):
        color = (20, 20, 20)
        if index in {3, 4}:
            color = (0, 130, 0)
        elif index in {5, 6}:
            color = (0, 0, 190)
        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 23
    return panel


def build_scene_board(
    *,
    rgb: np.ndarray,
    mask: np.ndarray,
    evidence_overlay: np.ndarray,
    silhouette: TopHypothesis,
    failure: TopHypothesis,
    dashboard_row: dict[str, str],
) -> np.ndarray:
    silhouette_overlay = draw_hypothesis_overlay(rgb, silhouette, color=(0, 180, 0))
    failure_overlay = draw_hypothesis_overlay(rgb, failure, color=(0, 0, 220))
    panels = [
        _add_title(_fit_panel(rgb), "RGB"),
        _add_title(_fit_panel(_mask_panel(mask, rgb)), "Target mask"),
        _add_title(_fit_panel(evidence_overlay), "Failure evidence"),
        _add_title(_fit_panel(silhouette_overlay), "Silhouette-only top-1"),
        _add_title(_fit_panel(failure_overlay), "Failure-aware top-1"),
        _add_title(_summary_panel(dashboard_row=dashboard_row, silhouette=silhouette, failure=failure), "Summary"),
    ]
    top = np.hstack(panels[:3])
    bottom = np.hstack(panels[3:])
    return np.vstack([top, bottom])


def generate_visual_board_report(
    *,
    replay_samples_dir: Path,
    masks_root: Path,
    evidence_dir: Path,
    ranking_csv: Path,
    dashboard_csv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    replay_samples_dir = Path(replay_samples_dir)
    masks_root = Path(masks_root)
    evidence_dir = Path(evidence_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_hypotheses = _read_top_hypotheses(ranking_csv)
    dashboard_rows = _read_csv(dashboard_csv)
    scene_records = []
    for dashboard_row in dashboard_rows:
        scene_id = dashboard_row["scene_id"]
        scene_hypotheses = top_hypotheses.get(scene_id)
        if scene_hypotheses is None:
            raise KeyError(f"missing ranking hypotheses for {scene_id}")
        silhouette = scene_hypotheses.get("silhouette_only")
        failure = scene_hypotheses.get("failure_aware")
        if silhouette is None or failure is None:
            raise KeyError(f"missing top-1 ranker pair for {scene_id}")

        rgb_path = replay_samples_dir / scene_id / "color.png"
        mask_path = masks_root / scene_id / "target_mask.png"
        evidence_path = evidence_dir / scene_id / "evidence_overlay.png"
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        evidence = cv2.imread(str(evidence_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_path)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if evidence is None:
            raise FileNotFoundError(evidence_path)

        board = build_scene_board(
            rgb=rgb,
            mask=mask,
            evidence_overlay=evidence,
            silhouette=silhouette,
            failure=failure,
            dashboard_row=dashboard_row,
        )
        board_path = output_dir / f"{scene_id}.png"
        cv2.imwrite(str(board_path), board)
        scene_records.append(
            {
                "scene_id": scene_id,
                "board_path": str(board_path),
                "silhouette_top": silhouette.hypothesis_id,
                "failure_top": failure.hypothesis_id,
                "weak_gt_pass": _bool(dashboard_row["weak_gt_pass"]),
            }
        )

    manifest = {
        "schema_version": "m4_visual_ranking_board_manifest_v1",
        "num_scenes": len(scene_records),
        "scenes": scene_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_index_markdown(manifest, output_dir / "index.md")
    return manifest


def write_index_markdown(manifest: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# M4 Visual Ranking Board",
        "",
        "Open each PNG to inspect RGB, mask, evidence, and top-1 hypothesis overlays.",
        "",
        "| scene_id | board | silhouette_top | failure_top | weak_gt |",
        "|---|---|---|---|---:|",
    ]
    for scene in manifest["scenes"]:
        board_name = Path(scene["board_path"]).name
        lines.append(
            "| "
            f"{scene['scene_id']} | "
            f"[{board_name}]({board_name}) | "
            f"{scene['silhouette_top']} | "
            f"{scene['failure_top']} | "
            f"{int(scene['weak_gt_pass'])} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-samples-dir",
        type=Path,
        default=Path("reports/m5_real_d435_replay_samples"),
    )
    parser.add_argument(
        "--masks-root",
        type=Path,
        default=Path("annotations/m5_real_d435_masks/masks"),
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("reports/m5_real_d435_masked_evidence"),
    )
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=Path("reports/m5_real_d435_ranking/m5_real_ranking.csv"),
    )
    parser.add_argument(
        "--dashboard-csv",
        type=Path,
        default=Path("reports/m4_real_dashboard/dashboard.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m4_visual_ranking_board"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = generate_visual_board_report(
        replay_samples_dir=args.replay_samples_dir,
        masks_root=args.masks_root,
        evidence_dir=args.evidence_dir,
        ranking_csv=args.ranking_csv,
        dashboard_csv=args.dashboard_csv,
        output_dir=args.output_dir,
    )
    print(f"Wrote {manifest['num_scenes']} M4 visual ranking boards to {args.output_dir}")


if __name__ == "__main__":
    main()
