#!/usr/bin/env python3
"""Build one compact dashboard from current real M4/M5 diagnostic reports."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DashboardRow:
    scene_id: str
    target_label: str | None
    shape_hint: str | None
    material_class: str
    object_family: str
    target_pixels: int
    valid_depth_ratio: float
    hole_ratio: float
    table_leakage_ratio: float
    foreground_ratio: float
    silhouette_top: str
    failure_top: str
    silhouette_shape: str
    failure_shape: str
    top1_changed: bool
    shape_changed: bool
    failure_score_delta: float
    visual_score_delta: float
    acceptable_shapes: str
    acceptable_shape_ok: bool
    failure_gain_ok: bool
    visual_drop_ok: bool
    shape_stability_ok: bool
    weak_gt_pass: bool


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float:
    return float(value)


def _int(value: Any) -> int:
    return int(float(value))


def _read_csv_by_scene(path: Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return {row["scene_id"]: row for row in rows}


def build_dashboard(
    *,
    evidence_summary_csv: Path,
    top1_comparison_csv: Path,
    weak_gt_eval_csv: Path,
) -> tuple[list[DashboardRow], dict[str, Any]]:
    evidence_by_scene = _read_csv_by_scene(evidence_summary_csv)
    top1_by_scene = _read_csv_by_scene(top1_comparison_csv)
    weak_gt_by_scene = _read_csv_by_scene(weak_gt_eval_csv)

    scene_ids = sorted(evidence_by_scene)
    missing_top1 = sorted(set(scene_ids) - set(top1_by_scene))
    missing_weak_gt = sorted(set(scene_ids) - set(weak_gt_by_scene))
    if missing_top1:
        raise KeyError(f"missing top-1 rows for scenes: {missing_top1}")
    if missing_weak_gt:
        raise KeyError(f"missing weak-GT rows for scenes: {missing_weak_gt}")

    rows: list[DashboardRow] = []
    for scene_id in scene_ids:
        evidence = evidence_by_scene[scene_id]
        top1 = top1_by_scene[scene_id]
        weak_gt = weak_gt_by_scene[scene_id]
        rows.append(
            DashboardRow(
                scene_id=scene_id,
                target_label=evidence.get("target_label") or top1.get("target_label"),
                shape_hint=evidence.get("shape_hint") or top1.get("shape_hint"),
                material_class=str(weak_gt["material_class"]),
                object_family=str(weak_gt["object_family"]),
                target_pixels=_int(evidence["target_pixels"]),
                valid_depth_ratio=round(_float(evidence["valid_depth_ratio"]), 6),
                hole_ratio=round(_float(evidence["hole_ratio"]), 6),
                table_leakage_ratio=round(_float(evidence["table_leakage_ratio"]), 6),
                foreground_ratio=round(_float(evidence["foreground_ratio"]), 6),
                silhouette_top=str(top1["silhouette_top"]),
                failure_top=str(top1["failure_top"]),
                silhouette_shape=str(top1["silhouette_shape"]),
                failure_shape=str(top1["failure_shape"]),
                top1_changed=_bool(top1["top1_changed"]),
                shape_changed=_bool(top1["shape_changed"]),
                failure_score_delta=round(_float(top1["failure_score_delta"]), 6),
                visual_score_delta=round(_float(top1["visual_score_delta"]), 6),
                acceptable_shapes=str(weak_gt["acceptable_shapes"]),
                acceptable_shape_ok=_bool(weak_gt["acceptable_shape_ok"]),
                failure_gain_ok=_bool(weak_gt["failure_gain_ok"]),
                visual_drop_ok=_bool(weak_gt["visual_drop_ok"]),
                shape_stability_ok=_bool(weak_gt["shape_stability_ok"]),
                weak_gt_pass=_bool(weak_gt["weak_gt_pass"]),
            )
        )

    summary = summarize_dashboard(rows)
    return rows, summary


def summarize_dashboard(rows: list[DashboardRow]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no dashboard rows")

    return {
        "schema_version": "m4_real_dashboard_v1",
        "num_scenes": len(rows),
        "weak_gt_pass_count": sum(1 for row in rows if row.weak_gt_pass),
        "top1_changed_count": sum(1 for row in rows if row.top1_changed),
        "shape_changed_count": sum(1 for row in rows if row.shape_changed),
        "mean_hole_ratio": round(float(np.mean([row.hole_ratio for row in rows])), 6),
        "mean_table_leakage_ratio": round(
            float(np.mean([row.table_leakage_ratio for row in rows])), 6
        ),
        "mean_failure_score_delta": round(
            float(np.mean([row.failure_score_delta for row in rows])), 6
        ),
        "mean_visual_score_delta": round(
            float(np.mean([row.visual_score_delta for row in rows])), 6
        ),
    }


def write_dashboard(
    *,
    rows: list[DashboardRow],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    payload = dict(summary)
    payload["rows"] = row_dicts

    (output_dir / "dashboard.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "dashboard.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)
    write_index_markdown(rows, summary, output_dir / "index.md")


def write_index_markdown(
    rows: list[DashboardRow],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        "# M4 Real Dashboard",
        "",
        "Compact dashboard for the current real D435 evidence/ranking chain.",
        "",
        "## Summary",
        "",
        f"- scenes: {summary['num_scenes']}",
        f"- weak_gt_pass: {summary['weak_gt_pass_count']}/{summary['num_scenes']}",
        f"- top1_changed: {summary['top1_changed_count']}/{summary['num_scenes']}",
        f"- shape_changed: {summary['shape_changed_count']}/{summary['num_scenes']}",
        f"- mean_hole_ratio: {summary['mean_hole_ratio']:.3f}",
        f"- mean_table_leakage_ratio: {summary['mean_table_leakage_ratio']:.3f}",
        f"- mean_failure_score_delta: {summary['mean_failure_score_delta']:.3f}",
        f"- mean_visual_score_delta: {summary['mean_visual_score_delta']:.3f}",
        "",
        "## Scene Rows",
        "",
        "| scene_id | material | family | hole | leak | silhouette_top | failure_top | weak_gt | failure_delta | visual_delta |",
        "|---|---|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.scene_id} | "
            f"{row.material_class} | "
            f"{row.object_family} | "
            f"{row.hole_ratio:.3f} | "
            f"{row.table_leakage_ratio:.3f} | "
            f"{row.silhouette_top} | "
            f"{row.failure_top} | "
            f"{int(row.weak_gt_pass)} | "
            f"{row.failure_score_delta:.3f} | "
            f"{row.visual_score_delta:.3f} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-summary-csv",
        type=Path,
        default=Path("reports/m5_real_d435_masked_evidence/summary.csv"),
    )
    parser.add_argument(
        "--top1-comparison-csv",
        type=Path,
        default=Path("reports/m5_real_d435_ranking/top1_comparison.csv"),
    )
    parser.add_argument(
        "--weak-gt-eval-csv",
        type=Path,
        default=Path("reports/m5_real_d435_weak_gt_eval/weak_gt_eval.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m4_real_dashboard"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows, summary = build_dashboard(
        evidence_summary_csv=args.evidence_summary_csv,
        top1_comparison_csv=args.top1_comparison_csv,
        weak_gt_eval_csv=args.weak_gt_eval_csv,
    )
    write_dashboard(rows=rows, summary=summary, output_dir=args.output_dir)
    print(f"Wrote M4 real dashboard for {summary['num_scenes']} scenes to {args.output_dir}")


if __name__ == "__main__":
    main()
