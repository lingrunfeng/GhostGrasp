#!/usr/bin/env python3
"""Evaluate real M5 ranking output against conservative weak-GT scene rules."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WeakGTEvalRow:
    scene_id: str
    target_label: str | None
    material_class: str
    object_family: str
    silhouette_top: str
    failure_top: str
    silhouette_shape: str
    failure_shape: str
    acceptable_shapes: str
    expected_failure_sensitive: bool
    require_shape_stability: bool
    top1_changed: bool
    shape_changed: bool
    failure_score_delta: float
    visual_score_delta: float
    hole_ratio: float | None
    table_leakage_ratio: float | None
    foreground_ratio: float | None
    acceptable_shape_ok: bool
    failure_gain_ok: bool
    visual_drop_ok: bool
    shape_stability_ok: bool
    weak_gt_pass: bool


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    return float(value)


def _read_csv_by_scene(path: Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return {row["scene_id"]: row for row in csv.DictReader(stream)}


def _read_weak_gt(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m5_real_weak_gt_v1":
        raise ValueError(f"unsupported weak-GT schema: {payload.get('schema_version')}")
    scenes = payload.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("weak-GT manifest must contain at least one scene")
    return scenes


def evaluate_weak_gt(
    *,
    weak_gt_path: Path,
    top1_comparison_csv: Path,
    evidence_summary_csv: Path,
) -> tuple[list[WeakGTEvalRow], dict[str, Any]]:
    top1_by_scene = _read_csv_by_scene(top1_comparison_csv)
    evidence_by_scene = _read_csv_by_scene(evidence_summary_csv)

    rows: list[WeakGTEvalRow] = []
    for scene in _read_weak_gt(weak_gt_path):
        scene_id = str(scene["scene_id"])
        if scene_id not in top1_by_scene:
            raise KeyError(f"missing top-1 ranking row for {scene_id}")
        top1 = top1_by_scene[scene_id]
        evidence = evidence_by_scene.get(scene_id, {})

        acceptable_shapes = [str(shape) for shape in scene.get("acceptable_shapes", [])]
        if not acceptable_shapes:
            raise ValueError(f"{scene_id} has no acceptable_shapes")
        failure_shape = str(top1["failure_shape"])
        visual_delta = float(top1["visual_score_delta"])
        failure_delta = float(top1["failure_score_delta"])
        expected_failure_sensitive = _bool(scene.get("expected_failure_sensitive", False))
        require_shape_stability = _bool(scene.get("require_shape_stability", False))
        min_failure_delta = float(scene.get("min_failure_score_delta", 0.0))
        max_visual_drop = float(scene.get("max_visual_score_drop", 0.12))

        acceptable_shape_ok = failure_shape in acceptable_shapes
        failure_gain_ok = (failure_delta >= min_failure_delta) if expected_failure_sensitive else True
        visual_drop_ok = visual_delta >= -max_visual_drop
        shape_changed = _bool(top1["shape_changed"])
        shape_stability_ok = (not shape_changed) if require_shape_stability else True
        weak_gt_pass = (
            acceptable_shape_ok
            and failure_gain_ok
            and visual_drop_ok
            and shape_stability_ok
        )

        rows.append(
            WeakGTEvalRow(
                scene_id=scene_id,
                target_label=scene.get("target_label") or top1.get("target_label"),
                material_class=str(scene.get("material_class", "unknown")),
                object_family=str(scene.get("object_family", "unknown")),
                silhouette_top=str(top1["silhouette_top"]),
                failure_top=str(top1["failure_top"]),
                silhouette_shape=str(top1["silhouette_shape"]),
                failure_shape=failure_shape,
                acceptable_shapes=";".join(acceptable_shapes),
                expected_failure_sensitive=expected_failure_sensitive,
                require_shape_stability=require_shape_stability,
                top1_changed=_bool(top1["top1_changed"]),
                shape_changed=shape_changed,
                failure_score_delta=round(failure_delta, 6),
                visual_score_delta=round(visual_delta, 6),
                hole_ratio=_float(evidence.get("hole_ratio")),
                table_leakage_ratio=_float(evidence.get("table_leakage_ratio")),
                foreground_ratio=_float(evidence.get("foreground_ratio")),
                acceptable_shape_ok=acceptable_shape_ok,
                failure_gain_ok=failure_gain_ok,
                visual_drop_ok=visual_drop_ok,
                shape_stability_ok=shape_stability_ok,
                weak_gt_pass=weak_gt_pass,
            )
        )

    failure_checked = [row for row in rows if row.expected_failure_sensitive]
    summary = {
        "schema_version": "m5_real_weak_gt_eval_v1",
        "num_scenes": len(rows),
        "weak_gt_pass_count": sum(1 for row in rows if row.weak_gt_pass),
        "acceptable_shape_pass_count": sum(1 for row in rows if row.acceptable_shape_ok),
        "failure_gain_checked_count": len(failure_checked),
        "failure_gain_pass_count": sum(1 for row in failure_checked if row.failure_gain_ok),
        "visual_drop_pass_count": sum(1 for row in rows if row.visual_drop_ok),
        "shape_stability_checked_count": sum(1 for row in rows if row.require_shape_stability),
        "shape_stability_pass_count": sum(
            1 for row in rows if row.require_shape_stability and row.shape_stability_ok
        ),
    }
    return rows, summary


def write_weak_gt_report(
    *,
    rows: list[WeakGTEvalRow],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    payload = dict(summary)
    payload["rows"] = row_dicts

    (output_dir / "weak_gt_eval.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if row_dicts:
        with (output_dir / "weak_gt_eval.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row_dicts[0].keys()))
            writer.writeheader()
            writer.writerows(row_dicts)
    else:
        (output_dir / "weak_gt_eval.csv").write_text("", encoding="utf-8")
    write_index_markdown(rows, summary, output_dir / "index.md")


def _summary_count(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key)
    return int(value) if value is not None else 0


def write_index_markdown(
    rows: list[WeakGTEvalRow],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        "# M5 Real Weak-GT Evaluation",
        "",
        "This is a conservative sanity check, not final 3D ground truth.",
        "",
        "## Summary",
        "",
        f"- scenes: {_summary_count(summary, 'num_scenes')}",
        f"- weak_gt_pass: {_summary_count(summary, 'weak_gt_pass_count')}/{_summary_count(summary, 'num_scenes')}",
        f"- acceptable_shape_pass: {_summary_count(summary, 'acceptable_shape_pass_count')}/{_summary_count(summary, 'num_scenes')}",
        f"- failure_gain_pass: {_summary_count(summary, 'failure_gain_pass_count')}/{_summary_count(summary, 'failure_gain_checked_count')}",
        f"- visual_drop_pass: {_summary_count(summary, 'visual_drop_pass_count')}/{_summary_count(summary, 'num_scenes')}",
        "",
        "## Scene Rows",
        "",
        "| scene_id | material | family | failure_top | acceptable | pass | failure_delta | visual_delta | hole | leak |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        hole = "" if row.hole_ratio is None else f"{row.hole_ratio:.3f}"
        leak = "" if row.table_leakage_ratio is None else f"{row.table_leakage_ratio:.3f}"
        lines.append(
            "| "
            f"{row.scene_id} | "
            f"{row.material_class} | "
            f"{row.object_family} | "
            f"{row.failure_top} | "
            f"{row.acceptable_shapes} | "
            f"{int(row.weak_gt_pass)} | "
            f"{row.failure_score_delta:.3f} | "
            f"{row.visual_score_delta:.3f} | "
            f"{hole} | "
            f"{leak} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weak-gt",
        type=Path,
        default=Path("annotations/m5_real_d435_weak_gt/weak_gt.json"),
    )
    parser.add_argument(
        "--top1-comparison-csv",
        type=Path,
        default=Path("reports/m5_real_d435_ranking/top1_comparison.csv"),
    )
    parser.add_argument(
        "--evidence-summary-csv",
        type=Path,
        default=Path("reports/m5_real_d435_masked_evidence/summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m5_real_d435_weak_gt_eval"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows, summary = evaluate_weak_gt(
        weak_gt_path=args.weak_gt,
        top1_comparison_csv=args.top1_comparison_csv,
        evidence_summary_csv=args.evidence_summary_csv,
    )
    write_weak_gt_report(rows=rows, summary=summary, output_dir=args.output_dir)
    print(
        "Wrote M5 real weak-GT evaluation for "
        f"{summary['num_scenes']} scenes to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
