#!/usr/bin/env python3
"""Build the M4 real-data algorithm gate from ranking and visual quality reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "m4_real_algorithm_gate_v1"
EVALUABLE_STATUSES = {"good", "questionable"}


def build_algorithm_gate_report(*, ranking_dir: Path, dashboard_json: Path) -> dict[str, Any]:
    ranking_dir = Path(ranking_dir)
    top1_payload = _read_json(ranking_dir / "top1_comparison.json")
    ranking_payload = _read_json(ranking_dir / "m5_real_ranking.json")
    dashboard_payload = _read_json(dashboard_json)
    quality_by_scene = _quality_by_scene(dashboard_payload)

    rows: list[dict[str, Any]] = []
    evaluable_failure_deltas: list[float] = []
    evaluable_visual_deltas: list[float] = []
    evaluable_top1_changed_count = 0
    evaluable_shape_changed_count = 0
    excluded_count = 0

    for item in sorted(top1_payload.get("rows", []), key=lambda row: str(row.get("scene_id", ""))):
        scene_id = str(item.get("scene_id", ""))
        quality = quality_by_scene.get(
            scene_id,
            {"status": "questionable", "reasons": ["scene missing from quality dashboard"]},
        )
        quality_status = str(quality.get("status", "questionable"))
        if quality_status in EVALUABLE_STATUSES:
            gate_decision = "evaluable"
            evaluable_failure_deltas.append(float(item.get("failure_score_delta", 0.0)))
            evaluable_visual_deltas.append(float(item.get("visual_score_delta", 0.0)))
            if bool(item.get("top1_changed", False)):
                evaluable_top1_changed_count += 1
            if bool(item.get("shape_changed", False)):
                evaluable_shape_changed_count += 1
        else:
            gate_decision = f"excluded_{quality_status}"
            excluded_count += 1

        rows.append(
            {
                "scene_id": scene_id,
                "target_label": item.get("target_label"),
                "shape_hint": item.get("shape_hint"),
                "quality_status": quality_status,
                "quality_reasons": list(quality.get("reasons", [])),
                "gate_decision": gate_decision,
                "silhouette_top": item.get("silhouette_top"),
                "failure_top": item.get("failure_top"),
                "silhouette_shape": item.get("silhouette_shape"),
                "failure_shape": item.get("failure_shape"),
                "top1_changed": bool(item.get("top1_changed", False)),
                "shape_changed": bool(item.get("shape_changed", False)),
                "failure_score_delta": float(item.get("failure_score_delta", 0.0)),
                "visual_score_delta": float(item.get("visual_score_delta", 0.0)),
            }
        )

    num_evaluable = len(evaluable_failure_deltas)
    mean_failure_delta = _mean(evaluable_failure_deltas)
    mean_visual_delta = _mean(evaluable_visual_deltas)
    gate_reasons: list[str] = []
    overall_status = "pass"
    if num_evaluable == 0:
        overall_status = "review"
        gate_reasons.append("no good/questionable real scenes are evaluable")
    if evaluable_top1_changed_count == 0:
        overall_status = "review"
        gate_reasons.append("failure-aware top-1 did not differ from silhouette-only on evaluable scenes")
    if mean_failure_delta <= 0.05:
        overall_status = "review"
        gate_reasons.append(f"mean failure-score delta is too small ({mean_failure_delta:.3f})")
    if not gate_reasons:
        gate_reasons.append(
            "failure evidence changes top-1 ranking and improves failure-score agreement on evaluable real scenes"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "overall_status": overall_status,
        "num_scenes": len(rows),
        "num_evaluable_scenes": num_evaluable,
        "num_excluded_scenes": excluded_count,
        "quality_counts": _quality_counts(rows),
        "evaluable_statuses": sorted(EVALUABLE_STATUSES),
        "excluded_statuses": sorted(
            {row["quality_status"] for row in rows if row["gate_decision"] != "evaluable"}
        ),
        "evaluable_top1_changed_count": evaluable_top1_changed_count,
        "evaluable_shape_changed_count": evaluable_shape_changed_count,
        "mean_evaluable_failure_score_delta": mean_failure_delta,
        "mean_evaluable_visual_score_delta": mean_visual_delta,
        "failure_aware_weights": ranking_payload.get("failure_aware_weights", {}),
        "gate_reasons": gate_reasons,
        "rows": rows,
    }


def write_algorithm_gate_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "algorithm_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(report["rows"], output_dir / "algorithm_gate.csv")
    (output_dir / "index.md").write_text(_render_markdown(report), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _quality_by_scene(dashboard_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    quality: dict[str, dict[str, Any]] = {}
    for card in dashboard_payload.get("cards", []):
        scene_id = str(card.get("scene_id", ""))
        card_quality = card.get("quality") or {}
        quality[scene_id] = {
            "status": str(card_quality.get("status", "questionable")),
            "reasons": list(card_quality.get("reasons", [])),
        }
    return quality


def _quality_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("quality_status", "questionable"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "scene_id",
        "target_label",
        "shape_hint",
        "quality_status",
        "gate_decision",
        "silhouette_top",
        "failure_top",
        "silhouette_shape",
        "failure_shape",
        "top1_changed",
        "shape_changed",
        "failure_score_delta",
        "visual_score_delta",
        "quality_reasons",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["quality_reasons"] = "; ".join(str(reason) for reason in row.get("quality_reasons", []))
            writer.writerow({name: payload.get(name) for name in fieldnames})


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M4 Real Algorithm Gate",
        "",
        f"- overall_status: {report['overall_status']}",
        f"- scenes: {report['num_scenes']}",
        f"- evaluable scenes: {report['num_evaluable_scenes']}",
        f"- excluded scenes: {report['num_excluded_scenes']}",
        f"- evaluable top-1 changed: {report['evaluable_top1_changed_count']}",
        f"- evaluable shape changed: {report['evaluable_shape_changed_count']}",
        f"- mean failure-score delta: {report['mean_evaluable_failure_score_delta']:.3f}",
        f"- mean visual-score delta: {report['mean_evaluable_visual_score_delta']:.3f}",
        "",
        "## Gate Reasons",
        "",
    ]
    for reason in report["gate_reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Scene Decisions",
            "",
            "| scene_id | quality | decision | silhouette_top | failure_top | top1_changed | failure_delta |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| "
            f"{row['scene_id']} | "
            f"{row['quality_status']} | "
            f"{row['gate_decision']} | "
            f"{row['silhouette_top']} | "
            f"{row['failure_top']} | "
            f"{int(row['top1_changed'])} | "
            f"{row['failure_score_delta']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranking-dir",
        type=Path,
        default=Path("reports/m5_real_d435_ranking"),
    )
    parser.add_argument(
        "--dashboard-json",
        type=Path,
        default=Path("reports/m4_real_external_mask_visual_dashboard/dashboard.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m4_real_algorithm_gate"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_algorithm_gate_report(
        ranking_dir=args.ranking_dir,
        dashboard_json=args.dashboard_json,
    )
    write_algorithm_gate_report(report, args.output_dir)
    print(f"Wrote M4 real algorithm gate to {args.output_dir / 'algorithm_gate.json'}")


if __name__ == "__main__":
    main()
