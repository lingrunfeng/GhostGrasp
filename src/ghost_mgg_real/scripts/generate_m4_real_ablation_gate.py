#!/usr/bin/env python3
"""Run real D435 external-mask ablations for M4 algorithm evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from generate_m5_masked_evidence_report import load_completed_mask_records
from generate_m5_real_ranking_report import (
    _frames_to_rgb_and_depth,
    _read_mask,
    evidence_maps_from_depth_background,
)
from ghost_mgg_core_py.evidence.types import EvidenceMaps
from ghost_mgg_core_py.ghost_mgg_v0 import (
    GHOST_MGG_V0_ABLATIONS,
    GhostMGGV0Config,
    run_ghost_mgg_v0,
)
from ghost_mgg_core_py.hypotheses.hypothesis_generator import generate_local_hypotheses
from ghost_mgg_core_py.hypotheses.primitives import PrimitiveHypothesis


SCHEMA_VERSION = "m4_real_ablation_gate_v1"
DEFAULT_ABLATIONS = (
    "full",
    "silhouette_only",
    "without_failure",
    "without_table_leakage",
    "without_weak_depth",
)
EVALUABLE_STATUSES = {"good", "questionable"}


def rank_scene_ablations(
    *,
    scene_id: str,
    target_label: str | None,
    shape_hint: str | None,
    target_mask: np.ndarray,
    current_depth: np.ndarray,
    background_depth: np.ndarray,
    hypotheses: list[PrimitiveHypothesis] | None = None,
    top_k: int = 3,
    ablation_names: tuple[str, ...] = DEFAULT_ABLATIONS,
) -> list[dict[str, Any]]:
    target = np.asarray(target_mask, dtype=bool)
    evidence = evidence_maps_from_depth_background(target, current_depth, background_depth)
    candidates = hypotheses or generate_local_hypotheses(
        target,
        shape_types=("box", "cylinder"),
        scale_factors=(0.65, 0.80, 0.95, 1.00, 1.10, 1.25),
        depth_m=1.0,
        height_m=0.08,
    )
    rows: list[dict[str, Any]] = []
    for ranker in ablation_names:
        spec = GHOST_MGG_V0_ABLATIONS[ranker]
        ablated_evidence = _zero_evidence_channels(evidence, tuple(spec.get("zero_channels", ())))
        ranked = run_ghost_mgg_v0(
            target,
            ablated_evidence,
            config=GhostMGGV0Config(top_k=top_k),
            weights=dict(spec["weights"]),
            hypotheses=candidates,
        )
        for rank_index, item in enumerate(ranked, start=1):
            hypothesis = item.hypothesis
            score = item.score
            rows.append(
                {
                    "scene_id": scene_id,
                    "target_label": target_label,
                    "shape_hint": shape_hint,
                    "ranker": ranker,
                    "rank": rank_index,
                    "hypothesis_id": str(hypothesis.hypothesis_id),
                    "shape_type": str(hypothesis.shape_type),
                    "center_u": float(hypothesis.center_uv[0]),
                    "center_v": float(hypothesis.center_uv[1]),
                    "size_u_px": float(hypothesis.size_px[0]),
                    "size_v_px": float(hypothesis.size_px[1]),
                    "visual_score": float(score.visual),
                    "failure_score": float(score.failure),
                    "depth_score": float(score.depth),
                    "prior_score": float(score.prior),
                    "total_score": float(score.total),
                    "zero_channels": list(spec.get("zero_channels", ())),
                    "validation_state": item.validation_state,
                }
            )
    return rows


def run_real_ablation_gate(
    *,
    data_dir: Path,
    annotations_root: Path,
    dashboard_json: Path,
    output_dir: Path,
    background_scene_id: str = "empty_table_001",
    top_k: int = 3,
) -> dict[str, Any]:
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
            rank_scene_ablations(
                scene_id=scene_id,
                target_label=record.get("target_label"),
                shape_hint=record.get("shape_hint"),
                target_mask=target_mask,
                current_depth=depth,
                background_depth=background_depth,
                top_k=top_k,
            )
        )

    report = build_ablation_gate_report(rows=rows, dashboard_json=dashboard_json)
    write_ablation_gate_report(report, output_dir)
    return report


def build_ablation_gate_report(*, rows: list[dict[str, Any]], dashboard_json: Path) -> dict[str, Any]:
    quality_by_scene = _quality_by_scene(_read_json(dashboard_json))
    top_rows = [row for row in rows if int(row.get("rank", 999)) == 1]
    by_scene: dict[str, dict[str, dict[str, Any]]] = {}
    for row in top_rows:
        by_scene.setdefault(str(row["scene_id"]), {})[str(row["ranker"])] = row

    report_rows: list[dict[str, Any]] = []
    ablation_summary: dict[str, dict[str, Any]] = {
        name: {
            "checked_count": 0,
            "top1_changed_count": 0,
            "shape_changed_count": 0,
            "mean_full_total_advantage": 0.0,
            "mean_full_failure_advantage": 0.0,
        }
        for name in DEFAULT_ABLATIONS
        if name != "full"
    }
    total_advantages: dict[str, list[float]] = {name: [] for name in ablation_summary}
    failure_advantages: dict[str, list[float]] = {name: [] for name in ablation_summary}
    excluded_count = 0
    evaluable_scenes: set[str] = set()

    for scene_id in sorted(by_scene):
        scene_rankers = by_scene[scene_id]
        full = scene_rankers.get("full")
        if full is None:
            continue
        quality = quality_by_scene.get(
            scene_id,
            {"status": "questionable", "reasons": ["scene missing from quality dashboard"]},
        )
        quality_status = str(quality.get("status", "questionable"))
        gate_decision = "evaluable" if quality_status in EVALUABLE_STATUSES else f"excluded_{quality_status}"
        if gate_decision == "evaluable":
            evaluable_scenes.add(scene_id)
        else:
            excluded_count += 1

        for ranker in DEFAULT_ABLATIONS:
            row = scene_rankers.get(ranker)
            if row is None:
                continue
            top1_changed = row["hypothesis_id"] != full["hypothesis_id"]
            shape_changed = row["shape_type"] != full["shape_type"]
            total_advantage = float(full["total_score"]) - float(row["total_score"])
            failure_advantage = float(full["failure_score"]) - float(row["failure_score"])
            report_row = {
                **row,
                "quality_status": quality_status,
                "quality_reasons": list(quality.get("reasons", [])),
                "gate_decision": gate_decision,
                "full_hypothesis_id": full["hypothesis_id"],
                "full_shape_type": full["shape_type"],
                "top1_changed_vs_full": bool(top1_changed) if ranker != "full" else False,
                "shape_changed_vs_full": bool(shape_changed) if ranker != "full" else False,
                "full_total_advantage": total_advantage,
                "full_failure_advantage": failure_advantage,
            }
            report_rows.append(report_row)
            if ranker != "full" and gate_decision == "evaluable":
                ablation_summary[ranker]["checked_count"] += 1
                if top1_changed:
                    ablation_summary[ranker]["top1_changed_count"] += 1
                if shape_changed:
                    ablation_summary[ranker]["shape_changed_count"] += 1
                total_advantages[ranker].append(total_advantage)
                failure_advantages[ranker].append(failure_advantage)

    for ranker, summary in ablation_summary.items():
        summary["mean_full_total_advantage"] = _mean(total_advantages[ranker])
        summary["mean_full_failure_advantage"] = _mean(failure_advantages[ranker])

    gate_reasons: list[str] = []
    overall_status = "pass"
    without_failure_changes = ablation_summary["without_failure"]["top1_changed_count"]
    without_leakage_changes = ablation_summary["without_table_leakage"]["top1_changed_count"]
    if not evaluable_scenes:
        overall_status = "review"
        gate_reasons.append("no good/questionable scenes are evaluable")
    if without_failure_changes == 0:
        overall_status = "review"
        gate_reasons.append("full ranker did not differ from without_failure on evaluable scenes")
    if without_leakage_changes == 0:
        overall_status = "review"
        gate_reasons.append("full ranker did not differ from without_table_leakage on evaluable scenes")
    if not gate_reasons:
        gate_reasons.append(
            "full failure-aware ranking changes top-1 relative to failure and table-leakage ablations"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "overall_status": overall_status,
        "num_scenes": len(by_scene),
        "num_evaluable_scenes": len(evaluable_scenes),
        "num_excluded_scenes": excluded_count,
        "rankers": list(DEFAULT_ABLATIONS),
        "ablation_summary": ablation_summary,
        "gate_reasons": gate_reasons,
        "rows": report_rows,
    }


def write_ablation_gate_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(report["rows"], output_dir / "ablation_gate.csv")
    (output_dir / "index.md").write_text(_render_markdown(report), encoding="utf-8")


def _zero_evidence_channels(evidence: EvidenceMaps, zero_channels: tuple[str, ...]) -> EvidenceMaps:
    values = evidence.as_dict()
    resolved = {}
    for name, value in values.items():
        resolved[name] = np.zeros_like(value, dtype=np.float32) if name in zero_channels else value
    return EvidenceMaps(**resolved)


def _quality_by_scene(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    quality = {}
    for card in payload.get("cards", []):
        card_quality = card.get("quality") or {}
        quality[str(card.get("scene_id", ""))] = {
            "status": str(card_quality.get("status", "questionable")),
            "reasons": list(card_quality.get("reasons", [])),
        }
    return quality


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(sum(values) / len(values)), 6)


def _write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "scene_id",
        "target_label",
        "shape_hint",
        "quality_status",
        "gate_decision",
        "ranker",
        "rank",
        "hypothesis_id",
        "shape_type",
        "full_hypothesis_id",
        "full_shape_type",
        "top1_changed_vs_full",
        "shape_changed_vs_full",
        "total_score",
        "failure_score",
        "visual_score",
        "depth_score",
        "full_total_advantage",
        "full_failure_advantage",
        "zero_channels",
        "quality_reasons",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["zero_channels"] = ";".join(str(item) for item in row.get("zero_channels", []))
            payload["quality_reasons"] = "; ".join(str(item) for item in row.get("quality_reasons", []))
            writer.writerow({name: payload.get(name) for name in fieldnames})


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M4 Real Ablation Gate",
        "",
        f"- overall_status: {report['overall_status']}",
        f"- scenes: {report['num_scenes']}",
        f"- evaluable scenes: {report['num_evaluable_scenes']}",
        f"- excluded scenes: {report['num_excluded_scenes']}",
        "",
        "## Gate Reasons",
        "",
    ]
    for reason in report["gate_reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Ablation Summary",
            "",
            "| ranker | checked | top1_changed | shape_changed | mean_total_advantage | mean_failure_advantage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for ranker, summary in report["ablation_summary"].items():
        lines.append(
            "| "
            f"{ranker} | "
            f"{summary.get('checked_count', 0)} | "
            f"{summary.get('top1_changed_count', 0)} | "
            f"{summary.get('shape_changed_count', 0)} | "
            f"{float(summary.get('mean_full_total_advantage', 0.0)):.3f} | "
            f"{float(summary.get('mean_full_failure_advantage', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Scene Rows",
            "",
            "| scene_id | quality | decision | ranker | hypothesis | full | top1_changed | total |",
            "|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in report["rows"]:
        if int(row.get("rank", 0)) != 1:
            continue
        lines.append(
            "| "
            f"{row['scene_id']} | "
            f"{row.get('quality_status', '')} | "
            f"{row.get('gate_decision', '')} | "
            f"{row['ranker']} | "
            f"{row['hypothesis_id']} | "
            f"{row.get('full_hypothesis_id', '')} | "
            f"{int(row.get('top1_changed_vs_full', False))} | "
            f"{float(row['total_score']):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/real_d435_m5"))
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("annotations/m5_real_d435_masks"),
    )
    parser.add_argument(
        "--dashboard-json",
        type=Path,
        default=Path("reports/m4_real_external_mask_visual_dashboard/dashboard.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m4_real_ablation_gate"),
    )
    parser.add_argument("--background-scene-id", default="empty_table_001")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_real_ablation_gate(
        data_dir=args.data_dir,
        annotations_root=args.annotations_root,
        dashboard_json=args.dashboard_json,
        output_dir=args.output_dir,
        background_scene_id=args.background_scene_id,
        top_k=args.top_k,
    )
    print(f"Wrote M4 real ablation gate to {args.output_dir / 'ablation_gate.json'}")
    print(f"overall_status={report['overall_status']}")


if __name__ == "__main__":
    main()
