from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ghost_mgg_core_py.evaluation.m3_dataset import M3EvidenceSample, load_m3_capture
from ghost_mgg_core_py.evaluation.m4_offline_ranking import M4OfflineRankingRow, run_offline_ranking
from ghost_mgg_core_py.hypotheses.primitives import PrimitiveHypothesis
from ghost_mgg_core_py.rendering.proxy_renderer import render_proxy
from ghost_mgg_core_py.scoring.score_terms import silhouette_iou


@dataclass(frozen=True)
class M4GeometryTruth:
    scenario_id: str
    shape_type: str
    center_u: float
    center_v: float
    size_u_px: float
    size_v_px: float


@dataclass(frozen=True)
class M4GeometryEvalRow:
    scenario_id: str
    failure_mode: str
    ranker: str
    truth_shape_type: str
    top1_hypothesis_id: str
    top1_shape_type: str
    top1_shape_correct: bool
    topk_contains_truth_shape: bool
    topk_contains_exact_proxy: bool
    top1_center_error_px: float
    top1_size_u_error_px: float
    top1_size_v_error_px: float
    top1_silhouette_iou: float
    top1_total_score: float
    top1_failure_score: float


def truth_from_sample(sample: M3EvidenceSample) -> M4GeometryTruth:
    arrays = sample.load_arrays()
    target_mask = arrays["target_mask"].astype(bool)
    if not target_mask.any():
        raise ValueError(f"sample target mask is empty: {sample.scenario_id}")

    rows, cols = np.nonzero(target_mask)
    min_u = float(cols.min())
    max_u = float(cols.max())
    min_v = float(rows.min())
    max_v = float(rows.max())
    return M4GeometryTruth(
        scenario_id=sample.scenario_id,
        shape_type="box",
        center_u=(min_u + max_u) / 2.0,
        center_v=(min_v + max_v) / 2.0,
        size_u_px=max_u - min_u + 1.0,
        size_v_px=max_v - min_v + 1.0,
    )


def evaluate_ranking_rows(
    capture_dir: str | Path,
    ranking_rows: list[M4OfflineRankingRow],
    top_k: int = 3,
    exact_iou_threshold: float = 0.98,
) -> list[M4GeometryEvalRow]:
    samples = {sample.scenario_id: sample for sample in load_m3_capture(capture_dir)}
    grouped: dict[tuple[str, str], list[M4OfflineRankingRow]] = {}
    for row in ranking_rows:
        grouped.setdefault((row.scenario_id, row.ranker), []).append(row)

    eval_rows: list[M4GeometryEvalRow] = []
    for (scenario_id, ranker), rows in sorted(grouped.items()):
        if scenario_id not in samples:
            raise KeyError(f"ranking row references missing sample: {scenario_id}")
        sample = samples[scenario_id]
        target_mask = sample.load_arrays()["target_mask"].astype(bool)
        truth = truth_from_sample(sample)
        ranked = sorted(rows, key=lambda item: item.rank)[: max(1, int(top_k))]
        top1 = ranked[0]

        top1_iou = _row_silhouette_iou(top1, target_mask)
        topk_contains_truth_shape = any(row.shape_type == truth.shape_type for row in ranked)
        topk_contains_exact_proxy = any(
            row.shape_type == truth.shape_type
            and _row_silhouette_iou(row, target_mask) >= exact_iou_threshold
            for row in ranked
        )

        eval_rows.append(
            M4GeometryEvalRow(
                scenario_id=scenario_id,
                failure_mode=sample.failure_mode,
                ranker=ranker,
                truth_shape_type=truth.shape_type,
                top1_hypothesis_id=top1.hypothesis_id,
                top1_shape_type=top1.shape_type,
                top1_shape_correct=top1.shape_type == truth.shape_type,
                topk_contains_truth_shape=topk_contains_truth_shape,
                topk_contains_exact_proxy=topk_contains_exact_proxy,
                top1_center_error_px=float(
                    np.hypot(top1.center_u - truth.center_u, top1.center_v - truth.center_v)
                ),
                top1_size_u_error_px=abs(top1.size_u_px - truth.size_u_px),
                top1_size_v_error_px=abs(top1.size_v_px - truth.size_v_px),
                top1_silhouette_iou=float(top1_iou),
                top1_total_score=top1.total_score,
                top1_failure_score=top1.failure_score,
            )
        )
    return eval_rows


def run_geometry_eval(capture_dir: str | Path, top_k: int = 3) -> list[M4GeometryEvalRow]:
    ranking_rows = run_offline_ranking(capture_dir, top_k=top_k)
    return evaluate_ranking_rows(capture_dir, ranking_rows, top_k=top_k)


def write_geometry_eval_reports(
    rows: list[M4GeometryEvalRow],
    output_csv: str | Path,
    output_json: str | Path,
) -> None:
    if not rows:
        raise ValueError("no geometry evaluation rows to write")
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    row_dicts = [asdict(row) for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)

    report = {
        "schema_version": "m4_geometry_ranking_eval_v1",
        "num_rows": len(rows),
        "aggregate_by_ranker": _aggregate_by_ranker(rows),
        "rows": row_dicts,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_ranking_rows_from_json(path: str | Path) -> list[M4OfflineRankingRow]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return [M4OfflineRankingRow(**row) for row in report.get("rows", [])]


def _row_silhouette_iou(row: M4OfflineRankingRow, target_mask: np.ndarray) -> float:
    hypothesis = PrimitiveHypothesis(
        hypothesis_id=row.hypothesis_id,
        shape_type=row.shape_type,
        center_uv=(row.center_u, row.center_v),
        size_px=(row.size_u_px, row.size_v_px),
        depth_m=1.20,
        height_m=0.08,
    )
    rendered = render_proxy(hypothesis, target_mask.shape)
    return silhouette_iou(rendered.silhouette, target_mask)


def _aggregate_by_ranker(rows: list[M4GeometryEvalRow]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for ranker in sorted({row.ranker for row in rows}):
        selected = [row for row in rows if row.ranker == ranker]
        count = float(len(selected))
        aggregate[ranker] = {
            "num_samples": len(selected),
            "top1_shape_accuracy": sum(row.top1_shape_correct for row in selected) / count,
            "topk_shape_recall": sum(row.topk_contains_truth_shape for row in selected) / count,
            "topk_exact_proxy_recall": sum(row.topk_contains_exact_proxy for row in selected) / count,
            "mean_top1_silhouette_iou": sum(row.top1_silhouette_iou for row in selected) / count,
            "mean_top1_center_error_px": sum(row.top1_center_error_px for row in selected) / count,
            "mean_top1_size_u_error_px": sum(row.top1_size_u_error_px for row in selected) / count,
            "mean_top1_size_v_error_px": sum(row.top1_size_v_error_px for row in selected) / count,
        }
    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--ranking-json", default=None, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--top-k", default=3, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.ranking_json is not None and args.ranking_json.exists():
        ranking_rows = load_ranking_rows_from_json(args.ranking_json)
        rows = evaluate_ranking_rows(args.capture_dir, ranking_rows, top_k=args.top_k)
    else:
        rows = run_geometry_eval(args.capture_dir, top_k=args.top_k)
    write_geometry_eval_reports(rows, args.output_csv, args.output_json)
    print(f"Wrote {len(rows)} M4 geometry evaluation rows")
    print(f"CSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
