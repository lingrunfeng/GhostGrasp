#!/usr/bin/env python3
"""Calibrate simple real-data GHOST-MGG score weights from M5 ranking rows."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_VISUAL_WEIGHTS = (0.75, 1.0, 1.25)
DEFAULT_FAILURE_WEIGHTS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
DEFAULT_DEPTH_WEIGHTS = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class CandidateRow:
    scene_id: str
    target_label: str | None
    shape_hint: str | None
    ranker: str
    rank: int
    hypothesis_id: str
    shape_type: str
    visual_score: float
    failure_score: float
    depth_score: float
    source_total_score: float


@dataclass(frozen=True)
class WeakGTRule:
    scene_id: str
    target_label: str | None
    material_class: str
    object_family: str
    acceptable_shapes: tuple[str, ...]
    expected_failure_sensitive: bool
    require_shape_stability: bool
    min_failure_score_delta: float
    max_visual_score_drop: float


@dataclass(frozen=True)
class CalibratedTop1Row:
    scene_id: str
    target_label: str | None
    material_class: str
    object_family: str
    hypothesis_id: str
    shape_type: str
    baseline_hypothesis_id: str
    baseline_shape_type: str
    expected_failure_sensitive: bool
    require_shape_stability: bool
    calibrated_total: float
    visual_score: float
    failure_score: float
    depth_score: float
    visual_score_delta: float
    failure_score_delta: float
    top1_changed: bool
    shape_changed: bool
    acceptable_shape_ok: bool
    failure_gain_ok: bool
    visual_drop_ok: bool
    shape_stability_ok: bool
    weak_gt_pass: bool


@dataclass(frozen=True)
class WeightGridRow:
    visual_weight: float
    failure_weight: float
    depth_weight: float
    num_scenes: int
    weak_gt_pass_count: int
    acceptable_shape_pass_count: int
    failure_gain_checked_count: int
    failure_gain_pass_count: int
    visual_drop_pass_count: int
    shape_stability_checked_count: int
    shape_stability_pass_count: int
    top1_changed_count: int
    shape_changed_count: int
    mean_failure_score_delta: float
    mean_visual_score_delta: float
    weight_l1: float


@dataclass(frozen=True)
class CalibrationResult:
    best_weights: dict[str, float]
    best_summary: dict[str, Any]
    summary: dict[str, Any]
    grid_rows: list[WeightGridRow]
    best_top1_rows: list[CalibratedTop1Row]
    best_top1_by_scene: dict[str, CalibratedTop1Row]


def calibrate_weights(
    *,
    ranking_csv: Path,
    weak_gt_json: Path,
    visual_weights: tuple[float, ...] = DEFAULT_VISUAL_WEIGHTS,
    failure_weights: tuple[float, ...] = DEFAULT_FAILURE_WEIGHTS,
    depth_weights: tuple[float, ...] = DEFAULT_DEPTH_WEIGHTS,
) -> CalibrationResult:
    candidates = _read_candidates(ranking_csv)
    weak_gt_rules = _read_weak_gt(weak_gt_json)
    candidates_by_scene = _group_candidates(candidates)
    baseline_by_scene = _baseline_candidates(candidates)

    grid_rows: list[WeightGridRow] = []
    top1_by_grid: list[list[CalibratedTop1Row]] = []
    for visual_weight, failure_weight, depth_weight in itertools.product(
        visual_weights, failure_weights, depth_weights
    ):
        top1_rows = _evaluate_weight_set(
            weak_gt_rules=weak_gt_rules,
            candidates_by_scene=candidates_by_scene,
            baseline_by_scene=baseline_by_scene,
            visual_weight=float(visual_weight),
            failure_weight=float(failure_weight),
            depth_weight=float(depth_weight),
        )
        grid_rows.append(
            _summarize_weight_set(
                top1_rows,
                visual_weight=float(visual_weight),
                failure_weight=float(failure_weight),
                depth_weight=float(depth_weight),
            )
        )
        top1_by_grid.append(top1_rows)

    if not grid_rows:
        raise ValueError("weight grid produced no rows")
    best_index = _best_grid_index(grid_rows)
    best_grid = grid_rows[best_index]
    best_top1_rows = top1_by_grid[best_index]
    best_weights = {
        "visual": best_grid.visual_weight,
        "failure": best_grid.failure_weight,
        "depth": best_grid.depth_weight,
    }
    best_summary = _grid_row_to_summary(best_grid)
    summary = {
        "schema_version": "m4_real_weight_calibration_v1",
        "num_scenes": best_grid.num_scenes,
        "num_weight_sets": len(grid_rows),
        "best_grid_index": best_index,
    }
    return CalibrationResult(
        best_weights=best_weights,
        best_summary=best_summary,
        summary=summary,
        grid_rows=grid_rows,
        best_top1_rows=best_top1_rows,
        best_top1_by_scene={row.scene_id: row for row in best_top1_rows},
    )


def write_calibration_report(result: CalibrationResult, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_dicts = [asdict(row) for row in result.grid_rows]
    top1_dicts = [asdict(row) for row in result.best_top1_rows]

    _write_json(
        output_dir / "best_weights.json",
        {
            "schema_version": "m4_real_weight_calibration_best_v1",
            "best_weights": result.best_weights,
            "best_summary": result.best_summary,
            "summary": result.summary,
        },
    )
    _write_json(
        output_dir / "calibration_grid.json",
        {
            "schema_version": "m4_real_weight_calibration_grid_v1",
            "rows": grid_dicts,
            "summary": result.summary,
        },
    )
    _write_json(
        output_dir / "calibrated_top1.json",
        {
            "schema_version": "m4_real_weight_calibration_top1_v1",
            "best_weights": result.best_weights,
            "rows": top1_dicts,
        },
    )
    _write_csv(output_dir / "calibration_grid.csv", grid_dicts)
    _write_csv(output_dir / "calibrated_top1.csv", top1_dicts)
    _write_index(result, output_dir / "index.md")


def _read_candidates(path: Path) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                CandidateRow(
                    scene_id=str(row["scene_id"]),
                    target_label=row.get("target_label") or None,
                    shape_hint=row.get("shape_hint") or None,
                    ranker=str(row["ranker"]),
                    rank=int(row["rank"]),
                    hypothesis_id=str(row["hypothesis_id"]),
                    shape_type=str(row["shape_type"]),
                    visual_score=float(row["visual_score"]),
                    failure_score=float(row["failure_score"]),
                    depth_score=float(row["depth_score"]),
                    source_total_score=float(row["total_score"]),
                )
            )
    if not rows:
        raise ValueError(f"ranking CSV has no rows: {path}")
    return rows


def _read_weak_gt(path: Path) -> list[WeakGTRule]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m5_real_weak_gt_v1":
        raise ValueError(f"unsupported weak-GT schema: {payload.get('schema_version')}")
    rules: list[WeakGTRule] = []
    for scene in payload.get("scenes", []):
        acceptable_shapes = tuple(str(shape) for shape in scene.get("acceptable_shapes", []))
        if not acceptable_shapes:
            raise ValueError(f"{scene.get('scene_id')} has no acceptable_shapes")
        rules.append(
            WeakGTRule(
                scene_id=str(scene["scene_id"]),
                target_label=scene.get("target_label"),
                material_class=str(scene.get("material_class", "unknown")),
                object_family=str(scene.get("object_family", "unknown")),
                acceptable_shapes=acceptable_shapes,
                expected_failure_sensitive=_bool(scene.get("expected_failure_sensitive", False)),
                require_shape_stability=_bool(scene.get("require_shape_stability", False)),
                min_failure_score_delta=float(scene.get("min_failure_score_delta", 0.0)),
                max_visual_score_drop=float(scene.get("max_visual_score_drop", 0.12)),
            )
        )
    if not rules:
        raise ValueError("weak-GT manifest has no scenes")
    return sorted(rules, key=lambda rule: rule.scene_id)


def _group_candidates(candidates: list[CandidateRow]) -> dict[str, list[CandidateRow]]:
    by_scene: dict[str, dict[str, CandidateRow]] = {}
    for candidate in candidates:
        by_scene.setdefault(candidate.scene_id, {})
        current = by_scene[candidate.scene_id].get(candidate.hypothesis_id)
        if current is None or candidate.rank < current.rank:
            by_scene[candidate.scene_id][candidate.hypothesis_id] = candidate
    return {
        scene_id: sorted(scene_candidates.values(), key=lambda row: row.hypothesis_id)
        for scene_id, scene_candidates in by_scene.items()
    }


def _baseline_candidates(candidates: list[CandidateRow]) -> dict[str, CandidateRow]:
    baseline: dict[str, CandidateRow] = {}
    for candidate in candidates:
        if candidate.ranker != "silhouette_only":
            continue
        current = baseline.get(candidate.scene_id)
        if current is None or candidate.rank < current.rank:
            baseline[candidate.scene_id] = candidate
    return baseline


def _evaluate_weight_set(
    *,
    weak_gt_rules: list[WeakGTRule],
    candidates_by_scene: dict[str, list[CandidateRow]],
    baseline_by_scene: dict[str, CandidateRow],
    visual_weight: float,
    failure_weight: float,
    depth_weight: float,
) -> list[CalibratedTop1Row]:
    rows = []
    for rule in weak_gt_rules:
        scene_candidates = candidates_by_scene.get(rule.scene_id)
        if not scene_candidates:
            raise KeyError(f"missing ranking candidates for {rule.scene_id}")
        baseline = baseline_by_scene.get(rule.scene_id)
        if baseline is None:
            raise KeyError(f"missing silhouette baseline for {rule.scene_id}")
        top = max(
            scene_candidates,
            key=lambda candidate: (
                _calibrated_total(candidate, visual_weight, failure_weight, depth_weight),
                candidate.visual_score,
                candidate.failure_score,
                candidate.hypothesis_id,
            ),
        )
        calibrated_total = _calibrated_total(top, visual_weight, failure_weight, depth_weight)
        failure_delta = top.failure_score - baseline.failure_score
        visual_delta = top.visual_score - baseline.visual_score
        acceptable_shape_ok = top.shape_type in rule.acceptable_shapes
        failure_gain_ok = (
            failure_delta >= rule.min_failure_score_delta
            if rule.expected_failure_sensitive
            else True
        )
        visual_drop_ok = visual_delta >= -rule.max_visual_score_drop
        shape_stability_ok = (
            top.shape_type == baseline.shape_type if rule.require_shape_stability else True
        )
        weak_gt_pass = (
            acceptable_shape_ok
            and failure_gain_ok
            and visual_drop_ok
            and shape_stability_ok
        )
        rows.append(
            CalibratedTop1Row(
                scene_id=rule.scene_id,
                target_label=rule.target_label or top.target_label,
                material_class=rule.material_class,
                object_family=rule.object_family,
                hypothesis_id=top.hypothesis_id,
                shape_type=top.shape_type,
                baseline_hypothesis_id=baseline.hypothesis_id,
                baseline_shape_type=baseline.shape_type,
                expected_failure_sensitive=rule.expected_failure_sensitive,
                require_shape_stability=rule.require_shape_stability,
                calibrated_total=round(calibrated_total, 6),
                visual_score=round(top.visual_score, 6),
                failure_score=round(top.failure_score, 6),
                depth_score=round(top.depth_score, 6),
                visual_score_delta=round(visual_delta, 6),
                failure_score_delta=round(failure_delta, 6),
                top1_changed=top.hypothesis_id != baseline.hypothesis_id,
                shape_changed=top.shape_type != baseline.shape_type,
                acceptable_shape_ok=acceptable_shape_ok,
                failure_gain_ok=failure_gain_ok,
                visual_drop_ok=visual_drop_ok,
                shape_stability_ok=shape_stability_ok,
                weak_gt_pass=weak_gt_pass,
            )
        )
    return rows


def _summarize_weight_set(
    rows: list[CalibratedTop1Row],
    *,
    visual_weight: float,
    failure_weight: float,
    depth_weight: float,
) -> WeightGridRow:
    if not rows:
        raise ValueError("cannot summarize empty calibrated top-1 rows")
    failure_checked = [row for row in rows if row.expected_failure_sensitive]
    shape_checked = [row for row in rows if row.require_shape_stability]
    return WeightGridRow(
        visual_weight=visual_weight,
        failure_weight=failure_weight,
        depth_weight=depth_weight,
        num_scenes=len(rows),
        weak_gt_pass_count=sum(1 for row in rows if row.weak_gt_pass),
        acceptable_shape_pass_count=sum(1 for row in rows if row.acceptable_shape_ok),
        failure_gain_checked_count=len(failure_checked),
        failure_gain_pass_count=sum(1 for row in failure_checked if row.failure_gain_ok),
        visual_drop_pass_count=sum(1 for row in rows if row.visual_drop_ok),
        shape_stability_checked_count=len(shape_checked),
        shape_stability_pass_count=sum(1 for row in shape_checked if row.shape_stability_ok),
        top1_changed_count=sum(1 for row in rows if row.top1_changed),
        shape_changed_count=sum(1 for row in rows if row.shape_changed),
        mean_failure_score_delta=round(
            sum(row.failure_score_delta for row in rows) / float(len(rows)), 6
        ),
        mean_visual_score_delta=round(
            sum(row.visual_score_delta for row in rows) / float(len(rows)), 6
        ),
        weight_l1=round(abs(visual_weight) + abs(failure_weight) + abs(depth_weight), 6),
    )


def _best_grid_index(rows: list[WeightGridRow]) -> int:
    def key(row: WeightGridRow) -> tuple[float, ...]:
        return (
            row.weak_gt_pass_count,
            row.failure_gain_pass_count,
            row.mean_failure_score_delta,
            row.mean_visual_score_delta,
            -row.weight_l1,
        )

    return max(range(len(rows)), key=lambda index: key(rows[index]))


def _grid_row_to_summary(row: WeightGridRow) -> dict[str, Any]:
    return {
        "num_scenes": row.num_scenes,
        "weak_gt_pass_count": row.weak_gt_pass_count,
        "acceptable_shape_pass_count": row.acceptable_shape_pass_count,
        "failure_gain_checked_count": row.failure_gain_checked_count,
        "failure_gain_pass_count": row.failure_gain_pass_count,
        "visual_drop_pass_count": row.visual_drop_pass_count,
        "shape_stability_checked_count": row.shape_stability_checked_count,
        "shape_stability_pass_count": row.shape_stability_pass_count,
        "top1_changed_count": row.top1_changed_count,
        "shape_changed_count": row.shape_changed_count,
        "mean_failure_score_delta": row.mean_failure_score_delta,
        "mean_visual_score_delta": row.mean_visual_score_delta,
    }


def _calibrated_total(
    candidate: CandidateRow, visual_weight: float, failure_weight: float, depth_weight: float
) -> float:
    return (
        visual_weight * candidate.visual_score
        + failure_weight * candidate.failure_score
        + depth_weight * candidate.depth_score
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_index(result: CalibrationResult, output_path: Path) -> None:
    lines = [
        "# M4 Real Weight Calibration",
        "",
        "Offline calibration over existing M5 real ranking candidates.",
        "",
        "## Best Weights",
        "",
        f"- visual: {result.best_weights['visual']:.2f}",
        f"- failure: {result.best_weights['failure']:.2f}",
        f"- depth: {result.best_weights['depth']:.2f}",
        f"- scenes: {result.best_summary['num_scenes']}",
        f"- weak_gt_pass: {result.best_summary['weak_gt_pass_count']}/{result.best_summary['num_scenes']}",
        f"- failure_gain_pass: {result.best_summary['failure_gain_pass_count']}/{result.best_summary['failure_gain_checked_count']}",
        f"- mean_failure_score_delta: {result.best_summary['mean_failure_score_delta']:.3f}",
        f"- mean_visual_score_delta: {result.best_summary['mean_visual_score_delta']:.3f}",
        "",
        "## Calibrated Top-1 Rows",
        "",
        "| scene_id | material | top1 | shape | baseline | pass | dF | dV |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in result.best_top1_rows:
        lines.append(
            "| "
            f"{row.scene_id} | "
            f"{row.material_class} | "
            f"{row.hypothesis_id} | "
            f"{row.shape_type} | "
            f"{row.baseline_hypothesis_id} | "
            f"{int(row.weak_gt_pass)} | "
            f"{row.failure_score_delta:.3f} | "
            f"{row.visual_score_delta:.3f} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=Path("reports/m5_real_d435_ranking/m5_real_ranking.csv"),
    )
    parser.add_argument(
        "--weak-gt",
        type=Path,
        default=Path("annotations/m5_real_d435_weak_gt/weak_gt.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m4_real_weight_calibration"),
    )
    parser.add_argument(
        "--visual-weights",
        default=",".join(str(value) for value in DEFAULT_VISUAL_WEIGHTS),
    )
    parser.add_argument(
        "--failure-weights",
        default=",".join(str(value) for value in DEFAULT_FAILURE_WEIGHTS),
    )
    parser.add_argument(
        "--depth-weights",
        default=",".join(str(value) for value in DEFAULT_DEPTH_WEIGHTS),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = calibrate_weights(
        ranking_csv=args.ranking_csv,
        weak_gt_json=args.weak_gt,
        visual_weights=_parse_float_tuple(args.visual_weights),
        failure_weights=_parse_float_tuple(args.failure_weights),
        depth_weights=_parse_float_tuple(args.depth_weights),
    )
    write_calibration_report(result, args.output_dir)
    print(
        "Wrote M4 real weight calibration "
        f"for {result.summary['num_scenes']} scenes to {args.output_dir}; "
        f"best_weights={result.best_weights}"
    )


if __name__ == "__main__":
    main()
