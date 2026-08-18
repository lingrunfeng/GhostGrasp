#!/usr/bin/env python3
"""Summarize current M4 readiness gates from generated real-data reports."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateCheck:
    gate_id: str
    status: str
    summary: str
    details: str


@dataclass(frozen=True)
class GateSummary:
    schema_version: str
    overall_status: str
    num_scenes: int
    best_weights: dict[str, float]
    key_counts: dict[str, int | float]
    checks: list[GateCheck]


def build_gate_summary(
    *,
    calibration_json: Path,
    dashboard_json: Path,
    weak_gt_json: Path,
    ranking_json: Path,
    visual_board_manifest: Path,
) -> GateSummary:
    calibration = _read_json(
        calibration_json,
        expected_schema="m4_real_weight_calibration_best_v1",
    )
    dashboard = _read_json(dashboard_json, expected_schema="m4_real_dashboard_v1")
    weak_gt = _read_json(weak_gt_json, expected_schema="m5_real_weak_gt_eval_v1")
    ranking = _read_json(ranking_json, expected_schema="m5_real_ranking_v1")
    visual_board = _read_json(
        visual_board_manifest,
        expected_schema="m4_visual_ranking_board_manifest_v1",
    )

    num_scenes = _int(dashboard["num_scenes"])
    best_weights = {
        "visual": _float(calibration["best_weights"].get("visual", 0.0)),
        "failure": _float(calibration["best_weights"].get("failure", 0.0)),
        "depth": _float(calibration["best_weights"].get("depth", 0.0)),
    }
    checks = [
        _check_calibration_weights(best_weights),
        _check_ranking_coverage(
            dashboard_scenes=num_scenes,
            ranking_scenes=_int(ranking["num_scenes"]),
            weak_gt_scenes=_int(weak_gt["num_scenes"]),
        ),
        _check_weak_gt(
            num_scenes=_int(weak_gt["num_scenes"]),
            weak_gt_pass_count=_int(weak_gt["weak_gt_pass_count"]),
        ),
        _check_failure_gain(
            checked_count=_int(weak_gt["failure_gain_checked_count"]),
            pass_count=_int(weak_gt["failure_gain_pass_count"]),
        ),
        _check_visual_boards(
            dashboard=dashboard,
            visual_board=visual_board,
        ),
    ]
    key_counts: dict[str, int | float] = {
        "dashboard_scenes": num_scenes,
        "ranking_scenes": _int(ranking["num_scenes"]),
        "weak_gt_pass_count": _int(weak_gt["weak_gt_pass_count"]),
        "failure_gain_checked_count": _int(weak_gt["failure_gain_checked_count"]),
        "failure_gain_pass_count": _int(weak_gt["failure_gain_pass_count"]),
        "visual_board_scenes": _int(visual_board["num_scenes"]),
        "top1_changed_count": _int(dashboard.get("top1_changed_count", 0)),
        "shape_changed_count": _int(dashboard.get("shape_changed_count", 0)),
        "mean_failure_score_delta": _float(dashboard.get("mean_failure_score_delta", 0.0)),
    }
    return GateSummary(
        schema_version="m4_gate_summary_v1",
        overall_status=_overall_status(checks),
        num_scenes=num_scenes,
        best_weights=best_weights,
        key_counts=key_counts,
        checks=checks,
    )


def write_gate_summary(summary: GateSummary, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    check_dicts = [asdict(check) for check in summary.checks]
    payload = {
        "schema_version": summary.schema_version,
        "overall_status": summary.overall_status,
        "num_scenes": summary.num_scenes,
        "best_weights": summary.best_weights,
        "key_counts": summary.key_counts,
        "checks": check_dicts,
    }
    (output_dir / "gate_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "gate_checks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["gate_id", "status", "summary", "details"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(check_dicts)
    _write_index(summary, output_dir / "index.md")


def _check_calibration_weights(weights: dict[str, float]) -> GateCheck:
    failure_weight = weights["failure"]
    if failure_weight > 0.0:
        return GateCheck(
            gate_id="calibration_weights",
            status="pass",
            summary=f"failure weight is positive ({failure_weight:.3f})",
            details=f"weights={_compact_json(weights)}",
        )
    return GateCheck(
        gate_id="calibration_weights",
        status="fail",
        summary="failure weight is zero; failure evidence is not influencing ranking",
        details=f"weights={_compact_json(weights)}",
    )


def _check_ranking_coverage(
    *,
    dashboard_scenes: int,
    ranking_scenes: int,
    weak_gt_scenes: int,
) -> GateCheck:
    if dashboard_scenes == ranking_scenes == weak_gt_scenes:
        return GateCheck(
            gate_id="ranking_coverage",
            status="pass",
            summary=f"all reports cover {dashboard_scenes} scenes",
            details=(
                f"dashboard={dashboard_scenes}, ranking={ranking_scenes}, "
                f"weak_gt={weak_gt_scenes}"
            ),
        )
    return GateCheck(
        gate_id="ranking_coverage",
        status="fail",
        summary="scene counts differ across ranking, dashboard, and weak-GT reports",
        details=(
            f"dashboard={dashboard_scenes}, ranking={ranking_scenes}, "
            f"weak_gt={weak_gt_scenes}"
        ),
    )


def _check_weak_gt(*, num_scenes: int, weak_gt_pass_count: int) -> GateCheck:
    status = "pass" if weak_gt_pass_count == num_scenes else "fail"
    return GateCheck(
        gate_id="weak_gt",
        status=status,
        summary=f"{weak_gt_pass_count}/{num_scenes} weak-GT scenes pass",
        details="weak-GT is conservative proxy-level validation, not metric 3D GT",
    )


def _check_failure_gain(*, checked_count: int, pass_count: int) -> GateCheck:
    if checked_count == 0:
        return GateCheck(
            gate_id="failure_gain",
            status="warn",
            summary="no failure-sensitive scenes were checked",
            details="add transparent/translucent annotated scenes before using this gate",
        )
    status = "pass" if pass_count == checked_count else "fail"
    return GateCheck(
        gate_id="failure_gain",
        status=status,
        summary=f"{pass_count}/{checked_count} failure-sensitive scenes pass",
        details="checks that failure-aware top-1 gains enough failure score",
    )


def _check_visual_boards(*, dashboard: dict[str, Any], visual_board: dict[str, Any]) -> GateCheck:
    dashboard_scene_ids = _scene_ids(dashboard.get("rows", []))
    board_scene_ids = _scene_ids(visual_board.get("scenes", []))
    missing = sorted(dashboard_scene_ids - board_scene_ids)
    extra = sorted(board_scene_ids - dashboard_scene_ids)
    if not missing and len(board_scene_ids) == _int(visual_board["num_scenes"]):
        return GateCheck(
            gate_id="visual_boards",
            status="pass",
            summary=f"{len(board_scene_ids)}/{len(dashboard_scene_ids)} visual boards generated",
            details="all dashboard scenes have a board entry",
        )
    return GateCheck(
        gate_id="visual_boards",
        status="fail",
        summary="visual board coverage does not match dashboard scenes",
        details=f"missing={missing}, extra={extra}",
    )


def _read_json(path: Path, *, expected_schema: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema != expected_schema:
        raise ValueError(f"{path} has schema {schema!r}, expected {expected_schema!r}")
    return payload


def _scene_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["scene_id"]) for row in rows}


def _overall_status(checks: list[GateCheck]) -> str:
    statuses = {check.status for check in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _int(value: Any) -> int:
    return int(float(value))


def _float(value: Any) -> float:
    return float(value)


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_index(summary: GateSummary, output_path: Path) -> None:
    lines = [
        "# M4 Gate Summary",
        "",
        f"overall_status: {summary.overall_status}",
        "",
        "This is an engineering readiness report for the current real-data M4 chain.",
        "It is not a final paper-result table.",
        "",
        "## Gate Checks",
        "",
        "| gate | status | summary |",
        "|---|---|---|",
    ]
    for check in summary.checks:
        lines.append(f"| {check.gate_id} | {check.status} | {check.summary} |")
    lines.extend(
        [
            "",
            "## Key Counts",
            "",
            f"- scenes: {summary.num_scenes}",
            f"- weak_gt_pass: {summary.key_counts['weak_gt_pass_count']}/{summary.num_scenes}",
            (
                "- failure_gain_pass: "
                f"{summary.key_counts['failure_gain_pass_count']}/"
                f"{summary.key_counts['failure_gain_checked_count']}"
            ),
            f"- top1_changed: {summary.key_counts['top1_changed_count']}/{summary.num_scenes}",
            f"- shape_changed: {summary.key_counts['shape_changed_count']}/{summary.num_scenes}",
            (
                "- mean_failure_score_delta: "
                f"{summary.key_counts['mean_failure_score_delta']:.3f}"
            ),
            "",
            "## Calibrated Weights",
            "",
            "| term | weight |",
            "|---|---:|",
        ]
    )
    for key in ("visual", "failure", "depth"):
        lines.append(f"| {key} | {summary.best_weights[key]:.3f} |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=Path("reports/m4_real_weight_calibration/best_weights.json"),
    )
    parser.add_argument(
        "--dashboard-json",
        type=Path,
        default=Path("reports/m4_real_dashboard/dashboard.json"),
    )
    parser.add_argument(
        "--weak-gt-json",
        type=Path,
        default=Path("reports/m5_real_d435_weak_gt_eval/weak_gt_eval.json"),
    )
    parser.add_argument(
        "--ranking-json",
        type=Path,
        default=Path("reports/m5_real_d435_ranking/m5_real_ranking.json"),
    )
    parser.add_argument(
        "--visual-board-manifest",
        type=Path,
        default=Path("reports/m4_visual_ranking_board/manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m4_gate_summary"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_gate_summary(
        calibration_json=args.calibration_json,
        dashboard_json=args.dashboard_json,
        weak_gt_json=args.weak_gt_json,
        ranking_json=args.ranking_json,
        visual_board_manifest=args.visual_board_manifest,
    )
    write_gate_summary(summary, args.output_dir)
    print(
        f"Wrote M4 gate summary ({summary.overall_status}) "
        f"for {summary.num_scenes} scenes to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
