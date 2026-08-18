from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "m4_joint_hypothesis_report_v1"
CSV_FIELDS = [
    "source_type",
    "rank_group",
    "joint_rank",
    "target_or_scene_id",
    "ranker",
    "hypothesis_id",
    "shape_type",
    "decision",
    "joint_score",
    "visual_score",
    "grasp_score",
    "moveit_pregrasp_planned",
    "descent_clearance_status",
    "path_points_count",
    "descent_points_count",
    "failure_reason",
]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path_for_report(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _ranking_index(ranking_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in ranking_rows:
        scene_id = row.get("scene_id") or row.get("scenario_id")
        key = (str(scene_id), str(row.get("ranker")), str(row.get("hypothesis_id")))
        current = output.get(key)
        if current is None or int(row.get("rank", 9999)) < int(current.get("rank", 9999)):
            output[key] = row
    return output


def _decision_for_real(grasp_row: dict[str, Any]) -> str:
    return "candidate" if bool(grasp_row.get("valid")) else "reject"


def _real_joint_score(ranking_row: dict[str, Any] | None, grasp_row: dict[str, Any]) -> float:
    visual_score = _as_float((ranking_row or {}).get("total_score"), 0.0)
    grasp_score = _as_float(grasp_row.get("score"), 0.0)
    validity_bonus = 0.05 if bool(grasp_row.get("valid")) else -0.25
    return max(0.0, min(1.0, 0.62 * visual_score + 0.33 * grasp_score + validity_bonus))


def _build_real_rows(
    ranking_rows: list[dict[str, Any]],
    graspability_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranking_by_key = _ranking_index(ranking_rows)
    rows = []
    for grasp_row in graspability_rows:
        scene_id = str(grasp_row.get("scene_id"))
        ranker = str(grasp_row.get("ranker"))
        hypothesis_id = str(grasp_row.get("hypothesis_id"))
        ranking_row = ranking_by_key.get((scene_id, ranker, hypothesis_id))
        visual_score = None if ranking_row is None else _as_float(ranking_row.get("total_score"))
        grasp_score = _as_float(grasp_row.get("score"))
        rows.append(
            {
                "source_type": "real_graspability",
                "rank_group": f"real:{scene_id}",
                "target_or_scene_id": scene_id,
                "ranker": ranker,
                "hypothesis_id": hypothesis_id,
                "shape_type": grasp_row.get("shape_type") or (ranking_row or {}).get("shape_type"),
                "decision": _decision_for_real(grasp_row),
                "joint_score": round(_real_joint_score(ranking_row, grasp_row), 6),
                "visual_score": visual_score,
                "grasp_score": grasp_score,
                "moveit_pregrasp_planned": None,
                "descent_clearance_status": None,
                "path_points_count": None,
                "descent_points_count": None,
                "failure_reason": grasp_row.get("failure_reason", ""),
            }
        )
    return rows


def _path_points_count(row: dict[str, Any]) -> int:
    for attempt in row.get("attempts", []):
        if attempt.get("planned"):
            return len(attempt.get("path_points_world", []))
    return 0


def _sim_decision(row: dict[str, Any]) -> str:
    if bool(row.get("planned")) and row.get("descent_clearance", {}).get("status") == "ok":
        return "executable"
    return "reject"


def _build_sim_rows(moveit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in moveit_rows:
        clearance_status = row.get("descent_clearance", {}).get("status")
        decision = _sim_decision(row)
        rows.append(
            {
                "source_type": "sim_moveit",
                "rank_group": "sim:m4_tabletop",
                "target_or_scene_id": str(row.get("target_id")),
                "ranker": "moveit_dryrun",
                "hypothesis_id": str(row.get("target_id")),
                "shape_type": row.get("shape_type"),
                "decision": decision,
                "joint_score": 1.0 if decision == "executable" else 0.0,
                "visual_score": None,
                "grasp_score": None,
                "moveit_pregrasp_planned": bool(row.get("pregrasp_planned")),
                "descent_clearance_status": clearance_status,
                "path_points_count": _path_points_count(row),
                "descent_points_count": len(row.get("descent_points_world", [])),
                "failure_reason": row.get("failure_reason", ""),
            }
        )
    return rows


def _rank_within_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["rank_group"]), []).append(row)

    ranked_rows = []
    for rank_group in sorted(grouped):
        group_rows = sorted(
            grouped[rank_group],
            key=lambda item: (-float(item["joint_score"]), item["decision"], item["hypothesis_id"]),
        )
        for index, row in enumerate(group_rows, start=1):
            ranked = dict(row)
            ranked["joint_rank"] = index
            ranked_rows.append(ranked)
    return ranked_rows


def build_joint_hypothesis_report(
    real_ranking_json: str | Path,
    graspability_json: str | Path,
    moveit_json: str | Path,
) -> dict[str, Any]:
    real_ranking = load_json(real_ranking_json)
    graspability = load_json(graspability_json)
    moveit = load_json(moveit_json)

    rows = _rank_within_groups(
        _build_real_rows(real_ranking.get("rows", []), graspability.get("rows", []))
        + _build_sim_rows(moveit.get("rows", []))
    )
    executable_rows = sum(1 for row in rows if row["decision"] in {"candidate", "executable"})
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "real_ranking_json": _path_for_report(real_ranking_json),
            "graspability_json": _path_for_report(graspability_json),
            "moveit_json": _path_for_report(moveit_json),
        },
        "summary": {
            "total_rows": len(rows),
            "real_rows": sum(1 for row in rows if row["source_type"] == "real_graspability"),
            "sim_moveit_rows": sum(1 for row in rows if row["source_type"] == "sim_moveit"),
            "executable_rows": executable_rows,
        },
        "rows": rows,
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_joint_hypothesis_reports(report: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "joint_hypotheses.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_path / "joint_hypotheses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})

    (output_path / "index.md").write_text(render_index(report), encoding="utf-8")


def render_index(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# M4 Joint Hypothesis Report",
        "",
        f"- total rows: {summary.get('total_rows', len(report.get('rows', [])))}",
        f"- executable/candidate rows: {summary.get('executable_rows', 0)}",
        f"- real graspability rows: {summary.get('real_rows', 0)}",
        f"- sim MoveIt rows: {summary.get('sim_moveit_rows', 0)}",
        "",
        "| group | rank | id | shape | decision | score | clearance |",
        "| --- | ---: | --- | --- | --- | ---: | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {group} | {rank} | {identity} | {shape} | {decision} | {score:.3f} | {clearance} |".format(
                group=row["rank_group"],
                rank=row["joint_rank"],
                identity=row["hypothesis_id"],
                shape=row.get("shape_type") or "",
                decision=row["decision"],
                score=float(row["joint_score"]),
                clearance=row.get("descent_clearance_status") or "",
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-ranking-json", default="reports/m5_real_d435_ranking/m5_real_ranking.json")
    parser.add_argument("--graspability-json", default="reports/m4_graspability_dryrun/graspability.json")
    parser.add_argument("--moveit-json", default="reports/m4_sim_moveit_dryrun/plan_results.json")
    parser.add_argument("--output-dir", default="reports/m4_joint_hypotheses")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_joint_hypothesis_report(
        args.real_ranking_json,
        args.graspability_json,
        args.moveit_json,
    )
    write_joint_hypothesis_reports(report, args.output_dir)
    print(
        "M4 joint hypothesis report wrote "
        f"{report['summary']['total_rows']} rows to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
