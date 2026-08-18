from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ghost_mgg_core_py.evaluation.synthetic_scene import make_failure_ranking_scene
from ghost_mgg_core_py.ghost_mgg_v0 import (
    GhostMGGV0Config,
    SILHOUETTE_ONLY_WEIGHTS,
    run_ghost_mgg_v0,
)


@dataclass(frozen=True)
class M4HardSyntheticRow:
    scene_id: str
    ranker: str
    ground_truth_id: str
    top1_hypothesis_id: str
    top1_correct: bool
    ground_truth_rank: int
    top1_visual_score: float
    top1_failure_score: float
    top1_total_score: float


def make_hard_synthetic_scenes():
    return (
        make_failure_ranking_scene("hard_center", 0),
        make_failure_ranking_scene("hard_right_1", 1),
        make_failure_ranking_scene("hard_right_2", 2),
        make_failure_ranking_scene("hard_left_1", -1),
        make_failure_ranking_scene("hard_left_2", -2),
    )


def run_hard_synthetic_eval() -> list[M4HardSyntheticRow]:
    rows: list[M4HardSyntheticRow] = []
    for scene in make_hard_synthetic_scenes():
        rows.extend(_rank_scene(scene, "silhouette_only", SILHOUETTE_ONLY_WEIGHTS))
        rows.extend(_rank_scene(scene, "failure_aware", None))
    return rows


def write_hard_synthetic_reports(
    rows: list[M4HardSyntheticRow],
    output_csv: str | Path,
    output_json: str | Path,
) -> None:
    if not rows:
        raise ValueError("no hard synthetic rows to write")
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
        "schema_version": "m4_hard_synthetic_eval_v1",
        "num_rows": len(rows),
        "aggregate_by_ranker": _aggregate_by_ranker(rows),
        "rows": row_dicts,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rank_scene(scene, ranker_name: str, weights: dict[str, float] | None) -> list[M4HardSyntheticRow]:
    ranked = run_ghost_mgg_v0(
        scene.target_mask,
        scene.evidence,
        config=GhostMGGV0Config(top_k=len(scene.candidates)),
        weights=weights,
        hypotheses=scene.candidates,
    )
    top1 = ranked[0]
    ground_truth_rank = next(
        index
        for index, item in enumerate(ranked, start=1)
        if item.hypothesis.hypothesis_id == scene.ground_truth_id
    )
    return [
        M4HardSyntheticRow(
            scene_id=scene.scene_id,
            ranker=ranker_name,
            ground_truth_id=scene.ground_truth_id,
            top1_hypothesis_id=top1.hypothesis.hypothesis_id,
            top1_correct=top1.hypothesis.hypothesis_id == scene.ground_truth_id,
            ground_truth_rank=ground_truth_rank,
            top1_visual_score=top1.score.visual,
            top1_failure_score=top1.score.failure,
            top1_total_score=top1.score.total,
        )
    ]


def _aggregate_by_ranker(rows: list[M4HardSyntheticRow]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for ranker in sorted({row.ranker for row in rows}):
        selected = [row for row in rows if row.ranker == ranker]
        count = float(len(selected))
        aggregate[ranker] = {
            "num_scenes": len(selected),
            "top1_accuracy": sum(row.top1_correct for row in selected) / count,
            "mean_ground_truth_rank": sum(row.ground_truth_rank for row in selected) / count,
        }
    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = run_hard_synthetic_eval()
    write_hard_synthetic_reports(rows, args.output_csv, args.output_json)
    print(f"Wrote {len(rows)} M4 hard synthetic rows")
    print(f"CSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
