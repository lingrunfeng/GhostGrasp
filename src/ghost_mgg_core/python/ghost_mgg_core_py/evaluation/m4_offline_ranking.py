from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ghost_mgg_core_py.evaluation.m3_dataset import M3EvidenceSample, load_m3_capture
from ghost_mgg_core_py.ghost_mgg_v0 import (
    GhostMGGV0Config,
    SILHOUETTE_ONLY_WEIGHTS,
    evidence_from_capture_arrays,
    run_ghost_mgg_v0,
)


@dataclass(frozen=True)
class M4OfflineRankingRow:
    scenario_id: str
    failure_mode: str
    ranker: str
    rank: int
    hypothesis_id: str
    shape_type: str
    center_u: float
    center_v: float
    size_u_px: float
    size_v_px: float
    visual_score: float
    failure_score: float
    depth_score: float
    total_score: float
    validation_state: str


def run_offline_ranking(capture_dir: str | Path, top_k: int = 3) -> list[M4OfflineRankingRow]:
    rows: list[M4OfflineRankingRow] = []
    for sample in load_m3_capture(capture_dir):
        rows.extend(rank_sample(sample, "silhouette_only", SILHOUETTE_ONLY_WEIGHTS, top_k))
        rows.extend(rank_sample(sample, "failure_aware", None, top_k))
    return rows


def rank_sample(
    sample: M3EvidenceSample,
    ranker_name: str,
    weights: dict[str, float] | None,
    top_k: int,
) -> list[M4OfflineRankingRow]:
    arrays = sample.load_arrays()
    target_mask = arrays["target_mask"].astype(bool)
    evidence = evidence_from_capture_arrays(arrays)
    ranked = run_ghost_mgg_v0(
        target_mask,
        evidence,
        config=GhostMGGV0Config(top_k=top_k),
        weights=weights,
    )

    rows = []
    for rank_index, item in enumerate(ranked, start=1):
        hypothesis = item.hypothesis
        score = item.score
        rows.append(
            M4OfflineRankingRow(
                scenario_id=sample.scenario_id,
                failure_mode=sample.failure_mode,
                ranker=ranker_name,
                rank=rank_index,
                hypothesis_id=hypothesis.hypothesis_id,
                shape_type=hypothesis.shape_type,
                center_u=float(hypothesis.center_uv[0]),
                center_v=float(hypothesis.center_uv[1]),
                size_u_px=float(hypothesis.size_px[0]),
                size_v_px=float(hypothesis.size_px[1]),
                visual_score=float(score.visual),
                failure_score=float(score.failure),
                depth_score=float(score.depth),
                total_score=float(score.total),
                validation_state=item.validation_state,
            )
        )
    return rows


def write_ranking_reports(
    rows: list[M4OfflineRankingRow],
    output_csv: str | Path,
    output_json: str | Path,
) -> None:
    if not rows:
        raise ValueError("no ranking rows to write")
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "m4_offline_ranking_v1",
                "num_rows": len(rows),
                "rows": row_dicts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--top-k", default=3, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = run_offline_ranking(args.capture_dir, top_k=args.top_k)
    write_ranking_reports(rows, args.output_csv, args.output_json)
    print(f"Wrote {len(rows)} M4 offline ranking rows")
    print(f"CSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
