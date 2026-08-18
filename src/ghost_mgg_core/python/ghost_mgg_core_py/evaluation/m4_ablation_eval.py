from __future__ import annotations

import argparse
from pathlib import Path

from ghost_mgg_core_py.evaluation.m3_dataset import M3EvidenceSample, load_m3_capture
from ghost_mgg_core_py.evaluation.m4_geometry_eval import (
    evaluate_ranking_rows,
    write_geometry_eval_reports,
)
from ghost_mgg_core_py.evaluation.m4_offline_ranking import (
    M4OfflineRankingRow,
    write_ranking_reports,
)
from ghost_mgg_core_py.ghost_mgg_v0 import (
    GHOST_MGG_V0_ABLATIONS,
    GhostMGGV0Config,
    evidence_from_capture_arrays,
    run_ghost_mgg_v0,
)


ABLATION_CONFIGS = GHOST_MGG_V0_ABLATIONS


def run_ablation_ranking(capture_dir: str | Path, top_k: int = 3) -> list[M4OfflineRankingRow]:
    rows: list[M4OfflineRankingRow] = []
    for sample in load_m3_capture(capture_dir):
        for ablation_name, config in ABLATION_CONFIGS.items():
            rows.extend(
                rank_ablation_sample(
                    sample,
                    ranker_name=ablation_name,
                    weights=config["weights"],
                    zero_channels=tuple(config["zero_channels"]),
                    top_k=top_k,
                )
            )
    return rows


def rank_ablation_sample(
    sample: M3EvidenceSample,
    ranker_name: str,
    weights: dict[str, float] | None,
    zero_channels: tuple[str, ...],
    top_k: int,
) -> list[M4OfflineRankingRow]:
    arrays = sample.load_arrays()
    target_mask = arrays["target_mask"].astype(bool)
    evidence = evidence_from_capture_arrays(arrays, zero_channels)
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


def run_ablation_eval(capture_dir: str | Path, top_k: int = 3):
    ranking_rows = run_ablation_ranking(capture_dir, top_k=top_k)
    eval_rows = evaluate_ranking_rows(capture_dir, ranking_rows, top_k=top_k)
    return ranking_rows, eval_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--ranking-csv", required=True, type=Path)
    parser.add_argument("--ranking-json", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--eval-json", required=True, type=Path)
    parser.add_argument("--top-k", default=3, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ranking_rows, eval_rows = run_ablation_eval(args.capture_dir, top_k=args.top_k)
    write_ranking_reports(ranking_rows, args.ranking_csv, args.ranking_json)
    write_geometry_eval_reports(eval_rows, args.eval_csv, args.eval_json)
    print(f"Wrote {len(ranking_rows)} M4 ablation ranking rows")
    print(f"Wrote {len(eval_rows)} M4 ablation evaluation rows")


if __name__ == "__main__":
    main()
