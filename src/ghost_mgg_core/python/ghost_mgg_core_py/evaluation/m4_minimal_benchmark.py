from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ghost_mgg_core_py.evaluation.m3_dataset import M3EvidenceSample, load_m3_capture


@dataclass(frozen=True)
class M4MinimalBenchmarkRow:
    scenario_id: str
    failure_mode: str
    total_pixels: int
    roi_pixels: int
    valid_depth_ratio: float
    hole_ratio: float
    table_leakage_ratio: float
    edge_ratio: float
    flying_point_ratio: float
    biased_depth_ratio: float
    silhouette_only_score: float
    mask_extrusion_score: float
    failure_aware_simple_score: float
    predicted_failure_family: str


def _predicted_failure_family(sample: M3EvidenceSample) -> str:
    ratios = {
        "hole": sample.ratio("hole_ratio"),
        "table_leakage": sample.ratio("table_leakage_ratio"),
        "edge_flying": sample.ratio("edge_ratio") + sample.ratio("flying_point_ratio"),
        "biased_depth": sample.ratio("biased_depth_ratio"),
    }
    best_name, best_value = max(ratios.items(), key=lambda item: item[1])
    if best_value <= 1e-6:
        return "none"
    return best_name


def score_sample(sample: M3EvidenceSample) -> M4MinimalBenchmarkRow:
    hole = sample.ratio("hole_ratio")
    leak = sample.ratio("table_leakage_ratio")
    edge = sample.ratio("edge_ratio")
    fly = sample.ratio("flying_point_ratio")
    biased = sample.ratio("biased_depth_ratio")
    valid = sample.ratio("valid_depth_ratio")

    failure_evidence = 0.45 * hole + 0.65 * leak + 0.35 * edge + 0.35 * fly + 0.30 * biased

    return M4MinimalBenchmarkRow(
        scenario_id=sample.scenario_id,
        failure_mode=sample.failure_mode,
        total_pixels=sample.count("total_pixels"),
        roi_pixels=sample.count("roi_pixels"),
        valid_depth_ratio=valid,
        hole_ratio=hole,
        table_leakage_ratio=leak,
        edge_ratio=edge,
        flying_point_ratio=fly,
        biased_depth_ratio=biased,
        silhouette_only_score=1.0,
        mask_extrusion_score=max(0.0, valid - 0.70 * hole - 0.90 * leak - 0.50 * fly),
        failure_aware_simple_score=min(1.0, failure_evidence),
        predicted_failure_family=_predicted_failure_family(sample),
    )


def run_benchmark(capture_dir: str | Path) -> list[M4MinimalBenchmarkRow]:
    return [score_sample(sample) for sample in load_m3_capture(capture_dir)]


def write_reports(
    rows: list[M4MinimalBenchmarkRow],
    output_csv: str | Path,
    output_json: str | Path,
) -> None:
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    row_dicts = [asdict(row) for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)

    summary = {
        "schema_version": "m4_minimal_benchmark_v1",
        "num_samples": len(rows),
        "rows": row_dicts,
        "notes": [
            "This is a minimal evidence benchmark over M3 captured summaries.",
            "It is not the final GHOST-MGG geometry ranking benchmark.",
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = run_benchmark(args.capture_dir)
    write_reports(rows, args.output_csv, args.output_json)
    print(f"Wrote {len(rows)} M4 minimal benchmark rows")
    print(f"CSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
