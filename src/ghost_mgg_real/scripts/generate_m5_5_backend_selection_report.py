#!/usr/bin/env python3
"""Batch-generate M5.5 ObservationQuality and backend-selection reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_m5_5_observation_quality import build_observation_quality  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scene_metadata_paths(sample_root: Path) -> list[Path]:
    return sorted(Path(sample_root).glob("*/metadata.json"))


def _evidence_path(evidence_root: Path, scene_id: str) -> Path:
    return Path(evidence_root) / scene_id / "evidence_summary.json"


def generate_backend_selection_report(
    *,
    sample_root: Path,
    evidence_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    sample_root = Path(sample_root)
    evidence_root = Path(evidence_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_rows: list[dict[str, Any]] = []
    for metadata_path in _scene_metadata_paths(sample_root):
        metadata = _read_json(metadata_path)
        scene_id = str(metadata.get("scene_id") or metadata_path.parent.name)
        evidence_summary_path = _evidence_path(evidence_root, scene_id)
        if not evidence_summary_path.exists():
            continue
        evidence_summary = _read_json(evidence_summary_path)
        quality = build_observation_quality(
            observation_id=scene_id,
            sample_metadata=metadata,
            topic_report={"overall_status": "pass"},
            target_summary=evidence_summary,
            planning_requested=False,
        )
        scene_dir = output_dir / scene_id
        _write_json(scene_dir / "observation_quality.json", quality)
        scene_rows.append(
            {
                "scene_id": scene_id,
                "target_label": evidence_summary.get("target_label"),
                "shape_hint": evidence_summary.get("shape_hint"),
                "recommended_backend": quality["recommended_backend"],
                "rgb_ok": quality["rgb_ok"],
                "ir_ok": quality["ir_ok"],
                "depth_ok": quality["depth_ok"],
                "target_valid_depth_ratio": quality["target_valid_depth_ratio"],
                "target_hole_ratio": quality["target_hole_ratio"],
                "target_table_leakage_ratio": quality["target_table_leakage_ratio"],
                "depth_failure_detected": quality["depth_failure_detected"],
                "reject_reasons": ",".join(quality.get("reject_reasons", [])),
                "depth_failure_reasons": ",".join(
                    quality.get("depth_failure_reasons", [])
                ),
                "observation_quality_path": f"{scene_id}/observation_quality.json",
            }
        )

    scene_rows.sort(key=lambda row: str(row["scene_id"]))
    report = {
        "schema_version": "m5_5_backend_selection_report_v1",
        "generated_at_utc": _utc_now(),
        "sample_root": str(sample_root),
        "evidence_root": str(evidence_root),
        "num_scenes": len(scene_rows),
        "backend_counts": _backend_counts(scene_rows),
        "scenes": scene_rows,
    }
    _write_json(output_dir / "backend_selection_report.json", report)
    _write_csv(output_dir / "backend_selection_report.csv", scene_rows)
    _write_index(output_dir / "index.md", report)
    return report


def _backend_counts(scene_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in scene_rows:
        backend = str(row["recommended_backend"])
        counts[backend] = counts.get(backend, 0) + 1
    return dict(sorted(counts.items()))


def _write_csv(path: Path, scene_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scene_id",
        "target_label",
        "shape_hint",
        "recommended_backend",
        "rgb_ok",
        "ir_ok",
        "depth_ok",
        "target_valid_depth_ratio",
        "target_hole_ratio",
        "target_table_leakage_ratio",
        "depth_failure_detected",
        "reject_reasons",
        "depth_failure_reasons",
        "observation_quality_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in scene_rows:
            writer.writerow(row)


def _write_index(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# M5.5 Backend Selection Report",
        "",
        f"- Scenes: {report['num_scenes']}",
        f"- Backend counts: {json.dumps(report['backend_counts'], sort_keys=True)}",
        "",
        "| scene | target | backend | valid | hole | leakage | reasons |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in report["scenes"]:
        lines.append(
            "| "
            f"{row['scene_id']} | "
            f"{row.get('target_label') or ''} | "
            f"{row['recommended_backend']} | "
            f"{float(row['target_valid_depth_ratio']):.3f} | "
            f"{float(row['target_hole_ratio']):.3f} | "
            f"{float(row['target_table_leakage_ratio']):.3f} | "
            f"{row.get('depth_failure_reasons') or row.get('reject_reasons') or ''} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=Path("reports/m5_real_d435_replay_samples"),
        help="Directory containing extracted replay sample metadata.",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path("reports/m5_real_d435_masked_evidence"),
        help="Directory containing masked evidence summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m5_5_backend_selection_report"),
        help="Output directory for the backend selection report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = generate_backend_selection_report(
        sample_root=args.sample_root,
        evidence_root=args.evidence_root,
        output_dir=args.output_dir,
    )
    print(
        "M5.5 backend report: "
        f"{report['num_scenes']} scenes -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
