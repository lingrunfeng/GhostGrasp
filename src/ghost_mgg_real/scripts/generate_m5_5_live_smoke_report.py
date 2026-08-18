#!/usr/bin/env python3
"""Summarize M5.5 live D435 smoke observations and backend decisions."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scene_row(bridge_root: Path, scene_id: str) -> dict[str, Any]:
    evidence_path = bridge_root / "live_masked_evidence" / scene_id / "evidence_summary.json"
    selection_path = bridge_root / "snapshots" / scene_id / "backend_selection.json"
    quality_path = bridge_root / "snapshots" / scene_id / "observation_quality.json"

    evidence = _read_json(evidence_path)
    selection = _read_json(selection_path)
    quality = _read_json(quality_path) if quality_path.exists() else {}

    return {
        "scene_id": scene_id,
        "target_label": evidence.get("target_label"),
        "shape_hint": evidence.get("shape_hint"),
        "recommended_backend": selection.get("recommended_backend"),
        "target_pixels": evidence.get("target_pixels"),
        "valid_depth_ratio": evidence.get("valid_depth_ratio"),
        "hole_ratio": evidence.get("hole_ratio"),
        "table_leakage_ratio": evidence.get("table_leakage_ratio"),
        "foreground_ratio": evidence.get("foreground_ratio"),
        "rgb_ok": quality.get("rgb_ok"),
        "ir_ok": quality.get("ir_ok"),
        "depth_ok": quality.get("depth_ok"),
        "depth_failure_detected": quality.get("depth_failure_detected"),
        "reject_reasons": ",".join(selection.get("reject_reasons", [])),
        "depth_failure_reasons": ",".join(selection.get("depth_failure_reasons", [])),
        "evidence_summary_path": str(evidence_path),
        "backend_selection_path": str(selection_path),
        "observation_quality_path": str(quality_path) if quality_path.exists() else None,
    }


def _backend_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        backend = str(row.get("recommended_backend"))
        counts[backend] = counts.get(backend, 0) + 1
    return dict(sorted(counts.items()))


def generate_live_smoke_report(
    *,
    bridge_root: Path,
    scene_ids: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    bridge_root = Path(bridge_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [_scene_row(bridge_root, scene_id) for scene_id in scene_ids]
    report = {
        "schema_version": "m5_5_live_smoke_report_v1",
        "generated_at_utc": _utc_now(),
        "bridge_root": str(bridge_root),
        "num_scenes": len(rows),
        "backend_counts": _backend_counts(rows),
        "scenes": rows,
    }
    _write_json(output_dir / "live_smoke_report.json", report)
    _write_csv(output_dir / "live_smoke_report.csv", rows)
    _write_index(output_dir / "index.md", report)
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scene_id",
        "target_label",
        "shape_hint",
        "recommended_backend",
        "target_pixels",
        "valid_depth_ratio",
        "hole_ratio",
        "table_leakage_ratio",
        "foreground_ratio",
        "rgb_ok",
        "ir_ok",
        "depth_ok",
        "depth_failure_detected",
        "reject_reasons",
        "depth_failure_reasons",
        "evidence_summary_path",
        "backend_selection_path",
        "observation_quality_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_ratio(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _write_index(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# M5.5 Live Smoke Report",
        "",
        f"- Scenes: {report['num_scenes']}",
        f"- Backend counts: {json.dumps(report['backend_counts'], sort_keys=True)}",
        "",
        "| scene | target | backend | valid | hole | leakage | reasons |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in report["scenes"]:
        reasons = row.get("depth_failure_reasons") or row.get("reject_reasons") or ""
        lines.append(
            "| "
            f"{row['scene_id']} | "
            f"{row.get('target_label') or ''} | "
            f"{row.get('recommended_backend') or ''} | "
            f"{_fmt_ratio(row.get('valid_depth_ratio'))} | "
            f"{_fmt_ratio(row.get('hole_ratio'))} | "
            f"{_fmt_ratio(row.get('table_leakage_ratio'))} | "
            f"{reasons} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge-root",
        type=Path,
        default=Path("reports/m5_5_real_online_bridge"),
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        dest="scene_ids",
        required=True,
        help="Scene id to include. Repeat for multiple scenes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m5_5_live_smoke_report"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = generate_live_smoke_report(
        bridge_root=args.bridge_root,
        scene_ids=args.scene_ids,
        output_dir=args.output_dir,
    )
    print(
        "M5.5 live smoke report: "
        f"{report['num_scenes']} scenes -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
