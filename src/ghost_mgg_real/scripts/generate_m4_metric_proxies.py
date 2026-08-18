#!/usr/bin/env python3
"""Convert M5 image-space ranking rows into table-anchored metric proxies."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricProxyRow:
    scene_id: str
    target_label: str | None
    shape_hint: str | None
    ranker: str
    rank: int
    hypothesis_id: str
    shape_type: str
    center_x_m: float
    center_y_m: float
    center_z_m: float
    width_m: float
    depth_m: float
    height_m: float
    table_z_m: float
    table_depth_m: float
    table_depth_source: str
    visual_score: float
    failure_score: float
    total_score: float
    source_center_u: float
    source_center_v: float
    source_size_u_px: float
    source_size_v_px: float


def pixel_proxy_to_metric(
    row: dict[str, Any],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    table_depth_m: float,
    table_z_m: float,
    primitive_height_m: float,
    table_depth_source: str = "constant",
) -> MetricProxyRow:
    z = _positive_float(table_depth_m, "table_depth_m")
    height = _positive_float(primitive_height_m, "primitive_height_m")
    resolved_fx = _positive_float(fx, "fx")
    resolved_fy = _positive_float(fy, "fy")
    center_u = _float(row["center_u"])
    center_v = _float(row["center_v"])
    size_u_px = _positive_float(row["size_u_px"], "size_u_px")
    size_v_px = _positive_float(row["size_v_px"], "size_v_px")

    return MetricProxyRow(
        scene_id=str(row["scene_id"]),
        target_label=row.get("target_label"),
        shape_hint=row.get("shape_hint"),
        ranker=str(row["ranker"]),
        rank=int(row["rank"]),
        hypothesis_id=str(row["hypothesis_id"]),
        shape_type=str(row["shape_type"]),
        center_x_m=round(((center_u - float(cx)) / resolved_fx) * z, 6),
        center_y_m=round(((center_v - float(cy)) / resolved_fy) * z, 6),
        center_z_m=round(float(table_z_m) + 0.5 * height, 6),
        width_m=round(size_u_px / resolved_fx * z, 6),
        depth_m=round(size_v_px / resolved_fy * z, 6),
        height_m=round(height, 6),
        table_z_m=round(float(table_z_m), 6),
        table_depth_m=round(z, 6),
        table_depth_source=table_depth_source,
        visual_score=_float(row.get("visual_score", 0.0)),
        failure_score=_float(row.get("failure_score", 0.0)),
        total_score=_float(row.get("total_score", 0.0)),
        source_center_u=center_u,
        source_center_v=center_v,
        source_size_u_px=size_u_px,
        source_size_v_px=size_v_px,
    )


def generate_metric_proxy_report(
    *,
    ranking_json: Path,
    output_dir: Path,
    fx: float = 554.0,
    fy: float = 554.0,
    cx: float = 320.0,
    cy: float = 240.0,
    table_depth_m: float = 1.0,
    frame_stats_csv: Path | None = Path("reports/m5_real_d435_frame_stats.csv"),
    table_z_m: float = 0.75,
    primitive_height_m: float = 0.04,
    rank: int = 1,
) -> list[MetricProxyRow]:
    payload = json.loads(Path(ranking_json).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m5_real_ranking_v1":
        raise ValueError(f"unsupported ranking schema: {payload.get('schema_version')}")
    scene_depths = load_aligned_depth_means(frame_stats_csv)
    rows = [
        pixel_proxy_to_metric(
            row,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            table_depth_m=scene_depths.get(str(row["scene_id"]), table_depth_m),
            table_depth_source=(
                "frame_stats_aligned_depth_mean"
                if str(row["scene_id"]) in scene_depths
                else "constant"
            ),
            table_z_m=table_z_m,
            primitive_height_m=primitive_height_m,
        )
        for row in payload.get("rows", [])
        if int(row.get("rank", -1)) == int(rank)
    ]
    if not rows:
        raise ValueError(f"no rank-{rank} rows in {ranking_json}")
    write_metric_proxy_report(rows, output_dir)
    return rows


def load_aligned_depth_means(frame_stats_csv: Path | None) -> dict[str, float]:
    if frame_stats_csv is None or not Path(frame_stats_csv).exists():
        return {}
    scene_depths: dict[str, float] = {}
    with Path(frame_stats_csv).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row.get("topic") != "/camera/camera/aligned_depth_to_color/image_raw":
                continue
            scene_id = str(row.get("scene_id", ""))
            if not scene_id:
                continue
            mean_depth = _positive_float(row.get("mean_valid_depth_m", 0.0), "mean_valid_depth_m")
            scene_depths[scene_id] = mean_depth
    return scene_depths


def write_metric_proxy_report(rows: list[MetricProxyRow], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    payload = {
        "schema_version": "m4_metric_proxies_v1",
        "num_rows": len(row_dicts),
        "num_scenes": len({row.scene_id for row in rows}),
        "rows": row_dicts,
    }
    (output_dir / "metric_proxies.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "metric_proxies.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_dicts[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(row_dicts)
    _write_index(rows, output_dir / "index.md")


def _write_index(rows: list[MetricProxyRow], output_path: Path) -> None:
    lines = [
        "# M4 Metric Proxies",
        "",
        "Table-anchored metric proxy export from M5 image-space ranking rows.",
        "",
        f"- rows: {len(rows)}",
        f"- scenes: {len({row.scene_id for row in rows})}",
        "",
        "| scene_id | ranker | hypothesis | shape | center xyz m | size xyz m | scale_source | total |",
        "|---|---|---|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.scene_id} | "
            f"{row.ranker} | "
            f"{row.hypothesis_id} | "
            f"{row.shape_type} | "
            f"({row.center_x_m:.3f}, {row.center_y_m:.3f}, {row.center_z_m:.3f}) | "
            f"({row.width_m:.3f}, {row.depth_m:.3f}, {row.height_m:.3f}) | "
            f"{row.table_depth_source} | "
            f"{row.total_score:.3f} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _float(value: Any) -> float:
    return float(value)


def _positive_float(value: Any, name: str) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranking-json",
        type=Path,
        default=Path("reports/m5_real_d435_ranking/m5_real_ranking.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m4_metric_proxies"))
    parser.add_argument("--fx", type=float, default=554.0)
    parser.add_argument("--fy", type=float, default=554.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--table-depth-m", type=float, default=1.0)
    parser.add_argument(
        "--frame-stats-csv",
        type=Path,
        default=Path("reports/m5_real_d435_frame_stats.csv"),
        help="Optional per-scene D435 frame statistics CSV used for aligned-depth metric scale.",
    )
    parser.add_argument("--table-z-m", type=float, default=0.75)
    parser.add_argument("--primitive-height-m", type=float, default=0.04)
    parser.add_argument("--rank", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = generate_metric_proxy_report(
        ranking_json=args.ranking_json,
        output_dir=args.output_dir,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        table_depth_m=args.table_depth_m,
        frame_stats_csv=args.frame_stats_csv,
        table_z_m=args.table_z_m,
        primitive_height_m=args.primitive_height_m,
        rank=args.rank,
    )
    print(f"Wrote {len(rows)} M4 metric proxy rows to {args.output_dir}")


if __name__ == "__main__":
    main()
