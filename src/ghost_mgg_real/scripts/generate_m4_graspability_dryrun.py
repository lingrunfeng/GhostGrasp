#!/usr/bin/env python3
"""Generate a cheap top-grasp graspability dry-run from M4 metric proxies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GraspabilityRow:
    scene_id: str
    ranker: str
    hypothesis_id: str
    shape_type: str
    grasp_id: str
    grasp_type: str
    grasp_x_m: float
    grasp_y_m: float
    grasp_z_m: float
    pregrasp_z_m: float
    approach_x: float
    approach_y: float
    approach_z: float
    grasp_width_axis: str
    grasp_width_base_m: float
    source_center_u: float
    source_center_v: float
    table_depth_m: float
    required_gripper_width_m: float
    gripper_width_margin_m: float
    nearest_neighbor_clearance_m: float
    workspace_radius_m: float
    table_clearance_m: float
    score: float
    valid: bool
    failure_reason: str


def score_top_grasp(
    proxy: dict[str, Any],
    *,
    neighbors: list[dict[str, Any]],
    max_gripper_width_m: float = 0.070,
    workspace_radius_m: float = 0.350,
    min_neighbor_clearance_m: float = 0.030,
    gripper_padding_m: float = 0.012,
    pregrasp_lift_m: float = 0.080,
) -> GraspabilityRow:
    width = _positive_float(proxy["width_m"], "width_m")
    depth = _positive_float(proxy["depth_m"], "depth_m")
    height = _positive_float(proxy["height_m"], "height_m")
    center_x = _float(proxy["center_x_m"])
    center_y = _float(proxy["center_y_m"])
    center_z = _float(proxy["center_z_m"])
    table_z = _float(proxy["table_z_m"])
    grasp_width_axis = "x" if width <= depth else "y"
    grasp_width_base = min(width, depth)
    required_width = round(grasp_width_base + float(gripper_padding_m), 6)
    gripper_margin = round(float(max_gripper_width_m) - required_width, 6)
    radius = math.hypot(center_x, center_y)
    table_clearance = round((center_z - 0.5 * height) - table_z, 6)
    neighbor_clearance = round(_nearest_neighbor_clearance(proxy, neighbors), 6)

    failure_reason = ""
    if gripper_margin < 0.0:
        failure_reason = "gripper_width"
    elif radius > workspace_radius_m:
        failure_reason = "workspace_radius"
    elif table_clearance < -1e-6:
        failure_reason = "table_penetration"
    elif neighbor_clearance < min_neighbor_clearance_m:
        failure_reason = "neighbor_clearance"

    valid = failure_reason == ""
    clearance_score = min(1.0, max(0.0, neighbor_clearance / max(min_neighbor_clearance_m, 1e-6)))
    width_score = min(1.0, max(0.0, gripper_margin / max(max_gripper_width_m, 1e-6)))
    workspace_score = min(1.0, max(0.0, 1.0 - radius / max(workspace_radius_m, 1e-6)))
    base_score = 0.50 + 0.25 * clearance_score + 0.15 * width_score + 0.10 * workspace_score
    score = round(base_score if valid else min(base_score, 0.30), 6)

    return GraspabilityRow(
        scene_id=str(proxy["scene_id"]),
        ranker=str(proxy["ranker"]),
        hypothesis_id=str(proxy["hypothesis_id"]),
        shape_type=str(proxy["shape_type"]),
        grasp_id=f"{proxy['hypothesis_id']}_top_dryrun",
        grasp_type="top",
        grasp_x_m=round(center_x, 6),
        grasp_y_m=round(center_y, 6),
        grasp_z_m=round(center_z + 0.5 * height + 0.005, 6),
        pregrasp_z_m=round(center_z + 0.5 * height + 0.005 + pregrasp_lift_m, 6),
        approach_x=0.0,
        approach_y=0.0,
        approach_z=-1.0,
        grasp_width_axis=grasp_width_axis,
        grasp_width_base_m=round(grasp_width_base, 6),
        source_center_u=round(_float(proxy.get("source_center_u", 0.0)), 6),
        source_center_v=round(_float(proxy.get("source_center_v", 0.0)), 6),
        table_depth_m=round(_float(proxy.get("table_depth_m", 0.0)), 6),
        required_gripper_width_m=required_width,
        gripper_width_margin_m=gripper_margin,
        nearest_neighbor_clearance_m=neighbor_clearance,
        workspace_radius_m=round(radius, 6),
        table_clearance_m=table_clearance,
        score=score,
        valid=valid,
        failure_reason=failure_reason,
    )


def generate_graspability_report(
    *,
    metric_proxies_json: Path,
    output_dir: Path,
    max_gripper_width_m: float = 0.070,
    workspace_radius_m: float = 0.350,
    min_neighbor_clearance_m: float = 0.030,
) -> list[GraspabilityRow]:
    payload = json.loads(Path(metric_proxies_json).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m4_metric_proxies_v1":
        raise ValueError(f"unsupported metric proxy schema: {payload.get('schema_version')}")
    proxies = list(payload.get("rows", []))
    rows = []
    for proxy in proxies:
        neighbors = [
            candidate
            for candidate in proxies
            if candidate.get("scene_id") == proxy.get("scene_id")
            and candidate.get("target_label") != proxy.get("target_label")
        ]
        rows.append(
            score_top_grasp(
                proxy,
                neighbors=neighbors,
                max_gripper_width_m=max_gripper_width_m,
                workspace_radius_m=workspace_radius_m,
                min_neighbor_clearance_m=min_neighbor_clearance_m,
            )
        )
    if not rows:
        raise ValueError(f"no metric proxy rows in {metric_proxies_json}")
    write_graspability_report(rows, output_dir)
    return rows


def write_graspability_report(rows: list[GraspabilityRow], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    payload = {
        "schema_version": "m4_graspability_dryrun_v1",
        "num_rows": len(row_dicts),
        "num_scenes": len({row.scene_id for row in rows}),
        "valid_count": sum(1 for row in rows if row.valid),
        "rows": row_dicts,
    }
    (output_dir / "graspability.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "graspability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_dicts[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(row_dicts)
    _write_index(rows, output_dir / "index.md")


def _nearest_neighbor_clearance(proxy: dict[str, Any], neighbors: list[dict[str, Any]]) -> float:
    if not neighbors:
        return 999.0
    px = _float(proxy["center_x_m"])
    py = _float(proxy["center_y_m"])
    pr = 0.5 * max(_positive_float(proxy["width_m"], "width_m"), _positive_float(proxy["depth_m"], "depth_m"))
    clearances = []
    for neighbor in neighbors:
        nx = _float(neighbor["center_x_m"])
        ny = _float(neighbor["center_y_m"])
        nr = 0.5 * max(
            _positive_float(neighbor["width_m"], "neighbor.width_m"),
            _positive_float(neighbor["depth_m"], "neighbor.depth_m"),
        )
        clearances.append(math.hypot(px - nx, py - ny) - pr - nr)
    return float(min(clearances))


def _write_index(rows: list[GraspabilityRow], output_path: Path) -> None:
    lines = [
        "# M4 Graspability Dry-Run",
        "",
        "Cheap headless top-grasp readiness check over M4 metric proxies.",
        "",
        f"- rows: {len(rows)}",
        f"- valid: {sum(1 for row in rows if row.valid)}/{len(rows)}",
        "",
        "| scene_id | hypothesis | valid | score | grasp_axis | required_width | width_margin | neighbor_clearance | reason |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.scene_id} | "
            f"{row.hypothesis_id} | "
            f"{int(row.valid)} | "
            f"{row.score:.3f} | "
            f"{row.grasp_width_axis} | "
            f"{row.required_gripper_width_m:.3f} | "
            f"{row.gripper_width_margin_m:.3f} | "
            f"{row.nearest_neighbor_clearance_m:.3f} | "
            f"{row.failure_reason} |"
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
        "--metric-proxies-json",
        type=Path,
        default=Path("reports/m4_metric_proxies/metric_proxies.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m4_graspability_dryrun"),
    )
    parser.add_argument("--max-gripper-width-m", type=float, default=0.070)
    parser.add_argument("--workspace-radius-m", type=float, default=0.350)
    parser.add_argument("--min-neighbor-clearance-m", type=float, default=0.030)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = generate_graspability_report(
        metric_proxies_json=args.metric_proxies_json,
        output_dir=args.output_dir,
        max_gripper_width_m=args.max_gripper_width_m,
        workspace_radius_m=args.workspace_radius_m,
        min_neighbor_clearance_m=args.min_neighbor_clearance_m,
    )
    valid_count = sum(1 for row in rows if row.valid)
    print(f"Wrote M4 graspability dry-run rows to {args.output_dir}: {valid_count}/{len(rows)} valid")


if __name__ == "__main__":
    main()
