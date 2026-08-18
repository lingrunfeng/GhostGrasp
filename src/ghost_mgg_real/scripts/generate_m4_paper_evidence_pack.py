#!/usr/bin/env python3
"""Generate a concise paper-evidence pack from M4 real replay reports."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "m4_paper_evidence_pack_v1"


def build_paper_evidence_pack(
    *,
    algorithm_gate_json: Path,
    ablation_gate_json: Path,
    failure_atlas_json: Path,
) -> dict[str, Any]:
    algorithm = _read_json(algorithm_gate_json)
    ablation = _read_json(ablation_gate_json)
    atlas = _read_json(failure_atlas_json)
    key_metrics = _key_metrics(algorithm, ablation, atlas)
    supported_claims = _supported_claims(key_metrics, algorithm, ablation, atlas)
    unsupported_claims = _unsupported_claims()
    next_steps = _next_steps()
    readiness = (
        "m4_real_replay_evidence_ready"
        if algorithm.get("overall_status") == "pass"
        and ablation.get("overall_status") == "pass"
        and key_metrics["failure_atlas_scenes"] > 0
        else "m4_needs_review"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "overall_readiness": readiness,
        "source_reports": {
            "algorithm_gate_json": str(algorithm_gate_json),
            "ablation_gate_json": str(ablation_gate_json),
            "failure_atlas_json": str(failure_atlas_json),
        },
        "key_metrics": key_metrics,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "next_steps": next_steps,
    }


def write_paper_evidence_pack(pack: dict[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence_pack.json").write_text(
        json.dumps(pack, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.md").write_text(_render_markdown(pack), encoding="utf-8")
    (output_dir / "index.html").write_text(_render_html(pack), encoding="utf-8")


def _key_metrics(algorithm: dict[str, Any], ablation: dict[str, Any], atlas: dict[str, Any]) -> dict[str, Any]:
    ablation_summary = ablation.get("ablation_summary", {})
    without_failure = ablation_summary.get("without_failure", {})
    without_table_leakage = ablation_summary.get("without_table_leakage", {})
    without_weak_depth = ablation_summary.get("without_weak_depth", {})
    categories = atlas.get("category_counts", {})
    return {
        "real_replay_scenes": int(algorithm.get("num_scenes", 0)),
        "evaluable_scenes": int(algorithm.get("num_evaluable_scenes", 0)),
        "excluded_scenes": int(algorithm.get("num_excluded_scenes", 0)),
        "algorithm_gate_status": algorithm.get("overall_status"),
        "failure_aware_top1_changed": int(algorithm.get("evaluable_top1_changed_count", 0)),
        "failure_aware_shape_changed": int(algorithm.get("evaluable_shape_changed_count", 0)),
        "mean_failure_score_delta": float(algorithm.get("mean_evaluable_failure_score_delta", 0.0)),
        "ablation_gate_status": ablation.get("overall_status"),
        "without_failure_top1_changed": int(without_failure.get("top1_changed_count", 0)),
        "without_table_leakage_top1_changed": int(
            without_table_leakage.get("top1_changed_count", 0)
        ),
        "without_weak_depth_top1_changed": int(without_weak_depth.get("top1_changed_count", 0)),
        "failure_atlas_scenes": int(atlas.get("num_scenes", 0)),
        "hole_dominant_count": int(categories.get("hole_dominant", 0)),
        "table_leakage_count": int(categories.get("table_leakage", 0)),
        "mixed_hole_leakage_count": int(categories.get("mixed_hole_leakage", 0)),
        "table_leakage_ablation_effect_count": int(
            categories.get("table_leakage_ablation_effect", 0)
        ),
        "ood_count": int(categories.get("ood_or_primitive_mismatch", 0)),
    }


def _supported_claims(
    metrics: dict[str, Any],
    algorithm: dict[str, Any],
    ablation: dict[str, Any],
    atlas: dict[str, Any],
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    if metrics["hole_dominant_count"] > 0 or metrics["table_leakage_count"] > 0:
        claims.append(
            {
                "claim_id": "real_d435_failure_evidence_is_observable",
                "status": "supported",
                "evidence": (
                    f"failure atlas: hole_dominant={metrics['hole_dominant_count']}, "
                    f"table_leakage={metrics['table_leakage_count']}, "
                    f"mixed={metrics['mixed_hole_leakage_count']}"
                ),
                "source": "reports/m4_failure_atlas",
            }
        )
    if algorithm.get("overall_status") == "pass" and metrics["failure_aware_top1_changed"] > 0:
        claims.append(
            {
                "claim_id": "failure_evidence_changes_real_replay_ranking",
                "status": "supported",
                "evidence": (
                    f"top-1 changed in {metrics['failure_aware_top1_changed']}/"
                    f"{metrics['evaluable_scenes']} evaluable scenes; "
                    f"mean failure-score delta={metrics['mean_failure_score_delta']:.3f}"
                ),
                "source": "reports/m4_real_algorithm_gate",
            }
        )
    if ablation.get("overall_status") == "pass":
        claims.append(
            {
                "claim_id": "failure_ablation_is_necessary_on_real_replay",
                "status": "supported",
                "evidence": (
                    f"without_failure top-1 changed {metrics['without_failure_top1_changed']}/"
                    f"{metrics['evaluable_scenes']} evaluable scenes"
                ),
                "source": "reports/m4_real_ablation_gate",
            }
        )
    if metrics["without_table_leakage_top1_changed"] > 0:
        claims.append(
            {
                "claim_id": "table_leakage_has_measurable_ablation_effect",
                "status": "supported",
                "evidence": (
                    f"without_table_leakage top-1 changed "
                    f"{metrics['without_table_leakage_top1_changed']}/"
                    f"{metrics['evaluable_scenes']} evaluable scenes"
                ),
                "source": "reports/m4_real_ablation_gate",
            }
        )
    if metrics["ood_count"] > 0:
        claims.append(
            {
                "claim_id": "ood_cases_are_separated_from_algorithm_claims",
                "status": "supported",
                "evidence": f"failure atlas marks {metrics['ood_count']} OOD/primitive-mismatch scene(s)",
                "source": "reports/m4_failure_atlas",
            }
        )
    return claims


def _unsupported_claims() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "real_robot_lift_and_hold_success",
            "status": "unsupported",
            "missing_evidence": "Requires M7 real robot execution with lift-and-hold outcome logs.",
        },
        {
            "claim_id": "statistically_significant_grasp_improvement",
            "status": "unsupported",
            "missing_evidence": "Requires paired real grasp trials, confidence intervals, and significance tests.",
        },
        {
            "claim_id": "joint_graspability_improves_real_geometry_choice",
            "status": "unsupported",
            "missing_evidence": "Current real replay evidence is geometry ranking only; robot graspability is not yet evaluated on real hardware.",
        },
        {
            "claim_id": "calibrated_confidence_and_abstention",
            "status": "unsupported",
            "missing_evidence": "Requires calibration split, confidence metrics, and abstention precision/recall.",
        },
        {
            "claim_id": "generic_arbitrary_object_reconstruction",
            "status": "out_of_scope",
            "missing_evidence": "The method is a primitive proxy ranker, not a full arbitrary-shape reconstruction method.",
        },
    ]


def _next_steps() -> list[dict[str, str]]:
    return [
        {
            "stage": "M5/M6",
            "task": "Add weak metric/pose ground truth for selected real replay scenes.",
            "reason": "Ranking changes are visible, but geometric accuracy still needs external validation.",
        },
        {
            "stage": "M6",
            "task": "Run hardware shadow mode with real D435, TF, MoveIt planning, and no motion.",
            "reason": "Verify camera-to-base timing and planning safety before robot execution.",
        },
        {
            "stage": "M7",
            "task": "Run constrained real grasp trials with lift-and-hold logging.",
            "reason": "Main RA-L endpoint is real lift-and-hold success, not offline replay ranking.",
        },
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _render_markdown(pack: dict[str, Any]) -> str:
    metrics = pack["key_metrics"]
    lines = [
        "# M4 Paper Evidence Pack",
        "",
        f"- overall_readiness: {pack['overall_readiness']}",
        f"- real replay scenes: {metrics['real_replay_scenes']}",
        f"- evaluable scenes: {metrics['evaluable_scenes']}",
        f"- excluded scenes: {metrics['excluded_scenes']}",
        "",
        "## Supported Claims",
        "",
        "| claim_id | evidence | source |",
        "|---|---|---|",
    ]
    for claim in pack["supported_claims"]:
        lines.append(f"| {claim['claim_id']} | {claim['evidence']} | {claim['source']} |")
    lines.extend(["", "## Unsupported Claims", "", "| claim_id | missing evidence |", "|---|---|"])
    for claim in pack["unsupported_claims"]:
        lines.append(f"| {claim['claim_id']} | {claim['missing_evidence']} |")
    lines.extend(["", "## Next Steps", "", "| stage | task | reason |", "|---|---|---|"])
    for step in pack["next_steps"]:
        lines.append(f"| {step['stage']} | {step['task']} | {step['reason']} |")
    lines.append("")
    return "\n".join(lines)


def _render_html(pack: dict[str, Any]) -> str:
    supported = "\n".join(_claim_item(claim) for claim in pack["supported_claims"])
    unsupported = "\n".join(_claim_item(claim, unsupported=True) for claim in pack["unsupported_claims"])
    next_steps = "\n".join(
        f"<li><strong>{_esc(step['stage'])}</strong>: {_esc(step['task'])}<br><span>{_esc(step['reason'])}</span></li>"
        for step in pack["next_steps"]
    )
    metrics = pack["key_metrics"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>M4 Paper Evidence Pack</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101216; --panel: #191d24; --line: #323a48; --text: #edf2f7; --muted: #a8b3c3; --green: #48d597; --orange: #f2b36b; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }}
    header {{ padding: 28px 32px 18px; border-bottom: 1px solid var(--line); background: #141820; }}
    main {{ padding: 24px 32px 42px; display: grid; gap: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 0 0 10px; font-size: 19px; }}
    .subtitle, span {{ color: var(--muted); line-height: 1.45; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; background: #151923; font-size: 13px; }}
    li {{ margin: 0 0 10px; }}
    .supported strong {{ color: var(--green); }}
    .unsupported strong {{ color: var(--orange); }}
  </style>
</head>
<body>
  <header>
    <h1>M4 Paper Evidence Pack</h1>
    <p class="subtitle">Claim ledger for current GHOST-MGG M4 real replay evidence. It separates supported claims from claims that still need robot or statistical evidence.</p>
  </header>
  <main>
    <section class="panel">
      <h2>Key Metrics</h2>
      <div class="metrics">
        <span class="pill">readiness: {_esc(pack['overall_readiness'])}</span>
        <span class="pill">evaluable: {metrics['evaluable_scenes']}</span>
        <span class="pill">top1 changed: {metrics['failure_aware_top1_changed']}</span>
        <span class="pill">without leakage changed: {metrics['without_table_leakage_top1_changed']}</span>
        <span class="pill">hole scenes: {metrics['hole_dominant_count']}</span>
        <span class="pill">leakage scenes: {metrics['table_leakage_count']}</span>
      </div>
    </section>
    <section class="panel supported"><h2>Supported Claims</h2><ul>{supported}</ul></section>
    <section class="panel unsupported"><h2>Unsupported Claims</h2><ul>{unsupported}</ul></section>
    <section class="panel"><h2>Next Steps</h2><ul>{next_steps}</ul></section>
  </main>
</body>
</html>
"""


def _claim_item(claim: dict[str, Any], unsupported: bool = False) -> str:
    detail_key = "missing_evidence" if unsupported else "evidence"
    return f"<li><strong>{_esc(claim['claim_id'])}</strong><br><span>{_esc(claim.get(detail_key, ''))}</span></li>"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm-gate-json",
        type=Path,
        default=Path("reports/m4_real_algorithm_gate/algorithm_gate.json"),
    )
    parser.add_argument(
        "--ablation-gate-json",
        type=Path,
        default=Path("reports/m4_real_ablation_gate/ablation_gate.json"),
    )
    parser.add_argument(
        "--failure-atlas-json",
        type=Path,
        default=Path("reports/m4_failure_atlas/failure_atlas.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m4_paper_evidence_pack"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pack = build_paper_evidence_pack(
        algorithm_gate_json=args.algorithm_gate_json,
        ablation_gate_json=args.ablation_gate_json,
        failure_atlas_json=args.failure_atlas_json,
    )
    write_paper_evidence_pack(pack, args.output_dir)
    print(f"Wrote M4 paper evidence pack to {args.output_dir / 'index.md'}")


if __name__ == "__main__":
    main()
