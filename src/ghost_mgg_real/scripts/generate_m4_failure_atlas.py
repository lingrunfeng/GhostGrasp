#!/usr/bin/env python3
"""Generate a visual failure-evidence atlas from real D435 M4 reports."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "m4_failure_atlas_v1"
CATEGORY_ORDER = (
    "hole_dominant",
    "table_leakage",
    "mixed_hole_leakage",
    "table_leakage_ablation_effect",
    "foreground_supported_control",
    "weak_failure",
    "ood_or_primitive_mismatch",
)


def build_failure_atlas(*, dashboard_json: Path, ablation_gate_json: Path) -> dict[str, Any]:
    dashboard = _read_json(dashboard_json)
    ablation = _read_json(ablation_gate_json)
    ablation_effects = _ablation_effects_by_scene(ablation)

    scenes: list[dict[str, Any]] = []
    for card in dashboard.get("cards", []):
        top = _top_hypothesis(card)
        scene_id = str(card.get("scene_id", ""))
        hole_ratio = float(card.get("hole_ratio", 0.0))
        leakage_ratio = float(card.get("table_leakage_ratio", 0.0))
        foreground_ratio = float(top.get("foreground_ratio", 0.0))
        valid_depth_ratio = float(top.get("valid_depth_ratio", 0.0))
        quality = card.get("quality") or {}
        effects = ablation_effects.get(scene_id, {})
        categories = _assign_categories(
            label=str(card.get("target_label", "")),
            quality_status=str(quality.get("status", "")),
            hole_ratio=hole_ratio,
            leakage_ratio=leakage_ratio,
            foreground_ratio=foreground_ratio,
            ablation_effects=effects,
        )
        scenes.append(
            {
                "scene_id": scene_id,
                "target_label": card.get("target_label"),
                "shape_hint": card.get("shape_hint"),
                "quality_status": quality.get("status", "unknown"),
                "quality_reasons": list(quality.get("reasons", [])),
                "categories": categories,
                "hole_ratio": hole_ratio,
                "table_leakage_ratio": leakage_ratio,
                "evidence_strength": round(hole_ratio + leakage_ratio, 6),
                "foreground_ratio": foreground_ratio,
                "valid_depth_ratio": valid_depth_ratio,
                "top_hypothesis_id": card.get("top_hypothesis_id"),
                "top_shape_type": card.get("top_shape_type"),
                "top_total_score": float(card.get("top_total_score", 0.0)),
                "ablation_effects": effects,
                "images": dict(card.get("images", {})),
            }
        )

    representatives = _representatives(scenes)
    category_counts = {category: 0 for category in CATEGORY_ORDER}
    for scene in scenes:
        for category in scene["categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "num_scenes": len(scenes),
        "category_order": list(CATEGORY_ORDER),
        "category_counts": category_counts,
        "representatives": representatives,
        "scenes": scenes,
    }


def write_failure_atlas(atlas: dict[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "failure_atlas.json").write_text(
        json.dumps(atlas, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.md").write_text(_render_markdown(atlas), encoding="utf-8")
    (output_dir / "index.html").write_text(_render_html(atlas), encoding="utf-8")


def _assign_categories(
    *,
    label: str,
    quality_status: str,
    hole_ratio: float,
    leakage_ratio: float,
    foreground_ratio: float,
    ablation_effects: dict[str, Any],
) -> list[str]:
    categories: list[str] = []
    lower_label = label.lower()
    if quality_status == "ood":
        categories.append("ood_or_primitive_mismatch")
    if "opaque" in lower_label and foreground_ratio >= 0.10:
        categories.append("foreground_supported_control")
    if hole_ratio >= 0.25 and hole_ratio >= max(0.01, leakage_ratio * 2.0):
        categories.append("hole_dominant")
    if leakage_ratio >= 0.08:
        categories.append("table_leakage")
    if hole_ratio >= 0.25 and leakage_ratio >= 0.08:
        categories.append("mixed_hole_leakage")
    leakage_effect = ablation_effects.get("without_table_leakage", {})
    if leakage_effect.get("top1_changed_vs_full") is True:
        categories.append("table_leakage_ablation_effect")
    if hole_ratio + leakage_ratio < 0.12 and quality_status != "ood":
        categories.append("weak_failure")
    if not categories:
        categories.append("weak_failure")
    return [category for category in CATEGORY_ORDER if category in set(categories)]


def _representatives(scenes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    reps: dict[str, dict[str, Any]] = {}
    for category in CATEGORY_ORDER:
        candidates = [scene for scene in scenes if category in scene["categories"]]
        if not candidates:
            continue
        selected = max(candidates, key=lambda scene: _representative_score(category, scene))
        reps[category] = {
            "scene_id": selected["scene_id"],
            "target_label": selected["target_label"],
            "quality_status": selected["quality_status"],
            "hole_ratio": selected["hole_ratio"],
            "table_leakage_ratio": selected["table_leakage_ratio"],
            "foreground_ratio": selected["foreground_ratio"],
            "reason": _category_reason(category),
            "images": selected["images"],
        }
    return reps


def _representative_score(category: str, scene: dict[str, Any]) -> float:
    if category == "hole_dominant":
        return float(scene["hole_ratio"])
    if category in {"table_leakage", "mixed_hole_leakage"}:
        return float(scene["table_leakage_ratio"]) + float(scene["hole_ratio"]) * 0.25
    if category == "table_leakage_ablation_effect":
        return float(
            scene.get("ablation_effects", {})
            .get("without_table_leakage", {})
            .get("full_total_advantage", 0.0)
        )
    if category == "foreground_supported_control":
        return float(scene["foreground_ratio"])
    if category == "weak_failure":
        return -float(scene["evidence_strength"])
    if category == "ood_or_primitive_mismatch":
        return float(scene["evidence_strength"])
    return 0.0


def _category_reason(category: str) -> str:
    return {
        "hole_dominant": "Depth returns are mostly missing inside the target mask.",
        "table_leakage": "Depth inside the target mask often matches the empty-table background.",
        "mixed_hole_leakage": "The same target contains both missing depth and table leakage.",
        "table_leakage_ablation_effect": "Removing table-leakage evidence changes the top hypothesis.",
        "foreground_supported_control": "Foreground support is present, useful as an opaque/control case.",
        "weak_failure": "The target has weak explicit failure evidence under the current thresholds.",
        "ood_or_primitive_mismatch": "The object or mask is outside the current primitive-family claim.",
    }.get(category, category)


def _ablation_effects_by_scene(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    effects: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows", []):
        if int(row.get("rank", 999)) != 1:
            continue
        scene_id = str(row.get("scene_id", ""))
        ranker = str(row.get("ranker", ""))
        if not scene_id or not ranker or ranker == "full":
            continue
        effects.setdefault(scene_id, {})[ranker] = {
            "top1_changed_vs_full": bool(row.get("top1_changed_vs_full", False)),
            "shape_changed_vs_full": bool(row.get("shape_changed_vs_full", False)),
            "full_total_advantage": float(row.get("full_total_advantage", 0.0)),
            "full_failure_advantage": float(row.get("full_failure_advantage", 0.0)),
            "hypothesis_id": row.get("hypothesis_id"),
        }
    return effects


def _top_hypothesis(card: dict[str, Any]) -> dict[str, Any]:
    hypotheses = sorted(card.get("hypotheses", []), key=lambda row: int(row.get("rank", 999)))
    return hypotheses[0] if hypotheses else {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _render_markdown(atlas: dict[str, Any]) -> str:
    lines = [
        "# M4 Failure Evidence Atlas",
        "",
        f"- scenes: {atlas['num_scenes']}",
        "",
        "## Category Counts",
        "",
        "| category | count |",
        "|---|---:|",
    ]
    for category in atlas["category_order"]:
        lines.append(f"| {category} | {atlas['category_counts'].get(category, 0)} |")
    lines.extend(["", "## Representative Scenes", "", "| category | scene_id | reason |", "|---|---|---|"])
    for category in atlas["category_order"]:
        rep = atlas["representatives"].get(category)
        if rep is None:
            continue
        lines.append(f"| {category} | {rep['scene_id']} | {rep['reason']} |")
    lines.extend(
        [
            "",
            "## All Scenes",
            "",
            "| scene_id | quality | categories | hole | leakage | foreground |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for scene in atlas["scenes"]:
        lines.append(
            "| "
            f"{scene['scene_id']} | "
            f"{scene['quality_status']} | "
            f"{', '.join(scene['categories'])} | "
            f"{scene['hole_ratio']:.3f} | "
            f"{scene['table_leakage_ratio']:.3f} | "
            f"{scene['foreground_ratio']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_html(atlas: dict[str, Any]) -> str:
    representative_cards = "\n".join(
        _render_representative_card(category, atlas["representatives"][category])
        for category in atlas["category_order"]
        if category in atlas["representatives"]
    )
    scene_rows = "\n".join(_render_scene_row(scene) for scene in atlas["scenes"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>M4 Failure Evidence Atlas</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101216;
      --panel: #191d24;
      --panel-2: #202631;
      --line: #323a48;
      --text: #edf2f7;
      --muted: #a8b3c3;
      --green: #48d597;
      --orange: #f2b36b;
      --blue: #6ab5ff;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: #141820;
    }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    .subtitle {{ color: var(--muted); max-width: 980px; line-height: 1.45; }}
    main {{ padding: 24px 32px 42px; display: grid; gap: 24px; }}
    h2 {{ margin: 0 0 12px; font-size: 19px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .card-body {{ padding: 16px; display: grid; gap: 12px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; color: var(--muted); font-size: 13px; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 4px 9px; background: #151923; }}
    .category {{ color: #07100b; background: var(--green); border: 0; font-weight: 760; }}
    .images {{ display: grid; grid-template-columns: repeat(4, minmax(80px, 1fr)); gap: 8px; }}
    figure {{ margin: 0; border: 1px solid var(--line); background: #080b10; border-radius: 6px; overflow: hidden; }}
    figure img {{ width: 100%; height: 120px; object-fit: contain; display: block; }}
    figcaption {{ padding: 6px 8px; font-size: 11px; color: var(--muted); border-top: 1px solid var(--line); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); background: #151923; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    @media (max-width: 900px) {{ header, main {{ padding-left: 16px; padding-right: 16px; }} .images {{ grid-template-columns: repeat(2, minmax(110px, 1fr)); }} }}
  </style>
</head>
<body>
  <header>
    <h1>M4 Failure Evidence Atlas</h1>
    <p class="subtitle">Representative real D435 scenes grouped by failure evidence. Inputs are real replay images, formal external masks, dashboard quality labels, and ablation effects. No Gazebo target truth is used.</p>
  </header>
  <main>
    <section>
      <h2>Representative Scenes</h2>
      <div class="grid">{representative_cards}</div>
    </section>
    <section>
      <h2>All Scenes</h2>
      <table>
        <thead><tr><th>scene</th><th>quality</th><th>categories</th><th class="num">hole</th><th class="num">leakage</th><th class="num">foreground</th></tr></thead>
        <tbody>{scene_rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def _render_representative_card(category: str, rep: dict[str, Any]) -> str:
    images = rep.get("images", {})
    return f"""
<article class="card">
  <div class="card-body">
    <div class="meta"><span class="pill category">{_esc(category)}</span><span class="pill">{_esc(rep.get('quality_status'))}</span></div>
    <strong>{_esc(rep.get('scene_id'))}</strong>
    <div class="meta">
      <span class="pill">hole {float(rep.get('hole_ratio', 0.0)):.3f}</span>
      <span class="pill">leak {float(rep.get('table_leakage_ratio', 0.0)):.3f}</span>
      <span class="pill">fg {float(rep.get('foreground_ratio', 0.0)):.3f}</span>
    </div>
    <p class="subtitle">{_esc(rep.get('reason'))}</p>
    <div class="images">
      {_figure('RGB', images.get('rgb', ''))}
      {_figure('Depth', images.get('aligned_depth', ''))}
      {_figure('Mask', images.get('formal_mask', ''))}
      {_figure('Evidence', images.get('evidence_overlay', ''))}
    </div>
  </div>
</article>
"""


def _render_scene_row(scene: dict[str, Any]) -> str:
    return f"""
<tr>
  <td>{_esc(scene['scene_id'])}</td>
  <td>{_esc(scene['quality_status'])}</td>
  <td>{_esc(', '.join(scene['categories']))}</td>
  <td class="num">{float(scene['hole_ratio']):.3f}</td>
  <td class="num">{float(scene['table_leakage_ratio']):.3f}</td>
  <td class="num">{float(scene['foreground_ratio']):.3f}</td>
</tr>
"""


def _figure(label: str, src: str) -> str:
    return f"""<figure><img src="{_esc(src)}" alt="{_esc(label)}"><figcaption>{_esc(label)}</figcaption></figure>"""


def _esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dashboard-json",
        type=Path,
        default=Path("reports/m4_real_external_mask_visual_dashboard/dashboard.json"),
    )
    parser.add_argument(
        "--ablation-gate-json",
        type=Path,
        default=Path("reports/m4_real_ablation_gate/ablation_gate.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m4_failure_atlas"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    atlas = build_failure_atlas(
        dashboard_json=args.dashboard_json,
        ablation_gate_json=args.ablation_gate_json,
    )
    write_failure_atlas(atlas, args.output_dir)
    print(f"Wrote M4 failure atlas to {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
