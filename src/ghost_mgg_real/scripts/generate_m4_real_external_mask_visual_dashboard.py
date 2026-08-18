#!/usr/bin/env python3
"""Generate a visual HTML dashboard for M4 real external-mask replay results."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "m4_real_external_mask_visual_dashboard_v1"
GATE_SUMMARY_SCHEMA_VERSION = "m4_real_external_mask_gate_summary_v1"
QUALITY_STATUSES = ("good", "questionable", "fail", "ood")


def _relative_path(path: Path, output_dir: Path) -> str:
    return os.path.relpath(Path(path).resolve(), start=Path(output_dir).resolve()).replace(os.sep, "/")


def build_scene_cards(
    *,
    hypotheses_json: Path,
    replay_samples_dir: Path,
    masked_evidence_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(hypotheses_json).read_text(encoding="utf-8"))
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("rows", []):
        by_scene.setdefault(str(row["scene_id"]), []).append(row)

    cards: list[dict[str, Any]] = []
    output_dir = Path(output_dir)
    for scene_id in sorted(by_scene):
        rows = sorted(by_scene[scene_id], key=lambda item: int(item.get("rank", 999)))
        top = rows[0]
        card = {
            "scene_id": scene_id,
            "target_label": top.get("target_label"),
            "shape_hint": top.get("shape_hint"),
            "top_hypothesis_id": top.get("hypothesis_id"),
            "top_shape_type": top.get("shape_type"),
            "top_total_score": float(top.get("total_score", 0.0)),
            "hole_ratio": float(top.get("hole_ratio", 0.0)),
            "table_leakage_ratio": float(top.get("table_leakage_ratio", 0.0)),
            "images": {
                "rgb": _relative_path(Path(replay_samples_dir) / scene_id / "color.png", output_dir),
                "aligned_depth": _relative_path(
                    Path(replay_samples_dir) / scene_id / "aligned_depth_viz.png",
                    output_dir,
                ),
                "formal_mask": _relative_path(
                    Path(masked_evidence_dir) / scene_id / "formal_mask.png",
                    output_dir,
                ),
                "evidence_overlay": _relative_path(
                    Path(masked_evidence_dir) / scene_id / "evidence_overlay.png",
                    output_dir,
                ),
            },
            "hypotheses": rows,
        }
        card["quality"] = assign_scene_verdict(card)
        cards.append(card)
    return cards


def assign_scene_verdict(card: dict[str, Any]) -> dict[str, Any]:
    """Assign a human-readable quality verdict for one real replay scene."""
    hypotheses = sorted(card.get("hypotheses", []), key=lambda item: int(item.get("rank", 999)))
    top = hypotheses[0] if hypotheses else {}
    second = hypotheses[1] if len(hypotheses) > 1 else {}
    top_score = float(card.get("top_total_score", top.get("total_score", 0.0)))
    second_score = float(second.get("total_score", 0.0))
    top_margin = round(top_score - second_score, 6) if second else round(top_score, 6)
    hole_ratio = float(card.get("hole_ratio", top.get("hole_ratio", 0.0)))
    leakage_ratio = float(card.get("table_leakage_ratio", top.get("table_leakage_ratio", 0.0)))
    evidence_strength = round(hole_ratio + leakage_ratio, 6)
    label = str(card.get("target_label") or top.get("target_label") or "").lower()
    shape_hint = str(card.get("shape_hint") or top.get("shape_hint") or "").lower()

    status = "good"
    reasons: list[str] = []

    if _is_ood_scene(label, shape_hint):
        status = "ood"
        reasons.append("object is outside the current box/cylinder/cup-like primitive family")
    elif top_score < 1.0:
        status = "fail"
        reasons.append(f"top score is too low ({top_score:.2f})")
    else:
        if "multi" in label:
            status = "questionable"
            reasons.append("multi-object scene may contaminate the target mask or background evidence")
        if top_score < 1.4:
            status = _max_status(status, "questionable")
            reasons.append(f"top score is weak ({top_score:.2f})")
        if top_margin < 0.15:
            status = _max_status(status, "questionable")
            reasons.append(f"top-1/top-2 margin is small ({top_margin:.2f})")
        if _needs_failure_evidence(label) and evidence_strength < 0.08:
            status = _max_status(status, "questionable")
            reasons.append("transparent/reflective scene has weak hole-or-leakage evidence")

    if not reasons:
        reasons.append("top hypothesis has a clear margin and usable evidence")

    return {
        "status": status,
        "label": status.replace("_", " ").title(),
        "reasons": reasons,
        "top_margin": top_margin,
        "evidence_strength": evidence_strength,
        "top_score": round(top_score, 6),
    }


def _is_ood_scene(label: str, shape_hint: str) -> bool:
    thin_or_irregular_terms = ("spoon", "fork", "knife", "thin", "teddy", "irregular", "handle")
    if any(term in label for term in thin_or_irregular_terms):
        return True
    return shape_hint == "unknown" and any(term in label for term in ("metal", "reflective"))


def _needs_failure_evidence(label: str) -> bool:
    return any(term in label for term in ("transparent", "glass", "jelly", "frosted", "metal", "reflective"))


def _max_status(current: str, candidate: str) -> str:
    severity = {"good": 0, "questionable": 1, "fail": 2, "ood": 3}
    return candidate if severity[candidate] > severity[current] else current


def write_visual_dashboard(
    *,
    cards: list[dict[str, Any]],
    output_dir: Path,
    source_hypotheses_json: Path,
) -> None:
    if not cards:
        raise ValueError("no scene cards to render")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for card in cards:
        card.setdefault("quality", assign_scene_verdict(card))
    gate_summary = summarize_quality(cards)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_hypotheses_json": str(source_hypotheses_json),
        "num_scenes": len(cards),
        "gate_summary": gate_summary,
        "cards": cards,
    }
    (output_dir / "dashboard.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gate_summary.json").write_text(
        json.dumps(gate_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gate_summary.md").write_text(_render_gate_summary_markdown(gate_summary), encoding="utf-8")
    (output_dir / "index.html").write_text(_render_html(cards, gate_summary), encoding="utf-8")


def summarize_quality(cards: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {status: 0 for status in QUALITY_STATUSES}
    scenes_by_status = {status: [] for status in QUALITY_STATUSES}
    for card in cards:
        quality = card.get("quality") or assign_scene_verdict(card)
        status = str(quality.get("status", "questionable"))
        if status not in counts:
            status = "questionable"
        counts[status] += 1
        scenes_by_status[status].append(card.get("scene_id", ""))
    return {
        "schema_version": GATE_SUMMARY_SCHEMA_VERSION,
        "num_scenes": len(cards),
        "status_counts": counts,
        "scenes_by_status": scenes_by_status,
        "gate_ready_count": counts["good"] + counts["questionable"],
        "needs_review_count": counts["fail"] + counts["ood"],
    }


def _render_gate_summary_markdown(summary: dict[str, Any]) -> str:
    counts = summary["status_counts"]
    lines = [
        "# M4 Real External-Mask Gate Summary",
        "",
        f"- scenes: {summary['num_scenes']}",
        f"- good: {counts['good']}",
        f"- questionable: {counts['questionable']}",
        f"- fail: {counts['fail']}",
        f"- ood: {counts['ood']}",
        f"- gate ready: {summary['gate_ready_count']}",
        f"- needs review: {summary['needs_review_count']}",
        "",
        "## Scenes By Status",
        "",
    ]
    for status in QUALITY_STATUSES:
        scenes = ", ".join(str(scene) for scene in summary["scenes_by_status"][status]) or "none"
        lines.append(f"- {status}: {scenes}")
    return "\n".join(lines) + "\n"


def _render_html(cards: list[dict[str, Any]], gate_summary: dict[str, Any]) -> str:
    rows = "\n".join(_render_card(card) for card in cards)
    counts = gate_summary["status_counts"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>M4 Real External-Mask Visual Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101216;
      --panel: #191d24;
      --panel-2: #202631;
      --text: #edf2f7;
      --muted: #9aa7b7;
      --line: #323a48;
      --green: #48d597;
      --blue: #6ab5ff;
      --orange: #f2b36b;
      --red: #ff6b6b;
      --purple: #c89cff;
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
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      font-weight: 720;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      max-width: 980px;
      line-height: 1.45;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .filter {{
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 12px;
      background: #0f141d;
      color: var(--text);
      cursor: pointer;
      font: inherit;
      font-size: 13px;
    }}
    .filter:hover {{
      border-color: var(--blue);
    }}
    main {{
      padding: 24px 32px 40px;
      display: grid;
      gap: 22px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .card-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 18px;
      font-weight: 680;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      background: #151923;
    }}
    .verdict {{
      margin-top: 12px;
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 13px;
    }}
    .status {{
      display: inline-flex;
      width: max-content;
      border-radius: 999px;
      padding: 5px 10px;
      font-weight: 760;
      color: #07100b;
      background: var(--green);
    }}
    .status-good {{ background: var(--green); }}
    .status-questionable {{ background: var(--orange); }}
    .status-fail {{ background: var(--red); }}
    .status-ood {{ background: var(--purple); }}
    .reasons {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.35;
    }}
    .top {{
      text-align: right;
      min-width: 190px;
      font-size: 13px;
      color: var(--muted);
    }}
    .top strong {{
      color: var(--green);
      font-size: 16px;
    }}
    .content {{
      padding: 18px 20px 20px;
      display: grid;
      gap: 18px;
    }}
    .images {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 12px;
    }}
    figure {{
      margin: 0;
      background: #0e1117;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }}
    figure img {{
      width: 100%;
      height: 180px;
      object-fit: contain;
      display: block;
      background: #06080c;
    }}
    figcaption {{
      padding: 8px 10px;
      font-size: 12px;
      color: var(--muted);
      border-top: 1px solid var(--line);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 620;
      background: #151923;
    }}
    td.num, th.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .rank1 td {{
      color: #f8fff9;
    }}
    .rank {{
      color: var(--orange);
      font-weight: 700;
    }}
    @media (max-width: 980px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .images {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }}
      .card-header {{ flex-direction: column; }}
      .top {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>M4 Real External-Mask Visual Dashboard</h1>
    <p class="subtitle">Visual acceptance board for the external mask contract: real D435 RGB/depth replay, completed formal masks, evidence overlays, and failure-aware top hypotheses. No Gazebo target truth is used.</p>
    <div class="summary" aria-label="Quality filters">
      <button class="filter" data-filter="all" onclick="filterByQuality('all')">All {gate_summary['num_scenes']}</button>
      <button class="filter" data-filter="good" onclick="filterByQuality('good')">Good {counts['good']}</button>
      <button class="filter" data-filter="questionable" onclick="filterByQuality('questionable')">Questionable {counts['questionable']}</button>
      <button class="filter" data-filter="fail" onclick="filterByQuality('fail')">Fail {counts['fail']}</button>
      <button class="filter" data-filter="ood" onclick="filterByQuality('ood')">OOD {counts['ood']}</button>
    </div>
  </header>
  <main>
    {rows}
  </main>
  <script>
    function filterByQuality(status) {{
      document.querySelectorAll('.card').forEach((card) => {{
        card.style.display = status === 'all' || card.dataset.quality === status ? '' : 'none';
      }});
    }}
  </script>
</body>
</html>
"""


def _render_card(card: dict[str, Any]) -> str:
    images = card["images"]
    quality = card.get("quality") or assign_scene_verdict(card)
    status = _esc(quality.get("status", "questionable"))
    reasons = "\n".join(f"<li>{_esc(reason)}</li>" for reason in quality.get("reasons", []))
    hypotheses = "\n".join(_render_hypothesis_row(row) for row in card["hypotheses"])
    return f"""
<section class="card" data-quality="{status}">
  <div class="card-header">
    <div>
      <h2>{_esc(card['scene_id'])}</h2>
      <div class="meta">
        <span class="pill">label: {_esc(card.get('target_label'))}</span>
        <span class="pill">shape hint: {_esc(card.get('shape_hint'))}</span>
        <span class="pill">hole: {float(card.get('hole_ratio', 0.0)):.3f}</span>
        <span class="pill">leakage: {float(card.get('table_leakage_ratio', 0.0)):.3f}</span>
      </div>
      <div class="verdict">
        <span class="status status-{status}">Quality Verdict: {_esc(quality.get('label'))}</span>
        <span>margin {float(quality.get('top_margin', 0.0)):.3f} · evidence {float(quality.get('evidence_strength', 0.0)):.3f}</span>
        <ul class="reasons">{reasons}</ul>
      </div>
    </div>
    <div class="top">
      top hypothesis<br>
      <strong>{_esc(card.get('top_hypothesis_id'))}</strong><br>
      {_esc(card.get('top_shape_type'))} · total {float(card.get('top_total_score', 0.0)):.3f}
    </div>
  </div>
  <div class="content">
    <div class="images">
      {_figure('RGB', images['rgb'])}
      {_figure('Aligned Depth', images['aligned_depth'])}
      {_figure('Formal Mask', images['formal_mask'])}
      {_figure('Evidence Overlay', images['evidence_overlay'])}
    </div>
    <div>
      <table>
        <thead>
          <tr>
            <th>Top-3 Hypotheses</th>
            <th>shape</th>
            <th class="num">total</th>
            <th class="num">failure</th>
            <th class="num">visual</th>
            <th class="num">inside hole</th>
            <th class="num">inside leak</th>
          </tr>
        </thead>
        <tbody>
          {hypotheses}
        </tbody>
      </table>
    </div>
  </div>
</section>
"""


def _render_hypothesis_row(row: dict[str, Any]) -> str:
    rank = int(row.get("rank", 0))
    failure_terms = row.get("failure_terms", {})
    inside_hole = float(row.get("failure_inside_hole", failure_terms.get("inside_hole", 0.0)))
    inside_leak = float(
        row.get("failure_inside_table_leakage", failure_terms.get("inside_table_leakage", 0.0))
    )
    row_class = " class=\"rank1\"" if rank == 1 else ""
    return f"""
<tr{row_class}>
  <td><span class="rank">R{rank}</span> {_esc(row.get('hypothesis_id'))}</td>
  <td>{_esc(row.get('shape_type'))}</td>
  <td class="num">{float(row.get('total_score', 0.0)):.3f}</td>
  <td class="num">{float(row.get('failure_score', 0.0)):.3f}</td>
  <td class="num">{float(row.get('visual_score', 0.0)):.3f}</td>
  <td class="num">{inside_hole:.3f}</td>
  <td class="num">{inside_leak:.3f}</td>
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
        "--hypotheses-json",
        type=Path,
        default=Path("reports/m4_real_external_mask_replay/m4_real_live_hypotheses.json"),
    )
    parser.add_argument(
        "--replay-samples-dir",
        type=Path,
        default=Path("reports/m5_real_d435_replay_samples"),
    )
    parser.add_argument(
        "--masked-evidence-dir",
        type=Path,
        default=Path("reports/m5_real_d435_masked_evidence"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m4_real_external_mask_visual_dashboard"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cards = build_scene_cards(
        hypotheses_json=args.hypotheses_json,
        replay_samples_dir=args.replay_samples_dir,
        masked_evidence_dir=args.masked_evidence_dir,
        output_dir=args.output_dir,
    )
    write_visual_dashboard(
        cards=cards,
        output_dir=args.output_dir,
        source_hypotheses_json=args.hypotheses_json,
    )
    print(f"Wrote M4 real external-mask visual dashboard to {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
