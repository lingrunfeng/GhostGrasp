#!/usr/bin/env python3
"""Generate one M6 shadow-mode perception/backend decision.

The decision is read-only and never authorizes real robot motion. It connects
an M6 shadow observation, an optional target mask, masked evidence, conservative
ObservationQuality, and BackendSelector into one auditable report.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_m5_5_live_masked_evidence import generate_live_masked_evidence  # noqa: E402
from generate_m5_5_observation_quality import build_observation_quality  # noqa: E402
from run_m5_5_backend_selector import select_backend  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _snapshot_dir_from_observation(observation: dict[str, Any], base_dir: Path) -> Path:
    snapshot_dir = Path(str(observation.get("snapshot", {}).get("dir", "")))
    if snapshot_dir.is_absolute():
        return snapshot_dir
    cwd_path = Path.cwd() / snapshot_dir
    if cwd_path.exists():
        return cwd_path
    return base_dir / snapshot_dir


def _topic_report_from_shadow_observation(observation: dict[str, Any]) -> dict[str, Any]:
    gates = observation.get("gate_checks", {})
    passed = bool(
        gates.get("has_snapshot")
        and gates.get("has_camera_to_base_tf")
        and gates.get("has_aligned_depth_raw")
    )
    missing = []
    if not gates.get("has_snapshot"):
        missing.append("snapshot")
    if not gates.get("has_camera_to_base_tf"):
        missing.append("base_link_to_camera_link_tf")
    if not gates.get("has_aligned_depth_raw"):
        missing.append("aligned_depth_raw")
    return {
        "schema_version": "m6_shadow_synthetic_topic_report_v1",
        "overall_status": "pass" if passed else "fail",
        "missing_topics": missing,
    }


def _load_target_summary(
    *,
    observation_id: str,
    snapshot_dir: Path,
    mask_path: Path | None,
    output_dir: Path,
    target_label: str,
    shape_hint: str,
) -> dict[str, Any] | None:
    if mask_path is None:
        return None
    return generate_live_masked_evidence(
        scene_id=observation_id,
        snapshot_dir=snapshot_dir,
        mask_path=Path(mask_path),
        output_dir=output_dir / "evidence",
        target_label=target_label,
        shape_hint=shape_hint,
    )


def _is_ready_for_shadow_planning(
    *,
    observation: dict[str, Any],
    quality: dict[str, Any],
    recommended_backend: str,
) -> bool:
    gates = observation.get("gate_checks", {})
    required_gates = (
        gates.get("has_snapshot"),
        gates.get("has_real_arm_joints"),
        gates.get("has_camera_to_base_tf"),
        gates.get("has_aligned_depth_raw"),
    )
    return bool(
        all(required_gates)
        and quality.get("mask_ok")
        and recommended_backend not in {"abstain", "manual_review"}
    )


def _render_index(report: dict[str, Any]) -> str:
    quality = report["quality"]
    target = report.get("target_summary") or {}
    lines = [
        "# M6 Shadow Decision",
        "",
        f"- observation_id: `{report['observation_id']}`",
        f"- safety_mode: `{report['safety_mode']}`",
        f"- motion_authorized: `{str(report['motion_authorized']).lower()}`",
        f"- recommended_backend: `{report['recommended_backend']}`",
        f"- ready_for_shadow_planning: `{str(report['ready_for_shadow_planning']).lower()}`",
        "",
        "## Target Evidence",
        "",
    ]
    if target:
        lines.extend(
            [
                f"- target_label: `{target.get('target_label')}`",
                f"- shape_hint: `{target.get('shape_hint')}`",
                f"- target_pixels: `{target.get('target_pixels')}`",
                f"- valid_depth_ratio: `{target.get('valid_depth_ratio', 0.0):.3f}`",
                f"- hole_ratio: `{target.get('hole_ratio', 0.0):.3f}`",
                f"- table_leakage_ratio: `{target.get('table_leakage_ratio', 0.0):.3f}`",
            ]
        )
    else:
        lines.append("- no target mask yet; decision must abstain")
    lines.extend(
        [
            "",
            "## Quality",
            "",
            f"- rgb_ok: `{quality.get('rgb_ok')}`",
            f"- ir_ok: `{quality.get('ir_ok')}`",
            f"- depth_ok: `{quality.get('depth_ok')}`",
            f"- mask_ok: `{quality.get('mask_ok')}`",
            f"- tf_ok: `{quality.get('tf_ok')}`",
            f"- reject_reasons: `{', '.join(report.get('reject_reasons', [])) or 'none'}`",
            f"- depth_failure_reasons: `{', '.join(report.get('depth_failure_reasons', [])) or 'none'}`",
            f"- caution_reasons: `{', '.join(report.get('caution_reasons', [])) or 'none'}`",
            "",
            "This report is shadow-only and does not authorize real motion.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_shadow_decision(
    *,
    shadow_observation_path: Path,
    mask_path: Path | None,
    output_dir: Path,
    target_label: str = "target",
    shape_hint: str = "unknown",
) -> dict[str, Any]:
    shadow_observation_path = Path(shadow_observation_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observation = _read_json(shadow_observation_path)
    observation_id = str(observation["observation_id"])
    snapshot_dir = _snapshot_dir_from_observation(
        observation,
        base_dir=shadow_observation_path.parent,
    )
    metadata_path = snapshot_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing snapshot metadata: {metadata_path}")
    sample_metadata = _read_json(metadata_path)
    target_summary = _load_target_summary(
        observation_id=observation_id,
        snapshot_dir=snapshot_dir,
        mask_path=mask_path,
        output_dir=output_dir,
        target_label=target_label,
        shape_hint=shape_hint,
    )
    topic_report = _topic_report_from_shadow_observation(observation)
    quality = build_observation_quality(
        observation_id=observation_id,
        sample_metadata=sample_metadata,
        topic_report=topic_report,
        target_summary=target_summary,
        planning_requested=True,
    )
    selection = select_backend(quality, planning_requested=True)
    recommended_backend = str(selection["recommended_backend"])
    ready_for_shadow_planning = _is_ready_for_shadow_planning(
        observation=observation,
        quality=quality,
        recommended_backend=recommended_backend,
    )
    report = {
        "schema_version": "m6_shadow_decision_v1",
        "generated_at_utc": _utc_now(),
        "observation_id": observation_id,
        "shadow_observation_path": str(shadow_observation_path),
        "safety_mode": "shadow_only_no_motion",
        "motion_authorized": False,
        "target_label": target_label,
        "shape_hint": shape_hint,
        "mask_path": str(mask_path) if mask_path else None,
        "target_summary": target_summary,
        "quality": quality,
        "backend_selection": selection,
        "recommended_backend": recommended_backend,
        "reject_reasons": list(selection.get("reject_reasons", [])),
        "depth_failure_reasons": list(selection.get("depth_failure_reasons", [])),
        "caution_reasons": list(selection.get("caution_reasons", [])),
        "ready_for_shadow_planning": ready_for_shadow_planning,
        "next_steps": [
            "if backend is abstain, fix missing mask/table/TF before planning",
            "if ready_for_shadow_planning is true, run MoveIt plan-only only",
            "do not execute real motion in M6",
        ],
    }
    _write_json(output_dir / "observation_quality.json", quality)
    _write_json(output_dir / "backend_selection.json", selection)
    _write_json(output_dir / "m6_shadow_decision.json", report)
    (output_dir / "index.md").write_text(_render_index(report), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-observation", type=Path, required=True)
    parser.add_argument("--mask-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-label", default="target")
    parser.add_argument("--shape-hint", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = generate_shadow_decision(
        shadow_observation_path=args.shadow_observation,
        mask_path=args.mask_path,
        output_dir=args.output_dir,
        target_label=args.target_label,
        shape_hint=args.shape_hint,
    )
    print(
        "M6 shadow decision: "
        f"{report['recommended_backend']} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
