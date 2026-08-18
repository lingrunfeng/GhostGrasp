#!/usr/bin/env python3
"""Freeze the M6 shadow-mode evidence into one auditable report.

The report is offline/read-only. It validates that saved real D435 + real
joint-state shadow observations include both the normal-depth and GHOST-MGG
routes, that MoveIt only planned, and that no report authorized real motion.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_expected_backend(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected backend must be observation_id=backend: {value}")
        scene_id, backend = value.split("=", 1)
        scene_id = scene_id.strip()
        backend = backend.strip()
        if not scene_id or not backend:
            raise ValueError(f"empty expected backend field: {value}")
        result[scene_id] = backend
    return result


def _scene_report(
    *,
    observation_id: str,
    expected_backend: str | None,
    observations_root: Path,
    decisions_root: Path,
    targets_root: Path,
) -> dict[str, Any]:
    observation_path = observations_root / observation_id / "m6_shadow_observation.json"
    decision_path = decisions_root / observation_id / "m6_shadow_decision.json"
    target_path = targets_root / observation_id / "m6_shadow_grasp_target.json"
    moveit_path = targets_root / observation_id / "moveit_plan_only_shadow_allowlist.json"

    observation = _read_json(observation_path)
    decision = _read_json(decision_path)
    target = _read_json(target_path)
    moveit = _read_json(moveit_path)

    gates = observation.get("gate_checks", {})
    backend = str(decision.get("recommended_backend", ""))
    moveit_summary = moveit.get("summary", {})
    target_payload = target.get("target", {})
    motion_authorized = bool(decision.get("motion_authorized")) or bool(
        target.get("motion_authorized")
    )
    checks = {
        "observation_gates_pass": all(
            bool(gates.get(name))
            for name in (
                "has_snapshot",
                "has_real_arm_joints",
                "has_camera_to_base_tf",
                "has_aligned_depth_raw",
            )
        ),
        "backend_matches_expected": expected_backend is None or backend == expected_backend,
        "ready_for_shadow_planning": bool(decision.get("ready_for_shadow_planning")),
        "target_valid": bool(target_payload.get("valid")),
        "moveit_all_planned": bool(moveit_summary.get("all_planned")),
        "motion_not_authorized": not motion_authorized,
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "observation_id": observation_id,
        "status": status,
        "expected_backend": expected_backend,
        "recommended_backend": backend,
        "depth_failure_reasons": list(decision.get("depth_failure_reasons", [])),
        "reject_reasons": list(decision.get("reject_reasons", [])),
        "ready_for_shadow_planning": bool(decision.get("ready_for_shadow_planning")),
        "moveit_all_planned": bool(moveit_summary.get("all_planned")),
        "motion_authorized": motion_authorized,
        "checks": checks,
        "evidence": {
            "observation": str(observation_path),
            "decision": str(decision_path),
            "target": str(target_path),
            "moveit_plan_only": str(moveit_path),
        },
    }


def generate_m6_shadow_freeze_report(
    *,
    output_dir: Path,
    scene_ids: list[str],
    expected_backends: dict[str, str],
    observations_root: Path = Path("reports/m6_shadow_observations"),
    decisions_root: Path = Path("reports/m6_shadow_decisions"),
    targets_root: Path = Path("reports/m6_shadow_grasp_targets"),
    readiness_path: Path = Path("reports/m6_shadow_readiness/m6_shadow_readiness.json"),
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    scenes = [
        _scene_report(
            observation_id=scene_id,
            expected_backend=expected_backends.get(scene_id),
            observations_root=Path(observations_root),
            decisions_root=Path(decisions_root),
            targets_root=Path(targets_root),
        )
        for scene_id in scene_ids
    ]
    readiness = _read_json(readiness_path)
    backend_counts = dict(sorted(Counter(scene["recommended_backend"] for scene in scenes).items()))
    motion_authorized = any(bool(scene["motion_authorized"]) for scene in scenes)
    has_required_routes = (
        int(backend_counts.get("normal_rgbd", 0)) >= 1
        and int(backend_counts.get("ghost_mgg", 0)) >= 1
    )
    checks = {
        "readiness_pass": readiness.get("overall_status") == "pass",
        "has_normal_and_ghost_routes": has_required_routes,
        "all_scenes_pass": all(scene["status"] == "pass" for scene in scenes),
        "motion_not_authorized": not motion_authorized,
    }
    overall_status = "pass" if all(checks.values()) else "fail"
    report = {
        "schema_version": "m6_shadow_freeze_v1",
        "generated_at_utc": _utc_now(),
        "overall_status": overall_status,
        "motion_authorized": False,
        "scene_count": len(scenes),
        "backend_counts": backend_counts,
        "checks": checks,
        "readiness_path": str(readiness_path),
        "scenes": scenes,
        "next_gate": "M7.1 low-amplitude empty-motion safety preflight",
    }
    _write_json(output_dir / "m6_shadow_freeze.json", report)
    _write_index(output_dir / "index.md", report)
    return report


def _write_index(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# M6 Shadow Freeze",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- motion_authorized: `{str(report['motion_authorized']).lower()}`",
        f"- scene_count: `{report['scene_count']}`",
        f"- backend_counts: `{report['backend_counts']}`",
        "",
        "## Gates",
        "",
        "| check | status |",
        "|---|---|",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"| {name} | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| observation | expected | selected | plan-only | motion | status |",
            "|---|---|---|---|---|---|",
        ]
    )
    for scene in report["scenes"]:
        lines.append(
            "| {observation_id} | {expected_backend} | {recommended_backend} | {plan} | {motion} | {status} |".format(
                observation_id=scene["observation_id"],
                expected_backend=scene.get("expected_backend") or "",
                recommended_backend=scene["recommended_backend"],
                plan="pass" if scene["moveit_all_planned"] else "fail",
                motion="authorized" if scene["motion_authorized"] else "not_authorized",
                status=scene["status"],
            )
        )
    lines.extend(
        [
            "",
            "This freeze report is evidence for M6 shadow mode only. It does not authorize real robot motion.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m6_shadow_freeze"))
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--expected-backend", action="append", default=[])
    parser.add_argument(
        "--observations-root",
        type=Path,
        default=Path("reports/m6_shadow_observations"),
    )
    parser.add_argument(
        "--decisions-root",
        type=Path,
        default=Path("reports/m6_shadow_decisions"),
    )
    parser.add_argument(
        "--targets-root",
        type=Path,
        default=Path("reports/m6_shadow_grasp_targets"),
    )
    parser.add_argument(
        "--readiness-path",
        type=Path,
        default=Path("reports/m6_shadow_readiness/m6_shadow_readiness.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = generate_m6_shadow_freeze_report(
        output_dir=args.output_dir,
        scene_ids=list(args.scene_id),
        expected_backends=_parse_expected_backend(list(args.expected_backend)),
        observations_root=args.observations_root,
        decisions_root=args.decisions_root,
        targets_root=args.targets_root,
        readiness_path=args.readiness_path,
    )
    print(f"M6 shadow freeze: {report['overall_status']} -> {args.output_dir}")


if __name__ == "__main__":
    main()
