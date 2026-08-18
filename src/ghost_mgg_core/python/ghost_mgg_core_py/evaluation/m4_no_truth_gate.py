from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ghost_mgg_core_py.evaluation.m4_hard_synthetic_eval import make_hard_synthetic_scenes
from ghost_mgg_core_py.ghost_mgg_v0 import (
    GHOST_MGG_V0_ABLATIONS,
    SILHOUETTE_ONLY_WEIGHTS,
    GhostMGGV0Config,
    run_ghost_mgg_v0,
)


@dataclass(frozen=True)
class NoTruthSyntheticRow:
    scene_id: str
    ranker: str
    ground_truth_id: str
    top1_hypothesis_id: str
    top1_correct: bool
    ground_truth_rank: int


FORBIDDEN_TRUTH_TOKENS = (
    "gz model",
    "gazebo_msgs",
    "material_id",
    "model_pose",
    "ground_truth_pose",
    "/world/",
)


def run_synthetic_no_truth_rows() -> list[NoTruthSyntheticRow]:
    rows: list[NoTruthSyntheticRow] = []
    rankers: tuple[tuple[str, dict[str, float] | None], ...] = (
        ("silhouette_only", SILHOUETTE_ONLY_WEIGHTS),
        ("full", None),
        ("without_failure", GHOST_MGG_V0_ABLATIONS["without_failure"]["weights"]),
    )
    for scene in make_hard_synthetic_scenes():
        for ranker_name, weights in rankers:
            ranked = run_ghost_mgg_v0(
                scene.target_mask,
                scene.evidence,
                config=GhostMGGV0Config(top_k=len(scene.candidates)),
                weights=weights,
                hypotheses=scene.candidates,
            )
            top1 = ranked[0]
            ground_truth_rank = next(
                index
                for index, item in enumerate(ranked, start=1)
                if item.hypothesis.hypothesis_id == scene.ground_truth_id
            )
            rows.append(
                NoTruthSyntheticRow(
                    scene_id=scene.scene_id,
                    ranker=ranker_name,
                    ground_truth_id=scene.ground_truth_id,
                    top1_hypothesis_id=top1.hypothesis.hypothesis_id,
                    top1_correct=top1.hypothesis.hypothesis_id == scene.ground_truth_id,
                    ground_truth_rank=ground_truth_rank,
                )
            )
    return rows


def load_real_dashboard_rows(real_dashboard_json: str | Path | None) -> list[dict[str, Any]]:
    if real_dashboard_json is None:
        return []
    path = Path(real_dashboard_json)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m4_real_dashboard_v1":
        raise ValueError(f"unsupported real dashboard schema: {payload.get('schema_version')}")
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("real dashboard rows must be a list")
    return [dict(row) for row in rows]


def load_multi_target_payload(multi_target_json: str | Path | None) -> dict[str, Any]:
    if multi_target_json is None:
        return {"summary": {}, "rows": []}
    path = Path(multi_target_json)
    if not path.exists():
        return {"summary": {}, "rows": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if summary.get("schema_version") != "m4_no_truth_multi_target_eval_v1":
        raise ValueError(
            f"unsupported multi-target schema: {summary.get('schema_version')}"
        )
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("multi-target rows must be a list")
    return {"summary": dict(summary), "rows": [dict(row) for row in rows]}


def load_scenario_sweep_payload(scenario_sweep_json: str | Path | None) -> dict[str, Any]:
    if scenario_sweep_json is None:
        return {"summary": {}, "rows": []}
    path = Path(scenario_sweep_json)
    if not path.exists():
        return {"summary": {}, "rows": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if summary.get("schema_version") != "m4_no_truth_scenario_sweep_v1":
        raise ValueError(
            f"unsupported scenario-sweep schema: {summary.get('schema_version')}"
        )
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("scenario-sweep rows must be a list")
    return {"summary": dict(summary), "rows": [dict(row) for row in rows]}


def load_dynamic_execute_payload(dynamic_execute_json: str | Path | None) -> dict[str, Any]:
    if dynamic_execute_json is None:
        return {}
    path = Path(dynamic_execute_json)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m4_joint_hypothesis_execute_v1":
        raise ValueError(
            f"unsupported dynamic-execute schema: {payload.get('schema_version')}"
        )
    return dict(payload)


def run_no_truth_audit(repo_root: str | Path | None = None) -> tuple[bool, list[str]]:
    root = Path(repo_root) if repo_root is not None else _repo_root()
    audited_paths = (
        root / "src" / "ghost_mgg_sim" / "scripts" / "m4_live_hypothesis_publisher_node.py",
        root / "src" / "ghost_mgg_core" / "python" / "ghost_mgg_core_py" / "evidence" / "depth_failure.py",
        root / "src" / "ghost_mgg_core" / "python" / "ghost_mgg_core_py" / "hypotheses" / "hypothesis_generator.py",
        root / "src" / "ghost_mgg_core" / "python" / "ghost_mgg_core_py" / "scoring" / "score_terms.py",
    )
    findings: list[str] = []
    for path in audited_paths:
        if not path.exists():
            findings.append(f"missing:{path.relative_to(root)}")
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_TRUTH_TOKENS:
            if token in text:
                findings.append(f"{path.relative_to(root)} contains forbidden token {token!r}")
    return not findings, findings


def summarize_no_truth_gate(
    *,
    synthetic_rows: list[NoTruthSyntheticRow],
    real_rows: list[dict[str, Any]],
    multi_target_payload: dict[str, Any] | None = None,
    scenario_sweep_payload: dict[str, Any] | None = None,
    dynamic_execute_payload: dict[str, Any] | None = None,
    ranked_fallback_payload: dict[str, Any] | None = None,
    no_truth_audit_pass: bool,
    audit_findings: list[str] | None = None,
) -> dict[str, Any]:
    full_acc = _top1_accuracy(synthetic_rows, "full")
    silhouette_acc = _top1_accuracy(synthetic_rows, "silhouette_only")
    without_failure_acc = _top1_accuracy(synthetic_rows, "without_failure")
    full_vs_silhouette_delta = full_acc - silhouette_acc
    without_failure_delta = full_acc - without_failure_acc
    multi_target_summary = dict((multi_target_payload or {}).get("summary", {}))
    live_multi_target_status = str(multi_target_summary.get("batch_status", "missing"))
    live_multi_target_dryrun_success_rate = float(
        multi_target_summary.get("dryrun_success_rate", 0.0)
    )
    live_multi_target_execute_success_rate = float(
        multi_target_summary.get("execute_success_rate", 0.0)
    )
    live_multi_target_no_truth_rate = float(
        multi_target_summary.get("no_truth_audit_pass_rate", 0.0)
    )
    live_multi_target_pose_accept_rate = float(
        multi_target_summary.get("pose_accept_rate", 0.0)
    )
    scenario_sweep_summary = dict((scenario_sweep_payload or {}).get("summary", {}))
    live_scenario_sweep_status = str(scenario_sweep_summary.get("batch_status", "missing"))
    live_scenario_hypothesis_success_rate = float(
        scenario_sweep_summary.get("hypothesis_success_rate", 0.0)
    )
    live_scenario_pose_accept_rate = float(
        scenario_sweep_summary.get("pose_accept_rate", 0.0)
    )
    live_scenario_no_truth_rate = float(
        scenario_sweep_summary.get("no_truth_audit_pass_rate", 0.0)
    )
    dynamic_execute = dict(dynamic_execute_payload or {})
    live_dynamic_execute_status = str(dynamic_execute.get("status_name", "missing"))
    live_dynamic_execute_event_published = bool(
        dynamic_execute.get("executed_event_published", False)
    )
    ranked_fallback = dict(ranked_fallback_payload or {})
    ranked_fallback_attempts = list(ranked_fallback.get("attempts", []))
    live_ranked_fallback_status = str(ranked_fallback.get("status_name", "missing"))
    live_ranked_fallback_attempt_count = len(ranked_fallback_attempts)
    live_ranked_fallback_first_attempt_simulated = bool(
        ranked_fallback_attempts
        and ranked_fallback_attempts[0].get("attempt_status_name") == "FAILED"
        and ranked_fallback_attempts[0].get("failure_reason") == "simulated_failure"
        and ranked_fallback_attempts[0].get("simulated") is True
    )
    gate_pass = (
        no_truth_audit_pass
        and full_vs_silhouette_delta > 0.0
        and live_multi_target_status == "pass"
        and live_scenario_sweep_status == "pass"
        and live_dynamic_execute_status == "SUCCEEDED"
        and live_ranked_fallback_status == "SUCCEEDED"
        and live_ranked_fallback_attempt_count >= 2
        and live_ranked_fallback_first_attempt_simulated
    )
    return {
        "schema_version": "m4_no_truth_gate_v1",
        "gate_status": "pass" if gate_pass else "fail",
        "synthetic_top1_success": full_acc,
        "real_replay_processed": len(real_rows),
        "live_multi_target_models": int(multi_target_summary.get("target_model_count", 0)),
        "live_multi_target_status": live_multi_target_status,
        "live_multi_target_dryrun_success_rate": live_multi_target_dryrun_success_rate,
        "live_multi_target_execute_success_rate": live_multi_target_execute_success_rate,
        "live_multi_target_no_truth_rate": live_multi_target_no_truth_rate,
        "live_multi_target_pose_accept_rate": live_multi_target_pose_accept_rate,
        "live_multi_target_shape_match_rate": float(
            multi_target_summary.get("shape_family_match_rate", 0.0)
        ),
        "live_scenario_sweep_count": int(scenario_sweep_summary.get("scenario_count", 0)),
        "live_scenario_sweep_status": live_scenario_sweep_status,
        "live_scenario_hypothesis_success_rate": live_scenario_hypothesis_success_rate,
        "live_scenario_pose_accept_rate": live_scenario_pose_accept_rate,
        "live_scenario_no_truth_rate": live_scenario_no_truth_rate,
        "live_dynamic_execute_status": live_dynamic_execute_status,
        "live_dynamic_execute_event_published": live_dynamic_execute_event_published,
        "live_ranked_fallback_status": live_ranked_fallback_status,
        "live_ranked_fallback_attempt_count": live_ranked_fallback_attempt_count,
        "live_ranked_fallback_first_attempt_simulated": live_ranked_fallback_first_attempt_simulated,
        "full_vs_silhouette_delta": full_vs_silhouette_delta,
        "without_failure_delta": without_failure_delta,
        "no_truth_audit_pass": bool(no_truth_audit_pass),
        "audit_findings": list(audit_findings or []),
        "known_gaps": [
            "This is M4.8-headless, not the full S0-S11 RA-L evidence gate.",
            "Real replay rows summarize current D435 bags; they are not robot lift-and-hold trials.",
            "Dynamic Gazebo smoke is validated by a separate launch-level script.",
            "Live multi-target shape-family match is reported, not used as a hard pass/fail gate.",
            "Live scenario sweep currently covers S0-S7, not the full S0-S11 matrix.",
        ],
    }


def build_no_truth_gate_report(
    *,
    output_dir: str | Path,
    real_dashboard_json: str | Path | None = None,
    multi_target_json: str | Path | None = None,
    scenario_sweep_json: str | Path | None = None,
    dynamic_execute_json: str | Path | None = None,
    ranked_fallback_json: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    synthetic_rows = run_synthetic_no_truth_rows()
    real_rows = load_real_dashboard_rows(real_dashboard_json)
    multi_target_payload = load_multi_target_payload(multi_target_json)
    scenario_sweep_payload = load_scenario_sweep_payload(scenario_sweep_json)
    dynamic_execute_payload = load_dynamic_execute_payload(dynamic_execute_json)
    ranked_fallback_payload = load_dynamic_execute_payload(ranked_fallback_json)
    audit_pass, audit_findings = run_no_truth_audit(repo_root)
    report = summarize_no_truth_gate(
        synthetic_rows=synthetic_rows,
        real_rows=real_rows,
        multi_target_payload=multi_target_payload,
        scenario_sweep_payload=scenario_sweep_payload,
        dynamic_execute_payload=dynamic_execute_payload,
        ranked_fallback_payload=ranked_fallback_payload,
        no_truth_audit_pass=audit_pass,
        audit_findings=audit_findings,
    )
    report["rows"] = _report_rows(
        synthetic_rows,
        real_rows,
        multi_target_payload.get("rows", []),
        scenario_sweep_payload.get("rows", []),
        dynamic_execute_payload,
        ranked_fallback_payload,
        report,
    )
    _write_report_files(report, output)
    return report


def _report_rows(
    synthetic_rows: list[NoTruthSyntheticRow],
    real_rows: list[dict[str, Any]],
    multi_target_rows: list[dict[str, Any]],
    scenario_sweep_rows: list[dict[str, Any]],
    dynamic_execute_payload: dict[str, Any],
    ranked_fallback_payload: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in synthetic_rows:
        row_dict = asdict(row)
        rows.append(
            {
                "category": "synthetic",
                "scenario_id": row.scene_id,
                "ranker": row.ranker,
                "ground_truth_id": row.ground_truth_id,
                "top1_hypothesis_id": row.top1_hypothesis_id,
                "top1_correct": row.top1_correct,
                "ground_truth_rank": row.ground_truth_rank,
                "detail": json.dumps(row_dict, sort_keys=True),
                "value": "",
            }
        )
    rows.extend(
        [
            {
                "category": "ablation",
                "scenario_id": "aggregate",
                "ranker": "full_vs_silhouette",
                "ground_truth_id": "",
                "top1_hypothesis_id": "",
                "top1_correct": "",
                "ground_truth_rank": "",
                "detail": "full top-1 accuracy minus silhouette-only top-1 accuracy",
                "value": summary["full_vs_silhouette_delta"],
            },
            {
                "category": "ablation",
                "scenario_id": "aggregate",
                "ranker": "without_failure",
                "ground_truth_id": "",
                "top1_hypothesis_id": "",
                "top1_correct": "",
                "ground_truth_rank": "",
                "detail": "full top-1 accuracy minus without-failure top-1 accuracy",
                "value": summary["without_failure_delta"],
            },
        ]
    )
    for row in real_rows:
        rows.append(
            {
                "category": "real_replay",
                "scenario_id": row.get("scene_id", ""),
                "ranker": "real_dashboard",
                "ground_truth_id": "",
                "top1_hypothesis_id": row.get("failure_top", ""),
                "top1_correct": row.get("weak_gt_pass", ""),
                "ground_truth_rank": "",
                "detail": row.get("target_label", ""),
                "value": row.get("failure_score_delta", ""),
            }
        )
    for row in multi_target_rows:
        rows.append(
            {
                "category": "live_multi_target",
                "scenario_id": row.get("case_id", ""),
                "ranker": row.get("target_model", ""),
                "ground_truth_id": row.get("expected_shape", ""),
                "top1_hypothesis_id": row.get("predicted_shape", ""),
                "top1_correct": row.get("shape_family_match", ""),
                "ground_truth_rank": "",
                "detail": json.dumps(
                    {
                        "hypothesis_id": row.get("hypothesis_id", ""),
                        "pose_error_m": row.get("pose_error_m", ""),
                        "dryrun_planned": row.get("dryrun_planned", ""),
                        "execute_status_name": row.get("execute_status_name", ""),
                        "no_truth_audit": row.get("no_truth_audit", ""),
                    },
                    sort_keys=True,
                ),
                "value": row.get("pose_error_m", ""),
            }
        )
    for row in scenario_sweep_rows:
        rows.append(
            {
                "category": "live_scenario_sweep",
                "scenario_id": row.get("scenario_id", ""),
                "ranker": row.get("target_model", ""),
                "ground_truth_id": "",
                "top1_hypothesis_id": row.get("hypothesis_id", ""),
                "top1_correct": row.get("hypothesis_count", 0) > 0,
                "ground_truth_rank": "",
                "detail": json.dumps(
                    {
                        "shape_type": row.get("shape_type", ""),
                        "pose_error_m": row.get("pose_error_m", ""),
                        "confidence": row.get("confidence", ""),
                        "no_truth_audit": row.get("no_truth_audit", ""),
                    },
                    sort_keys=True,
                ),
                "value": row.get("pose_error_m", ""),
            }
        )
    if dynamic_execute_payload:
        rows.append(
            {
                "category": "live_dynamic_execute",
                "scenario_id": "dynamic_rerun",
                "ranker": dynamic_execute_payload.get("hypotheses_topic", ""),
                "ground_truth_id": "",
                "top1_hypothesis_id": dynamic_execute_payload.get(
                    "selected_hypothesis_id", ""
                ),
                "top1_correct": dynamic_execute_payload.get("status_name", "")
                == "SUCCEEDED",
                "ground_truth_rank": "",
                "detail": json.dumps(
                    {
                        "final_status_name": dynamic_execute_payload.get(
                            "final_status_name", ""
                        ),
                        "executed_event_published": dynamic_execute_payload.get(
                            "executed_event_published", ""
                        ),
                        "runtime_sec": dynamic_execute_payload.get("runtime_sec", ""),
                    },
                    sort_keys=True,
                ),
                "value": dynamic_execute_payload.get("status_name", ""),
            }
        )
    if ranked_fallback_payload:
        rows.append(
            {
                "category": "live_ranked_fallback",
                "scenario_id": "simulated_top1_failure",
                "ranker": ranked_fallback_payload.get("hypotheses_topic", ""),
                "ground_truth_id": "",
                "top1_hypothesis_id": ranked_fallback_payload.get(
                    "selected_hypothesis_id", ""
                ),
                "top1_correct": ranked_fallback_payload.get("status_name", "")
                == "SUCCEEDED",
                "ground_truth_rank": "",
                "detail": json.dumps(
                    {
                        "attempt_count": summary.get(
                            "live_ranked_fallback_attempt_count", ""
                        ),
                        "first_attempt_simulated": summary.get(
                            "live_ranked_fallback_first_attempt_simulated", ""
                        ),
                        "final_status_name": ranked_fallback_payload.get(
                            "final_status_name", ""
                        ),
                        "executed_event_published": ranked_fallback_payload.get(
                            "executed_event_published", ""
                        ),
                    },
                    sort_keys=True,
                ),
                "value": ranked_fallback_payload.get("status_name", ""),
            }
        )
    rows.append(
        {
            "category": "no_truth_audit",
            "scenario_id": "source_scan",
            "ranker": "audit",
            "ground_truth_id": "",
            "top1_hypothesis_id": "",
            "top1_correct": summary["no_truth_audit_pass"],
            "ground_truth_rank": "",
            "detail": "; ".join(summary["audit_findings"]),
            "value": summary["no_truth_audit_pass"],
        }
    )
    return rows


def _write_report_files(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_fields = [
        "category",
        "scenario_id",
        "ranker",
        "ground_truth_id",
        "top1_hypothesis_id",
        "top1_correct",
        "ground_truth_rank",
        "detail",
        "value",
    ]
    with (output_dir / "gate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(report["rows"])
    (output_dir / "index.md").write_text(_render_index(report), encoding="utf-8")


def _render_index(report: dict[str, Any]) -> str:
    return (
        "# M4 No-Truth Gate\n\n"
        f"- Gate status: `{report['gate_status']}`\n"
        f"- Synthetic top-1 success: `{report['synthetic_top1_success']:.3f}`\n"
        f"- Full vs silhouette delta: `{report['full_vs_silhouette_delta']:.3f}`\n"
        f"- Full vs without-failure delta: `{report['without_failure_delta']:.3f}`\n"
        f"- Real replay processed: `{report['real_replay_processed']}`\n"
        f"- Live multi-target models: `{report['live_multi_target_models']}`\n"
        f"- Live multi-target dry-run success: `{report['live_multi_target_dryrun_success_rate']:.3f}`\n"
        f"- Live multi-target pose accept: `{report['live_multi_target_pose_accept_rate']:.3f}`\n"
        f"- Live multi-target shape match: `{report['live_multi_target_shape_match_rate']:.3f}`\n"
        f"- Live scenario sweep count: `{report['live_scenario_sweep_count']}`\n"
        f"- Live scenario hypothesis success: `{report['live_scenario_hypothesis_success_rate']:.3f}`\n"
        f"- Live scenario pose accept: `{report['live_scenario_pose_accept_rate']:.3f}`\n"
        f"- Live dynamic execute status: `{report['live_dynamic_execute_status']}`\n"
        f"- Live ranked fallback status: `{report['live_ranked_fallback_status']}`\n"
        f"- Live ranked fallback attempts: `{report['live_ranked_fallback_attempt_count']}`\n"
        f"- No-truth audit pass: `{report['no_truth_audit_pass']}`\n\n"
        "## Known Gaps\n\n"
        + "\n".join(f"- {gap}" for gap in report["known_gaps"])
        + "\n"
    )


def _top1_accuracy(rows: list[NoTruthSyntheticRow], ranker: str) -> float:
    selected = [row for row in rows if row.ranker == ranker]
    if not selected:
        return 0.0
    return float(sum(row.top1_correct for row in selected)) / float(len(selected))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=Path("reports/m4_no_truth_gate"), type=Path)
    parser.add_argument(
        "--real-dashboard-json",
        default=Path("reports/m4_real_dashboard/dashboard.json"),
        type=Path,
    )
    parser.add_argument(
        "--multi-target-json",
        default=Path("reports/m4_no_truth_multi_target/multi_target.json"),
        type=Path,
    )
    parser.add_argument(
        "--scenario-sweep-json",
        default=Path("reports/m4_no_truth_scenario_sweep/scenario_sweep.json"),
        type=Path,
    )
    parser.add_argument(
        "--dynamic-execute-json",
        default=Path("reports/m4_no_truth_live_dynamic_execute/result.json"),
        type=Path,
    )
    parser.add_argument(
        "--ranked-fallback-json",
        default=Path("reports/m4_no_truth_live_ranked_fallback/result.json"),
        type=Path,
    )
    parser.add_argument("--repo-root", default=None, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_no_truth_gate_report(
        output_dir=args.output_dir,
        real_dashboard_json=args.real_dashboard_json,
        multi_target_json=args.multi_target_json,
        scenario_sweep_json=args.scenario_sweep_json,
        dynamic_execute_json=args.dynamic_execute_json,
        ranked_fallback_json=args.ranked_fallback_json,
        repo_root=args.repo_root,
    )
    print(f"Wrote M4 no-truth gate: {args.output_dir}")
    print(f"gate_status={report['gate_status']}")
    print(f"synthetic_top1_success={report['synthetic_top1_success']:.3f}")
    print(f"real_replay_processed={report['real_replay_processed']}")
    print(f"live_multi_target_models={report['live_multi_target_models']}")
    print(f"live_scenario_sweep_count={report['live_scenario_sweep_count']}")
    print(f"live_dynamic_execute_status={report['live_dynamic_execute_status']}")
    print(f"live_ranked_fallback_status={report['live_ranked_fallback_status']}")


if __name__ == "__main__":
    main()
