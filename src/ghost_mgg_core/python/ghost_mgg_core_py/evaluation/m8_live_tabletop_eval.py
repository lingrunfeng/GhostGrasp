from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FORBIDDEN_TRUTH_TOKENS = (
    "gz model",
    "gazebo_truth",
    "model_pose",
    "ground_truth_pose",
    "sdf_dimension",
    "model://",
)


@dataclass(frozen=True)
class M8MovedObjectSnapshot:
    object_id: str
    shape_type: str
    center_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    yaw_rad: float


@dataclass(frozen=True)
class M8LiveHypothesisSnapshot:
    hypothesis_id: str
    component_id: int
    shape_type: str
    center_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    yaw_rad: float
    provenance: str


@dataclass(frozen=True)
class M8TruthGeometry:
    object_id: str
    shape_type: str
    center_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    yaw_rad: float


@dataclass(frozen=True)
class M8HypothesisGeometry:
    hypothesis_id: str
    shape_type: str
    center_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    yaw_rad: float
    provenance: str
    score_total: float = 0.0
    score_visual: float = 0.0
    score_failure: float = 0.0
    score_depth: float = 0.0
    score_prior: float = 0.0


@dataclass(frozen=True)
class M8DynamicEvalRow:
    object_id: str
    matched_before_hypothesis_id: str
    matched_after_hypothesis_id: str
    before_hypothesis_count: int
    after_hypothesis_count: int
    before_center_error_m: float
    after_center_error_m: float
    update_latency_sec: float
    size_drift_m: float
    yaw_drift_rad: float
    no_truth_audit_pass: bool
    status: str


@dataclass(frozen=True)
class M8StrictGeometryEvalRow:
    object_id: str
    truth_shape_type: str
    matched_hypothesis_id: str
    hypothesis_shape_type: str
    center_error_m: float
    size_error_m: float
    yaw_error_rad: float
    shape_match: bool
    no_truth_audit_pass: bool
    status: str


def evaluate_strict_geometry_snapshot(
    *,
    truths: list[M8TruthGeometry],
    hypotheses: list[M8HypothesisGeometry],
    max_center_error_m: float = 0.008,
    max_size_error_m: float = 0.010,
    max_yaw_error_rad: float = math.radians(12.0),
) -> list[M8StrictGeometryEvalRow]:
    rows: list[M8StrictGeometryEvalRow] = []
    no_truth_pass = _no_truth_audit_pass(hypotheses)
    for truth in truths:
        match = _best_strict_geometry_hypothesis(
            truth,
            hypotheses,
            max_center_error_m=float(max_center_error_m),
            max_size_error_m=float(max_size_error_m),
            max_yaw_error_rad=float(max_yaw_error_rad),
        )
        if match is None:
            rows.append(
                M8StrictGeometryEvalRow(
                    object_id=str(truth.object_id),
                    truth_shape_type=str(truth.shape_type),
                    matched_hypothesis_id="",
                    hypothesis_shape_type="",
                    center_error_m=float("inf"),
                    size_error_m=float("inf"),
                    yaw_error_rad=float("inf"),
                    shape_match=False,
                    no_truth_audit_pass=bool(no_truth_pass),
                    status="fail",
                )
            )
            continue

        center_error = _distance_xy(truth.center_xy_m, match.center_xy_m)
        size_error = _size_drift(truth.size_xy_m, match.size_xy_m)
        yaw_error = _geometry_yaw_error(truth, match)
        shape_match = _shape_matches(str(truth.shape_type), str(match.shape_type))
        passed = (
            shape_match
            and center_error <= float(max_center_error_m)
            and size_error <= float(max_size_error_m)
            and yaw_error <= float(max_yaw_error_rad)
            and no_truth_pass
        )
        rows.append(
            M8StrictGeometryEvalRow(
                object_id=str(truth.object_id),
                truth_shape_type=str(truth.shape_type),
                matched_hypothesis_id=str(match.hypothesis_id),
                hypothesis_shape_type=str(match.shape_type),
                center_error_m=float(center_error),
                size_error_m=float(size_error),
                yaw_error_rad=float(yaw_error),
                shape_match=bool(shape_match),
                no_truth_audit_pass=bool(no_truth_pass),
                status="pass" if passed else "fail",
            )
        )
    return rows


def summarize_strict_geometry_rows(rows: list[M8StrictGeometryEvalRow]) -> dict[str, Any]:
    pass_count = sum(row.status == "pass" for row in rows)
    count = len(rows)
    return {
        "gate_status": "pass" if count > 0 and pass_count == count else "fail",
        "row_count": count,
        "pass_count": pass_count,
        "pass_rate": float(pass_count / count) if count else 0.0,
        "max_center_error_m": max((row.center_error_m for row in rows), default=0.0),
        "max_size_error_m": max((row.size_error_m for row in rows), default=0.0),
        "max_yaw_error_rad": max((row.yaw_error_rad for row in rows), default=0.0),
        "shape_match_rate": (
            float(sum(row.shape_match for row in rows) / count) if count else 0.0
        ),
        "no_truth_pass_rate": (
            float(sum(row.no_truth_audit_pass for row in rows) / count) if count else 0.0
        ),
    }


def evaluate_dynamic_tabletop_update(
    *,
    before_object: M8MovedObjectSnapshot,
    after_object: M8MovedObjectSnapshot,
    before_hypotheses: list[M8LiveHypothesisSnapshot],
    after_hypotheses: list[M8LiveHypothesisSnapshot],
    update_latency_sec: float,
    max_center_error_m: float = 0.030,
    max_update_latency_sec: float = 2.0,
    max_size_drift_m: float = 0.030,
    max_yaw_drift_rad: float = math.radians(20.0),
) -> M8DynamicEvalRow:
    before_match = _nearest_hypothesis(before_object.center_xy_m, before_hypotheses)
    after_match = _nearest_hypothesis(after_object.center_xy_m, after_hypotheses)
    before_error = (
        _distance_xy(before_object.center_xy_m, before_match.center_xy_m)
        if before_match is not None
        else float("inf")
    )
    after_error = (
        _distance_xy(after_object.center_xy_m, after_match.center_xy_m)
        if after_match is not None
        else float("inf")
    )
    size_drift = (
        _size_drift(after_object.size_xy_m, after_match.size_xy_m)
        if after_match is not None
        else float("inf")
    )
    yaw_drift = (
        abs(_normalize_half_turn(after_object.yaw_rad - after_match.yaw_rad))
        if after_match is not None
        else float("inf")
    )
    no_truth_pass = _no_truth_audit_pass(before_hypotheses + after_hypotheses)
    pass_status = (
        before_match is not None
        and after_match is not None
        and before_error <= float(max_center_error_m)
        and after_error <= float(max_center_error_m)
        and float(update_latency_sec) <= float(max_update_latency_sec)
        and size_drift <= float(max_size_drift_m)
        and yaw_drift <= float(max_yaw_drift_rad)
        and no_truth_pass
    )
    return M8DynamicEvalRow(
        object_id=str(after_object.object_id),
        matched_before_hypothesis_id=before_match.hypothesis_id if before_match else "",
        matched_after_hypothesis_id=after_match.hypothesis_id if after_match else "",
        before_hypothesis_count=len(before_hypotheses),
        after_hypothesis_count=len(after_hypotheses),
        before_center_error_m=float(before_error),
        after_center_error_m=float(after_error),
        update_latency_sec=float(update_latency_sec),
        size_drift_m=float(size_drift),
        yaw_drift_rad=float(yaw_drift),
        no_truth_audit_pass=bool(no_truth_pass),
        status="pass" if pass_status else "fail",
    )


def summarize_dynamic_rows(rows: list[M8DynamicEvalRow]) -> dict[str, Any]:
    pass_count = sum(row.status == "pass" for row in rows)
    count = len(rows)
    no_truth_pass_count = sum(row.no_truth_audit_pass for row in rows)
    return {
        "gate_status": "pass" if count > 0 and pass_count == count else "fail",
        "row_count": count,
        "pass_count": pass_count,
        "pass_rate": float(pass_count / count) if count else 0.0,
        "no_truth_pass_rate": float(no_truth_pass_count / count) if count else 0.0,
        "max_after_center_error_m": max((row.after_center_error_m for row in rows), default=0.0),
        "max_update_latency_sec": max((row.update_latency_sec for row in rows), default=0.0),
        "max_size_drift_m": max((row.size_drift_m for row in rows), default=0.0),
        "max_yaw_drift_rad": max((row.yaw_drift_rad for row in rows), default=0.0),
    }


def write_dynamic_eval_report(rows: list[M8DynamicEvalRow], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = summarize_dynamic_rows(rows)
    payload = {
        "schema_version": "m8_live_tabletop_headless_eval_v1",
        "summary": summary,
        "rows": [asdict(row) for row in rows],
    }
    (output_path / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_path / "index.md").write_text(_markdown_report(payload), encoding="utf-8")


def make_synthetic_dynamic_rows() -> list[M8DynamicEvalRow]:
    rows: list[M8DynamicEvalRow] = []
    cases = (
        (
            M8MovedObjectSnapshot("red_cube", "box", (0.03, 0.18), (0.04, 0.04), 0.25),
            M8MovedObjectSnapshot("red_cube", "box", (0.11, 0.22), (0.04, 0.04), 0.55),
            M8LiveHypothesisSnapshot(
                "component_1_box",
                1,
                "box",
                (0.031, 0.179),
                (0.041, 0.040),
                0.24,
                "m4_live_no_truth_live_tabletop_core",
            ),
            M8LiveHypothesisSnapshot(
                "component_1_box",
                1,
                "box",
                (0.109, 0.221),
                (0.040, 0.041),
                0.54,
                "m4_live_no_truth_live_tabletop_core",
            ),
            0.55,
        ),
        (
            M8MovedObjectSnapshot("green_cylinder", "cylinder", (-0.02, 0.20), (0.035, 0.035), 0.0),
            M8MovedObjectSnapshot("green_cylinder", "cylinder", (0.02, 0.24), (0.035, 0.035), 0.0),
            M8LiveHypothesisSnapshot(
                "component_2_cylinder",
                2,
                "cylinder",
                (-0.021, 0.199),
                (0.036, 0.036),
                0.0,
                "m4_live_no_truth_live_tabletop_core",
            ),
            M8LiveHypothesisSnapshot(
                "component_2_cylinder",
                2,
                "cylinder",
                (0.021, 0.239),
                (0.036, 0.036),
                0.0,
                "m4_live_no_truth_live_tabletop_core",
            ),
            0.40,
        ),
    )
    for before_object, after_object, before_hypothesis, after_hypothesis, latency in cases:
        rows.append(
            evaluate_dynamic_tabletop_update(
                before_object=before_object,
                after_object=after_object,
                before_hypotheses=[before_hypothesis],
                after_hypotheses=[after_hypothesis],
                update_latency_sec=latency,
            )
        )
    return rows


def _nearest_hypothesis(
    center_xy_m: tuple[float, float], hypotheses: list[M8LiveHypothesisSnapshot]
) -> M8LiveHypothesisSnapshot | None:
    if not hypotheses:
        return None
    return min(hypotheses, key=lambda hypothesis: _distance_xy(center_xy_m, hypothesis.center_xy_m))


def _nearest_geometry_hypothesis(
    center_xy_m: tuple[float, float], hypotheses: list[M8HypothesisGeometry]
) -> M8HypothesisGeometry | None:
    if not hypotheses:
        return None
    return min(hypotheses, key=lambda hypothesis: _distance_xy(center_xy_m, hypothesis.center_xy_m))


def _best_strict_geometry_hypothesis(
    truth: M8TruthGeometry,
    hypotheses: list[M8HypothesisGeometry],
    *,
    max_center_error_m: float,
    max_size_error_m: float,
    max_yaw_error_rad: float,
) -> M8HypothesisGeometry | None:
    shape_matching: list[tuple[float, float, float, M8HypothesisGeometry]] = []
    for hypothesis in hypotheses:
        if not _shape_matches(str(truth.shape_type), str(hypothesis.shape_type)):
            continue
        center_error = _distance_xy(truth.center_xy_m, hypothesis.center_xy_m)
        size_error = _size_drift(truth.size_xy_m, hypothesis.size_xy_m)
        yaw_error = _geometry_yaw_error(truth, hypothesis)
        if (
            center_error <= float(max_center_error_m)
            and size_error <= float(max_size_error_m)
            and yaw_error <= float(max_yaw_error_rad)
        ):
            shape_matching.append((center_error, size_error, yaw_error, hypothesis))
    if shape_matching:
        return min(shape_matching, key=lambda item: (item[0], item[1], item[2]))[3]
    return _nearest_geometry_hypothesis(truth.center_xy_m, hypotheses)


def _distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _size_drift(expected: tuple[float, float], actual: tuple[float, float]) -> float:
    expected_sorted = sorted((float(expected[0]), float(expected[1])))
    actual_sorted = sorted((float(actual[0]), float(actual[1])))
    return max(
        abs(expected_sorted[0] - actual_sorted[0]),
        abs(expected_sorted[1] - actual_sorted[1]),
    )


def _normalize_half_turn(angle: float) -> float:
    normalized = float(angle)
    while normalized <= -math.pi / 2.0:
        normalized += math.pi
    while normalized > math.pi / 2.0:
        normalized -= math.pi
    return normalized


def _normalize_quarter_turn(angle: float) -> float:
    normalized = float(angle)
    while normalized <= -math.pi / 4.0:
        normalized += math.pi / 2.0
    while normalized > math.pi / 4.0:
        normalized -= math.pi / 2.0
    return normalized


def _geometry_yaw_error(truth: M8TruthGeometry, hypothesis: M8HypothesisGeometry) -> float:
    truth_shape = str(truth.shape_type).lower()
    if truth_shape == "cylinder":
        return 0.0
    if _is_squareish(truth.size_xy_m):
        if _is_squareish(hypothesis.size_xy_m):
            return 0.0
        return abs(_normalize_quarter_turn(float(truth.yaw_rad) - float(hypothesis.yaw_rad)))
    return abs(_normalize_half_turn(float(truth.yaw_rad) - float(hypothesis.yaw_rad)))


def _is_squareish(size_xy_m: tuple[float, float], max_aspect: float = 1.12) -> bool:
    smaller = max(1e-9, min(float(size_xy_m[0]), float(size_xy_m[1])))
    larger = max(float(size_xy_m[0]), float(size_xy_m[1]))
    return bool(larger / smaller <= float(max_aspect))


def _shape_matches(truth_shape: str, hypothesis_shape: str) -> bool:
    truth = str(truth_shape).lower()
    hypothesis = str(hypothesis_shape).lower()
    if truth == "box":
        return hypothesis in {"box", "bbox"}
    if truth == "bbox":
        return hypothesis in {"box", "bbox"}
    if truth == "cylinder":
        return hypothesis == "cylinder"
    return hypothesis in {truth, "bbox", "box"}


def _no_truth_audit_pass(hypotheses: list[Any]) -> bool:
    for hypothesis in hypotheses:
        provenance = str(hypothesis.provenance).lower()
        if any(token in provenance for token in FORBIDDEN_TRUTH_TOKENS):
            return False
    return True


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = dict(payload["summary"])
    lines = [
        "# M8 Live Tabletop Headless Eval",
        "",
        f"- gate_status: `{summary['gate_status']}`",
        f"- rows: `{summary['row_count']}`",
        f"- pass_rate: `{summary['pass_rate']:.3f}`",
        f"- no_truth_pass_rate: `{summary['no_truth_pass_rate']:.3f}`",
        f"- max_after_center_error_m: `{summary['max_after_center_error_m']:.4f}`",
        f"- max_update_latency_sec: `{summary['max_update_latency_sec']:.3f}`",
        f"- max_size_drift_m: `{summary['max_size_drift_m']:.4f}`",
        f"- max_yaw_drift_rad: `{summary['max_yaw_drift_rad']:.4f}`",
        "",
        "| object | status | matched_after | center_error_m | latency_s | size_drift_m | yaw_drift_rad |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {object_id} | {status} | {matched_after_hypothesis_id} | "
            "{after_center_error_m:.4f} | {update_latency_sec:.3f} | "
            "{size_drift_m:.4f} | {yaw_drift_rad:.4f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an M8 live tabletop headless eval report.")
    parser.add_argument(
        "--output-dir",
        default=Path("reports/m8_live_tabletop_headless_eval"),
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = make_synthetic_dynamic_rows()
    write_dynamic_eval_report(rows, args.output_dir)
    summary = summarize_dynamic_rows(rows)
    print(f"M8 live tabletop headless eval: {summary['gate_status']} -> {args.output_dir}")


if __name__ == "__main__":
    main()
