import csv
import json
from pathlib import Path

from ghost_mgg_core_py.evaluation.m4_joint_hypothesis_report import (
    build_joint_hypothesis_report,
    write_joint_hypothesis_reports,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_m4_joint_hypothesis_report_merges_real_and_sim_validation(tmp_path):
    real_ranking = tmp_path / "m5_real_ranking.json"
    graspability = tmp_path / "graspability.json"
    moveit = tmp_path / "moveit.json"

    _write_json(
        real_ranking,
        {
            "schema_version": "m5_real_d435_ranking_v1",
            "rows": [
                {
                    "scene_id": "scene_a",
                    "ranker": "failure_aware",
                    "rank": 1,
                    "hypothesis_id": "box_s1.00",
                    "shape_type": "box",
                    "total_score": 0.82,
                    "failure_score": 0.34,
                    "visual_score": 0.78,
                },
                {
                    "scene_id": "scene_a",
                    "ranker": "silhouette_only",
                    "rank": 1,
                    "hypothesis_id": "cylinder_s1.00",
                    "shape_type": "cylinder",
                    "total_score": 0.70,
                    "failure_score": 0.0,
                    "visual_score": 0.70,
                },
            ],
        },
    )
    _write_json(
        graspability,
        {
            "schema_version": "m4_graspability_dryrun_v1",
            "rows": [
                {
                    "scene_id": "scene_a",
                    "ranker": "failure_aware",
                    "hypothesis_id": "box_s1.00",
                    "grasp_id": "box_s1.00_top",
                    "grasp_type": "top",
                    "shape_type": "box",
                    "score": 0.91,
                    "valid": True,
                    "failure_reason": "",
                    "gripper_width_margin_m": 0.022,
                },
                {
                    "scene_id": "scene_a",
                    "ranker": "silhouette_only",
                    "hypothesis_id": "cylinder_s1.00",
                    "grasp_id": "cylinder_s1.00_top",
                    "grasp_type": "top",
                    "shape_type": "cylinder",
                    "score": 0.40,
                    "valid": False,
                    "failure_reason": "width",
                    "gripper_width_margin_m": -0.01,
                },
            ],
        },
    )
    _write_json(
        moveit,
        {
            "schema_version": "m4_sim_moveit_dryrun_v1",
            "rows": [
                {
                    "target_id": "red_cube",
                    "shape_type": "box",
                    "planned": True,
                    "pregrasp_planned": True,
                    "grasp_checked": False,
                    "descent_points_world": [{"x": 0.0, "y": 0.0, "z": 0.86}],
                    "descent_clearance": {"status": "ok", "min_clearance_m": 0.04},
                    "attempts": [{"planned": True, "path_points_world": [{}, {}, {}]}],
                },
                {
                    "target_id": "blue_cylinder",
                    "shape_type": "cylinder",
                    "planned": False,
                    "pregrasp_planned": False,
                    "grasp_checked": False,
                    "descent_points_world": [],
                    "descent_clearance": {"status": "low", "min_clearance_m": 0.01},
                    "attempts": [],
                },
            ],
        },
    )

    report = build_joint_hypothesis_report(real_ranking, graspability, moveit)

    assert report["schema_version"] == "m4_joint_hypothesis_report_v1"
    assert report["summary"]["real_rows"] == 2
    assert report["summary"]["sim_moveit_rows"] == 2
    assert report["summary"]["executable_rows"] == 2

    rows = report["rows"]
    real_rows = [row for row in rows if row["source_type"] == "real_graspability"]
    sim_rows = [row for row in rows if row["source_type"] == "sim_moveit"]

    assert real_rows[0]["rank_group"] == "real:scene_a"
    assert real_rows[0]["joint_rank"] == 1
    assert real_rows[0]["decision"] == "candidate"
    assert real_rows[0]["ranker"] == "failure_aware"
    assert real_rows[0]["moveit_pregrasp_planned"] is None

    red_cube = next(row for row in sim_rows if row["target_or_scene_id"] == "red_cube")
    assert red_cube["decision"] == "executable"
    assert red_cube["joint_score"] == 1.0
    assert red_cube["moveit_pregrasp_planned"] is True
    assert red_cube["descent_clearance_status"] == "ok"
    assert red_cube["path_points_count"] == 3
    assert red_cube["descent_points_count"] == 1

    blue = next(row for row in sim_rows if row["target_or_scene_id"] == "blue_cylinder")
    assert blue["decision"] == "reject"
    assert blue["joint_score"] == 0.0


def test_m4_joint_hypothesis_report_writes_json_csv_and_index(tmp_path):
    report = {
        "schema_version": "m4_joint_hypothesis_report_v1",
        "summary": {"total_rows": 1, "executable_rows": 1},
        "rows": [
            {
                "source_type": "sim_moveit",
                "rank_group": "sim:m4_tabletop",
                "joint_rank": 1,
                "target_or_scene_id": "red_cube",
                "hypothesis_id": "red_cube",
                "shape_type": "box",
                "decision": "executable",
                "joint_score": 1.0,
                "visual_score": None,
                "grasp_score": None,
                "moveit_pregrasp_planned": True,
                "descent_clearance_status": "ok",
                "failure_reason": "",
            }
        ],
    }

    output_dir = tmp_path / "joint"
    write_joint_hypothesis_reports(report, output_dir)

    written = json.loads((output_dir / "joint_hypotheses.json").read_text(encoding="utf-8"))
    assert written["schema_version"] == "m4_joint_hypothesis_report_v1"

    with (output_dir / "joint_hypotheses.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["decision"] == "executable"
    assert rows[0]["descent_clearance_status"] == "ok"

    index_text = (output_dir / "index.md").read_text(encoding="utf-8")
    assert "# M4 Joint Hypothesis Report" in index_text
    assert "red_cube" in index_text
    assert "executable" in index_text


def test_m4_joint_hypothesis_shell_entrypoint_exists():
    script = Path(__file__).resolve().parents[3] / "scripts" / "run_m4_joint_hypothesis_report.sh"
    text = script.read_text(encoding="utf-8")

    assert "m4_joint_hypothesis_report" in text
    assert "reports/m4_joint_hypotheses" in text
