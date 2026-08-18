import json

from ghost_mgg_core_py.evaluation.m4_hard_synthetic_eval import (
    run_hard_synthetic_eval,
    write_hard_synthetic_reports,
)
from ghost_mgg_core_py.evaluation.synthetic_scene import make_failure_ranking_scene
from ghost_mgg_core_py.ghost_mgg_v0 import (
    GhostMGGV0Config,
    SILHOUETTE_ONLY_WEIGHTS,
    run_ghost_mgg_v0,
)


def test_ghost_mgg_v0_runner_exposes_failure_aware_ranking():
    scene = make_failure_ranking_scene("hard", shift_px=1)

    silhouette_only = run_ghost_mgg_v0(
        scene.target_mask,
        scene.evidence,
        config=GhostMGGV0Config(top_k=2),
        weights=SILHOUETTE_ONLY_WEIGHTS,
        hypotheses=scene.candidates,
    )
    failure_aware = run_ghost_mgg_v0(
        scene.target_mask,
        scene.evidence,
        config=GhostMGGV0Config(top_k=2),
        hypotheses=scene.candidates,
    )

    assert silhouette_only[0].hypothesis.hypothesis_id != scene.ground_truth_id
    assert failure_aware[0].hypothesis.hypothesis_id == scene.ground_truth_id
    assert failure_aware[0].score.failure > failure_aware[1].score.failure
    assert set(failure_aware[0].score.as_dict()) == {
        "visual",
        "failure",
        "depth",
        "physical",
        "grasp",
        "prior",
        "total",
    }


def test_m4_hard_synthetic_eval_shows_failure_aware_gain():
    rows = run_hard_synthetic_eval()
    by_ranker = {}
    for row in rows:
        by_ranker.setdefault(row.ranker, []).append(row)

    silhouette_accuracy = sum(row.top1_correct for row in by_ranker["silhouette_only"]) / len(
        by_ranker["silhouette_only"]
    )
    failure_accuracy = sum(row.top1_correct for row in by_ranker["failure_aware"]) / len(
        by_ranker["failure_aware"]
    )

    assert len(rows) == 10
    assert silhouette_accuracy == 0.0
    assert failure_accuracy == 1.0


def test_m4_hard_synthetic_eval_writes_aggregate_report(tmp_path):
    rows = run_hard_synthetic_eval()
    output_csv = tmp_path / "hard.csv"
    output_json = tmp_path / "hard.json"

    write_hard_synthetic_reports(rows, output_csv, output_json)

    assert "scene_id,ranker" in output_csv.read_text(encoding="utf-8")
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == "m4_hard_synthetic_eval_v1"
    assert report["aggregate_by_ranker"]["failure_aware"]["top1_accuracy"] == 1.0
    assert report["aggregate_by_ranker"]["silhouette_only"]["top1_accuracy"] == 0.0
