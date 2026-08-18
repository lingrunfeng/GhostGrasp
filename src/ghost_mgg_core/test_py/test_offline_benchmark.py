from ghost_mgg_core_py.evaluation.offline_benchmark import run_minimal_m1_benchmark
from ghost_mgg_core_py.evaluation.synthetic_scene import make_failure_ranking_scene
from ghost_mgg_core_py.scoring.joint_ranker import rank_hypotheses


def test_synthetic_scene_failure_evidence_changes_top_rank():
    scene = make_failure_ranking_scene("scene_a", shift_px=2)

    silhouette_only = rank_hypotheses(
        scene.candidates,
        scene.target_mask,
        scene.evidence,
        weights={"visual": 1.0, "failure": 0.0, "depth": 0.0, "prior": 1.0},
    )
    full_model = rank_hypotheses(scene.candidates, scene.target_mask, scene.evidence)
    failure_only = rank_hypotheses(
        scene.candidates,
        scene.target_mask,
        scene.evidence,
        weights={"visual": 0.0, "failure": 1.0, "depth": 0.0, "prior": 0.0},
    )

    assert silhouette_only[0].hypothesis.hypothesis_id != scene.ground_truth_id
    assert failure_only[0].hypothesis.hypothesis_id == scene.ground_truth_id
    assert full_model[0].hypothesis.hypothesis_id == scene.ground_truth_id
    assert full_model[0].score.failure > full_model[1].score.failure


def test_minimal_m1_benchmark_reports_failure_aware_gain():
    result = run_minimal_m1_benchmark()

    assert result["num_scenes"] == 2
    assert result["silhouette_only_top1"] < result["full_model_top1"]
    assert result["full_model_top1"] == 2
    assert result["abstain_count"] == 0
    assert set(result) == {"num_scenes", "silhouette_only_top1", "full_model_top1", "abstain_count"}
