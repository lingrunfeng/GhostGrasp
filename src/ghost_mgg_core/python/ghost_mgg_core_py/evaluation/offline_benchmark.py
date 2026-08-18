from ghost_mgg_core_py.evaluation.synthetic_scene import make_failure_ranking_scene
from ghost_mgg_core_py.scoring.joint_ranker import rank_hypotheses


SILHOUETTE_ONLY_WEIGHTS = {"visual": 1.0, "failure": 0.0, "depth": 0.0, "prior": 1.0}


def run_minimal_m1_benchmark() -> dict[str, int]:
    scenes = (
        make_failure_ranking_scene("scene_shift_1", 1),
        make_failure_ranking_scene("scene_shift_2", 2),
    )

    silhouette_only_top1 = 0
    full_model_top1 = 0
    abstain_count = 0

    for scene in scenes:
        silhouette_only = rank_hypotheses(
            scene.candidates,
            scene.target_mask,
            scene.evidence,
            weights=SILHOUETTE_ONLY_WEIGHTS,
        )
        full_model = rank_hypotheses(scene.candidates, scene.target_mask, scene.evidence)

        if silhouette_only[0].hypothesis.hypothesis_id == scene.ground_truth_id:
            silhouette_only_top1 += 1
        if full_model[0].hypothesis.hypothesis_id == scene.ground_truth_id:
            full_model_top1 += 1
        abstain_count += int(full_model[0].validation_state == "rejected")

    return {
        "num_scenes": len(scenes),
        "silhouette_only_top1": silhouette_only_top1,
        "full_model_top1": full_model_top1,
        "abstain_count": abstain_count,
    }
