import importlib.util
from pathlib import Path


SIM_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SIM_DIR / "scripts" / "m4_perception_hypothesis_publisher_node.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "m4_perception_hypothesis_publisher_node", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_reports():
    joint_report = {
        "schema_version": "m4_joint_hypothesis_report_v1",
        "rows": [
            {
                "source_type": "real_graspability",
                "rank_group": "real:scene_a",
                "target_or_scene_id": "scene_a",
                "ranker": "failure_aware",
                "hypothesis_id": "box_s1.00",
                "shape_type": "box",
                "decision": "candidate",
                "joint_score": 0.91,
                "visual_score": 0.82,
                "grasp_score": 0.77,
                "failure_reason": "",
                "joint_rank": 1,
            },
            {
                "source_type": "real_graspability",
                "rank_group": "real:scene_a",
                "target_or_scene_id": "scene_a",
                "ranker": "silhouette_only",
                "hypothesis_id": "cylinder_s1.00",
                "shape_type": "cylinder",
                "decision": "reject",
                "joint_score": 0.31,
                "visual_score": 0.71,
                "grasp_score": 0.30,
                "failure_reason": "gripper_width",
                "joint_rank": 2,
            },
            {
                "source_type": "real_graspability",
                "rank_group": "real:scene_b",
                "target_or_scene_id": "scene_b",
                "ranker": "failure_aware",
                "hypothesis_id": "box_s1.10",
                "shape_type": "box",
                "decision": "candidate",
                "joint_score": 0.88,
                "visual_score": 0.80,
                "grasp_score": 0.70,
                "failure_reason": "",
                "joint_rank": 1,
            },
            {
                "source_type": "sim_moveit",
                "rank_group": "sim:m4_tabletop",
                "target_or_scene_id": "glass_block",
                "ranker": "moveit_dryrun",
                "hypothesis_id": "glass_block",
                "shape_type": "box",
                "decision": "executable",
                "joint_score": 1.0,
                "joint_rank": 1,
            },
        ],
    }
    metric_report = {
        "schema_version": "m4_metric_proxies_v1",
        "rows": [
            {
                "scene_id": "scene_a",
                "target_label": "jelly",
                "ranker": "failure_aware",
                "rank": 1,
                "hypothesis_id": "box_s1.00",
                "shape_type": "box",
                "center_x_m": 0.12,
                "center_y_m": -0.04,
                "center_z_m": 0.78,
                "width_m": 0.045,
                "depth_m": 0.038,
                "height_m": 0.060,
                "table_z_m": 0.75,
                "table_depth_m": 1.05,
            },
            {
                "scene_id": "scene_a",
                "target_label": "jelly",
                "ranker": "silhouette_only",
                "rank": 1,
                "hypothesis_id": "cylinder_s1.00",
                "shape_type": "cylinder",
                "center_x_m": 0.11,
                "center_y_m": -0.05,
                "center_z_m": 0.77,
                "width_m": 0.080,
                "depth_m": 0.080,
                "height_m": 0.040,
                "table_z_m": 0.75,
                "table_depth_m": 1.05,
            },
        ],
    }
    graspability_report = {
        "schema_version": "m4_graspability_dryrun_v1",
        "rows": [
            {
                "scene_id": "scene_a",
                "ranker": "failure_aware",
                "hypothesis_id": "box_s1.00",
                "shape_type": "box",
                "grasp_id": "box_s1.00_top_dryrun",
                "grasp_type": "top",
                "grasp_x_m": 0.12,
                "grasp_y_m": -0.04,
                "grasp_z_m": 0.815,
                "pregrasp_z_m": 0.895,
                "approach_x": 0.0,
                "approach_y": 0.0,
                "approach_z": -1.0,
                "required_gripper_width_m": 0.050,
                "score": 0.77,
                "valid": True,
                "failure_reason": "",
            },
            {
                "scene_id": "scene_a",
                "ranker": "silhouette_only",
                "hypothesis_id": "cylinder_s1.00",
                "shape_type": "cylinder",
                "grasp_id": "cylinder_s1.00_top_dryrun",
                "grasp_type": "top",
                "grasp_x_m": 0.11,
                "grasp_y_m": -0.05,
                "grasp_z_m": 0.795,
                "pregrasp_z_m": 0.875,
                "approach_x": 0.0,
                "approach_y": 0.0,
                "approach_z": -1.0,
                "required_gripper_width_m": 0.092,
                "score": 0.30,
                "valid": False,
                "failure_reason": "gripper_width",
            },
        ],
    }
    return joint_report, metric_report, graspability_report


def test_build_perception_hypotheses_filters_real_scene_and_skips_rejected_rows():
    module = load_module()
    joint_report, metric_report, graspability_report = sample_reports()

    hypotheses = module.build_perception_hypotheses(
        joint_report,
        metric_report,
        graspability_report,
        scene_id="scene_a",
        frame_id="world",
    )

    assert [hypothesis.hypothesis_id for hypothesis in hypotheses] == ["box_s1.00"]
    hypothesis = hypotheses[0]
    assert hypothesis.shape_type == module.GeometryHypothesis.SHAPE_BOX
    assert hypothesis.validation_state == module.GeometryHypothesis.VALIDATION_VALID
    assert hypothesis.pose_base.header.frame_id == "world"
    assert hypothesis.pose_base.pose.position.x == 0.12
    assert hypothesis.pose_base.pose.position.y == -0.04
    assert hypothesis.pose_base.pose.position.z == 0.78
    assert hypothesis.dimensions_m.x == 0.045
    assert hypothesis.dimensions_m.y == 0.038
    assert hypothesis.dimensions_m.z == 0.060
    assert hypothesis.score.visual == 0.82
    assert hypothesis.score.grasp == 0.77
    assert hypothesis.score.total == 0.91
    assert hypothesis.confidence == 0.91
    assert hypothesis.provenance == "m4_perception:scene_a:failure_aware:rank=1"

    grasp = hypothesis.grasp_candidates[0]
    assert grasp.grasp_id == "box_s1.00_top_dryrun"
    assert grasp.grasp_type == module.GraspCandidate.GRASP_TYPE_TOP
    assert grasp.validation_state == module.GraspCandidate.VALIDATION_VALID
    assert grasp.grasp_pose.pose.position.z == 0.815
    assert grasp.pregrasp_pose.pose.position.z == 0.895
    assert grasp.approach_vector.z == -1.0
    assert grasp.gripper_width_m == 0.050


def test_build_perception_hypotheses_can_include_rejected_rows():
    module = load_module()
    joint_report, metric_report, graspability_report = sample_reports()

    hypotheses = module.build_perception_hypotheses(
        joint_report,
        metric_report,
        graspability_report,
        scene_id="scene_a",
        frame_id="world",
        include_rejected=True,
    )

    assert [hypothesis.hypothesis_id for hypothesis in hypotheses] == [
        "box_s1.00",
        "cylinder_s1.00",
    ]
    rejected = hypotheses[1]
    assert rejected.shape_type == module.GeometryHypothesis.SHAPE_CYLINDER
    assert rejected.validation_state == module.GeometryHypothesis.VALIDATION_REJECTED
    assert rejected.grasp_candidates[0].validation_state == module.GraspCandidate.VALIDATION_REJECTED
    assert rejected.failure_reason == "gripper_width"


def test_build_message_populates_contract_fields():
    module = load_module()
    joint_report, metric_report, graspability_report = sample_reports()

    message = module.build_hypothesis_array(
        joint_report,
        metric_report,
        graspability_report,
        scene_id="scene_a",
        frame_id="world",
        trial_id="trial_1",
        observation_id="obs_1",
        backend_name="ghost_mgg_m4_perception",
    )

    assert message.header.frame_id == "world"
    assert message.trial_id == "trial_1"
    assert message.observation_id == "obs_1"
    assert message.backend_name == "ghost_mgg_m4_perception"
    assert len(message.hypotheses) == 1
