import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "src" / "ghost_mgg_real" / "scripts"


def _load_script(name: str):
    script_path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_m6_scene(root: Path, observation_id: str, backend: str, planned: bool) -> None:
    _write_json(
        root / "observations" / observation_id / "m6_shadow_observation.json",
        {
            "schema_version": "m6_shadow_observation_v1",
            "observation_id": observation_id,
            "safety_mode": "shadow_only_no_motion",
            "gate_checks": {
                "has_snapshot": True,
                "has_real_arm_joints": True,
                "has_camera_to_base_tf": True,
                "has_aligned_depth_raw": True,
            },
        },
    )
    _write_json(
        root / "decisions" / observation_id / "m6_shadow_decision.json",
        {
            "schema_version": "m6_shadow_decision_v1",
            "observation_id": observation_id,
            "motion_authorized": False,
            "recommended_backend": backend,
            "ready_for_shadow_planning": True,
            "depth_failure_reasons": ["target_hole_ratio_high"] if backend == "ghost_mgg" else [],
            "reject_reasons": [],
        },
    )
    _write_json(
        root / "targets" / observation_id / "m6_shadow_grasp_target.json",
        {
            "schema_version": "m6_shadow_grasp_target_v1",
            "observation_id": observation_id,
            "motion_authorized": False,
            "target": {"target_id": observation_id, "valid": True, "grasp_type": "top_grasp"},
        },
    )
    _write_json(
        root / "targets" / observation_id / "moveit_plan_only_shadow_allowlist.json",
        {
            "schema_version": "m4_sim_moveit_dryrun_v1",
            "summary": {"planned": 1 if planned else 0, "total": 1, "all_planned": planned},
        },
    )


def test_m6_shadow_freeze_report_requires_normal_and_ghost_shadow_scenes(tmp_path):
    module = _load_script("generate_m6_shadow_freeze_report")
    _write_m6_scene(tmp_path, "green_scene", "normal_rgbd", True)
    _write_m6_scene(tmp_path, "jelly_scene", "ghost_mgg", True)
    readiness_path = tmp_path / "readiness" / "m6_shadow_readiness.json"
    _write_json(readiness_path, {"schema_version": "m6_shadow_readiness_v1", "overall_status": "pass"})

    report = module.generate_m6_shadow_freeze_report(
        output_dir=tmp_path / "freeze",
        scene_ids=["green_scene", "jelly_scene"],
        expected_backends={"green_scene": "normal_rgbd", "jelly_scene": "ghost_mgg"},
        observations_root=tmp_path / "observations",
        decisions_root=tmp_path / "decisions",
        targets_root=tmp_path / "targets",
        readiness_path=readiness_path,
    )

    assert report["overall_status"] == "pass"
    assert report["motion_authorized"] is False
    assert report["backend_counts"] == {"ghost_mgg": 1, "normal_rgbd": 1}
    assert all(scene["moveit_all_planned"] for scene in report["scenes"])
    assert all(scene["motion_authorized"] is False for scene in report["scenes"])
    assert (tmp_path / "freeze" / "m6_shadow_freeze.json").exists()
    assert "M6 Shadow Freeze" in (tmp_path / "freeze" / "index.md").read_text()


def test_m7_safety_preflight_blocks_without_operator_authorization(tmp_path):
    module = _load_script("generate_m7_safety_preflight")
    freeze_path = tmp_path / "freeze" / "m6_shadow_freeze.json"
    _write_json(
        freeze_path,
        {
            "schema_version": "m6_shadow_freeze_v1",
            "overall_status": "pass",
            "motion_authorized": False,
            "scene_count": 2,
            "backend_counts": {"ghost_mgg": 1, "normal_rgbd": 1},
        },
    )

    report = module.generate_m7_safety_preflight(
        m6_freeze_path=freeze_path,
        output_dir=tmp_path / "preflight",
        operator_authorized=False,
    )

    assert report["overall_status"] == "blocked"
    assert report["motion_authorized"] is False
    assert "operator_authorization_missing" in report["blockers"]
    assert report["limits"]["max_joint_delta_deg"] <= 2.0
    assert report["limits"]["max_speed"] <= 5
    assert "确认进入 M7.1" in report["required_operator_phrase"]
    assert (tmp_path / "preflight" / "m7_safety_preflight.json").exists()
    assert "M7 Safety Preflight" in (tmp_path / "preflight" / "index.md").read_text()


def test_m6_freeze_and_m7_preflight_shells_are_non_motion_contracts():
    freeze_source = (REPO_ROOT / "scripts" / "run_m6_shadow_freeze_report.sh").read_text(
        encoding="utf-8"
    )
    preflight_source = (REPO_ROOT / "scripts" / "run_m7_safety_preflight.sh").read_text(
        encoding="utf-8"
    )
    combined = freeze_source + "\n" + preflight_source

    for required in [
        "m6_shadow_live_003",
        "m6_shadow_jelly_live_001",
        "generate_m6_shadow_freeze_report.py",
        "generate_m7_safety_preflight.py",
        "M7_OPERATOR_AUTHORIZED",
        "reports/m7_safety_preflight",
    ]:
        assert required in combined

    for forbidden in [
        "send_angles",
        "send_coords",
        "ros2 action send_goal",
        "FollowJointTrajectory",
        "moveit_sim_execute_server",
        "execute_m4_joint_hypothesis_action.py",
    ]:
        assert forbidden not in combined
