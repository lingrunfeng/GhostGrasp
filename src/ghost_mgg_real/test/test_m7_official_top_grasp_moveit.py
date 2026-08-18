import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "src" / "ghost_mgg_real" / "scripts"


def _load_module(name: str):
    script_path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_top_grasp_template_uses_vertical_pregrasp_pattern():
    module = _load_module("generate_m7_official_top_grasp_targets")

    template = module.official_top_grasp_template()

    assert template["template_id"] == "elephant_official_top_grasp_v1"
    assert template["approach_axis_base"] == [0.0, 0.0, -1.0]
    assert template["pregrasp_clearance_m"] == 0.07
    assert template["retreat_lift_m"] == 0.07
    assert template["moveit_execution_allowed"] is False
    assert template["official_reference_pose_mm_deg"] == [-177.5, 1.91, 173.49]


def test_build_seed_targets_writes_moveit_compatible_synthetic_targets(tmp_path):
    module = _load_module("generate_m7_official_top_grasp_targets")
    output_dir = tmp_path / "m7_official_top_grasp"

    report = module.generate_official_top_grasp_seed_targets(
        output_dir=output_dir,
        table_top_z_m=-0.0127,
        seed_target_specs=[
            "seed_green:0.005:0.236:cylinder:0.020:0.025:0.0",
            "seed_box:-0.025:0.220:box:0.035:0.025:0.040:0.35",
        ],
    )

    targets_path = output_dir / "m7_official_top_grasp_seed_targets.json"
    targets = json.loads(targets_path.read_text(encoding="utf-8"))

    assert report["schema_version"] == "m7_official_top_grasp_seed_targets_v1"
    assert report["safety_mode"] == "moveit_plan_only_no_hardware_motion"
    assert report["motion_authorized"] is False
    assert targets["schema_version"] == "m4_sim_grasp_targets_v1"
    assert [row["target_id"] for row in targets["rows"]] == ["seed_green", "seed_box"]
    assert targets["rows"][0]["shape_type"] == "cylinder"
    assert targets["rows"][0]["center_z_m"] == -0.0002
    assert targets["rows"][0]["grasp_type"] == "top_grasp"
    assert targets["rows"][0]["pregrasp_clearance_m"] == 0.07
    assert targets["rows"][0]["synthetic_source"] == "manual_seed_not_real_perception"
    assert targets["rows"][1]["shape_type"] == "box"
    assert targets["rows"][1]["size_z_m"] == 0.04
    assert (output_dir / "index.md").exists()


def test_m7_official_top_grasp_moveit_script_is_plan_only():
    script_path = REPO_ROOT / "scripts" / "run_m7_official_top_grasp_moveit_plan_only.sh"
    script_text = script_path.read_text(encoding="utf-8")

    for required in [
        "generate_m7_official_top_grasp_targets.py",
        "probe_m4_sim_grasp_moveit.py",
        "--planning-frame base_link",
        "--check-grasp-stage",
        "m7_official_top_grasp_moveit_plan_only.json",
        "moveit_plan_only_no_hardware_motion",
    ]:
        assert required in script_text

    for forbidden in [
        "send_angles",
        "send_coords",
        "set_gripper_state",
        "FollowJointTrajectory",
        "moveit_sim_execute_server",
        "run_m7_target_based_executor.py --execute",
    ]:
        assert forbidden not in script_text


def test_m7_real_state_official_top_grasp_moveit_script_reads_joints_only():
    script_path = REPO_ROOT / "scripts" / "run_m7_real_state_official_top_grasp_moveit_plan_only.sh"
    script_text = script_path.read_text(encoding="utf-8")

    for required in [
        "m6_mycobot_state_bridge.launch.py",
        "use_fake_joint_states:=false",
        "generate_m7_official_top_grasp_targets.py",
        "probe_m4_sim_grasp_moveit.py",
        "--planning-frame base_link",
        "--check-grasp-stage",
        "real_joint_state_once.yaml",
        "m7_real_state_official_top_grasp_moveit_plan_only.json",
    ]:
        assert required in script_text

    for forbidden in [
        "send_angles",
        "send_coords",
        "set_gripper_state",
        "FollowJointTrajectory",
        "moveit_sim_execute_server",
        "run_m7_target_based_executor.py --execute",
    ]:
        assert forbidden not in script_text
