import importlib.util
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "generate_m7_green_cylinder_grasp_gate.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_m7_green_cylinder_grasp_gate",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _valid_decision(path: Path):
    return _write_json(
        path,
        {
            "recommended_backend": "normal_rgbd",
            "target_label": "green_cylinder",
            "ready_for_shadow_planning": True,
            "depth_failure_reasons": [],
            "caution_reasons": ["contact_shadow_leakage_but_depth_usable"],
        },
    )


def _valid_target(path: Path):
    return _write_json(
        path,
        {
            "target": {
                "shape_type": "cylinder",
                "valid": True,
                "center_x_m": 0.0,
                "center_y_m": 0.24,
                "center_z_m": 0.035,
                "required_gripper_width_m": 0.043,
            }
        },
    )


def _valid_moveit_plan(path: Path, target_angles_deg: list[float]):
    return _write_json(
        path,
        {
            "summary": {"all_planned": True},
            "rows": [
                {
                    "planned": True,
                    "descent_clearance": {"status": "ok"},
                    "attempts": [
                        {
                            "final_joint_positions": [
                                math.radians(value) for value in target_angles_deg
                            ]
                        }
                    ],
                }
            ],
        },
    )


def test_green_cylinder_gate_blocks_large_pregrasp_joint_delta(tmp_path):
    module = _load_module()
    decision = _write_json(
        tmp_path / "decision.json",
        {
            "recommended_backend": "normal_rgbd",
            "target_label": "green_cylinder",
            "ready_for_shadow_planning": True,
            "depth_failure_reasons": [],
            "caution_reasons": ["contact_shadow_leakage_but_depth_usable"],
        },
    )
    target = _write_json(
        tmp_path / "target.json",
        {
            "target": {
                "shape_type": "cylinder",
                "valid": True,
                "center_x_m": 0.0,
                "center_y_m": 0.24,
                "center_z_m": 0.035,
                "required_gripper_width_m": 0.043,
            }
        },
    )
    moveit_plan = _write_json(
        tmp_path / "plan.json",
        {
            "summary": {"all_planned": True},
            "rows": [
                {
                    "planned": True,
                    "descent_clearance": {"status": "ok"},
                    "attempts": [
                        {
                            "final_joint_positions": [
                                0.0,
                                0.0,
                                -2.5,
                                0.0,
                                0.0,
                                0.0,
                            ]
                        }
                    ],
                }
            ],
        },
    )
    current = _write_json(
        tmp_path / "current.json",
        {"angles_deg": [0.0, 0.0, 80.0, 0.0, 0.0, 0.0]},
    )

    report = module.generate_gate(
        decision_path=decision,
        target_path=target,
        moveit_plan_path=moveit_plan,
        current_state_path=current,
        output_dir=tmp_path / "gate",
        execute_requested=True,
        operator_phrase=module.REQUIRED_OPERATOR_PHRASE,
        max_joint_delta_deg=60.0,
    )

    assert report["overall_status"] == "blocked"
    assert report["motion_authorized"] is False
    assert "pregrasp_joint_delta_too_large" in report["blockers"]
    assert report["joint_delta"]["max_abs_delta_deg"] > 60.0


def test_green_cylinder_gate_shell_is_non_actuating():
    source = (REPO_ROOT / "scripts" / "run_m7_green_cylinder_grasp_gate.sh").read_text()

    assert "generate_m7_green_cylinder_grasp_gate.py" in source
    assert "get_angles" in source
    assert "get_coords" in source
    assert "send_angles" not in source
    assert "send_coords" not in source
    assert "set_gripper" not in source


def test_green_cylinder_gate_blocks_current_to_home_joint_delta(tmp_path):
    module = _load_module()
    decision = _valid_decision(tmp_path / "decision.json")
    target = _valid_target(tmp_path / "target.json")
    moveit_plan = _valid_moveit_plan(tmp_path / "plan.json", [1.0, -22.0, -68.0, 0.0, 0.0, 0.0])
    current = _write_json(
        tmp_path / "current.json",
        {"angles_deg": [120.0, 30.0, -70.0, 0.0, 0.0, 0.0]},
    )

    report = module.generate_gate(
        decision_path=decision,
        target_path=target,
        moveit_plan_path=moveit_plan,
        current_state_path=current,
        output_dir=tmp_path / "gate",
        execute_requested=True,
        operator_phrase=module.REQUIRED_OPERATOR_PHRASE,
        max_joint_delta_deg=60.0,
        max_home_to_pregrasp_delta_deg=70.0,
    )

    assert report["overall_status"] == "blocked"
    assert "current_to_home_joint_delta_too_large" in report["blockers"]
    assert report["joint_delta"]["current_to_home"]["max_abs_delta_deg"] == 120.0


def test_green_cylinder_gate_blocks_home_to_pregrasp_joint_delta(tmp_path):
    module = _load_module()
    decision = _valid_decision(tmp_path / "decision.json")
    target = _valid_target(tmp_path / "target.json")
    moveit_plan = _valid_moveit_plan(tmp_path / "plan.json", [100.0, -22.0, -68.0, 0.0, 0.0, 0.0])
    current = _write_json(
        tmp_path / "current.json",
        {"angles_deg": [0.0, -21.0, -70.0, 0.0, 0.0, 0.0]},
    )

    report = module.generate_gate(
        decision_path=decision,
        target_path=target,
        moveit_plan_path=moveit_plan,
        current_state_path=current,
        output_dir=tmp_path / "gate",
        execute_requested=True,
        operator_phrase=module.REQUIRED_OPERATOR_PHRASE,
        max_joint_delta_deg=60.0,
        max_home_to_pregrasp_delta_deg=70.0,
    )

    assert report["overall_status"] == "blocked"
    assert "home_to_pregrasp_joint_delta_too_large" in report["blockers"]
    assert report["joint_delta"]["home_to_pregrasp"]["max_abs_delta_deg"] == 100.0
