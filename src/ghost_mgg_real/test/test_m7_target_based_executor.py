import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "run_m7_target_based_executor.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_m7_target_based_executor", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _ready_gate() -> dict:
    return {
        "overall_status": "ready_for_separate_real_execute",
        "blockers": [],
        "joint_delta": {
            "available": True,
            "current_angles_deg": [0.0, -30.0, 40.0, 0.0, 10.0, 20.0],
            "pregrasp_target_angles_deg": [1.0, -32.0, 42.0, 1.0, 11.0, 21.0],
            "max_abs_delta_deg": 2.0,
        },
    }


def _planned_moveit() -> dict:
    return {
        "summary": {"all_planned": True},
        "rows": [
            {
                "planned": True,
                "pregrasp_z_m": 0.15,
                "grasp_z_m": 0.055,
                "descent_clearance": {"status": "ok"},
                "attempts": [
                    {
                        "final_joint_positions": [
                            0.0174532925,
                            -0.5585053606,
                            0.7330382858,
                            0.0174532925,
                            0.1919862177,
                            0.3665191429,
                        ]
                    }
                ],
            }
        ],
    }


def test_target_based_executor_blocks_gate_blockers(tmp_path):
    module = _load_module()
    gate = _write_json(
        tmp_path / "gate.json",
        {
            **_ready_gate(),
            "overall_status": "blocked",
            "blockers": ["pregrasp_joint_delta_too_large"],
        },
    )
    moveit = _write_json(tmp_path / "moveit.json", _planned_moveit())

    report = module.generate_target_based_execution_report(
        gate_path=gate,
        moveit_plan_path=moveit,
        output_dir=tmp_path / "out",
        execute=True,
        operator_phrase=module.REQUIRED_OPERATOR_PHRASE,
        target_based_confirmed=True,
        process_table="",
        execute_remote_fn=lambda **_: {"available": True},
    )

    assert report["overall_status"] == "blocked"
    assert report["motion_authorized"] is False
    assert "gate_not_ready" in report["blockers"]
    assert "pregrasp_joint_delta_too_large" in report["blockers"]


def test_target_based_executor_blocks_local_shadow_bridge_process(tmp_path):
    module = _load_module()
    gate = _write_json(tmp_path / "gate.json", _ready_gate())
    moveit = _write_json(tmp_path / "moveit.json", _planned_moveit())

    report = module.generate_target_based_execution_report(
        gate_path=gate,
        moveit_plan_path=moveit,
        output_dir=tmp_path / "out",
        execute=True,
        operator_phrase=module.REQUIRED_OPERATOR_PHRASE,
        target_based_confirmed=True,
        process_table="123 python3 m6_ssh_joint_state_bridge.py",
        execute_remote_fn=lambda **_: {"available": True},
    )

    assert report["overall_status"] == "blocked"
    assert report["motion_authorized"] is False
    assert "local_shadow_bridge_running" in report["blockers"]


def test_target_based_executor_dry_run_computes_pregrasp_and_descent(tmp_path):
    module = _load_module()
    gate = _write_json(tmp_path / "gate.json", _ready_gate())
    moveit = _write_json(tmp_path / "moveit.json", _planned_moveit())

    report = module.generate_target_based_execution_report(
        gate_path=gate,
        moveit_plan_path=moveit,
        output_dir=tmp_path / "out",
        execute=False,
        operator_phrase="",
        target_based_confirmed=False,
        process_table="",
        execute_remote_fn=lambda **_: {"available": True},
    )

    assert report["overall_status"] == "dry_run_ready"
    assert report["motion_authorized"] is False
    assert report["commanded_motion"]["descend_mm"] == 95.0
    assert report["commanded_motion"]["pregrasp_angles_deg"] == [1.0, -32.0, 42.0, 1.0, 11.0, 21.0]


def test_target_based_executor_remote_script_contains_full_sequence():
    module = _load_module()

    script = module.build_remote_target_based_script(
        serial_port="/dev/ttyAMA0",
        baud=1000000,
        pregrasp_angles_deg=[1.0, -32.0, 42.0, 1.0, 11.0, 21.0],
        descend_mm=30.0,
        lift_mm=35.0,
        speed=5,
        gripper_speed=20,
        pregrasp_settle_sec=2.0,
        settle_sec=2.0,
        gripper_settle_sec=1.0,
        min_z_mm=120.0,
    )

    assert "mc.send_angles(pregrasp_angles_deg, speed)" in script
    assert "mc.send_coords(grasp_coords, speed, 1)" in script
    assert "mc.set_gripper_state(1, gripper_speed)" in script
    assert "mc.send_coords(lift_coords, speed, 1)" in script


def test_target_based_executor_remote_script_moves_home_before_pregrasp():
    module = _load_module()

    script = module.build_remote_target_based_script(
        serial_port="/dev/ttyAMA0",
        baud=1000000,
        home_angles_deg=[0.0, 30.0, -70.0, 0.0, 0.0, 0.0],
        pregrasp_angles_deg=[1.0, -32.0, 42.0, 1.0, 11.0, 21.0],
        descend_mm=70.0,
        lift_mm=35.0,
        speed=3,
        gripper_speed=20,
        home_settle_sec=3.0,
        pregrasp_settle_sec=3.0,
        settle_sec=2.0,
        gripper_settle_sec=1.0,
        min_z_mm=120.0,
    )

    assert "mc.send_angles(home_angles_deg, speed)" in script
    assert script.index("mc.send_angles(home_angles_deg, speed)") < script.index(
        "mc.send_angles(pregrasp_angles_deg, speed)"
    )
    assert "mc.send_coords(grasp_coords, speed, 1)" in script
    assert "mc.set_gripper_state(1, gripper_speed)" in script
    assert "mc.send_coords(lift_coords, speed, 1)" in script
