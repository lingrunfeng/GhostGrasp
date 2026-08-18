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


def _write_preflight(path: Path, *, ready: bool) -> None:
    _write_json(
        path,
        {
            "schema_version": "m7_safety_preflight_v1",
            "overall_status": "ready_for_operator_controlled_m7_1" if ready else "blocked",
            "motion_authorized": False,
            "required_operator_phrase": "确认进入 M7.1，允许真实低幅度空载运动；机械臂周围已清空，我已准备好断电/急停。",
            "limits": {
                "max_joint_delta_deg": 2.0,
                "max_speed": 5,
                "allowed_joint_scope": "single distal wrist joint only",
            },
            "blockers": [] if ready else ["operator_authorization_missing"],
        },
    )


def test_m7_low_amplitude_default_generates_dry_run_plan_without_motion(tmp_path):
    module = _load_script("run_m7_low_amplitude_motion_test")
    preflight_path = tmp_path / "m7_safety_preflight.json"
    _write_preflight(preflight_path, ready=False)

    report = module.generate_m7_low_amplitude_motion_test(
        preflight_path=preflight_path,
        output_dir=tmp_path / "motion",
        execute=False,
    )

    assert report["schema_version"] == "m7_low_amplitude_motion_test_v1"
    assert report["overall_status"] == "dry_run_only"
    assert report["motion_authorized"] is False
    assert report["commanded_motion"]["joint_number"] == 6
    assert report["commanded_motion"]["delta_deg"] <= 2.0
    assert report["commanded_motion"]["speed"] <= 5
    assert report["readback"]["available"] is False
    assert (tmp_path / "motion" / "m7_low_amplitude_motion_test.json").exists()
    assert "M7 Low-Amplitude Motion Test" in (tmp_path / "motion" / "index.md").read_text()


def test_m7_low_amplitude_execute_blocks_without_exact_operator_phrase(tmp_path):
    module = _load_script("run_m7_low_amplitude_motion_test")
    preflight_path = tmp_path / "m7_safety_preflight.json"
    _write_preflight(preflight_path, ready=True)

    report = module.generate_m7_low_amplitude_motion_test(
        preflight_path=preflight_path,
        output_dir=tmp_path / "motion",
        execute=True,
        operator_phrase="wrong phrase",
    )

    assert report["overall_status"] == "blocked"
    assert report["motion_authorized"] is False
    assert "operator_phrase_mismatch" in report["blockers"]
    assert report["readback"]["available"] is False


def test_m7_motion_helpers_enforce_limits_and_record_return_error():
    module = _load_script("run_m7_low_amplitude_motion_test")
    start = [-21.0, 1.0, 1.0, -144.0, 70.0, -1.0]

    target = module.build_target_angles(start, joint_number=6, delta_deg=2.0)
    assert target[:5] == start[:5]
    assert target[5] == 1.0

    readback = module.build_readback_summary(
        start_angles=start,
        target_angles=target,
        after_target_angles=[-21.0, 1.0, 1.0, -144.0, 70.0, 0.7],
        final_angles=[-21.0, 1.0, 1.0, -144.0, 70.0, -0.3],
        return_tolerance_deg=1.0,
    )
    assert readback["available"] is True
    assert readback["after_target_j6_delta_from_start_deg"] == 1.7
    assert readback["final_max_abs_delta_from_start_deg"] == 0.7
    assert readback["returned_within_tolerance"] is True

    for kwargs in [
        {"joint_number": 5, "delta_deg": 1.0, "speed": 5},
        {"joint_number": 6, "delta_deg": 2.1, "speed": 5},
        {"joint_number": 6, "delta_deg": 1.0, "speed": 6},
    ]:
        try:
            module.validate_motion_limits(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected limit rejection for {kwargs}")


def test_m7_low_amplitude_shell_defaults_to_dry_run_and_requires_explicit_execute_flag():
    script = (REPO_ROOT / "scripts" / "run_m7_low_amplitude_motion_test.sh").read_text(
        encoding="utf-8"
    )

    for required in [
        "run_m7_safety_preflight.sh",
        "run_m7_low_amplitude_motion_test.py",
        "M7_EXECUTE_REAL_MOTION",
        "M7_OPERATOR_PHRASE",
        "reports/m7_low_amplitude_motion_test",
    ]:
        assert required in script

    assert "--execute" in script
    assert 'M7_EXECUTE_REAL_MOTION:-0' in script
    assert "send_angles" not in script
    assert "send_coords" not in script
    assert "ros2 action send_goal" not in script
