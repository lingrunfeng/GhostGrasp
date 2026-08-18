import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "run_m7_current_pregrasp_grasp.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_m7_current_pregrasp_grasp", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_pregrasp_limits_reject_large_descent():
    module = _load_module()

    with pytest.raises(ValueError, match="descend_mm"):
        module.validate_motion_limits(
            descend_mm=40.0,
            lift_mm=20.0,
            speed=5,
            gripper_speed=20,
            min_z_mm=120.0,
        )


def test_current_pregrasp_report_blocks_execute_without_confirmation(tmp_path):
    module = _load_module()

    report = module.generate_current_pregrasp_grasp_report(
        output_dir=tmp_path / "report",
        execute=True,
        operator_phrase=module.REQUIRED_OPERATOR_PHRASE,
        current_pregrasp_confirmed=False,
    )

    assert report["overall_status"] == "blocked"
    assert report["motion_authorized"] is False
    assert "current_pregrasp_not_confirmed" in report["blockers"]


def test_current_pregrasp_shell_defaults_to_dry_run_and_requires_explicit_flags():
    source = (REPO_ROOT / "scripts" / "run_m7_current_pregrasp_grasp.sh").read_text()

    assert "M7_EXECUTE_REAL_GRASP:-0" in source
    assert "M7_CURRENT_PREGRASP_CONFIRMED:-0" in source
    assert "--execute" in source
    assert "--current-pregrasp-confirmed" in source
