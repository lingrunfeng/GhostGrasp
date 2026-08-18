import importlib.util
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "src" / "ghost_mgg_real" / "scripts" / "m7_real_grasp_marker_node.py"


def load_marker_module():
    spec = importlib.util.spec_from_file_location("m7_real_grasp_marker_node", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_yaw_to_quaternion_z_matches_planar_rotation():
    module = load_marker_module()

    quat = module.yaw_to_quaternion_z(math.pi / 2.0)

    assert quat["x"] == 0.0
    assert quat["y"] == 0.0
    assert abs(quat["z"] - math.sqrt(0.5)) < 1e-9
    assert abs(quat["w"] - math.sqrt(0.5)) < 1e-9


def test_marker_source_uses_target_yaw_for_body_orientation():
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "yaw_to_quaternion_z(float(target.get(\"yaw_rad\", 0.0)))" in source
    assert "body.pose.orientation.w = 1.0" not in source
