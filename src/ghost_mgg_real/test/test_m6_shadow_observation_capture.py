import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "capture_m6_shadow_observation.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("capture_m6_shadow_observation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_shadow_observation_report_records_snapshot_joints_and_tf(tmp_path):
    module = load_module()
    output_dir = tmp_path / "m6_shadow_observation"
    snapshot_manifest = {
        "observation_id": "m6_shadow_test_001",
        "copied_files": ["color.png", "aligned_depth_raw.npy"],
        "missing_topics": [],
    }
    joint_state = {
        "name": [
            "link1_to_link2",
            "link2_to_link3",
            "link3_to_link4",
            "link4_to_link5",
            "link5_to_link6",
            "link6_to_link6_flange",
        ],
        "position": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
        "stamp_sec": 12,
        "stamp_nanosec": 34,
    }
    camera_to_base = {
        "parent_frame": "base_link",
        "child_frame": "camera_link",
        "translation": {"x": 0.0, "y": 0.39, "z": 0.16},
        "rotation_quat": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }

    report = module.write_shadow_observation_report(
        observation_id="m6_shadow_test_001",
        output_dir=output_dir,
        snapshot_dir=output_dir / "snapshot",
        snapshot_manifest=snapshot_manifest,
        joint_state=joint_state,
        camera_to_base=camera_to_base,
    )

    assert report["schema_version"] == "m6_shadow_observation_v1"
    assert report["safety_mode"] == "shadow_only_no_motion"
    assert report["observation_id"] == "m6_shadow_test_001"
    assert report["snapshot"]["manifest"]["copied_files"] == [
        "color.png",
        "aligned_depth_raw.npy",
    ]
    assert report["joint_state"]["name"][0] == "link1_to_link2"
    assert report["camera_to_base"]["translation"]["y"] == 0.39
    assert report["gate_checks"]["has_real_arm_joints"] is True
    assert report["gate_checks"]["has_camera_to_base_tf"] is True
    assert (output_dir / "m6_shadow_observation.json").exists()
    assert (output_dir / "index.md").exists()


def test_shadow_observation_scripts_are_read_only_and_documented():
    capture_script = SCRIPT_PATH.read_text(encoding="utf-8")
    shell_script = (REPO_ROOT / "scripts" / "capture_m6_shadow_observation.sh").read_text(
        encoding="utf-8"
    )
    docs = (REPO_ROOT / "docs" / "m6_shadow_observation.md").read_text(
        encoding="utf-8"
    )

    for required in [
        "capture_live_topic_snapshot",
        "/joint_states",
        "base_link",
        "camera_link",
        "shadow_only_no_motion",
        "m6_shadow_observation.json",
    ]:
        assert required in capture_script

    assert "capture_m6_shadow_observation.py" in shell_script
    assert "不会发送真实运动命令" in docs
    assert "scripts/run_m6_rough_tf_shadow_inspect.sh" in docs
    assert "scripts/capture_m6_shadow_observation.sh" in docs

    forbidden_tokens = [
        "send_angles",
        "send_coords",
        "FollowJointTrajectory",
        "ExecuteGrasp",
        "ros2 action send_goal",
        "moveit_sim_execute_server",
    ]
    for source in (capture_script, shell_script):
        for token in forbidden_tokens:
            assert token not in source
