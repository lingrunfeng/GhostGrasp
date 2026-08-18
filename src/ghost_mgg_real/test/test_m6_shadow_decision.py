import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "src" / "ghost_mgg_real" / "scripts"


def _load_script(name: str):
    script_path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_shadow_observation(root: Path, observation_id: str = "m6_shadow_unit_001") -> Path:
    obs_dir = root / observation_id
    snapshot_dir = obs_dir / "snapshot"
    snapshot_dir.mkdir(parents=True)

    metadata = {
        "scene_id": observation_id,
        "frames": {
            "/camera/camera/color/image_raw": {
                "encoding": "rgb8",
                "nonzero_ratio": 1.0,
                "mean_intensity": 100.0,
            },
            "/camera/camera/depth/image_rect_raw": {
                "encoding": "16UC1",
                "valid_ratio": 0.85,
            },
            "/camera/camera/aligned_depth_to_color/image_raw": {
                "encoding": "16UC1",
                "valid_ratio": 0.85,
            },
            "/camera/camera/infra1/image_rect_raw": {
                "encoding": "mono8",
                "nonzero_ratio": 1.0,
                "mean_intensity": 75.0,
            },
            "/camera/camera/infra2/image_rect_raw": {
                "encoding": "mono8",
                "nonzero_ratio": 1.0,
                "mean_intensity": 76.0,
            },
        },
    }
    (snapshot_dir / "metadata.json").write_text(json.dumps(metadata))
    cv2.imwrite(str(snapshot_dir / "color.png"), np.zeros((5, 5, 3), dtype=np.uint8))
    depth = np.full((5, 5), 1000, dtype=np.uint16)
    depth[2, 2] = 0
    depth[1, 1] = 820
    np.save(snapshot_dir / "aligned_depth_raw.npy", depth)

    observation = {
        "schema_version": "m6_shadow_observation_v1",
        "observation_id": observation_id,
        "safety_mode": "shadow_only_no_motion",
        "snapshot": {
            "dir": str(snapshot_dir),
            "manifest": {
                "copied_files": [
                    "color.png",
                    "metadata.json",
                    "aligned_depth_raw.npy",
                ],
                "missing_topics": [],
            },
        },
        "joint_state": {
            "name": [
                "link1_to_link2",
                "link2_to_link3",
                "link3_to_link4",
                "link4_to_link5",
                "link5_to_link6",
                "link6_to_link6_flange",
            ],
            "position": [0, 0, 0, 0, 0, 0],
        },
        "camera_to_base": {
            "parent_frame": "base_link",
            "child_frame": "camera_link",
            "translation": {"x": 0.0, "y": 0.39, "z": 0.16},
        },
        "gate_checks": {
            "has_snapshot": True,
            "has_real_arm_joints": True,
            "has_camera_to_base_tf": True,
            "has_aligned_depth_raw": True,
        },
    }
    observation_path = obs_dir / "m6_shadow_observation.json"
    observation_path.write_text(json.dumps(observation))
    return observation_path


def test_shadow_decision_abstains_without_target_mask(tmp_path):
    module = _load_script("generate_m6_shadow_decision")
    observation_path = _write_shadow_observation(tmp_path)
    output_dir = tmp_path / "decision"

    report = module.generate_shadow_decision(
        shadow_observation_path=observation_path,
        mask_path=None,
        output_dir=output_dir,
        target_label="transparent_jelly_cup",
        shape_hint="cup_like",
    )

    assert report["schema_version"] == "m6_shadow_decision_v1"
    assert report["safety_mode"] == "shadow_only_no_motion"
    assert report["motion_authorized"] is False
    assert report["recommended_backend"] == "abstain"
    assert "target_mask_missing" in report["reject_reasons"]
    assert report["ready_for_shadow_planning"] is False
    assert (output_dir / "m6_shadow_decision.json").exists()
    assert (output_dir / "observation_quality.json").exists()
    assert (output_dir / "index.md").exists()


def test_shadow_decision_with_mask_runs_evidence_and_backend_selector(tmp_path):
    module = _load_script("generate_m6_shadow_decision")
    observation_path = _write_shadow_observation(tmp_path, "m6_shadow_masked_001")
    mask_path = tmp_path / "target_mask.png"
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 255
    cv2.imwrite(str(mask_path), mask)

    report = module.generate_shadow_decision(
        shadow_observation_path=observation_path,
        mask_path=mask_path,
        output_dir=tmp_path / "decision",
        target_label="transparent_jelly_cup",
        shape_hint="cup_like",
    )

    assert report["recommended_backend"] == "ghost_mgg"
    assert report["ready_for_shadow_planning"] is True
    assert report["target_summary"]["target_pixels"] == 9
    assert report["target_summary"]["hole_ratio"] == 1 / 9
    assert report["target_summary"]["table_leakage_ratio"] == 7 / 9
    assert report["quality"]["tf_ok"] is True
    assert report["quality"]["planning_requested"] is True
    assert "target_table_leakage_ratio_high" in report["depth_failure_reasons"]
    assert (tmp_path / "decision" / "evidence" / "m6_shadow_masked_001" / "evidence_overlay.png").exists()


def test_direct_mask_tool_writes_binary_mask_from_polygon(tmp_path):
    module = _load_script("annotate_direct_mask")
    image_path = tmp_path / "color.png"
    mask_path = tmp_path / "target_mask.png"
    cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))

    summary = module.save_polygon_mask(
        image_path=image_path,
        output_path=mask_path,
        polygon=[(2, 2), (7, 2), (7, 7), (2, 7)],
    )

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert summary["schema_version"] == "direct_mask_annotation_v1"
    assert summary["mask_pixels"] > 0
    assert mask is not None
    assert set(np.unique(mask)).issubset({0, 255})
    assert (tmp_path / "target_mask_overlay.png").exists()
