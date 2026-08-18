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


def test_project_mask_depth_to_camera_points_uses_camera_info_intrinsics():
    module = _load_script("generate_m6_shadow_grasp_target")
    depth_mm = np.array([[1000, 1000], [0, 2000]], dtype=np.uint16)
    mask = np.array([[True, False], [True, True]])
    camera_info = {
        "width": 2,
        "height": 2,
        "k": [2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0],
    }

    points = module.project_mask_depth_to_camera_points(
        mask=mask,
        depth_mm=depth_mm,
        camera_info=camera_info,
    )

    assert points.shape == (2, 3)
    assert np.allclose(points[0], [0.0, 0.0, 1.0])
    assert np.allclose(points[1], [1.0, 0.5, 2.0])


def test_build_shadow_grasp_target_outputs_moveit_compatible_cylinder(tmp_path):
    module = _load_script("generate_m6_shadow_grasp_target")
    depth_mm = np.full((5, 5), 1000, dtype=np.uint16)
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    camera_info = {
        "frame_id": "camera_color_optical_frame",
        "width": 5,
        "height": 5,
        "k": [100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0],
    }
    transform = {
        "parent_frame": "base_link",
        "child_frame": "camera_color_optical_frame",
        "translation": {"x": 0.1, "y": 0.2, "z": 0.3},
        "rotation_quat": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }

    row = module.build_target_row_from_points(
        observation_id="m6_shadow_target_001",
        shape_hint="cylinder",
        points_base=module.transform_points(
            module.project_mask_depth_to_camera_points(
                mask=mask,
                depth_mm=depth_mm,
                camera_info=camera_info,
            ),
            transform,
        ),
        object_height_m=0.025,
        pregrasp_clearance_m=0.095,
    )

    assert row["target_id"] == "m6_shadow_target_001"
    assert row["shape_type"] == "cylinder"
    assert row["center_x_m"] == 0.1
    assert row["center_y_m"] == 0.2
    assert row["center_z_m"] == 1.2875
    assert row["height_m"] == 0.025
    assert row["radius_m"] > 0.0
    assert row["required_gripper_width_m"] >= 2.0 * row["radius_m"]
    assert row["grasp_type"] == "top_grasp"
    assert row["valid"] is True


def test_refine_mask_with_depth_support_removes_far_projection_tail():
    module = _load_script("generate_m6_shadow_grasp_target")
    depth_mm = np.full((8, 12), 1200, dtype=np.uint16)
    mask = np.zeros(depth_mm.shape, dtype=bool)
    mask[2:5, 2:5] = True
    mask[3:5, 5:10] = True
    depth_mm[2:5, 2:5] = 820

    refined, diagnostics = module.refine_mask_with_depth_support(
        mask=mask,
        depth_mm=depth_mm,
        foreground_depth_window_mm=80,
        min_supported_pixels=4,
    )

    assert int(mask.sum()) == 19
    assert int(refined.sum()) == 9
    assert refined[3, 3]
    assert not refined[3, 8]
    assert diagnostics["removed_far_depth_pixels"] == 10
    assert diagnostics["fallback_used"] is False


def test_oriented_footprint_recovers_rotated_rectangular_points():
    module = _load_script("generate_m6_shadow_grasp_target")
    yaw = np.deg2rad(32.0)
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float64,
    )
    local = np.array(
        [
            [-0.030, -0.012],
            [-0.030, 0.012],
            [0.030, -0.012],
            [0.030, 0.012],
            [0.000, 0.000],
        ],
        dtype=np.float64,
    )
    points_xy = local @ rotation.T + np.array([0.12, -0.04])

    footprint = module.estimate_oriented_footprint(points_xy, square_anisotropy_threshold=1.05)

    assert np.allclose(footprint["center_xy_m"], [0.12, -0.04], atol=1e-6)
    assert footprint["size_x_m"] > footprint["size_y_m"]
    assert abs(footprint["size_x_m"] - 0.06) < 0.004
    assert abs(footprint["size_y_m"] - 0.024) < 0.004
    assert abs(np.cos(footprint["yaw_rad"] - yaw)) > 0.995


def test_box_target_row_includes_yaw_and_grasp_yaw_candidates():
    module = _load_script("generate_m6_shadow_grasp_target")
    yaw = np.deg2rad(-25.0)
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float64,
    )
    local = np.array(
        [[x, y] for x in np.linspace(-0.028, 0.028, 4) for y in np.linspace(-0.010, 0.010, 3)],
        dtype=np.float64,
    )
    xy = local @ rotation.T + np.array([0.04, 0.18])
    z = np.full((xy.shape[0], 1), 0.79)
    points_base = np.hstack([xy, z])

    row = module.build_target_row_from_points(
        observation_id="m7_oriented_box_001",
        shape_hint="box",
        points_base=points_base,
        object_height_m=0.025,
        pregrasp_clearance_m=0.095,
    )

    assert row["shape_type"] == "box"
    assert abs(np.cos(row["yaw_rad"] - yaw)) > 0.99
    assert row["size_x_m"] > row["size_y_m"]
    assert row["footprint"]["estimator"] == "pca_obb"
    assert row["ranked_geometry_hypotheses"][0]["shape_type"] == "box"
    assert row["ranked_geometry_hypotheses"][0]["rank"] == 1
    assert row["ranked_geometry_hypotheses"][0]["yaw_rad"] == row["yaw_rad"]
    assert row["ranked_geometry_hypotheses"][1]["shape_type"] == "cylinder"
    assert [candidate["label"] for candidate in row["grasp_yaw_candidates"]] == [
        "box_short_axis",
        "box_long_axis",
    ]
    assert row["grasp_yaw_candidates"][0]["required_gripper_width_m"] < row["grasp_yaw_candidates"][1]["required_gripper_width_m"]


def test_square_mask_canonicalizes_polluted_box_footprint():
    module = _load_script("generate_m6_shadow_grasp_target")
    xy = np.array(
        [[x, y] for x in np.linspace(-0.027, 0.027, 8) for y in np.linspace(-0.013, 0.013, 4)],
        dtype=np.float64,
    ) + np.array([0.05, 0.19])
    z = np.full((xy.shape[0], 1), 0.79)
    points_base = np.hstack([xy, z])
    square_mask = np.zeros((100, 100), dtype=bool)
    square_mask[20:90, 15:85] = True

    row = module.build_target_row_from_points(
        observation_id="m7_square_box_polluted_depth",
        shape_hint="box",
        points_base=points_base,
        object_height_m=0.025,
        pregrasp_clearance_m=0.095,
        refined_mask=square_mask,
    )

    assert row["shape_type"] == "box"
    assert row["footprint"]["estimator"] == "pca_obb_mask_square_canonical"
    assert row["size_x_m"] == row["size_y_m"]
    assert 0.024 <= row["size_x_m"] <= 0.030
    assert row["yaw_rad"] == 0.0


def test_perspective_square_mask_canonicalizes_polluted_box_footprint():
    module = _load_script("generate_m6_shadow_grasp_target")
    xy = np.array(
        [[x, y] for x in np.linspace(-0.027, 0.027, 8) for y in np.linspace(-0.011, 0.011, 4)],
        dtype=np.float64,
    ) + np.array([0.05, 0.19])
    z = np.full((xy.shape[0], 1), 0.79)
    points_base = np.hstack([xy, z])
    perspective_square_mask = np.zeros((120, 120), dtype=bool)
    perspective_square_mask[20:94, 30:87] = True

    row = module.build_target_row_from_points(
        observation_id="m7_perspective_square_box_polluted_depth",
        shape_hint="box",
        points_base=points_base,
        object_height_m=0.025,
        pregrasp_clearance_m=0.095,
        refined_mask=perspective_square_mask,
    )

    assert row["footprint"]["estimator"] == "pca_obb_mask_square_canonical"
    assert row["size_x_m"] == row["size_y_m"]
    assert row["yaw_rad"] == 0.0


def test_generate_shadow_grasp_target_report_writes_targets_json(tmp_path):
    module = _load_script("generate_m6_shadow_grasp_target")
    snapshot_dir = tmp_path / "snapshot"
    output_dir = tmp_path / "target"
    snapshot_dir.mkdir()
    depth_mm = np.full((5, 5), 1000, dtype=np.uint16)
    np.save(snapshot_dir / "aligned_depth_raw.npy", depth_mm)
    cv2.imwrite(str(snapshot_dir / "color.png"), np.zeros((5, 5, 3), dtype=np.uint8))
    (snapshot_dir / "aligned_depth_camera_info.json").write_text(
        json.dumps(
            {
                "frame_id": "camera_color_optical_frame",
                "width": 5,
                "height": 5,
                "k": [100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0],
            }
        )
    )
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 255
    mask_path = tmp_path / "target_mask.png"
    cv2.imwrite(str(mask_path), mask)
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "observation_id": "m6_shadow_target_002",
                "recommended_backend": "normal_rgbd",
                "ready_for_shadow_planning": True,
                "target_label": "green_cylinder",
                "shape_hint": "cylinder",
                "mask_path": str(mask_path),
                "shadow_observation_path": str(tmp_path / "observation.json"),
            }
        )
    )
    (tmp_path / "observation.json").write_text(
        json.dumps(
            {
                "observation_id": "m6_shadow_target_002",
                "snapshot": {"dir": str(snapshot_dir)},
            }
        )
    )
    transform = {
        "parent_frame": "base_link",
        "child_frame": "camera_color_optical_frame",
        "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation_quat": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }

    report = module.generate_shadow_grasp_target(
        shadow_decision_path=decision_path,
        output_dir=output_dir,
        camera_to_base_transform=transform,
        object_height_m=0.025,
        pregrasp_clearance_m=0.095,
    )

    targets = json.loads((output_dir / "m6_shadow_grasp_targets.json").read_text())
    assert report["schema_version"] == "m6_shadow_grasp_target_v1"
    assert report["mask_refinement"]["schema_version"] == "m7_mask_refinement_v1"
    assert report["target"]["footprint"]["schema_version"] == "m7_oriented_footprint_v1"
    assert report["target"]["grasp_yaw_candidates"]
    assert targets["schema_version"] == "m4_sim_grasp_targets_v1"
    assert targets["rows"][0]["target_id"] == "m6_shadow_target_002"
    assert (output_dir / "index.md").exists()
