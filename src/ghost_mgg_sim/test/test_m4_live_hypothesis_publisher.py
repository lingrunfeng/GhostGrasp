import importlib.util
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image


SIM_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SIM_DIR / "scripts" / "m4_live_hypothesis_publisher_node.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "m4_live_hypothesis_publisher_node", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sloped_table_depth(height=80, width=100):
    rows = np.arange(height, dtype=np.float32)[:, None]
    return np.repeat(1.12 + rows * 0.0015, width, axis=1)


def with_foreground_patch(center_u, center_v, *, height=80, width=100):
    depth = sloped_table_depth(height=height, width=width)
    depth[center_v - 5 : center_v + 5, center_u - 7 : center_u + 7] -= 0.12
    return depth


def with_low_contrast_foreground_patch(center_u, center_v, *, height=80, width=100):
    depth = sloped_table_depth(height=height, width=width)
    depth[center_v - 5 : center_v + 5, center_u - 7 : center_u + 7] -= 0.025
    return depth


def mask_center(mask):
    rows, cols = np.nonzero(mask)
    return float(cols.mean()), float(rows.mean())


def test_foreground_mask_from_depth_tracks_moved_depth_patch():
    module = load_module()

    left_mask = module.foreground_mask_from_depth(
        with_foreground_patch(30, 42),
        depth_margin_m=0.04,
        min_component_area_px=20,
    )
    right_mask = module.foreground_mask_from_depth(
        with_foreground_patch(68, 42),
        depth_margin_m=0.04,
        min_component_area_px=20,
    )

    left_center_u, left_center_v = mask_center(left_mask)
    right_center_u, right_center_v = mask_center(right_mask)
    assert 28.0 <= left_center_u <= 31.0
    assert 40.0 <= left_center_v <= 44.0
    assert 66.0 <= right_center_u <= 69.0
    assert 40.0 <= right_center_v <= 44.0
    assert right_center_u - left_center_u > 30.0


def test_foreground_mask_from_depth_falls_back_for_low_contrast_tabletop_objects():
    module = load_module()

    mask = module.foreground_mask_from_depth(
        with_low_contrast_foreground_patch(48, 41),
        depth_margin_m=0.035,
        min_component_area_px=20,
    )

    center_u, center_v = mask_center(mask)
    assert 46.0 <= center_u <= 50.0
    assert 39.0 <= center_v <= 43.0


def test_foreground_mask_from_depth_rejects_large_robot_foreground_component():
    module = load_module()
    depth = sloped_table_depth(height=120, width=140)
    depth[12:96, 8:86] -= 0.16
    depth[72:84, 104:118] -= 0.12

    mask = module.foreground_mask_from_depth(
        depth,
        depth_margin_m=0.04,
        min_component_area_px=20,
    )

    center_u, center_v = mask_center(mask)
    assert 108.0 <= center_u <= 114.0
    assert 76.0 <= center_v <= 82.0
    assert int(mask.sum()) < 260


def test_foreground_mask_from_depth_keeps_locked_target_instead_of_far_object():
    module = load_module()
    depth = sloped_table_depth(height=120, width=160)
    depth[58:70, 44:58] -= 0.10
    depth[70:86, 122:142] -= 0.16

    unlocked = module.foreground_mask_from_depth(
        depth,
        depth_margin_m=0.04,
        min_component_area_px=20,
    )
    locked = module.foreground_mask_from_depth(
        depth,
        depth_margin_m=0.04,
        min_component_area_px=20,
        locked_center_uv=(51.0, 64.0),
        max_locked_center_distance_px=28.0,
    )

    unlocked_center_u, _ = mask_center(unlocked)
    locked_center_u, locked_center_v = mask_center(locked)
    assert unlocked_center_u > 120.0
    assert 48.0 <= locked_center_u <= 54.0
    assert 62.0 <= locked_center_v <= 67.0


def test_foreground_components_mask_from_depth_keeps_multiple_tabletop_objects():
    module = load_module()
    depth = sloped_table_depth(height=120, width=160)
    depth[58:70, 44:58] -= 0.10
    depth[70:86, 122:142] -= 0.16

    mask = module.foreground_components_mask_from_depth(
        depth,
        depth_margin_m=0.04,
        min_component_area_px=20,
    )

    components = module.extract_components(mask, min_area_px=20)
    centers = sorted(component.centroid_uv[0] for component in components)
    assert len(components) == 2
    assert 48.0 <= centers[0] <= 54.0
    assert 130.0 <= centers[1] <= 134.0


def test_foreground_components_mask_from_depth_keeps_tall_irregular_tabletop_object():
    module = load_module()
    depth = sloped_table_depth(height=120, width=160)
    depth[18:96, 112:126] -= 0.12
    depth[80:96, 100:138] -= 0.12

    mask = module.foreground_components_mask_from_depth(
        depth,
        depth_margin_m=0.04,
        min_component_area_px=20,
    )

    components = module.extract_components(mask, min_area_px=20)
    assert len(components) == 1
    assert components[0].bbox_xyxy[3] - components[0].bbox_xyxy[1] > 70


def test_foreground_components_mask_from_depth_accumulates_tall_and_low_contrast_objects():
    module = load_module()
    depth = sloped_table_depth(height=120, width=160)
    depth[18:96, 112:126] -= 0.12
    depth[80:96, 100:138] -= 0.12
    depth[58:70, 44:58] -= 0.015
    depth[70:86, 72:88] -= 0.015

    mask = module.foreground_components_mask_from_depth(
        depth,
        depth_margin_m=0.04,
        min_component_area_px=20,
    )

    components = module.extract_components(mask, min_area_px=20)
    centers = sorted(component.centroid_uv[0] for component in components)
    assert len(components) == 3
    assert 48.0 <= centers[0] <= 54.0
    assert 78.0 <= centers[1] <= 84.0
    assert 116.0 <= centers[2] <= 124.0


def test_foreground_components_mask_from_depth_does_not_expand_seed_components_with_weak_bridge():
    module = load_module()
    depth = np.full((80, 140), 1.0, dtype=np.float32)
    depth[34:46, 34:48] = 0.900
    depth[34:46, 92:106] = 0.900
    depth[38:42, 48:92] = 0.985

    mask = module.foreground_components_mask_from_depth(
        depth,
        depth_margin_m=0.050,
        min_component_area_px=20,
    )

    components = module.extract_components(mask, min_area_px=20)
    centers = sorted(component.centroid_uv[0] for component in components)
    assert len(components) == 2
    assert centers[0] < 50.0
    assert centers[1] > 90.0
    assert not mask[40, 70]


def test_sparse_foreground_fit_does_not_expand_to_shadow_hole_support():
    module = load_module()
    fit = module.GeometryFit(
        component_id=1,
        shape_type="box",
        center_xy_m=(0.10, 0.20),
        size_x_m=0.030,
        size_y_m=0.024,
        size_z_m=0.040,
        yaw_rad=0.20,
        provenance="metric_points_visible_fit",
        bottom_z_m=0.0,
        center_z_m=0.02,
    )
    shadow_hole_support = module.TableFootprint(
        center_xy_m=(0.13, 0.23),
        size_x_m=0.120,
        size_y_m=0.090,
        width_px=80.0,
        height_px=60.0,
        yaw_rad=0.75,
    )

    result = module.expand_sparse_foreground_fit_with_support(
        fit,
        "foreground",
        shadow_hole_support,
    )

    assert result.size_x_m == fit.size_x_m
    assert result.size_y_m == fit.size_y_m
    assert result.center_xy_m == fit.center_xy_m
    assert "sparse_foreground_amodal" not in result.provenance
    assert "shadow_support_suppressed_for_tight_fit" in result.provenance


def test_foreground_mask_from_depth_abstains_when_locked_target_disappears_far_from_candidates():
    module = load_module()
    depth = sloped_table_depth(height=120, width=160)
    depth[70:86, 122:142] -= 0.16

    mask = module.foreground_mask_from_depth(
        depth,
        depth_margin_m=0.04,
        min_component_area_px=20,
        locked_center_uv=(51.0, 64.0),
        max_locked_center_distance_px=28.0,
    )

    assert not mask.any()


def test_target_color_mask_from_image_selects_red_observation_not_depth_blob():
    module = load_module()
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    rgb[25:38, 15:29] = (210, 30, 20)
    rgb[48:65, 70:88] = (30, 40, 210)

    mask = module.target_color_mask_from_image(
        rgb,
        color_hint="red",
        min_area_px=20,
    )

    center_u, center_v = mask_center(mask)
    assert 19.0 <= center_u <= 24.0
    assert 30.0 <= center_v <= 34.0
    assert int(mask.sum()) >= 120


def test_decode_mask_image_accepts_mono8_and_8uc1():
    module = load_module()
    values = np.zeros((10, 12), dtype=np.uint8)
    values[3:7, 4:9] = 255
    msg = Image()
    msg.height = values.shape[0]
    msg.width = values.shape[1]
    msg.step = values.shape[1]
    msg.encoding = "mono8"
    msg.is_bigendian = False
    msg.data = values.reshape(-1).tolist()

    decoded = module.decode_mask_image(msg)
    assert decoded.dtype == bool
    assert decoded.shape == values.shape
    assert decoded[5, 6]

    msg.encoding = "8UC1"
    decoded_8uc1 = module.decode_mask_image(msg)
    assert np.array_equal(decoded, decoded_8uc1)


def test_external_target_mask_takes_priority_over_color_hint_and_depth_blob():
    module = load_module()
    rclpy.init()
    node = None
    try:
        node = module.M4LiveHypothesisPublisherNode()
        node.target_color_hint = "red"
        capture = CapturePublisher()
        node.mask_pub = capture

        depth = sloped_table_depth(height=80, width=100)
        depth[45:58, 72:88] -= 0.12
        node.handle_raw_depth(make_depth_msg(depth))

        rgb = np.zeros((80, 100, 3), dtype=np.uint8)
        rgb[20:34, 12:26] = (220, 20, 20)
        node.handle_color(make_color_msg(rgb))

        external = np.zeros((80, 100), dtype=np.uint8)
        external[50:64, 44:58] = 255
        node.handle_external_mask(make_mask_msg(external))
        node.handle_raw_depth(make_depth_msg(depth))

        assert capture.messages
        data = np.asarray(capture.messages[-1].data, dtype=np.uint8).reshape(
            capture.messages[-1].height,
            capture.messages[-1].step,
        )[:, : capture.messages[-1].width]
        mask = data > 0
        center_u, center_v = mask_center(mask)
        assert 49.0 <= center_u <= 52.0
        assert 56.0 <= center_v <= 59.0
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_external_target_mask_required_disables_color_and_depth_fallbacks():
    module = load_module()
    rclpy.init()
    node = None
    try:
        node = module.M4LiveHypothesisPublisherNode()
        node.require_external_mask = True
        node.target_color_hint = "red"
        capture = CapturePublisher()
        node.mask_pub = capture

        depth = sloped_table_depth(height=80, width=100)
        depth[45:58, 72:88] -= 0.12
        node.handle_raw_depth(make_depth_msg(depth))

        rgb = np.zeros((80, 100, 3), dtype=np.uint8)
        rgb[20:34, 12:26] = (220, 20, 20)
        node.handle_color(make_color_msg(rgb))

        assert capture.messages
        data = np.asarray(capture.messages[-1].data, dtype=np.uint8).reshape(
            capture.messages[-1].height,
            capture.messages[-1].step,
        )[:, : capture.messages[-1].width]
        assert not (data > 0).any()
        assert node.latest_target_mask is not None
        assert not node.latest_target_mask.any()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_live_node_color_hint_does_not_lock_to_stale_target_center():
    module = load_module()
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "target_color_mask_from_image(" in source
    assert "locked_center_uv=None" in source


def test_evidence_from_depth_and_mask_marks_holes_and_foreground():
    module = load_module()
    depth = with_foreground_patch(45, 40)
    mask = module.foreground_mask_from_depth(depth, depth_margin_m=0.04, min_component_area_px=20)
    corrupted = depth.copy()
    corrupted[38:42, 43:47] = 0.0

    evidence = module.evidence_from_depth_and_mask(corrupted, mask)

    assert evidence.hole[40, 45] == 1.0
    assert evidence.foreground_support[35, 40] == 1.0
    assert evidence.valid[35, 40] == 1.0
    assert evidence.valid[40, 45] == 0.0


def test_build_live_hypothesis_array_uses_camera_data_not_gazebo_truth():
    module = load_module()
    raw_depth = with_foreground_patch(50, 42)
    corrupted_depth = raw_depth.copy()
    corrupted_depth[41:44, 47:53] = 0.0
    mask = module.foreground_mask_from_depth(
        raw_depth,
        depth_margin_m=0.04,
        min_component_area_px=20,
    )

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        top_k=2,
    )

    assert message.header.frame_id == "d435_depth_optical_frame"
    assert message.backend_name == "ghost_mgg_m4_live"
    assert len(message.hypotheses) >= 1
    top = message.hypotheses[0]
    assert top.hypothesis_id.startswith(("box_", "cylinder_"))
    assert top.pose_base.header.frame_id == "d435_depth_optical_frame"
    assert abs(top.pose_base.pose.position.x) < 0.05
    assert top.pose_base.pose.position.z > 0.80
    assert 0.005 < top.dimensions_m.x < 0.30
    assert 0.005 < top.dimensions_m.y < 0.30
    assert top.grasp_candidates
    assert top.grasp_candidates[0].grasp_type == module.GraspCandidate.GRASP_TYPE_TOP
    assert top.grasp_candidates[0].approach_vector.z == -1.0
    assert "gazebo" not in top.provenance.lower()
    assert "gz model" not in top.provenance.lower()


def test_bbox_shape_constant_uses_box_marker_for_unknown_irregular_proxy():
    module = load_module()

    assert module.shape_constant("bbox") == module.GeometryHypothesis.SHAPE_BOX


def test_build_live_hypothesis_array_publishes_no_truth_ghost_mgg_v1_scores():
    module = load_module()
    raw_depth = tabletop_depth_with_object_patch()
    corrupted_depth = raw_depth.copy()
    corrupted_depth[40:44, 47:52] = 0.0
    corrupted_depth[44:47, 47:52] = 0.250
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[33:53, 38:62] = True

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    top = message.hypotheses[0]
    assert "ghost_mgg_v1" in top.provenance
    assert "no_truth" in top.provenance
    assert "gazebo" not in top.provenance.lower()
    assert top.score.failure > 0.0
    assert top.score.depth > 0.0


def test_live_hypothesis_provenance_reports_evidence_diagnostics_without_truth():
    module = load_module()
    raw_depth = np.full((90, 120), 0.200, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[42:56, 53:67] = True
    mask[18:42, 53:67] = True
    raw_depth[mask] = 0.225
    corrupted_depth[mask] = 0.0

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=120, height=90, fx=120.0, fy=120.0, cx=60.0, cy=45.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.200,
        primitive_height_m=0.040,
        top_k=1,
    )

    provenance = message.hypotheses[0].provenance.lower()
    for token in (
        "evidence_diag:",
        "support_source=",
        "final_xy=",
        "final_size=",
        "final_yaw_deg=",
        "shadow_xy=",
        "full_xy=",
        "residual_n=",
        "fit_n=",
    ):
        assert token in provenance
    assert "gazebo" not in provenance
    assert "sdf" not in provenance


def downward_camera_to_world_transform() -> TransformStamped:
    transform = TransformStamped()
    transform.header.frame_id = "world"
    transform.child_frame_id = "d435_depth_optical_frame"
    transform.transform.translation.x = 0.0
    transform.transform.translation.y = 0.0
    transform.transform.translation.z = 1.0
    transform.transform.rotation.x = 1.0
    transform.transform.rotation.y = 0.0
    transform.transform.rotation.z = 0.0
    transform.transform.rotation.w = 0.0
    return transform


def identity_camera_to_world_transform() -> TransformStamped:
    transform = TransformStamped()
    transform.header.frame_id = "world"
    transform.child_frame_id = "d435_depth_optical_frame"
    transform.transform.rotation.w = 1.0
    return transform


def z_yaw_camera_to_world_transform(yaw_rad: float) -> TransformStamped:
    transform = TransformStamped()
    transform.header.frame_id = "world"
    transform.child_frame_id = "d435_depth_optical_frame"
    transform.transform.rotation.z = math.sin(0.5 * float(yaw_rad))
    transform.transform.rotation.w = math.cos(0.5 * float(yaw_rad))
    return transform


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def constant_depth_patch(depth_m: float, *, height=80, width=100):
    depth = np.ones((height, width), dtype=np.float32)
    depth[37:47, 43:57] = float(depth_m)
    return depth


def tabletop_depth_with_object_patch(
    *,
    table_depth_m=0.250,
    object_depth_m=0.225,
    height=80,
    width=100,
):
    depth = np.full((height, width), float(table_depth_m), dtype=np.float32)
    depth[38:48, 44:56] = float(object_depth_m)
    return depth


def yawed_rectangle_mask(*, height=80, width=100, center_u=50.0, center_v=40.0):
    rows, cols = np.indices((height, width))
    du = cols.astype(np.float32) - float(center_u)
    dv = rows.astype(np.float32) - float(center_v)
    yaw = np.deg2rad(35.0)
    local_u = np.cos(yaw) * du + np.sin(yaw) * dv
    local_v = -np.sin(yaw) * du + np.cos(yaw) * dv
    return (np.abs(local_u) <= 13.0) & (np.abs(local_v) <= 5.0)


def yawed_near_square_mask(*, height=80, width=100, center_u=50.0, center_v=40.0):
    rows, cols = np.indices((height, width))
    du = cols.astype(np.float32) - float(center_u)
    dv = rows.astype(np.float32) - float(center_v)
    yaw = np.deg2rad(45.0)
    local_u = np.cos(yaw) * du + np.sin(yaw) * dv
    local_v = -np.sin(yaw) * du + np.cos(yaw) * dv
    return (np.abs(local_u) <= 9.0) & (np.abs(local_v) <= 6.0)


def test_principal_yaw_ignores_near_square_masks():
    module = load_module()

    assert module.principal_yaw_from_mask(yawed_near_square_mask()) is None


def test_build_live_hypothesis_array_table_anchors_base_pose_and_top_grasp():
    module = load_module()
    raw_depth = with_foreground_patch(50, 42)
    mask = module.foreground_mask_from_depth(
        raw_depth,
        depth_margin_m=0.04,
        min_component_area_px=20,
    )

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    assert message.header.frame_id == "world"
    assert len(message.hypotheses) == 1
    top = message.hypotheses[0]
    assert top.pose_camera.header.frame_id == "d435_depth_optical_frame"
    assert top.pose_base.header.frame_id == "world"
    assert abs(top.pose_base.pose.position.x) < 0.03
    assert abs(top.pose_base.pose.position.y) < 0.04
    assert 0.762 <= top.pose_base.pose.position.z <= 0.771
    assert top.pose_base.pose.orientation.w == 1.0
    assert top.pose_base.pose.orientation.x == 0.0
    assert top.grasp_candidates[0].grasp_pose.header.frame_id == "world"
    assert 0.779 <= top.grasp_candidates[0].grasp_pose.pose.position.z <= 0.796
    assert 0.859 <= top.grasp_candidates[0].pregrasp_pose.pose.position.z <= 0.876
    assert top.grasp_candidates[0].approach_vector.z == -1.0


def test_build_live_hypothesis_array_estimates_height_from_table_and_visible_depth():
    module = load_module()
    raw_depth = constant_depth_patch(0.225)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[37:47, 43:57] = True

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    top = message.hypotheses[0]
    assert 0.022 <= top.dimensions_m.z <= 0.028
    assert 0.761 <= top.pose_base.pose.position.z <= 0.764
    assert 0.779 <= top.grasp_candidates[0].grasp_pose.pose.position.z <= 0.782


def test_build_live_hypothesis_array_uses_footprint_height_prior_for_hole_only_target():
    module = load_module()
    raw_depth = tabletop_depth_with_object_patch()
    corrupted_depth = raw_depth.copy()
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[38:48, 44:56] = True
    corrupted_depth[mask] = 0.0

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    top = message.hypotheses[0]
    assert 0.018 <= top.dimensions_m.z <= 0.030
    assert abs(top.pose_base.pose.position.z - (0.75 + 0.5 * top.dimensions_m.z)) < 1e-9


def test_build_live_hypothesis_array_rejects_rear_hole_tail_as_geometry_support():
    module = load_module()
    yaw_rad = 0.65
    raw_depth = np.full((90, 120), 0.200, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[42:56, 53:67] = True
    mask[18:42, 53:67] = True
    raw_depth[mask] = 0.225
    corrupted_depth[mask] = 0.0

    components = module.extract_components(mask, min_area_px=1)
    message = module.build_live_tabletop_component_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        components=components,
        intrinsics=module.CameraModel(width=120, height=90, fx=120.0, fy=120.0, cx=60.0, cy=45.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=z_yaw_camera_to_world_transform(yaw_rad),
        table_z_m=0.200,
        primitive_height_m=0.040,
        min_tabletop_proxy_height_m=0.005,
    )

    top = message.hypotheses[0]
    assert top.hypothesis_id.startswith("component_1_bbox")
    assert abs(top.pose_base.pose.orientation.z) > 0.10
    assert 0.018 <= top.dimensions_m.x <= 0.035
    assert 0.018 <= top.dimensions_m.y <= 0.035
    assert 0.018 <= top.dimensions_m.z <= 0.035
    assert "shadow_shape=unknown" in top.provenance
    assert "support=shadow_trimmed" in top.provenance


def test_build_live_hypothesis_array_uses_sparse_foreground_before_rear_shadow_tail():
    module = load_module()
    yaw_rad = 0.50
    raw_depth = np.full((90, 120), 0.200, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[42:56, 53:67] = True
    mask[18:42, 53:67] = True
    raw_depth[mask] = 0.225
    corrupted_depth[mask] = 0.0
    corrupted_depth[48:52, 56:64] = 0.225

    components = module.extract_components(mask, min_area_px=1)
    message = module.build_live_tabletop_component_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        components=components,
        intrinsics=module.CameraModel(width=120, height=90, fx=120.0, fy=120.0, cx=60.0, cy=45.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=z_yaw_camera_to_world_transform(yaw_rad),
        table_z_m=0.200,
        primitive_height_m=0.040,
        min_tabletop_proxy_height_m=0.005,
    )

    top = message.hypotheses[0]
    assert "support=foreground" in top.provenance
    assert "shadow_support_suppressed_for_tight_fit" in top.provenance
    assert top.dimensions_m.x >= 0.010
    assert top.dimensions_m.y >= 0.005
    assert top.dimensions_m.x < 0.020
    assert top.dimensions_m.y < 0.012
    pose_yaw = yaw_from_quaternion(top.pose_base.pose.orientation)
    assert abs(module.normalize_yaw_half_turn(pose_yaw - yaw_rad)) < math.radians(12.0)


def test_sparse_foreground_shadow_support_keeps_visible_geometry_tight():
    module = load_module()
    fit = module.GeometryFit(
        component_id=3,
        shape_type="box",
        center_xy_m=(0.010, -0.020),
        size_x_m=0.018,
        size_y_m=0.017,
        size_z_m=0.020,
        yaw_rad=math.radians(20.0),
        provenance="metric_points_oriented_footprint",
        bottom_z_m=0.75,
        center_z_m=0.76,
    )
    support = module.TableFootprint(
        center_xy_m=(0.040, 0.015),
        size_x_m=0.032,
        size_y_m=0.031,
        width_px=32.0,
        height_px=31.0,
        yaw_rad=math.radians(70.0),
    )

    expanded = module.expand_sparse_foreground_fit_with_support(
        fit,
        "foreground",
        support,
        min_area_gain=1.2,
    )

    assert expanded.center_xy_m == fit.center_xy_m
    assert expanded.size_x_m == fit.size_x_m
    assert expanded.size_y_m == fit.size_y_m
    assert "shadow_support_suppressed_for_tight_fit" in expanded.provenance


def test_sparse_foreground_box_alignment_uses_visible_edges_for_center_and_yaw():
    module = load_module()
    yaw_rad = math.radians(30.0)
    center = np.array([0.040, -0.020], dtype=np.float64)
    side = 0.026
    axis_x = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)
    axis_y = np.array([-math.sin(yaw_rad), math.cos(yaw_rad)], dtype=np.float64)
    offsets = []
    for value in np.linspace(-0.5 * side, 0.5 * side, 9):
        offsets.append(0.5 * side * axis_x + value * axis_y)
        offsets.append(value * axis_x - 0.5 * side * axis_y)
    edge_points = center + np.asarray(offsets, dtype=np.float64)
    sparse_fit = module.GeometryFit(
        component_id=2,
        shape_type="box",
        center_xy_m=(float(edge_points[:, 0].mean()), float(edge_points[:, 1].mean())),
        size_x_m=0.014,
        size_y_m=0.026,
        size_z_m=side,
        yaw_rad=math.radians(-18.0),
        provenance="metric_points_oriented_footprint",
        bottom_z_m=0.74,
        center_z_m=0.753,
    )
    support = module.TableFootprint(
        center_xy_m=(float(center[0] + 0.006), float(center[1] - 0.004)),
        size_x_m=0.038,
        size_y_m=0.034,
        width_px=38.0,
        height_px=34.0,
        yaw_rad=math.radians(4.0),
    )

    aligned = module.align_sparse_foreground_box_to_visible_edges(
        sparse_fit,
        "foreground",
        support,
        edge_points,
    )

    center_error = math.hypot(aligned.center_xy_m[0] - center[0], aligned.center_xy_m[1] - center[1])
    yaw_delta = module.normalize_yaw_half_turn(aligned.yaw_rad - yaw_rad)
    yaw_error = min(
        abs(yaw_delta),
        abs(module.normalize_yaw_half_turn(yaw_delta - 0.5 * math.pi)),
        abs(module.normalize_yaw_half_turn(yaw_delta + 0.5 * math.pi)),
    )
    assert center_error <= 0.004
    assert yaw_error <= math.radians(4.0)
    assert 0.023 <= aligned.size_x_m <= 0.030
    assert 0.023 <= aligned.size_y_m <= 0.030
    assert "foreground_edge_aligned" in aligned.provenance


def test_sparse_foreground_bbox_alignment_uses_visible_edges_for_yaw():
    module = load_module()
    yaw_rad = math.radians(42.0)
    center = np.array([0.025, 0.035], dtype=np.float64)
    side = 0.028
    axis_x = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)
    axis_y = np.array([-math.sin(yaw_rad), math.cos(yaw_rad)], dtype=np.float64)
    offsets = []
    for value in np.linspace(-0.5 * side, 0.5 * side, 11):
        offsets.append(-0.5 * side * axis_x + value * axis_y)
        offsets.append(value * axis_x + 0.5 * side * axis_y)
    edge_points = center + np.asarray(offsets, dtype=np.float64)
    sparse_fit = module.GeometryFit(
        component_id=7,
        shape_type="bbox",
        center_xy_m=(float(edge_points[:, 0].mean()), float(edge_points[:, 1].mean())),
        size_x_m=0.012,
        size_y_m=0.025,
        size_z_m=0.026,
        yaw_rad=math.radians(-8.0),
        provenance="metric_points_oriented_footprint",
        bottom_z_m=0.74,
        center_z_m=0.753,
    )
    support = module.TableFootprint(
        center_xy_m=(float(center[0] - 0.004), float(center[1] + 0.003)),
        size_x_m=0.035,
        size_y_m=0.033,
        width_px=35.0,
        height_px=33.0,
        yaw_rad=math.radians(0.0),
    )

    aligned = module.align_sparse_foreground_box_to_visible_edges(
        sparse_fit,
        "foreground",
        support,
        edge_points,
    )

    yaw_delta = module.normalize_yaw_half_turn(aligned.yaw_rad - yaw_rad)
    yaw_error = min(
        abs(yaw_delta),
        abs(module.normalize_yaw_half_turn(yaw_delta - 0.5 * math.pi)),
        abs(module.normalize_yaw_half_turn(yaw_delta + 0.5 * math.pi)),
    )
    assert aligned.shape_type == "bbox"
    assert yaw_error <= math.radians(4.0)
    assert "foreground_edge_aligned" in aligned.provenance


def test_large_shadow_component_uses_compact_residual_foreground_points():
    module = load_module()
    raw_depth = np.full((120, 160), 0.200, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[52:72, 30:70] = True
    mask[20:56, 64:92] = True
    raw_depth[mask] = 0.225
    corrupted_depth[mask] = 0.0
    residual_foreground = np.zeros_like(mask)
    residual_foreground[60:63, 46:50] = True
    corrupted_depth[residual_foreground] = 0.225

    intrinsics = module.CameraModel(width=160, height=120, fx=120.0, fy=120.0, cx=80.0, cy=60.0)
    transform = z_yaw_camera_to_world_transform(0.25)
    components = module.extract_components(mask, min_area_px=1)
    message = module.build_live_tabletop_component_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        components=components,
        intrinsics=intrinsics,
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=transform,
        table_z_m=0.200,
        primitive_height_m=0.040,
        min_tabletop_proxy_height_m=0.005,
    )

    top = message.hypotheses[0]
    residual_points = module.base_points_from_depth(
        depth_m=corrupted_depth,
        intrinsics=intrinsics,
        camera_to_base_transform=transform,
    )[residual_foreground]
    residual_center = residual_points[:, 0:2].mean(axis=0)
    center_error = math.hypot(
        top.pose_base.pose.position.x - residual_center[0],
        top.pose_base.pose.position.y - residual_center[1],
    )

    assert "support=foreground" in top.provenance
    assert "shadow_support_suppressed_for_tight_fit" in top.provenance
    assert center_error <= 0.010


def test_build_live_hypothesis_array_trims_far_shadow_tail_before_fitting_unknown_support():
    module = load_module()
    yaw_rad = 0.35
    raw_depth = np.full((90, 120), 0.200, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[42:58, 35:85] = True
    mask[30:44, 62:88] = True
    mask[15:30, 62:88] = True
    raw_depth[mask] = 0.225
    corrupted_depth[mask] = 0.0

    components = module.extract_components(mask, min_area_px=1)
    message = module.build_live_tabletop_component_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        components=components,
        intrinsics=module.CameraModel(width=120, height=90, fx=120.0, fy=120.0, cx=60.0, cy=45.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=z_yaw_camera_to_world_transform(yaw_rad),
        table_z_m=0.200,
        primitive_height_m=0.040,
        min_tabletop_proxy_height_m=0.005,
    )

    top = message.hypotheses[0]
    assert top.dimensions_m.x >= 0.060
    assert top.dimensions_m.y <= 0.055
    assert "support=shadow_trimmed" in top.provenance


def test_build_live_hypothesis_array_anchors_yawed_hole_only_support_to_compact_base():
    module = load_module()
    yaw_rad = 0.30
    raw_depth = np.full((120, 160), 0.200, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    rows, cols = np.indices(raw_depth.shape)
    du = cols.astype(np.float32) - 80.0
    dv = rows.astype(np.float32) - 70.0
    base_yaw_rad = math.radians(35.0)
    local_u = math.cos(base_yaw_rad) * du + math.sin(base_yaw_rad) * dv
    local_v = -math.sin(base_yaw_rad) * du + math.cos(base_yaw_rad) * dv
    compact_base = (np.abs(local_u) <= 13.0) & (np.abs(local_v) <= 8.0)
    rear_shadow_tail = (74 <= cols) & (cols <= 92) & (30 <= rows) & (rows < 60)
    mask = compact_base | rear_shadow_tail
    raw_depth[mask] = 0.225
    corrupted_depth[mask] = 0.0

    intrinsics = module.CameraModel(
        width=160,
        height=120,
        fx=120.0,
        fy=120.0,
        cx=80.0,
        cy=60.0,
    )
    transform = z_yaw_camera_to_world_transform(yaw_rad)
    components = module.extract_components(mask, min_area_px=1)
    message = module.build_live_tabletop_component_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        components=components,
        intrinsics=intrinsics,
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=transform,
        table_z_m=0.200,
        primitive_height_m=0.040,
        min_tabletop_proxy_height_m=0.005,
    )

    top = message.hypotheses[0]
    compact_points = module.table_points_from_mask(
        mask=compact_base,
        intrinsics=intrinsics,
        camera_to_base_transform=transform,
        table_z_m=0.200,
    )
    compact_footprint = module.table_footprint_from_points(compact_points)
    assert compact_footprint is not None
    center_error = math.hypot(
        top.pose_base.pose.position.x - compact_footprint.center_xy_m[0],
        top.pose_base.pose.position.y - compact_footprint.center_xy_m[1],
    )
    pose_yaw = yaw_from_quaternion(top.pose_base.pose.orientation)
    yaw_error = abs(module.normalize_yaw_half_turn(pose_yaw - compact_footprint.yaw_rad))

    assert center_error <= 0.012
    assert yaw_error <= math.radians(30.0)
    assert top.dimensions_m.x <= 0.040
    assert top.dimensions_m.y <= 0.040


def test_build_live_hypothesis_array_uses_nearby_sparse_foreground_before_shadow_projection():
    module = load_module()
    raw_depth = np.full((90, 120), 0.200, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[35:55, 55:73] = True
    raw_depth[mask] = 0.225
    corrupted_depth[mask] = 0.0
    corrupted_depth[40:55, 42:52] = 0.225

    components = module.extract_components(mask, min_area_px=1)
    message = module.build_live_tabletop_component_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        components=components,
        intrinsics=module.CameraModel(width=120, height=90, fx=120.0, fy=120.0, cx=60.0, cy=45.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=z_yaw_camera_to_world_transform(0.0),
        table_z_m=0.200,
        primitive_height_m=0.040,
        min_tabletop_proxy_height_m=0.005,
    )

    top = message.hypotheses[0]
    assert top.pose_base.pose.position.x < -0.010
    assert "support=nearby_foreground" in top.provenance


def test_nearby_foreground_does_not_steal_points_from_neighbor_component():
    module = load_module()
    raw_depth = np.full((90, 120), 0.200, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    mask = np.zeros_like(raw_depth, dtype=bool)
    front_shadow = np.zeros_like(mask)
    rear_visible = np.zeros_like(mask)
    front_shadow[35:55, 55:73] = True
    rear_visible[40:55, 76:86] = True
    mask[front_shadow | rear_visible] = True
    raw_depth[mask] = 0.225
    corrupted_depth[front_shadow] = 0.0
    corrupted_depth[rear_visible] = 0.225

    components = module.extract_components(mask, min_area_px=1)
    assert len(components) == 2

    message = module.build_live_tabletop_component_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        components=components,
        intrinsics=module.CameraModel(width=120, height=90, fx=120.0, fy=120.0, cx=60.0, cy=45.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=z_yaw_camera_to_world_transform(0.0),
        table_z_m=0.200,
        primitive_height_m=0.040,
        min_tabletop_proxy_height_m=0.005,
    )

    front = next(
        hypothesis
        for hypothesis in message.hypotheses
        if "component_id=1" in hypothesis.provenance
    )
    rear = next(
        hypothesis
        for hypothesis in message.hypotheses
        if "component_id=2" in hypothesis.provenance
    )
    assert "support=nearby_foreground" not in front.provenance
    assert "support=foreground" in rear.provenance


def test_build_live_hypothesis_array_projects_xy_footprint_to_table_plane():
    module = load_module()
    raw_depth = constant_depth_patch(0.250)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[37:47, 43:57] = True

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    top = message.hypotheses[0]
    assert top.hypothesis_id == "box_s1.00"
    assert 0.034 <= top.dimensions_m.x <= 0.036
    assert 0.024 <= top.dimensions_m.y <= 0.026


def test_build_live_hypothesis_array_adds_box_prior_for_corner_supported_masks():
    module = load_module()
    raw_depth = constant_depth_patch(0.225)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[37:47, 43:57] = True

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    top = message.hypotheses[0]
    assert top.hypothesis_id == "box_s1.00"
    assert top.score.prior > 0.40


def test_build_live_hypothesis_array_penalizes_cylinder_for_non_circular_visible_blob():
    module = load_module()
    raw_depth = np.full((120, 160), 0.250, dtype=np.float32)
    mask = np.zeros_like(raw_depth, dtype=bool)
    rows, cols = np.indices(mask.shape)
    u0, v0 = 80.0, 60.0
    blob = ((cols - u0) / 22.0) ** 2 + ((rows - v0) / 18.0) ** 2 <= 1.0
    notch = (cols < 70) & (rows < 54)
    mask[blob & ~notch] = True
    raw_depth[mask] = 0.225

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=160, height=120, fx=180.0, fy=180.0, cx=80.0, cy=60.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=2,
    )

    top = message.hypotheses[0]
    assert top.shape_type == module.GeometryHypothesis.SHAPE_BOX
    assert top.hypothesis_id.startswith("box_")


def test_build_live_hypothesis_array_shrinks_loose_mask_to_visible_foreground_extent():
    module = load_module()
    raw_depth = tabletop_depth_with_object_patch()
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[33:53, 38:62] = True

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    top = message.hypotheses[0]
    assert top.hypothesis_id == "box_s1.00"
    assert 0.024 <= top.dimensions_m.x <= 0.034
    assert 0.021 <= top.dimensions_m.y <= 0.030
    assert -0.018 <= top.pose_base.pose.position.x <= 0.018
    assert -0.012 <= top.pose_base.pose.position.y <= 0.020


def test_build_live_hypothesis_array_does_not_shrink_to_sparse_depth_residue():
    module = load_module()
    raw_depth = np.full((80, 100), 0.250, dtype=np.float32)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[30:50, 30:70] = True
    raw_depth[mask] = 0.225

    corrupted_depth = np.full_like(raw_depth, 0.250)
    corrupted_depth[37:43, 47:53] = 0.225

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    top = message.hypotheses[0]
    assert top.hypothesis_id == "box_s1.00"
    assert top.dimensions_m.x >= 0.080
    assert top.dimensions_m.y >= 0.040


def test_build_live_hypothesis_array_prefers_mask_footprint_when_sparse_depth_residue_is_valid():
    module = load_module()
    raw_depth = np.full((80, 100), 0.250, dtype=np.float32)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[30:50, 30:70] = True
    raw_depth[mask] = 0.225

    corrupted_depth = np.full_like(raw_depth, 0.250)
    corrupted_depth[36:44, 46:54] = 0.225

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    top = message.hypotheses[0]
    assert top.hypothesis_id == "box_s1.00"
    assert top.dimensions_m.x >= 0.080
    assert top.dimensions_m.y >= 0.040


def test_build_live_hypothesis_array_does_not_apply_scene_size_floor_to_tiny_live_mask():
    module = load_module()
    raw_depth = np.full((80, 100), 0.250, dtype=np.float32)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[38:44, 47:53] = True
    raw_depth[mask] = 0.225

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    top = message.hypotheses[0]
    assert 0.010 <= top.dimensions_m.x < 0.025
    assert 0.010 <= top.dimensions_m.y < 0.025


def test_build_live_hypothesis_array_uses_small_honest_proxy_for_tiny_weak_foreground_frame():
    module = load_module()
    raw_depth = np.full((80, 100), 0.250, dtype=np.float32)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[38:44, 47:53] = True
    raw_depth[mask] = 0.245

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    top = message.hypotheses[0]
    assert 0.010 <= top.dimensions_m.x < 0.025
    assert 0.010 <= top.dimensions_m.y < 0.025
    assert 0.010 <= top.dimensions_m.z < 0.025


def test_table_footprint_uses_contact_band_for_tall_loose_silhouette():
    module = load_module()
    mask = np.zeros((80, 100), dtype=bool)
    mask[30:48, 30:70] = True
    mask[48:52, 45:56] = True

    footprint = module.table_footprint_from_mask(
        mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
    )

    assert footprint is not None
    assert 0.020 <= footprint.size_x_m <= 0.035
    assert 0.020 <= footprint.size_y_m <= 0.035
    assert abs(footprint.center_xy_m[0]) < 0.020


def test_table_footprint_does_not_shrink_to_smooth_perspective_tip():
    module = load_module()
    mask = np.zeros((90, 120), dtype=bool)
    center_u = 60
    for offset, row in enumerate(range(30, 61)):
        width = 58 if offset < 10 else max(2, 58 - 3 * (offset - 10))
        half = width // 2
        mask[row, center_u - half : center_u + half] = True

    transform = downward_camera_to_world_transform()
    transform.transform.translation.z = 0.300
    footprint = module.table_footprint_from_mask(
        mask=mask,
        intrinsics=module.CameraModel(width=120, height=90, fx=100.0, fy=100.0, cx=60.0, cy=45.0),
        camera_to_base_transform=transform,
        table_z_m=0.200,
    )

    assert footprint is not None
    assert 0.022 <= footprint.size_x_m <= 0.032


def test_base_frame_dimensions_do_not_reinflate_foreground_footprint_by_rank_scale():
    module = load_module()
    raw_depth = tabletop_depth_with_object_patch()
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[33:53, 38:62] = True

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=10,
    )

    box_dimensions = [
        (hypothesis.dimensions_m.x, hypothesis.dimensions_m.y)
        for hypothesis in message.hypotheses
        if hypothesis.hypothesis_id.startswith("box_s")
    ]
    assert len(box_dimensions) >= 3
    widths = [width for width, _ in box_dimensions]
    depths = [depth for _, depth in box_dimensions]
    assert max(widths) - min(widths) < 0.001
    assert max(depths) - min(depths) < 0.001


def test_table_anchored_live_candidates_include_non_truth_yaw_grid():
    module = load_module()
    mask = yawed_rectangle_mask()
    zeros = np.zeros(mask.shape, dtype=np.float32)
    evidence = module.EvidenceMaps(
        valid=zeros.copy(),
        hole=mask.astype(np.float32),
        table_leakage=zeros.copy(),
        edge=zeros.copy(),
        flying_point=zeros.copy(),
        foreground_support=zeros.copy(),
    )

    candidates = module.generate_live_table_anchored_hypotheses(
        mask=mask,
        evidence=evidence,
        metric_footprint=module.TableFootprint(
            center_xy_m=(0.07, 0.012),
            size_x_m=0.026,
            size_y_m=0.026,
            width_px=30.0,
            height_px=30.0,
            yaw_rad=0.0,
        ),
        table_z_m=0.740,
        config=module.GhostMGGV0Config(scale_factors=(1.0,), height_m=0.026),
    )

    box_yaws = {round(math.degrees(candidate.yaw_rad)) for candidate in candidates if candidate.shape_type == "box"}
    assert 0 in box_yaws
    assert 45 in box_yaws
    assert -45 in box_yaws


def test_table_anchored_live_candidates_refine_xy_center_without_truth():
    module = load_module()
    mask = yawed_rectangle_mask()
    zeros = np.zeros(mask.shape, dtype=np.float32)
    evidence = module.EvidenceMaps(
        valid=zeros.copy(),
        hole=mask.astype(np.float32),
        table_leakage=zeros.copy(),
        edge=zeros.copy(),
        flying_point=zeros.copy(),
        foreground_support=zeros.copy(),
    )

    candidates = module.generate_live_table_anchored_hypotheses(
        mask=mask,
        evidence=evidence,
        metric_footprint=module.TableFootprint(
            center_xy_m=(0.0, 0.0),
            size_x_m=0.026,
            size_y_m=0.026,
            width_px=20.0,
            height_px=20.0,
            yaw_rad=0.0,
        ),
        table_z_m=0.750,
        config=module.GhostMGGV0Config(scale_factors=(1.0,), height_m=0.026),
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        camera_to_base_transform=downward_camera_to_world_transform(),
    )

    box_candidates = [candidate for candidate in candidates if candidate.shape_type == "box"]
    metric_centers = {
        (round(candidate.center_xy_m[0], 4), round(candidate.center_xy_m[1], 4))
        for candidate in box_candidates
    }
    pixel_centers = {
        (round(candidate.center_uv[0], 2), round(candidate.center_uv[1], 2))
        for candidate in box_candidates
    }
    assert (0.0, 0.0) in metric_centers
    assert len(metric_centers) > 1
    assert len(pixel_centers) > 1
    assert any(abs(x) >= 0.0025 or abs(y) >= 0.0025 for x, y in metric_centers)


def test_table_anchored_square_foreground_does_not_bury_box_scale_candidate():
    module = load_module()
    mask = np.zeros((120, 160), dtype=bool)
    mask[38:83, 58:103] = True
    zeros = np.zeros(mask.shape, dtype=np.float32)
    foreground = zeros.copy()
    foreground[mask] = 1.0
    evidence = module.EvidenceMaps(
        valid=np.ones(mask.shape, dtype=np.float32),
        hole=zeros.copy(),
        table_leakage=zeros.copy(),
        edge=zeros.copy(),
        flying_point=zeros.copy(),
        foreground_support=foreground,
    )

    candidates = module.generate_live_table_anchored_hypotheses(
        mask=mask,
        evidence=evidence,
        metric_footprint=module.TableFootprint(
            center_xy_m=(0.02, 0.02),
            size_x_m=0.0496,
            size_y_m=0.0496,
            width_px=45.0,
            height_px=45.0,
            yaw_rad=math.radians(3.0),
        ),
        table_z_m=0.750,
        config=module.GhostMGGV0Config(scale_factors=(0.90, 1.00), height_m=0.045),
        intrinsics=module.CameraModel(width=160, height=120, fx=180.0, fy=180.0, cx=80.0, cy=60.0),
        camera_to_base_transform=downward_camera_to_world_transform(),
    )

    priors = {candidate.hypothesis_id: candidate.prior_score for candidate in candidates}
    assert priors["box_s0.90"] > 0.0
    assert priors["box_s0.90"] > priors["cylinder_s1.00"]


def test_foreground_footprint_centers_on_robust_bounds_not_point_density():
    module = load_module()
    raw_depth = np.full((80, 100), 0.200, dtype=np.float32)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[30:50, 30:70] = True
    raw_depth[30:50, 30:48] = 0.225
    raw_depth[30:50, 66:70] = 0.225

    footprint = module.foreground_footprint_from_depth(
        mask=mask,
        raw_depth_m=raw_depth,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        camera_to_base_transform=identity_camera_to_world_transform(),
        table_z_m=0.200,
        min_foreground_height_m=0.006,
        min_mask_fraction=0.05,
    )

    assert footprint is not None
    assert -0.003 <= footprint.center_xy_m[0] <= 0.001
    assert -0.003 <= footprint.center_xy_m[1] <= 0.002


def test_foreground_footprint_estimates_base_frame_yaw_and_oriented_size():
    module = load_module()
    yaw_rad = 0.65
    raw_depth = np.full((80, 100), 0.200, dtype=np.float32)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[30:50, 30:70] = True
    raw_depth[mask] = 0.225

    footprint = module.foreground_footprint_from_depth(
        mask=mask,
        raw_depth_m=raw_depth,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        camera_to_base_transform=z_yaw_camera_to_world_transform(yaw_rad),
        table_z_m=0.200,
        min_foreground_height_m=0.006,
        min_mask_fraction=0.05,
    )

    assert footprint is not None
    assert abs(module.normalize_yaw_half_turn(footprint.yaw_rad - yaw_rad)) < math.radians(3.0)
    assert 0.080 <= footprint.size_x_m <= 0.095
    assert 0.038 <= footprint.size_y_m <= 0.050


def test_foreground_footprint_keeps_near_square_proxy_square_despite_yaw_ambiguity():
    module = load_module()
    yaw_rad = 0.65
    raw_depth = np.full((80, 100), 0.200, dtype=np.float32)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[34:47, 43:57] = True
    raw_depth[mask] = 0.225

    footprint = module.foreground_footprint_from_depth(
        mask=mask,
        raw_depth_m=raw_depth,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        camera_to_base_transform=z_yaw_camera_to_world_transform(yaw_rad),
        table_z_m=0.200,
        min_foreground_height_m=0.006,
        min_mask_fraction=0.05,
    )

    assert footprint is not None
    assert abs(footprint.size_x_m - footprint.size_y_m) < 1e-6


def test_canonicalize_square_yaw_uses_human_readable_ninety_degree_equivalent():
    module = load_module()

    yaw_rad = module.canonicalize_square_yaw(math.radians(-58.0))

    assert abs(math.degrees(yaw_rad) - 32.0) < 1e-6


def test_build_live_hypothesis_array_uses_base_footprint_yaw_for_world_pose():
    module = load_module()
    yaw_rad = 0.65
    raw_depth = np.full((80, 100), 0.200, dtype=np.float32)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[30:50, 30:70] = True
    raw_depth[mask] = 0.225

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=z_yaw_camera_to_world_transform(yaw_rad),
        table_z_m=0.200,
        primitive_height_m=0.04,
        top_k=1,
    )

    top = message.hypotheses[0]
    pose_yaw = yaw_from_quaternion(top.pose_base.pose.orientation)
    assert abs(module.normalize_yaw_half_turn(pose_yaw - yaw_rad)) < math.radians(3.0)


def test_build_live_hypothesis_array_prefers_yawed_box_for_diagonal_foreground_mask():
    module = load_module()
    mask = yawed_rectangle_mask()
    raw_depth = np.ones(mask.shape, dtype=np.float32)
    raw_depth[mask] = np.float32(0.225)

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0),
        frame_id="d435_depth_optical_frame",
        top_k=1,
    )

    top = message.hypotheses[0]
    assert top.hypothesis_id.startswith("box_yaw")
    assert top.shape_type == module.GeometryHypothesis.SHAPE_BOX
    assert abs(top.pose_base.pose.orientation.z) > 0.10


def test_build_live_hypothesis_array_returns_one_hypothesis_per_tabletop_component():
    module = load_module()
    raw_depth = np.full((90, 120), 0.250, dtype=np.float32)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[34:48, 25:41] = True
    mask[38:54, 78:94] = True
    raw_depth[mask] = 0.225

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=120, height=90, fx=120.0, fy=120.0, cx=60.0, cy=45.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    assert len(message.hypotheses) == 2
    assert all(hypothesis.hypothesis_id.startswith("component_") for hypothesis in message.hypotheses)
    assert all("component_id=" in hypothesis.provenance for hypothesis in message.hypotheses)
    centers_x = sorted(hypothesis.pose_base.pose.position.x for hypothesis in message.hypotheses)
    assert centers_x[1] - centers_x[0] > 0.06


def test_build_live_hypothesis_array_splits_close_objects_joined_by_shadow_bridge():
    module = load_module()
    raw_depth = np.full((90, 120), 0.250, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    mask = np.zeros_like(raw_depth, dtype=bool)
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    left[38:52, 42:56] = True
    right[38:52, 62:76] = True
    bridge = np.zeros_like(mask)
    bridge[43:47, 56:62] = True
    mask[left | right | bridge] = True
    raw_depth[mask] = 0.225
    corrupted_depth[left | right] = 0.225
    corrupted_depth[bridge] = 0.0

    assert len(module.extract_components(mask, min_area_px=1)) == 1

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=120, height=90, fx=120.0, fy=120.0, cx=60.0, cy=45.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    assert len(message.hypotheses) == 2
    centers_x = sorted(hypothesis.pose_base.pose.position.x for hypothesis in message.hypotheses)
    assert centers_x[1] - centers_x[0] > 0.025
    assert all("component_split=foreground_islands" in h.provenance for h in message.hypotheses)


def test_build_live_hypothesis_array_splits_visible_objects_joined_by_hole_bridge():
    module = load_module()
    raw_depth = np.full((90, 140), 0.250, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    mask = np.zeros_like(raw_depth, dtype=bool)
    left_visible = np.zeros_like(mask)
    right_visible = np.zeros_like(mask)
    hole_bridge = np.zeros_like(mask)
    left_visible[38:52, 30:46] = True
    right_visible[38:52, 94:110] = True
    hole_bridge[39:51, 46:94] = True
    mask[left_visible | right_visible | hole_bridge] = True
    raw_depth[mask] = 0.225
    corrupted_depth[left_visible | right_visible] = 0.225
    corrupted_depth[hole_bridge] = 0.0

    assert len(module.extract_components(mask, min_area_px=1)) == 1

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=140, height=90, fx=140.0, fy=140.0, cx=70.0, cy=45.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    assert len(message.hypotheses) == 2
    centers_x = sorted(hypothesis.pose_base.pose.position.x for hypothesis in message.hypotheses)
    assert centers_x[1] - centers_x[0] > 0.040
    assert all("component_split=foreground_islands" in h.provenance for h in message.hypotheses)
    assert all("hole_only_bridge_between_foreground_islands" in h.provenance for h in message.hypotheses)


def test_build_live_hypothesis_array_splits_visible_objects_joined_by_weak_visible_bridge():
    module = load_module()
    raw_depth = np.full((120, 160), 0.250, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    rows, cols = np.indices(raw_depth.shape)
    front_cylinder = (cols - 70.0) ** 2 + (rows - 76.0) ** 2 <= 10.0**2
    rear_bar = (62 <= cols) & (cols <= 108) & (45 <= rows) & (rows <= 58)
    hole_bridge = (68 <= cols) & (cols <= 90) & (58 < rows) & (rows < 66)
    weak_visible_bridge = (72 <= cols) & (cols <= 76) & (58 <= rows) & (rows <= 68)
    mask = front_cylinder | rear_bar | hole_bridge | weak_visible_bridge
    raw_depth[mask] = 0.225
    corrupted_depth[front_cylinder | rear_bar | weak_visible_bridge] = 0.225
    corrupted_depth[hole_bridge & ~weak_visible_bridge] = 0.0

    foreground = mask & np.isfinite(corrupted_depth) & (corrupted_depth > 0.0)
    assert len(module.extract_components(mask, min_area_px=1)) == 1
    assert len(module.extract_components(foreground, min_area_px=1)) == 1

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=160, height=120, fx=160.0, fy=160.0, cx=80.0, cy=60.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    assert len(message.hypotheses) == 2
    assert all("visible_weak_neck_ownership" in h.provenance for h in message.hypotheses)
    cylinder = next(
        hypothesis
        for hypothesis in message.hypotheses
        if hypothesis.shape_type == module.GeometryHypothesis.SHAPE_CYLINDER
    )
    assert cylinder.dimensions_m.x <= 0.034
    assert cylinder.dimensions_m.y <= 0.034
    assert all(max(h.dimensions_m.x, h.dimensions_m.y) <= 0.070 for h in message.hypotheses)


def test_build_live_hypothesis_array_keeps_visible_circular_foreground_as_cylinder_with_shadow_tail():
    module = load_module()
    raw_depth = np.full((100, 140), 0.250, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    rows, cols = np.indices(raw_depth.shape)
    circle = (cols - 70.0) ** 2 + (rows - 56.0) ** 2 <= 10.0**2
    tail = (60 <= cols) & (cols <= 80) & (25 <= rows) & (rows < 46)
    mask = circle | tail
    raw_depth[mask] = 0.225
    corrupted_depth[circle] = 0.225
    corrupted_depth[tail] = 0.0

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=140, height=100, fx=140.0, fy=140.0, cx=70.0, cy=50.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    assert len(message.hypotheses) == 1
    top = message.hypotheses[0]
    assert top.shape_type == module.GeometryHypothesis.SHAPE_CYLINDER
    assert "support=foreground" in top.provenance
    assert top.dimensions_m.x <= 0.034
    assert top.dimensions_m.y <= 0.034


def test_build_live_hypothesis_array_uses_round_mask_evidence_for_hole_only_cylinder():
    module = load_module()
    raw_depth = np.full((130, 170), 0.250, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    rows, cols = np.indices(raw_depth.shape)
    mask = (cols - 85.0) ** 2 + (rows - 70.0) ** 2 <= 20.0**2
    raw_depth[mask] = 0.225
    corrupted_depth[mask] = 0.0

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=170, height=130, fx=170.0, fy=170.0, cx=85.0, cy=65.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    assert len(message.hypotheses) == 1
    top = message.hypotheses[0]
    assert top.shape_type == module.GeometryHypothesis.SHAPE_CYLINDER
    assert "shadow_shape=mask_round_cylinder" in top.provenance
    assert top.dimensions_m.x == top.dimensions_m.y


def test_build_live_hypothesis_array_does_not_turn_square_hole_only_mask_into_cylinder():
    module = load_module()
    raw_depth = np.full((100, 140), 0.250, dtype=np.float32)
    corrupted_depth = raw_depth.copy()
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[45:66, 60:81] = True
    raw_depth[mask] = 0.225
    corrupted_depth[mask] = 0.0

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=corrupted_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=140, height=100, fx=140.0, fy=140.0, cx=70.0, cy=50.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    assert len(message.hypotheses) == 1
    top = message.hypotheses[0]
    assert top.shape_type != module.GeometryHypothesis.SHAPE_CYLINDER
    assert "shadow_shape=unknown" in top.provenance
    footprint_aspect = max(top.dimensions_m.x, top.dimensions_m.y) / min(
        top.dimensions_m.x,
        top.dimensions_m.y,
    )
    assert footprint_aspect <= 1.08


def test_shadow_unknown_bbox_squares_near_square_support_before_returning():
    module = load_module()
    component_mask = np.zeros((40, 40), dtype=bool)
    component_mask[10:31, 9:32] = True
    component_mask[2:7, 5:36] = True
    component = module.PixelComponent(
        component_id=7,
        area_px=int(component_mask.sum()),
        bbox_xyxy=(5, 2, 35, 30),
        centroid_uv=(20.0, 20.0),
        mask=component_mask,
    )
    fit = module.GeometryFit(
        component_id=7,
        shape_type="box",
        center_xy_m=(0.0, 0.0),
        size_x_m=0.040,
        size_y_m=0.025,
        size_z_m=0.020,
        yaw_rad=math.radians(75.0),
        provenance="test",
        bottom_z_m=0.75,
        center_z_m=0.76,
    )
    support = module.TableFootprint(
        center_xy_m=(0.01, 0.02),
        size_x_m=0.024,
        size_y_m=0.017,
        width_px=23.0,
        height_px=21.0,
        yaw_rad=math.radians(75.0),
    )

    coerced = module.coerce_projected_shadow_fit_to_unknown_bbox(
        fit,
        "shadow_trimmed",
        support,
        component,
    )

    assert coerced.shape_type == "bbox"
    assert coerced.size_x_m == coerced.size_y_m == 0.024
    assert abs(math.degrees(coerced.yaw_rad)) <= 45.0


def test_shadow_unknown_bbox_uses_full_mask_yaw_for_squareish_support():
    module = load_module()
    component_mask = np.zeros((44, 44), dtype=bool)
    component_mask[12:33, 11:34] = True
    component = module.PixelComponent(
        component_id=8,
        area_px=int(component_mask.sum()),
        bbox_xyxy=(11, 12, 33, 32),
        centroid_uv=(22.0, 22.0),
        mask=component_mask,
    )
    fit = module.GeometryFit(
        component_id=8,
        shape_type="box",
        center_xy_m=(0.0, 0.0),
        size_x_m=0.041,
        size_y_m=0.030,
        size_z_m=0.020,
        yaw_rad=math.radians(61.0),
        provenance="test",
        bottom_z_m=0.75,
        center_z_m=0.76,
    )
    compact_support = module.TableFootprint(
        center_xy_m=(0.012, 0.018),
        size_x_m=0.024,
        size_y_m=0.019,
        width_px=23.0,
        height_px=20.0,
        yaw_rad=math.radians(61.0),
    )
    full_mask_support = module.TableFootprint(
        center_xy_m=(0.010, 0.019),
        size_x_m=0.043,
        size_y_m=0.031,
        width_px=63.0,
        height_px=72.0,
        yaw_rad=math.radians(45.0),
    )

    coerced = module.coerce_projected_shadow_fit_to_unknown_bbox(
        fit,
        "shadow_trimmed",
        compact_support,
        component,
        full_mask_support,
    )

    assert coerced.shape_type == "bbox"
    assert coerced.center_xy_m == compact_support.center_xy_m
    assert coerced.size_x_m == coerced.size_y_m == max(
        compact_support.size_x_m,
        compact_support.size_y_m,
    )
    assert abs(module.normalize_yaw_half_turn(coerced.yaw_rad - math.radians(45.0))) <= math.radians(2.0)
    assert "full_mask_yaw" in coerced.provenance


def test_shadow_unknown_bbox_uses_full_mask_yaw_when_shadow_stretches_component_bbox():
    module = load_module()
    component_mask = np.zeros((64, 72), dtype=bool)
    component_mask[18:32, 16:34] = True
    component_mask[23:29, 34:56] = True
    component = module.PixelComponent(
        component_id=9,
        area_px=int(component_mask.sum()),
        bbox_xyxy=(16, 18, 55, 39),
        centroid_uv=(35.5, 28.5),
        mask=component_mask,
    )
    fit = module.GeometryFit(
        component_id=9,
        shape_type="box",
        center_xy_m=(0.0, 0.0),
        size_x_m=0.084,
        size_y_m=0.050,
        size_z_m=0.040,
        yaw_rad=math.radians(61.0),
        provenance="test",
        bottom_z_m=0.75,
        center_z_m=0.76,
    )
    compact_support = module.TableFootprint(
        center_xy_m=(0.034, 0.060),
        size_x_m=0.063,
        size_y_m=0.050,
        width_px=63.0,
        height_px=50.0,
        yaw_rad=math.radians(61.0),
    )
    full_mask_support = module.TableFootprint(
        center_xy_m=(0.034, 0.060),
        size_x_m=0.084,
        size_y_m=0.052,
        width_px=84.0,
        height_px=52.0,
        yaw_rad=math.radians(45.0),
    )

    coerced = module.coerce_projected_shadow_fit_to_unknown_bbox(
        fit,
        "shadow_trimmed",
        compact_support,
        component,
        full_mask_support,
    )

    assert coerced.shape_type == "bbox"
    assert coerced.center_xy_m == compact_support.center_xy_m
    assert coerced.size_x_m == compact_support.size_x_m
    assert coerced.size_y_m == compact_support.size_y_m
    assert abs(module.normalize_yaw_half_turn(coerced.yaw_rad - math.radians(45.0))) <= math.radians(2.0)
    assert "full_mask_yaw" in coerced.provenance


def test_shadow_unknown_bbox_squares_cube_like_compact_support_with_low_fill_shadow():
    module = load_module()
    component_mask = np.zeros((64, 72), dtype=bool)
    component_mask[18:32, 16:34] = True
    component_mask[23:29, 34:56] = True
    component = module.PixelComponent(
        component_id=10,
        area_px=int(component_mask.sum()),
        bbox_xyxy=(16, 18, 55, 31),
        centroid_uv=(30.0, 25.0),
        mask=component_mask,
    )
    fit = module.GeometryFit(
        component_id=10,
        shape_type="box",
        center_xy_m=(0.0, 0.0),
        size_x_m=0.046,
        size_y_m=0.031,
        size_z_m=0.0188,
        yaw_rad=math.radians(61.0),
        provenance="test",
        bottom_z_m=0.75,
        center_z_m=0.7594,
    )
    compact_support = module.TableFootprint(
        center_xy_m=(0.034, 0.060),
        size_x_m=0.0245,
        size_y_m=0.0187,
        width_px=24.0,
        height_px=19.0,
        yaw_rad=math.radians(61.0),
    )
    full_mask_support = module.TableFootprint(
        center_xy_m=(0.034, 0.060),
        size_x_m=0.045,
        size_y_m=0.029,
        width_px=70.0,
        height_px=48.0,
        yaw_rad=math.radians(45.0),
    )

    coerced = module.coerce_projected_shadow_fit_to_unknown_bbox(
        fit,
        "shadow_trimmed",
        compact_support,
        component,
        full_mask_support,
    )

    assert coerced.shape_type == "bbox"
    assert coerced.center_xy_m == compact_support.center_xy_m
    assert coerced.size_x_m == coerced.size_y_m == compact_support.size_x_m
    assert abs(module.normalize_yaw_half_turn(coerced.yaw_rad - math.radians(45.0))) <= math.radians(2.0)
    assert "full_mask_yaw" in coerced.provenance


def test_shadow_unknown_bbox_snaps_weak_square_yaw_to_table_axis():
    module = load_module()
    component_mask = np.zeros((80, 100), dtype=bool)
    component_mask[30:52, 42:66] = True
    component = module.PixelComponent(
        component_id=1,
        area_px=int(component_mask.sum()),
        bbox_xyxy=(42, 30, 65, 51),
        centroid_uv=(53.5, 40.5),
        mask=component_mask,
    )
    fit = module.GeometryFit(
        component_id=1,
        shape_type="box",
        center_xy_m=(0.0, 0.0),
        size_x_m=0.030,
        size_y_m=0.026,
        size_z_m=0.025,
        yaw_rad=math.radians(-14.0),
        provenance="test",
        bottom_z_m=0.74,
        center_z_m=0.7525,
    )
    compact_support = module.TableFootprint(
        center_xy_m=(0.01, 0.02),
        size_x_m=0.0255,
        size_y_m=0.0212,
        width_px=18.0,
        height_px=17.0,
        yaw_rad=math.radians(-13.0),
    )
    full_mask_support = module.TableFootprint(
        center_xy_m=(0.01, 0.02),
        size_x_m=0.040,
        size_y_m=0.029,
        width_px=28.0,
        height_px=21.0,
        yaw_rad=math.radians(-16.5),
    )

    coerced = module.coerce_projected_shadow_fit_to_unknown_bbox(
        fit,
        "shadow_trimmed",
        compact_support,
        component,
        full_mask_support,
    )

    assert abs(module.normalize_yaw_half_turn(coerced.yaw_rad)) <= math.radians(1.0)
    assert "weak_square_yaw_axis_snap" in coerced.provenance


def test_shadow_unknown_bbox_uses_round_corner_evidence_for_moved_cylinder():
    module = load_module()
    rows, cols = np.indices((80, 100))
    component_mask = ((cols - 50.0) / 11.0) ** 2 + ((rows - 40.0) / 13.0) ** 2 <= 1.0
    component_mask[25:32, 42:58] = True
    component = module.PixelComponent(
        component_id=3,
        area_px=int(component_mask.sum()),
        bbox_xyxy=(39, 25, 62, 54),
        centroid_uv=(50.0, 40.0),
        mask=component_mask,
    )
    fit = module.GeometryFit(
        component_id=3,
        shape_type="box",
        center_xy_m=(0.0, 0.0),
        size_x_m=0.030,
        size_y_m=0.026,
        size_z_m=0.025,
        yaw_rad=math.radians(-21.0),
        provenance="test",
        bottom_z_m=0.74,
        center_z_m=0.7525,
    )
    compact_support = module.TableFootprint(
        center_xy_m=(0.0568, 0.0109),
        size_x_m=0.0207,
        size_y_m=0.0241,
        width_px=21.0,
        height_px=25.0,
        yaw_rad=math.radians(-24.0),
    )
    full_mask_support = module.TableFootprint(
        center_xy_m=(0.0445, 0.0159),
        size_x_m=0.0338,
        size_y_m=0.0222,
        width_px=34.0,
        height_px=24.0,
        yaw_rad=math.radians(-21.5),
    )

    coerced = module.coerce_projected_shadow_fit_to_unknown_bbox(
        fit,
        "shadow_trimmed",
        compact_support,
        component,
        full_mask_support,
    )

    assert coerced.shape_type == "cylinder"
    assert coerced.size_x_m == coerced.size_y_m
    assert coerced.center_xy_m == compact_support.center_xy_m
    assert "shadow_shape=mask_round_cylinder" in coerced.provenance


def test_shadow_unknown_bbox_accepts_round_metric_support_with_mild_pixel_skew():
    module = load_module()
    rows, cols = np.indices((80, 100))
    component_mask = ((cols - 50.0) / 11.0) ** 2 + ((rows - 40.0) / 13.0) ** 2 <= 1.0
    component_mask[25:32, 42:58] = True
    component = module.PixelComponent(
        component_id=3,
        area_px=int(component_mask.sum()),
        bbox_xyxy=(39, 25, 62, 54),
        centroid_uv=(50.0, 40.0),
        mask=component_mask,
    )
    fit = module.GeometryFit(
        component_id=3,
        shape_type="box",
        center_xy_m=(0.0, 0.0),
        size_x_m=0.030,
        size_y_m=0.026,
        size_z_m=0.025,
        yaw_rad=math.radians(-21.0),
        provenance="test",
        bottom_z_m=0.74,
        center_z_m=0.7525,
    )
    compact_support = module.TableFootprint(
        center_xy_m=(0.0345, 0.0479),
        size_x_m=0.0248,
        size_y_m=0.0225,
        width_px=21.0,
        height_px=28.0,
        yaw_rad=math.radians(-18.5),
    )
    full_mask_support = module.TableFootprint(
        center_xy_m=(0.0290, 0.0503),
        size_x_m=0.0355,
        size_y_m=0.0223,
        width_px=36.0,
        height_px=25.0,
        yaw_rad=math.radians(-24.5),
    )

    coerced = module.coerce_projected_shadow_fit_to_unknown_bbox(
        fit,
        "shadow_trimmed",
        compact_support,
        component,
        full_mask_support,
    )

    assert coerced.shape_type == "cylinder"
    assert coerced.size_x_m == coerced.size_y_m
    assert "shadow_shape=mask_round_cylinder" in coerced.provenance


def test_shadow_unknown_bbox_accepts_high_fill_round_metric_support():
    module = load_module()
    component_mask = np.zeros((80, 100), dtype=bool)
    component_mask[25:54, 39:62] = True
    component_mask[25:30, 39:44] = False
    component_mask[25:30, 57:62] = False
    component_mask[49:54, 39:44] = False
    component_mask[49:54, 57:62] = False
    component = module.PixelComponent(
        component_id=3,
        area_px=int(component_mask.sum()),
        bbox_xyxy=(39, 25, 61, 53),
        centroid_uv=(50.0, 40.0),
        mask=component_mask,
    )
    fit = module.GeometryFit(
        component_id=3,
        shape_type="box",
        center_xy_m=(0.0, 0.0),
        size_x_m=0.030,
        size_y_m=0.026,
        size_z_m=0.025,
        yaw_rad=math.radians(-21.0),
        provenance="test",
        bottom_z_m=0.74,
        center_z_m=0.7525,
    )
    compact_support = module.TableFootprint(
        center_xy_m=(0.0345, 0.0479),
        size_x_m=0.0248,
        size_y_m=0.0225,
        width_px=22.0,
        height_px=22.0,
        yaw_rad=math.radians(-18.5),
    )
    full_mask_support = module.TableFootprint(
        center_xy_m=(0.0290, 0.0503),
        size_x_m=0.0355,
        size_y_m=0.0223,
        width_px=36.0,
        height_px=25.0,
        yaw_rad=math.radians(-24.5),
    )

    coerced = module.coerce_projected_shadow_fit_to_unknown_bbox(
        fit,
        "shadow_trimmed",
        compact_support,
        component,
        full_mask_support,
    )

    assert module.component_bbox_fill_ratio(component) >= 0.84
    assert module.component_bbox_aspect(component) > 1.25
    assert coerced.shape_type == "cylinder"
    assert "shadow_shape=mask_round_cylinder" in coerced.provenance


def test_build_live_hypothesis_array_uses_bbox_for_irregular_tabletop_component_without_truth():
    module = load_module()
    raw_depth = np.full((100, 140), 0.250, dtype=np.float32)
    mask = np.zeros_like(raw_depth, dtype=bool)
    mask[36:62, 38:48] = True
    mask[54:64, 38:70] = True
    mask[42:58, 96:112] = True
    raw_depth[mask] = 0.225

    message = module.build_live_hypothesis_array(
        raw_depth_m=raw_depth,
        corrupted_depth_m=raw_depth,
        target_mask=mask,
        intrinsics=module.CameraModel(width=140, height=100, fx=140.0, fy=140.0, cx=70.0, cy=50.0),
        frame_id="d435_depth_optical_frame",
        base_frame_id="world",
        camera_to_base_transform=downward_camera_to_world_transform(),
        table_z_m=0.75,
        primitive_height_m=0.04,
        top_k=1,
    )

    assert len(message.hypotheses) == 2
    irregular = max(message.hypotheses, key=lambda hypothesis: hypothesis.dimensions_m.x)
    assert irregular.shape_type == module.GeometryHypothesis.SHAPE_BOX
    assert "fit=bbox" in irregular.provenance
    forbidden_tokens = ("gazebo", "gz model", "sdf", "model://", "nurse_abstract_target")
    for hypothesis in message.hypotheses:
        provenance = hypothesis.provenance.lower()
        assert not any(token in provenance for token in forbidden_tokens)


def make_depth_msg(depth):
    msg = Image()
    msg.header.frame_id = "d435_depth_optical_frame"
    msg.height, msg.width = depth.shape
    msg.encoding = "32FC1"
    msg.is_bigendian = False
    msg.step = msg.width * 4
    msg.data = np.asarray(depth, dtype="<f4").tobytes()
    return msg


def make_color_msg(rgb):
    msg = Image()
    msg.header.frame_id = "d435_color_optical_frame"
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = msg.width * 3
    msg.data = np.asarray(rgb, dtype=np.uint8).reshape(-1).tolist()
    return msg


def make_mask_msg(mask):
    msg = Image()
    msg.header.frame_id = "d435_depth_optical_frame"
    msg.height, msg.width = mask.shape
    msg.encoding = "mono8"
    msg.is_bigendian = False
    msg.step = msg.width
    msg.data = np.asarray(mask, dtype=np.uint8).reshape(-1).tolist()
    return msg


class CapturePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def test_node_publishes_target_mask_from_raw_depth_before_corrupted_depth():
    module = load_module()
    rclpy.init()
    node = None
    try:
        node = module.M4LiveHypothesisPublisherNode()
        capture = CapturePublisher()
        node.mask_pub = capture

        node.handle_raw_depth(make_depth_msg(with_foreground_patch(40, 42)))

        assert len(capture.messages) == 1
        assert capture.messages[0].encoding == "mono8"
        assert any(capture.messages[0].data)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_node_locks_initial_target_mask_center_against_far_candidate():
    module = load_module()
    rclpy.init()
    node = None
    try:
        node = module.M4LiveHypothesisPublisherNode()
        node.max_locked_center_distance_px = 28.0
        capture = CapturePublisher()
        node.mask_pub = capture

        first_depth = sloped_table_depth(height=120, width=160)
        first_depth[58:70, 44:58] -= 0.10
        node.handle_raw_depth(make_depth_msg(first_depth))

        second_depth = first_depth.copy()
        second_depth[70:86, 122:142] -= 0.16
        node.handle_raw_depth(make_depth_msg(second_depth))

        assert node.locked_target_center_uv is not None
        assert len(capture.messages) == 2
        data = np.asarray(capture.messages[-1].data, dtype=np.uint8).reshape(
            capture.messages[-1].height,
            capture.messages[-1].step,
        )[:, : capture.messages[-1].width]
        mask = data > 0
        center_u, center_v = mask_center(mask)
        assert 48.0 <= center_u <= 54.0
        assert 62.0 <= center_v <= 67.0
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_node_does_not_reprocess_same_corrupted_frame_on_timer():
    module = load_module()
    rclpy.init()
    node = None
    try:
        node = module.M4LiveHypothesisPublisherNode()
        node.hypothesis_pub = CapturePublisher()
        node.latest_raw_depth = with_foreground_patch(40, 42)
        node.latest_raw_header = make_depth_msg(node.latest_raw_depth).header
        node.latest_target_mask = node.latest_raw_depth < sloped_table_depth()
        node.latest_camera_model = module.CameraModel(
            width=node.latest_raw_depth.shape[1],
            height=node.latest_raw_depth.shape[0],
            fx=120.0,
            fy=120.0,
            cx=50.0,
            cy=40.0,
        )
        transform = TransformStamped()
        transform.transform.rotation.w = 1.0
        node.lookup_camera_to_base_transform = lambda: transform

        build_calls = []

        def fake_build_live_hypothesis_array(**kwargs):
            build_calls.append(kwargs)
            return module.GeometryHypothesisArray()

        original_builder = module.build_live_hypothesis_array
        module.build_live_hypothesis_array = fake_build_live_hypothesis_array
        try:
            node.handle_corrupted_depth(make_depth_msg(node.latest_raw_depth))
            node.try_publish()
            node.try_publish()
            node.handle_corrupted_depth(make_depth_msg(node.latest_raw_depth))
        finally:
            module.build_live_hypothesis_array = original_builder

        assert len(build_calls) == 2
        assert len(node.hypothesis_pub.messages) == 2
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_table_foreground_gate_keeps_only_workspace_height_supported_components():
    module = load_module()
    depth = np.full((60, 80), np.nan, dtype=np.float32)
    depth[20:30, 20:30] = 0.08
    depth[22:30, 50:58] = 0.40
    mask = np.zeros(depth.shape, dtype=bool)
    mask[20:30, 20:30] = True
    mask[22:30, 50:58] = True

    transform = TransformStamped()
    transform.transform.rotation.w = 1.0
    camera_model = module.CameraModel(width=80, height=60, fx=100.0, fy=100.0, cx=40.0, cy=30.0)

    gated = module.table_foreground_supported_mask(
        mask=mask,
        depth_m=depth,
        intrinsics=camera_model,
        camera_to_base_transform=transform,
        table_z_m=0.0,
        min_height_m=0.01,
        max_height_m=0.20,
        min_component_area_px=20,
        workspace_min_x_m=-0.10,
        workspace_max_x_m=0.02,
        workspace_min_y_m=-0.02,
        workspace_max_y_m=0.02,
    )

    assert gated[20:30, 20:30].any()
    assert not gated[22:30, 50:58].any()


def test_stable_hypothesis_gate_tracks_pose_before_shape_and_hysteresis_holds_shape():
    module = load_module()
    tracks = []
    first = module.GeometryHypothesisArray()
    first.backend_name = "ghost_mgg_m4_live"
    first.hypotheses = [
        _hypothesis_at(module, "stable_candidate", module.GeometryHypothesis.SHAPE_CYLINDER, 0.10, 0.20),
        _hypothesis_at(module, "single_frame_noise", module.GeometryHypothesis.SHAPE_BOX, -0.30, 0.10),
    ]

    filtered, tracks = module.filter_stable_hypothesis_array(
        first,
        tracks,
        min_observations=2,
        max_center_jump_m=0.04,
        max_misses=1,
        shape_switch_observations=3,
        smoothing_alpha=0.5,
    )
    assert filtered.hypotheses == []

    second = module.GeometryHypothesisArray()
    second.backend_name = "ghost_mgg_m4_live"
    second.hypotheses = [
        _hypothesis_at(module, "stable_candidate_moved", module.GeometryHypothesis.SHAPE_BOX, 0.11, 0.195),
        _hypothesis_at(module, "new_single_frame_noise", module.GeometryHypothesis.SHAPE_CYLINDER, -0.10, -0.20),
    ]

    filtered, tracks = module.filter_stable_hypothesis_array(
        second,
        tracks,
        min_observations=2,
        max_center_jump_m=0.04,
        max_misses=1,
        shape_switch_observations=3,
        smoothing_alpha=0.5,
    )

    assert len(filtered.hypotheses) == 1
    assert filtered.hypotheses[0].hypothesis_id == "track_1_cylinder"
    assert filtered.hypotheses[0].shape_type == module.GeometryHypothesis.SHAPE_CYLINDER
    assert 0.104 <= filtered.hypotheses[0].pose_base.pose.position.x <= 0.106


def test_stable_hypothesis_gate_damps_dimension_spike_without_slowing_center():
    module = load_module()
    tracks = []
    first = module.GeometryHypothesisArray()
    first.hypotheses = [
        _hypothesis_at(module, "stable_box", module.GeometryHypothesis.SHAPE_BOX, 0.10, 0.20)
    ]
    first.hypotheses[0].dimensions_m.x = 0.040
    first.hypotheses[0].dimensions_m.y = 0.040
    first.hypotheses[0].dimensions_m.z = 0.050

    filtered, tracks = module.filter_stable_hypothesis_array(
        first,
        tracks,
        min_observations=1,
        max_center_jump_m=0.08,
        max_misses=1,
        shape_switch_observations=5,
        smoothing_alpha=0.50,
        dimension_smoothing_alpha=0.10,
        dimension_max_step_ratio=0.20,
    )
    assert len(filtered.hypotheses) == 1

    second = module.GeometryHypothesisArray()
    second.hypotheses = [
        _hypothesis_at(module, "stable_box_spike", module.GeometryHypothesis.SHAPE_BOX, 0.12, 0.20)
    ]
    second.hypotheses[0].dimensions_m.x = 0.040
    second.hypotheses[0].dimensions_m.y = 0.120
    second.hypotheses[0].dimensions_m.z = 0.050

    filtered, tracks = module.filter_stable_hypothesis_array(
        second,
        tracks,
        min_observations=1,
        max_center_jump_m=0.08,
        max_misses=1,
        shape_switch_observations=5,
        smoothing_alpha=0.50,
        dimension_smoothing_alpha=0.10,
        dimension_max_step_ratio=0.20,
    )

    assert len(filtered.hypotheses) == 1
    top = filtered.hypotheses[0]
    assert 0.109 <= top.pose_base.pose.position.x <= 0.111
    assert top.dimensions_m.y < 0.050


def test_stable_component_mask_filters_one_frame_noise_before_geometry_fit():
    module = load_module()
    tracks = []
    transform = TransformStamped()
    transform.transform.rotation.w = 1.0
    camera_model = module.CameraModel(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0)

    first_mask = np.zeros((80, 100), dtype=bool)
    first_mask[30:40, 30:40] = True
    first_mask[10:20, 76:88] = True
    first_stable, tracks = module.filter_stable_component_mask(
        first_mask,
        tracks,
        intrinsics=camera_model,
        camera_to_base_transform=transform,
        table_z_m=1.0,
        min_component_area_px=20,
        min_observations=2,
        max_center_jump_m=0.04,
        max_misses=1,
    )
    assert not first_stable.any()

    second_mask = np.zeros((80, 100), dtype=bool)
    second_mask[31:41, 31:41] = True
    second_mask[55:65, 72:84] = True
    second_stable, tracks = module.filter_stable_component_mask(
        second_mask,
        tracks,
        intrinsics=camera_model,
        camera_to_base_transform=transform,
        table_z_m=1.0,
        min_component_area_px=20,
        min_observations=2,
        max_center_jump_m=0.04,
        max_misses=1,
    )

    assert second_stable[31:41, 31:41].any()
    assert not second_stable[55:65, 72:84].any()


def _hypothesis_at(module, hypothesis_id, shape_type, x, y):
    hypothesis = module.GeometryHypothesis()
    hypothesis.hypothesis_id = hypothesis_id
    hypothesis.shape_type = shape_type
    hypothesis.pose_base = module.make_pose("base_link", x, y, 0.05, module.yaw_orientation(0.0))
    return hypothesis


def test_sparse_foreground_alignment_keeps_elongated_visible_cloud_unsquared():
    module = load_module()
    xs = np.linspace(-0.025, 0.025, 11)
    ys = np.linspace(-0.045, 0.045, 19)
    center = np.array([0.010, 0.150], dtype=np.float64)
    points = center + np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    fit = module.GeometryFit(
        component_id=1,
        shape_type="box",
        center_xy_m=(0.010, 0.150),
        size_x_m=0.050,
        size_y_m=0.090,
        size_z_m=0.065,
        yaw_rad=0.0,
        provenance="metric_points_oriented_footprint",
        bottom_z_m=0.0,
        center_z_m=0.032,
    )
    squareish_hole_support = module.TableFootprint(
        center_xy_m=(-0.020, 0.140),
        size_x_m=0.040,
        size_y_m=0.055,
        width_px=40.0,
        height_px=55.0,
        yaw_rad=0.0,
    )

    aligned = module.align_sparse_foreground_box_to_visible_edges(
        fit,
        "foreground",
        squareish_hole_support,
        points,
    )

    assert "foreground_edge_aligned" not in aligned.provenance
    assert abs(aligned.size_x_m - 0.050) < 1e-9
    assert abs(aligned.size_y_m - 0.090) < 1e-9
    assert abs(aligned.center_xy_m[0] - 0.010) < 1e-9


def test_component_hole_adjacency_fraction_flags_hole_rim_but_not_open_table():
    module = load_module()
    depth = np.full((60, 80), 1.0, dtype=np.float32)
    depth[10:40, 40:70] = 0.0

    rim_mask = np.zeros((60, 80), dtype=bool)
    rim_mask[20:30, 30:40] = True
    assert module.component_hole_adjacency_fraction(rim_mask, depth) >= 0.20

    open_table_mask = np.zeros((60, 80), dtype=bool)
    open_table_mask[45:55, 5:15] = True
    assert module.component_hole_adjacency_fraction(open_table_mask, depth) < 0.05


def test_hole_adjacent_silhouette_fit_becomes_existence_proxy():
    module = load_module()
    fit = module.GeometryFit(
        component_id=3,
        shape_type="box",
        center_xy_m=(0.020, 0.180),
        size_x_m=0.060,
        size_y_m=0.020,
        size_z_m=0.030,
        yaw_rad=0.4,
        provenance="mask_projection",
        bottom_z_m=0.0,
        center_z_m=0.015,
    )
    points = np.array(
        [[0.040, 0.190], [0.050, 0.200], [0.060, 0.210], [0.050, 0.210]],
        dtype=np.float64,
    )

    proxy = module.coerce_hole_adjacent_fit_to_existence_proxy(fit, points)

    assert proxy.shape_type == "bbox"
    assert abs(proxy.size_x_m - module.HOLE_EXISTENCE_PROXY_SIZE_M) < 1e-9
    assert abs(proxy.size_y_m - module.HOLE_EXISTENCE_PROXY_SIZE_M) < 1e-9
    assert proxy.yaw_rad == 0.0
    assert "hole_existence_only" in proxy.provenance
    assert abs(proxy.center_xy_m[0] - 0.050) < 1e-6
    assert abs(proxy.center_xy_m[1] - 0.205) < 1e-6


def test_stable_hypothesis_gate_blocks_merged_candidate_from_stealing_neighbor_track():
    module = load_module()
    tracks = []
    first = module.GeometryHypothesisArray()
    small = _hypothesis_at(module, "small", module.GeometryHypothesis.SHAPE_BOX, 0.0, 0.20)
    small.dimensions_m.x = 0.040
    small.dimensions_m.y = 0.016
    small.dimensions_m.z = 0.020
    big = _hypothesis_at(module, "big", module.GeometryHypothesis.SHAPE_BOX, 0.078, 0.20)
    big.dimensions_m.x = 0.088
    big.dimensions_m.y = 0.088
    big.dimensions_m.z = 0.065
    first.hypotheses = [small, big]

    filtered, tracks = module.filter_stable_hypothesis_array(
        first,
        tracks,
        min_observations=1,
        max_center_jump_m=0.08,
        max_misses=2,
        shape_switch_observations=5,
        smoothing_alpha=0.45,
        dimension_smoothing_alpha=0.12,
        dimension_max_step_ratio=0.20,
    )
    assert len(filtered.hypotheses) == 2
    assert len(tracks) == 2
    small_track_id = tracks[0].track_id
    big_track_id = tracks[1].track_id

    second = module.GeometryHypothesisArray()
    merged = _hypothesis_at(module, "merged", module.GeometryHypothesis.SHAPE_BOX, 0.030, 0.20)
    merged.dimensions_m.x = 0.086
    merged.dimensions_m.y = 0.086
    merged.dimensions_m.z = 0.065
    second.hypotheses = [merged]

    filtered, tracks = module.filter_stable_hypothesis_array(
        second,
        tracks,
        min_observations=1,
        max_center_jump_m=0.08,
        max_misses=2,
        shape_switch_observations=5,
        smoothing_alpha=0.45,
        dimension_smoothing_alpha=0.12,
        dimension_max_step_ratio=0.20,
    )

    assert len(filtered.hypotheses) == 1
    assert filtered.hypotheses[0].hypothesis_id == f"track_{big_track_id}_box"
    surviving_small = [t for t in tracks if t.track_id == small_track_id]
    assert len(surviving_small) == 1
    assert surviving_small[0].misses == 1
    assert abs(surviving_small[0].center_xy_m[0] - 0.0) < 1e-9
    assert abs(surviving_small[0].size_xyz_m[0] - 0.040) < 1e-9


def test_drop_low_evidence_hypotheses_keeps_existence_proxy_and_scored_fits():
    module = load_module()
    message = module.GeometryHypothesisArray()
    good = _hypothesis_at(module, "good", module.GeometryHypothesis.SHAPE_BOX, 0.1, 0.2)
    good.score.total = 0.8
    noise = _hypothesis_at(module, "noise", module.GeometryHypothesis.SHAPE_BOX, -0.1, 0.2)
    noise.score.total = 0.0
    noise.provenance = "support=mask_full"
    proxy = _hypothesis_at(module, "proxy", module.GeometryHypothesis.SHAPE_BOX, -0.2, 0.2)
    proxy.score.total = 0.0
    proxy.provenance = "support=hole_evidence;hole_existence_only"
    message.hypotheses = [good, noise, proxy]

    kept = module.drop_low_evidence_hypotheses(message)

    assert [h.hypothesis_id for h in kept.hypotheses] == ["good", "proxy"]


def test_stable_hypothesis_gate_holds_established_track_through_brief_dropout():
    module = load_module()
    tracks = []
    for _ in range(4):
        frame = module.GeometryHypothesisArray()
        frame.header.frame_id = "base_link"
        steady = _hypothesis_at(module, "steady", module.GeometryHypothesis.SHAPE_BOX, 0.05, 0.20)
        steady.dimensions_m.x = 0.040
        steady.dimensions_m.y = 0.016
        steady.dimensions_m.z = 0.020
        frame.hypotheses = [steady]
        filtered, tracks = module.filter_stable_hypothesis_array(
            frame,
            tracks,
            min_observations=1,
            max_center_jump_m=0.08,
            max_misses=2,
            shape_switch_observations=5,
            smoothing_alpha=0.45,
            dimension_smoothing_alpha=0.12,
            dimension_max_step_ratio=0.20,
            output_hold_misses=2,
        )
    assert len(filtered.hypotheses) == 1
    track_id = tracks[0].track_id

    empty = module.GeometryHypothesisArray()
    empty.header.frame_id = "base_link"
    filtered, tracks = module.filter_stable_hypothesis_array(
        empty,
        tracks,
        min_observations=1,
        max_center_jump_m=0.08,
        max_misses=2,
        shape_switch_observations=5,
        smoothing_alpha=0.45,
        dimension_smoothing_alpha=0.12,
        dimension_max_step_ratio=0.20,
        output_hold_misses=2,
    )

    assert len(filtered.hypotheses) == 1
    held = filtered.hypotheses[0]
    assert held.hypothesis_id == f"track_{track_id}_box"
    assert "stable_output_hold" in held.provenance
    assert abs(held.pose_base.pose.position.x - 0.05) < 1e-6
    assert abs(held.dimensions_m.x - 0.040) < 1e-6
    assert held.grasp_candidates == []


def _seed_split_scene(module):
    height, width = 60, 80
    z = np.full((height, width), np.nan)
    z[45:55, :] = 0.0  # visible table strip
    points = np.zeros((height, width, 3))
    xs = np.linspace(-0.10, 0.30, width)[None, :]
    ys = np.linspace(0.30, 0.05, height)[:, None]
    points[:, :, 0] = xs
    points[:, :, 1] = ys
    mask = np.zeros((height, width), dtype=bool)
    # two 20x20 object blobs bridged by a 3px-tall smear band => one component
    mask[15:35, 10:30] = True
    mask[15:35, 40:60] = True
    mask[24:27, 30:40] = True
    z[mask] = 0.05
    points[:, :, 2] = z
    component = module.pixel_component_from_mask(1, mask)
    return points, z, component


def test_seed_split_divides_hole_merged_component_between_recent_tracks():
    module = load_module()
    points, z, component = _seed_split_scene(module)
    seed_a = (float(points[25, 20, 0]), float(points[25, 20, 1]))
    seed_b = (float(points[25, 50, 0]), float(points[25, 50, 1]))

    split, seeded_ids = module.split_components_by_recent_track_seeds(
        [component],
        points_base=points,
        z_map=z,
        table_z_m=0.0,
        seed_centers_xy=[seed_a, seed_b],
    )

    assert len(split) == 2
    assert seeded_ids == {1, 2}
    areas = sorted(int(c.area_px) for c in split)
    assert areas[0] >= 400
    centroids_u = sorted(c.centroid_uv[0] for c in split)
    assert centroids_u[0] < 32 and centroids_u[1] > 38


def test_seed_split_leaves_component_alone_without_two_reachable_seeds():
    module = load_module()
    points, z, component = _seed_split_scene(module)
    seed_a = (float(points[25, 20, 0]), float(points[25, 20, 1]))
    far_seed = (5.0, 5.0)

    split, seeded_ids = module.split_components_by_recent_track_seeds(
        [component],
        points_base=points,
        z_map=z,
        table_z_m=0.0,
        seed_centers_xy=[seed_a, far_seed],
    )

    assert len(split) == 1
    assert seeded_ids == set()
    assert int(split[0].area_px) == int(component.area_px)


def test_seed_split_rejects_when_clusters_collapse_onto_one_object():
    module = load_module()
    height, width = 60, 80
    z = np.full((height, width), np.nan)
    points = np.zeros((height, width, 3))
    xs = np.linspace(-0.10, 0.30, width)[None, :]
    ys = np.linspace(0.30, 0.05, height)[:, None]
    points[:, :, 0] = xs
    points[:, :, 1] = ys
    mask = np.zeros((height, width), dtype=bool)
    mask[15:35, 10:30] = True  # single lone object
    z[mask] = 0.05
    points[:, :, 2] = z
    component = module.pixel_component_from_mask(1, mask)
    inside = (float(points[25, 18, 0]), float(points[25, 18, 1]))
    nearby = (float(points[25, 33, 0]), float(points[25, 33, 1]))

    split, seeded_ids = module.split_components_by_recent_track_seeds(
        [component],
        points_base=points,
        z_map=z,
        table_z_m=0.0,
        seed_centers_xy=[inside, nearby],
    )

    assert len(split) == 1
    assert seeded_ids == set()


def _sized_hypothesis(module, hid, shape, x, y, sx, sy, sz, score=0.9):
    h = _hypothesis_at(module, hid, shape, x, y)
    h.dimensions_m.x = sx
    h.dimensions_m.y = sy
    h.dimensions_m.z = sz
    h.score.total = score
    return h


def _run_filter(module, frame_hyps, tracks):
    frame = module.GeometryHypothesisArray()
    frame.header.frame_id = "base_link"
    frame.hypotheses = frame_hyps
    return module.filter_stable_hypothesis_array(
        frame, tracks, min_observations=1, max_center_jump_m=0.08, max_misses=2,
        shape_switch_observations=8, smoothing_alpha=0.45,
        dimension_smoothing_alpha=0.12, dimension_max_step_ratio=0.20,
        output_hold_misses=2,
    )


def test_poisoned_track_relocks_to_consistent_smaller_cylinder_within_frames():
    module = load_module()
    tracks = []
    # birth from a merged transitional fit: huge box
    poisoned = _sized_hypothesis(
        module, "merged", module.GeometryHypothesis.SHAPE_BOX, 0.0, 0.20, 0.129, 0.035, 0.065
    )
    filtered, tracks = _run_filter(module, [poisoned], tracks)
    assert filtered.hypotheses[0].dimensions_m.x > 0.12

    # afterwards the real object produces clean small cylinder fits every frame
    for frame_index in range(1, 5):
        cyl = _sized_hypothesis(
            module, "cyl", module.GeometryHypothesis.SHAPE_CYLINDER, 0.0, 0.20,
            0.027, 0.027, 0.065, score=0.9,
        )
        filtered, tracks = _run_filter(module, [cyl], tracks)
    top = filtered.hypotheses[0]
    assert top.dimensions_m.x <= 0.030  # snapped, not slowly unwinding
    assert top.shape_type == module.GeometryHypothesis.SHAPE_CYLINDER
    assert "stable_fast_relock" in top.provenance or top.dimensions_m.x <= 0.030


def test_dimension_hold_escapes_when_larger_size_persists_without_neighbor():
    module = load_module()
    tracks = []
    small = _sized_hypothesis(
        module, "small", module.GeometryHypothesis.SHAPE_BOX, 0.0, 0.20, 0.030, 0.030, 0.03
    )
    filtered, tracks = _run_filter(module, [small], tracks)

    last = None
    for frame_index in range(1, 8):
        big = _sized_hypothesis(
            module, "big", module.GeometryHypothesis.SHAPE_BOX, 0.0, 0.20, 0.090, 0.090, 0.03
        )
        last, tracks = _run_filter(module, [big], tracks)
    assert last.hypotheses[0].dimensions_m.x >= 0.085  # escaped the freeze


def test_dimension_hold_stays_frozen_when_neighbor_track_explains_expansion():
    module = load_module()
    tracks = []
    a = _sized_hypothesis(
        module, "a", module.GeometryHypothesis.SHAPE_BOX, 0.0, 0.20, 0.030, 0.030, 0.03
    )
    b = _sized_hypothesis(
        module, "b", module.GeometryHypothesis.SHAPE_BOX, 0.07, 0.20, 0.030, 0.030, 0.03
    )
    filtered, tracks = _run_filter(module, [a, b], tracks)

    last = None
    for frame_index in range(1, 10):
        merged = _sized_hypothesis(
            module, "merged", module.GeometryHypothesis.SHAPE_BOX, 0.02, 0.20, 0.100, 0.090, 0.03
        )
        b_again = _sized_hypothesis(
            module, "b", module.GeometryHypothesis.SHAPE_BOX, 0.07, 0.20, 0.030, 0.030, 0.03
        )
        last, tracks = _run_filter(module, [b_again, merged], tracks)
    sizes = sorted(h.dimensions_m.x for h in last.hypotheses)
    assert sizes[0] <= 0.045  # neither track ballooned to the merged fit


def test_seed_split_carves_halo_when_only_one_seed_reaches_merged_component():
    module = load_module()
    points, z, component = _seed_split_scene(module)
    # only the LEFT object's track survives; the right object was just carried
    # here by hand and its stale seed is far outside the component
    seed_left = (float(points[25, 20, 0]), float(points[25, 20, 1]))
    stale_far_seed = (5.0, 5.0)

    split, seeded_ids = module.split_components_by_recent_track_seeds(
        [component],
        points_base=points,
        z_map=z,
        table_z_m=0.0,
        seed_centers_xy=[seed_left, stale_far_seed],
        seed_half_extents_m=[0.050, 0.014],
    )

    assert len(split) == 2
    assert seeded_ids == {1, 2}
    centroids_u = sorted(c.centroid_uv[0] for c in split)
    assert centroids_u[0] < 32 and centroids_u[1] > 38


def test_seed_split_single_seed_does_not_carve_dense_lone_object():
    module = load_module()
    height, width = 60, 80
    z = np.full((height, width), np.nan)
    points = np.zeros((height, width, 3))
    xs = np.linspace(-0.10, 0.30, width)[None, :]
    ys = np.linspace(0.30, 0.05, height)[:, None]
    points[:, :, 0] = xs
    points[:, :, 1] = ys
    mask = np.zeros((height, width), dtype=bool)
    mask[15:35, 10:50] = True  # one solid wide object (no valley inside)
    z[mask] = 0.05
    points[:, :, 2] = z
    component = module.pixel_component_from_mask(1, mask)
    seed_inside = (float(points[25, 15, 0]), float(points[25, 15, 1]))

    split, seeded_ids = module.split_components_by_recent_track_seeds(
        [component],
        points_base=points,
        z_map=z,
        table_z_m=0.0,
        seed_centers_xy=[seed_inside, (5.0, 5.0)],
        seed_half_extents_m=[0.012, 0.014],
    )

    assert len(split) == 1
    assert seeded_ids == set()


def test_merged_candidate_covering_two_established_tracks_never_births_a_track():
    module = load_module()
    tracks = []
    for _ in range(4):
        frame = module.GeometryHypothesisArray()
        frame.header.frame_id = "base_link"
        a = _sized_hypothesis(module, "a", module.GeometryHypothesis.SHAPE_BOX, 0.0, 0.20, 0.030, 0.030, 0.03)
        b = _sized_hypothesis(module, "b", module.GeometryHypothesis.SHAPE_BOX, 0.10, 0.20, 0.030, 0.030, 0.03)
        filtered, tracks = _run_filter(module, [a, b], tracks)
    known_ids = {t.track_id for t in tracks}

    merged = _sized_hypothesis(
        module, "merged", module.GeometryHypothesis.SHAPE_BOX, 0.05, 0.20, 0.160, 0.060, 0.05
    )
    filtered, tracks = _run_filter(module, [merged], tracks)

    assert {t.track_id for t in tracks} == known_ids  # no new track born
    held = {h.hypothesis_id for h in filtered.hypotheses}
    assert len(held) == 2  # both objects stay on display via holds


def test_ghost_seed_pool_remembers_dead_tracks_and_expires():
    module = load_module()
    dead = module.StableHypothesisTrack(
        shape_type=1, center_xy_m=(0.05, 0.10), observations=4,
        size_xyz_m=(0.06, 0.01, 0.02), track_id=3,
    )
    survivor = module.StableHypothesisTrack(
        shape_type=1, center_xy_m=(0.20, 0.20), observations=9,
        size_xyz_m=(0.03, 0.03, 0.05), track_id=1,
    )
    pool = module.update_ghost_seed_pool([], [dead, survivor], [survivor])
    assert len(pool) == 1
    assert pool[0]["center"] == (0.05, 0.10)
    assert abs(pool[0]["half_extent"] - 0.03) < 1e-9

    # a live track reclaiming the spot suppresses the ghost
    reborn = module.StableHypothesisTrack(
        shape_type=1, center_xy_m=(0.055, 0.105), observations=2,
        size_xyz_m=(0.06, 0.01, 0.02), track_id=7,
    )
    pool = module.update_ghost_seed_pool(pool, [survivor, reborn], [survivor, reborn])
    assert pool == []

    # ttl expiry
    pool = module.update_ghost_seed_pool([], [dead, survivor], [survivor])
    for _ in range(module.GHOST_SEED_TTL_PUBLISHES):
        pool = module.update_ghost_seed_pool(pool, [survivor], [survivor])
    assert pool == []


def test_one_seed_carve_keeps_sparse_outskirts_with_the_seed():
    module = load_module()
    height, width = 60, 80
    z = np.full((height, width), np.nan)
    points = np.zeros((height, width, 3))
    xs = np.linspace(-0.10, 0.30, width)[None, :]
    ys = np.linspace(0.30, 0.05, height)[:, None]
    points[:, :, 0] = xs
    points[:, :, 1] = ys
    mask = np.zeros((height, width), dtype=bool)
    mask[15:35, 10:30] = True   # left object body (seeded)
    mask[24:27, 4:10] = True    # sparse outskirts just left of the body
    mask[15:35, 40:60] = True   # newly arrived right object
    mask[24:27, 30:40] = True   # bridge
    z[mask] = 0.05
    points[:, :, 2] = z
    component = module.pixel_component_from_mask(1, mask)
    seed_left = (float(points[25, 20, 0]), float(points[25, 20, 1]))

    split, seeded_ids = module.split_components_by_recent_track_seeds(
        [component],
        points_base=points,
        z_map=z,
        table_z_m=0.0,
        seed_centers_xy=[seed_left, (5.0, 5.0)],
        seed_half_extents_m=[0.050, 0.014],
    )

    assert len(split) == 2
    left = min(split, key=lambda c: c.centroid_uv[0])
    left_mask = np.asarray(left.mask, dtype=bool)
    assert left_mask[25, 6]  # the sparse outskirts stayed with the seeded object


def test_shape_vote_window_converges_despite_interrupting_box_frames():
    module = load_module()
    tracks = []
    born_box = _sized_hypothesis(
        module, "b", module.GeometryHypothesis.SHAPE_BOX, 0.0, 0.20, 0.027, 0.025, 0.065
    )
    filtered, tracks = _run_filter(module, [born_box], tracks)

    # cylinder evidence interrupted by a box frame every third frame — the old
    # consecutive counter never converged on this pattern
    pattern = [
        module.GeometryHypothesis.SHAPE_CYLINDER,
        module.GeometryHypothesis.SHAPE_CYLINDER,
        module.GeometryHypothesis.SHAPE_BOX,
        module.GeometryHypothesis.SHAPE_CYLINDER,
        module.GeometryHypothesis.SHAPE_CYLINDER,
        module.GeometryHypothesis.SHAPE_BOX,
        module.GeometryHypothesis.SHAPE_CYLINDER,
    ]
    for shape in pattern:
        cand = _sized_hypothesis(module, "c", shape, 0.0, 0.20, 0.027, 0.027, 0.065, score=0.9)
        filtered, tracks = _run_filter(module, [cand], tracks)

    assert filtered.hypotheses[0].shape_type == module.GeometryHypothesis.SHAPE_CYLINDER


def test_established_track_survives_long_gap_with_same_identity():
    module = load_module()
    tracks = []
    for _ in range(30):
        h = _sized_hypothesis(
            module, "obj", module.GeometryHypothesis.SHAPE_BOX, -0.16, 0.03, 0.024, 0.011, 0.02
        )
        filtered, tracks = _run_filter(module, [h], tracks)
    veteran_id = tracks[0].track_id

    for _ in range(7):  # 7-frame detection gap (budget = 2 + 30//5 = 8)
        empty = module.GeometryHypothesisArray()
        empty.header.frame_id = "base_link"
        filtered, tracks = _run_filter(module, [], tracks)
        assert any(int(t.track_id) == veteran_id for t in tracks)

    h = _sized_hypothesis(
        module, "obj", module.GeometryHypothesis.SHAPE_BOX, -0.16, 0.03, 0.024, 0.011, 0.02
    )
    filtered, tracks = _run_filter(module, [h], tracks)
    assert filtered.hypotheses[0].hypothesis_id == f"track_{veteran_id}_box"


def _ring_scene(module, *, interior="table"):
    height, width = 80, 80
    z = np.full((height, width), np.nan)
    points = np.zeros((height, width, 3))
    xs = np.linspace(-0.10, 0.10, width)[None, :]
    ys = np.linspace(0.25, 0.05, height)[:, None]
    points[:, :, 0] = xs
    points[:, :, 1] = ys
    rows, cols = np.indices((height, width))
    cx_px, cy_px = 40, 40
    radius_px = 20  # ~50mm radius at 2.5mm/px
    rr = np.hypot(rows - cy_px, cols - cx_px)
    ring = (np.abs(rr - radius_px) <= 2.5)
    arc_top = ring & (rows < cy_px - 5)
    arc_bottom = ring & (rows > cy_px + 5)
    z[arc_top] = 0.05
    z[arc_bottom] = 0.05
    interior_mask = rr <= radius_px - 6
    if interior == "table":
        z[interior_mask] = 0.001
    elif interior == "elevated":
        z[interior_mask] = 0.05
    points[:, :, 2] = z
    comp_a = module.pixel_component_from_mask(1, arc_top)
    comp_b = module.pixel_component_from_mask(2, arc_bottom)
    hole = ~np.isfinite(z)
    return points, z, hole, [comp_a, comp_b]


def test_transparent_ring_merges_two_rim_arcs_into_one_cylinder():
    module = load_module()
    points, z, hole, comps = _ring_scene(module, interior="table")

    merged, ring_fits = module.merge_transparent_ring_components(
        comps, points_base=points, z_map=z, hole_mask=hole, table_z_m=0.0,
    )

    assert len(merged) == 1
    assert len(ring_fits) == 1
    (cx, cy, radius, height) = list(ring_fits.values())[0]
    assert abs(radius - 0.050) <= 0.008  # 20 px at ~2.5mm/px
    assert 0.03 <= height <= 0.07


def test_transparent_ring_accepts_elevated_interior_upside_down_bowl():
    # an upside-down glass bowl returns rim arcs plus an ELEVATED dome patch:
    # the hull-circle evidence must still assemble it into one object
    module = load_module()
    points, z, hole, comps = _ring_scene(module, interior="elevated")

    merged, ring_fits = module.merge_transparent_ring_components(
        comps, points_base=points, z_map=z, hole_mask=hole, table_z_m=0.0,
    )

    assert len(ring_fits) == 1
    assert len(merged) == 1


def test_transparent_ring_refuses_points_outside_the_circle():
    module = load_module()
    points, z, hole, comps = _ring_scene(module, interior="table")
    # a solid bar poking OUT of the circle breaks the one-round-object story
    bar = np.zeros_like(np.asarray(comps[0].mask, dtype=bool))
    bar[36:44, 62:78] = True
    z2 = z.copy()
    z2[bar] = 0.05
    points2 = points.copy()
    points2[:, :, 2] = z2
    comps2 = [
        module.pixel_component_from_mask(1, np.asarray(comps[0].mask, dtype=bool) | bar),
        comps[1],
    ]

    merged, ring_fits = module.merge_transparent_ring_components(
        comps2, points_base=points2, z_map=z2,
        hole_mask=~np.isfinite(z2), table_z_m=0.0,
    )

    assert len(ring_fits) == 0


def test_transparent_ring_refuses_non_cocircular_bars():
    module = load_module()
    height, width = 80, 80
    z = np.full((height, width), np.nan)
    points = np.zeros((height, width, 3))
    xs = np.linspace(-0.10, 0.10, width)[None, :]
    ys = np.linspace(0.25, 0.05, height)[:, None]
    points[:, :, 0] = xs
    points[:, :, 1] = ys
    bar_a = np.zeros((height, width), dtype=bool)
    bar_a[20:24, 20:60] = True
    bar_b = np.zeros((height, width), dtype=bool)
    bar_b[56:60, 20:60] = True
    z[bar_a] = 0.05
    z[bar_b] = 0.05
    # empty interior between the bars (transparent-looking) — residuals must veto
    rows, cols = np.indices((height, width))
    z[(rows > 26) & (rows < 54) & (cols > 24) & (cols < 56)] = 0.001
    points[:, :, 2] = z
    comps = [
        module.pixel_component_from_mask(1, bar_a),
        module.pixel_component_from_mask(2, bar_b),
    ]

    merged, ring_fits = module.merge_transparent_ring_components(
        comps, points_base=points, z_map=z,
        hole_mask=~np.isfinite(z), table_z_m=0.0,
    )

    assert len(ring_fits) == 0


def _straight_down_camera_and_transform(module, height=120, width=160):
    camera = module.CameraModel(
        width=width, height=height, fx=140.0, fy=140.0, cx=80.0, cy=60.0
    )
    transform = TransformStamped()
    transform.transform.translation.x = 0.0
    transform.transform.translation.y = 0.20
    transform.transform.translation.z = 0.50
    transform.transform.rotation.x = 1.0
    transform.transform.rotation.w = 0.0
    return camera, transform


def test_flat_table_warp_component_lacks_core_height_support():
    module = load_module()
    camera, transform = _straight_down_camera_and_transform(module)
    depth = np.full((120, 160), 0.5, dtype=np.float32)
    mask = np.zeros((120, 160), dtype=bool)
    # a broad warp blob riding 3.5mm above the table: clears min_height_m but
    # never rises clear of the sensor noise floor
    mask[20:50, 20:60] = True
    depth[20:50, 20:60] = 0.4965
    # a real 20mm-tall object
    mask[70:100, 90:130] = True
    depth[70:100, 90:130] = 0.48
    supported = module.table_foreground_supported_mask(
        mask=mask,
        depth_m=depth,
        intrinsics=camera,
        camera_to_base_transform=transform,
        table_z_m=0.0,
        min_height_m=0.003,
        max_height_m=0.2,
        min_component_area_px=900,
    )
    assert not supported[20:50, 20:60].any()
    assert supported[70:100, 90:130].all()


def test_rear_footprint_in_front_shadow_cone_is_dropped():
    module = load_module()
    def fp(x, y):
        return module.TableFootprint(
            center_xy_m=(x, y), size_x_m=0.05, size_y_m=0.05,
            width_px=10.0, height_px=10.0, yaw_rad=0.0,
        )
    camera_xy = (0.0, 0.0)
    front = fp(0.05, 0.0)
    rear = fp(0.15, 0.0)  # directly behind `front` seen from the camera
    kept = module._drop_shadowed_transparent_footprints([front, rear], camera_xy)
    assert kept == [front]
    lateral_a = fp(0.10, 0.10)
    lateral_b = fp(0.10, -0.10)  # side by side: nobody shadows anybody
    kept = module._drop_shadowed_transparent_footprints([lateral_a, lateral_b], camera_xy)
    assert kept == [lateral_a, lateral_b]


def _field_footprints(
    module,
    depth,
    camera,
    transform,
    *,
    rgb=None,
    occluders=(),
    seconds=6.0,
    steps=15,
    workspace=(-2.0, 2.0, -2.0, 2.0),
    field=None,
):
    field = field or module.TransparentEvidenceField(workspace)
    dt = seconds / steps
    for _ in range(steps):
        anchor, hole, rgb_votes, bare = module.transparent_evidence_votes(
            depth_m=depth,
            intrinsics=camera,
            camera_to_base_transform=transform,
            table_z_m=0.0,
            occluders_xy_half=list(occluders),
            workspace=workspace,
            rgb_saliency_mask=rgb,
        )
        field.update(anchor, hole, rgb_votes, dt)  # bare decay exercised via node path
    camera_xy = (
        float(transform.transform.translation.x),
        float(transform.transform.translation.y),
    )
    return field.extract(camera_xy), field


def test_unexplained_hole_yields_footprint_and_camera_ray_shadow_is_explained():
    module = load_module()
    depth = np.full((120, 160), 0.5, dtype=np.float32)
    depth[40:70, 30:70] = 0.0
    camera, transform = _straight_down_camera_and_transform(module)
    footprints, _ = _field_footprints(module, depth, camera, transform)
    assert len(footprints) == 1
    center = footprints[0].center_xy_m

    camera_xy = (0.0, 0.20)
    direction = np.asarray([camera_xy[0] - center[0], camera_xy[1] - center[1]])
    direction = direction / np.linalg.norm(direction)
    occluder = (center[0] + 0.09 * direction[0], center[1] + 0.09 * direction[1], 0.02)
    explained, _ = _field_footprints(
        module, depth, camera, transform, occluders=[occluder]
    )
    assert explained == []

    contained, _ = _field_footprints(
        module, depth, camera, transform, occluders=[(center[0], center[1], 0.12)]
    )
    assert contained == []


def test_field_hysteresis_holds_then_forgets_after_evidence_stops():
    module = load_module()
    depth = np.full((120, 160), 0.5, dtype=np.float32)
    depth[40:70, 30:70] = 0.0
    clean = np.full((120, 160), 0.5, dtype=np.float32)
    camera, transform = _straight_down_camera_and_transform(module)
    footprints, field = _field_footprints(module, depth, camera, transform)
    assert len(footprints) == 1
    # evidence disappears: the accumulated mass sustains the object briefly
    footprints, field = _field_footprints(
        module, clean, camera, transform, seconds=0.5, steps=3, field=field
    )
    assert len(footprints) == 1
    # ... but a forgotten object does not survive many time constants
    footprints, field = _field_footprints(
        module, clean, camera, transform, seconds=10.0, steps=20, field=field
    )
    assert footprints == []


def test_fragmented_hole_evidence_bridged_by_specks_is_one_footprint():
    module = load_module()
    depth = np.full((120, 160), 0.5, dtype=np.float32)
    depth[40:50, 30:42] = 0.0
    depth[44:48, 44:48] = 0.488
    depth[40:50, 52:64] = 0.0
    camera, transform = _straight_down_camera_and_transform(module)
    footprints, _ = _field_footprints(module, depth, camera, transform)
    assert len(footprints) == 1


def test_separated_evidence_clusters_stay_separate_footprints():
    module = load_module()
    depth = np.full((120, 160), 0.5, dtype=np.float32)
    depth[40:50, 20:32] = 0.0
    depth[40:50, 100:112] = 0.0
    camera, transform = _straight_down_camera_and_transform(module)
    footprints, _ = _field_footprints(module, depth, camera, transform)
    assert len(footprints) == 2


def test_small_persistent_hole_is_existence_evidence():
    module = load_module()
    depth = np.full((120, 160), 0.5, dtype=np.float32)
    depth[53:64, 70:81] = 0.0
    camera, transform = _straight_down_camera_and_transform(module)
    footprints, _ = _field_footprints(module, depth, camera, transform)
    assert len(footprints) == 1


def test_fov_truncated_speck_evidence_is_ignored():
    module = load_module()
    depth = np.full((120, 160), 0.5, dtype=np.float32)
    depth[0:6, 70:90] = 0.46
    camera, transform = _straight_down_camera_and_transform(module)
    footprints, _ = _field_footprints(module, depth, camera, transform)
    assert footprints == []


def test_hole_bordering_offplane_void_is_ignored():
    module = load_module()
    depth = np.full((120, 160), 0.8, dtype=np.float32)
    depth[:, :30] = 0.5
    depth[40:70, 30:70] = 0.0
    camera, transform = _straight_down_camera_and_transform(module)
    footprints, _ = _field_footprints(module, depth, camera, transform)
    assert footprints == []


def test_floor_dropoff_is_not_refraction_evidence():
    module = load_module()
    depth = np.full((120, 160), 0.5, dtype=np.float32)
    depth[30:90, 90:150] = 0.9
    camera, transform = _straight_down_camera_and_transform(module)
    footprints, _ = _field_footprints(module, depth, camera, transform)
    assert footprints == []


def test_rgb_saliency_marks_structure_not_smooth_illumination():
    module = load_module()
    rows, cols = 120, 160
    base = np.full((rows, cols), 150.0, dtype=np.float32)
    # smooth illumination ramp: no saliency anywhere
    ramp = base * np.linspace(0.6, 1.0, cols, dtype=np.float32)[None, :]
    smooth_rgb = np.stack([ramp] * 3, axis=2)
    saliency = module.rgb_transparency_saliency_mask(smooth_rgb)
    assert not saliency.any()
    # a striped refraction outline pops out of the same smooth frame
    striped = ramp.copy()
    striped[50:64, 60:80] += np.where(np.arange(20) % 2 == 0, 25.0, -25.0)[None, :]
    saliency = module.rgb_transparency_saliency_mask(np.stack([striped] * 3, axis=2))
    assert saliency[52:62, 62:78].any()
    assert not saliency[10:40, 10:40].any()


def test_rgb_saliency_alone_never_births_an_object():
    module = load_module()
    depth = np.full((120, 160), 0.5, dtype=np.float32)
    camera, transform = _straight_down_camera_and_transform(module)
    rgb_mask = np.zeros((120, 160), dtype=bool)
    rgb_mask[50:62, 60:75] = True
    footprints, _ = _field_footprints(module, depth, camera, transform, rgb=rgb_mask)
    assert footprints == []
    # the same saliency plus a small adjacent hole = a transparent object
    depth_with_hole = depth.copy()
    depth_with_hole[52:60, 52:60] = 0.0
    footprints, _ = _field_footprints(
        module, depth_with_hole, camera, transform, rgb=rgb_mask
    )
    assert len(footprints) == 1


def test_instance_box_touching_top_border_still_votes():
    module = load_module()
    depth = np.full((120, 160), 0.5, dtype=np.float32)
    depth[30:40, 60:75] = 0.0  # depth oddity near the base
    camera, transform = _straight_down_camera_and_transform(module)
    # a standing object whose box top leaves the frame; base band in-frame
    weights = np.zeros((120, 160), dtype=np.float32)
    weights[0:44, 55:80] = module.DETECTOR_BOX_BODY_WEIGHT
    weights[30:44, 55:80] = module.FIELD_WEIGHT_RGB
    field = module.TransparentEvidenceField((-2.0, 2.0, -2.0, 2.0))
    for _ in range(15):
        anchor, hole, rgb, bare = module.transparent_evidence_votes(
            depth_m=depth,
            intrinsics=camera,
            camera_to_base_transform=transform,
            table_z_m=0.0,
            occluders_xy_half=[],
            workspace=(-2.0, 2.0, -2.0, 2.0),
            rgb_saliency_mask=weights,
            rgb_instance_evidence=True,
        )
        field.update(anchor, hole, rgb, 0.4)
    assert rgb.shape[0] > 0  # top-border contact must not kill the box
    footprints = field.extract((0.0, 0.2))
    assert len(footprints) == 1
    # the same geometry as anonymous saliency stays rejected (FOV junk rule)
    anchor, hole, rgb, bare = module.transparent_evidence_votes(
        depth_m=depth,
        intrinsics=camera,
        camera_to_base_transform=transform,
        table_z_m=0.0,
        occluders_xy_half=[],
        workspace=(-2.0, 2.0, -2.0, 2.0),
        rgb_saliency_mask=weights > 0,
        rgb_instance_evidence=False,
    )
    assert rgb.shape[0] == 0


def test_instance_box_with_base_band_cut_by_bottom_border_is_rejected():
    module = load_module()
    depth = np.full((120, 160), 0.5, dtype=np.float32)
    camera, transform = _straight_down_camera_and_transform(module)
    weights = np.zeros((120, 160), dtype=np.float32)
    weights[80:120, 55:80] = module.DETECTOR_BOX_BODY_WEIGHT
    weights[105:120, 55:80] = module.FIELD_WEIGHT_RGB  # band runs into bottom
    anchor, hole, rgb, bare = module.transparent_evidence_votes(
        depth_m=depth,
        intrinsics=camera,
        camera_to_base_transform=transform,
        table_z_m=0.0,
        occluders_xy_half=[],
        workspace=(-2.0, 2.0, -2.0, 2.0),
        rgb_saliency_mask=weights,
        rgb_instance_evidence=True,
    )
    assert rgb.shape[0] == 0


def test_refraction_artifacts_behind_transparent_are_dropped():
    module = load_module()
    fp = module.TableFootprint(
        center_xy_m=(0.11, 0.19), size_x_m=0.05, size_y_m=0.04,
        width_px=10.0, height_px=10.0, yaw_rad=0.0,
    )
    camera_xy = (0.0, 0.39)

    def hyp(x, y, sx, sy):
        message = module.empty_live_hypothesis_array(
            frame_id="f", base_frame_id="b", use_base_frame=True
        )
        h = module.GeometryHypothesis()
        h.hypothesis_id = "component_1_bbox_r1"
        h.provenance = "m4_live_no_truth"
        h.pose_base.pose.position.x = x
        h.pose_base.pose.position.y = y
        h.dimensions_m.x = sx
        h.dimensions_m.y = sy
        message.hypotheses.append(h)
        return message

    # small speckle box straight behind the transparent footprint: dropped
    behind = module.drop_refraction_artifacts_behind_transparent(
        hyp(0.13, 0.13, 0.045, 0.02), [fp], camera_xy
    )
    assert len(behind.hypotheses) == 0
    # a big object behind survives (bounded rule)
    big = module.drop_refraction_artifacts_behind_transparent(
        hyp(0.13, 0.13, 0.09, 0.05), [fp], camera_xy
    )
    assert len(big.hypotheses) == 1
    # a small object OFF the shadow cone survives
    lateral = module.drop_refraction_artifacts_behind_transparent(
        hyp(0.25, 0.19, 0.045, 0.02), [fp], camera_xy
    )
    assert len(lateral.hypotheses) == 1


def test_field_merges_fragments_near_same_single_seed():
    module = load_module()
    field = module.TransparentEvidenceField((-2.0, 2.0, -2.0, 2.0))
    # three sub-25mm fragments spread within 6cm of one seed (far/oblique
    # hole shatter) — individually below the pinpoint floor
    votes = []
    for cx, cy in [(0.116, 0.156), (0.130, 0.198), (0.146, 0.131)]:
        for dx in (-0.004, 0.0, 0.004):
            for dy in (-0.004, 0.0, 0.004):
                votes.append((cx + dx, cy + dy, 30.0))
    for _ in range(10):
        field.update([], votes, [], 0.1)
    seed = (0.128, 0.150)
    # no seed: fragments die on the pinpoint floor individually
    assert field.extract((0.0, 0.39)) == []
    merged = field.extract((0.0, 0.39), instance_seeds_xy=[seed])
    assert len(merged) == 1
    largest = max(merged[0].size_x_m, merged[0].size_y_m)
    assert largest >= module.FIELD_FOOTPRINT_MIN_MAX_DIM_M
    # two seeds each owning fragments must NOT cross-merge into one
    field2 = module.TransparentEvidenceField((-2.0, 2.0, -2.0, 2.0))
    for _ in range(10):
        field2.update([], votes, [], 0.1)
    two = field2.extract(
        (0.0, 0.39), instance_seeds_xy=[(0.116, 0.156), (0.146, 0.131)]
    )
    assert len(two) <= 2


def test_field_yaw_hysteresis_freezes_degenerate_and_adopts_decisive():
    module = load_module()
    field = module.TransparentEvidenceField((-2.0, 2.0, -2.0, 2.0))
    # elongated blob along ~40 deg: oriented yaw is meaningful
    yaw_true = 0.7
    votes = []
    for u in np.linspace(-0.035, 0.035, 18):
        for v in np.linspace(-0.008, 0.008, 5):
            x = 0.10 + u * math.cos(yaw_true) - v * math.sin(yaw_true)
            y = 0.15 + u * math.sin(yaw_true) + v * math.cos(yaw_true)
            votes.append((x, y, 30.0))
    for _ in range(10):
        field.update([], votes, [], 0.1)
    first = field.extract((0.0, 0.39))
    assert len(first) == 1
    assert abs(first[0].yaw_rad - yaw_true) < 0.15
    # remembered yaw survives an identical frame (no spurious rotation)
    second = field.extract((0.0, 0.39))
    assert abs(second[0].yaw_rad - first[0].yaw_rad) < 1e-6
    # a WRONG remembered yaw is abandoned when the true orientation is
    # decisively tighter (rotation adoption still works)
    field.yaw_grid[field.yaw_valid] = 0.0
    third = field.extract((0.0, 0.39))
    assert abs(third[0].yaw_rad - yaw_true) < 0.15


def test_refraction_filter_protects_dense_components():
    module = load_module()
    fp = module.TableFootprint(
        center_xy_m=(0.11, 0.19), size_x_m=0.05, size_y_m=0.04,
        width_px=10.0, height_px=10.0, yaw_rad=0.0,
    )
    camera_xy = (0.0, 0.39)

    def hyp(prov):
        message = module.empty_live_hypothesis_array(
            frame_id="f", base_frame_id="b", use_base_frame=True
        )
        h = module.GeometryHypothesis()
        h.hypothesis_id = "component_1_bbox_r1"
        h.provenance = prov
        h.pose_base.pose.position.x = 0.13
        h.pose_base.pose.position.y = 0.13
        h.dimensions_m.x = 0.043
        h.dimensions_m.y = 0.030
        message.hypotheses.append(h)
        return message

    # dense component straight behind the footprint: PROTECTED (real object)
    dense = module.drop_refraction_artifacts_behind_transparent(
        hyp("live_tabletop_core;component_area_px=1520;coverage=0.937;"),
        [fp], camera_xy,
    )
    assert len(dense.hypotheses) == 1
    # sparse speckle in the same spot: dropped
    sparse = module.drop_refraction_artifacts_behind_transparent(
        hyp("live_tabletop_core;component_area_px=383;coverage=0.486;"),
        [fp], camera_xy,
    )
    assert len(sparse.hypotheses) == 0


def test_dense_occluders_exclude_instance_seed_self_returns():
    module = load_module()
    message = module.empty_live_hypothesis_array(
        frame_id="f", base_frame_id="b", use_base_frame=True
    )
    for x, y in [(0.10, 0.20), (0.00, 0.30)]:
        h = module.GeometryHypothesis()
        h.hypothesis_id = "component_1_bbox_r1"
        h.provenance = "live_tabletop_core;component_area_px=1500;coverage=0.95;"
        h.pose_base.pose.position.x = x
        h.pose_base.pose.position.y = y
        h.dimensions_m.x = 0.03
        h.dimensions_m.y = 0.02
        message.hypotheses.append(h)
    both = module.dense_occluders_xy_half(message)
    assert len(both) == 2
    # the dense hypothesis AT the instance seed is the detected transparent
    # object's own partial return: excluded from confirmed occluders
    kept = module.dense_occluders_xy_half(
        message, exclude_seeds_xy=[(0.105, 0.195)]
    )
    assert len(kept) == 1
    assert abs(kept[0][0] - 0.0) < 1e-6


def test_shadow_drop_exempts_seed_owned_instances():
    module = load_module()

    def fp(x, y, s=0.05):
        return module.TableFootprint(
            center_xy_m=(x, y), size_x_m=s, size_y_m=s,
            width_px=10.0, height_px=10.0, yaw_rad=0.0,
        )
    camera_xy = (0.0, 0.39)
    front = fp(0.045, 0.27)
    rear = fp(0.035, 0.13)   # deep in front's shadow wedge
    # no seeds: rear is eaten (historic ghost behavior preserved)
    out = module._drop_shadowed_transparent_footprints([front, rear], camera_xy)
    assert [f.center_xy_m for f in out] == [front.center_xy_m]
    # rear owns its own detector seed: both survive
    out = module._drop_shadowed_transparent_footprints(
        [front, rear], camera_xy,
        instance_seeds_xy=[(0.043, 0.286), (0.034, 0.143)],
    )
    assert len(out) == 2
    # a blob near TWO seeds is bridging junk, not an instance: still eaten
    blob = fp(0.04, 0.2)
    out = module._drop_shadowed_transparent_footprints(
        [front, blob], camera_xy,
        instance_seeds_xy=[(0.043, 0.286), (0.02, 0.21), (0.06, 0.19)],
    )
    assert [f.center_xy_m for f in out] == [front.center_xy_m]


def test_table_plane_self_check_flags_offset_and_tilt():
    module = load_module()
    import numpy as np

    class FakeTf:
        pass

    # synthesize: identity-ish camera looking straight down from 0.5m
    cam = module.CameraModel(fx=400.0, fy=400.0, cx=160.0, cy=120.0, width=320, height=240)
    from geometry_msgs.msg import TransformStamped
    tf = TransformStamped()
    tf.transform.translation.z = 0.5
    # optical frame straight down: rotate so +z(optical) maps to -z(base)
    tf.transform.rotation.x = 1.0
    tf.transform.rotation.w = 0.0
    flat = np.full((240, 320), 0.5, dtype=np.float32)
    ok = module.table_plane_self_check(flat, cam, tf, 0.0, (-1.0, 1.0, -1.0, 1.0))
    assert ok is None
    shifted = module.table_plane_self_check(flat, cam, tf, 0.008, (-1.0, 1.0, -1.0, 1.0))
    assert shifted is not None and "TABLE_Z_M" in shifted


def test_consolidate_partial_returns_into_owned_footprint():
    module = load_module()
    fp = module.TableFootprint(
        center_xy_m=(0.05, 0.25), size_x_m=0.11, size_y_m=0.11,
        width_px=10.0, height_px=10.0, yaw_rad=0.0,
    )
    seed = (0.05, 0.24)

    def msg(hyps):
        message = module.empty_live_hypothesis_array(
            frame_id="f", base_frame_id="b", use_base_frame=True
        )
        for x, y, sx, sy in hyps:
            h = module.GeometryHypothesis()
            h.hypothesis_id = "component_1_bbox_r1"
            h.provenance = "live_tabletop_core;component_area_px=600;coverage=0.99;"
            h.pose_base.pose.position.x = x
            h.pose_base.pose.position.y = y
            h.dimensions_m.x = sx
            h.dimensions_m.y = sy
            message.hypotheses.append(h)
        return message

    # small fragments inside the owned footprint: consolidated away
    out = module.consolidate_instance_partial_returns(
        msg([(0.04, 0.24, 0.03, 0.02), (0.07, 0.27, 0.025, 0.02)]), [fp], [seed]
    )
    assert len(out.hypotheses) == 0
    # a footprint-sized real (metal-cylinder class) survives
    out = module.consolidate_instance_partial_returns(
        msg([(0.05, 0.25, 0.08, 0.08)]), [fp], [seed]
    )
    assert len(out.hypotheses) == 1
    # fragment outside the footprint survives
    out = module.consolidate_instance_partial_returns(
        msg([(0.20, 0.25, 0.03, 0.02)]), [fp], [seed]
    )
    assert len(out.hypotheses) == 1
    # no seeds -> untouched (dark without detector keeps old behavior)
    out = module.consolidate_instance_partial_returns(
        msg([(0.04, 0.24, 0.03, 0.02)]), [fp], None
    )
    assert len(out.hypotheses) == 1


def test_footprint_overlap_suppression_for_big_objects():
    module = load_module()
    fp = module.TableFootprint(
        center_xy_m=(0.05, 0.22), size_x_m=0.11, size_y_m=0.10,
        width_px=10.0, height_px=10.0, yaw_rad=0.0,
    )
    # big surviving real covering the same object (offset 4cm, still >30% of
    # the smaller rect): suppressed
    assert module._footprint_overlaps_occluder(fp, [(0.05, 0.26, 0.05, 0.045)])
    # small real elsewhere: not suppressed
    assert not module._footprint_overlaps_occluder(fp, [(0.25, 0.22, 0.02, 0.02)])
