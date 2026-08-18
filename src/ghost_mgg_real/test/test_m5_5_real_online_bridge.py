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


def test_required_topic_contract_contains_rgb_depth_ir_points_and_tf():
    module = _load_script("check_m5_5_real_online_topics")

    topics = module.required_d435_topics()

    assert topics["/camera/camera/color/image_raw"] == "sensor_msgs/msg/Image"
    assert topics["/camera/camera/aligned_depth_to_color/image_raw"] == "sensor_msgs/msg/Image"
    assert topics["/camera/camera/infra1/image_rect_raw"] == "sensor_msgs/msg/Image"
    assert topics["/camera/camera/infra2/image_rect_raw"] == "sensor_msgs/msg/Image"
    assert topics["/camera/camera/depth/color/points"] == "sensor_msgs/msg/PointCloud2"
    assert topics["/tf_static"] == "tf2_msgs/msg/TFMessage"


def test_topic_check_report_schema_marks_missing_topics():
    module = _load_script("check_m5_5_real_online_topics")

    report = module.build_topic_check_report(
        observed_topics={"/camera/camera/color/image_raw": "sensor_msgs/msg/Image"},
        now_sec=10.0,
    )

    assert report["schema_version"] == "m5_5_real_online_topic_check_v1"
    assert report["overall_status"] == "fail"
    assert "/camera/camera/depth/image_rect_raw" in report["missing_topics"]
    assert "/camera/camera/color/image_raw" in report["present_topics"]


def test_topic_check_report_fails_type_mismatch():
    module = _load_script("check_m5_5_real_online_topics")
    observed = module.required_d435_topics()
    observed["/camera/camera/depth/color/points"] = "sensor_msgs/msg/Image"

    report = module.build_topic_check_report(observed_topics=observed, now_sec=12.5)

    assert report["overall_status"] == "fail"
    assert report["type_mismatches"] == {
        "/camera/camera/depth/color/points": {
            "expected": "sensor_msgs/msg/PointCloud2",
            "observed": "sensor_msgs/msg/Image",
        }
    }


def test_observation_quality_from_replay_metadata_reports_stream_reliability(tmp_path):
    module = _load_script("generate_m5_5_observation_quality")
    metadata = {
        "scene_id": "live_jelly_001",
        "frames": {
            "/camera/camera/color/image_raw": {
                "encoding": "rgb8",
                "nonzero_ratio": 1.0,
                "mean_intensity": 108.0,
            },
            "/camera/camera/depth/image_rect_raw": {
                "encoding": "16UC1",
                "valid_ratio": 0.81,
            },
            "/camera/camera/aligned_depth_to_color/image_raw": {
                "encoding": "16UC1",
                "valid_ratio": 0.95,
            },
            "/camera/camera/infra1/image_rect_raw": {
                "encoding": "mono8",
                "nonzero_ratio": 1.0,
                "mean_intensity": 90.0,
            },
            "/camera/camera/infra2/image_rect_raw": {
                "encoding": "mono8",
                "nonzero_ratio": 1.0,
                "mean_intensity": 91.0,
            },
        },
    }

    quality = module.build_observation_quality(
        observation_id="live_jelly_001",
        sample_metadata=metadata,
        topic_report={"overall_status": "pass"},
        target_summary=None,
        planning_requested=False,
    )

    assert quality["schema_version"] == "m5_5_observation_quality_v1"
    assert quality["rgb_ok"] is True
    assert quality["ir_ok"] is True
    assert quality["depth_ok"] is True
    assert quality["aligned_depth_ok"] is True
    assert quality["mask_ok"] is False
    assert quality["recommended_backend"] == "abstain"
    assert "target_mask_missing" in quality["reject_reasons"]


def test_observation_quality_uses_target_failure_metrics_for_ghost_mgg():
    module = _load_script("generate_m5_5_observation_quality")

    quality = module.build_observation_quality(
        observation_id="transparent_target",
        sample_metadata={
            "frames": {
                "/camera/camera/color/image_raw": {
                    "encoding": "rgb8",
                    "nonzero_ratio": 1.0,
                    "mean_intensity": 100.0,
                },
                "/camera/camera/depth/image_rect_raw": {
                    "encoding": "16UC1",
                    "valid_ratio": 0.70,
                },
                "/camera/camera/aligned_depth_to_color/image_raw": {
                    "encoding": "16UC1",
                    "valid_ratio": 0.72,
                },
                "/camera/camera/infra1/image_rect_raw": {
                    "encoding": "mono8",
                    "nonzero_ratio": 1.0,
                    "mean_intensity": 80.0,
                },
                "/camera/camera/infra2/image_rect_raw": {
                    "encoding": "mono8",
                    "nonzero_ratio": 1.0,
                    "mean_intensity": 81.0,
                },
            }
        },
        topic_report={"overall_status": "pass"},
        target_summary={
            "target_pixels": 100,
            "valid_depth_ratio": 0.45,
            "hole_ratio": 0.42,
            "table_leakage_ratio": 0.12,
        },
        planning_requested=False,
    )

    assert quality["mask_ok"] is True
    assert quality["depth_failure_detected"] is True
    assert quality["recommended_backend"] == "ghost_mgg"
    assert quality["target_hole_ratio"] == 0.42


def test_backend_selector_prefers_normal_ir_ghost_and_abstain():
    module = _load_script("run_m5_5_backend_selector")

    normal = module.select_backend(
        {
            "stale": False,
            "rgb_ok": True,
            "ir_ok": True,
            "depth_ok": True,
            "mask_ok": True,
            "table_ok": True,
            "tf_ok": True,
            "target_valid_depth_ratio": 0.8,
            "target_hole_ratio": 0.05,
            "target_table_leakage_ratio": 0.0,
        }
    )
    assert normal["recommended_backend"] == "normal_rgbd"

    ir_depth = module.select_backend(
        {
            "stale": False,
            "rgb_ok": False,
            "ir_ok": True,
            "depth_ok": True,
            "mask_ok": True,
            "table_ok": True,
            "tf_ok": True,
            "target_valid_depth_ratio": 0.72,
            "target_hole_ratio": 0.05,
            "target_table_leakage_ratio": 0.0,
        }
    )
    assert ir_depth["recommended_backend"] == "ir_depth"

    ghost = module.select_backend(
        {
            "stale": False,
            "rgb_ok": True,
            "ir_ok": True,
            "depth_ok": True,
            "mask_ok": True,
            "table_ok": True,
            "tf_ok": True,
            "target_valid_depth_ratio": 0.45,
            "target_hole_ratio": 0.4,
            "target_table_leakage_ratio": 0.1,
        }
    )
    assert ghost["recommended_backend"] == "ghost_mgg"

    abstain = module.select_backend(
        {
            "stale": True,
            "rgb_ok": True,
            "ir_ok": True,
            "depth_ok": True,
            "mask_ok": True,
            "table_ok": True,
            "tf_ok": True,
            "target_valid_depth_ratio": 0.8,
            "target_hole_ratio": 0.0,
            "target_table_leakage_ratio": 0.0,
        }
    )
    assert abstain["recommended_backend"] == "abstain"
    assert "stale_observation" in abstain["reject_reasons"]


def test_backend_selector_routes_high_table_leakage_to_ghost_even_when_valid_depth_is_high():
    module = _load_script("run_m5_5_backend_selector")

    selection = module.select_backend(
        {
            "stale": False,
            "rgb_ok": True,
            "ir_ok": True,
            "depth_ok": True,
            "mask_ok": True,
            "table_ok": True,
            "tf_ok": True,
            "target_valid_depth_ratio": 0.95,
            "target_hole_ratio": 0.02,
            "target_table_leakage_ratio": 0.30,
        }
    )

    assert selection["recommended_backend"] == "ghost_mgg"
    assert "target_table_leakage_ratio_high" in selection["depth_failure_reasons"]


def test_backend_selector_treats_contact_shadow_leakage_as_normal_when_depth_is_usable():
    module = _load_script("run_m5_5_backend_selector")

    selection = module.select_backend(
        {
            "stale": False,
            "rgb_ok": True,
            "ir_ok": True,
            "depth_ok": True,
            "mask_ok": True,
            "table_ok": True,
            "tf_ok": True,
            "target_valid_depth_ratio": 0.768,
            "target_hole_ratio": 0.232,
            "target_table_leakage_ratio": 0.172,
        },
        planning_requested=True,
    )

    assert selection["recommended_backend"] == "normal_rgbd"
    assert selection["depth_failure_reasons"] == []
    assert "contact_shadow_leakage_but_depth_usable" in selection["caution_reasons"]


def test_backend_selector_accepts_borderline_mask_edge_holes_when_depth_is_usable():
    module = _load_script("run_m5_5_backend_selector")

    selection = module.select_backend(
        {
            "stale": False,
            "rgb_ok": True,
            "ir_ok": True,
            "depth_ok": True,
            "mask_ok": True,
            "table_ok": True,
            "tf_ok": True,
            "target_valid_depth_ratio": 0.77,
            "target_hole_ratio": 0.229,
            "target_table_leakage_ratio": 0.026,
        },
        planning_requested=True,
    )

    assert selection["recommended_backend"] == "normal_rgbd"
    assert "borderline_hole_ratio_but_depth_usable" in selection["caution_reasons"]


def test_backend_selector_cli_writes_decision_json(tmp_path):
    module = _load_script("run_m5_5_backend_selector")
    quality_path = tmp_path / "quality.json"
    output_path = tmp_path / "selection.json"
    quality_path.write_text(
        json.dumps(
            {
                "stale": False,
                "rgb_ok": True,
                "ir_ok": True,
                "depth_ok": True,
                "mask_ok": True,
                "table_ok": True,
                "tf_ok": True,
                "target_valid_depth_ratio": 0.4,
                "target_hole_ratio": 0.5,
                "target_table_leakage_ratio": 0.02,
            }
        )
    )

    result = module.run_selector(quality_path=quality_path, output_path=output_path)

    saved = json.loads(output_path.read_text())
    assert result["recommended_backend"] == "ghost_mgg"
    assert saved["recommended_backend"] == "ghost_mgg"


def test_backend_selection_report_pairs_replay_metadata_with_evidence(tmp_path):
    module = _load_script("generate_m5_5_backend_selection_report")
    sample_root = tmp_path / "samples"
    evidence_root = tmp_path / "evidence"
    output_dir = tmp_path / "report"

    _write_scene_metadata(
        sample_root / "opaque_box" / "metadata.json",
        scene_id="opaque_box",
        depth_valid_ratio=0.9,
        rgb_mean=110.0,
    )
    _write_evidence_summary(
        evidence_root / "opaque_box" / "evidence_summary.json",
        scene_id="opaque_box",
        target_label="opaque_box",
        valid_depth_ratio=0.82,
        hole_ratio=0.08,
        table_leakage_ratio=0.01,
    )
    _write_scene_metadata(
        sample_root / "jelly" / "metadata.json",
        scene_id="jelly",
        depth_valid_ratio=0.7,
        rgb_mean=100.0,
    )
    _write_evidence_summary(
        evidence_root / "jelly" / "evidence_summary.json",
        scene_id="jelly",
        target_label="transparent_jelly_cup",
        valid_depth_ratio=0.35,
        hole_ratio=0.5,
        table_leakage_ratio=0.12,
    )

    report = module.generate_backend_selection_report(
        sample_root=sample_root,
        evidence_root=evidence_root,
        output_dir=output_dir,
    )

    backends = {row["scene_id"]: row["recommended_backend"] for row in report["scenes"]}
    assert backends == {"jelly": "ghost_mgg", "opaque_box": "normal_rgbd"}
    assert (output_dir / "backend_selection_report.json").exists()
    assert (output_dir / "backend_selection_report.csv").exists()
    assert "opaque_box" in (output_dir / "index.md").read_text()


def test_capture_offline_replay_sample_snapshot_writes_manifest(tmp_path):
    module = _load_script("capture_m5_5_real_online_snapshot")
    sample_dir = tmp_path / "sample_scene"
    output_dir = tmp_path / "snapshot"
    sample_dir.mkdir()
    (sample_dir / "metadata.json").write_text(json.dumps({"scene_id": "sample_scene"}))
    (sample_dir / "color.png").write_bytes(b"color")
    (sample_dir / "depth_viz.png").write_bytes(b"depth")

    manifest = module.capture_offline_sample_snapshot(
        offline_sample_dir=sample_dir,
        output_dir=output_dir,
        observation_id="live_000001",
    )

    saved = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "m5_5_real_online_snapshot_v1"
    assert saved["observation_id"] == "live_000001"
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "color.png").exists()
    assert "offline_replay_sample" in saved["source"]


def test_live_snapshot_frame_collector_writes_scene_outputs(tmp_path):
    module = _load_script("capture_m5_5_real_online_snapshot")
    extractor = _load_script("extract_m5_replay_samples")
    frames = {
        "/camera/camera/color/image_raw": _msg(
            "rgb8", 1, 1, bytes([10, 20, 30])
        ),
        "/camera/camera/depth/image_rect_raw": _msg(
            "16UC1", 1, 1, b"\xe8\x03"
        ),
        "/camera/camera/aligned_depth_to_color/image_raw": _msg(
            "16UC1", 1, 1, b"\xe8\x03"
        ),
        "/camera/camera/infra1/image_rect_raw": _msg("mono8", 1, 1, bytes([42])),
        "/camera/camera/infra2/image_rect_raw": _msg("mono8", 1, 1, bytes([43])),
    }

    manifest = module.write_live_snapshot_outputs(
        observation_id="live_000002",
        frames=frames,
        output_dir=tmp_path,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert manifest["source"] == "live_ros_topics"
    assert manifest["observation_id"] == "live_000002"
    assert metadata["scene_id"] == "live_000002"
    assert (tmp_path / "color.png").exists()
    assert (tmp_path / "depth_raw.npy").exists()
    assert int(np.load(tmp_path / "depth_raw.npy")[0, 0]) == 1000
    assert (tmp_path / "aligned_depth_raw.npy").exists()
    assert extractor.decode_image_msg(frames["/camera/camera/color/image_raw"]).shape == (1, 1, 3)


def test_live_snapshot_writes_aligned_depth_camera_info(tmp_path):
    module = _load_script("capture_m5_5_real_online_snapshot")
    frames = {
        "/camera/camera/color/image_raw": _msg("rgb8", 1, 1, bytes([10, 20, 30])),
        "/camera/camera/depth/image_rect_raw": _msg("16UC1", 1, 1, b"\xe8\x03"),
        "/camera/camera/aligned_depth_to_color/image_raw": _msg("16UC1", 1, 1, b"\xe8\x03"),
        "/camera/camera/infra1/image_rect_raw": _msg("mono8", 1, 1, bytes([42])),
        "/camera/camera/infra2/image_rect_raw": _msg("mono8", 1, 1, bytes([43])),
        "/camera/camera/aligned_depth_to_color/camera_info": _camera_info_msg(
            width=1,
            height=1,
            frame_id="camera_color_optical_frame",
            k=[500.0, 0.0, 0.5, 0.0, 501.0, 0.6, 0.0, 0.0, 1.0],
        ),
    }

    manifest = module.write_live_snapshot_outputs(
        observation_id="live_with_info",
        frames=frames,
        output_dir=tmp_path,
    )

    camera_info = json.loads((tmp_path / "aligned_depth_camera_info.json").read_text())
    assert "aligned_depth_camera_info.json" in manifest["copied_files"]
    assert camera_info["frame_id"] == "camera_color_optical_frame"
    assert camera_info["width"] == 1
    assert camera_info["height"] == 1
    assert camera_info["k"] == [500.0, 0.0, 0.5, 0.0, 501.0, 0.6, 0.0, 0.0, 1.0]


def test_live_masked_evidence_estimates_table_leakage_from_outside_mask(tmp_path):
    module = _load_script("generate_m5_5_live_masked_evidence")
    snapshot_dir = tmp_path / "snapshot"
    mask_path = tmp_path / "target_mask.png"
    output_dir = tmp_path / "evidence"
    snapshot_dir.mkdir()

    depth = np.full((5, 5), 1000, dtype=np.uint16)
    depth[2, 2] = 0
    depth[1, 1] = 820
    np.save(snapshot_dir / "aligned_depth_raw.npy", depth)
    cv2.imwrite(str(snapshot_dir / "color.png"), np.zeros((5, 5, 3), dtype=np.uint8))
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 255
    cv2.imwrite(str(mask_path), mask)

    summary = module.generate_live_masked_evidence(
        scene_id="live_target",
        snapshot_dir=snapshot_dir,
        mask_path=mask_path,
        output_dir=output_dir,
    )

    assert summary["scene_id"] == "live_target"
    assert summary["target_pixels"] == 9
    assert summary["hole_ratio"] == 1 / 9
    assert summary["table_leakage_ratio"] == 7 / 9
    assert summary["foreground_ratio"] == 1 / 9
    assert (output_dir / "live_target" / "evidence_summary.json").exists()


def test_live_smoke_report_summarizes_ghost_and_normal_backends(tmp_path):
    module = _load_script("generate_m5_5_live_smoke_report")
    bridge_root = tmp_path / "bridge"
    output_dir = tmp_path / "report"
    _write_live_scene_outputs(
        bridge_root,
        scene_id="live_jelly",
        target_label="transparent_jelly_cup",
        backend="ghost_mgg",
        valid_depth_ratio=0.45,
        hole_ratio=0.4,
        leakage_ratio=0.2,
    )
    _write_live_scene_outputs(
        bridge_root,
        scene_id="live_cylinder",
        target_label="green_cylinder",
        backend="normal_rgbd",
        valid_depth_ratio=0.82,
        hole_ratio=0.1,
        leakage_ratio=0.02,
    )

    report = module.generate_live_smoke_report(
        bridge_root=bridge_root,
        scene_ids=["live_jelly", "live_cylinder"],
        output_dir=output_dir,
    )

    assert report["schema_version"] == "m5_5_live_smoke_report_v1"
    assert report["backend_counts"] == {"ghost_mgg": 1, "normal_rgbd": 1}
    assert report["scenes"][0]["scene_id"] == "live_jelly"
    assert (output_dir / "live_smoke_report.json").exists()
    assert (output_dir / "live_smoke_report.csv").exists()
    assert "transparent_jelly_cup" in (output_dir / "index.md").read_text()


def test_m6_shadow_readiness_reports_passed_and_blocked_gates(tmp_path):
    module = _load_script("generate_m6_shadow_readiness")
    live_report_path = tmp_path / "live_smoke_report.json"
    topic_check_path = tmp_path / "topic_check.json"
    output_dir = tmp_path / "m6"
    live_report_path.write_text(
        json.dumps(
            {
                "schema_version": "m5_5_live_smoke_report_v1",
                "backend_counts": {"ghost_mgg": 1, "normal_rgbd": 1},
                "scenes": [
                    {
                        "scene_id": "live_jelly",
                        "recommended_backend": "ghost_mgg",
                    },
                    {
                        "scene_id": "live_cylinder",
                        "recommended_backend": "normal_rgbd",
                    },
                ],
            }
        )
    )
    topic_check_path.write_text(
        json.dumps(
            {
                "schema_version": "m5_5_real_online_topic_check_v1",
                "overall_status": "pass",
                "missing_topics": [],
            }
        )
    )

    report = module.generate_m6_shadow_readiness(
        live_smoke_report_path=live_report_path,
        topic_check_path=topic_check_path,
        output_dir=output_dir,
        real_tf_checked=False,
        moveit_shadow_checked=False,
    )

    gate_status = {gate["gate_id"]: gate["status"] for gate in report["gates"]}
    assert gate_status["m5_5_live_backend_switch"] == "pass"
    assert gate_status["d435_topic_contract"] == "pass"
    assert gate_status["real_camera_to_base_tf"] == "blocked"
    assert gate_status["moveit_shadow_planning"] == "blocked"
    assert report["overall_status"] == "blocked"
    assert (output_dir / "m6_shadow_readiness.json").exists()
    assert "M6 Shadow Readiness" in (output_dir / "index.md").read_text()


def test_m6_shadow_readiness_records_passed_moveit_shadow_evidence(tmp_path):
    module = _load_script("generate_m6_shadow_readiness")
    live_report_path = tmp_path / "live_smoke_report.json"
    topic_check_path = tmp_path / "topic_check.json"
    output_dir = tmp_path / "m6"
    live_report_path.write_text(
        json.dumps(
            {
                "schema_version": "m5_5_live_smoke_report_v1",
                "backend_counts": {"ghost_mgg": 1, "normal_rgbd": 1},
            }
        )
    )
    topic_check_path.write_text(
        json.dumps(
            {
                "schema_version": "m5_5_real_online_topic_check_v1",
                "overall_status": "pass",
                "missing_topics": [],
            }
        )
    )

    report = module.generate_m6_shadow_readiness(
        live_smoke_report_path=live_report_path,
        topic_check_path=topic_check_path,
        output_dir=output_dir,
        real_tf_checked=False,
        moveit_shadow_checked=True,
        moveit_shadow_evidence_path="reports/m6_shadow_grasp_targets/test/moveit_plan_only_shadow_allowlist.json",
    )

    gates = {gate["gate_id"]: gate for gate in report["gates"]}
    moveit_gate = gates["moveit_shadow_planning"]
    assert moveit_gate["status"] == "pass"
    assert (
        moveit_gate["evidence"]
        == "reports/m6_shadow_grasp_targets/test/moveit_plan_only_shadow_allowlist.json"
    )
    assert "Plan-only MoveIt request succeeded" in moveit_gate["detail"]
    assert "Run MoveIt shadow planning" not in report["next_required_live_steps"]


def test_m6_shadow_readiness_records_passed_real_tf_evidence(tmp_path):
    module = _load_script("generate_m6_shadow_readiness")
    live_report_path = tmp_path / "live_smoke_report.json"
    topic_check_path = tmp_path / "topic_check.json"
    output_dir = tmp_path / "m6"
    live_report_path.write_text(
        json.dumps(
            {
                "schema_version": "m5_5_live_smoke_report_v1",
                "backend_counts": {"ghost_mgg": 1, "normal_rgbd": 1},
            }
        )
    )
    topic_check_path.write_text(
        json.dumps(
            {
                "schema_version": "m5_5_real_online_topic_check_v1",
                "overall_status": "pass",
                "missing_topics": [],
            }
        )
    )

    report = module.generate_m6_shadow_readiness(
        live_smoke_report_path=live_report_path,
        topic_check_path=topic_check_path,
        output_dir=output_dir,
        real_tf_checked=True,
        real_tf_evidence_path="reports/m6_shadow_observations/live_001/m6_shadow_observation.json",
    )

    gates = {gate["gate_id"]: gate for gate in report["gates"]}
    tf_gate = gates["real_camera_to_base_tf"]
    assert tf_gate["status"] == "pass"
    assert (
        tf_gate["evidence"]
        == "reports/m6_shadow_observations/live_001/m6_shadow_observation.json"
    )
    assert "Verify camera-to-base TF" not in report["next_required_live_steps"]


def test_m6_shadow_readiness_records_real_mycobot_state_bridge_evidence(tmp_path):
    module = _load_script("generate_m6_shadow_readiness")
    live_report_path = tmp_path / "live_smoke_report.json"
    topic_check_path = tmp_path / "topic_check.json"
    output_dir = tmp_path / "m6"
    live_report_path.write_text(
        json.dumps(
            {
                "schema_version": "m5_5_live_smoke_report_v1",
                "backend_counts": {"ghost_mgg": 1, "normal_rgbd": 1},
            }
        )
    )
    topic_check_path.write_text(
        json.dumps(
            {
                "schema_version": "m5_5_real_online_topic_check_v1",
                "overall_status": "pass",
                "missing_topics": [],
            }
        )
    )

    report = module.generate_m6_shadow_readiness(
        live_smoke_report_path=live_report_path,
        topic_check_path=topic_check_path,
        output_dir=output_dir,
        real_tf_checked=False,
        moveit_shadow_checked=True,
        mycobot_state_bridge_checked=True,
        real_state_moveit_shadow_checked=True,
    )

    gates = {gate["gate_id"]: gate for gate in report["gates"]}
    assert gates["mycobot_state_bridge"]["status"] == "pass"
    assert (
        gates["mycobot_state_bridge"]["evidence"]
        == "reports/m6_mycobot_state_bridge_smoke_shadow_gripper/joint_state_once.yaml"
    )
    assert gates["real_state_moveit_shadow"]["status"] == "pass"
    assert "real /joint_states" in gates["real_state_moveit_shadow"]["detail"]
    assert "Run myCobot state bridge smoke" not in report["next_required_live_steps"]


def test_m6_shadow_readiness_has_no_next_steps_when_all_gates_pass(tmp_path):
    module = _load_script("generate_m6_shadow_readiness")
    live_report_path = tmp_path / "live_smoke_report.json"
    topic_check_path = tmp_path / "topic_check.json"
    output_dir = tmp_path / "m6"
    live_report_path.write_text(
        json.dumps(
            {
                "schema_version": "m5_5_live_smoke_report_v1",
                "backend_counts": {"ghost_mgg": 1, "normal_rgbd": 1},
            }
        )
    )
    topic_check_path.write_text(
        json.dumps(
            {
                "schema_version": "m5_5_real_online_topic_check_v1",
                "overall_status": "pass",
                "missing_topics": [],
            }
        )
    )

    report = module.generate_m6_shadow_readiness(
        live_smoke_report_path=live_report_path,
        topic_check_path=topic_check_path,
        output_dir=output_dir,
        real_tf_checked=True,
        moveit_shadow_checked=True,
        mycobot_state_bridge_checked=True,
        real_state_moveit_shadow_checked=True,
    )

    assert report["overall_status"] == "pass"
    assert report["next_required_live_steps"] == []
    assert "None; ready for next M6 gate." in (output_dir / "index.md").read_text()


def _msg(encoding: str, height: int, width: int, data: bytes):
    class Message:
        pass

    msg = Message()
    msg.encoding = encoding
    msg.height = height
    msg.width = width
    msg.data = data
    return msg


def _camera_info_msg(*, width: int, height: int, frame_id: str, k: list[float]):
    class Header:
        pass

    class Message:
        pass

    msg = Message()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.k = k
    msg.d = []
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [k[0], 0.0, k[2], 0.0, 0.0, k[4], k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.distortion_model = "plumb_bob"
    return msg


def _write_scene_metadata(
    path: Path,
    *,
    scene_id: str,
    depth_valid_ratio: float,
    rgb_mean: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "frames": {
                    "/camera/camera/color/image_raw": {
                        "encoding": "rgb8",
                        "nonzero_ratio": 1.0,
                        "mean_intensity": rgb_mean,
                    },
                    "/camera/camera/depth/image_rect_raw": {
                        "encoding": "16UC1",
                        "valid_ratio": depth_valid_ratio,
                    },
                    "/camera/camera/aligned_depth_to_color/image_raw": {
                        "encoding": "16UC1",
                        "valid_ratio": depth_valid_ratio,
                    },
                    "/camera/camera/infra1/image_rect_raw": {
                        "encoding": "mono8",
                        "nonzero_ratio": 1.0,
                        "mean_intensity": 90.0,
                    },
                    "/camera/camera/infra2/image_rect_raw": {
                        "encoding": "mono8",
                        "nonzero_ratio": 1.0,
                        "mean_intensity": 91.0,
                    },
                },
            }
        )
    )


def _write_evidence_summary(
    path: Path,
    *,
    scene_id: str,
    target_label: str,
    valid_depth_ratio: float,
    hole_ratio: float,
    table_leakage_ratio: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "m5_masked_evidence_scene_v1",
                "scene_id": scene_id,
                "target_label": target_label,
                "shape_hint": "box",
                "target_pixels": 100,
                "valid_depth_ratio": valid_depth_ratio,
                "hole_ratio": hole_ratio,
                "table_leakage_ratio": table_leakage_ratio,
            }
        )
    )


def _write_live_scene_outputs(
    bridge_root: Path,
    *,
    scene_id: str,
    target_label: str,
    backend: str,
    valid_depth_ratio: float,
    hole_ratio: float,
    leakage_ratio: float,
) -> None:
    evidence_dir = bridge_root / "live_masked_evidence" / scene_id
    snapshot_dir = bridge_root / "snapshots" / scene_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "evidence_summary.json").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "target_label": target_label,
                "shape_hint": "unknown",
                "target_pixels": 100,
                "valid_depth_ratio": valid_depth_ratio,
                "hole_ratio": hole_ratio,
                "table_leakage_ratio": leakage_ratio,
                "foreground_ratio": 0.1,
            }
        )
    )
    (snapshot_dir / "backend_selection.json").write_text(
        json.dumps(
            {
                "schema_version": "m5_5_backend_selection_v1",
                "recommended_backend": backend,
                "reject_reasons": [],
                "depth_failure_reasons": (
                    ["target_hole_ratio_high"] if backend == "ghost_mgg" else []
                ),
            }
        )
    )
    (snapshot_dir / "observation_quality.json").write_text(
        json.dumps({"recommended_backend": backend, "rgb_ok": True, "depth_ok": True})
    )
