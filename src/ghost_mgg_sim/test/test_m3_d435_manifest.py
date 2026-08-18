from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_PACKAGE = REPO_ROOT / "src" / "ghost_mgg_sim"


def test_m3_d435_manifest_declares_simulation_contract_and_real_replay_guard():
    manifest_path = SIM_PACKAGE / "config" / "d435_sim_calibration_manifest.yaml"
    assert manifest_path.exists()

    text = manifest_path.read_text(encoding="utf-8")

    for required in [
        "profile_id: m3_sim_placeholder_640x480_10hz",
        "source: gazebo_sim_placeholder",
        "requires_real_device_export_before_m5: true",
        "IR streams are monocular proxy images for topic/TF contract validation",
        "This manifest is a versioned simulation contract, not a real D435 factory calibration.",
        "d435_color_optical_frame",
        "d435_depth_optical_frame",
        "d435_infra1_optical_frame",
        "d435_infra2_optical_frame",
        "/ghost_mgg/d435/color/image_raw",
        "/ghost_mgg/d435/depth/image_rect_raw",
        "/ghost_mgg/d435/depth/m3_corrupted",
        "/ghost_mgg/d435/target_mask",
        "/ghost_mgg/d435/evidence/summary",
        "/ghost_mgg/d435/infra1/image_rect_raw",
        "/ghost_mgg/d435/infra2/image_rect_raw",
        "encoding: 32FC1",
        "role: left_ir_proxy",
        "role: right_ir_proxy",
        "baseline_from_depth_frame_m: -0.026",
        "baseline_from_depth_frame_m: 0.026",
        "failure_mode: mixed",
        "evidence_source: target_mask",
    ]:
        assert required in text


def test_ghost_mgg_sim_installs_d435_manifest_directory():
    cmake_text = (SIM_PACKAGE / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "DIRECTORY config launch models worlds" in cmake_text
    assert "evidence_summary_logger_node" in cmake_text
    assert "src/evidence_summary_logger_node.cpp" in cmake_text


def test_m3_failure_scenario_config_declares_s0_to_s7_presets():
    config_path = SIM_PACKAGE / "config" / "m3_failure_scenarios.yaml"
    assert config_path.exists()

    text = config_path.read_text(encoding="utf-8")

    for required in [
        "S0:",
        "label: normal_day_rgbd_baseline",
        "S1:",
        "label: low_light_rgb_degraded_ir_depth_available_placeholder",
        "S2:",
        "label: dark_ir_proxy_transparent_failure",
        "S3:",
        "failure_mode: hole",
        "S4:",
        "failure_mode: table_leakage",
        "S5:",
        "failure_mode: edge_flying",
        "S6:",
        "failure_mode: mixed",
        "S7:",
        "failure_mode: reflective",
        "flying_point_stride: 4",
        "biased_depth_offset_m: -0.05",
    ]:
        assert required in text
