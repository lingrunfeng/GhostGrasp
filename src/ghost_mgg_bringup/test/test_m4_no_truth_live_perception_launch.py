from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_DIR = REPO_ROOT / "src" / "ghost_mgg_sim"


def test_m4_no_truth_live_launch_uses_camera_driven_live_hypotheses():
    launch_path = BRINGUP_DIR / "launch" / "m4_no_truth_live_perception.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text(encoding="utf-8")
    for required in [
        "m3_failure_scenario.launch.py",
        "GroupAction(",
        "scoped=True",
        "'enable_target_mask_emulator': 'false'",
        "'show_rviz': 'false'",
        "'point_cloud_stride': point_cloud_stride",
        "'world_file': world_file",
        "'world_name': world_name",
        "m4_live_hypothesis_publisher_node.py",
        "m4_no_truth_live_perception.rviz",
        "m4_no_truth_live_rviz",
        "/ghost_mgg/d435/depth/image_rect_raw",
        "/ghost_mgg/d435/depth/m3_corrupted",
        "/ghost_mgg/d435/target_mask",
        "/ghost_mgg/d435/external_target_mask",
        "/ghost_mgg/m4_live_hypotheses",
        "'base_frame_id': base_frame_id",
        "'table_z_m': ParameterValue(table_z_m, value_type=float)",
        "'enable_target_lock': ParameterValue(enable_target_lock, value_type=bool)",
        "'max_locked_center_distance_px': 60.0",
        "'publish_rate_hz': ParameterValue(publish_rate_hz, value_type=float)",
        "'processing_stride': ParameterValue(processing_stride, value_type=int)",
        "DeclareLaunchArgument('enable_target_lock', default_value='true')",
        "DeclareLaunchArgument('publish_rate_hz', default_value='5.0')",
        "DeclareLaunchArgument('processing_stride', default_value='2')",
        "DeclareLaunchArgument('point_cloud_stride', default_value='2')",
        "DeclareLaunchArgument('world_file', default_value='m2_tabletop_visual.sdf')",
        "DeclareLaunchArgument('world_name', default_value='ghost_mgg_m2_visual')",
        "DeclareLaunchArgument('base_frame_id', default_value='world')",
        "DeclareLaunchArgument('table_z_m', default_value='0.7400')",
        "DeclareLaunchArgument('target_color_hint', default_value='red')",
        "DeclareLaunchArgument('external_target_mask_topic', default_value='/ghost_mgg/d435/external_target_mask')",
        "DeclareLaunchArgument('require_external_target_mask', default_value='false')",
        "/ghost_mgg/d435/color/image_raw",
        "'color_topic': '/ghost_mgg/d435/color/image_raw'",
        "'external_mask_topic': external_target_mask_topic",
        "'require_external_mask': ParameterValue(require_external_target_mask, value_type=bool)",
        "'target_color_hint': target_color_hint",
        "'hypotheses_topic': live_hypothesis_topic",
        "'marker_topic': live_marker_topic",
        "'max_visible_hypotheses': ParameterValue(max_visible_hypotheses, value_type=int)",
        "DeclareLaunchArgument('top_k', default_value='5')",
        "DeclareLaunchArgument('max_visible_hypotheses', default_value='5')",
        "'live_marker_topic', default_value='/ghost_mgg/m4_joint_contract_markers'",
    ]:
        assert required in launch_text

    forbidden = [
        "gz model",
        "refresh_m4_scene_targets",
        "m4_sim_grasp_targets",
        "current_targets.json",
    ]
    for value in forbidden:
        assert value not in launch_text


def test_m7_2f_sim_algorithm_inspect_script_disables_target_lock_for_manual_gazebo_moves():
    script_path = REPO_ROOT / "scripts" / "run_m7_2f_sim_algorithm_inspect.sh"

    assert script_path.exists()
    text = script_path.read_text(encoding="utf-8")
    for required in [
        "m4_no_truth_live_perception.launch.py",
        "headless:=false",
        "show_rviz:=true",
        "enable_target_lock:=\"${enable_target_lock}\"",
        "enable_target_lock=\"${ENABLE_TARGET_LOCK:-false}\"",
        "max_visible_hypotheses=\"${MAX_VISIBLE_HYPOTHESES:-1}\"",
        "target_color_hint:=\"${target_color_hint}\"",
        "scenario_id:=\"${scenario_id}\"",
    ]:
        assert required in text


def test_m8_live_tabletop_inspect_script_uses_depth_all_target_mode():
    script_path = REPO_ROOT / "scripts" / "run_m8_live_tabletop_inspect.sh"

    assert script_path.exists()
    text = script_path.read_text(encoding="utf-8")
    for required in [
        "m4_no_truth_live_perception.launch.py",
        "headless:=false",
        "show_rviz:=true",
        "target_color_hint:=\"${target_color_hint}\"",
        "target_color_hint=\"${TARGET_COLOR_HINT:-none}\"",
        "enable_target_lock:=\"${enable_target_lock}\"",
        "enable_target_lock=\"${ENABLE_TARGET_LOCK:-false}\"",
        "max_visible_hypotheses=\"${MAX_VISIBLE_HYPOTHESES:-12}\"",
        "top_k=\"${TOP_K:-12}\"",
        "docs/m8_live_tabletop_inspect.md",
        "M8_CLEAN_STALE_PROCESSES:-1",
        "cleanup_stale_processes",
        "hypothesis_markers_node",
        "m4_live_hypothesis_publisher_node.py",
        "m4_no_truth_live_perception.launch.py",
        "xargs -r kill 2>/dev/null || true",
    ]:
        assert required in text


def test_m8_depth_failure_completion_inspect_script_uses_failure_scenario():
    script_path = REPO_ROOT / "scripts" / "run_m8_depth_failure_completion_inspect.sh"

    assert script_path.exists()
    text = script_path.read_text(encoding="utf-8")
    for required in [
        "m4_no_truth_live_perception.launch.py",
        "headless:=false",
        "show_rviz:=true",
        "scenario_id=\"${SCENARIO_ID:-S6}\"",
        "target_color_hint=\"${TARGET_COLOR_HINT:-none}\"",
        "enable_target_lock=\"${ENABLE_TARGET_LOCK:-false}\"",
        "top_k=\"${TOP_K:-12}\"",
        "max_visible_hypotheses=\"${MAX_VISIBLE_HYPOTHESES:-12}\"",
        "docs/m8_depth_failure_completion_inspect.md",
        "Observed Corrupted Depth Points",
        "M4 Live Proxy And Grasp",
    ]:
        assert required in text


def test_m8_update_rate_benchmark_entrypoint_exists_and_uses_live_hypotheses():
    script_path = REPO_ROOT / "scripts" / "run_m8_update_rate_benchmark.sh"
    benchmark_path = SIM_DIR / "scripts" / "m8_update_rate_benchmark.py"
    cmake_text = (SIM_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert script_path.exists()
    assert benchmark_path.exists()
    assert "scripts/m8_update_rate_benchmark.py" in cmake_text

    script_text = script_path.read_text(encoding="utf-8")
    for required in [
        "m4_no_truth_live_perception.launch.py",
        "scenario_id:=\"${scenario_id}\"",
        "target_color_hint:=none",
        "enable_target_lock:=false",
        "m8_update_rate_benchmark.py",
        "reports/m8_update_rate_benchmark",
        "/world/${world_name}/set_pose",
    ]:
        assert required in script_text

    benchmark_text = benchmark_path.read_text(encoding="utf-8")
    for required in [
        "/ghost_mgg/m4_live_hypotheses",
        "median_latency_sec",
        "p95_latency_sec",
        "first_changed_hypothesis",
        "--fail-on-gate",
        "center_threshold_m",
        "subprocess.run",
    ]:
        assert required in benchmark_text


def test_m8_shape_library_inspect_entrypoint_uses_separate_world():
    script_path = REPO_ROOT / "scripts" / "run_m8_shape_library_inspect.sh"

    assert script_path.exists()
    text = script_path.read_text(encoding="utf-8")
    for required in [
        "m4_no_truth_live_perception.launch.py",
        "world_file:=m8_shape_library.sdf",
        "world_name:=ghost_mgg_m8_shape_library",
        "target_color_hint:=none",
        "enable_target_lock:=false",
        "max_visible_hypotheses",
        "point_cloud_stride",
    ]:
        assert required in text


def test_m8_strict_geometry_matrix_entrypoint_uses_shape_library_metadata():
    script_path = REPO_ROOT / "scripts" / "run_m8_strict_geometry_matrix.sh"
    benchmark_path = SIM_DIR / "scripts" / "m8_strict_geometry_benchmark.py"

    assert script_path.exists()
    script_text = script_path.read_text(encoding="utf-8")
    for required in [
        "world_file:=m8_shape_library.sdf",
        "world_name:=ghost_mgg_m8_shape_library",
        "m8_small_cube",
        "m8_rect_box_2x1",
        "m8_short_cylinder",
        "m8_tri_prism",
        "reports/m8_geometry_benchmark_matrix",
        "--models",
    ]:
        assert required in script_text

    benchmark_text = benchmark_path.read_text(encoding="utf-8")
    for required in [
        '"m8_small_cube": (0.018, 0.018)',
        '"m8_rect_box_2x1": (0.050, 0.025)',
        '"m8_short_cylinder": (0.028, 0.028)',
        '"m8_tri_prism": (0.036, 0.031)',
        '"m8_hex_prism": (0.036, 0.031)',
        '"m8_tall_cylinder": "cylinder"',
    ]:
        assert required in benchmark_text


def test_m3_launch_can_disable_fixed_roi_target_mask_emulator():
    launch_text = (
        BRINGUP_DIR / "launch" / "m3_failure_camera_inspect.launch.py"
    ).read_text(encoding="utf-8")

    assert "enable_target_mask_emulator" in launch_text
    assert "condition=IfCondition(enable_target_mask_emulator)" in launch_text
    assert "DeclareLaunchArgument('enable_target_mask_emulator', default_value='true')" in launch_text


def test_m4_no_truth_rviz_only_shows_live_perception_layers():
    rviz_path = SIM_DIR / "rviz" / "m4_no_truth_live_perception.rviz"

    assert rviz_path.exists()
    rviz_text = rviz_path.read_text(encoding="utf-8")
    for required in [
        "Raw Ideal Depth Points (debug only)",
        "/ghost_mgg/d435/depth/points",
        "Enabled: false\n      Invert Rainbow: false",
        "Observed Corrupted Depth Points (algorithm input)",
        "Enabled: true\n      Invert Rainbow: false",
        "Enabled: false",
        "/ghost_mgg/m4_joint_contract_markers",
        "M4 Live Proxy And Grasp",
    ]:
        assert required in rviz_text

    for stale_topic in [
        "/ghost_mgg/m4_graspability_markers",
        "/ghost_mgg/m4_sim_grasp_markers",
        "/ghost_mgg/m4_joint_hypothesis_markers",
        "/ghost_mgg/m4_real_ranking_markers",
    ]:
        assert stale_topic not in rviz_text


def test_live_hypothesis_node_is_installed_and_test_registered():
    sim_cmake = (SIM_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    bringup_cmake = (BRINGUP_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "scripts/m4_live_hypothesis_publisher_node.py" in sim_cmake
    assert (
        "ament_add_pytest_test(test_m4_live_hypothesis_publisher "
        "test/test_m4_live_hypothesis_publisher.py)"
    ) in sim_cmake
    assert (
        "ament_add_pytest_test(test_m4_no_truth_live_perception_launch "
        "test/test_m4_no_truth_live_perception_launch.py)"
    ) in bringup_cmake


def test_m4_no_truth_live_dynamic_smoke_contract():
    smoke_path = REPO_ROOT / "scripts" / "smoke_m4_no_truth_live_dynamic.sh"

    assert smoke_path.exists()
    text = smoke_path.read_text(encoding="utf-8")
    for required in [
        "m4_no_truth_live_perception.launch.py",
        "/ghost_mgg/m4_live_hypotheses",
        "gz service",
        "move_non_target_clutter_out_of_view",
        "non_target_models=(",
        "math.sin(float(\"${yaw}\") * 0.5)",
        "math.cos(float(\"${yaw}\") * 0.5)",
        "initial_hypothesis_center=",
        "moved_hypothesis_center=",
        "wait_for_moved_hypothesis",
        "delta_norm=",
        "delta_norm_ok=true",
        "no_truth_audit=true",
        "m4_live_no_truth_ghost_mgg_v1",
    ]:
        assert required in text
    forbidden = [
        "refresh_m4_scene_targets.py",
        "current_targets.json",
        "m4_sim_grasp_targets",
    ]
    for value in forbidden:
        assert value not in text
