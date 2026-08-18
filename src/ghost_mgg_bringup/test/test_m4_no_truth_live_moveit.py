from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m4_no_truth_live_moveit_launch_wires_live_hypotheses_to_moveit():
    launch_path = BRINGUP_DIR / "launch" / "m4_no_truth_live_moveit_execute.launch.py"
    assert launch_path.exists()
    source = launch_path.read_text(encoding="utf-8")

    for required in [
        "m4_no_truth_live_perception.launch.py",
        "GroupAction",
        "scoped=True",
        "'live_hypothesis_topic': live_hypothesis_topic",
        "'live_marker_topic': '/ghost_mgg/m4_live_debug_markers'",
        "'world_file': world_file",
        "'world_name': world_name",
        "'enable_target_lock': enable_target_lock",
        "'point_cloud_stride': point_cloud_stride",
        "'target_color_hint': target_color_hint",
        "'external_target_mask_topic': external_target_mask_topic",
        "'require_external_target_mask': require_external_target_mask",
        "m2_move_group.launch.py",
        "moveit_sim_execute_server",
        "'action_name': '/grasp_executors/moveit_sim/execute'",
        "'hypotheses_topic': live_hypothesis_topic",
        "'marker_topic': live_marker_topic",
        "'hide_executed_hypotheses': True",
        "DeclareLaunchArgument('live_hypothesis_topic', default_value='/ghost_mgg/m4_live_hypotheses')",
        "DeclareLaunchArgument('live_marker_topic', default_value='/ghost_mgg/m4_joint_contract_markers')",
        "DeclareLaunchArgument('world_file', default_value='m2_tabletop_visual.sdf')",
        "DeclareLaunchArgument('world_name', default_value='ghost_mgg_m2_visual')",
        "DeclareLaunchArgument('enable_target_lock', default_value='false')",
        "DeclareLaunchArgument('point_cloud_stride', default_value='2')",
        "DeclareLaunchArgument('target_color_hint', default_value='red')",
        "DeclareLaunchArgument('external_target_mask_topic', default_value='/ghost_mgg/d435/external_target_mask')",
        "DeclareLaunchArgument('require_external_target_mask', default_value='false')",
        "DeclareLaunchArgument('show_rviz', default_value='false')",
        "DeclareLaunchArgument('verify_lift_and_hold', default_value='false')",
        "DeclareLaunchArgument('auto_execute', default_value='false')",
        "m4_no_truth_auto_execute_client",
        "execute_m4_joint_hypothesis_action.py",
        "condition=IfCondition(auto_execute)",
        "--hypotheses-topic",
        "--fallback-until-success",
        "--reset-executed-before-run",
        "--max-attempts",
        "--stable-hypotheses-count",
        "--stable-center-tolerance-m",
        "auto_execute_output_json",
        "'hide_all_after_success': True",
    ]:
        assert required in source

    for forbidden in [
        "m4_joint_hypotheses",
        "m4_joint_hypothesis_publisher_node.py",
        "m4_sim_grasp_targets.json",
        "current_targets.json",
    ]:
        assert forbidden not in source


def test_m4_no_truth_execute_client_is_installed_for_launch_auto_execute():
    cmake_path = BRINGUP_DIR / "CMakeLists.txt"
    cmake_text = cmake_path.read_text(encoding="utf-8")

    for required in [
        "install(PROGRAMS",
        "../../scripts/execute_m4_joint_hypothesis_action.py",
        "DESTINATION lib/${PROJECT_NAME}",
    ]:
        assert required in cmake_text


def test_m4_no_truth_live_moveit_smokes_cover_dryrun_execute_and_dynamic_rerun():
    dryrun = REPO_ROOT / "scripts" / "smoke_m4_no_truth_live_moveit_dryrun.sh"
    execute = REPO_ROOT / "scripts" / "smoke_m4_no_truth_live_moveit_execute.sh"
    dynamic = REPO_ROOT / "scripts" / "smoke_m4_no_truth_live_dynamic_execute.sh"
    fallback = REPO_ROOT / "scripts" / "smoke_m4_no_truth_live_ranked_fallback.sh"

    for path in [dryrun, execute, dynamic, fallback]:
        assert path.exists()
        source = path.read_text(encoding="utf-8")
        assert "m4_no_truth_live_moveit_execute.launch.py" in source
        assert "/ghost_mgg/m4_live_hypotheses" in source
        assert "m4_live_no_truth_ghost_mgg_v1" in source

    dryrun_text = dryrun.read_text(encoding="utf-8")
    for required in [
        "probe_m4_sim_grasp_moveit.py",
        "--input-mode hypotheses",
        "--hypotheses-topic /ghost_mgg/m4_live_hypotheses",
        "\"source_input\": \"/ghost_mgg/m4_live_hypotheses\"",
        "no_truth_audit=true",
        "M4 no-truth live MoveIt dry-run smoke passed",
    ]:
        assert required in dryrun_text

    execute_text = execute.read_text(encoding="utf-8")
    for required in [
        "execute_m4_joint_hypothesis_action.py",
        "--hypotheses-topic /ghost_mgg/m4_live_hypotheses",
        "--execute-action /grasp_executors/moveit_sim/execute",
        "--fallback-until-success",
        "--max-attempts",
        "--fallback-delay-sec",
        "--stable-hypotheses-count",
        "--stable-center-tolerance-m",
        "\"hypotheses_topic\": \"/ghost_mgg/m4_live_hypotheses\"",
        "status_name=SUCCEEDED",
        "no_truth_audit=true",
        "M4 no-truth live ExecuteGrasp smoke passed",
    ]:
        assert required in execute_text

    dynamic_text = dynamic.read_text(encoding="utf-8")
    for required in [
        "set_model_pose",
        "move_non_target_clutter_out_of_view",
        "initial_hypothesis_center=",
        "moved_hypothesis_center=",
        "clean_hypothesis_line",
        "awk '/^-?[0-9]+(\\.[0-9]+)?[[:space:]]+-?[0-9]+/",
        "[[ -z \"${initial_output}\" ]]",
        "delta_norm_ok=true",
        "--fallback-delay-sec",
        "rerun_execute_status=SUCCEEDED",
        "M4 no-truth live dynamic execute smoke passed",
    ]:
        assert required in dynamic_text

    fallback_text = fallback.read_text(encoding="utf-8")
    for required in [
        "execute_m4_joint_hypothesis_action.py",
        "--hypotheses-topic /ghost_mgg/m4_live_hypotheses",
        "--simulate-first-failure",
        "--fallback-until-success",
        "--max-attempts 3",
        "attempt_status_name",
        "simulated_failure",
        "fallback_attempt_count_ok=true",
        "fallback_final_status=SUCCEEDED",
        "M4 no-truth live ranked fallback smoke passed",
    ]:
        assert required in fallback_text


def test_m8_live_gazebo_execute_matrix_covers_direct_execute_scenarios():
    matrix = REPO_ROOT / "scripts" / "run_m8_live_gazebo_execute_matrix.sh"
    assert matrix.exists()
    source = matrix.read_text(encoding="utf-8")

    for required in [
        "normal_table",
        "depth_failure_table",
        "complex_table",
        "m4_no_truth_live_moveit_execute.launch.py",
        "grasp_m4_live_once.sh",
        "reports/m8_live_gazebo_execute_matrix",
        "m8_shape_library.sdf",
        "scenario_id:=S0",
        "scenario_id:=S3",
        "world_file:=m8_shape_library.sdf",
        "target_color_hint:=none",
        "enable_target_lock:=false",
        "verify_lift_and_hold:=false",
        "GHOST_MGG_M4_LIVE_OUTPUT_JSON",
        "matrix_summary.json",
        "matrix_summary.md",
    ]:
        assert required in source


def test_m8_live_gazebo_dynamic_rerun_matrix_moves_refreshes_and_executes():
    matrix = REPO_ROOT / "scripts" / "run_m8_live_gazebo_dynamic_rerun_matrix.sh"
    assert matrix.exists()
    source = matrix.read_text(encoding="utf-8")

    for required in [
        "normal_table",
        "depth_failure_table",
        "complex_table",
        "red_cube",
        "m8_medium_cube",
        "pose_a",
        "pose_b",
        "set_model_pose",
        "wait_for_hypothesis_near_target",
        "grasp_m4_live_once.sh",
        "--target-id",
        "reports/m8_live_gazebo_dynamic_rerun_matrix",
        "dynamic_summary.json",
        "dynamic_summary.md",
        "m8_shape_library.sdf",
        "scenario_id:=S0",
        "scenario_id:=S3",
        "target_color_hint:=none",
        "enable_target_lock:=false",
        "verify_lift_and_hold:=false",
        "target_error_m",
    ]:
        assert required in source
