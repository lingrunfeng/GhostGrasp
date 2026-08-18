from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m4_joint_hypothesis_execute_launch_starts_moveit_executor():
    launch_path = BRINGUP_DIR / "launch" / "m4_joint_hypothesis_moveit_execute.launch.py"
    assert launch_path.exists()
    launch_text = launch_path.read_text(encoding="utf-8")

    for required in [
        "LaunchConfiguration('headless')",
        "LaunchConfiguration('show_rviz')",
        "'headless': headless",
        "DeclareLaunchArgument('headless', default_value='true')",
        "DeclareLaunchArgument('show_rviz', default_value='false')",
        "rviz2",
        "m4_joint_execute.rviz",
        "IfCondition(show_rviz)",
        "m4_sim_grasp_moveit_dryrun.launch.py",
        "sim_grasp_targets_path",
        "joint_hypothesis_report_path",
        "joint_hypothesis_topic",
        "executed_hypotheses_topic",
        "m4_joint_hypothesis_markers_node.py",
        "'marker_topic': '/ghost_mgg/m4_joint_hypothesis_markers'",
        "'executed_hypotheses_topic': executed_hypotheses_topic",
        "m4_joint_contract_markers_node",
        "executable='hypothesis_markers_node'",
        "'marker_topic': '/ghost_mgg/m4_joint_contract_markers'",
        "'hide_executed_hypotheses': True",
        "moveit_sim_execute_server",
        "'action_name': '/grasp_executors/moveit_sim/execute'",
        "'add_m2_scene_obstacles': True",
        "'use_grasp_orientation': True",
        "'verify_lift_and_hold': False",
        "'enable_gripper': True",
        "MoveItConfigsBuilder",
    ]:
        assert required in launch_text


def test_m4_joint_execute_rviz_is_execution_view_not_camera_view():
    rviz_path = REPO_ROOT / "src" / "ghost_mgg_sim" / "rviz" / "m4_joint_execute.rviz"
    assert rviz_path.exists()
    rviz_text = rviz_path.read_text(encoding="utf-8")

    for required in [
        "RobotModel",
        "TF",
        "M4 Sim Grasp Markers",
        "/ghost_mgg/m4_sim_grasp_markers",
        "M4 Joint Hypothesis Markers",
        "/ghost_mgg/m4_joint_hypothesis_markers",
        "M4 Joint Contract Markers",
        "/ghost_mgg/m4_joint_contract_markers",
    ]:
        assert required in rviz_text

    for camera_topic in [
        "/ghost_mgg/d435/depth/points",
        "/ghost_mgg/d435/depth/m3_points",
        "/ghost_mgg/d435/color/image_raw",
        "/ghost_mgg/d435/depth/image_rect_raw",
    ]:
        assert camera_topic not in rviz_text


def test_m4_execute_script_sends_top_valid_hypothesis_action_goal():
    script_path = REPO_ROOT / "scripts" / "execute_m4_joint_hypothesis_action.py"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "ExecuteGrasp",
        "GeometryHypothesisArray",
        "wait_for_hypotheses",
        "select_valid_hypothesis",
        "async_send_goal",
        "async_get_result",
        "publish_execution_event",
        "reset_execution_events",
        "--hypotheses-topic",
        "--execute-action",
        "--executed-topic",
        "--reset-executed-before-run",
        "--stable-hypotheses-count",
        "--stable-center-tolerance-m",
        "--target-x",
        "--target-y",
        "--target-tolerance-m",
        "target_xy",
        "target_error_m",
        "--output-json",
        "/ghost_mgg/m4_executed_hypotheses",
        "m4_joint_hypothesis_execute_v1",
        "selected_pose_base",
        "selected_shape_type",
        "dimensions_m",
    ]:
        assert required in source


def test_m4_execute_script_supports_ranked_fallback_attempt_log():
    script_path = REPO_ROOT / "scripts" / "execute_m4_joint_hypothesis_action.py"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "select_valid_hypotheses",
        "execute_hypothesis",
        "--fallback-until-success",
        "--simulate-failure-id",
        "--simulate-first-failure",
        "--max-attempts",
        "--fallback-delay-sec",
        "--no-publish-executed-event",
        "attempts",
        "attempt_status_name",
        "simulated_failure",
        "simulate_first_failure",
        "parse_known_args",
        "rclpy.spin_once(node, timeout_sec=0.05)",
        "final_status_name",
    ]:
        assert required in source


def test_m4_joint_hypothesis_execute_smoke_runs_launch_and_action_client():
    smoke_path = REPO_ROOT / "scripts" / "smoke_m4_joint_hypothesis_execute.sh"
    assert smoke_path.exists()
    smoke_text = smoke_path.read_text(encoding="utf-8")

    for required in [
        "m4_joint_hypothesis_moveit_execute.launch.py",
        "execute_m4_joint_hypothesis_action.py",
        "/ghost_mgg/m4_joint_hypotheses ghost_mgg_interfaces/msg/GeometryHypothesisArray",
        "/grasp_executors/moveit_sim/execute",
        "status_name",
        "SUCCEEDED",
        "--fallback-until-success",
        "--simulate-failure-id blue_cylinder",
        "--executed-topic /ghost_mgg/m4_executed_hypotheses",
        "--reset-executed-before-run",
        "attempt_status_name",
        "M4 joint hypothesis ExecuteGrasp smoke passed",
    ]:
        assert required in smoke_text


def test_grasp_once_wrapper_runs_m4_execute_client_with_fallback_defaults():
    script_path = REPO_ROOT / "scripts" / "grasp_m4_once.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "execute_m4_joint_hypothesis_action.py",
        "--hypotheses-topic /ghost_mgg/m4_joint_hypotheses",
        "--execute-action /grasp_executors/moveit_sim/execute",
        "--fallback-until-success",
        "--reset-executed-before-run",
        "--max-attempts",
        "GHOST_MGG_M4_MAX_ATTEMPTS",
        "manual_result.json",
    ]:
        assert required in source


def test_grasp_live_once_wrapper_uses_current_live_hypothesis_contract():
    script_path = REPO_ROOT / "scripts" / "grasp_m4_live_once.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "execute_m4_joint_hypothesis_action.py",
        "--hypotheses-topic /ghost_mgg/m4_live_hypotheses",
        "--execute-action /grasp_executors/moveit_sim/execute",
        "--executed-topic /ghost_mgg/m4_executed_hypotheses",
        "--fallback-until-success",
        "--reset-executed-before-run",
        "--stable-hypotheses-count",
        "--stable-center-tolerance-m",
        "GHOST_MGG_M4_LIVE_MAX_ATTEMPTS",
        "GHOST_MGG_M4_LIVE_STABLE_COUNT",
        "manual_live_once.json",
    ]:
        assert required in source


def test_manual_roi_mask_publisher_is_installed_and_targets_external_mask_topic():
    script_path = REPO_ROOT / "scripts" / "publish_m4_target_roi_mask.py"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")
    cmake_text = (BRINGUP_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    for required in [
        "sensor_msgs.msg",
        "Image",
        "/ghost_mgg/d435/external_target_mask",
        "--bbox",
        "--width",
        "--height",
        "DurabilityPolicy.TRANSIENT_LOCAL",
        "ReliabilityPolicy.RELIABLE",
        "mono8",
    ]:
        assert required in source

    assert "../../scripts/publish_m4_target_roi_mask.py" in cmake_text


def test_color_mask_adapter_is_installed_and_targets_external_mask_topic():
    script_path = REPO_ROOT / "scripts" / "publish_m4_color_target_mask.py"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")
    cmake_text = (BRINGUP_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    for required in [
        "sensor_msgs.msg",
        "Image",
        "/ghost_mgg/d435/color/image_raw",
        "/ghost_mgg/d435/external_target_mask",
        "--color-hint",
        "DurabilityPolicy.TRANSIENT_LOCAL",
        "ReliabilityPolicy.RELIABLE",
        "mono8",
        "m4_external_color_mask_adapter",
    ]:
        assert required in source

    assert "../../scripts/publish_m4_color_target_mask.py" in cmake_text
