from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m6_site_check_script_records_environment_without_motion_commands():
    script = REPO_ROOT / "scripts" / "collect_m6_mycobot_shadow_site_log.sh"
    source = script.read_text(encoding="utf-8")

    for required in [
        "M6 myCobot Shadow Site Check",
        "ip -br addr",
        "ip -br link",
        "ip neigh show dev",
        "ros2 topic list -t",
        "ros2 pkg list",
        "pymycobot",
        "serial.tools.list_ports",
        "reports/m6_mycobot_shadow_site_check",
        "No real motion command is sent by this script.",
    ]:
        assert required in source

    for forbidden in [
        "send_angles",
        "send_coords",
        "set_encoder",
        "set_encoders",
        "ros2 action send_goal",
        "moveit_sim_execute_server",
    ]:
        assert forbidden not in source


def test_m6_shadow_smoke_launches_plan_only_moveit_without_execution_paths():
    script = REPO_ROOT / "scripts" / "smoke_m6_moveit_shadow_plan_only.sh"
    source = script.read_text(encoding="utf-8")

    for required in [
        "m6_shadow_move_group.launch.py",
        "show_rviz:=false",
        "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST",
        "unset ROS_STATIC_PEERS",
        "ros2 daemon stop",
        "--no-daemon --spin-time 1",
        "probe_m2_moveit_plan.py",
        "/plan_kinematic_path",
        "/joint_states",
        "/tf",
        "/move_group",
        "M6 MoveIt shadow plan-only smoke passed",
    ]:
        assert required in source

    for forbidden in [
        "ros2 action send_goal",
        "follow_joint_trajectory",
        "moveit_sim_execute_server",
        "execute_m4_joint_hypothesis_action.py",
        "send_angles",
        "send_coords",
    ]:
        assert forbidden not in source


def test_m6_shadow_readiness_shell_records_verified_tf_and_latest_plan_evidence():
    script = REPO_ROOT / "scripts" / "run_m6_shadow_readiness.sh"
    source = script.read_text(encoding="utf-8")

    for required in [
        "--real-tf-checked",
        "M6_MOVEIT_SHADOW_EVIDENCE_PATH",
        "M6_REAL_STATE_MOVEIT_SHADOW_EVIDENCE_PATH",
        "--moveit-shadow-evidence-path",
        "--real-state-moveit-shadow-evidence-path",
        "reports/m6_shadow_grasp_targets",
    ]:
        assert required in source
