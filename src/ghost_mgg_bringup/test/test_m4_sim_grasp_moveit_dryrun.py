from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m4_sim_grasp_moveit_dryrun_launch_is_planning_only():
    launch_path = BRINGUP_DIR / "launch" / "m4_sim_grasp_moveit_dryrun.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text(encoding="utf-8")

    for required in [
        "m2_visual_scene.launch.py",
        "LaunchConfiguration('headless')",
        "'headless': headless",
        "DeclareLaunchArgument('headless', default_value='true')",
        "m2_move_group.launch.py",
        "m4_sim_grasp_markers_node",
        "'marker_topic': '/ghost_mgg/m4_sim_grasp_markers'",
        "'targets_path': sim_grasp_targets_path",
        "TimerAction(period=13.0, actions=[move_group])",
        "TimerAction(period=14.0, actions=[sim_grasp_markers])",
    ]:
        assert required in launch_text

    for forbidden in [
        "rviz2",
        "bt_runner_node",
        "moveit_sim_execute_server",
        "mask_extrusion_recovery_server",
        "ExecuteGrasp",
    ]:
        assert forbidden not in launch_text


def test_m4_sim_grasp_moveit_probe_uses_pose_constraints_and_scene_obstacles():
    probe_text = (REPO_ROOT / "scripts" / "probe_m4_sim_grasp_moveit.py").read_text(
        encoding="utf-8"
    )

    for required in [
        "m4_sim_grasp_targets.json",
        "GetMotionPlan",
        "ApplyPlanningScene",
        "AllowedCollisionEntry",
        "PositionConstraint",
        "OrientationConstraint",
        "start_state.joint_state",
        "final_joint_names",
        "final_joint_positions",
        "m4_table_collision",
        "--planning-frame",
        "args.planning_frame",
        "--table-center-x-m",
        "--table-center-y-m",
        "m4_obstacle_",
        "--omit-target-obstacles",
        "--allowed-collision-pair",
        "allowed_collision_pair",
        "omit_target_obstacles",
        "gripper_tcp",
        "top_down_quaternion",
        "pregrasp_clearance_m",
        "schema_version",
        "m4_sim_moveit_dryrun_v1",
        "target_id=",
        "status=planned",
    ]:
        assert required in probe_text


def test_m4_sim_grasp_moveit_smoke_runs_dryrun_launch_and_probe():
    smoke_text = (
        REPO_ROOT / "scripts" / "smoke_m4_sim_grasp_moveit_dryrun.sh"
    ).read_text(encoding="utf-8")

    for required in [
        "m4_sim_grasp_moveit_dryrun.launch.py",
        "probe_m4_sim_grasp_moveit.py",
        "/ghost_mgg/m4_sim_grasp_markers visualization_msgs/msg/MarkerArray",
        "red_cube",
        "blue_cylinder",
        "green_cylinder",
        "glass_block",
        "M4 sim grasp MoveIt dry-run smoke passed",
    ]:
        assert required in smoke_text
