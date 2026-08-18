from pathlib import Path


SIM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
BRINGUP_DIR = REPO_ROOT / "src" / "ghost_mgg_bringup"


def test_probe_records_fk_path_points_for_rviz_plan_markers():
    probe_text = (REPO_ROOT / "scripts" / "probe_m4_sim_grasp_moveit.py").read_text(
        encoding="utf-8"
    )

    for required in [
        "GetPositionFK",
        "/compute_fk",
        "fk_link_names",
        "gripper_tcp",
        "sample_trajectory_points",
        "path_points_world",
        "path_point_stride",
        "build_descent_points_world",
        "descent_points_world",
        "descent_clearance",
        "table_top_z_m",
        "descent_samples",
    ]:
        assert required in probe_text


def test_sim_package_installs_moveit_plan_marker_node():
    cmake_text = (SIM_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    package_text = (SIM_DIR / "package.xml").read_text(encoding="utf-8")
    marker_node = SIM_DIR / "scripts" / "m4_sim_moveit_plan_markers_node.py"
    source = marker_node.read_text(encoding="utf-8")

    assert marker_node.exists()
    assert "install(PROGRAMS" in cmake_text
    assert "scripts/m4_sim_moveit_plan_markers_node.py" in cmake_text
    assert "<exec_depend>rclpy</exec_depend>" in package_text
    for required in [
        "/ghost_mgg/m4_sim_moveit_plan_markers",
        "m4_sim_moveit_plan_path",
        "m4_sim_moveit_plan_goal",
        "m4_sim_moveit_descent_path",
        "m4_sim_moveit_descent_goal",
        "m4_sim_moveit_plan_text",
        "path_points_world",
        "descent_points_world",
        "descent_clearance",
        "LINE_STRIP",
        "SPHERE",
        "TEXT_VIEW_FACING",
        "planned={planned}/{total}",
        "clearance=ok",
        "transient_local",
    ]:
        assert required in source


def test_m4_inspect_launch_and_rviz_show_moveit_plan_markers():
    launch_text = (
        BRINGUP_DIR / "launch" / "m4_offline_ranking_inspect.launch.py"
    ).read_text(encoding="utf-8")
    rviz_text = (SIM_DIR / "rviz" / "m3_failure_camera_inspect.rviz").read_text(
        encoding="utf-8"
    )

    for required in [
        "moveit_plan_report_path",
        "m4_sim_moveit_plan_markers_node.py",
        "'marker_topic': '/ghost_mgg/m4_sim_moveit_plan_markers'",
        "'report_path': moveit_plan_report_path",
    ]:
        assert required in launch_text

    assert "M4 Sim MoveIt Plan Markers" in rviz_text
    assert "/ghost_mgg/m4_sim_moveit_plan_markers" in rviz_text


def test_m4_marker_smoke_checks_moveit_plan_marker_topic():
    smoke_text = (
        REPO_ROOT / "scripts" / "smoke_m4_offline_ranking_markers.sh"
    ).read_text(encoding="utf-8")

    for required in [
        "GHOST_MGG_M4_MOVEIT_PLAN_REPORT_PATH",
        "smoke_m4_sim_grasp_moveit_dryrun.sh",
        "moveit_plan_report_path:=",
        "/ghost_mgg/m4_sim_moveit_plan_markers visualization_msgs/msg/MarkerArray",
        "m4_sim_moveit_plan_path",
        "m4_sim_moveit_plan_goal",
        "m4_sim_moveit_descent_path",
        "clearance=ok",
        "planned=4/4",
    ]:
        assert required in smoke_text
