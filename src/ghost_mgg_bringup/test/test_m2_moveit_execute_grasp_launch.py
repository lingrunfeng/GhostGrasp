from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_backend_package_builds_moveit_execute_server():
    backends_dir = REPO_ROOT / "src" / "ghost_mgg_backends"
    cmake_text = (backends_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    package_text = (backends_dir / "package.xml").read_text(encoding="utf-8")
    executor_text = (
        backends_dir / "src" / "moveit_sim_execute_server.cpp"
    ).read_text(encoding="utf-8")

    for required in [
        "moveit_sim_execute_server",
        "src/moveit_sim_execute_server.cpp",
        "src/moveit_sim_execute_server_main.cpp",
        "moveit_ros_planning_interface",
        "moveit_msgs",
        "shape_msgs",
    ]:
        assert required in cmake_text

    for required in [
        "<depend>moveit_ros_planning_interface</depend>",
        "<depend>moveit_msgs</depend>",
        "<depend>shape_msgs</depend>",
    ]:
        assert required in package_text

    assert 'constexpr const char * kDefaultEndEffectorLink = "gripper_tcp";' in executor_text


def test_bt_registry_exposes_moveit_sim_executor():
    registry_path = REPO_ROOT / "src" / "ghost_mgg_bt" / "config" / "backend_registry.yaml"
    registry_text = registry_path.read_text(encoding="utf-8")

    assert "moveit_sim:" in registry_text
    assert "execute_action: /grasp_executors/moveit_sim/execute" in registry_text


def test_m2_mask_extrusion_moveit_loop_wires_moveit_executor_and_bt():
    launch_path = BRINGUP_DIR / "launch" / "m2_mask_extrusion_moveit_loop.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text(encoding="utf-8")

    for required in [
        "DeclareLaunchArgument('headless', default_value='true')",
        "DeclareLaunchArgument('preferred_hypothesis_id', default_value='')",
        "DeclareLaunchArgument('strict_preferred_hypothesis', default_value='false')",
        "DeclareLaunchArgument('max_hypotheses', default_value='4')",
        "LaunchConfiguration('headless')",
        "LaunchConfiguration('preferred_hypothesis_id')",
        "LaunchConfiguration('strict_preferred_hypothesis')",
        "LaunchConfiguration('max_hypotheses')",
        "m2_visual_scene.launch.py",
        "launch_arguments={'headless': headless}.items()",
        "m2_move_group.launch.py",
        "mask_extrusion_recovery_server",
        "'preferred_hypothesis_id': preferred_hypothesis_id",
        "'strict_preferred_hypothesis': strict_preferred_hypothesis",
        "moveit_sim_execute_server",
        "'action_name': '/grasp_executors/moveit_sim/execute'",
        "'end_effector_link': 'gripper_tcp'",
        "'gripper_trajectory_action_name': '/gripper_group_controller/follow_joint_trajectory'",
        "'enable_gripper': True",
        "'gripper_open_position': 0.15",
        "'gripper_close_position': -0.38",
        "'gripper_max_effort': 45.0",
        "'gripper_motion_duration_sec': 0.60",
        "'wait_for_close_gripper_result': True",
        "'settle_time_sec': 0.80",
        "'verify_lift_and_hold': True",
        "'min_lift_z_delta': 0.010",
        "'lift_hold_duration_sec': 0.50",
        "'lift_hold_sample_period_sec': 0.10",
        "'use_grasp_orientation': True",
        "'goal_position_tolerance_m': 0.006",
        "'goal_orientation_tolerance_rad': 0.45",
        "'add_m2_scene_obstacles': True",
        "'scene_object_padding_m': 0.006",
        "'use_cartesian_top_grasp_segments': False",
        "'cartesian_step_m': 0.004",
        "'cartesian_jump_threshold': 0.0",
        "'min_cartesian_fraction': 0.95",
        "hypothesis_markers_node",
        "observation_cache_node",
        "depth_to_point_cloud_node",
        "depth_to_mono8_node",
        "/ghost_mgg/d435/depth/image_viz",
        "m2_world_to_base_link_tf",
        "'--x', '-0.171207'",
        "'--y', '0.228790'",
        "'--z', '0.765000'",
        "'--yaw', '-2.180640'",
        "m2_world_to_d435_link_tf",
        "'--x', '0.286775'",
        "'--y', '-0.079936'",
        "'--z', '1.105590'",
        "'--pitch', '0.950000'",
        "'--yaw', '2.754700'",
        "bt_runner_node",
        "'backend_name': 'mask_extrusion'",
        "'executor_name': 'moveit_sim'",
        "'execute_timeout_sec': 65.0",
        "'max_hypotheses': max_hypotheses",
        "'trial_log_dir': 'log/ghost_mgg_trials/m2_mask_extrusion_moveit'",
    ]:
        assert required in launch_text

    assert "mycobot_sim_execute_server" not in launch_text
    assert "'executor_name': 'mycobot_sim'" not in launch_text


def test_m2_mask_extrusion_moveit_inspect_launch_adds_gazebo_gui_and_rviz():
    launch_path = BRINGUP_DIR / "launch" / "m2_mask_extrusion_moveit_inspect.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text(encoding="utf-8")

    assert "m2_mask_extrusion_moveit_loop.launch.py" in launch_text
    assert "'headless': 'false'" in launch_text
    assert "'preferred_hypothesis_id': preferred_hypothesis_id" in launch_text
    assert "DeclareLaunchArgument('preferred_hypothesis_id', default_value='')" in launch_text
    assert "rviz2" in launch_text
    assert "m2_camera_inspect.rviz" in launch_text
    assert "m2_mask_extrusion_moveit_inspect_rviz" in launch_text
    assert "m2_visual_scene.launch.py" not in launch_text


def test_m2_mask_extrusion_moveit_smoke_checks_moveit_closed_loop():
    smoke_path = REPO_ROOT / "scripts" / "smoke_m2_mask_extrusion_moveit_loop.sh"
    assert smoke_path.exists()

    smoke_text = smoke_path.read_text(encoding="utf-8")

    assert "ros2 launch ghost_mgg_bringup m2_mask_extrusion_moveit_loop.launch.py" in smoke_text
    assert "ros2 node list --no-daemon --spin-time 2" in smoke_text
    assert "/move_group" in smoke_text
    assert "/ghost_mgg/hypotheses/mask_extrusion ghost_mgg_interfaces/msg/GeometryHypothesisArray" in smoke_text
    assert "/ghost_mgg/hypothesis_markers visualization_msgs/msg/MarkerArray" in smoke_text
    assert "log/ghost_mgg_trials/m2_mask_extrusion_moveit" in smoke_text
    assert '"backend_name":"mask_extrusion"' in smoke_text
    assert '"final_status":"succeeded"' in smoke_text
    assert 'preferred_hypothesis_id:="' in smoke_text
    assert "strict_preferred_hypothesis:=true" in smoke_text
    assert "max_hypotheses:=1" in smoke_text
    assert 'selected_hypothesis_id' in smoke_text
    assert '${preferred_hypothesis_id}' in smoke_text
    assert "gz model -m ghost_mgg_mycobot_280 -p" in smoke_text
    assert "gripper_group_controller" in smoke_text
    assert "target_model_lifted" in smoke_text
    assert "gz model -m ${target_model_name} -p" in smoke_text
    assert "min_lift_z_delta = 0.010" in smoke_text
    assert "verify_lift_and_hold" in smoke_text

    all_targets_smoke = (
        REPO_ROOT / "scripts" / "smoke_m2_mask_extrusion_moveit_all_targets.sh"
    ).read_text(encoding="utf-8")
    for target in [
        "mask_extrusion_glass_block",
        "mask_extrusion_red_cube",
        "mask_extrusion_blue_cylinder",
        "mask_extrusion_green_cylinder",
    ]:
        assert target in all_targets_smoke
    assert "smoke_m2_mask_extrusion_moveit_loop.sh" in all_targets_smoke
