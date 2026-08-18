from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m2_mask_extrusion_launch_selects_real_baseline_backend():
    launch_path = BRINGUP_DIR / "launch" / "m2_mask_extrusion_bt_loop.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text()

    assert "m2_visual_scene.launch.py" in launch_text
    assert "mask_extrusion_recovery_server" in launch_text
    assert "dummy_recovery_server" not in launch_text
    assert "'action_name': '/geometry_backends/mask_extrusion/recover'" in launch_text
    assert "'hypotheses_topic': '/ghost_mgg/hypotheses/mask_extrusion'" in launch_text
    assert "DeclareLaunchArgument('preferred_hypothesis_id', default_value='')" in launch_text
    assert "LaunchConfiguration('preferred_hypothesis_id')" in launch_text
    assert "'preferred_hypothesis_id': preferred_hypothesis_id" in launch_text
    assert "hypothesis_markers_node" in launch_text
    assert "'hypotheses_topic': '/ghost_mgg/hypotheses/mask_extrusion'" in launch_text
    assert "'marker_topic': '/ghost_mgg/hypothesis_markers'" in launch_text
    assert "mycobot_sim_execute_server" in launch_text
    assert "'trajectory_action_name': '/arm_controller/follow_joint_trajectory'" in launch_text
    assert "bt_runner_node" in launch_text
    assert "m2_sim_closed_loop.xml" in launch_text
    assert "'backend_name': 'mask_extrusion'" in launch_text
    assert "'executor_name': 'mycobot_sim'" in launch_text
    assert "'max_hypotheses': 4" in launch_text
    assert "'trial_log_dir': 'log/ghost_mgg_trials/m2_mask_extrusion'" in launch_text


def test_m2_mask_extrusion_loop_includes_camera_observation_chain():
    launch_path = BRINGUP_DIR / "launch" / "m2_mask_extrusion_bt_loop.launch.py"
    launch_text = launch_path.read_text(encoding="utf-8")

    for required in [
        "ros_gz_bridge",
        "parameter_bridge",
        "/ghost_mgg/d435/color/image_raw",
        "/ghost_mgg/d435/depth/image_rect_raw",
        "/ghost_mgg/d435/color/camera_info",
        "/ghost_mgg/d435/depth/camera_info",
        "depth_to_point_cloud_node",
        "depth_to_mono8_node",
        "/ghost_mgg/d435/depth/points",
        "/ghost_mgg/d435/depth/image_viz",
        "observation_cache_node",
        "/ghost_mgg/observations/latest",
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
        "d435_color_optical_frame",
        "d435_depth_optical_frame",
        "launch_arguments={'headless': 'true'}.items()",
    ]:
        assert required in launch_text


def test_m2_mask_extrusion_inspect_launch_adds_rviz_only():
    launch_path = BRINGUP_DIR / "launch" / "m2_mask_extrusion_inspect.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text(encoding="utf-8")

    assert "m2_mask_extrusion_bt_loop.launch.py" in launch_text
    assert "rviz2" in launch_text
    assert "m2_camera_inspect.rviz" in launch_text
    assert "m2_mask_extrusion_inspect_rviz" in launch_text
    assert "m2_visual_scene.launch.py" not in launch_text


def test_m2_mask_extrusion_smoke_checks_backend_observation_and_markers():
    smoke_path = REPO_ROOT / "scripts" / "smoke_m2_mask_extrusion_bt_loop.sh"
    assert smoke_path.exists()

    smoke_text = smoke_path.read_text(encoding="utf-8")

    assert "ros2 launch ghost_mgg_bringup m2_mask_extrusion_bt_loop.launch.py" in smoke_text
    assert "/ghost_mgg/observations/latest ghost_mgg_interfaces/msg/ObservationRef" in smoke_text
    assert "/ghost_mgg/hypotheses/mask_extrusion ghost_mgg_interfaces/msg/GeometryHypothesisArray" in smoke_text
    assert "/ghost_mgg/hypothesis_markers visualization_msgs/msg/MarkerArray" in smoke_text
    assert "/ghost_mgg/d435/depth/points sensor_msgs/msg/PointCloud2" in smoke_text
    assert "log/ghost_mgg_trials/m2_mask_extrusion" in smoke_text
    assert "backend_name: mask_extrusion" in smoke_text
    assert "mask_extrusion_red_cube" in smoke_text
    assert "mask_extrusion_glass_block" in smoke_text
    assert "mask_extrusion_green_cylinder" in smoke_text
    assert "shape_type: 1" in smoke_text
