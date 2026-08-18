from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m3_failure_camera_inspect_launch_adds_failure_emulator_outputs():
    launch_path = BRINGUP_DIR / "launch" / "m3_failure_camera_inspect.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text(encoding="utf-8")

    for required in [
        "DeclareLaunchArgument('headless', default_value='true')",
        "DeclareLaunchArgument('show_rviz', default_value='false')",
        "DeclareLaunchArgument('scenario_id', default_value='custom')",
        "DeclareLaunchArgument('failure_mode', default_value='mixed')",
        "DeclareLaunchArgument('roi_center_u_ratio', default_value='0.50')",
        "DeclareLaunchArgument('roi_center_v_ratio', default_value='0.58')",
        "DeclareLaunchArgument('roi_width_ratio', default_value='0.22')",
        "DeclareLaunchArgument('roi_height_ratio', default_value='0.22')",
        "DeclareLaunchArgument('table_leak_depth_m', default_value='1.20')",
        "DeclareLaunchArgument('flying_point_offset_m', default_value='0.12')",
        "DeclareLaunchArgument('biased_depth_offset_m', default_value='-0.04')",
        "DeclareLaunchArgument('edge_band_pixels', default_value='2')",
        "DeclareLaunchArgument('flying_point_stride', default_value='5')",
        "DeclareLaunchArgument('point_cloud_stride', default_value='1')",
        "DeclareLaunchArgument('world_file', default_value='m2_tabletop_visual.sdf')",
        "DeclareLaunchArgument('world_name', default_value='ghost_mgg_m2_visual')",
        "DeclareLaunchArgument('pattern_seed', default_value='0')",
        "m2_visual_scene.launch.py",
        "'world_file': world_file",
        "'world_name': world_name",
        "/ghost_mgg/d435/infra1/image_rect_raw@sensor_msgs/msg/Image[gz.msgs.Image",
        "/ghost_mgg/d435/infra1/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        "/ghost_mgg/d435/infra2/image_rect_raw@sensor_msgs/msg/Image[gz.msgs.Image",
        "/ghost_mgg/d435/infra2/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        "d435_infra1_frame",
        "d435_infra1_optical_frame",
        "d435_infra2_frame",
        "d435_infra2_optical_frame",
        "depth_failure_injector_node",
        "'failure_mode': failure_mode",
        "'input_depth_topic': '/ghost_mgg/d435/depth/image_rect_raw'",
        "'output_depth_topic': '/ghost_mgg/d435/depth/m3_corrupted'",
        "'hole_mask_topic': '/ghost_mgg/d435/evidence/hole_mask'",
        "'table_leakage_mask_topic': '/ghost_mgg/d435/evidence/table_leakage_mask'",
        "'edge_mask_topic': '/ghost_mgg/d435/evidence/edge_mask'",
        "'flying_point_mask_topic': '/ghost_mgg/d435/evidence/flying_point_mask'",
        "'biased_depth_mask_topic': '/ghost_mgg/d435/evidence/biased_depth_mask'",
        "'summary_topic': '/ghost_mgg/d435/evidence/summary'",
        "'roi_center_u_ratio': ParameterValue(roi_center_u_ratio, value_type=float)",
        "'roi_center_v_ratio': ParameterValue(roi_center_v_ratio, value_type=float)",
        "'roi_width_ratio': ParameterValue(roi_width_ratio, value_type=float)",
        "'roi_height_ratio': ParameterValue(roi_height_ratio, value_type=float)",
        "'table_leak_depth_m': ParameterValue(table_leak_depth_m, value_type=float)",
        "'flying_point_offset_m': ParameterValue(flying_point_offset_m, value_type=float)",
        "'biased_depth_offset_m': ParameterValue(biased_depth_offset_m, value_type=float)",
        "'edge_band_pixels': ParameterValue(edge_band_pixels, value_type=int)",
        "'flying_point_stride': ParameterValue(flying_point_stride, value_type=int)",
        "'pattern_seed': ParameterValue(pattern_seed, value_type=int)",
        "m3_raw_depth_to_point_cloud_node",
        "'points_topic': '/ghost_mgg/d435/depth/points'",
        "'pixel_stride': ParameterValue(point_cloud_stride, value_type=int)",
        "m3_raw_depth_to_mono8_node",
        "'preview_topic': '/ghost_mgg/d435/depth/image_viz'",
        "m3_target_mask_emulator_node",
        "'mask_topic': '/ghost_mgg/d435/target_mask'",
        "'use_mask_topic': True",
        "'target_mask_topic': '/ghost_mgg/d435/target_mask'",
        "'infra1_topic': '/ghost_mgg/d435/infra1/image_rect_raw'",
        "'infra2_topic': '/ghost_mgg/d435/infra2/image_rect_raw'",
        "m3_corrupted_depth_to_point_cloud_node",
        "'depth_topic': '/ghost_mgg/d435/depth/m3_corrupted'",
        "'points_topic': '/ghost_mgg/d435/depth/m3_points'",
        "'pixel_stride': ParameterValue(point_cloud_stride, value_type=int)",
        "m3_corrupted_depth_to_mono8_node",
        "'preview_topic': '/ghost_mgg/d435/depth/m3_image_viz'",
        "m3_evidence_summary_logger_node",
        "evidence_summary_logger_node",
        "'scenario_id': scenario_id",
        "'log_dir': 'log/ghost_mgg_trials/m3_failure'",
        "m3_failure_camera_inspect.rviz",
        "IfCondition(show_rviz)",
    ]:
        assert required in launch_text


def test_m3_failure_smoke_checks_corrupted_depth_and_evidence_topics():
    smoke_path = REPO_ROOT / "scripts" / "smoke_m3_failure_camera_inspect.sh"
    assert smoke_path.exists()

    smoke_text = smoke_path.read_text(encoding="utf-8")

    for required in [
        "ros2 launch ghost_mgg_bringup m3_failure_camera_inspect.launch.py",
        "headless:=true",
        "show_rviz:=false",
        'scenario_id:="${scenario_id}"',
        'failure_mode:="${failure_mode}"',
        "/ghost_mgg/d435/depth/image_viz sensor_msgs/msg/Image",
        "/ghost_mgg/d435/depth/points sensor_msgs/msg/PointCloud2",
        "/ghost_mgg/d435/infra1/image_rect_raw sensor_msgs/msg/Image",
        "/ghost_mgg/d435/infra2/image_rect_raw sensor_msgs/msg/Image",
        "/ghost_mgg/d435/target_mask sensor_msgs/msg/Image",
        "/ghost_mgg/d435/depth/m3_corrupted sensor_msgs/msg/Image",
        "/ghost_mgg/d435/depth/m3_image_viz sensor_msgs/msg/Image",
        "/ghost_mgg/d435/depth/m3_points sensor_msgs/msg/PointCloud2",
        "/ghost_mgg/d435/evidence/hole_mask sensor_msgs/msg/Image",
        "/ghost_mgg/d435/evidence/table_leakage_mask sensor_msgs/msg/Image",
        "/ghost_mgg/d435/evidence/edge_mask sensor_msgs/msg/Image",
        "/ghost_mgg/d435/evidence/flying_point_mask sensor_msgs/msg/Image",
        "/ghost_mgg/d435/evidence/biased_depth_mask sensor_msgs/msg/Image",
        "/ghost_mgg/d435/evidence/summary std_msgs/msg/String",
        'grep -Fq "\\"failure_mode\\":\\"${failure_mode}\\""',
        '"evidence_source":"target_mask"',
        '"hole_pixels":',
        '"table_leakage_pixels":',
        '"edge_pixels":',
        '"flying_point_pixels":',
        '"biased_depth_pixels":',
        "has_ir: true",
        "has_mask: true",
        "find \"${trial_log_dir}\"",
        "scenario_id",
        "_evidence.jsonl",
    ]:
        assert required in smoke_text


def test_m3_failure_scenario_launch_declares_s0_to_s7_presets():
    launch_path = BRINGUP_DIR / "launch" / "m3_failure_scenario.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text(encoding="utf-8")

    for required in [
        "DeclareLaunchArgument('scenario_id', default_value='S6')",
        "DeclareLaunchArgument('point_cloud_stride', default_value='1')",
        "DeclareLaunchArgument('world_file', default_value='m2_tabletop_visual.sdf')",
        "DeclareLaunchArgument('world_name', default_value='ghost_mgg_m2_visual')",
        "m3_failure_camera_inspect.launch.py",
        "'S0'",
        "'S1'",
        "'S2'",
        "'S3'",
        "'S4'",
        "'S5'",
        "'S6'",
        "'S7'",
        "'failure_mode': 'disabled'",
        "'failure_mode': 'hole'",
        "'failure_mode': 'table_leakage'",
        "'failure_mode': 'edge_flying'",
        "'failure_mode': 'reflective'",
        "'edge_band_pixels': '3'",
        "'flying_point_stride': '4'",
        "'biased_depth_offset_m': '-0.05'",
    ]:
        assert required in launch_text


def test_m3_failure_scenario_smoke_runs_s0_to_s7_matrix():
    smoke_path = REPO_ROOT / "scripts" / "smoke_m3_failure_scenarios.sh"
    assert smoke_path.exists()

    smoke_text = smoke_path.read_text(encoding="utf-8")

    for required in [
        "smoke_m3_failure_camera_inspect.sh",
        "run_scenario S0 disabled",
        "run_scenario S1 disabled",
        "run_scenario S2 mixed",
        "run_scenario S3 hole",
        "run_scenario S4 table_leakage",
        "run_scenario S5 edge_flying",
        "run_scenario S6 mixed",
        "run_scenario S7 reflective",
        "GHOST_MGG_M3_SCENARIO_ID",
        "GHOST_MGG_M3_FAILURE_MODE",
        "M3 S0-S7 scenario smoke passed",
    ]:
        assert required in smoke_text
