from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_PACKAGE = REPO_ROOT / "src" / "ghost_mgg_sim"
VENDORED_DESCRIPTION = REPO_ROOT / "src" / "ghost_mgg_mycobot_description"


def read_text(path):
    return path.read_text(encoding="utf-8")


def package_files(root):
    return [path for path in root.rglob("*") if path.is_file()]


def assert_no_forbidden_text(paths, forbidden_text):
    offenders = []
    for path in paths:
        try:
            text = read_text(path)
        except UnicodeDecodeError:
            continue

        for forbidden in forbidden_text:
            if forbidden in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {forbidden!r}")

    assert not offenders, "\n".join(offenders)


def test_vendored_description_package_contains_required_assets():
    required_paths = [
        "package.xml",
        "CMakeLists.txt",
        "LICENSE",
        "urdf/robots/mycobot_280.urdf.xacro",
        "urdf/mech/mycobot_280_arm.urdf.xacro",
        "urdf/mech/adaptive_gripper.urdf.xacro",
        "meshes/mycobot_280/visual/link1.dae",
        "meshes/adaptive_gripper/visual/gripper_base.dae",
    ]

    missing = [
        relative_path
        for relative_path in required_paths
        if not (VENDORED_DESCRIPTION / relative_path).exists()
    ]

    assert not missing, "Missing vendored assets: " + ", ".join(missing)


def test_adaptive_gripper_defines_tcp_link_near_finger_center():
    gripper_text = read_text(
        VENDORED_DESCRIPTION / "urdf" / "mech" / "adaptive_gripper.urdf.xacro"
    )

    assert 'link name="${prefix}gripper_tcp"' in gripper_text
    assert 'joint name="${prefix}gripper_base_to_${prefix}gripper_tcp" type="fixed"' in gripper_text
    assert '<parent link="${prefix}gripper_base"/>' in gripper_text
    assert '<child link="${prefix}gripper_tcp"/>' in gripper_text
    assert '<origin xyz="0.0 0.055 -0.010" rpy="0 0 0"/>' in gripper_text


@pytest.mark.parametrize("xacro_path", sorted(VENDORED_DESCRIPTION.rglob("*.xacro")))
def test_vendored_xacro_references_vendored_description_package(xacro_path):
    text = read_text(xacro_path)

    assert "$(find mycobot_description)" not in text
    if "$(find " in text:
        assert "$(find ghost_mgg_mycobot_description)" in text


def test_vendored_description_package_has_no_external_or_moveit_references():
    assert_no_forbidden_text(
        package_files(VENDORED_DESCRIPTION),
        [
            "mycobot_moveit_config",
            "/home/student26/Leorover-team8/src",
            "/home/student26/Leorover-team8/install",
        ],
    )


def test_visual_scene_launch_uses_vendored_description_package_only():
    launch_text = read_text(SIM_PACKAGE / "launch" / "m2_visual_scene.launch.py")

    assert "ghost_mgg_mycobot_description" in launch_text
    for forbidden in [
        "/home/student26/Leorover-team8/src",
        "/home/student26/Leorover-team8/install",
        "FindPackageShare('mycobot_description')",
        'FindPackageShare("mycobot_description")',
        "$(find mycobot_description)",
    ]:
        assert forbidden not in launch_text


def test_visual_scene_spawn_initializes_robot_description_parameter():
    launch_text = read_text(SIM_PACKAGE / "launch" / "m2_visual_scene.launch.py")

    assert "executable='create'" in launch_text
    assert "'-param'," in launch_text
    assert "'robot_description'," in launch_text
    assert "parameters=[{'robot_description': robot_description}]" in launch_text


def test_visual_scene_bridges_gazebo_clock_into_ros_time():
    launch_text = read_text(SIM_PACKAGE / "launch" / "m2_visual_scene.launch.py")

    assert "package='ros_gz_bridge'" in launch_text
    assert "executable='parameter_bridge'" in launch_text
    assert "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" in launch_text


def test_visual_scene_can_run_gazebo_server_only_for_smoke_tests():
    launch_text = read_text(SIM_PACKAGE / "launch" / "m2_visual_scene.launch.py")

    assert "DeclareLaunchArgument(" in launch_text
    assert "'headless'" in launch_text
    assert "default_value='false'" in launch_text
    assert "LaunchConfiguration('headless')" in launch_text
    assert "'-s -r '" in launch_text


def test_visual_scene_world_can_be_overridden_for_m8_shape_library():
    launch_text = read_text(SIM_PACKAGE / "launch" / "m2_visual_scene.launch.py")

    for required in [
        "world_file = LaunchConfiguration('world_file')",
        "world_name = LaunchConfiguration('world_name')",
        "PathJoinSubstitution([sim_share, 'worlds', world_file])",
        "'-world',",
        "world_name,",
        "DeclareLaunchArgument('world_file', default_value='m2_tabletop_visual.sdf')",
        "DeclareLaunchArgument('world_name', default_value='ghost_mgg_m2_visual')",
    ]:
        assert required in launch_text


def test_visual_scene_robot_uses_controlled_pregrasp_pose_near_targets():
    launch_text = read_text(SIM_PACKAGE / "launch" / "m2_visual_scene.launch.py")
    base_text = read_text(
        VENDORED_DESCRIPTION / "urdf" / "mech" / "g_shape_base_v2_0.urdf.xacro"
    )

    assert "use_gazebo:=true" in launch_text
    assert "'-x'," in launch_text
    assert "'-0.171207'," in launch_text
    assert "'-y'," in launch_text
    assert "'0.228790'," in launch_text
    assert "'-z'," in launch_text
    assert "'0.765000'," in launch_text
    assert "'-Y'," in launch_text
    assert "'-2.180640'," in launch_text
    red_cube_dx = 0.070172 - (-0.171207)
    red_cube_dy = 0.012156 - 0.228790
    red_cube_pregrasp_dz = (0.7525 + 0.095) - 0.765
    red_cube_pregrasp_range = (
        red_cube_dx**2 + red_cube_dy**2 + red_cube_pregrasp_dz**2
    ) ** 0.5
    assert red_cube_pregrasp_range < 0.34
    assert "joint_state_broadcaster" in launch_text
    assert "arm_controller" in launch_text
    assert "/arm_controller/follow_joint_trajectory" in launch_text
    assert "control_msgs/action/FollowJointTrajectory" in launch_text
    assert '<mass value="200.0"/>' in base_text
    assert 'ixx="0.333"' in base_text
    assert "<kinematic>true</kinematic>" in base_text
    assert "<mu1>10.0</mu1>" in base_text
    assert "<mu2>10.0</mu2>" in base_text
    for joint_name in [
        "link1_to_link2",
        "link2_to_link3",
        "link3_to_link4",
        "link4_to_link5",
        "link5_to_link6",
        "link6_to_link6_flange",
    ]:
        assert joint_name in launch_text


def test_visual_scene_objects_are_centimeter_scale_and_colored():
    world_text = read_text(SIM_PACKAGE / "worlds" / "m2_tabletop_visual.sdf")
    expected_world_poses = [
        "<pose>0.070172 0.012156 0.752500 0 0 0</pose>",
        "<pose>0.115000 0.075000 0.752500 0 0 0</pose>",
        "<pose>-0.035000 -0.047344 0.752500 0 0 0</pose>",
        "<pose>0.002000 0.100000 0.752500 0 0 0</pose>",
    ]

    for expected_pose in expected_world_poses:
        assert expected_pose in world_text

    red_text = read_text(SIM_PACKAGE / "models" / "red_cube" / "model.sdf")
    blue_text = read_text(SIM_PACKAGE / "models" / "blue_cylinder" / "model.sdf")
    green_text = read_text(SIM_PACKAGE / "models" / "green_cylinder" / "model.sdf")
    glass_text = read_text(SIM_PACKAGE / "models" / "glass_block" / "model.sdf")

    assert "<size>0.025 0.025 0.025</size>" in red_text
    assert "<radius>0.0125</radius>" in blue_text
    assert "<radius>0.0125</radius>" in green_text
    assert "<length>0.025</length>" in blue_text
    assert "<length>0.025</length>" in green_text
    assert "<transparency>0.55</transparency>" in glass_text


def test_square_visual_scene_targets_are_axis_aligned_for_m8_inspection():
    for world_name in ["m2_tabletop_visual.sdf", "m2_tabletop_tuning.sdf"]:
        world_text = read_text(SIM_PACKAGE / "worlds" / world_name)

        assert "<pose>0.070172 0.012156 0.752500 0 0 0</pose>" in world_text
        assert "<pose>0.002000 0.100000 0.752500 0 0 0</pose>" in world_text


def test_visual_scene_objects_are_dynamic_with_contact_friction_for_retention():
    object_model_paths = [
        SIM_PACKAGE / "models" / "red_cube" / "model.sdf",
        SIM_PACKAGE / "models" / "blue_cylinder" / "model.sdf",
        SIM_PACKAGE / "models" / "green_cylinder" / "model.sdf",
        SIM_PACKAGE / "models" / "glass_block" / "model.sdf",
    ]

    for model_path in object_model_paths:
        text = read_text(model_path)
        assert "<static>false</static>" in text
        assert "<inertial>" in text
        assert "<mass>0.020</mass>" in text
        assert "<surface>" in text
        assert "<friction>" in text
        assert "<ode>" in text
        assert "<mu>1.5</mu>" in text
        assert "<mu2>1.5</mu2>" in text
        assert "<contact>" in text
        assert "<kp>1000000.0</kp>" in text
        assert "<kd>10.0</kd>" in text


def test_visual_scene_camera_stand_is_tabletop_and_downward_facing():
    world_text = read_text(SIM_PACKAGE / "worlds" / "m2_tabletop_visual.sdf")
    stand_text = read_text(SIM_PACKAGE / "models" / "d435_stand_only" / "model.sdf")
    camera_text = read_text(SIM_PACKAGE / "models" / "d435_camera_head" / "model.sdf")

    assert "<shadows>false</shadows>" in world_text
    assert "<cast_shadows>false</cast_shadows>" in world_text
    assert "<direction>-0.55 0.18 -1.0</direction>" in world_text
    assert "<pose>0.410788 -0.079176 0.740000 0 0 -0.386880</pose>" in world_text
    assert "<pose>0.286775 -0.079936 1.105590 0 0.950000 2.754700</pose>" in world_text
    assert 'filename="gz-sim-physics-system"' in world_text
    assert "gz::sim::systems::Physics" in world_text
    assert 'filename="gz-sim-sensors-system"' in world_text
    assert "gz::sim::systems::Sensors" in world_text
    assert 'filename="gz-sim-user-commands-system"' in world_text
    assert "gz::sim::systems::UserCommands" in world_text
    assert 'filename="gz-sim-scene-broadcaster-system"' in world_text
    assert "gz::sim::systems::SceneBroadcaster" in world_text
    assert "tabletop_base_visual" in stand_text
    assert "short_pole_visual" in stand_text
    assert "short_support_arm_visual" in stand_text
    assert "<sensor " not in stand_text
    assert '<visual name="downward_camera_head_visual">' in camera_text
    assert "<pose>-0.020 0 0.025 0 0 0</pose>" in camera_text
    assert "<size>0.030 0.095 0.032</size>" in camera_text
    assert "lens_down_left_visual" in camera_text
    assert "lens_down_right_visual" in camera_text
    assert "<pose>-0.004 -0.026 0.000 0 1.5708 0</pose>" in camera_text
    assert "<pose>-0.004 0.026 0.000 0 1.5708 0</pose>" in camera_text
    assert '<sensor name="target_rgb_camera" type="camera">' in camera_text
    assert "<pose>0 0 0 0 0 0</pose>" in camera_text
    assert "<topic>ghost_mgg/d435/color/image_raw</topic>" in camera_text
    assert '<sensor name="target_depth_camera" type="depth_camera">' in camera_text
    assert "<topic>ghost_mgg/d435/depth/image_rect_raw</topic>" in camera_text
    assert "<format>R_FLOAT32</format>" in camera_text
    assert '<sensor name="target_infra1_camera" type="camera">' in camera_text
    assert '<sensor name="target_infra2_camera" type="camera">' in camera_text
    assert "<topic>ghost_mgg/d435/infra1/image_rect_raw</topic>" in camera_text
    assert "<topic>ghost_mgg/d435/infra2/image_rect_raw</topic>" in camera_text
    assert "<format>L8</format>" in camera_text
    assert "<horizontal_fov>0.72</horizontal_fov>" in camera_text
    assert "<width>640</width>" in camera_text
    assert "<height>480</height>" in camera_text


def test_visual_scene_world_includes_required_models_and_world_name():
    world_text = read_text(SIM_PACKAGE / "worlds" / "m2_tabletop_visual.sdf")

    for required in [
        "model://table",
        "model://d435_stand_only",
        "model://d435_camera_head",
        "model://red_cube",
        "model://blue_cylinder",
        "model://green_cylinder",
        "model://glass_block",
        "ghost_mgg_m2_visual",
    ]:
        assert required in world_text

    assert "model://d435_stand</uri>" not in world_text
    assert "model://primitive_objects" not in world_text


def test_m2_rviz_scene_launch_and_marker_node_are_self_contained():
    launch_text = read_text(SIM_PACKAGE / "launch" / "m2_rviz_scene.launch.py")
    visual_scene_launch_text = read_text(SIM_PACKAGE / "launch" / "m2_visual_scene.launch.py")
    marker_text = read_text(SIM_PACKAGE / "src" / "m2_scene_markers_node.cpp")

    for required in [
        "m2_visual_scene.launch.py",
        "m2_scene_markers_node",
        "rviz2",
        "m2_scene.rviz",
    ]:
        assert required in launch_text
    assert "ghost_mgg_mycobot_description" in visual_scene_launch_text

    for required in [
        "visualization_msgs::msg::MarkerArray",
        "table_frame",
        "base_link",
        "d435_color_optical_frame",
        "red_cube",
        "blue_cylinder",
        "green_cylinder",
        "glass_block",
        "camera_frustum",
    ]:
        assert required in marker_text


def test_m8_shape_library_world_contains_varied_generalization_targets():
    world_path = SIM_PACKAGE / "worlds" / "m8_shape_library.sdf"
    assert world_path.exists()
    world_text = read_text(world_path)

    for required in [
        '<world name="ghost_mgg_m8_shape_library">',
        "<name>m8_small_cube</name>",
        "<name>m8_medium_cube</name>",
        "<name>m8_large_cube</name>",
        "<name>m8_rect_box_2x1</name>",
        "<name>m8_rect_box_3x1</name>",
        "<name>m8_short_cylinder</name>",
        "<name>m8_tall_cylinder</name>",
        "<name>m8_tri_prism</name>",
        "<name>m8_hex_prism</name>",
        "m8_unknown_handle_target",
        "m8_unknown_thin_plate",
        "m8_unknown_concave_target",
        "m8_unknown_blob_target",
        "nurse_abstract_target",
    ]:
        assert required in world_text

    for model_name in [
        "m8_small_cube",
        "m8_medium_cube",
        "m8_large_cube",
        "m8_rect_box_2x1",
        "m8_rect_box_3x1",
        "m8_short_cylinder",
        "m8_tall_cylinder",
        "m8_tri_prism",
        "m8_hex_prism",
    ]:
        assert (SIM_PACKAGE / "models" / model_name / "model.sdf").exists()


def test_m2_scene_marker_node_uses_camera_body_frames_before_optical_frames():
    marker_text = read_text(SIM_PACKAGE / "src" / "m2_scene_markers_node.cpp")

    for required in [
        '"d435_color_frame"',
        '"d435_depth_frame"',
        '"d435_color_optical_frame"',
        '"d435_depth_optical_frame"',
        "-1.5708",
    ]:
        assert required in marker_text

    assert '"d435_mount_frame",\n      "d435_color_optical_frame"' not in marker_text
    assert '"d435_color_optical_frame",\n      "d435_depth_optical_frame"' not in marker_text


def test_m2_rviz_config_has_robot_model_tf_and_markers():
    rviz_text = read_text(SIM_PACKAGE / "rviz" / "m2_scene.rviz")

    for required in [
        "rviz_default_plugins/RobotModel",
        "rviz_default_plugins/TF",
        "rviz_default_plugins/MarkerArray",
        "/ghost_mgg/m2_scene_markers",
    ]:
        assert required in rviz_text


def test_m2_camera_inspect_rviz_config_has_image_and_point_cloud_displays():
    rviz_text = read_text(SIM_PACKAGE / "rviz" / "m2_camera_inspect.rviz")

    for required in [
        "rviz_default_plugins/Image",
        "rviz_default_plugins/PointCloud2",
        "rviz_default_plugins/MarkerArray",
        "Durability Policy: Transient Local",
        "/ghost_mgg/d435/color/image_raw",
        "/ghost_mgg/d435/depth/image_viz",
        "/ghost_mgg/d435/depth/points",
        "/ghost_mgg/hypothesis_markers",
    ]:
        assert required in rviz_text


def test_m2_hypothesis_marker_visualizer_is_installed():
    cmake_text = read_text(SIM_PACKAGE / "CMakeLists.txt")
    package_text = read_text(SIM_PACKAGE / "package.xml")

    assert "add_executable(hypothesis_markers_node src/hypothesis_markers_node.cpp)" in cmake_text
    assert "hypothesis_markers_node" in cmake_text
    assert "ghost_mgg_interfaces" in cmake_text
    assert "<depend>visualization_msgs</depend>" in package_text


def test_m2_depth_preview_visualizer_is_installed():
    cmake_text = read_text(SIM_PACKAGE / "CMakeLists.txt")

    for required in [
        "add_library(depth_to_mono8 src/depth_to_mono8.cpp)",
        "add_executable(depth_to_mono8_node src/depth_to_mono8_node.cpp)",
        "target_link_libraries(depth_to_mono8_node depth_to_mono8)",
        "depth_to_mono8_node",
    ]:
        assert required in cmake_text


def test_m2_camera_inspect_launch_starts_observation_cache_without_image_actions():
    launch_text = read_text(SIM_PACKAGE.parent / "ghost_mgg_bringup" / "launch" / "m2_gazebo_camera_inspect.launch.py")
    sim_cmake_text = read_text(SIM_PACKAGE / "CMakeLists.txt")
    sim_package_text = read_text(SIM_PACKAGE / "package.xml")

    for required in [
        "observation_cache_node",
        "/ghost_mgg/observations/latest",
        "/ghost_mgg/d435/color/image_raw",
        "/ghost_mgg/d435/depth/image_rect_raw",
        "/ghost_mgg/d435/color/camera_info",
        "/ghost_mgg/d435/depth/camera_info",
        "depth_to_mono8_node",
        "/ghost_mgg/d435/depth/image_viz",
    ]:
        assert required in launch_text

    assert "add_executable(observation_cache_node src/observation_cache_node.cpp)" in sim_cmake_text
    assert "<depend>ghost_mgg_interfaces</depend>" in sim_package_text
    assert "sensor_msgs/msg/Image rgb" not in read_text(REPO_ROOT / "src" / "ghost_mgg_interfaces" / "action" / "RecoverGeometry.action")


def test_m2_camera_inspect_rviz_defaults_depth_points_to_axis_heatmap():
    rviz_text = read_text(SIM_PACKAGE / "rviz" / "m2_camera_inspect.rviz")

    for required in [
        "Name: Depth Points",
        "Color Transformer: AxisColor",
        "Axis: Z",
        "Invert Rainbow: false",
        "Autocompute Value Bounds:",
        "Value: true",
    ]:
        assert required in rviz_text


def test_m3_failure_camera_inspect_rviz_exposes_failure_evidence_streams():
    rviz_text = read_text(SIM_PACKAGE / "rviz" / "m3_failure_camera_inspect.rviz")

    for required in [
        "Raw Depth Preview",
        "M3 Corrupted Depth Preview",
        "Target Mask",
        "Hole Evidence",
        "Table Leakage Evidence",
        "Edge Evidence",
        "Flying Point Evidence",
        "Biased Depth Evidence",
        "Raw Depth Points",
        "M3 Corrupted Depth Points",
        "IR Left Proxy",
        "IR Right Proxy",
        "/ghost_mgg/d435/depth/image_viz",
        "/ghost_mgg/d435/depth/m3_image_viz",
        "/ghost_mgg/d435/target_mask",
        "/ghost_mgg/d435/evidence/hole_mask",
        "/ghost_mgg/d435/evidence/table_leakage_mask",
        "/ghost_mgg/d435/evidence/edge_mask",
        "/ghost_mgg/d435/evidence/flying_point_mask",
        "/ghost_mgg/d435/evidence/biased_depth_mask",
        "/ghost_mgg/d435/depth/points",
        "/ghost_mgg/d435/depth/m3_points",
        "/ghost_mgg/d435/infra1/image_rect_raw",
        "/ghost_mgg/d435/infra2/image_rect_raw",
        "Color Transformer: AxisColor",
        "Axis: Z",
    ]:
        assert required in rviz_text


def test_pose_tuning_launch_starts_split_camera_assets_and_rviz():
    launch_text = read_text(SIM_PACKAGE / "launch" / "m2_pose_tuning.launch.py")

    for required in [
        "m2_tabletop_tuning.sdf",
        "rviz2",
        "m2_camera_inspect.rviz",
        "d435_stand_only",
        "d435_camera_head",
        "camera_x",
        "camera_y",
        "camera_z",
        "camera_roll",
        "camera_pitch",
        "camera_yaw",
        "stand_x",
        "stand_y",
        "stand_z",
        "stand_yaw",
        "/ghost_mgg/d435/depth/points",
        "depth_to_point_cloud_node",
        "depth_to_mono8_node",
        "/ghost_mgg/d435/depth/image_viz",
        "observation_cache_node",
        "default_value='-0.171207'",
        "default_value='0.228790'",
        "default_value='0.765000'",
        "default_value='-2.180640'",
        "default_value='0.286775'",
        "default_value='-0.079936'",
        "default_value='1.105590'",
    ]:
        assert required in launch_text


def test_pose_tuning_world_uses_independent_draggable_targets():
    world_text = read_text(SIM_PACKAGE / "worlds" / "m2_tabletop_tuning.sdf")

    assert "ghost_mgg_m2_tuning" in world_text
    assert "model://table" in world_text
    assert "model://d435_stand" not in world_text
    assert "model://d435_stand_only" not in world_text
    assert "model://d435_camera_head" not in world_text
    assert "model://primitive_objects" not in world_text

    for required in [
        "model://red_cube",
        "model://blue_cylinder",
        "model://green_cylinder",
        "model://glass_block",
    ]:
        assert required in world_text


def test_pose_tuning_target_models_are_independent_single_objects():
    expected_models = {
        "red_cube": ["red_cube_visual", "<size>0.025 0.025 0.025</size>"],
        "blue_cylinder": ["blue_cylinder_visual", "<radius>0.0125</radius>"],
        "green_cylinder": ["green_cylinder_visual", "<radius>0.0125</radius>"],
        "glass_block": [
            "glass_block_visual",
            "<size>0.025 0.025 0.025</size>",
            "<transparency>0.55</transparency>",
        ],
    }

    for model_name, required_fragments in expected_models.items():
        model_text = read_text(SIM_PACKAGE / "models" / model_name / "model.sdf")

        assert f'<model name="{model_name}">' in model_text
        assert "<static>false</static>" in model_text
        for required in required_fragments:
            assert required in model_text

        other_names = set(expected_models) - {model_name}
        for other_name in other_names:
            assert f"{other_name}_visual" not in model_text


def test_visual_world_includes_lightweight_nurse_abstract_target():
    world_text = read_text(SIM_PACKAGE / "worlds" / "m2_tabletop_visual.sdf")

    for required in [
        '<model name="nurse_abstract_target">',
        "<static>false</static>",
        "https://fuel.gazebosim.org/1.0/openrobotics/models/nurse/2/files/meshes/Nurse.obj",
        "https://fuel.gazebosim.org/1.0/openrobotics/models/nurse/2/files/meshes/Nurse_Col.obj",
        "<scale>0.12 0.12 0.12</scale>",
        "<mass>0.005</mass>",
        "<pose>0.029075 0.189698 0.769565 1.570800 -0.000001 1.027280</pose>",
    ]:
        assert required in world_text


def test_split_stand_model_has_no_camera_or_sensors():
    stand_text = read_text(SIM_PACKAGE / "models" / "d435_stand_only" / "model.sdf")

    for required in [
        '<model name="d435_stand_only">',
        "tabletop_base_visual",
        "short_pole_visual",
        "short_support_arm_visual",
    ]:
        assert required in stand_text

    for forbidden in [
        "<sensor ",
        "downward_camera_head_visual",
        "lens_down_left_visual",
        "target_rgb_camera",
        "target_depth_camera",
    ]:
        assert forbidden not in stand_text


def test_split_camera_head_model_has_sensors_but_no_stand_geometry():
    camera_text = read_text(SIM_PACKAGE / "models" / "d435_camera_head" / "model.sdf")

    for required in [
        '<model name="d435_camera_head">',
        "downward_camera_head_visual",
        "lens_down_left_visual",
        "lens_down_right_visual",
        '<sensor name="target_rgb_camera" type="camera">',
        "<topic>ghost_mgg/d435/color/image_raw</topic>",
        '<sensor name="target_depth_camera" type="depth_camera">',
        "<topic>ghost_mgg/d435/depth/image_rect_raw</topic>",
        "<horizontal_fov>0.72</horizontal_fov>",
        "<width>640</width>",
        "<height>480</height>",
    ]:
        assert required in camera_text

    for forbidden in [
        "tabletop_base_visual",
        "short_pole_visual",
        "short_support_arm_visual",
    ]:
        assert forbidden not in camera_text


def test_cmake_installs_assets_and_registers_pytest():
    cmake_text = read_text(SIM_PACKAGE / "CMakeLists.txt")

    assert "DIRECTORY config launch models worlds" in cmake_text
    assert "robot_description_publisher_node" in cmake_text
    assert "if(BUILD_TESTING)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert (
        "ament_add_pytest_test(test_m2_visual_scene_assets "
        "test/test_m2_visual_scene_assets.py)"
    ) in cmake_text.replace("\n", " ")


def test_package_declares_pytest_test_dependency():
    package_text = read_text(SIM_PACKAGE / "package.xml")

    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_text
    assert "<exec_depend>ros_gz_bridge</exec_depend>" in package_text
    assert "<exec_depend>rviz2</exec_depend>" in package_text
    assert "<depend>std_msgs</depend>" in package_text
