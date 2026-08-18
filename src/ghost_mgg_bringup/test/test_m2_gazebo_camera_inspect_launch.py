from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m2_gazebo_camera_inspect_launch_bridges_camera_topics_and_starts_rviz():
    launch_path = BRINGUP_DIR / "launch" / "m2_gazebo_camera_inspect.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text(encoding="utf-8")

    for required in [
        "m2_visual_scene.launch.py",
        "ros_gz_bridge",
        "parameter_bridge",
        "/ghost_mgg/d435/color/image_raw",
        "/ghost_mgg/d435/depth/image_rect_raw",
        "/ghost_mgg/d435/color/camera_info",
        "/ghost_mgg/d435/depth/camera_info",
        "d435_color_optical_frame",
        "d435_depth_optical_frame",
        "m2_world_to_base_link_tf",
        "'--child-frame-id', 'base_link'",
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
        "robot_description_publisher_node",
        "depth_to_point_cloud_node",
        "depth_to_mono8_node",
        "/ghost_mgg/d435/depth/image_viz",
        "m2_camera_inspect.rviz",
    ]:
        assert required in launch_text


def test_m2_camera_inspect_separates_sensor_body_and_ros_optical_frames():
    launch_path = BRINGUP_DIR / "launch" / "m2_gazebo_camera_inspect.launch.py"
    launch_text = launch_path.read_text(encoding="utf-8")

    for required in [
        "m2_d435_link_to_color_frame_tf",
        "m2_d435_link_to_depth_frame_tf",
        "m2_d435_color_frame_to_optical_tf",
        "m2_d435_depth_frame_to_optical_tf",
        "'--child-frame-id', 'd435_color_frame'",
        "'--child-frame-id', 'd435_depth_frame'",
        "'--frame-id', 'd435_color_frame'",
        "'--child-frame-id', 'd435_color_optical_frame'",
        "'--frame-id', 'd435_depth_frame'",
        "'--child-frame-id', 'd435_depth_optical_frame'",
    ]:
        assert required in launch_text

    for required_identity in [
        "'--x', '0.0'",
        "'--y', '0.0'",
        "'--z', '0.0'",
        "'--roll', '0.0'",
        "'--pitch', '0.0'",
        "'--yaw', '0.0'",
    ]:
        assert required_identity in launch_text

    assert "m2_d435_link_to_color_optical_tf" not in launch_text
    assert "m2_d435_link_to_depth_optical_tf" not in launch_text


def test_m2_camera_inspect_smoke_checks_sensor_contract():
    smoke_text = (
        REPO_ROOT / "scripts" / "smoke_m2_gazebo_camera_inspect.sh"
    ).read_text(encoding="utf-8")

    assert "ros2 daemon stop" in smoke_text
    assert (
        "ros2 topic echo --no-daemon /ghost_mgg/d435/color/image_raw "
        "sensor_msgs/msg/Image --once"
    ) in smoke_text
    assert (
        "ros2 topic echo --no-daemon --qos-reliability reliable "
        "--qos-durability transient_local --qos-depth 1 --once --field data "
        "/robot_description std_msgs/msg/String"
    ) in smoke_text
    assert (
        "ros2 topic echo --no-daemon /ghost_mgg/d435/depth/image_rect_raw "
        "sensor_msgs/msg/Image --once"
    ) in smoke_text
    assert (
        "ros2 topic echo --no-daemon /ghost_mgg/d435/color/camera_info "
        "sensor_msgs/msg/CameraInfo --once"
    ) in smoke_text
    assert (
        "ros2 topic echo --no-daemon /ghost_mgg/d435/depth/camera_info "
        "sensor_msgs/msg/CameraInfo --once"
    ) in smoke_text
    assert (
        "ros2 topic echo --no-daemon /ghost_mgg/d435/depth/points "
        "sensor_msgs/msg/PointCloud2 --once"
    ) in smoke_text
    assert (
        "ros2 topic echo --no-daemon /ghost_mgg/d435/depth/image_viz "
        "sensor_msgs/msg/Image --once"
    ) in smoke_text
    assert "tf2_echo world base_link" in smoke_text
    assert "tf2_echo world d435_color_optical_frame" in smoke_text
    assert "tf2_echo world d435_depth_optical_frame" in smoke_text
