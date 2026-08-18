from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_DIR = REPO_ROOT / "src" / "ghost_mgg_real"


def test_real_package_declares_realsense_runtime_assets():
    package_xml = (REAL_DIR / "package.xml").read_text(encoding="utf-8")
    cmake_text = (REAL_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "<exec_depend>realsense2_camera</exec_depend>" in package_xml
    assert "<exec_depend>rviz2</exec_depend>" in package_xml
    assert "install(" in cmake_text
    assert "launch rviz scripts" in cmake_text


def test_d435_launch_matches_ros2_realsense_argument_contract():
    launch_text = (REAL_DIR / "launch" / "d435_realsense.launch.py").read_text(
        encoding="utf-8"
    )

    for required in [
        "realsense2_camera",
        "rs_launch.py",
        "depth_module.depth_profile",
        "rgb_camera.color_profile",
        "depth_module.infra_profile",
        "enable_infra1",
        "enable_infra2",
        "pointcloud.enable",
        "align_depth.enable",
        "initial_reset",
        "DeclareLaunchArgument('depth_profile', default_value='1280x720x30')",
        "DeclareLaunchArgument('color_profile', default_value='1280x720x30')",
        "DeclareLaunchArgument('infra_profile', default_value='1280x720x30')",
    ]:
        assert required in launch_text


def test_d435_inspect_launch_starts_camera_and_rviz_together():
    launch_text = (
        REAL_DIR / "launch" / "d435_realsense_inspect.launch.py"
    ).read_text(encoding="utf-8")

    for required in [
        "d435_realsense.launch.py",
        "rviz2",
        "d435_realsense.rviz",
        "enable_pointcloud': 'true'",
        "enable_align_depth': 'true'",
        "enable_infra': 'true'",
    ]:
        assert required in launch_text


def test_d435_rviz_uses_real_realsense_topics_and_camera_link_frame():
    rviz_text = (REAL_DIR / "rviz" / "d435_realsense.rviz").read_text(
        encoding="utf-8"
    )

    for required in [
        "Fixed Frame: camera_link",
        "/camera/camera/depth/color/points",
        "/camera/camera/color/image_raw",
        "/camera/camera/depth/image_rect_raw",
        "/camera/camera/infra1/image_rect_raw",
        "/camera/camera/infra2/image_rect_raw",
    ]:
        assert required in rviz_text


def test_d435_smoke_checks_usb3_topics_encodings_and_point_cloud_frame():
    smoke_text = (REAL_DIR / "scripts" / "smoke_d435_realsense.sh").read_text(
        encoding="utf-8"
    )

    for required in [
        "Device USB type: 3.",
        "RealSense Node Is Up!",
        "/camera/camera/depth/image_rect_raw encoding 16UC1",
        "/camera/camera/color/image_raw encoding rgb8",
        "/camera/camera/infra1/image_rect_raw encoding mono8",
        "/camera/camera/depth/color/points header.frame_id camera_depth_optical_frame",
    ]:
        assert required in smoke_text
