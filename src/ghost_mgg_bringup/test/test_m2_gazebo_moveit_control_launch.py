from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m2_gazebo_moveit_control_launch_combines_scene_and_delayed_moveit_rviz():
    launch_path = BRINGUP_DIR / "launch" / "m2_gazebo_moveit_control.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text(encoding="utf-8")

    for required in [
        "ghost_mgg_sim",
        "m2_visual_scene.launch.py",
        "ghost_mgg_moveit_config",
        "m2_moveit_rviz.launch.py",
        "package='tf2_ros'",
        "static_transform_publisher",
        "'--frame-id', 'world'",
        "'--child-frame-id', 'base_link'",
        "'--x', '-0.171207'",
        "'--y', '0.228790'",
        "'--z', '0.765000'",
        "'--yaw', '-2.180640'",
        "TimerAction(period=13.0",
    ]:
        assert required in launch_text

    for forbidden in [
        "m2_mask_extrusion_bt_loop.launch.py",
        "bt_runner_node",
        "dummy_recovery_server",
        "mask_extrusion_recovery_server",
    ]:
        assert forbidden not in launch_text


def test_m2_gazebo_moveit_control_smoke_checks_clock_and_end_effector_tf():
    smoke_text = (
        REPO_ROOT / "scripts" / "smoke_m2_gazebo_moveit_control.sh"
    ).read_text(encoding="utf-8")

    assert "ros2 daemon stop" in smoke_text
    assert (
        "ros2 topic echo --no-daemon /clock "
        "rosgraph_msgs/msg/Clock --once"
    ) in smoke_text
    assert (
        "ros2 topic echo --no-daemon /joint_states "
        "sensor_msgs/msg/JointState --once"
    ) in smoke_text
    assert "ros2 control list_controllers --spin-time 2 -c /controller_manager" in smoke_text
    assert "ros2 node list --no-daemon --spin-time 2" in smoke_text
    assert "ros2 run tf2_ros tf2_echo world link6_flange" in smoke_text
    assert "ros2 run tf2_ros tf2_echo world gripper_base" in smoke_text
    assert "probe_m2_moveit_plan.py" in smoke_text
    assert "error_code=1" in smoke_text
