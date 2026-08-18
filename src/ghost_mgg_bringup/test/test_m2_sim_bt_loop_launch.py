from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]


def test_m2_sim_bt_loop_launch_wires_visual_scene_backend_executor_and_bt():
    launch_path = BRINGUP_DIR / "launch" / "m2_sim_bt_loop.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text()

    assert "m2_visual_scene.launch.py" in launch_text
    assert "dummy_recovery_server" in launch_text
    assert "mycobot_sim_execute_server" in launch_text
    assert "'trajectory_action_name': '/arm_controller/follow_joint_trajectory'" in launch_text
    assert "'trajectory_server_timeout_sec': 6.0" in launch_text
    assert "bt_runner_node" in launch_text
    assert "m2_sim_closed_loop.xml" in launch_text
    assert "'backend_name': 'dummy'" in launch_text
    assert "'executor_name': 'mycobot_sim'" in launch_text
    assert "'target_label': 'm2_sim_target'" in launch_text
    assert "'recover_timeout_sec': 2.0" in launch_text
    assert "'execute_timeout_sec': 8.0" in launch_text
