import importlib.util
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "m6_ssh_joint_state_bridge.py"
)


def load_bridge_module():
    spec = importlib.util.spec_from_file_location("m6_ssh_joint_state_bridge", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_remote_angle_line_accepts_json_object():
    module = load_bridge_module()

    angles = module.parse_remote_angle_line('{"angles": [-21.26, 1.14, 1.31, -144.66, 70.4, -1.05]}')

    assert angles == [-21.26, 1.14, 1.31, -144.66, 70.4, -1.05]


def test_parse_remote_angle_line_accepts_plain_list():
    module = load_bridge_module()

    angles = module.parse_remote_angle_line("[-21.26, 1.14, 1.31, -144.66, 70.4, -1.05]")

    assert angles == [-21.26, 1.14, 1.31, -144.66, 70.4, -1.05]


def test_parse_remote_angle_line_rejects_bad_samples():
    module = load_bridge_module()

    assert module.parse_remote_angle_line("current pymycobot library version: 4.0.2") is None
    assert module.parse_remote_angle_line('{"angles": [1, 2, 3]}') is None
    assert module.parse_remote_angle_line('{"angles": [1, 2, 3, 4, 5, "bad"]}') is None


def test_degrees_to_radians_maps_vendor_to_ghost_mgg_joint_names():
    module = load_bridge_module()

    names, positions = module.degrees_to_ghost_joint_state([0, 90, -90, 180, -180, 45])

    assert names == [
        "link1_to_link2",
        "link2_to_link3",
        "link3_to_link4",
        "link4_to_link5",
        "link5_to_link6",
        "link6_to_link6_flange",
    ]
    assert positions == [
        0.0,
        math.pi / 2.0,
        -math.pi / 2.0,
        math.pi,
        -math.pi,
        math.pi / 4.0,
    ]


def test_append_shadow_gripper_state_adds_open_gripper_defaults():
    module = load_bridge_module()

    names, positions = module.degrees_to_ghost_joint_state([0, 0, 0, 0, 0, 0])
    names, positions = module.append_shadow_gripper_state(names, positions, gripper_position=0.15)

    assert names == [
        "link1_to_link2",
        "link2_to_link3",
        "link3_to_link4",
        "link4_to_link5",
        "link5_to_link6",
        "link6_to_link6_flange",
        "gripper_controller",
        "gripper_base_to_gripper_left2",
        "gripper_left3_to_gripper_left1",
        "gripper_base_to_gripper_right3",
        "gripper_base_to_gripper_right2",
        "gripper_right3_to_gripper_right1",
    ]
    assert positions[:6] == [0.0] * 6
    assert positions[6:] == [0.15, 0.15, -0.15, -0.15, -0.15, 0.15]


def test_remote_reader_script_is_read_only():
    module = load_bridge_module()

    remote_script = module.build_remote_reader_script(
        serial_port="/dev/ttyAMA0",
        baud=1000000,
        sample_hz=4.0,
    )

    assert "get_angles" in remote_script
    assert "/dev/ttyAMA0" in remote_script
    assert "1000000" in remote_script
    forbidden_tokens = [
        "send_angles",
        "send_coords",
        "set_gripper",
        "set_encoder",
        "set_encoders",
        "set_basic_output",
        "release_all_servos",
        "power_on",
        "power_off",
    ]
    for token in forbidden_tokens:
        assert token not in remote_script


def test_ssh_command_targets_robot_without_embedding_password():
    module = load_bridge_module()

    command = module.build_ssh_command(
        host="10.42.0.169",
        user="elephant",
        remote_script="print('ok')",
        connect_timeout_s=4,
    )

    assert command[:2] == ["ssh", "-o"]
    assert "elephant@10.42.0.169" in command
    assert "python3 -u -" in " ".join(command)
    assert "trunk" not in " ".join(command)


def test_launch_file_uses_safe_m6_defaults():
    launch_text = (
        REPO_ROOT
        / "src"
        / "ghost_mgg_real"
        / "launch"
        / "m6_mycobot_state_bridge.launch.py"
    ).read_text()

    assert "m6_ssh_joint_state_bridge.py" in launch_text
    assert "10.42.0.169" in launch_text
    assert "elephant" in launch_text
    assert "/dev/ttyAMA0" in launch_text
    assert "1000000" in launch_text
    assert "publish_shadow_gripper_joints" in launch_text


def test_package_installs_bridge_executable_and_pytest():
    cmake_text = (REPO_ROOT / "src" / "ghost_mgg_real" / "CMakeLists.txt").read_text()
    package_text = (REPO_ROOT / "src" / "ghost_mgg_real" / "package.xml").read_text()

    assert "scripts/m6_ssh_joint_state_bridge.py" in cmake_text
    assert "test_m6_mycobot_state_bridge" in cmake_text
    assert "<exec_depend>rclpy</exec_depend>" in package_text


def test_safe_rclpy_shutdown_ignores_already_shutdown_context():
    module = load_bridge_module()

    class AlreadyShutdownRclpy:
        class _Rclpy:
            class RCLError(Exception):
                pass

        _rclpy_pybind11 = _Rclpy()

        def shutdown(self):
            raise self._rclpy_pybind11.RCLError(
                "failed to shutdown: rcl_shutdown already called on the given context"
            )

    module.safe_rclpy_shutdown(AlreadyShutdownRclpy())


def test_spin_bridge_until_shutdown_treats_keyboard_interrupt_as_clean_exit():
    module = load_bridge_module()

    class FakeBridge:
        def __init__(self):
            self.shutdown_called = False

        def spin(self):
            raise KeyboardInterrupt

        def shutdown(self):
            self.shutdown_called = True

    bridge = FakeBridge()
    module.spin_bridge_until_shutdown(bridge)

    assert bridge.shutdown_called is True


def test_spin_bridge_until_shutdown_treats_shutdown_keyboard_interrupt_as_clean_exit():
    module = load_bridge_module()

    class FakeBridge:
        def __init__(self):
            self.spin_called = False

        def spin(self):
            self.spin_called = True

        def shutdown(self):
            raise KeyboardInterrupt

    bridge = FakeBridge()
    module.spin_bridge_until_shutdown(bridge)

    assert bridge.spin_called is True


def test_real_state_moveit_shadow_smoke_uses_bridge_and_disables_fake_joint_states():
    script_text = (
        REPO_ROOT / "scripts" / "smoke_m6_real_state_moveit_shadow_plan_only.sh"
    ).read_text()

    assert "m6_mycobot_state_bridge.launch.py" in script_text
    assert "m6_shadow_move_group.launch.py" in script_text
    assert "use_fake_joint_states:=false" in script_text
    assert "probe_m2_moveit_plan.py" in script_text
    assert "ros2 action send_goal" not in script_text
    assert "ExecuteGrasp" not in script_text
