import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "m7_mycobot_control_server.py"
)


def load_control_module():
    spec = importlib.util.spec_from_file_location("m7_mycobot_control_server", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_remote_control_script_owns_one_mycobot_and_supports_state_and_commands():
    module = load_control_module()

    remote_script = module.build_remote_control_script(
        serial_port="/dev/ttyAMA0",
        baud=1000000,
        sample_hz=4.0,
    )

    assert "mc = MyCobot280('/dev/ttyAMA0', 1000000)" in remote_script
    assert "read_state" in remote_script
    assert "go_home" in remote_script
    assert "guarded_taught_grasp" in remote_script
    assert "send_angles" in remote_script
    assert "send_coords" in remote_script
    assert "set_gripper_state" in remote_script
    assert remote_script.count("MyCobot280(") == 1
    for forbidden in [
        "release_all_servos",
        "power_off",
        "power_on",
        "set_basic_output",
        "set_encoder",
        "set_encoders",
    ]:
        assert forbidden not in remote_script


def test_parse_remote_control_line_separates_state_and_response():
    module = load_control_module()

    state = module.parse_remote_control_line(
        '{"type":"state","angles":[0,1,2,3,4,5],"coords":[1,2,3,4,5,6],"gripper":97}'
    )
    response = module.parse_remote_control_line(
        '{"type":"response","id":"abc","ok":true,"status":"executed_home","result":{"x":1}}'
    )

    assert state["type"] == "state"
    assert state["angles"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert state["coords"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert state["gripper"] == 97
    assert response["type"] == "response"
    assert response["id"] == "abc"
    assert response["ok"] is True


def test_build_home_command_requires_confirmation_for_real_execution():
    module = load_control_module()

    command = module.build_home_command(
        request_id="home-1",
        execute=True,
        operator_phrase=module.REQUIRED_OPERATOR_PHRASE,
        target_angles_deg=[0, 30, -70, 0, 0, 0],
        speed=3,
        max_delta_deg=90,
        open_gripper=True,
        gripper_speed=20,
        staged_home=True,
    )

    assert command["id"] == "home-1"
    assert command["command"] == "go_home"
    assert command["execute"] is True
    assert command["operator_phrase"] == module.REQUIRED_OPERATOR_PHRASE
    assert command["target_angles_deg"] == [0.0, 30.0, -70.0, 0.0, 0.0, 0.0]
    assert command["open_gripper"] is True
    assert command["gripper_speed"] == 20.0
    assert command["staged_home"] is True


def test_standard_home_joint_and_home_script_open_gripper():
    module = load_control_module()
    home_script = (REPO_ROOT / "scripts" / "home_m7_real.sh").read_text()
    remote_script = module.build_remote_control_script(
        serial_port="/dev/ttyAMA0",
        baud=1000000,
        sample_hz=4.0,
    )

    assert module.STANDARD_TOP_GRASP_HOME_JOINT_DEG == [
        0.0,
        30.0,
        -70.0,
        0.0,
        0.0,
        0.0,
    ]
    assert "target_angles_deg: [0.0, 30.0, -70.0, 0.0, 0.0, 0.0]" in home_script
    assert "staged_home: true" in home_script
    assert "open_gripper: true" in home_script
    assert "gripper_speed: 20.0" in home_script
    assert "mc.set_gripper_state(0, gripper_speed)" in remote_script
    assert "build_joint_stages" in remote_script
    assert "executed_staged_home" in remote_script


def test_guarded_grasp_command_validates_min_z_and_speed():
    module = load_control_module()

    command = module.build_guarded_taught_grasp_command(
        request_id="grasp-1",
        execute=True,
        operator_phrase=module.REQUIRED_OPERATOR_PHRASE,
        pregrasp_coords_mm_deg=[241.5, 42.6, 140.0, 174.61, -16.86, -60.89],
        grasp_coords_mm_deg=[241.5, 42.6, 126.4, 174.61, -16.86, -60.89],
        lift_coords_mm_deg=[241.5, 42.6, 161.4, 174.61, -16.86, -60.89],
        speed=3,
        gripper_speed=20,
        min_z_mm=120.0,
    )

    assert command["command"] == "guarded_taught_grasp"
    assert command["execute"] is True
    assert command["pregrasp_coords_mm_deg"][2] == 140.0
    assert command["grasp_coords_mm_deg"][2] == 126.4
    assert command["lift_coords_mm_deg"][2] == 161.4


def test_launch_and_scripts_use_control_server_not_ssh_motion_scripts():
    launch_text = (
        REPO_ROOT / "src" / "ghost_mgg_real" / "launch" / "m7_real_control_inspect.launch.py"
    ).read_text()
    control_script = (REPO_ROOT / "scripts" / "run_m7_real_control_inspect.sh").read_text()
    grasp_script = (REPO_ROOT / "scripts" / "grasp_m7_real_once.sh").read_text()
    cmake_text = (REPO_ROOT / "src" / "ghost_mgg_real" / "CMakeLists.txt").read_text()

    assert "m7_mycobot_control_server.py" in launch_text
    assert "m7_real_grasp_marker_node.py" in launch_text
    assert "d435_realsense.launch.py" in launch_text
    assert "m6_shadow_move_group.launch.py" in launch_text
    assert "m7_real_control.rviz" in launch_text
    assert "m7_real_control_inspect.launch.py" in control_script
    assert "ros2 service call /ghost_mgg/mycobot/guarded_taught_grasp" in grasp_script
    assert "ssh" not in grasp_script
    assert "m7_mycobot_control_server.py" in cmake_text
