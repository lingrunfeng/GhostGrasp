from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DESCRIPTION_PACKAGE = REPO_ROOT / "src" / "ghost_mgg_mycobot_description"

ARM_JOINTS = [
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_link6",
    "link6_to_link6_flange",
]

GRIPPER_JOINTS = [
    "gripper_controller",
    "gripper_base_to_gripper_left2",
    "gripper_left3_to_gripper_left1",
    "gripper_base_to_gripper_right3",
    "gripper_base_to_gripper_right2",
    "gripper_right3_to_gripper_right1",
]


def run_gazebo_control_xacro():
    xacro_path = DESCRIPTION_PACKAGE / "urdf" / "robots" / "mycobot_280.urdf.xacro"

    result = subprocess.run(
        [
            "xacro",
            str(xacro_path),
            "add_world:=false",
            "use_camera:=false",
            "use_gazebo:=true",
            "use_gripper:=true",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_xacro_generates_gazebo_ros2_control_contract():
    robot_description = run_gazebo_control_xacro()
    assert "<mimic" not in robot_description

    for required in [
        '<ros2_control name="RobotSystem" type="system">',
        "gz_ros2_control/GazeboSimSystem",
        "gz_ros2_control::GazeboSimROS2ControlPlugin",
        "ros2_controllers.yaml",
    ]:
        assert required in robot_description

    robot = ET.fromstring(robot_description)
    ros2_control = robot.find("./ros2_control[@name='RobotSystem']")
    assert ros2_control is not None
    assert ros2_control.attrib["type"] == "system"

    control_joints = {
        joint.attrib["name"]: joint for joint in ros2_control.findall("./joint")
    }
    assert set(ARM_JOINTS).issubset(control_joints)
    assert set(GRIPPER_JOINTS).issubset(control_joints)

    for joint_name in ARM_JOINTS:
        control_joint = control_joints[joint_name]
        command_interfaces = [
            interface.attrib["name"]
            for interface in control_joint.findall("./command_interface")
        ]
        state_interfaces = [
            interface.attrib["name"]
            for interface in control_joint.findall("./state_interface")
        ]
        assert command_interfaces == ["position"]
        assert state_interfaces == ["position"]

    for joint_name in GRIPPER_JOINTS:
        control_joint = control_joints[joint_name]
        command_interfaces = [
            interface.attrib["name"]
            for interface in control_joint.findall("./command_interface")
        ]
        state_interfaces = [
            interface.attrib["name"]
            for interface in control_joint.findall("./state_interface")
        ]
        assert command_interfaces == ["position"]
        assert "position" in state_interfaces


def test_ros2_controller_yaml_declares_arm_position_controller():
    controller_path = (
        DESCRIPTION_PACKAGE / "config" / "mycobot_280" / "ros2_controllers.yaml"
    )
    controller_config = yaml.safe_load(controller_path.read_text(encoding="utf-8"))

    controller_manager = controller_config["controller_manager"]["ros__parameters"]
    assert (
        controller_manager["joint_state_broadcaster"]["type"]
        == "joint_state_broadcaster/JointStateBroadcaster"
    )
    assert (
        controller_manager["arm_controller"]["type"]
        == "joint_trajectory_controller/JointTrajectoryController"
    )
    assert (
        controller_manager["gripper_group_controller"]["type"]
        == "joint_trajectory_controller/JointTrajectoryController"
    )

    arm_controller = controller_config["arm_controller"]["ros__parameters"]
    assert arm_controller["joints"] == ARM_JOINTS
    assert arm_controller["command_interfaces"] == ["position"]
    assert arm_controller["state_interfaces"] == ["position"]

    gripper_controller = controller_config["gripper_group_controller"]["ros__parameters"]
    assert gripper_controller["joints"] == GRIPPER_JOINTS
    assert gripper_controller["command_interfaces"] == ["position"]
    assert gripper_controller["state_interfaces"] == ["position"]
