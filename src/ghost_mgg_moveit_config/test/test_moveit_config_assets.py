from pathlib import Path
import importlib.util
import xml.etree.ElementTree as ET

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
MOVEIT_PACKAGE = REPO_ROOT / "src" / "ghost_mgg_moveit_config"

ARM_JOINTS = [
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_link6",
    "link6_to_link6_flange",
]

SAFE_GROUP_STATES = {
    "home": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "pregrasp": [0.0, -0.85, 1.35, -0.85, 0.65, 0.0],
}


def read_text(relative_path):
    return (MOVEIT_PACKAGE / relative_path).read_text(encoding="utf-8")


def load_move_group_launch_module():
    launch_path = MOVEIT_PACKAGE / "launch" / "m2_move_group.launch.py"
    spec = importlib.util.spec_from_file_location("m2_move_group_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_m6_shadow_launch_module():
    launch_path = MOVEIT_PACKAGE / "launch" / "m6_shadow_move_group.launch.py"
    spec = importlib.util.spec_from_file_location("m6_shadow_move_group_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_srdf_declares_arm_group_joints_and_world_virtual_joint():
    srdf_path = MOVEIT_PACKAGE / "config" / "ghost_mgg_mycobot_280.srdf"
    assert srdf_path.exists()

    robot = ET.parse(srdf_path).getroot()
    assert robot.attrib["name"] == "mycobot_280"

    arm_group = robot.find("./group[@name='arm']")
    assert arm_group is not None

    group_joints = [joint.attrib["name"] for joint in arm_group.findall("joint")]
    assert group_joints == ARM_JOINTS

    virtual_joint = robot.find("./virtual_joint[@name='world_joint']")
    assert virtual_joint is not None
    assert virtual_joint.attrib["type"] == "fixed"
    assert virtual_joint.attrib["parent_frame"] == "world"
    assert virtual_joint.attrib["child_link"] == "base_link"


def test_moveit_controller_config_wires_arm_follow_joint_trajectory_controller():
    controller_config = yaml.safe_load(read_text("config/moveit_controllers.yaml"))

    assert (
        controller_config["moveit_controller_manager"]
        == "moveit_simple_controller_manager/MoveItSimpleControllerManager"
    )

    simple_controller_manager = controller_config["moveit_simple_controller_manager"]
    assert simple_controller_manager["controller_names"] == ["arm_controller"]

    arm_controller = simple_controller_manager["arm_controller"]
    assert arm_controller["type"] == "FollowJointTrajectory"
    assert arm_controller["action_ns"] == "follow_joint_trajectory"
    assert arm_controller["default"] is True
    assert arm_controller["joints"] == ARM_JOINTS


def test_joint_limits_define_acceleration_limits_for_time_parameterization():
    joint_limits = yaml.safe_load(read_text("config/joint_limits.yaml"))["joint_limits"]

    for joint_name in ARM_JOINTS:
        limits = joint_limits[joint_name]
        assert limits["has_velocity_limits"] is True
        assert limits["max_velocity"] > 0.0
        assert limits["has_acceleration_limits"] is True
        assert limits["max_acceleration"] > 0.0


def test_srdf_declares_safe_named_group_states_for_arm():
    srdf_path = MOVEIT_PACKAGE / "config" / "ghost_mgg_mycobot_280.srdf"
    robot = ET.parse(srdf_path).getroot()

    for state_name, expected_positions in SAFE_GROUP_STATES.items():
        group_state = robot.find(
            f"./group_state[@name='{state_name}'][@group='arm']"
        )
        assert group_state is not None

        state_joints = group_state.findall("joint")
        assert [joint.attrib["name"] for joint in state_joints] == ARM_JOINTS
        assert [float(joint.attrib["value"]) for joint in state_joints] == expected_positions


def test_srdf_allows_closed_gripper_fingertip_contact():
    srdf_text = read_text("config/ghost_mgg_mycobot_280.srdf")

    assert (
        '<disable_collisions link1="gripper_left1" link2="gripper_right1" '
        'reason="ClosedGripper"/>'
    ) in srdf_text


def test_move_group_and_rviz_launch_files_reference_expected_packages_and_nodes():
    move_group_launch = read_text("launch/m2_move_group.launch.py")
    rviz_launch = read_text("launch/m2_moveit_rviz.launch.py")

    for required in [
        "ghost_mgg_mycobot_description",
        "ghost_mgg_moveit_config",
        "move_group",
    ]:
        assert required in move_group_launch

    assert "rviz2" in rviz_launch
    assert "name='m2_moveit_rviz'" in rviz_launch
    assert "name='rviz2'" not in rviz_launch
    assert "m2_moveit.rviz" in rviz_launch
    assert "'use_sim_time': True" in move_group_launch
    assert "'use_sim_time': True" in rviz_launch


def test_rviz_config_contains_motion_planning_robot_model_and_tf_displays():
    rviz_text = read_text("rviz/m2_moveit.rviz")

    for required in [
        "MotionPlanning",
        "RobotModel",
        "TF",
    ]:
        assert required in rviz_text


def test_moveit_builder_exposes_joint_limits_at_effective_parameter_key():
    move_group_launch = load_move_group_launch_module()
    moveit_dict = move_group_launch.build_m2_moveit_config().to_dict()

    planning = moveit_dict["robot_description_planning"]
    assert "joint_limits" in planning
    assert "robot_description_planning" not in planning
    assert "link1_to_link2" in planning["joint_limits"]


def test_m6_shadow_move_group_launch_is_plan_only_and_uses_shadow_joint_states():
    launch_text = read_text("launch/m6_shadow_move_group.launch.py")

    for required in [
        "joint_state_publisher",
        "robot_state_publisher",
        "move_group",
        "'allow_trajectory_execution': False",
        "'moveit_manage_controllers': False",
        "'publish_robot_description': True",
        "'publish_robot_description_semantic': True",
        "'use_sim_time': False",
        "'use_gazebo': 'false'",
        "'use_camera': 'false'",
        "m6_shadow_moveit_rviz",
        "DeclareLaunchArgument('show_rviz', default_value='false')",
        "DeclareLaunchArgument('use_fake_joint_states', default_value='true')",
        "condition=IfCondition(use_fake_joint_states)",
    ]:
        assert required in launch_text

    for forbidden in [
        "controller_manager",
        "ros2_control_node",
        "moveit_controllers.yaml",
        "FollowJointTrajectory",
        "moveit_sim_execute_server",
        "execute_m4_joint_hypothesis_action.py",
    ]:
        assert forbidden not in launch_text


def test_m6_shadow_moveit_config_omits_execution_controller_manager():
    shadow_launch = load_m6_shadow_launch_module()
    moveit_dict = shadow_launch.build_m6_shadow_moveit_config().to_dict()

    assert "moveit_controller_manager" not in moveit_dict
    assert "moveit_simple_controller_manager" not in moveit_dict
    assert moveit_dict["robot_description_planning"]["joint_limits"].keys() >= set(ARM_JOINTS)
