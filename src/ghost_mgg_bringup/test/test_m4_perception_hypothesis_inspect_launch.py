from pathlib import Path


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_DIR = REPO_ROOT / "src" / "ghost_mgg_sim"


def test_m4_perception_hypothesis_inspect_launch_exists_and_starts_publisher():
    launch_path = BRINGUP_DIR / "launch" / "m4_perception_hypothesis_inspect.launch.py"
    assert launch_path.exists()

    launch_text = launch_path.read_text(encoding="utf-8")
    for required in [
        "m4_offline_ranking_inspect.launch.py",
        "m4_perception_hypothesis_publisher_node.py",
        "scene_id",
        "joint_hypothesis_topic",
        "/ghost_mgg/m4_perception_hypotheses",
        "ghost_mgg_m4_perception",
        "'joint_report_path': joint_hypothesis_report_path",
        "'metric_proxy_report_path': metric_proxy_report_path",
        "'graspability_report_path': graspability_report_path",
        "'hypothesis_topic': perception_hypothesis_topic",
    ]:
        assert required in launch_text


def test_m4_perception_hypothesis_node_is_installed_and_registered_for_testing():
    cmake_text = (SIM_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    bringup_cmake_text = (BRINGUP_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "scripts/m4_perception_hypothesis_publisher_node.py" in cmake_text
    assert (
        "ament_add_pytest_test(test_m4_perception_hypothesis_publisher "
        "test/test_m4_perception_hypothesis_publisher.py)"
    ) in cmake_text
    assert (
        "ament_add_pytest_test(test_m4_perception_hypothesis_inspect_launch "
        "test/test_m4_perception_hypothesis_inspect_launch.py)"
    ) in bringup_cmake_text
