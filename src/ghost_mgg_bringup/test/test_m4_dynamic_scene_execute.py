from pathlib import Path
import importlib.util


BRINGUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m4_dynamic_scene_execute_launch_uses_scene_snapshot_targets():
    launch_path = BRINGUP_DIR / "launch" / "m4_dynamic_scene_execute.launch.py"
    assert launch_path.exists()
    launch_text = launch_path.read_text(encoding="utf-8")

    for required in [
        "m4_joint_hypothesis_moveit_execute.launch.py",
        "LaunchConfiguration('headless')",
        "LaunchConfiguration('show_rviz')",
        "dynamic_targets_path",
        "reports/m4_scene_snapshot/current_targets.json",
        "'sim_grasp_targets_path': dynamic_targets_path",
        "DeclareLaunchArgument('headless', default_value='true')",
        "DeclareLaunchArgument('show_rviz', default_value='false')",
    ]:
        assert required in launch_text


def test_refresh_m4_scene_targets_reads_gazebo_model_poses():
    script_path = REPO_ROOT / "scripts" / "refresh_m4_scene_targets.py"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "gz model -m",
        "parse_gz_model_pose",
        "refresh_targets_from_gazebo",
        "m4_sim_grasp_targets_v1",
        "red_cube",
        "blue_cylinder",
        "green_cylinder",
        "glass_block",
        "reports/m4_scene_snapshot/current_targets.json",
        "center_x_m",
        "center_y_m",
        "center_z_m",
        "yaw_rad",
    ]:
        assert required in source


def test_refresh_m4_scene_targets_parses_gazebo_pose_and_updates_rows(monkeypatch):
    script_path = REPO_ROOT / "scripts" / "refresh_m4_scene_targets.py"
    spec = importlib.util.spec_from_file_location("refresh_m4_scene_targets", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    sample_output = """
Requesting state for world [ghost_mgg_m2_visual]...

Model: [81]
  - Name: red_cube
  - Pose [ XYZ (m) ] [ RPY (rad) ]:
    [0.123000 -0.045000 0.752500]
    [0.000000 0.000000 1.570000]
"""
    xyz, rpy = module.parse_gz_model_pose(sample_output)
    assert xyz == (0.123, -0.045, 0.7525)
    assert rpy == (0.0, 0.0, 1.57)

    def fake_query(model_name, timeout_sec):
        assert model_name == "red_cube"
        assert timeout_sec == 4.0
        return xyz, rpy

    monkeypatch.setattr(module, "query_gazebo_model_pose", fake_query)
    refreshed = module.refresh_targets_from_gazebo(
        {
            "schema_version": "m4_sim_grasp_targets_v1",
            "rows": [
                {
                    "target_id": "red_cube",
                    "shape_type": "box",
                    "center_x_m": 0.0,
                    "center_y_m": 0.0,
                    "center_z_m": 0.0,
                    "yaw_rad": 0.0,
                }
            ],
        },
        timeout_sec=4.0,
        allow_missing=False,
    )
    row = refreshed["rows"][0]
    assert row["center_x_m"] == 0.123
    assert row["center_y_m"] == -0.045
    assert row["center_z_m"] == 0.7525
    assert row["yaw_rad"] == 1.57
    assert row["pose_source"] == "gz_model_pose"


def test_refresh_and_grasp_wrapper_refreshes_targets_before_grasping():
    script_path = REPO_ROOT / "scripts" / "refresh_and_grasp_m4_once.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "refresh_m4_scene_targets.py",
        "reports/m4_scene_snapshot/current_targets.json",
        "GHOST_MGG_M4_DYNAMIC_TARGETS_PATH",
        "GHOST_MGG_M4_REFRESH_SETTLE_SEC",
        "grasp_m4_once.sh",
        "--reset-executed-before-run",
        "--base-targets",
        "--output",
    ]:
        assert required in source
