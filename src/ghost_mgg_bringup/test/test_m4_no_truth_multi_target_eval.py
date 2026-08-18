from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_m4_no_truth_multi_target_runner_contract():
    script_path = REPO_ROOT / "scripts" / "run_m4_no_truth_multi_target_eval.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "m4_no_truth_live_moveit_execute.launch.py",
        "/ghost_mgg/m4_live_hypotheses",
        "red_cube_box,red_cube,box",
        "blue_cylinder_center,blue_cylinder,cylinder",
        "green_cylinder_offset,green_cylinder,cylinder",
        "glass_block_box,glass_block,box",
        "move_non_target_clutter_out_of_view",
        "cleanup_stale_m4_processes",
        "print(1.10 + 0.08 * int(\"${index}\"))",
        "last_rejected_hypothesis=",
        "awk '/^\\{.*\\}$/ {line=$0} END {print line}'",
        "top_k:=1",
        "max_visible_hypotheses:=1",
        "probe_m4_sim_grasp_moveit.py",
        "--omit-target-obstacles",
        "execute_m4_joint_hypothesis_action.py",
        "reports/m4_no_truth_multi_target",
        "multi_target.json",
        "multi_target.csv",
        "index.md",
        "pose_error_m",
        "expected_shape",
        "predicted_shape",
        "shape_family_match",
        "M4 no-truth multi-target eval passed",
        "M4 no-truth multi-target eval failed: no accepted hypothesis",
        "deadline=$((SECONDS + 90))",
        "[[ -z \"${output}\" ]]",
        "2>/dev/null <<'PY'",
    ]:
        assert required in source

    assert "uses the current top live hypothesis" in source
    assert "--target-id" not in source


def test_m4_no_truth_multi_target_registered_in_cmake():
    cmake_text = (REPO_ROOT / "src" / "ghost_mgg_bringup" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert "test_m4_no_truth_multi_target_eval" in cmake_text
