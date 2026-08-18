import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "generate_m4_visual_ranking_board.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m4_visual_ranking_board", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_draw_hypothesis_overlay_modifies_expected_region():
    module = _load_module()
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    hypothesis = module.TopHypothesis(
        scene_id="scene_a",
        ranker="failure_aware",
        hypothesis_id="box_s1.00",
        shape_type="box",
        center_u=50.0,
        center_v=40.0,
        size_u_px=30.0,
        size_v_px=20.0,
        visual_score=0.8,
        failure_score=0.6,
        total_score=3.0,
    )

    rendered = module.draw_hypothesis_overlay(image, hypothesis, color=(0, 0, 255))

    assert rendered.sum() > 0
    assert rendered[30, 35, 2] > 0 or rendered[50, 65, 2] > 0


def test_generate_visual_board_report_writes_scene_png_manifest_and_index(tmp_path):
    module = _load_module()
    scene_id = "scene_a"
    replay_dir = tmp_path / "replay"
    masks_root = tmp_path / "masks"
    evidence_dir = tmp_path / "evidence"
    output_dir = tmp_path / "boards"
    (replay_dir / scene_id).mkdir(parents=True)
    (masks_root / scene_id).mkdir(parents=True)
    (evidence_dir / scene_id).mkdir(parents=True)

    rgb = np.full((80, 100, 3), 160, dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[25:55, 35:65] = 255
    evidence = rgb.copy()
    evidence[:, :, 1] = 220
    cv2.imwrite(str(replay_dir / scene_id / "color.png"), rgb)
    cv2.imwrite(str(masks_root / scene_id / "target_mask.png"), mask)
    cv2.imwrite(str(evidence_dir / scene_id / "evidence_overlay.png"), evidence)

    ranking_csv = tmp_path / "ranking.csv"
    dashboard_csv = tmp_path / "dashboard.csv"
    _write_csv(
        ranking_csv,
        [
            {
                "scene_id": scene_id,
                "target_label": "cup",
                "shape_hint": "cup_like",
                "ranker": "silhouette_only",
                "rank": 1,
                "hypothesis_id": "cylinder_s1.00",
                "shape_type": "cylinder",
                "center_u": 50,
                "center_v": 40,
                "size_u_px": 40,
                "size_v_px": 30,
                "visual_score": 0.9,
                "failure_score": -0.1,
                "depth_score": 0.0,
                "total_score": 0.9,
                "validation_state": "accepted",
            },
            {
                "scene_id": scene_id,
                "target_label": "cup",
                "shape_hint": "cup_like",
                "ranker": "failure_aware",
                "rank": 1,
                "hypothesis_id": "box_s1.00",
                "shape_type": "box",
                "center_u": 50,
                "center_v": 40,
                "size_u_px": 30,
                "size_v_px": 20,
                "visual_score": 0.8,
                "failure_score": 0.6,
                "depth_score": 0.0,
                "total_score": 3.0,
                "validation_state": "accepted",
            },
        ],
    )
    _write_csv(
        dashboard_csv,
        [
            {
                "scene_id": scene_id,
                "target_label": "cup",
                "shape_hint": "cup_like",
                "material_class": "transparent",
                "object_family": "jelly_cup",
                "target_pixels": 900,
                "valid_depth_ratio": 0.4,
                "hole_ratio": 0.6,
                "table_leakage_ratio": 0.2,
                "foreground_ratio": 0.03,
                "silhouette_top": "cylinder_s1.00",
                "failure_top": "box_s1.00",
                "silhouette_shape": "cylinder",
                "failure_shape": "box",
                "top1_changed": "True",
                "shape_changed": "True",
                "failure_score_delta": 0.7,
                "visual_score_delta": -0.1,
                "acceptable_shapes": "box;cylinder",
                "acceptable_shape_ok": "True",
                "failure_gain_ok": "True",
                "visual_drop_ok": "True",
                "shape_stability_ok": "True",
                "weak_gt_pass": "True",
            }
        ],
    )

    manifest = module.generate_visual_board_report(
        replay_samples_dir=replay_dir,
        masks_root=masks_root,
        evidence_dir=evidence_dir,
        ranking_csv=ranking_csv,
        dashboard_csv=dashboard_csv,
        output_dir=output_dir,
    )

    assert manifest["schema_version"] == "m4_visual_ranking_board_manifest_v1"
    assert manifest["num_scenes"] == 1
    board_path = output_dir / f"{scene_id}.png"
    assert board_path.exists()
    board = cv2.imread(str(board_path), cv2.IMREAD_COLOR)
    assert board is not None
    assert board.shape[0] > 80
    assert board.shape[1] > 100
    assert (output_dir / "index.md").exists()
    saved_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert saved_manifest["scenes"][0]["scene_id"] == scene_id
