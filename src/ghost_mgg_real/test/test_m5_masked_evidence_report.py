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
    / "generate_m5_masked_evidence_report.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m5_masked_evidence_report", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compute_masked_evidence_summary_counts_failure_states():
    module = _load_module()
    target_mask = np.array([[True, True, True, True, True]], dtype=bool)
    current_depth = np.array([[0, 1000, 850, 1200, 920]], dtype=np.uint16)
    background_depth = np.array([[1000, 1005, 1000, 1000, 0]], dtype=np.uint16)

    summary, evidence = module.compute_masked_evidence_summary(
        scene_id="scene_a",
        target_mask=target_mask,
        current_depth=current_depth,
        background_depth=background_depth,
        leakage_tolerance_mm=15,
        foreground_margin_mm=50,
    )

    assert summary["target_pixels"] == 5
    assert summary["valid_depth_ratio"] == 0.8
    assert summary["comparable_depth_ratio"] == 0.6
    assert summary["hole_ratio"] == 0.2
    assert summary["table_leakage_ratio"] == 0.2
    assert summary["foreground_ratio"] == 0.2
    assert summary["background_shift_ratio"] == 0.2
    assert summary["unexplained_valid_ratio"] == 0.2
    assert evidence["hole"][0, 0]
    assert evidence["table_leakage"][0, 1]
    assert evidence["foreground"][0, 2]
    assert evidence["background_shift"][0, 3]
    assert evidence["unexplained_valid"][0, 4]


def test_load_completed_mask_records_filters_rasterized_entries(tmp_path):
    module = _load_module()
    annotations_root = tmp_path / "annotations"
    mask_dir = annotations_root / "masks" / "scene_a"
    mask_dir.mkdir(parents=True)
    mask_path = mask_dir / "target_mask.png"
    cv2.imwrite(str(mask_path), np.ones((3, 3), dtype=np.uint8) * 255)
    task_path = annotations_root / "tasks" / "scene_a.json"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps({"target_label": "cup", "shape_hint": "cup_like"}))
    manifest = {
        "tasks": [
            {
                "scene_id": "scene_a",
                "status": "rasterized",
                "mask_path": str(mask_path),
                "task_path": str(task_path),
            },
            {"scene_id": "scene_b", "status": "skipped"},
        ]
    }
    manifest_path = annotations_root / "masks" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    records = module.load_completed_mask_records(annotations_root)

    assert len(records) == 1
    assert records[0]["scene_id"] == "scene_a"
    assert records[0]["target_label"] == "cup"
    assert records[0]["shape_hint"] == "cup_like"
    assert records[0]["mask_path"] == mask_path


def test_write_masked_evidence_outputs_creates_review_artifacts(tmp_path):
    module = _load_module()
    rgb = np.zeros((4, 4, 3), dtype=np.uint8) + 100
    target_mask = np.zeros((4, 4), dtype=bool)
    target_mask[1:3, 1:3] = True
    evidence = {key: np.zeros((4, 4), dtype=bool) for key in module.EVIDENCE_CHANNELS}
    evidence["hole"][1, 1] = True
    summary = {
        "schema_version": "m5_masked_evidence_scene_v1",
        "scene_id": "scene_a",
        "target_pixels": 4,
        "hole_ratio": 0.25,
    }

    output = module.write_masked_evidence_outputs(
        scene_id="scene_a",
        rgb=rgb,
        target_mask=target_mask,
        evidence=evidence,
        summary=summary,
        output_dir=tmp_path,
    )

    scene_dir = tmp_path / "scene_a"
    assert (scene_dir / "formal_mask.png").exists()
    assert (scene_dir / "evidence_overlay.png").exists()
    saved = json.loads((scene_dir / "evidence_summary.json").read_text())
    assert saved["scene_id"] == "scene_a"
    assert output["outputs"]["formal_mask"] == "scene_a/formal_mask.png"
