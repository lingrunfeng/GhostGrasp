import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "generate_m5_real_evidence_previews.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m5_real_evidence_previews", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_target_mask_combines_rgb_and_depth_change():
    module = _load_module()
    background_rgb = np.zeros((5, 5, 3), dtype=np.uint8) + 100
    current_rgb = background_rgb.copy()
    current_rgb[2, 2] = [160, 160, 160]
    background_depth = np.zeros((5, 5), dtype=np.uint16) + 1000
    current_depth = background_depth.copy()
    current_depth[1, 1] = 800

    mask = module.build_target_mask(
        current_rgb,
        background_rgb,
        current_depth,
        background_depth,
        rgb_threshold=20,
        depth_threshold_mm=50,
        morph_kernel=1,
    )

    assert mask[2, 2]
    assert mask[1, 1]
    assert not mask[0, 0]


def test_compute_evidence_channels_from_depth_and_background():
    module = _load_module()
    target_mask = np.array([[True, True, True, True]], dtype=bool)
    current_depth = np.array([[0, 1000, 850, 1200]], dtype=np.uint16)
    background_depth = np.array([[1000, 1005, 1000, 1000]], dtype=np.uint16)

    evidence = module.compute_evidence_channels(
        target_mask,
        current_depth,
        background_depth,
        leakage_tolerance_mm=15,
        foreground_margin_mm=50,
    )

    assert evidence["hole"][0, 0]
    assert evidence["table_leakage"][0, 1]
    assert evidence["foreground"][0, 2]
    assert evidence["background_shift"][0, 3]


def test_write_evidence_outputs_creates_images_and_summary(tmp_path):
    module = _load_module()
    scene_id = "daylight_transparent_jelly_cup_001"
    rgb = np.zeros((3, 3, 3), dtype=np.uint8) + 100
    target_mask = np.zeros((3, 3), dtype=bool)
    target_mask[1, 1] = True
    evidence = {
        "hole": target_mask.copy(),
        "table_leakage": np.zeros((3, 3), dtype=bool),
        "foreground": np.zeros((3, 3), dtype=bool),
        "background_shift": np.zeros((3, 3), dtype=bool),
    }

    summary = module.write_evidence_outputs(scene_id, rgb, target_mask, evidence, tmp_path)

    scene_dir = tmp_path / scene_id
    assert (scene_dir / "target_mask.png").exists()
    assert (scene_dir / "evidence_overlay.png").exists()
    metadata = json.loads((scene_dir / "evidence_summary.json").read_text())
    assert metadata["scene_id"] == scene_id
    assert metadata["target_pixels"] == 1
    assert metadata["hole_ratio"] == 1.0
    assert summary["scene_id"] == scene_id
