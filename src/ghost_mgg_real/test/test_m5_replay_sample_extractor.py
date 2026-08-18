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
    / "extract_m5_replay_samples.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("extract_m5_replay_samples", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Message:
    pass


def _msg(encoding: str, height: int, width: int, data: bytes) -> Message:
    msg = Message()
    msg.encoding = encoding
    msg.height = height
    msg.width = width
    msg.data = data
    return msg


def test_decode_rgb_mono_and_depth_images():
    module = _load_module()

    rgb = _msg("rgb8", 1, 2, bytes([255, 0, 0, 0, 255, 0]))
    mono = _msg("mono8", 1, 2, bytes([10, 20]))
    depth = _msg(
        "16UC1",
        2,
        2,
        np.array([0, 1000, 2000, 3000], dtype=np.uint16).tobytes(),
    )

    rgb_array = module.decode_image_msg(rgb)
    mono_array = module.decode_image_msg(mono)
    depth_array = module.decode_image_msg(depth)

    assert rgb_array.shape == (1, 2, 3)
    assert rgb_array[0, 0].tolist() == [255, 0, 0]
    assert mono_array.tolist() == [[10, 20]]
    assert depth_array.dtype == np.uint16
    assert int(depth_array[1, 1]) == 3000


def test_depth_viz_ignores_zero_holes_and_writes_png(tmp_path):
    module = _load_module()
    depth = np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)
    output_path = tmp_path / "depth_viz.png"

    metadata = module.write_image_png(output_path, depth, encoding="16UC1")

    written = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    assert written.shape == (2, 2)
    assert written.dtype == np.uint8
    assert metadata["encoding"] == "16UC1"
    assert metadata["valid_ratio"] == 0.75
    assert metadata["mean_valid_depth_m"] == 2.0


def test_write_scene_outputs_manifest_and_expected_files(tmp_path):
    module = _load_module()
    output_dir = tmp_path / "samples"
    scene_id = "daylight_transparent_jelly_cup_001"
    frames = {
        "/camera/camera/color/image_raw": _msg(
            "rgb8", 1, 1, bytes([10, 20, 30])
        ),
        "/camera/camera/depth/image_rect_raw": _msg(
            "16UC1", 1, 1, np.array([1000], dtype=np.uint16).tobytes()
        ),
        "/camera/camera/infra1/image_rect_raw": _msg("mono8", 1, 1, bytes([42])),
        "/camera/camera/infra2/image_rect_raw": _msg("mono8", 1, 1, bytes([43])),
    }

    scene_manifest = module.write_scene_sample_outputs(scene_id, frames, output_dir)

    scene_dir = output_dir / scene_id
    assert (scene_dir / "color.png").exists()
    assert (scene_dir / "depth_viz.png").exists()
    assert (scene_dir / "infra1.png").exists()
    assert (scene_dir / "infra2.png").exists()
    metadata = json.loads((scene_dir / "metadata.json").read_text())
    assert metadata["scene_id"] == scene_id
    assert scene_manifest["scene_id"] == scene_id
    assert scene_manifest["outputs"]["depth"] == f"{scene_id}/depth_viz.png"


def test_write_index_markdown_links_scene_images(tmp_path):
    module = _load_module()
    output_dir = tmp_path / "samples"
    output_dir.mkdir()
    manifest = {
        "num_scenes": 1,
        "scenes": [
            {
                "scene_id": "daylight_transparent_jelly_cup_001",
                "outputs": {
                    "color": "daylight_transparent_jelly_cup_001/color.png",
                    "depth": "daylight_transparent_jelly_cup_001/depth_viz.png",
                    "infra1": "daylight_transparent_jelly_cup_001/infra1.png",
                },
                "missing_topics": [],
            }
        ],
    }

    module.write_index_markdown(manifest, output_dir)

    markdown = (output_dir / "index.md").read_text()
    assert "# M5 Replay Samples" in markdown
    assert "daylight_transparent_jelly_cup_001" in markdown
    assert "color.png" in markdown
    assert "depth_viz.png" in markdown
