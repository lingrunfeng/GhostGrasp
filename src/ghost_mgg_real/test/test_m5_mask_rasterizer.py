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
    / "rasterize_m5_mask_annotations.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("rasterize_m5_mask_annotations", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_color(path: Path, height: int = 20, width: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 80
    image[:, :, 2] = 160
    assert cv2.imwrite(str(path), image)


def _write_task(
    annotations_root: Path,
    scene_id: str,
    color_path: Path,
    status: str,
    polygons: list[list[list[int]]],
) -> Path:
    task_path = annotations_root / "tasks" / f"{scene_id}.json"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task = {
        "schema_version": "m5_mask_annotation_task_v1",
        "scene_id": scene_id,
        "status": status,
        "target_label": scene_id,
        "shape_hint": "unknown",
        "include_in_benchmark": True,
        "image_paths": {"color": str(color_path)},
        "polygons": polygons,
    }
    task_path.write_text(json.dumps(task), encoding="utf-8")
    return task_path


def test_rasterize_polygons_fills_binary_region():
    module = _load_module()

    mask = module.rasterize_polygons(
        polygons=[[[2, 2], [7, 2], [7, 7], [2, 7]]],
        height=10,
        width=10,
    )

    assert mask.dtype == np.bool_
    assert mask[4, 4]
    assert not mask[0, 0]
    assert mask.sum() > 20


def test_rasterize_completed_task_writes_mask_overlay_and_summary(tmp_path):
    module = _load_module()
    annotations_root = tmp_path / "annotations"
    color_path = tmp_path / "replay" / "scene_a" / "color.png"
    _write_color(color_path)
    task_path = _write_task(
        annotations_root,
        "scene_a",
        color_path,
        "complete",
        [[[5, 5], [14, 5], [14, 14], [5, 14]]],
    )

    summary = module.rasterize_annotation_task(task_path, annotations_root)

    assert summary["status"] == "rasterized"
    mask_path = annotations_root / "masks" / "scene_a" / "target_mask.png"
    overlay_path = annotations_root / "masks" / "scene_a" / "annotation_overlay.png"
    summary_path = annotations_root / "masks" / "scene_a" / "mask_summary.json"
    assert mask_path.exists()
    assert overlay_path.exists()
    assert summary_path.exists()

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(mask)).issubset({0, 255})
    assert int(mask.sum()) > 0
    saved_summary = json.loads(summary_path.read_text())
    assert saved_summary["scene_id"] == "scene_a"
    assert saved_summary["mask_pixels"] == int((mask > 0).sum())


def test_rasterize_annotations_skips_pending_tasks_and_writes_manifest(tmp_path):
    module = _load_module()
    annotations_root = tmp_path / "annotations"
    color_path = tmp_path / "replay" / "color.png"
    _write_color(color_path)
    _write_task(
        annotations_root,
        "complete_scene",
        color_path,
        "complete",
        [[[3, 3], [8, 3], [8, 8], [3, 8]]],
    )
    _write_task(annotations_root, "pending_scene", color_path, "pending", [])

    manifest = module.rasterize_annotations(annotations_root)

    assert manifest["num_tasks"] == 2
    assert manifest["num_rasterized"] == 1
    assert manifest["num_skipped"] == 1
    skipped = [item for item in manifest["tasks"] if item["status"] == "skipped"]
    assert skipped[0]["scene_id"] == "pending_scene"
    assert (annotations_root / "masks" / "manifest.json").exists()
