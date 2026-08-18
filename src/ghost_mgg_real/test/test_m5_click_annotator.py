import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT / "src" / "ghost_mgg_real" / "scripts" / "annotate_m5_mask.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("annotate_m5_mask", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_color(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 0] = 30
    image[:, :, 1] = 100
    image[:, :, 2] = 160
    assert cv2.imwrite(str(path), image)


def _write_task(annotations_root: Path, scene_id: str, color_path: Path) -> Path:
    task_path = annotations_root / "tasks" / f"{scene_id}.json"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task = {
        "schema_version": "m5_mask_annotation_task_v1",
        "scene_id": scene_id,
        "status": "pending",
        "target_label": "transparent_jelly_cup",
        "shape_hint": "cup_like",
        "include_in_benchmark": True,
        "image_paths": {"color": str(color_path)},
        "polygons": [],
    }
    task_path.write_text(json.dumps(task), encoding="utf-8")
    return task_path


def test_task_path_for_scene_uses_annotation_root():
    module = _load_module()

    path = module.task_path_for_scene(
        Path("annotations/m5_real_d435_masks"),
        "daylight_transparent_jelly_cup_001",
    )

    assert path == (
        Path("annotations/m5_real_d435_masks")
        / "tasks"
        / "daylight_transparent_jelly_cup_001.json"
    )


def test_save_completed_polygon_updates_task_and_rasterizes(tmp_path):
    module = _load_module()
    annotations_root = tmp_path / "annotations"
    color_path = tmp_path / "replay" / "scene_a" / "color.png"
    _write_color(color_path)
    task_path = _write_task(annotations_root, "scene_a", color_path)

    summary = module.save_completed_polygon(
        task_path=task_path,
        annotations_root=annotations_root,
        polygon=[(5, 5), (18, 5), (18, 18), (5, 18)],
    )

    task = json.loads(task_path.read_text())
    assert task["status"] == "complete"
    assert task["polygons"] == [[[5, 5], [18, 5], [18, 18], [5, 18]]]
    assert summary["status"] == "rasterized"
    assert (annotations_root / "masks" / "scene_a" / "target_mask.png").exists()
    assert (annotations_root / "masks" / "scene_a" / "annotation_overlay.png").exists()


def test_save_completed_polygon_rejects_too_few_points(tmp_path):
    module = _load_module()
    annotations_root = tmp_path / "annotations"
    color_path = tmp_path / "replay" / "scene_a" / "color.png"
    _write_color(color_path)
    task_path = _write_task(annotations_root, "scene_a", color_path)

    try:
        module.save_completed_polygon(
            task_path=task_path,
            annotations_root=annotations_root,
            polygon=[(5, 5), (18, 5)],
        )
    except ValueError as exc:
        assert "at least 3 points" in str(exc)
    else:
        raise AssertionError("expected ValueError")
