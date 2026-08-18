import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "prepare_m5_mask_annotation_tasks.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_m5_mask_annotation_tasks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_replay_manifest(root: Path) -> Path:
    manifest = {
        "schema_version": "m5_replay_samples_manifest_v1",
        "num_scenes": 3,
        "scenes": [
            {
                "scene_id": "daylight_transparent_jelly_cup_001",
                "outputs": {
                    "color": "daylight_transparent_jelly_cup_001/color.png",
                    "depth": "daylight_transparent_jelly_cup_001/depth_viz.png",
                    "infra1": "daylight_transparent_jelly_cup_001/infra1.png",
                    "infra2": "daylight_transparent_jelly_cup_001/infra2.png",
                },
                "missing_topics": [],
            },
            {
                "scene_id": "daylight_multi_objects_001",
                "outputs": {
                    "color": "daylight_multi_objects_001/color.png",
                    "depth": "daylight_multi_objects_001/depth_viz.png",
                },
                "missing_topics": [],
            },
            {
                "scene_id": "empty_table_001",
                "outputs": {
                    "color": "empty_table_001/color.png",
                    "depth": "empty_table_001/depth_viz.png",
                },
                "missing_topics": [],
            },
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_evidence_manifest(root: Path) -> Path:
    manifest = {
        "schema_version": "m5_real_evidence_preview_manifest_v1",
        "num_scenes": 1,
        "scenes": [
            {
                "scene_id": "daylight_transparent_jelly_cup_001",
                "outputs": {
                    "target_mask": "daylight_transparent_jelly_cup_001/target_mask.png",
                    "evidence_overlay": "daylight_transparent_jelly_cup_001/evidence_overlay.png",
                },
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_infer_scene_defaults_from_scene_ids():
    module = _load_module()

    jelly = module.infer_scene_defaults("daylight_transparent_jelly_cup_001")
    multi = module.infer_scene_defaults("daylight_multi_objects_001")
    empty = module.infer_scene_defaults("empty_table_001")

    assert jelly["target_label"] == "transparent_jelly_cup"
    assert jelly["shape_hint"] == "cup_like"
    assert jelly["include_in_benchmark"] is True
    assert multi["target_label"] == "multi_objects"
    assert multi["include_in_benchmark"] is False
    assert empty["target_label"] == "empty_table"
    assert empty["include_in_benchmark"] is False


def test_prepare_annotation_tasks_writes_templates(tmp_path):
    module = _load_module()
    replay_root = tmp_path / "replay"
    evidence_root = tmp_path / "evidence"
    annotations_root = tmp_path / "annotations"
    replay_manifest = _write_replay_manifest(replay_root)
    evidence_manifest = _write_evidence_manifest(evidence_root)

    manifest = module.prepare_annotation_tasks(
        replay_manifest_path=replay_manifest,
        evidence_manifest_path=evidence_manifest,
        annotations_root=annotations_root,
        overwrite=False,
    )

    assert manifest["num_tasks"] == 3
    task_path = annotations_root / "tasks" / "daylight_transparent_jelly_cup_001.json"
    task = json.loads(task_path.read_text())
    assert task["schema_version"] == "m5_mask_annotation_task_v1"
    assert task["status"] == "pending"
    assert task["target_label"] == "transparent_jelly_cup"
    assert task["shape_hint"] == "cup_like"
    assert task["image_paths"]["color"].endswith("color.png")
    assert task["image_paths"]["seed_mask"].endswith("target_mask.png")
    assert task["polygons"] == []


def test_prepare_annotation_tasks_preserves_existing_without_overwrite(tmp_path):
    module = _load_module()
    replay_root = tmp_path / "replay"
    annotations_root = tmp_path / "annotations"
    replay_manifest = _write_replay_manifest(replay_root)
    existing_path = annotations_root / "tasks" / "daylight_transparent_jelly_cup_001.json"
    existing_path.parent.mkdir(parents=True)
    existing_path.write_text(
        json.dumps({"status": "complete", "polygons": [[[1, 1], [2, 1], [2, 2]]]}),
        encoding="utf-8",
    )

    module.prepare_annotation_tasks(
        replay_manifest_path=replay_manifest,
        evidence_manifest_path=None,
        annotations_root=annotations_root,
        overwrite=False,
    )

    preserved = json.loads(existing_path.read_text())
    assert preserved["status"] == "complete"
    assert preserved["polygons"] == [[[1, 1], [2, 1], [2, 2]]]
