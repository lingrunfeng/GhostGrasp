#!/usr/bin/env python3
"""Prepare human-editable target-mask annotation tasks for M5 real D435 samples."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_SCHEMA_VERSION = "m5_mask_annotation_task_v1"
MANIFEST_SCHEMA_VERSION = "m5_mask_annotation_manifest_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def infer_scene_defaults(scene_id: str) -> dict[str, Any]:
    """Infer conservative label metadata from the M5 scene id."""
    if "empty_table" in scene_id:
        return {
            "target_label": "empty_table",
            "shape_hint": "unknown",
            "include_in_benchmark": False,
        }
    if "multi_objects" in scene_id:
        return {
            "target_label": "multi_objects",
            "shape_hint": "unknown",
            "include_in_benchmark": False,
        }
    if "jelly_cup" in scene_id:
        return {
            "target_label": "transparent_jelly_cup",
            "shape_hint": "cup_like",
            "include_in_benchmark": True,
        }
    if "glass_cup" in scene_id:
        return {
            "target_label": "glass_cup",
            "shape_hint": "cup_like",
            "include_in_benchmark": True,
        }
    if "frosted_plastic_bowl" in scene_id:
        return {
            "target_label": "frosted_plastic_bowl",
            "shape_hint": "cup_like",
            "include_in_benchmark": True,
        }
    if "opaque_box" in scene_id:
        return {
            "target_label": "opaque_box",
            "shape_hint": "box",
            "include_in_benchmark": True,
        }
    if "metal_spoon" in scene_id:
        return {
            "target_label": "metal_spoon",
            "shape_hint": "unknown",
            "include_in_benchmark": True,
        }
    if "reflective_object" in scene_id:
        return {
            "target_label": "reflective_object",
            "shape_hint": "unknown",
            "include_in_benchmark": True,
        }
    return {
        "target_label": scene_id,
        "shape_hint": "unknown",
        "include_in_benchmark": True,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _output_path(manifest_path: Path, scene: dict[str, Any], output_key: str) -> str | None:
    output = scene.get("outputs", {}).get(output_key)
    if not output:
        return None
    return _portable_path(manifest_path.parent / output)


def _evidence_seed_masks(evidence_manifest_path: Path | None) -> dict[str, str]:
    if evidence_manifest_path is None:
        return {}
    manifest = _load_json(evidence_manifest_path)
    seed_masks: dict[str, str] = {}
    for scene in manifest.get("scenes", []):
        scene_id = scene.get("scene_id")
        target_mask = _output_path(evidence_manifest_path, scene, "target_mask")
        if scene_id and target_mask:
            seed_masks[str(scene_id)] = target_mask
    return seed_masks


def _task_from_scene(
    replay_manifest_path: Path,
    scene: dict[str, Any],
    seed_masks: dict[str, str],
) -> dict[str, Any]:
    scene_id = str(scene["scene_id"])
    defaults = infer_scene_defaults(scene_id)
    image_paths = {
        key: value
        for key in ("color", "depth", "aligned_depth", "infra1", "infra2")
        if (value := _output_path(replay_manifest_path, scene, key)) is not None
    }
    if scene_id in seed_masks:
        image_paths["seed_mask"] = seed_masks[scene_id]

    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "scene_id": scene_id,
        "status": "pending",
        "target_label": defaults["target_label"],
        "shape_hint": defaults["shape_hint"],
        "include_in_benchmark": defaults["include_in_benchmark"],
        "image_paths": image_paths,
        "polygons": [],
    }


def prepare_annotation_tasks(
    replay_manifest_path: Path,
    evidence_manifest_path: Path | None,
    annotations_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one annotation task JSON per replay-sample scene."""
    replay_manifest_path = Path(replay_manifest_path)
    evidence_manifest_path = Path(evidence_manifest_path) if evidence_manifest_path else None
    annotations_root = Path(annotations_root)

    replay_manifest = _load_json(replay_manifest_path)
    seed_masks = _evidence_seed_masks(evidence_manifest_path)
    tasks_dir = annotations_root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    task_records = []
    for scene in replay_manifest.get("scenes", []):
        scene_id = str(scene["scene_id"])
        task_path = tasks_dir / f"{scene_id}.json"
        wrote = False
        if overwrite or not task_path.exists():
            task = _task_from_scene(replay_manifest_path, scene, seed_masks)
            task_path.write_text(
                json.dumps(task, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            wrote = True
        task_records.append(
            {
                "scene_id": scene_id,
                "task_path": _portable_path(task_path),
                "written": wrote,
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "replay_manifest_path": _portable_path(replay_manifest_path),
        "evidence_manifest_path": (
            _portable_path(evidence_manifest_path) if evidence_manifest_path else None
        ),
        "annotations_root": _portable_path(annotations_root),
        "num_tasks": len(task_records),
        "tasks": task_records,
    }
    annotations_root.mkdir(parents=True, exist_ok=True)
    (annotations_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        default=Path("reports/m5_real_d435_replay_samples/manifest.json"),
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=Path("reports/m5_real_d435_evidence_preview/manifest.json"),
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("annotations/m5_real_d435_masks"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evidence_manifest = args.evidence_manifest if args.evidence_manifest.exists() else None
    manifest = prepare_annotation_tasks(
        replay_manifest_path=args.replay_manifest,
        evidence_manifest_path=evidence_manifest,
        annotations_root=args.annotations_root,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote annotation task manifest: {args.annotations_root / 'manifest.json'} "
        f"({manifest['num_tasks']} tasks)"
    )


if __name__ == "__main__":
    main()
