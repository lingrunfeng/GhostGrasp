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
    / "generate_m4_real_external_mask_replay.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m4_real_external_mask_replay", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_scene():
    target_mask = np.zeros((24, 24), dtype=bool)
    target_mask[6:18, 7:17] = True
    current_depth = np.full((24, 24), 1000, dtype=np.uint16)
    background_depth = np.full((24, 24), 1000, dtype=np.uint16)
    current_depth[target_mask] = 0
    current_depth[14:18, 9:15] = 955
    return target_mask, current_depth, background_depth


def test_rank_external_mask_scene_returns_live_hypothesis_contract_rows():
    module = _load_module()
    target_mask, current_depth, background_depth = _synthetic_scene()

    rows = module.rank_external_mask_scene(
        scene_id="daylight_transparent_jelly_cup_001",
        target_label="transparent_jelly_cup",
        shape_hint="cup_like",
        target_mask=target_mask,
        current_depth=current_depth,
        background_depth=background_depth,
        top_k=2,
    )

    assert len(rows) == 2
    first = rows[0]
    assert first["scene_id"] == "daylight_transparent_jelly_cup_001"
    assert first["rank"] == 1
    assert first["ranker"] == "failure_aware"
    assert first["hypothesis_id"]
    assert first["shape_type"] in {"box", "cylinder"}
    assert first["score"]["total"] == first["total_score"]
    assert first["score"]["failure"] == first["failure_score"]
    assert first["mask_pixels"] == int(target_mask.sum())
    assert first["hole_ratio"] > 0.0
    assert "external_mask_replay" in first["provenance"]
    assert "no_truth" in first["provenance"]
    assert "gazebo" not in first["provenance"].lower()


def test_write_real_external_mask_replay_report_creates_json_csv_and_index(tmp_path):
    module = _load_module()
    target_mask, current_depth, background_depth = _synthetic_scene()
    rows = module.rank_external_mask_scene(
        scene_id="scene_a",
        target_label="cup",
        shape_hint="cup_like",
        target_mask=target_mask,
        current_depth=current_depth,
        background_depth=background_depth,
        top_k=1,
    )

    module.write_external_mask_replay_reports(
        rows,
        tmp_path,
        source_data_dir=Path("data/real_d435_m5"),
        annotations_root=Path("annotations/m5_real_d435_masks"),
        background_scene_id="empty_table_001",
        top_k=1,
    )

    json_path = tmp_path / "m4_real_live_hypotheses.json"
    csv_path = tmp_path / "m4_real_live_hypotheses.csv"
    index_path = tmp_path / "index.md"
    assert json_path.exists()
    assert csv_path.exists()
    assert index_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "m4_real_external_mask_replay_v1"
    assert payload["contract"] == "external_mask_replay_to_live_hypotheses"
    assert payload["num_scenes"] == 1
    assert payload["num_hypotheses"] == 1
    assert payload["rows"][0]["ranker"] == "failure_aware"
    assert payload["rows"][0]["score"]["visual"] == payload["rows"][0]["visual_score"]
    assert b"\r\n" not in csv_path.read_bytes()
    markdown = index_path.read_text(encoding="utf-8")
    assert "M4 Real External-Mask Replay" in markdown
    assert "external mask contract" in markdown
    assert "scene_a" in markdown


def test_real_external_mask_replay_script_contract_is_no_truth():
    script_path = REPO_ROOT / "scripts" / "run_m4_real_external_mask_replay.sh"
    assert script_path.exists()
    source = script_path.read_text(encoding="utf-8")

    for required in [
        "generate_m4_real_external_mask_replay.py",
        "--data-dir data/real_d435_m5",
        "--annotations-root annotations/m5_real_d435_masks",
        "--output-dir reports/m4_real_external_mask_replay",
        "m4_real_live_hypotheses.json",
        "m4_real_live_hypotheses.csv",
        "M4 real external-mask replay passed",
    ]:
        assert required in source

    for forbidden in [
        "gz model",
        "gz service",
        "current_targets.json",
        "m4_sim_grasp_targets.json",
    ]:
        assert forbidden not in source
