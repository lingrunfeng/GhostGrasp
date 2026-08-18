import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "ghost_mgg_real"
    / "scripts"
    / "analyze_m5_real_d435_bags.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_m5_real_d435_bags", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_metadata(scene_dir: Path, duration_ns: int, topics: dict[str, int]) -> None:
    scene_dir.mkdir(parents=True)
    metadata = {
        "rosbag2_bagfile_information": {
            "version": 9,
            "storage_identifier": "mcap",
            "duration": {"nanoseconds": duration_ns},
            "starting_time": {"nanoseconds_since_epoch": 1783000000000000000},
            "message_count": sum(topics.values()),
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": topic,
                        "type": "sensor_msgs/msg/Image",
                        "serialization_format": "cdr",
                    },
                    "message_count": count,
                }
                for topic, count in topics.items()
            ],
        }
    }
    (scene_dir / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    (scene_dir / f"{scene_dir.name}_0.mcap").write_bytes(b"fake")


def test_parse_metadata_reports_required_topic_completeness(tmp_path):
    module = _load_module()
    scene_dir = tmp_path / "daylight_transparent_jelly_cup_001"
    topics = {topic: 30 for topic in module.REQUIRED_TOPICS}
    _write_metadata(scene_dir, duration_ns=15_000_000_000, topics=topics)

    summary = module.parse_bag_metadata(scene_dir)

    assert summary.scene_id == "daylight_transparent_jelly_cup_001"
    assert summary.duration_sec == 15.0
    assert summary.message_count == 30 * len(module.REQUIRED_TOPICS)
    assert summary.bag_size_bytes == 4
    assert summary.missing_required_topics == []
    assert summary.is_complete is True
    assert summary.topic_counts["/camera/camera/depth/image_rect_raw"] == 30


def test_parse_metadata_lists_missing_required_topics(tmp_path):
    module = _load_module()
    scene_dir = tmp_path / "bad_scene"
    topics = {
        "/camera/camera/color/image_raw": 10,
        "/camera/camera/depth/image_rect_raw": 10,
    }
    _write_metadata(scene_dir, duration_ns=5_000_000_000, topics=topics)

    summary = module.parse_bag_metadata(scene_dir)

    assert summary.is_complete is False
    assert "/camera/camera/infra1/image_rect_raw" in summary.missing_required_topics
    assert "/tf_static" in summary.missing_required_topics


def test_image_metric_helpers_measure_depth_validity_and_ir_brightness():
    module = _load_module()

    class Message:
        pass

    depth_msg = Message()
    depth_msg.encoding = "16UC1"
    depth_msg.height = 2
    depth_msg.width = 3
    depth_msg.data = np.array([0, 1000, 2000, 0, 500, 3000], dtype=np.uint16).tobytes()

    depth_metrics = module.image_metrics_from_msg(depth_msg)

    assert depth_metrics["valid_ratio"] == 4 / 6
    assert depth_metrics["mean_valid_depth_m"] == 1.625

    ir_msg = Message()
    ir_msg.encoding = "mono8"
    ir_msg.height = 2
    ir_msg.width = 2
    ir_msg.data = bytes([0, 10, 20, 30])

    ir_metrics = module.image_metrics_from_msg(ir_msg)

    assert ir_metrics["mean_intensity"] == 15.0
    assert ir_metrics["nonzero_ratio"] == 0.75


def test_write_reports_creates_machine_and_human_readable_outputs(tmp_path):
    module = _load_module()
    data_dir = tmp_path / "bags"
    scene_dir = data_dir / "daylight_transparent_jelly_cup_visible_points_001"
    topics = {topic: 20 for topic in module.REQUIRED_TOPICS}
    _write_metadata(scene_dir, duration_ns=10_000_000_000, topics=topics)

    summaries = [module.parse_bag_metadata(scene_dir)]
    output_dir = tmp_path / "reports"
    image_stats = {
        scene_dir.name: {
            "/camera/camera/depth/image_rect_raw": {
                "frames_sampled": 2.0,
                "valid_ratio": 0.5,
                "mean_valid_depth_m": 0.8,
                "min_valid_depth_m": 0.4,
                "max_valid_depth_m": 1.2,
            }
        }
    }
    module.write_reports(summaries, image_stats, output_dir)

    manifest = json.loads((output_dir / "m5_real_d435_manifest.json").read_text())
    csv_text = (output_dir / "m5_real_d435_manifest.csv").read_text()
    markdown = (output_dir / "m5_real_d435_summary.md").read_text()

    assert manifest["schema_version"] == "m5_real_d435_manifest_v1"
    assert manifest["num_bags"] == 1
    assert "scene_id,duration_sec,message_count" in csv_text
    assert "min_valid_depth_m" in (output_dir / "m5_real_d435_frame_stats.csv").read_text()
    assert "daylight_transparent_jelly_cup_visible_points_001" in markdown
    assert "required topics complete" in markdown
