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
    / "generate_m4_metric_proxies.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_m4_metric_proxies", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ranking_json(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "m5_real_ranking_v1",
                "num_scenes": 1,
                "num_rows": 2,
                "rows": [
                    {
                        "scene_id": "scene_a",
                        "target_label": "cup",
                        "shape_hint": "cup_like",
                        "ranker": "failure_aware",
                        "rank": 1,
                        "hypothesis_id": "box_s1.00",
                        "shape_type": "box",
                        "center_u": 370.0,
                        "center_v": 260.0,
                        "size_u_px": 80.0,
                        "size_v_px": 40.0,
                        "visual_score": 0.8,
                        "failure_score": 0.6,
                        "depth_score": 0.0,
                        "total_score": 1.4,
                        "validation_state": "accepted",
                    },
                    {
                        "scene_id": "scene_a",
                        "target_label": "cup",
                        "shape_hint": "cup_like",
                        "ranker": "failure_aware",
                        "rank": 2,
                        "hypothesis_id": "box_s1.10",
                        "shape_type": "box",
                        "center_u": 370.0,
                        "center_v": 260.0,
                        "size_u_px": 88.0,
                        "size_v_px": 44.0,
                        "visual_score": 0.7,
                        "failure_score": 0.5,
                        "depth_score": 0.0,
                        "total_score": 1.2,
                        "validation_state": "accepted",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _frame_stats_csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "scene_id,topic,frames_sampled,valid_ratio,mean_valid_depth_m,min_valid_depth_m,max_valid_depth_m,mean_intensity,nonzero_ratio,saturated_ratio",
                "scene_a,/camera/camera/aligned_depth_to_color/image_raw,12.0,0.90,0.250,0.18,0.45,,,",
                "scene_a,/camera/camera/depth/image_rect_raw,12.0,0.80,0.400,0.15,8.0,,,",
                "scene_b,/camera/camera/aligned_depth_to_color/image_raw,12.0,0.90,0.300,0.18,0.45,,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_pixel_proxy_to_metric_uses_pinhole_and_table_anchor():
    module = _load_module()
    row = {
        "scene_id": "scene_a",
        "ranker": "failure_aware",
        "rank": 1,
        "hypothesis_id": "box_s1.00",
        "shape_type": "box",
        "center_u": 370.0,
        "center_v": 260.0,
        "size_u_px": 80.0,
        "size_v_px": 40.0,
        "visual_score": 0.8,
        "failure_score": 0.6,
        "total_score": 1.4,
    }

    proxy = module.pixel_proxy_to_metric(
        row,
        fx=500.0,
        fy=400.0,
        cx=320.0,
        cy=240.0,
        table_depth_m=1.0,
        table_z_m=0.75,
        primitive_height_m=0.04,
    )

    assert proxy.scene_id == "scene_a"
    assert proxy.hypothesis_id == "box_s1.00"
    assert proxy.center_x_m == 0.1
    assert proxy.center_y_m == 0.05
    assert proxy.center_z_m == 0.77
    assert proxy.width_m == 0.16
    assert proxy.depth_m == 0.10
    assert proxy.height_m == 0.04


def test_metric_proxy_report_uses_per_scene_aligned_depth_from_frame_stats(tmp_path):
    module = _load_module()
    ranking_path = _ranking_json(tmp_path / "ranking.json")
    frame_stats_path = _frame_stats_csv(tmp_path / "frame_stats.csv")
    output_dir = tmp_path / "metric"

    rows = module.generate_metric_proxy_report(
        ranking_json=ranking_path,
        output_dir=output_dir,
        frame_stats_csv=frame_stats_path,
        fx=500.0,
        fy=400.0,
        cx=320.0,
        cy=240.0,
        table_depth_m=1.0,
        table_z_m=0.75,
        primitive_height_m=0.04,
    )

    assert rows[0].table_depth_m == 0.25
    assert rows[0].table_depth_source == "frame_stats_aligned_depth_mean"
    assert rows[0].center_x_m == 0.025
    assert rows[0].width_m == 0.04
    payload = json.loads((output_dir / "metric_proxies.json").read_text())
    assert payload["rows"][0]["table_depth_source"] == "frame_stats_aligned_depth_mean"


def test_metric_proxy_report_writes_rank1_rows_json_csv_and_index(tmp_path):
    module = _load_module()
    ranking_path = _ranking_json(tmp_path / "ranking.json")
    output_dir = tmp_path / "metric"

    rows = module.generate_metric_proxy_report(
        ranking_json=ranking_path,
        output_dir=output_dir,
        fx=500.0,
        fy=400.0,
        cx=320.0,
        cy=240.0,
        table_depth_m=1.0,
        table_z_m=0.75,
        primitive_height_m=0.04,
    )

    assert len(rows) == 1
    assert rows[0].rank == 1
    payload = json.loads((output_dir / "metric_proxies.json").read_text())
    assert payload["schema_version"] == "m4_metric_proxies_v1"
    assert payload["num_rows"] == 1
    assert payload["rows"][0]["center_z_m"] == 0.77
    assert (output_dir / "metric_proxies.csv").exists()
    index = (output_dir / "index.md").read_text()
    assert "M4 Metric Proxies" in index
    assert "scene_a" in index
