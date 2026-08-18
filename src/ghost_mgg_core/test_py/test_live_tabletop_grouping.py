import numpy as np

from ghost_mgg_core_py.live_tabletop.components import extract_components
from ghost_mgg_core_py.live_tabletop.grouping import group_components_by_foreground_islands


def test_splits_two_independent_objects_connected_by_thin_failure_bridge():
    mask = np.zeros((24, 42), dtype=bool)
    mask[8:16, 4:12] = True
    mask[8:16, 28:36] = True
    mask[11:13, 12:28] = True
    component = extract_components(mask, min_area_px=1)[0]
    z_map = np.full(mask.shape, 0.74, dtype=float)
    z_map[8:16, 4:12] = 0.80
    z_map[8:16, 28:36] = 0.80
    z_map[11:13, 12:28] = np.nan

    result = group_components_by_foreground_islands(
        [component],
        z_map=z_map,
        table_z_m=0.74,
        min_island_area_px=8,
        min_foreground_fraction=0.35,
    )

    assert len(result.components) == 2
    assert len(result.split_component_ids) == 2
    centers_u = sorted(float(item.centroid_uv[0]) for item in result.components)
    assert centers_u[0] < 16.0
    assert centers_u[1] > 24.0
    assert {decision.action for decision in result.decisions} == {"split"}
    assert result.decisions[0].reason == "separate_foreground_islands"


def test_split_components_do_not_keep_depth_hole_bridge_as_object_area():
    mask = np.zeros((28, 48), dtype=bool)
    mask[8:16, 5:13] = True
    mask[8:16, 31:39] = True
    mask[11:13, 13:31] = True
    mask[6:18, 24:31] = True
    component = extract_components(mask, min_area_px=1)[0]
    z_map = np.full(mask.shape, 0.74, dtype=float)
    z_map[8:16, 5:13] = 0.80
    z_map[8:16, 31:39] = 0.80
    z_map[11:13, 13:31] = np.nan
    z_map[6:18, 24:31] = np.nan

    result = group_components_by_foreground_islands(
        [component],
        z_map=z_map,
        table_z_m=0.74,
        min_island_area_px=8,
        min_foreground_fraction=0.30,
    )

    assert len(result.components) == 2
    widths = sorted(item.bbox_xyxy[2] - item.bbox_xyxy[0] + 1 for item in result.components)
    assert widths == [8, 8]
    assert sum(item.area_px for item in result.components) == 128


def test_splits_sparse_foreground_islands_when_large_hole_bridge_lowers_foreground_fraction():
    mask = np.zeros((36, 76), dtype=bool)
    mask[12:20, 6:14] = True
    mask[12:20, 56:64] = True
    mask[13:19, 14:56] = True
    component = extract_components(mask, min_area_px=1)[0]
    z_map = np.full(mask.shape, 0.74, dtype=float)
    z_map[12:20, 6:14] = 0.80
    z_map[12:20, 56:64] = 0.80
    z_map[13:19, 14:56] = np.nan

    result = group_components_by_foreground_islands(
        [component],
        z_map=z_map,
        table_z_m=0.74,
        min_island_area_px=8,
        min_foreground_fraction=0.35,
    )

    assert len(result.components) == 2
    assert result.decisions[0].action == "split"
    assert result.decisions[0].reason == "separate_sparse_foreground_islands"
    centers_u = sorted(float(item.centroid_uv[0]) for item in result.components)
    assert centers_u[0] < 16.0
    assert centers_u[1] > 52.0


def test_splits_visible_islands_connected_only_by_hole_bridge_even_when_not_sparse():
    mask = np.zeros((30, 62), dtype=bool)
    mask[10:20, 6:18] = True
    mask[10:20, 44:56] = True
    mask[9:21, 18:44] = True
    component = extract_components(mask, min_area_px=1)[0]
    z_map = np.full(mask.shape, 0.74, dtype=float)
    z_map[10:20, 6:18] = 0.80
    z_map[10:20, 44:56] = 0.80
    z_map[9:21, 18:44] = np.nan

    result = group_components_by_foreground_islands(
        [component],
        z_map=z_map,
        table_z_m=0.74,
        min_island_area_px=8,
        min_foreground_fraction=0.35,
    )

    assert len(result.components) == 2
    assert result.decisions[0].action == "split"
    assert result.decisions[0].reason == "hole_only_bridge_between_foreground_islands"
    widths = sorted(item.bbox_xyxy[2] - item.bbox_xyxy[0] + 1 for item in result.components)
    assert widths == [12, 12]


def test_keeps_complex_single_object_when_foreground_islands_have_broad_silhouette_neck():
    mask = np.zeros((28, 44), dtype=bool)
    mask[7:16, 5:14] = True
    mask[12:21, 29:38] = True
    mask[10:19, 14:29] = True
    component = extract_components(mask, min_area_px=1)[0]
    z_map = np.full(mask.shape, 0.74, dtype=float)
    z_map[7:16, 5:14] = 0.80
    z_map[12:21, 29:38] = 0.80
    z_map[10:19, 14:29] = np.nan

    result = group_components_by_foreground_islands(
        [component],
        z_map=z_map,
        table_z_m=0.74,
        min_island_area_px=8,
        min_foreground_fraction=0.25,
    )

    assert len(result.components) == 1
    assert not result.split_component_ids
    assert result.components[0].bbox_xyxy == component.bbox_xyxy
    assert result.decisions[0].action == "keep"
    assert result.decisions[0].reason == "continuous_silhouette_neck"
