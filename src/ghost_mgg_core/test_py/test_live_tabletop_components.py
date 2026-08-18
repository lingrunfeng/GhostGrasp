import numpy as np

from ghost_mgg_core_py.live_tabletop.components import extract_components


def test_extracts_two_tabletop_components_from_height_mask():
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:3, 1:4] = True
    mask[4:7, 6:9] = True

    components = extract_components(mask, min_area_px=2)

    assert len(components) == 2
    assert [component.area_px for component in components] == [9, 6]
    assert components[0].bbox_xyxy == (6, 4, 8, 6)
    assert components[1].bbox_xyxy == (1, 1, 3, 2)


def test_ignores_table_pixels_below_min_area():
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:3, 1:4] = True
    mask[6, 8] = True

    components = extract_components(mask, min_area_px=2)

    assert len(components) == 1
    assert components[0].area_px == 6
    assert components[0].bbox_xyxy == (1, 1, 3, 2)


def test_component_bbox_records_pixel_extent_and_area():
    mask = np.zeros((6, 7), dtype=bool)
    mask[2:5, 3:6] = True

    components = extract_components(mask, min_area_px=1)

    assert len(components) == 1
    component = components[0]
    assert component.component_id == 1
    assert component.area_px == 9
    assert component.bbox_xyxy == (3, 2, 5, 4)
    assert component.centroid_uv == (4.0, 3.0)
    assert component.mask.dtype == bool
    assert component.mask.shape == mask.shape
    assert int(np.count_nonzero(component.mask)) == 9


def test_extract_components_keeps_diagonal_pixels_separate():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True

    components = extract_components(mask, min_area_px=1)

    assert len(components) == 2
    assert [component.area_px for component in components] == [1, 1]
