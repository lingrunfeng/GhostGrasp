from __future__ import annotations

import numpy as np

from ghost_mgg_core_py.live_tabletop.types import PixelComponent

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only on minimal installs.
    cv2 = None


def extract_components(binary_mask: np.ndarray, min_area_px: int = 1) -> list[PixelComponent]:
    mask = np.asarray(binary_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("binary_mask must be a 2D array")
    min_area = int(min_area_px)
    if min_area < 1:
        raise ValueError("min_area_px must be at least 1")
    if not mask.any():
        return []
    if cv2 is not None:
        return _extract_components_cv2(mask, min_area)

    visited = np.zeros(mask.shape, dtype=bool)
    components: list[PixelComponent] = []
    next_component_id = 1
    rows, cols = mask.shape

    for row in range(rows):
        for col in range(cols):
            if not mask[row, col] or visited[row, col]:
                continue
            pixels = _collect_component(mask, visited, row, col)
            if len(pixels) >= min_area:
                components.append(_component_from_pixels(next_component_id, mask.shape, pixels))
            next_component_id += 1

    return sorted(components, key=lambda component: (-component.area_px, component.component_id))


def _extract_components_cv2(mask: np.ndarray, min_area_px: int) -> list[PixelComponent]:
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4
    )
    components: list[PixelComponent] = []
    for label_id in range(1, int(labels_count)):
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px < int(min_area_px):
            continue
        left = int(stats[label_id, cv2.CC_STAT_LEFT])
        top = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        component_mask = labels == label_id
        components.append(
            PixelComponent(
                component_id=int(label_id),
                area_px=area_px,
                bbox_xyxy=(
                    left,
                    top,
                    left + width - 1,
                    top + height - 1,
                ),
                centroid_uv=(
                    float(centroids[label_id, 0]),
                    float(centroids[label_id, 1]),
                ),
                mask=component_mask,
            )
        )
    return sorted(components, key=lambda component: (-component.area_px, component.component_id))


def _collect_component(
    mask: np.ndarray, visited: np.ndarray, start_row: int, start_col: int
) -> list[tuple[int, int]]:
    queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
    visited[start_row, start_col] = True
    pixels: list[tuple[int, int]] = []
    rows, cols = mask.shape

    while queue:
        row, col = queue.popleft()
        pixels.append((row, col))
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if (
                0 <= next_row < rows
                and 0 <= next_col < cols
                and mask[next_row, next_col]
                and not visited[next_row, next_col]
            ):
                visited[next_row, next_col] = True
                queue.append((next_row, next_col))

    return pixels


def _component_from_pixels(
    component_id: int, shape: tuple[int, int], pixels: list[tuple[int, int]]
) -> PixelComponent:
    rows = np.array([row for row, _ in pixels], dtype=int)
    cols = np.array([col for _, col in pixels], dtype=int)
    component_mask = np.zeros(shape, dtype=bool)
    component_mask[rows, cols] = True

    return PixelComponent(
        component_id=component_id,
        area_px=int(len(pixels)),
        bbox_xyxy=(
            int(np.min(cols)),
            int(np.min(rows)),
            int(np.max(cols)),
            int(np.max(rows)),
        ),
        centroid_uv=(float(np.mean(cols)), float(np.mean(rows))),
        mask=component_mask,
    )
