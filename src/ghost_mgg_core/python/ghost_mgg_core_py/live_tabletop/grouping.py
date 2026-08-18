from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ghost_mgg_core_py.live_tabletop.components import extract_components
from ghost_mgg_core_py.live_tabletop.types import PixelComponent


@dataclass(frozen=True)
class ComponentGroupingDecision:
    original_component_id: int
    output_component_ids: tuple[int, ...]
    action: str
    reason: str


@dataclass(frozen=True)
class ComponentGroupingResult:
    components: list[PixelComponent]
    split_component_ids: set[int]
    reason_by_component_id: dict[int, str]
    decisions: list[ComponentGroupingDecision]


def group_components_by_foreground_islands(
    components: list[PixelComponent],
    *,
    z_map: np.ndarray,
    table_z_m: float,
    min_foreground_height_m: float = 0.006,
    min_island_area_px: int = 12,
    min_foreground_fraction: float = 0.35,
    min_centroid_distance_px: float = 8.0,
    min_neck_width_px: float = 4.0,
    min_neck_ratio: float = 0.65,
) -> ComponentGroupingResult:
    z_values = np.asarray(z_map, dtype=float)
    finite_z = np.isfinite(z_values)
    foreground_z_min = float(table_z_m) + float(min_foreground_height_m)

    grouped_components: list[PixelComponent] = []
    split_component_ids: set[int] = set()
    reason_by_component_id: dict[int, str] = {}
    decisions: list[ComponentGroupingDecision] = []
    next_component_id = 1

    for component in components:
        component_mask = np.asarray(component.mask, dtype=bool)
        foreground = component_mask & finite_z & (z_values > foreground_z_min)
        islands = extract_components(foreground, min_area_px=int(min_island_area_px))
        weak_neck_split_masks: list[np.ndarray] = []
        if len(islands) < 2:
            weak_neck_split_masks = split_single_foreground_island_by_weak_neck(
                foreground,
                component_mask=component_mask,
                finite_z=finite_z,
                min_island_area_px=int(min_island_area_px),
            )
            if len(weak_neck_split_masks) >= 2:
                should_split = True
                reason = "visible_weak_neck_ownership"
            else:
                should_split, reason = should_split_component_from_foreground_islands(
                    component,
                    islands,
                    min_foreground_fraction=float(min_foreground_fraction),
                    min_centroid_distance_px=float(min_centroid_distance_px),
                    min_neck_width_px=float(min_neck_width_px),
                    min_neck_ratio=float(min_neck_ratio),
                )
        else:
            should_split, reason = should_split_component_from_foreground_islands(
                component,
                islands,
                min_foreground_fraction=float(min_foreground_fraction),
                min_centroid_distance_px=float(min_centroid_distance_px),
                min_neck_width_px=float(min_neck_width_px),
                min_neck_ratio=float(min_neck_ratio),
            )

        if should_split:
            split_masks = weak_neck_split_masks or [
                np.asarray(island.mask, dtype=bool).copy() for island in islands
            ]
            valid_split_masks = [
                item for item in split_masks if int(np.count_nonzero(item)) >= int(min_island_area_px)
            ]
        else:
            valid_split_masks = []

        if len(valid_split_masks) < 2:
            rebuilt = pixel_component_from_mask(next_component_id, component_mask)
            if rebuilt is None:
                decisions.append(
                    ComponentGroupingDecision(
                        original_component_id=int(component.component_id),
                        output_component_ids=(),
                        action="drop",
                        reason="empty_component",
                    )
                )
                continue
            grouped_components.append(rebuilt)
            reason_by_component_id[int(rebuilt.component_id)] = reason
            decisions.append(
                ComponentGroupingDecision(
                    original_component_id=int(component.component_id),
                    output_component_ids=(int(rebuilt.component_id),),
                    action="keep",
                    reason=reason,
                )
            )
            next_component_id += 1
            continue

        output_ids: list[int] = []
        for split_mask in valid_split_masks:
            rebuilt = pixel_component_from_mask(next_component_id, split_mask)
            if rebuilt is None:
                continue
            grouped_components.append(rebuilt)
            split_component_ids.add(int(rebuilt.component_id))
            reason_by_component_id[int(rebuilt.component_id)] = reason
            output_ids.append(int(rebuilt.component_id))
            next_component_id += 1

        decisions.append(
            ComponentGroupingDecision(
                original_component_id=int(component.component_id),
                output_component_ids=tuple(output_ids),
                action="split",
                reason=reason,
            )
        )

    return ComponentGroupingResult(
        components=grouped_components,
        split_component_ids=split_component_ids,
        reason_by_component_id=reason_by_component_id,
        decisions=decisions,
    )


def split_single_foreground_island_by_weak_neck(
    foreground_mask: np.ndarray,
    *,
    component_mask: np.ndarray,
    finite_z: np.ndarray,
    min_island_area_px: int,
    min_hole_fraction: float = 0.03,
) -> list[np.ndarray]:
    foreground = np.asarray(foreground_mask, dtype=bool)
    component = np.asarray(component_mask, dtype=bool)
    finite = np.asarray(finite_z, dtype=bool)
    if foreground.shape != component.shape or finite.shape != component.shape:
        raise ValueError("foreground_mask, component_mask, and finite_z must have the same shape")
    if int(np.count_nonzero(foreground)) < int(min_island_area_px) * 2:
        return []

    hole_fraction = float(np.count_nonzero(component & ~finite)) / float(max(1, np.count_nonzero(component)))
    if hole_fraction < float(min_hole_fraction):
        return []

    for radius_px in (2, 3):
        opened = _binary_dilate(_binary_erode(foreground, radius_px), radius_px)
        seed_components = extract_components(opened, min_area_px=int(min_island_area_px))
        if len(seed_components) < 2:
            continue
        if should_accept_weak_neck_seeds(
            foreground,
            seed_components,
            min_island_area_px=int(min_island_area_px),
        ):
            return [np.asarray(seed.mask, dtype=bool).copy() for seed in seed_components]
    return []


def should_accept_weak_neck_seeds(
    foreground: np.ndarray,
    seed_components: list[PixelComponent],
    *,
    min_island_area_px: int,
    min_centroid_distance_px: float = 8.0,
    max_largest_fraction: float = 0.88,
) -> bool:
    seeds = [component for component in seed_components if int(component.area_px) >= int(min_island_area_px)]
    if len(seeds) < 2:
        return False
    total_area = sum(int(component.area_px) for component in seeds)
    if total_area <= 0:
        return False
    largest_fraction = max(int(component.area_px) for component in seeds) / float(total_area)
    if largest_fraction > float(max_largest_fraction):
        return False
    pair = farthest_foreground_island_pair(seeds)
    if pair is None:
        return False
    first, second, distance_px = pair
    if float(distance_px) < float(min_centroid_distance_px):
        return False
    neck_width_px = silhouette_neck_width_px(foreground, first.centroid_uv, second.centroid_uv)
    smaller_width_px = math.sqrt(float(max(1, min(first.area_px, second.area_px))))
    return neck_width_px <= max(4.0, 0.85 * smaller_width_px)


def _binary_erode(mask: np.ndarray, radius_px: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    radius = int(radius_px)
    if radius <= 0:
        return binary.copy()
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    eroded = np.ones_like(binary, dtype=bool)
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            eroded &= padded[
                radius + row_offset : radius + row_offset + binary.shape[0],
                radius + col_offset : radius + col_offset + binary.shape[1],
            ]
    return eroded


def _binary_dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    radius = int(radius_px)
    if radius <= 0:
        return binary.copy()
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(binary, dtype=bool)
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            dilated |= padded[
                radius + row_offset : radius + row_offset + binary.shape[0],
                radius + col_offset : radius + col_offset + binary.shape[1],
            ]
    return dilated


def pixel_component_from_mask(component_id: int, mask: np.ndarray) -> PixelComponent | None:
    component_mask = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(component_mask)
    if rows.size == 0:
        return None
    return PixelComponent(
        component_id=int(component_id),
        area_px=int(rows.size),
        bbox_xyxy=(
            int(np.min(cols)),
            int(np.min(rows)),
            int(np.max(cols)),
            int(np.max(rows)),
        ),
        centroid_uv=(float(np.mean(cols)), float(np.mean(rows))),
        mask=component_mask,
    )


def should_split_component_from_foreground_islands(
    component: PixelComponent,
    islands: list[PixelComponent],
    *,
    min_foreground_fraction: float,
    min_centroid_distance_px: float = 8.0,
    min_neck_width_px: float = 4.0,
    min_neck_ratio: float = 0.65,
) -> tuple[bool, str]:
    if len(islands) < 2:
        return False, "single_foreground_island"

    foreground_area = sum(int(island.area_px) for island in islands)
    foreground_fraction = foreground_area / float(max(1, int(component.area_px)))
    sparse_foreground = foreground_fraction < float(min_foreground_fraction)

    largest_fraction = max(int(island.area_px) for island in islands) / float(
        max(1, foreground_area)
    )
    if largest_fraction > 0.88:
        return False, "dominant_foreground_island"

    pair = farthest_foreground_island_pair(islands)
    if pair is None:
        return False, "single_foreground_island"
    first, second, centroid_distance_px = pair
    if centroid_distance_px < float(min_centroid_distance_px):
        return False, "close_foreground_islands"

    neck_width_px = silhouette_neck_width_px(component.mask, first.centroid_uv, second.centroid_uv)
    smaller_island_width_px = math.sqrt(float(max(1, min(first.area_px, second.area_px))))
    required_neck_px = max(
        float(min_neck_width_px),
        float(min_neck_ratio) * smaller_island_width_px,
    )
    if neck_width_px >= required_neck_px:
        foreground_union = np.zeros_like(np.asarray(component.mask, dtype=bool))
        for island in islands:
            foreground_union |= np.asarray(island.mask, dtype=bool)
        foreground_neck_width_px = silhouette_neck_width_px(
            foreground_union,
            first.centroid_uv,
            second.centroid_uv,
        )
        foreground_bridge_missing = foreground_neck_width_px < max(
            1.0,
            0.35 * required_neck_px,
        )
        if sparse_foreground:
            if foreground_bridge_missing:
                return True, "separate_sparse_foreground_islands"
        elif foreground_bridge_missing and foreground_fraction < 0.50:
            return True, "hole_only_bridge_between_foreground_islands"
        return False, "continuous_silhouette_neck"

    if sparse_foreground:
        return True, "separate_sparse_foreground_islands"
    return True, "separate_foreground_islands"


def farthest_foreground_island_pair(
    islands: list[PixelComponent],
) -> tuple[PixelComponent, PixelComponent, float] | None:
    if len(islands) < 2:
        return None
    centroids = np.asarray([island.centroid_uv for island in islands], dtype=float)
    distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    first_index, second_index = np.unravel_index(int(np.argmax(distances)), distances.shape)
    if first_index == second_index:
        return None
    return (
        islands[int(first_index)],
        islands[int(second_index)],
        float(distances[first_index, second_index]),
    )


def silhouette_neck_width_px(
    component_mask: np.ndarray,
    first_centroid_uv: tuple[float, float],
    second_centroid_uv: tuple[float, float],
) -> float:
    rows, cols = np.nonzero(np.asarray(component_mask, dtype=bool))
    if rows.size == 0:
        return 0.0

    first = np.asarray(first_centroid_uv, dtype=np.float64)
    second = np.asarray(second_centroid_uv, dtype=np.float64)
    axis = second - first
    distance = float(np.linalg.norm(axis))
    if distance < 1e-6:
        return float("inf")

    axis /= distance
    lateral_axis = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    pixels = np.column_stack([cols.astype(np.float64), rows.astype(np.float64)])
    relative = pixels - first
    longitudinal = relative @ axis
    lateral = relative @ lateral_axis

    central = (longitudinal >= 0.25 * distance) & (longitudinal <= 0.75 * distance)
    if not np.any(central):
        return 0.0

    central_longitudinal = longitudinal[central]
    central_lateral = lateral[central]
    bin_edges = np.linspace(0.25 * distance, 0.75 * distance, num=6)
    widths: list[float] = []
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (central_longitudinal >= start) & (central_longitudinal <= end)
        if not np.any(in_bin):
            widths.append(0.0)
            continue
        values = central_lateral[in_bin]
        widths.append(float(values.max() - values.min() + 1.0))

    return float(min(widths)) if widths else 0.0
