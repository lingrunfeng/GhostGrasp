# GHOST-MGG Real Hardware Bringup

This package contains real-device launch and inspection assets. It is intentionally
separate from `ghost_mgg_sim` so simulation defaults do not leak into hardware runs.

## D435 Quick Check

```bash
cd ~/Leorover-team8/Ghost-MGG
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch ghost_mgg_real d435_realsense.launch.py
```

In another terminal:

```bash
rviz2 -d install/ghost_mgg_real/share/ghost_mgg_real/rviz/d435_realsense.rviz
```

Or start the camera and RViz together:

```bash
ros2 launch ghost_mgg_real d435_realsense_inspect.launch.py
```

Expected topics:

```text
/camera/camera/color/image_raw
/camera/camera/depth/image_rect_raw
/camera/camera/infra1/image_rect_raw
/camera/camera/infra2/image_rect_raw
/camera/camera/depth/color/points
```

Expected frames:

```text
camera_link
camera_color_optical_frame
camera_depth_optical_frame
camera_infra1_optical_frame
camera_infra2_optical_frame
```

The wrapper's default namespace and camera name are both `camera`, so the topic
prefix is `/camera/camera/...`.

## M5 Bag Analysis

Raw bags live under `data/real_d435_m5/` and are intentionally ignored by git.
After collection, generate the small manifest and statistics reports with:

```bash
cd ~/Leorover-team8/Ghost-MGG
source /opt/ros/jazzy/setup.bash
source install/setup.bash

src/ghost_mgg_real/scripts/analyze_m5_real_d435_bags.py \
  --data-dir data/real_d435_m5 \
  --output-dir reports \
  --max-frames-per-topic 12
```

Outputs:

```text
reports/m5_real_d435_manifest.json
reports/m5_real_d435_manifest.csv
reports/m5_real_d435_frame_stats.csv
reports/m5_real_d435_summary.md
```

To extract one replay-check frame bundle per bag:

```bash
src/ghost_mgg_real/scripts/extract_m5_replay_samples.py \
  --data-dir data/real_d435_m5 \
  --output-dir reports/m5_real_d435_replay_samples
```

Each scene directory contains:

```text
color.png
depth_viz.png
aligned_depth_viz.png
infra1.png
infra2.png
metadata.json
```

To generate first-pass real failure-evidence previews using `empty_table_001` as
the background reference:

```bash
src/ghost_mgg_real/scripts/generate_m5_real_evidence_previews.py \
  --data-dir data/real_d435_m5 \
  --output-dir reports/m5_real_d435_evidence_preview \
  --background-scene-id empty_table_001
```

These outputs are coarse background-difference previews for engineering review,
not final paper labels.

## M5 Target Mask Annotation

Generate editable annotation task JSON files from the replay samples and coarse
evidence previews:

```bash
src/ghost_mgg_real/scripts/prepare_m5_mask_annotation_tasks.py \
  --replay-manifest reports/m5_real_d435_replay_samples/manifest.json \
  --evidence-manifest reports/m5_real_d435_evidence_preview/manifest.json \
  --annotations-root annotations/m5_real_d435_masks
```

Each task lives at:

```text
annotations/m5_real_d435_masks/tasks/<scene_id>.json
```

To finish one scene, edit its task JSON:

```json
"status": "complete",
"polygons": [
  [[120, 80], [160, 82], [158, 130], [118, 128]]
]
```

Then rasterize completed tasks:

```bash
src/ghost_mgg_real/scripts/rasterize_m5_mask_annotations.py \
  --annotations-root annotations/m5_real_d435_masks
```

Completed scenes produce:

```text
annotations/m5_real_d435_masks/masks/<scene_id>/target_mask.png
annotations/m5_real_d435_masks/masks/<scene_id>/annotation_overlay.png
annotations/m5_real_d435_masks/masks/<scene_id>/mask_summary.json
```

Pending scenes are skipped and listed in
`annotations/m5_real_d435_masks/masks/manifest.json`.

For interactive click annotation, run one scene at a time:

```bash
src/ghost_mgg_real/scripts/annotate_m5_mask.py \
  --scene-id daylight_transparent_jelly_cup_001
```

Controls:

```text
left click   add polygon point
right click  undo last point
u            undo last point
r            reset polygon
s            save task JSON and generate mask/overlay
q or Esc      quit without saving
```

The first five scenes to annotate are:

```text
daylight_transparent_jelly_cup_001
daylight_transparent_jelly_cup_yaw45_001
daylight_transparent_jelly_cup_visible_points_001
daylight_frosted_plastic_bowl_001
daylight_opaque_box_001
```

If RGB is hard to see, switch the display image while saving the same task:

```bash
src/ghost_mgg_real/scripts/annotate_m5_mask.py \
  --scene-id daylight_transparent_jelly_cup_001 \
  --image-key infra1
```

After completing masks, generate the formal-mask evidence report:

```bash
src/ghost_mgg_real/scripts/generate_m5_masked_evidence_report.py \
  --data-dir data/real_d435_m5 \
  --annotations-root annotations/m5_real_d435_masks \
  --output-dir reports/m5_real_d435_masked_evidence \
  --background-scene-id empty_table_001
```

The report compares each target mask against the empty-table background depth and
stores:

```text
reports/m5_real_d435_masked_evidence/index.md
reports/m5_real_d435_masked_evidence/summary.csv
reports/m5_real_d435_masked_evidence/<scene_id>/evidence_overlay.png
```

To run the first real-data primitive ranking comparison:

```bash
src/ghost_mgg_real/scripts/generate_m5_real_ranking_report.py \
  --data-dir data/real_d435_m5 \
  --annotations-root annotations/m5_real_d435_masks \
  --output-dir reports/m5_real_d435_ranking \
  --background-scene-id empty_table_001 \
  --top-k 3 \
  --failure-aware-weights-json reports/m4_real_weight_calibration/best_weights.json
```

This compares `silhouette_only` and `failure_aware` ranking for completed masks
and writes:

```text
reports/m5_real_d435_ranking/index.md
reports/m5_real_d435_ranking/m5_real_ranking.csv
reports/m5_real_d435_ranking/m5_real_ranking.json
reports/m5_real_d435_ranking/top1_comparison.csv
reports/m5_real_d435_ranking/top1_comparison.json
```

`top1_comparison.*` is the quickest engineering readout: one row per annotated
real scene comparing the silhouette-only top hypothesis against the
failure-aware top hypothesis, including whether the top-1 changed and the
failure/visual score deltas.

To run the conservative weak-GT sanity check for those real ranking results:

```bash
src/ghost_mgg_real/scripts/generate_m5_real_weak_gt_eval.py \
  --weak-gt annotations/m5_real_d435_weak_gt/weak_gt.json \
  --top1-comparison-csv reports/m5_real_d435_ranking/top1_comparison.csv \
  --evidence-summary-csv reports/m5_real_d435_masked_evidence/summary.csv \
  --output-dir reports/m5_real_d435_weak_gt_eval
```

This report checks only weak proxy-level expectations: accepted primitive
families, failure-score gain for transparent/translucent scenes, visual-score
drop bounds, and shape stability for the opaque box. It is not metric 3D ground
truth.

To build a compact M4 dashboard from the three real-data reports:

```bash
src/ghost_mgg_real/scripts/generate_m4_real_dashboard.py \
  --evidence-summary-csv reports/m5_real_d435_masked_evidence/summary.csv \
  --top1-comparison-csv reports/m5_real_d435_ranking/top1_comparison.csv \
  --weak-gt-eval-csv reports/m5_real_d435_weak_gt_eval/weak_gt_eval.csv \
  --output-dir reports/m4_real_dashboard
```

The dashboard entrypoint is:

```text
reports/m4_real_dashboard/index.md
```

To run the M4.1 real-data weight calibration over the existing candidate score
decomposition:

```bash
src/ghost_mgg_real/scripts/generate_m4_real_weight_calibration.py \
  --ranking-csv reports/m5_real_d435_ranking/m5_real_ranking.csv \
  --weak-gt annotations/m5_real_d435_weak_gt/weak_gt.json \
  --output-dir reports/m4_real_weight_calibration
```

This report searches a small interpretable grid over visual, failure, and depth
weights. It is a calibration sanity check, not final paper training.

The calibration entrypoint is:

```text
reports/m4_real_weight_calibration/index.md
```

To generate visual ranking boards for direct inspection:

```bash
src/ghost_mgg_real/scripts/generate_m4_visual_ranking_board.py \
  --replay-samples-dir reports/m5_real_d435_replay_samples \
  --masks-root annotations/m5_real_d435_masks/masks \
  --evidence-dir reports/m5_real_d435_masked_evidence \
  --ranking-csv reports/m5_real_d435_ranking/m5_real_ranking.csv \
  --dashboard-csv reports/m4_real_dashboard/dashboard.csv \
  --output-dir reports/m4_visual_ranking_board
```

Open the board index or a single scene image:

```bash
xdg-open reports/m4_visual_ranking_board/index.md
xdg-open reports/m4_visual_ranking_board/daylight_transparent_jelly_cup_001.png
```

To build one compact M4 readiness gate summary from the generated reports:

```bash
src/ghost_mgg_real/scripts/generate_m4_gate_summary.py \
  --output-dir reports/m4_gate_summary
```

The gate-summary entrypoint is:

```text
reports/m4_gate_summary/index.md
```
