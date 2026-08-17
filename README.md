# GhostGrasp

### Grasping Transparent Objects in Darkness from Depth-Failure Evidence

**Runfeng Ling — MSc Robotics, The University of Manchester**

[English](README.md) | [简体中文](README.zh-CN.md)

[Watch on YouTube](https://www.youtube.com/watch?v=1ZADqmrzXGo) · [Watch on Bilibili](https://www.bilibili.com/video/BV1yHbq69ERM)

> **Release status:** this repository currently hosts the GhostGrasp project page and visual material. The paper, source code, trained models, configurations, and evaluation tools will be added in a later release.

![GhostGrasp overview: transparent-object depth failures remain visible to active infrared sensing and support top grasping in daylight and darkness.](assets/ghostgrasp_overview.png)

*GhostGrasp treats structured RGB-D sensing failures as evidence. When visible RGB disappears, active infrared sensing still supports persistent object hypotheses and the same top-grasp task.*

## Overview

Transparent objects often appear to an RGB-D camera as missing depth, isolated returns, or measurements that pass through the object and reach the table. Surface-completion methods attempt to repair these measurements before grasp planning. GhostGrasp takes the opposite approach: it keeps the failures and uses their spatial and temporal structure as evidence that an object is present.

The system integrates depth-failure evidence on a table-plane grid, maintains object hypotheses over time, and converts each stable hypothesis into an action-oriented proxy containing a footprint, centre, opening width, yaw candidates, and grasp height. An RGB or raw-IR detector can locate and update a hypothesis, but it cannot create one without supporting depth evidence.

## Key results

The final system was evaluated with a fixed Intel RealSense D435, a myCobot 280, and CPU-only perception.

| Measure | Daylight | Darkness |
|---|---:|---:|
| Maintained-hypothesis frame coverage | 100% | 96.9% |
| Transparent-object grasp success | 34/50 (68%) | 35/50 (70%) |
| Perception-core runtime | 52 ms/frame | 88 ms/frame |

- **150 real-robot trials:** 100 transparent-object trials and 50 opaque controls.
- **Zero position failures** across all 100 transparent-object trials; recorded failures occurred after contact, mainly through low-friction slip or gripper–shape mismatch.
- **CPU-only deployment:** transparent-specific processing takes 8–11 ms, with the instance detector running asynchronously at 6 Hz.
- **Failure evidence is essential:** removing the hole channel reduces coverage to 0–2% on the recorded stress cases.

These figures describe the tested fixed-view tabletop setup; they are not claims about arbitrary cameras, scenes, or grasp types.

## Method

![GhostGrasp pipeline from RGB, raw infrared, and failed depth to persistent hypotheses and a top-grasp proxy.](assets/ghostgrasp_pipeline.png)

The pipeline has four main stages:

1. **Depth-failure evidence** — holes, isolated/specular returns, and below-table refraction vote for object existence on a 4 mm table-plane grid.
2. **Illumination-aware instance cues** — the detector uses RGB in normal light and the raw active-IR dot pattern in darkness. Detections provide identity and position, not existence by themselves.
3. **Persistent hypothesis lifecycle** — temporal accumulation, hysteresis, merging, shadow-cone deduplication, and retirement maintain stable object hypotheses through missed detections and lighting changes.
4. **Action-oriented proxy** — each anchored hypothesis exposes the quantities required by the tested top-grasp executor without reconstructing a complete surface.

## RViz visualisation and digital twin

The following RViz views combine the registered point cloud, object proxies, coordinate frames, camera/IR panels, and the myCobot model. They provide a direct visual bridge between the sensor observations, the maintained hypotheses, and the target sent to the robot. The same ROS 2 interfaces also support simulation-based testing of the recovery and execution logic.

### Multi-object perception

| Daylight | Darkness |
|---|---|
| ![RViz daylight scene with four maintained object proxies, active-IR view, and RGB detections.](assets/rviz_multi_object_daylight.png) | ![RViz dark scene with four maintained object proxies and raw active-IR detections.](assets/rviz_multi_object_darkness.png) |
| RGB supplies the daylight instance cue while the point cloud and active-IR stream remain visible. | Visible appearance is removed; raw active IR and accumulated depth-failure evidence maintain the four proxies. |

### Grasp execution views

| Daylight grasp | Dark grasp |
|---|---|
| ![RViz and physical daylight grasp view with the robot model aligned to the registered point cloud.](assets/rviz_grasp_daylight.png) | ![RViz and physical dark grasp view with the robot model, active-IR panel, and maintained transparent-object target.](assets/rviz_grasp_darkness.png) |
| The physical camera inset and RViz model show the opaque-control grasp in a common robot frame. | The dark scene uses the raw-IR cue to maintain the transparent target while the same robot-side grasp path is executed. |

The coloured boxes are action proxies rather than reconstructed object meshes. Their centres and dimensions are the quantities passed to the downstream grasp interface.

### Assorted tabletop clutter

![Assorted tabletop objects in daylight and darkness, with the corresponding RViz point cloud and maintained action proxies.](assets/tabletop_clutter_daylight_darkness.jpg)

This qualitative view shows assorted tabletop objects under room light (left) and with the lights off (right). The lower panels show the corresponding RViz point cloud and maintained proxies. It illustrates how the same visualisation remains readable as RGB appearance disappears; this scene is supporting system evidence and is separate from the reported transparent-object grasp totals.

### Robot-model verification

<p align="center">
  <img src="assets/rviz_robot_model.png" width="82%" alt="myCobot 280 URDF and gripper model displayed in RViz with joint controls and coordinate frames.">
</p>

The myCobot 280 URDF, gripper joints, and TF frames were checked in RViz before connecting perception targets to robot-side planning and execution. This model provides the digital counterpart used by the simulation and plan-only checks.

## Robot integration

![ROS 2 architecture showing the deployed perception-to-motion path and supporting calibration, visualisation, logging, and safety interfaces.](assets/ros2_system_architecture.png)

The deployed real-robot path was:

```text
geometry hypothesis → bridge → guarded taught-grasp service → myCobot
```

The perception output is carried through a standard ROS 2 hypothesis interface. A guarded executor checks target age, motion enablement, grasp-height limits, and low-speed execution before commanding the robot. This path was used for all 150 reported trials.

## Demonstration videos

- [YouTube — GhostGrasp demonstration](https://www.youtube.com/watch?v=1ZADqmrzXGo)
- [Bilibili — GhostGrasp demonstration](https://www.bilibili.com/video/BV1yHbq69ERM)

The videos show daylight and dark grasping, active-IR observations, maintained hypotheses, light transitions, multi-object perception, and representative failures.

## Current scope

GhostGrasp currently targets fixed-view, indoor tabletop perception and single-target top grasping. The deployed executor uses nominal object height for automatic grasp height. Moving cameras, side grasps, and dense clutter are outside the reported evaluation.

Planned extensions include executing the predicted yaw and gripper opening, adding contact feedback to reduce slip, evaluating more active-stereo cameras and scenes, and estimating height from multiple views when nominal metadata are unavailable.

## Repository status

This public folder intentionally does **not** contain the complete implementation yet. The planned release package includes:

- ROS 2 perception, interface, bridge, and guarded-execution packages;
- evidence-field and hypothesis-lifecycle implementation;
- launch files and frozen deployment configurations;
- trained RGB/raw-IR detector weights;
- replay, ablation, runtime, and evaluation scripts;
- documentation for reproducing the fixed-view tabletop setup.

Release instructions and software licensing will be added together with the code.

## Citation

The paper citation will be added when the manuscript is publicly available. Until then, the project may be referenced as:

```bibtex
@mastersthesis{ling2026ghostgrasp,
  author = {Runfeng Ling},
  title  = {GhostGrasp: Grasping Transparent Objects in Darkness from Depth-Failure Evidence},
  school = {The University of Manchester},
  type   = {MSc dissertation},
  year   = {2026}
}
```

## Contact

Please use the GitHub issue tracker for project questions after the repository is published.
