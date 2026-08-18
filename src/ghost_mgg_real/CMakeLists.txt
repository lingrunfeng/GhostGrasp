cmake_minimum_required(VERSION 3.16)
project(ghost_mgg_real)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch rviz scripts
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/annotate_direct_mask.py
    scripts/capture_m6_shadow_observation.py
    scripts/generate_m6_shadow_decision.py
    scripts/generate_m6_shadow_freeze_report.py
    scripts/generate_m6_shadow_grasp_target.py
    scripts/m6_ssh_joint_state_bridge.py
    scripts/m7_mycobot_control_server.py
    scripts/m7_real_grasp_marker_node.py
    scripts/generate_m7_safety_preflight.py
    scripts/generate_m7_official_top_grasp_targets.py
    scripts/run_m7_low_amplitude_motion_test.py
    scripts/run_m7_target_based_executor.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_d435_realsense_launch test/test_d435_realsense_launch.py)
  ament_add_pytest_test(test_m5_real_d435_analysis test/test_m5_real_d435_analysis.py)
  ament_add_pytest_test(test_m5_replay_sample_extractor test/test_m5_replay_sample_extractor.py)
  ament_add_pytest_test(test_m5_real_evidence_preview test/test_m5_real_evidence_preview.py)
  ament_add_pytest_test(test_m5_mask_annotation_tasks test/test_m5_mask_annotation_tasks.py)
  ament_add_pytest_test(test_m5_mask_rasterizer test/test_m5_mask_rasterizer.py)
  ament_add_pytest_test(test_m5_click_annotator test/test_m5_click_annotator.py)
  ament_add_pytest_test(test_m5_masked_evidence_report test/test_m5_masked_evidence_report.py)
  ament_add_pytest_test(test_m5_real_ranking_report test/test_m5_real_ranking_report.py)
  ament_add_pytest_test(test_m5_real_weak_gt_eval test/test_m5_real_weak_gt_eval.py)
  ament_add_pytest_test(test_m5_5_real_online_bridge test/test_m5_5_real_online_bridge.py)
  ament_add_pytest_test(test_m4_real_dashboard test/test_m4_real_dashboard.py)
  ament_add_pytest_test(test_m4_visual_ranking_board test/test_m4_visual_ranking_board.py)
  ament_add_pytest_test(test_m4_real_weight_calibration test/test_m4_real_weight_calibration.py)
  ament_add_pytest_test(test_m4_gate_summary test/test_m4_gate_summary.py)
  ament_add_pytest_test(test_m4_metric_proxies test/test_m4_metric_proxies.py)
  ament_add_pytest_test(test_m4_graspability_dryrun test/test_m4_graspability_dryrun.py)
  ament_add_pytest_test(test_m4_real_external_mask_replay test/test_m4_real_external_mask_replay.py)
  ament_add_pytest_test(test_m4_real_external_mask_visual_dashboard test/test_m4_real_external_mask_visual_dashboard.py)
  ament_add_pytest_test(test_m4_real_algorithm_gate test/test_m4_real_algorithm_gate.py)
  ament_add_pytest_test(test_m4_real_ablation_gate test/test_m4_real_ablation_gate.py)
  ament_add_pytest_test(test_m4_failure_atlas test/test_m4_failure_atlas.py)
  ament_add_pytest_test(test_m4_paper_evidence_pack test/test_m4_paper_evidence_pack.py)
  ament_add_pytest_test(test_m6_mycobot_state_bridge test/test_m6_mycobot_state_bridge.py)
  ament_add_pytest_test(test_m6_rough_tf_shadow_inspect test/test_m6_rough_tf_shadow_inspect.py)
  ament_add_pytest_test(test_m6_shadow_observation_capture test/test_m6_shadow_observation_capture.py)
  ament_add_pytest_test(test_m6_shadow_decision test/test_m6_shadow_decision.py)
  ament_add_pytest_test(test_m6_shadow_grasp_target test/test_m6_shadow_grasp_target.py)
  ament_add_pytest_test(test_m6_shadow_moveit_plan_only test/test_m6_shadow_moveit_plan_only.py)
  ament_add_pytest_test(test_m6_freeze_m7_safety test/test_m6_freeze_m7_safety.py)
  ament_add_pytest_test(test_m7_official_top_grasp_moveit test/test_m7_official_top_grasp_moveit.py)
  ament_add_pytest_test(test_m7_low_amplitude_motion test/test_m7_low_amplitude_motion.py)
  ament_add_pytest_test(test_m7_target_based_executor test/test_m7_target_based_executor.py)
  ament_add_pytest_test(test_m7_mycobot_control_server test/test_m7_mycobot_control_server.py)
  ament_add_pytest_test(test_m7_real_grasp_marker_node test/test_m7_real_grasp_marker_node.py)
  ament_add_pytest_test(test_m7_refresh_grasp_trial test/test_m7_refresh_grasp_trial.py)
  ament_add_pytest_test(test_m8_real_shadow_inspect test/test_m8_real_shadow_inspect.py)
endif()

ament_package()
