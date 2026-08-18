cmake_minimum_required(VERSION 3.16)
project(ghost_mgg_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  ../../scripts/execute_m4_joint_hypothesis_action.py
  ../../scripts/publish_m4_color_target_mask.py
  ../../scripts/publish_m4_target_roi_mask.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_m2_sim_bt_loop_launch test/test_m2_sim_bt_loop_launch.py)
  ament_add_pytest_test(test_m2_mask_extrusion_launch test/test_m2_mask_extrusion_launch.py)
  ament_add_pytest_test(test_m2_moveit_rviz_launch test/test_m2_moveit_rviz_launch.py)
  ament_add_pytest_test(test_m2_gazebo_moveit_control_launch test/test_m2_gazebo_moveit_control_launch.py)
  ament_add_pytest_test(test_m2_gazebo_camera_inspect_launch test/test_m2_gazebo_camera_inspect_launch.py)
  ament_add_pytest_test(test_m2_moveit_execute_grasp_launch test/test_m2_moveit_execute_grasp_launch.py)
  ament_add_pytest_test(test_m3_failure_camera_inspect_launch test/test_m3_failure_camera_inspect_launch.py)
  ament_add_pytest_test(test_m4_sim_grasp_moveit_dryrun test/test_m4_sim_grasp_moveit_dryrun.py)
  ament_add_pytest_test(test_m4_joint_hypothesis_moveit_dryrun test/test_m4_joint_hypothesis_moveit_dryrun.py)
  ament_add_pytest_test(test_m4_joint_hypothesis_execute test/test_m4_joint_hypothesis_execute.py)
  ament_add_pytest_test(test_m4_dynamic_scene_execute test/test_m4_dynamic_scene_execute.py)
  ament_add_pytest_test(test_m4_no_truth_live_perception_launch test/test_m4_no_truth_live_perception_launch.py)
  ament_add_pytest_test(test_m4_no_truth_live_moveit test/test_m4_no_truth_live_moveit.py)
  ament_add_pytest_test(test_m4_no_truth_live_batch test/test_m4_no_truth_live_batch.py)
  ament_add_pytest_test(test_m4_no_truth_multi_target_eval test/test_m4_no_truth_multi_target_eval.py)
  ament_add_pytest_test(test_m4_no_truth_scenario_sweep test/test_m4_no_truth_scenario_sweep.py)
  ament_add_pytest_test(test_m4_no_truth_headless_suite test/test_m4_no_truth_headless_suite.py)
  ament_add_pytest_test(test_m4_no_truth_dynamic_rerun_eval test/test_m4_no_truth_dynamic_rerun_eval.py)
  ament_add_pytest_test(test_m4_no_truth_external_mask_dynamic_rerun test/test_m4_no_truth_external_mask_dynamic_rerun.py)
  ament_add_pytest_test(test_m4_perception_hypothesis_inspect_launch test/test_m4_perception_hypothesis_inspect_launch.py)
endif()

ament_package()
