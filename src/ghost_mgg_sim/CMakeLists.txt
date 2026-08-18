cmake_minimum_required(VERSION 3.16)
project(ghost_mgg_sim)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ghost_mgg_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(m2_scene_markers_node src/m2_scene_markers_node.cpp)
target_include_directories(m2_scene_markers_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(m2_scene_markers_node PUBLIC cxx_std_17)
ament_target_dependencies(
  m2_scene_markers_node
  geometry_msgs
  rclcpp
  tf2_ros
  visualization_msgs
)

add_library(depth_to_point_cloud src/depth_to_point_cloud.cpp)
target_include_directories(depth_to_point_cloud PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(depth_to_point_cloud PUBLIC cxx_std_17)
ament_target_dependencies(depth_to_point_cloud sensor_msgs)

add_library(depth_to_mono8 src/depth_to_mono8.cpp)
target_include_directories(depth_to_mono8 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(depth_to_mono8 PUBLIC cxx_std_17)
ament_target_dependencies(depth_to_mono8 sensor_msgs)

add_library(depth_failure_injector src/depth_failure_injector.cpp)
target_include_directories(depth_failure_injector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(depth_failure_injector PUBLIC cxx_std_17)
ament_target_dependencies(depth_failure_injector sensor_msgs)

add_executable(depth_to_point_cloud_node src/depth_to_point_cloud_node.cpp)
target_include_directories(depth_to_point_cloud_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(depth_to_point_cloud_node PUBLIC cxx_std_17)
target_link_libraries(depth_to_point_cloud_node depth_to_point_cloud)
ament_target_dependencies(depth_to_point_cloud_node rclcpp sensor_msgs)

add_executable(depth_to_mono8_node src/depth_to_mono8_node.cpp)
target_include_directories(depth_to_mono8_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(depth_to_mono8_node PUBLIC cxx_std_17)
target_link_libraries(depth_to_mono8_node depth_to_mono8)
ament_target_dependencies(depth_to_mono8_node rclcpp sensor_msgs)

add_executable(depth_failure_injector_node src/depth_failure_injector_node.cpp)
target_include_directories(depth_failure_injector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(depth_failure_injector_node PUBLIC cxx_std_17)
target_link_libraries(depth_failure_injector_node depth_failure_injector)
ament_target_dependencies(depth_failure_injector_node rclcpp sensor_msgs std_msgs)

add_executable(target_mask_emulator_node src/target_mask_emulator_node.cpp)
target_compile_features(target_mask_emulator_node PUBLIC cxx_std_17)
ament_target_dependencies(target_mask_emulator_node rclcpp sensor_msgs)

add_executable(evidence_summary_logger_node src/evidence_summary_logger_node.cpp)
target_compile_features(evidence_summary_logger_node PUBLIC cxx_std_17)
ament_target_dependencies(evidence_summary_logger_node rclcpp std_msgs)

add_executable(robot_description_publisher_node src/robot_description_publisher_node.cpp)
target_compile_features(robot_description_publisher_node PUBLIC cxx_std_17)
ament_target_dependencies(robot_description_publisher_node rclcpp std_msgs)

add_executable(observation_cache_node src/observation_cache_node.cpp)
target_compile_features(observation_cache_node PUBLIC cxx_std_17)
ament_target_dependencies(
  observation_cache_node
  ghost_mgg_interfaces
  rclcpp
  sensor_msgs
)

add_executable(hypothesis_markers_node src/hypothesis_markers_node.cpp)
target_compile_features(hypothesis_markers_node PUBLIC cxx_std_17)
ament_target_dependencies(
  hypothesis_markers_node
  geometry_msgs
  ghost_mgg_interfaces
  rclcpp
  std_msgs
  visualization_msgs
)

add_executable(m4_offline_ranking_markers_node src/m4_offline_ranking_markers_node.cpp)
target_compile_features(m4_offline_ranking_markers_node PUBLIC cxx_std_17)
ament_target_dependencies(
  m4_offline_ranking_markers_node
  geometry_msgs
  rclcpp
  visualization_msgs
)

add_executable(m4_real_ranking_markers_node src/m4_real_ranking_markers_node.cpp)
target_compile_features(m4_real_ranking_markers_node PUBLIC cxx_std_17)
ament_target_dependencies(
  m4_real_ranking_markers_node
  geometry_msgs
  rclcpp
  visualization_msgs
)

add_executable(m4_graspability_markers_node src/m4_graspability_markers_node.cpp)
target_compile_features(m4_graspability_markers_node PUBLIC cxx_std_17)
ament_target_dependencies(
  m4_graspability_markers_node
  geometry_msgs
  rclcpp
  visualization_msgs
)

add_executable(m4_sim_grasp_markers_node src/m4_sim_grasp_markers_node.cpp)
target_compile_features(m4_sim_grasp_markers_node PUBLIC cxx_std_17)
ament_target_dependencies(
  m4_sim_grasp_markers_node
  geometry_msgs
  rclcpp
  visualization_msgs
)

install(TARGETS
  m2_scene_markers_node
  depth_to_point_cloud
  depth_to_mono8
  depth_failure_injector
  depth_to_point_cloud_node
  depth_to_mono8_node
  depth_failure_injector_node
  target_mask_emulator_node
  evidence_summary_logger_node
  robot_description_publisher_node
  observation_cache_node
  hypothesis_markers_node
  m4_offline_ranking_markers_node
  m4_real_ranking_markers_node
  m4_graspability_markers_node
  m4_sim_grasp_markers_node
  EXPORT export_${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(PROGRAMS
  scripts/m4_joint_hypothesis_markers_node.py
  scripts/m4_joint_hypothesis_publisher_node.py
  scripts/m4_live_hypothesis_publisher_node.py
  scripts/m4_perception_hypothesis_publisher_node.py
  scripts/m4_sim_moveit_plan_markers_node.py
  scripts/m8_strict_geometry_benchmark.py
  scripts/m8_transparent_detector_node.py
  scripts/m8_update_rate_benchmark.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch models worlds
  DESTINATION share/${PROJECT_NAME}
)
install(
  DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)
install(
  DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_m2_visual_scene_assets test/test_m2_visual_scene_assets.py)
  ament_add_pytest_test(test_m3_d435_manifest test/test_m3_d435_manifest.py)
  ament_add_pytest_test(test_m4_real_ranking_markers test/test_m4_real_ranking_markers.py)
  ament_add_pytest_test(test_m4_graspability_markers test/test_m4_graspability_markers.py)
  ament_add_pytest_test(test_m4_joint_hypothesis_markers test/test_m4_joint_hypothesis_markers.py)
  ament_add_pytest_test(test_m4_joint_hypothesis_topic test/test_m4_joint_hypothesis_topic.py)
  ament_add_pytest_test(test_m4_live_hypothesis_publisher test/test_m4_live_hypothesis_publisher.py)
  ament_add_pytest_test(test_m4_perception_hypothesis_publisher test/test_m4_perception_hypothesis_publisher.py)
  ament_add_pytest_test(test_m4_sim_moveit_plan_markers test/test_m4_sim_moveit_plan_markers.py)
  ament_add_gtest(test_depth_to_point_cloud test/test_depth_to_point_cloud.cpp)
  target_link_libraries(test_depth_to_point_cloud depth_to_point_cloud depth_to_mono8 depth_failure_injector)
  ament_target_dependencies(test_depth_to_point_cloud sensor_msgs)
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(geometry_msgs ghost_mgg_interfaces rclcpp sensor_msgs std_msgs tf2_ros visualization_msgs)
ament_package()
