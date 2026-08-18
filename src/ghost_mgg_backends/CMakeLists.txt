cmake_minimum_required(VERSION 3.16)
project(ghost_mgg_backends)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(ghost_mgg_interfaces REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_library(m2_scene_primitives_lib
  src/m2_scene_primitives.cpp
)
target_include_directories(m2_scene_primitives_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(m2_scene_primitives_lib PUBLIC cxx_std_17)
ament_target_dependencies(m2_scene_primitives_lib
  ghost_mgg_interfaces
)

add_library(dummy_backend_lib
  src/dummy_recovery_server.cpp
  src/mask_extrusion_recovery_server.cpp
  src/dummy_execute_server.cpp
  src/mycobot_sim_execute_server.cpp
  src/moveit_sim_execute_server.cpp
)
target_include_directories(dummy_backend_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(dummy_backend_lib PUBLIC cxx_std_17)
ament_target_dependencies(dummy_backend_lib
  rclcpp
  rclcpp_action
  control_msgs
  ghost_mgg_interfaces
  moveit_msgs
  moveit_ros_planning_interface
  shape_msgs
  trajectory_msgs
)
target_link_libraries(dummy_backend_lib
  m2_scene_primitives_lib
)

add_executable(dummy_recovery_server src/dummy_recovery_server_main.cpp)
target_link_libraries(dummy_recovery_server dummy_backend_lib)
ament_target_dependencies(dummy_recovery_server rclcpp)

add_executable(mask_extrusion_recovery_server src/mask_extrusion_recovery_server_main.cpp)
target_link_libraries(mask_extrusion_recovery_server dummy_backend_lib)
ament_target_dependencies(mask_extrusion_recovery_server rclcpp)

add_executable(dummy_execute_server src/dummy_execute_server_main.cpp)
target_link_libraries(dummy_execute_server dummy_backend_lib)
ament_target_dependencies(dummy_execute_server rclcpp)

add_executable(mycobot_sim_execute_server src/mycobot_sim_execute_server_main.cpp)
target_link_libraries(mycobot_sim_execute_server dummy_backend_lib)
ament_target_dependencies(
  mycobot_sim_execute_server
  rclcpp
  rclcpp_action
  control_msgs
  ghost_mgg_interfaces
  trajectory_msgs
)

add_executable(moveit_sim_execute_server src/moveit_sim_execute_server_main.cpp)
target_link_libraries(moveit_sim_execute_server dummy_backend_lib)
ament_target_dependencies(
  moveit_sim_execute_server
  rclcpp
  rclcpp_action
  control_msgs
  ghost_mgg_interfaces
  moveit_msgs
  moveit_ros_planning_interface
  shape_msgs
)

install(TARGETS
  m2_scene_primitives_lib
  dummy_backend_lib
  dummy_recovery_server
  mask_extrusion_recovery_server
  dummy_execute_server
  mycobot_sim_execute_server
  moveit_sim_execute_server
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_dummy_servers test/test_dummy_servers.cpp)
  target_link_libraries(test_dummy_servers dummy_backend_lib)
  ament_target_dependencies(test_dummy_servers rclcpp rclcpp_action ghost_mgg_interfaces)

  ament_add_gtest(test_mask_extrusion_recovery_server
    test/test_mask_extrusion_recovery_server.cpp)
  target_link_libraries(test_mask_extrusion_recovery_server dummy_backend_lib)
  target_compile_definitions(test_mask_extrusion_recovery_server PRIVATE
    GHOST_MGG_BACKENDS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  )
  ament_target_dependencies(
    test_mask_extrusion_recovery_server
    rclcpp
    rclcpp_action
    ghost_mgg_interfaces
  )

  ament_add_gtest(test_mycobot_sim_execute_server test/test_mycobot_sim_execute_server.cpp)
  target_link_libraries(test_mycobot_sim_execute_server dummy_backend_lib)
  target_compile_definitions(test_mycobot_sim_execute_server PRIVATE
    GHOST_MGG_BACKENDS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  )
  ament_target_dependencies(
    test_mycobot_sim_execute_server
    rclcpp
    rclcpp_action
    control_msgs
    ghost_mgg_interfaces
    trajectory_msgs
  )

  ament_add_gtest(test_moveit_sim_execute_server test/test_moveit_sim_execute_server.cpp)
  target_link_libraries(test_moveit_sim_execute_server dummy_backend_lib)
  target_compile_definitions(test_moveit_sim_execute_server PRIVATE
    GHOST_MGG_BACKENDS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  )
  ament_target_dependencies(
    test_moveit_sim_execute_server
    rclcpp
    rclcpp_action
    ghost_mgg_interfaces
    moveit_msgs
    moveit_ros_planning_interface
    shape_msgs
  )
endif()

ament_export_include_directories(include)
ament_export_libraries(dummy_backend_lib m2_scene_primitives_lib)
ament_export_dependencies(
  rclcpp
  rclcpp_action
  control_msgs
  ghost_mgg_interfaces
  moveit_msgs
  moveit_ros_planning_interface
  shape_msgs
  trajectory_msgs
)
ament_package()
