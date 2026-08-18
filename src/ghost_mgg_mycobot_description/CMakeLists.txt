cmake_minimum_required(VERSION 3.8)
project(ghost_mgg_mycobot_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config meshes rviz urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES README.md LICENSE
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_mycobot_control_assets test/test_mycobot_control_assets.py)
endif()

ament_package()
