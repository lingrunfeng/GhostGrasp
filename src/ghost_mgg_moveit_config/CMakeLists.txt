cmake_minimum_required(VERSION 3.16)
project(ghost_mgg_moveit_config)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_moveit_config_assets test/test_moveit_config_assets.py)
endif()

ament_package()
