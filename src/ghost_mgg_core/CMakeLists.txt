cmake_minimum_required(VERSION 3.16)
project(ghost_mgg_core)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

add_library(${PROJECT_NAME}
  src/trial_log_event.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/
  DESTINATION include
)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

ament_python_install_package(ghost_mgg_core_py
  PACKAGE_DIR python/ghost_mgg_core_py
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_gtest(test_status_codes test/test_status_codes.cpp)
  target_link_libraries(test_status_codes ${PROJECT_NAME})
  ament_add_pytest_test(test_core_python test_py)
endif()

ament_package()
