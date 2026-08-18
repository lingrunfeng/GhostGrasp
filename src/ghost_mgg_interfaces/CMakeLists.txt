cmake_minimum_required(VERSION 3.16)
project(ghost_mgg_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObservationRef.msg"
  "msg/ScoreBreakdown.msg"
  "msg/GraspCandidate.msg"
  "msg/GeometryHypothesis.msg"
  "msg/GeometryHypothesisArray.msg"
  "msg/FailureEvidenceSummary.msg"
  "srv/MyCobotGoHome.srv"
  "srv/MyCobotGuardedTaughtGrasp.srv"
  "action/RecoverGeometry.action"
  "action/ExecuteGrasp.action"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
