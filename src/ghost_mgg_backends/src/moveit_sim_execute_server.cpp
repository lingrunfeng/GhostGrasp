#include "ghost_mgg_backends/moveit_sim_execute_server.hpp"

#include "ghost_mgg_backends/m2_scene_primitives.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <future>
#include <functional>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

namespace ghost_mgg_backends
{
namespace
{
using namespace std::chrono_literals;

constexpr const char * kDefaultActionName = "/grasp_executors/moveit_sim/execute";
constexpr const char * kDefaultPlanningGroup = "arm";
constexpr const char * kDefaultEndEffectorLink = "gripper_tcp";
constexpr const char * kDefaultTableFrame = "world";
constexpr const char * kDefaultTableObjectId = "m2_table_collision";
constexpr double kDefaultSceneObjectPaddingM = 0.006;
constexpr double kDefaultCartesianStepM = 0.004;
constexpr double kDefaultCartesianJumpThreshold = 0.0;
constexpr double kDefaultMinCartesianFraction = 0.95;

std::int32_t duration_seconds(double duration_sec)
{
  return static_cast<std::int32_t>(std::floor(std::max(0.0, duration_sec)));
}

std::uint32_t duration_nanoseconds(double duration_sec)
{
  const auto positive_duration = std::max(0.0, duration_sec);
  const auto seconds = std::floor(positive_duration);
  return static_cast<std::uint32_t>(
    std::round((positive_duration - seconds) * 1000000000.0));
}

double positive_or_default(double value, double fallback)
{
  return std::isfinite(value) && value > 0.0 ? value : fallback;
}

double clamped_scale(double value, double fallback)
{
  if (!std::isfinite(value)) {
    return fallback;
  }
  return std::clamp(value, 0.01, 1.0);
}

double nonzero_dimension(double value)
{
  if (!std::isfinite(value) || value <= 0.0) {
    return 0.001;
  }
  return value;
}

std::string frame_or_world(const std::string & frame_id)
{
  return frame_id.empty() ? "world" : frame_id;
}

std::string stage_failure(const std::string & stage, const std::string & reason)
{
  return stage + " " + reason;
}

bool moveit_succeeded(const moveit::core::MoveItErrorCode & code)
{
  return code == moveit::core::MoveItErrorCode::SUCCESS;
}

struct PipeCloser
{
  void operator()(FILE * pipe) const
  {
    if (pipe != nullptr) {
      (void)pclose(pipe);
    }
  }
};

std::string read_command_output(const std::string & command)
{
  std::array<char, 256> buffer{};
  std::string output;
  std::unique_ptr<FILE, PipeCloser> pipe(popen(command.c_str(), "r"));
  if (!pipe) {
    return output;
  }

  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
    output += buffer.data();
  }
  return output;
}

std::optional<double> query_gz_model_center_z(const std::string & model_name)
{
  const auto output = read_command_output(
    "timeout 2s gz model -m " + model_name + " -p 2>&1");
  return parse_gz_model_center_z(output);
}

}  // namespace

std::optional<GraspCandidate> select_first_valid_grasp_candidate(
  const GeometryHypothesis & hypothesis)
{
  for (const auto & candidate : hypothesis.grasp_candidates) {
    if (candidate.validation_state == GraspCandidate::VALIDATION_VALID) {
      return candidate;
    }
  }
  return std::nullopt;
}

std::string pose_reference_frame_for_target(
  const geometry_msgs::msg::PoseStamped & target,
  const std::string & fallback_frame)
{
  if (!target.header.frame_id.empty()) {
    return target.header.frame_id;
  }
  return fallback_frame.empty() ? "base_link" : fallback_frame;
}

moveit_msgs::msg::CollisionObject make_proxy_collision_object(
  const GeometryHypothesis & hypothesis,
  const std::string & id_prefix)
{
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = frame_or_world(hypothesis.pose_base.header.frame_id);
  object.id = id_prefix + "_" + hypothesis.hypothesis_id + "_proxy";
  object.operation = moveit_msgs::msg::CollisionObject::ADD;

  shape_msgs::msg::SolidPrimitive primitive;
  if (hypothesis.shape_type == GeometryHypothesis::SHAPE_CYLINDER ||
      hypothesis.shape_type == GeometryHypothesis::SHAPE_CUP_LIKE)
  {
    primitive.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    primitive.dimensions.resize(2);
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] =
      nonzero_dimension(hypothesis.dimensions_m.z);
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] =
      0.5 * std::max(
        nonzero_dimension(hypothesis.dimensions_m.x),
        nonzero_dimension(hypothesis.dimensions_m.y));
  } else {
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions = {
      nonzero_dimension(hypothesis.dimensions_m.x),
      nonzero_dimension(hypothesis.dimensions_m.y),
      nonzero_dimension(hypothesis.dimensions_m.z)};
  }

  object.primitives.push_back(primitive);
  object.primitive_poses.push_back(hypothesis.pose_base.pose);
  return object;
}

moveit_msgs::msg::CollisionObject make_padded_proxy_collision_object(
  const GeometryHypothesis & hypothesis,
  const std::string & id_prefix,
  double padding_m)
{
  auto object = make_proxy_collision_object(hypothesis, id_prefix);
  const auto padding = std::max(0.0, padding_m);
  for (auto & primitive : object.primitives) {
    if (primitive.type == shape_msgs::msg::SolidPrimitive::BOX &&
      primitive.dimensions.size() >= 3)
    {
      primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] += 2.0 * padding;
      primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] += 2.0 * padding;
      primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] += 2.0 * padding;
    } else if (primitive.type == shape_msgs::msg::SolidPrimitive::CYLINDER &&
      primitive.dimensions.size() >= 2)
    {
      primitive.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] +=
        2.0 * padding;
      primitive.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] += padding;
    }
  }
  return object;
}

std::optional<std::string> canonical_m2_model_name_for_hypothesis_id(
  const std::string & hypothesis_id)
{
  if (hypothesis_id == "mask_extrusion_glass_block" || hypothesis_id == "glass_block") {
    return std::string{"glass_block"};
  }
  if (hypothesis_id == "mask_extrusion_red_cube" || hypothesis_id == "red_cube") {
    return std::string{"red_cube"};
  }
  if (hypothesis_id == "mask_extrusion_blue_cylinder" || hypothesis_id == "blue_cylinder") {
    return std::string{"blue_cylinder"};
  }
  if (hypothesis_id == "mask_extrusion_green_cylinder" || hypothesis_id == "green_cylinder") {
    return std::string{"green_cylinder"};
  }
  return std::nullopt;
}

bool m2_hypotheses_refer_to_same_model(
  const std::string & first_hypothesis_id,
  const std::string & second_hypothesis_id)
{
  if (first_hypothesis_id == second_hypothesis_id) {
    return true;
  }
  const auto first_model = canonical_m2_model_name_for_hypothesis_id(first_hypothesis_id);
  const auto second_model = canonical_m2_model_name_for_hypothesis_id(second_hypothesis_id);
  return first_model.has_value() && second_model.has_value() && *first_model == *second_model;
}

std::vector<moveit_msgs::msg::CollisionObject> make_m2_scene_obstacle_collision_objects(
  const std::string & active_hypothesis_id,
  const std::string & id_prefix,
  double padding_m)
{
  std::vector<moveit_msgs::msg::CollisionObject> objects;
  for (const auto & hypothesis : make_m2_mask_extrusion_hypotheses(99)) {
    if (m2_hypotheses_refer_to_same_model(hypothesis.hypothesis_id, active_hypothesis_id)) {
      continue;
    }
    objects.push_back(make_padded_proxy_collision_object(hypothesis, id_prefix, padding_m));
  }
  return objects;
}

moveit_msgs::msg::CollisionObject make_table_collision_object(
  const std::string & frame_id,
  const std::string & object_id,
  double x,
  double y,
  double z,
  double size_x,
  double size_y,
  double size_z)
{
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = frame_or_world(frame_id);
  object.id = object_id;
  object.operation = moveit_msgs::msg::CollisionObject::ADD;

  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions = {
    nonzero_dimension(size_x),
    nonzero_dimension(size_y),
    nonzero_dimension(size_z)};
  object.primitives.push_back(primitive);

  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation.w = 1.0;
  object.primitive_poses.push_back(pose);
  return object;
}

GripperCommand::Goal make_gripper_command_goal(double position, double max_effort)
{
  GripperCommand::Goal goal;
  goal.command.position = position;
  goal.command.max_effort = max_effort;
  return goal;
}

GripperTrajectory::Goal make_gripper_trajectory_goal(
  double command_position,
  double duration_sec)
{
  GripperTrajectory::Goal goal;
  goal.trajectory.joint_names = {
    "gripper_controller",
    "gripper_base_to_gripper_left2",
    "gripper_left3_to_gripper_left1",
    "gripper_base_to_gripper_right3",
    "gripper_base_to_gripper_right2",
    "gripper_right3_to_gripper_right1"};

  const auto command = std::clamp(command_position, -0.70, 0.15);
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = {
    command,
    std::clamp(command, -0.80, 0.50),
    std::clamp(-command, -0.50, 0.50),
    std::clamp(-command, -0.15, 0.70),
    std::clamp(-command, -0.50, 0.80),
    std::clamp(command, -0.50, 0.50)};
  point.time_from_start.sec = duration_seconds(duration_sec);
  point.time_from_start.nanosec = duration_nanoseconds(duration_sec);
  if (point.time_from_start.nanosec >= 1000000000u) {
    ++point.time_from_start.sec;
    point.time_from_start.nanosec -= 1000000000u;
  }
  goal.trajectory.points.push_back(point);
  return goal;
}

bool should_wait_for_gripper_command_result(
  const std::string & stage,
  bool wait_for_close_result)
{
  return stage != "close_gripper" || wait_for_close_result;
}

std::optional<std::string> m2_model_name_for_hypothesis_id(
  const std::string & hypothesis_id)
{
  return canonical_m2_model_name_for_hypothesis_id(hypothesis_id);
}

std::optional<double> parse_gz_model_center_z(
  const std::string & gz_model_output)
{
  std::istringstream lines(gz_model_output);
  std::string line;
  bool read_next_vector = false;
  const std::regex vector_line{
    R"(\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\])"};

  while (std::getline(lines, line)) {
    if (read_next_vector) {
      std::smatch match;
      if (std::regex_search(line, match, vector_line) && match.size() == 4) {
        try {
          return std::stod(match[3].str());
        } catch (const std::exception &) {
          return std::nullopt;
        }
      }
    }
    if (line.find("Pose [ XYZ") != std::string::npos) {
      read_next_vector = true;
    }
  }
  return std::nullopt;
}

bool target_lift_delta_satisfies(
  double initial_center_z,
  double observed_center_z,
  double min_lift_z_delta)
{
  return std::isfinite(initial_center_z) && std::isfinite(observed_center_z) &&
         std::isfinite(min_lift_z_delta) &&
         observed_center_z - initial_center_z >= min_lift_z_delta;
}

MoveItSimExecuteServer::MoveItSimExecuteServer(const rclcpp::NodeOptions & options)
: rclcpp::Node("moveit_sim_execute_server", options)
{
  const auto action_name =
    this->declare_parameter<std::string>("action_name", kDefaultActionName);
  const auto gripper_trajectory_action_name =
    this->declare_parameter<std::string>(
      "gripper_trajectory_action_name", "/gripper_group_controller/follow_joint_trajectory");
  this->declare_parameter<std::string>("planning_group", kDefaultPlanningGroup);
  this->declare_parameter<std::string>("end_effector_link", kDefaultEndEffectorLink);
  this->declare_parameter<std::string>("table_frame", kDefaultTableFrame);
  this->declare_parameter<std::string>("table_object_id", kDefaultTableObjectId);
  this->declare_parameter<std::vector<double>>("table_position", {0.05, 0.05, 0.715});
  this->declare_parameter<std::vector<double>>("table_size", {0.42, 0.36, 0.03});
  this->declare_parameter<double>("planning_time_sec", 5.0);
  this->declare_parameter<double>("velocity_scaling", 0.20);
  this->declare_parameter<double>("acceleration_scaling", 0.20);
  this->declare_parameter<bool>("use_grasp_orientation", false);
  this->declare_parameter<double>("goal_position_tolerance_m", 0.006);
  this->declare_parameter<double>("goal_orientation_tolerance_rad", 0.45);
  this->declare_parameter<bool>("clear_scene_each_goal", true);
  this->declare_parameter<bool>("add_m2_scene_obstacles", true);
  this->declare_parameter<double>("scene_object_padding_m", kDefaultSceneObjectPaddingM);
  this->declare_parameter<bool>("use_cartesian_top_grasp_segments", false);
  this->declare_parameter<double>("cartesian_step_m", kDefaultCartesianStepM);
  this->declare_parameter<double>("cartesian_jump_threshold", kDefaultCartesianJumpThreshold);
  this->declare_parameter<double>("min_cartesian_fraction", kDefaultMinCartesianFraction);
  this->declare_parameter<bool>("enable_gripper", true);
  this->declare_parameter<double>("gripper_open_position", 0.15);
  this->declare_parameter<double>("gripper_close_position", -0.70);
  this->declare_parameter<double>("gripper_max_effort", 45.0);
  this->declare_parameter<double>("gripper_command_timeout_sec", 4.0);
  this->declare_parameter<double>("gripper_motion_duration_sec", 0.30);
  this->declare_parameter<bool>("wait_for_close_gripper_result", false);
  this->declare_parameter<double>("settle_time_sec", 0.25);
  this->declare_parameter<bool>("verify_lift_and_hold", false);
  this->declare_parameter<double>("min_lift_z_delta", 0.010);
  this->declare_parameter<double>("lift_hold_duration_sec", 0.50);
  this->declare_parameter<double>("lift_hold_sample_period_sec", 0.10);

  gripper_client_ = rclcpp_action::create_client<GripperTrajectory>(
    this,
    gripper_trajectory_action_name);

  action_server_ = rclcpp_action::create_server<MoveItExecuteGrasp>(
    this,
    action_name,
    std::bind(&MoveItSimExecuteServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
    std::bind(&MoveItSimExecuteServer::handle_cancel, this, std::placeholders::_1),
    std::bind(&MoveItSimExecuteServer::handle_accepted, this, std::placeholders::_1));
}

rclcpp_action::GoalResponse MoveItSimExecuteServer::handle_goal(
  const rclcpp_action::GoalUUID &,
  std::shared_ptr<const MoveItExecuteGrasp::Goal> goal)
{
  if (!goal || goal->hypothesis.hypothesis_id.empty()) {
    return rclcpp_action::GoalResponse::REJECT;
  }

  bool expected = false;
  if (!execute_active_.compare_exchange_strong(expected, true)) {
    RCLCPP_WARN(this->get_logger(), "rejecting MoveIt execute goal while another goal is active");
    return rclcpp_action::GoalResponse::REJECT;
  }

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse MoveItSimExecuteServer::handle_cancel(
  const std::shared_ptr<GoalHandleExecuteGrasp>)
{
  return rclcpp_action::CancelResponse::ACCEPT;
}

void MoveItSimExecuteServer::handle_accepted(
  const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle)
{
  std::thread{std::bind(&MoveItSimExecuteServer::execute, this, goal_handle)}.detach();
}

void MoveItSimExecuteServer::publish_feedback(
  const std::shared_ptr<GoalHandleExecuteGrasp> & goal_handle,
  const std::string & stage,
  double progress) const
{
  auto feedback = std::make_shared<MoveItExecuteGrasp::Feedback>();
  feedback->stage = stage;
  feedback->progress = progress;
  goal_handle->publish_feedback(feedback);
}

void MoveItSimExecuteServer::execute(
  const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle)
{
  const auto active_guard = std::unique_ptr<void, std::function<void(void *)>>(
    this,
    [this](void *) { execute_active_.store(false); });

  const auto goal = goal_handle->get_goal();
  auto result = std::make_shared<MoveItExecuteGrasp::Result>();
  result->hypothesis_id = goal->hypothesis.hypothesis_id;

  const auto start = this->now();
  const auto finish_with_failure =
    [&](std::uint8_t status, const std::string & reason) {
      result->status = status;
      result->failure_reason = reason;
      result->runtime_sec = (this->now() - start).seconds();
      if (status == kExecuteStatusCanceled) {
        goal_handle->canceled(result);
      } else {
        goal_handle->abort(result);
      }
    };

  std::vector<GraspCandidate> grasp_candidates;
  for (const auto & candidate : goal->hypothesis.grasp_candidates) {
    if (candidate.validation_state == GraspCandidate::VALIDATION_VALID) {
      grasp_candidates.push_back(candidate);
    }
  }
  if (grasp_candidates.empty()) {
    finish_with_failure(kExecuteStatusFailed, "hypothesis has no valid grasp candidate");
    return;
  }

  const auto planning_group =
    this->get_parameter("planning_group").as_string();
  const auto end_effector_link =
    this->get_parameter("end_effector_link").as_string();
  const auto table_frame =
    this->get_parameter("table_frame").as_string();
  const auto table_object_id =
    this->get_parameter("table_object_id").as_string();
  const auto table_position =
    this->get_parameter("table_position").as_double_array();
  const auto table_size =
    this->get_parameter("table_size").as_double_array();
  const auto planning_time_sec =
    positive_or_default(this->get_parameter("planning_time_sec").as_double(), 5.0);
  const auto velocity_scaling =
    clamped_scale(this->get_parameter("velocity_scaling").as_double(), 0.20);
  const auto acceleration_scaling =
    clamped_scale(this->get_parameter("acceleration_scaling").as_double(), 0.20);
  const auto use_grasp_orientation =
    this->get_parameter("use_grasp_orientation").as_bool();
  const auto goal_position_tolerance_m =
    positive_or_default(this->get_parameter("goal_position_tolerance_m").as_double(), 0.006);
  const auto goal_orientation_tolerance_rad =
    positive_or_default(this->get_parameter("goal_orientation_tolerance_rad").as_double(), 0.45);
  const auto clear_scene_each_goal =
    this->get_parameter("clear_scene_each_goal").as_bool();
  const auto add_m2_scene_obstacles =
    this->get_parameter("add_m2_scene_obstacles").as_bool();
  const auto scene_object_padding_m =
    std::max(0.0, this->get_parameter("scene_object_padding_m").as_double());
  const auto use_cartesian_top_grasp_segments =
    this->get_parameter("use_cartesian_top_grasp_segments").as_bool();
  const auto cartesian_step_m =
    positive_or_default(this->get_parameter("cartesian_step_m").as_double(), kDefaultCartesianStepM);
  const auto cartesian_jump_threshold =
    std::max(0.0, this->get_parameter("cartesian_jump_threshold").as_double());
  (void)cartesian_jump_threshold;
  const auto min_cartesian_fraction =
    clamped_scale(
      this->get_parameter("min_cartesian_fraction").as_double(),
      kDefaultMinCartesianFraction);
  const auto enable_gripper =
    this->get_parameter("enable_gripper").as_bool();
  const auto gripper_open_position =
    this->get_parameter("gripper_open_position").as_double();
  const auto gripper_close_position =
    this->get_parameter("gripper_close_position").as_double();
  const auto gripper_max_effort =
    positive_or_default(this->get_parameter("gripper_max_effort").as_double(), 45.0);
  const auto gripper_command_timeout_sec =
    positive_or_default(this->get_parameter("gripper_command_timeout_sec").as_double(), 4.0);
  const auto gripper_motion_duration_sec =
    positive_or_default(this->get_parameter("gripper_motion_duration_sec").as_double(), 0.30);
  const auto wait_for_close_gripper_result =
    this->get_parameter("wait_for_close_gripper_result").as_bool();
  const auto settle_time_sec =
    positive_or_default(this->get_parameter("settle_time_sec").as_double(), 0.25);
  const auto verify_lift_and_hold =
    this->get_parameter("verify_lift_and_hold").as_bool();
  const auto min_lift_z_delta =
    positive_or_default(this->get_parameter("min_lift_z_delta").as_double(), 0.010);
  const auto lift_hold_duration_sec =
    positive_or_default(this->get_parameter("lift_hold_duration_sec").as_double(), 0.50);
  const auto lift_hold_sample_period_sec =
    positive_or_default(this->get_parameter("lift_hold_sample_period_sec").as_double(), 0.10);

  if (table_position.size() != 3 || table_size.size() != 3) {
    finish_with_failure(kExecuteStatusFailed, "table_position and table_size must have 3 values");
    return;
  }

  try {
    const auto obstacle_prefix = goal->trial_id + "_m2_obstacle";

    publish_feedback(goal_handle, "planning_scene", 0.05);
    moveit::planning_interface::PlanningSceneInterface planning_scene;
    std::vector<std::string> collision_object_ids{table_object_id};
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    collision_objects.push_back(make_table_collision_object(
      table_frame,
      table_object_id,
      table_position[0],
      table_position[1],
      table_position[2],
      table_size[0],
      table_size[1],
      table_size[2]));
    if (add_m2_scene_obstacles) {
      auto obstacle_objects = make_m2_scene_obstacle_collision_objects(
        goal->hypothesis.hypothesis_id,
        obstacle_prefix,
        scene_object_padding_m);
      for (const auto & object : obstacle_objects) {
        collision_object_ids.push_back(object.id);
      }
      collision_objects.insert(
        collision_objects.end(),
        obstacle_objects.begin(),
        obstacle_objects.end());
    }
    planning_scene.applyCollisionObjects(collision_objects);

    auto move_group = moveit::planning_interface::MoveGroupInterface(
      shared_from_this(),
      planning_group);
    move_group.setPlanningTime(planning_time_sec);
    move_group.setMaxVelocityScalingFactor(velocity_scaling);
    move_group.setMaxAccelerationScalingFactor(acceleration_scaling);
    move_group.setGoalPositionTolerance(goal_position_tolerance_m);
    move_group.setGoalOrientationTolerance(goal_orientation_tolerance_rad);
    if (!end_effector_link.empty()) {
      move_group.setEndEffectorLink(end_effector_link);
    }

    const auto plan_and_execute =
      [&](
        const geometry_msgs::msg::PoseStamped & target,
        const std::string & stage,
        bool force_pose_orientation) {
        if (goal_handle->is_canceling()) {
          return stage_failure(stage, "canceled");
        }

        publish_feedback(goal_handle, "plan_" + stage, stage == "pregrasp" ? 0.20 : 0.50);
        move_group.setStartStateToCurrentState();
        if (use_grasp_orientation || force_pose_orientation) {
          move_group.setPoseTarget(target, end_effector_link);
        } else {
          const auto reference_frame = pose_reference_frame_for_target(
            target, move_group.getPlanningFrame());
          move_group.setPoseReferenceFrame(reference_frame);
          move_group.setPositionTarget(
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
            end_effector_link);
        }

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        const auto plan_code = move_group.plan(plan);
        if (!moveit_succeeded(plan_code)) {
          move_group.clearPoseTargets();
          return stage_failure(stage, "planning failed");
        }

        publish_feedback(goal_handle, "execute_" + stage, stage == "pregrasp" ? 0.35 : 0.70);
        const auto execute_code = move_group.execute(plan);
        move_group.clearPoseTargets();
        if (!moveit_succeeded(execute_code)) {
          return stage_failure(stage, "execution failed");
        }
        return std::string{};
      };

    const auto execute_cartesian =
      [&](
        const geometry_msgs::msg::PoseStamped & target,
        const std::string & stage,
        double plan_progress,
        double execute_progress) {
        if (goal_handle->is_canceling()) {
          return stage_failure(stage, "canceled");
        }

        publish_feedback(goal_handle, "plan_" + stage, plan_progress);
        move_group.clearPoseTargets();
        move_group.setStartStateToCurrentState();

        std::vector<geometry_msgs::msg::Pose> waypoints{target.pose};
        moveit_msgs::msg::RobotTrajectory trajectory;
        const auto fraction = move_group.computeCartesianPath(
          waypoints,
          cartesian_step_m,
          trajectory,
          true);
        if (fraction < min_cartesian_fraction) {
          std::ostringstream reason;
          reason << "cartesian path incomplete fraction=" << fraction
                 << " required=" << min_cartesian_fraction;
          return stage_failure(stage, reason.str());
        }

        publish_feedback(goal_handle, "execute_" + stage, execute_progress);
        const auto execute_code = move_group.execute(trajectory);
        if (!moveit_succeeded(execute_code)) {
          return stage_failure(stage, "execution failed");
        }
        return std::string{};
      };

    const auto command_gripper =
      [&](double position, const std::string & stage, double progress, bool wait_for_result) {
        if (!enable_gripper) {
          return std::string{};
        }
        if (goal_handle->is_canceling()) {
          return stage_failure(stage, "canceled");
        }
        if (!gripper_client_->wait_for_action_server(
            std::chrono::duration<double>(gripper_command_timeout_sec)))
        {
          return stage_failure(stage, "gripper action server unavailable");
        }

        publish_feedback(goal_handle, stage, progress);
        (void)gripper_max_effort;
        const auto gripper_goal = make_gripper_trajectory_goal(
          position,
          gripper_motion_duration_sec);
        auto goal_future = gripper_client_->async_send_goal(gripper_goal);
        if (goal_future.wait_for(std::chrono::duration<double>(gripper_command_timeout_sec)) !=
          std::future_status::ready)
        {
          return stage_failure(stage, "gripper goal send timeout");
        }

        const auto gripper_goal_handle = goal_future.get();
        if (!gripper_goal_handle) {
          return stage_failure(stage, "gripper goal rejected");
        }
        if (!wait_for_result) {
          return std::string{};
        }

        auto result_future = gripper_client_->async_get_result(gripper_goal_handle);
        if (result_future.wait_for(std::chrono::duration<double>(gripper_command_timeout_sec)) !=
          std::future_status::ready)
        {
          gripper_client_->async_cancel_goal(gripper_goal_handle);
          return stage_failure(stage, "gripper result timeout");
        }

        const auto wrapped_result = result_future.get();
        if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED) {
          return stage_failure(stage, "gripper command failed");
        }
        return std::string{};
      };

    const auto verify_target_lift_and_hold =
      [&](const GeometryHypothesis & hypothesis, const std::string & stage) {
        if (!verify_lift_and_hold) {
          return std::string{};
        }

        const auto model_name = m2_model_name_for_hypothesis_id(hypothesis.hypothesis_id);
        if (!model_name.has_value()) {
          return stage_failure(stage, "has no configured M2 Gazebo model mapping");
        }

        RCLCPP_INFO(
          this->get_logger(),
          "verify_lift_and_hold target_model=%s min_lift_z_delta=%.4f hold_sec=%.3f",
          model_name->c_str(),
          min_lift_z_delta,
          lift_hold_duration_sec);
        const auto initial_center_z = hypothesis.pose_base.pose.position.z;
        const auto deadline = std::chrono::steady_clock::now() +
          std::chrono::duration<double>(lift_hold_duration_sec);

        while (std::chrono::steady_clock::now() < deadline) {
          if (goal_handle->is_canceling()) {
            return stage_failure(stage, "canceled");
          }
          std::this_thread::sleep_for(
            std::chrono::duration<double>(lift_hold_sample_period_sec));
        }

        if (goal_handle->is_canceling()) {
          return stage_failure(stage, "canceled");
        }

        const auto observed_center_z = query_gz_model_center_z(*model_name);
        if (!observed_center_z.has_value()) {
          return stage_failure(stage, "could not read Gazebo model pose for " + *model_name);
        }
        if (!target_lift_delta_satisfies(
            initial_center_z,
            *observed_center_z,
            min_lift_z_delta))
        {
          std::ostringstream reason;
          reason << "target " << *model_name << " not lifted: z_delta="
                 << (*observed_center_z - initial_center_z)
                 << " required=" << min_lift_z_delta;
          return stage_failure(stage, reason.str());
        }

        RCLCPP_INFO(
          this->get_logger(),
          "verify_lift_and_hold succeeded target_model=%s",
          model_name->c_str());
        return std::string{};
      };

    std::vector<std::string> grasp_failures;
    for (const auto & selected_grasp : grasp_candidates) {
      const auto grasp_prefix = selected_grasp.grasp_id.empty() ?
        std::string{"grasp_candidate"} :
        selected_grasp.grasp_id;
      const auto is_top_grasp =
        selected_grasp.grasp_type == GraspCandidate::GRASP_TYPE_TOP;
      const auto use_cartesian_segments =
        use_cartesian_top_grasp_segments && is_top_grasp;

      if (const auto reason =
          command_gripper(
            gripper_open_position,
            "open_gripper",
            0.12,
            should_wait_for_gripper_command_result("open_gripper", wait_for_close_gripper_result));
        !reason.empty())
      {
        if (reason.find("canceled") != std::string::npos) {
          finish_with_failure(kExecuteStatusCanceled, reason);
          return;
        }
        finish_with_failure(kExecuteStatusFailed, grasp_prefix + ": " + reason);
        return;
      }

      if (const auto reason =
          plan_and_execute(selected_grasp.pregrasp_pose, "pregrasp", is_top_grasp);
        !reason.empty())
      {
        if (reason.find("canceled") != std::string::npos) {
          finish_with_failure(kExecuteStatusCanceled, reason);
          return;
        }
        grasp_failures.push_back(grasp_prefix + ": " + reason);
        continue;
      }

      const auto grasp_reason = use_cartesian_segments ?
        execute_cartesian(selected_grasp.grasp_pose, "vertical_descent", 0.48, 0.66) :
        plan_and_execute(selected_grasp.grasp_pose, "grasp", is_top_grasp);
      if (!grasp_reason.empty())
      {
        if (grasp_reason.find("canceled") != std::string::npos) {
          finish_with_failure(kExecuteStatusCanceled, grasp_reason);
          return;
        }
        grasp_failures.push_back(grasp_prefix + ": " + grasp_reason);
        continue;
      }

      if (const auto reason =
          command_gripper(
            gripper_close_position,
            "close_gripper",
            0.76,
            should_wait_for_gripper_command_result("close_gripper", wait_for_close_gripper_result));
        !reason.empty())
      {
        if (reason.find("canceled") != std::string::npos) {
          finish_with_failure(kExecuteStatusCanceled, reason);
          return;
        }
        grasp_failures.push_back(grasp_prefix + ": " + reason);
        continue;
      }

      publish_feedback(goal_handle, "settle_grasp", 0.80);
      std::this_thread::sleep_for(std::chrono::duration<double>(settle_time_sec));

      const auto retreat_reason = use_cartesian_segments ?
        execute_cartesian(selected_grasp.pregrasp_pose, "vertical_retreat", 0.82, 0.88) :
        plan_and_execute(selected_grasp.pregrasp_pose, "retreat", is_top_grasp);
      if (!retreat_reason.empty())
      {
        if (retreat_reason.find("canceled") != std::string::npos) {
          finish_with_failure(kExecuteStatusCanceled, retreat_reason);
          return;
        }
        grasp_failures.push_back(grasp_prefix + ": " + retreat_reason);
        continue;
      }

      publish_feedback(goal_handle, "verify_lift_and_hold", 0.90);
      if (const auto reason =
          verify_target_lift_and_hold(goal->hypothesis, "verify_lift_and_hold");
        !reason.empty())
      {
        if (reason.find("canceled") != std::string::npos) {
          finish_with_failure(kExecuteStatusCanceled, reason);
          return;
        }
        grasp_failures.push_back(grasp_prefix + ": " + reason);
        continue;
      }

      if (clear_scene_each_goal) {
        planning_scene.removeCollisionObjects(collision_object_ids);
      }

      publish_feedback(goal_handle, "succeeded", 1.0);
      result->status = kExecuteStatusSucceeded;
      result->failure_reason.clear();
      result->runtime_sec = (this->now() - start).seconds();
      goal_handle->succeed(result);
      return;
    }

    if (clear_scene_each_goal) {
      planning_scene.removeCollisionObjects(collision_object_ids);
    }

    std::ostringstream failure_stream;
    failure_stream << "all grasp candidates failed";
    for (const auto & failure : grasp_failures) {
      failure_stream << "; " << failure;
    }
    finish_with_failure(kExecuteStatusFailed, failure_stream.str());
  } catch (const std::exception & ex) {
    finish_with_failure(kExecuteStatusFailed, std::string{"MoveIt exception: "} + ex.what());
  }
}

}  // namespace ghost_mgg_backends
