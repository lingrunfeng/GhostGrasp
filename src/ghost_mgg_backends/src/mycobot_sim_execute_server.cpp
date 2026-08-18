#include "ghost_mgg_backends/mycobot_sim_execute_server.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <builtin_interfaces/msg/duration.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

namespace ghost_mgg_backends
{
namespace
{
using namespace std::chrono_literals;

constexpr const char * kDefaultExecuteActionName = "/grasp_executors/mycobot_sim/execute";
constexpr const char * kDefaultTrajectoryActionName = "/arm_controller/follow_joint_trajectory";

double elapsed_seconds(const rclcpp::Time & start, const rclcpp::Clock & clock)
{
  return (clock.now() - start).seconds();
}

builtin_interfaces::msg::Duration seconds_to_duration(double seconds)
{
  builtin_interfaces::msg::Duration duration;
  const auto clamped_seconds = std::max(0.0, seconds);
  const auto whole_seconds = std::floor(clamped_seconds);
  auto nanoseconds = std::llround((clamped_seconds - whole_seconds) * 1'000'000'000.0);
  duration.sec = static_cast<std::int32_t>(whole_seconds);
  if (nanoseconds >= 1'000'000'000) {
    ++duration.sec;
    nanoseconds -= 1'000'000'000;
  }
  duration.nanosec = static_cast<std::uint32_t>(nanoseconds);
  return duration;
}

MyCobotExecuteOutcome succeeded_outcome()
{
  return {true, kExecuteStatusSucceeded, ""};
}

MyCobotExecuteOutcome failed_outcome(const std::string & reason)
{
  return {false, kExecuteStatusFailed, reason};
}

MyCobotExecuteOutcome timeout_outcome(const std::string & reason)
{
  return {false, kExecuteStatusTimeout, reason};
}

MyCobotExecuteOutcome canceled_outcome(const std::string & reason)
{
  return {false, kExecuteStatusCanceled, reason};
}

std::string trajectory_failure_reason(
  const std::string & stage,
  const std::string & reason)
{
  return stage + " trajectory " + reason;
}

double seconds_until(const rclcpp::Time & deadline, const rclcpp::Clock & clock)
{
  return (deadline - clock.now()).seconds();
}

std::chrono::milliseconds wait_step(double remaining_sec)
{
  const auto step_ms = std::max(
    1,
    std::min(20, static_cast<int>(std::ceil(remaining_sec * 1000.0))));
  return std::chrono::milliseconds(step_ms);
}

enum class FutureWaitStatus
{
  ready,
  timeout,
  canceled,
  shutdown
};

template<typename FutureT>
FutureWaitStatus wait_for_future(
  FutureT & future,
  const rclcpp::Clock & clock,
  const rclcpp::Time & deadline,
  const std::shared_ptr<MyCobotSimExecuteServer::GoalHandleExecuteGrasp> & goal_handle,
  const std::function<void()> & on_cancel_or_timeout = nullptr)
{
  while (rclcpp::ok()) {
    if (goal_handle->is_canceling()) {
      if (on_cancel_or_timeout) {
        on_cancel_or_timeout();
      }
      return FutureWaitStatus::canceled;
    }

    const auto remaining_sec = seconds_until(deadline, clock);
    if (remaining_sec <= 0.0) {
      if (on_cancel_or_timeout) {
        on_cancel_or_timeout();
      }
      return FutureWaitStatus::timeout;
    }

    if (future.wait_for(wait_step(remaining_sec)) == std::future_status::ready) {
      return FutureWaitStatus::ready;
    }
  }

  return FutureWaitStatus::shutdown;
}

}  // namespace

std::vector<std::string> default_mycobot_joint_names()
{
  return {
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_link6",
    "link6_to_link6_flange"};
}

MyCobotTrajectoryPreset default_mycobot_trajectory_preset()
{
  return {
    {0.0, -0.6, 0.9, 0.0, -0.3, 0.0},
    {0.1, -0.5, 1.0, 0.0, -0.4, 0.1},
    {0.0, -0.7, 0.8, 0.0, -0.2, 0.0},
    1.0};
}

MyCobotPresetValidation validate_mycobot_preset(
  const std::vector<std::string> & joint_names,
  const MyCobotTrajectoryPreset & preset)
{
  if (joint_names.empty()) {
    return {false, "joint_names must not be empty"};
  }

  const auto expected_size = joint_names.size();
  const auto validate_positions =
    [expected_size](const std::vector<double> & positions, const char * field_name) {
      if (positions.size() != expected_size) {
        std::ostringstream reason;
        reason << field_name << " expected " << expected_size
               << " values but got " << positions.size();
        return reason.str();
      }
      return std::string{};
    };

  if (const auto reason = validate_positions(preset.pregrasp_positions, "pregrasp_positions");
    !reason.empty())
  {
    return {false, reason};
  }
  if (const auto reason = validate_positions(preset.grasp_positions, "grasp_positions");
    !reason.empty())
  {
    return {false, reason};
  }
  if (const auto reason = validate_positions(preset.retreat_positions, "retreat_positions");
    !reason.empty())
  {
    return {false, reason};
  }
  if (!std::isfinite(preset.segment_duration_sec) || preset.segment_duration_sec <= 0.0) {
    return {false, "segment_duration_sec must be positive"};
  }

  return {true, ""};
}

std::vector<MyCobotFollowJointTrajectory::Goal> make_mycobot_trajectory_goals(
  const std::vector<std::string> & joint_names,
  const MyCobotTrajectoryPreset & preset)
{
  const auto make_goal = [&](const std::vector<double> & positions) {
    MyCobotFollowJointTrajectory::Goal goal;
    goal.trajectory.joint_names = joint_names;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = positions;
    point.time_from_start = seconds_to_duration(preset.segment_duration_sec);
    goal.trajectory.points.push_back(point);
    return goal;
  };

  return {
    make_goal(preset.pregrasp_positions),
    make_goal(preset.grasp_positions),
    make_goal(preset.retreat_positions)};
}

MyCobotExecuteOutcome map_mycobot_trajectory_outcome(
  rclcpp_action::ResultCode result_code,
  const MyCobotFollowJointTrajectory::Result & trajectory_result,
  const std::string & stage)
{
  if (result_code == rclcpp_action::ResultCode::CANCELED) {
    return canceled_outcome(trajectory_failure_reason(stage, "canceled"));
  }

  if (result_code == rclcpp_action::ResultCode::ABORTED) {
    return failed_outcome(trajectory_failure_reason(stage, "aborted"));
  }

  if (result_code != rclcpp_action::ResultCode::SUCCEEDED) {
    return failed_outcome(trajectory_failure_reason(stage, "finished with unknown result code"));
  }

  if (trajectory_result.error_code == MyCobotFollowJointTrajectory::Result::SUCCESSFUL) {
    return succeeded_outcome();
  }

  std::ostringstream reason;
  reason << stage << " trajectory failed with error_code " << trajectory_result.error_code;
  if (!trajectory_result.error_string.empty()) {
    reason << ": " << trajectory_result.error_string;
  }
  return failed_outcome(reason.str());
}

MyCobotSimExecuteServer::MyCobotSimExecuteServer(const rclcpp::NodeOptions & options)
: rclcpp::Node("mycobot_sim_execute_server", options)
{
  const auto execute_action_name =
    this->declare_parameter<std::string>("action_name", kDefaultExecuteActionName);
  const auto trajectory_action_name =
    this->declare_parameter<std::string>("trajectory_action_name", kDefaultTrajectoryActionName);
  this->declare_parameter<std::vector<std::string>>(
    "joint_names", default_mycobot_joint_names());

  const auto default_preset = default_mycobot_trajectory_preset();
  this->declare_parameter<std::vector<double>>(
    "pregrasp_positions", default_preset.pregrasp_positions);
  this->declare_parameter<std::vector<double>>(
    "grasp_positions", default_preset.grasp_positions);
  this->declare_parameter<std::vector<double>>(
    "retreat_positions", default_preset.retreat_positions);
  this->declare_parameter<double>("segment_duration_sec", default_preset.segment_duration_sec);
  this->declare_parameter<double>("default_max_runtime_sec", 10.0);
  this->declare_parameter<double>("trajectory_server_timeout_sec", 2.0);

  trajectory_client_ = rclcpp_action::create_client<MyCobotFollowJointTrajectory>(
    this, trajectory_action_name);

  using namespace std::placeholders;
  action_server_ = rclcpp_action::create_server<MyCobotExecuteGrasp>(
    this,
    execute_action_name,
    std::bind(&MyCobotSimExecuteServer::handle_goal, this, _1, _2),
    std::bind(&MyCobotSimExecuteServer::handle_cancel, this, _1),
    std::bind(&MyCobotSimExecuteServer::handle_accepted, this, _1));
}

rclcpp_action::GoalResponse MyCobotSimExecuteServer::handle_goal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const MyCobotExecuteGrasp::Goal> goal)
{
  (void)uuid;
  bool expected_active = false;
  if (!execute_active_.compare_exchange_strong(expected_active, true)) {
    RCLCPP_WARN(
      this->get_logger(), "rejected myCobot execute goal trial_id=%s because another goal is active",
      goal->trial_id.c_str());
    return rclcpp_action::GoalResponse::REJECT;
  }

  RCLCPP_INFO(
    this->get_logger(), "accepted myCobot execute goal trial_id=%s hypothesis_id=%s",
    goal->trial_id.c_str(), goal->hypothesis.hypothesis_id.c_str());
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse MyCobotSimExecuteServer::handle_cancel(
  const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle)
{
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "accepted myCobot execute cancel request");
  cancel_active_or_pending_trajectories();
  return rclcpp_action::CancelResponse::ACCEPT;
}

void MyCobotSimExecuteServer::handle_accepted(
  const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle)
{
  std::thread{std::bind(&MyCobotSimExecuteServer::execute, this, goal_handle)}.detach();
}

void MyCobotSimExecuteServer::execute(
  const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle)
{
  const auto start = this->now();
  const auto goal = goal_handle->get_goal();
  auto result = std::make_shared<MyCobotExecuteGrasp::Result>();
  result->hypothesis_id = goal->hypothesis.hypothesis_id;

  const auto finish = [&](const MyCobotExecuteOutcome & outcome) {
    result->status = outcome.status;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->failure_reason = outcome.failure_reason;
    clear_active_trajectory();
    execute_active_.store(false);

    if (outcome.status == kExecuteStatusCanceled) {
      goal_handle->canceled(result);
      return;
    }
    goal_handle->succeed(result);
  };

  publish_feedback(goal_handle, "accepted", 0.05);

  const auto joint_names = joint_names_from_parameters();
  const auto preset = preset_from_parameters();
  if (const auto validation = validate_mycobot_preset(joint_names, preset); !validation.valid) {
    publish_feedback(goal_handle, "failed", 1.0);
    finish(failed_outcome("invalid myCobot trajectory preset: " + validation.failure_reason));
    return;
  }

  auto max_runtime_sec = goal->max_runtime_sec;
  if (!std::isfinite(max_runtime_sec) || max_runtime_sec <= 0.0) {
    max_runtime_sec = this->get_parameter("default_max_runtime_sec").as_double();
  }
  if (!std::isfinite(max_runtime_sec) || max_runtime_sec <= 0.0) {
    publish_feedback(goal_handle, "failed", 1.0);
    finish(failed_outcome("default_max_runtime_sec must be positive"));
    return;
  }
  const auto deadline = start + rclcpp::Duration::from_seconds(max_runtime_sec);

  if (goal_handle->is_canceling()) {
    finish(canceled_outcome("myCobot execute canceled"));
    return;
  }

  const auto server_ready = wait_for_trajectory_server(goal_handle, deadline);
  if (!server_ready.succeeded) {
    publish_feedback(goal_handle, "failed", 1.0);
    finish(server_ready);
    return;
  }

  const auto trajectory_goals = make_mycobot_trajectory_goals(joint_names, preset);
  const std::vector<std::pair<std::string, double>> stages = {
    {"pregrasp", 0.25},
    {"grasp", 0.55},
    {"retreat", 0.85}};

  for (std::size_t i = 0; i < trajectory_goals.size(); ++i) {
    publish_feedback(goal_handle, stages[i].first, stages[i].second);
    const auto stage_outcome = execute_trajectory_stage(
      goal_handle, trajectory_goals[i], stages[i].first, deadline);
    if (!stage_outcome.succeeded) {
      publish_feedback(goal_handle, "failed", 1.0);
      finish(stage_outcome);
      return;
    }
  }

  publish_feedback(goal_handle, "completed", 1.0);
  finish(succeeded_outcome());
}

MyCobotExecuteOutcome MyCobotSimExecuteServer::wait_for_trajectory_server(
  const std::shared_ptr<GoalHandleExecuteGrasp> & goal_handle,
  const rclcpp::Time & deadline)
{
  const auto server_wait_start = this->now();
  const auto configured_timeout_sec =
    this->get_parameter("trajectory_server_timeout_sec").as_double();
  const auto server_timeout_sec =
    std::isfinite(configured_timeout_sec) && configured_timeout_sec > 0.0 ?
    configured_timeout_sec : 0.0;

  while (rclcpp::ok()) {
    if (goal_handle->is_canceling()) {
      return canceled_outcome("myCobot execute canceled while waiting for trajectory server");
    }
    if (seconds_until(deadline, *this->get_clock()) <= 0.0) {
      return timeout_outcome("timed out waiting for trajectory action server");
    }
    if (server_timeout_sec > 0.0 &&
      elapsed_seconds(server_wait_start, *this->get_clock()) >= server_timeout_sec)
    {
      return timeout_outcome("trajectory action server unavailable");
    }
    if (trajectory_client_->wait_for_action_server(50ms)) {
      return succeeded_outcome();
    }
  }

  return failed_outcome("ROS shutdown while waiting for trajectory action server");
}

MyCobotExecuteOutcome MyCobotSimExecuteServer::execute_trajectory_stage(
  const std::shared_ptr<GoalHandleExecuteGrasp> & execute_goal_handle,
  const MyCobotFollowJointTrajectory::Goal & trajectory_goal,
  const std::string & stage,
  const rclcpp::Time & deadline)
{
  auto send_goal_future = trajectory_client_->async_send_goal(trajectory_goal);
  const auto cancel_any_trajectory = [this]() {
    cancel_active_or_pending_trajectories();
  };
  switch (wait_for_future(
      send_goal_future, *this->get_clock(), deadline, execute_goal_handle, cancel_any_trajectory))
  {
    case FutureWaitStatus::ready:
      break;
    case FutureWaitStatus::canceled:
      return canceled_outcome("myCobot execute canceled while sending " + stage + " trajectory");
    case FutureWaitStatus::timeout:
      return timeout_outcome("timed out sending " + stage + " trajectory");
    case FutureWaitStatus::shutdown:
      return failed_outcome("ROS shutdown while sending " + stage + " trajectory");
  }

  auto follow_goal_handle = send_goal_future.get();
  if (!follow_goal_handle) {
    return failed_outcome(stage + " trajectory goal rejected");
  }
  {
    std::lock_guard<std::mutex> lock(active_trajectory_mutex_);
    active_trajectory_goal_ = follow_goal_handle;
  }

  std::shared_future<FollowGoalHandle::WrappedResult> result_future;
  try {
    result_future = trajectory_client_->async_get_result(follow_goal_handle);
  } catch (const std::exception & ex) {
    clear_active_trajectory();
    return failed_outcome(stage + " trajectory result unavailable: " + std::string(ex.what()));
  }

  switch (wait_for_future(
      result_future, *this->get_clock(), deadline, execute_goal_handle, cancel_any_trajectory))
  {
    case FutureWaitStatus::ready:
      break;
    case FutureWaitStatus::canceled:
      clear_active_trajectory();
      return canceled_outcome("myCobot execute canceled during " + stage + " trajectory");
    case FutureWaitStatus::timeout:
      clear_active_trajectory();
      return timeout_outcome("timed out during " + stage + " trajectory");
    case FutureWaitStatus::shutdown:
      clear_active_trajectory();
      return failed_outcome("ROS shutdown during " + stage + " trajectory");
  }

  clear_active_trajectory();
  const auto wrapped_result = result_future.get();
  if (!wrapped_result.result) {
    return failed_outcome(stage + " trajectory returned an empty result");
  }
  return map_mycobot_trajectory_outcome(wrapped_result.code, *wrapped_result.result, stage);
}

MyCobotTrajectoryPreset MyCobotSimExecuteServer::preset_from_parameters() const
{
  MyCobotTrajectoryPreset preset;
  preset.pregrasp_positions = this->get_parameter("pregrasp_positions").as_double_array();
  preset.grasp_positions = this->get_parameter("grasp_positions").as_double_array();
  preset.retreat_positions = this->get_parameter("retreat_positions").as_double_array();
  preset.segment_duration_sec = this->get_parameter("segment_duration_sec").as_double();
  return preset;
}

std::vector<std::string> MyCobotSimExecuteServer::joint_names_from_parameters() const
{
  return this->get_parameter("joint_names").as_string_array();
}

void MyCobotSimExecuteServer::cancel_active_or_pending_trajectories()
{
  FollowGoalHandle::SharedPtr active_goal;
  {
    std::lock_guard<std::mutex> lock(active_trajectory_mutex_);
    active_goal = active_trajectory_goal_;
  }
  if (active_goal) {
    trajectory_client_->async_cancel_goal(active_goal);
  }
  trajectory_client_->async_cancel_all_goals();
}

void MyCobotSimExecuteServer::clear_active_trajectory()
{
  std::lock_guard<std::mutex> lock(active_trajectory_mutex_);
  active_trajectory_goal_.reset();
}

void MyCobotSimExecuteServer::publish_feedback(
  const std::shared_ptr<GoalHandleExecuteGrasp> & goal_handle,
  const std::string & stage,
  double progress) const
{
  auto feedback = std::make_shared<MyCobotExecuteGrasp::Feedback>();
  feedback->stage = stage;
  feedback->progress = progress;
  goal_handle->publish_feedback(feedback);
}

}  // namespace ghost_mgg_backends
