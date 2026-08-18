#include "ghost_mgg_backends/dummy_execute_server.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace ghost_mgg_backends
{
namespace
{
constexpr const char * kDefaultActionName = "/grasp_executors/dummy/execute";

double elapsed_seconds(const rclcpp::Time & start, const rclcpp::Clock & clock)
{
  return (clock.now() - start).seconds();
}
}  // namespace

DummyExecuteOutcome evaluate_dummy_execute(
  const std::string & mode,
  const std::string & hypothesis_id)
{
  if (mode == "always_succeed" || mode == "delay") {
    return {true, kExecuteStatusSucceeded, ""};
  }

  if (mode == "fail_first_then_succeed") {
    if (hypothesis_id == "h1") {
      return {false, kExecuteStatusFailed, "dummy executor rejected first hypothesis"};
    }
    return {true, kExecuteStatusSucceeded, ""};
  }

  if (mode == "fail_all") {
    return {false, kExecuteStatusFailed, "dummy executor configured to fail all hypotheses"};
  }

  return {false, kExecuteStatusFailed, "unknown dummy execute mode: " + mode};
}

DummyExecuteServer::DummyExecuteServer(const rclcpp::NodeOptions & options)
: rclcpp::Node("dummy_execute_server", options)
{
  const auto action_name = this->declare_parameter<std::string>("action_name", kDefaultActionName);
  this->declare_parameter<std::string>("mode", "fail_first_then_succeed");
  this->declare_parameter<int>("response_delay_ms", 0);

  using namespace std::placeholders;
  action_server_ = rclcpp_action::create_server<ExecuteGrasp>(
    this,
    action_name,
    std::bind(&DummyExecuteServer::handle_goal, this, _1, _2),
    std::bind(&DummyExecuteServer::handle_cancel, this, _1),
    std::bind(&DummyExecuteServer::handle_accepted, this, _1));
}

rclcpp_action::GoalResponse DummyExecuteServer::handle_goal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const ExecuteGrasp::Goal> goal)
{
  (void)uuid;
  RCLCPP_INFO(
    this->get_logger(), "accepted dummy execute goal trial_id=%s hypothesis_id=%s",
    goal->trial_id.c_str(), goal->hypothesis.hypothesis_id.c_str());
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse DummyExecuteServer::handle_cancel(
  const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle)
{
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "accepted dummy execute cancel request");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void DummyExecuteServer::handle_accepted(
  const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle)
{
  std::thread{std::bind(&DummyExecuteServer::execute, this, goal_handle)}.detach();
}

void DummyExecuteServer::execute(
  const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle)
{
  const auto start = this->now();
  const auto goal = goal_handle->get_goal();
  auto result = std::make_shared<ExecuteGrasp::Result>();
  result->hypothesis_id = goal->hypothesis.hypothesis_id;

  publish_feedback(goal_handle, "accepted", 0.05);

  auto mode = this->get_parameter("mode").as_string();
  auto response_delay_ms = static_cast<int>(this->get_parameter("response_delay_ms").as_int());
  if (mode == "delay" && response_delay_ms <= 0) {
    response_delay_ms = 1000;
  }

  publish_feedback(goal_handle, "executing", 0.50);
  if (!sleep_cancelable(goal_handle, response_delay_ms)) {
    result->status = kExecuteStatusCanceled;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->failure_reason = "dummy execute canceled";
    goal_handle->canceled(result);
    return;
  }

  const auto outcome = evaluate_dummy_execute(mode, goal->hypothesis.hypothesis_id);
  publish_feedback(goal_handle, "verifying", 0.85);
  if (goal_handle->is_canceling()) {
    result->status = kExecuteStatusCanceled;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->failure_reason = "dummy execute canceled";
    goal_handle->canceled(result);
    return;
  }

  result->status = outcome.status;
  result->runtime_sec = elapsed_seconds(start, *this->get_clock());
  result->failure_reason = outcome.failure_reason;
  publish_feedback(goal_handle, "completed", 1.0);
  goal_handle->succeed(result);
}

bool DummyExecuteServer::sleep_cancelable(
  const std::shared_ptr<GoalHandleExecuteGrasp> & goal_handle,
  int delay_ms)
{
  auto remaining_ms = std::max(delay_ms, 0);
  while (remaining_ms > 0) {
    if (goal_handle->is_canceling()) {
      return false;
    }
    const auto step_ms = std::min(remaining_ms, 20);
    std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
    remaining_ms -= step_ms;
  }
  return !goal_handle->is_canceling();
}

void DummyExecuteServer::publish_feedback(
  const std::shared_ptr<GoalHandleExecuteGrasp> & goal_handle,
  const std::string & stage,
  double progress) const
{
  auto feedback = std::make_shared<ExecuteGrasp::Feedback>();
  feedback->stage = stage;
  feedback->progress = progress;
  goal_handle->publish_feedback(feedback);
}

}  // namespace ghost_mgg_backends
