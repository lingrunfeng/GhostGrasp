#include "ghost_mgg_backends/mask_extrusion_recovery_server.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <thread>

#include "ghost_mgg_backends/m2_scene_primitives.hpp"
#include "ghost_mgg_backends/recover_status.hpp"

namespace ghost_mgg_backends
{
namespace
{
constexpr const char * kBackendName = "mask_extrusion";
constexpr const char * kDefaultActionName = "/geometry_backends/mask_extrusion/recover";
constexpr const char * kSuccessDiagnostics =
  "mask_extrusion baseline completed using configured M2 scene prior";

double elapsed_seconds(const rclcpp::Time & start, const rclcpp::Clock & clock)
{
  return (clock.now() - start).seconds();
}
}  // namespace

MaskExtrusionRecoveryServer::MaskExtrusionRecoveryServer(const rclcpp::NodeOptions & options)
: rclcpp::Node("mask_extrusion_recovery_server", options)
{
  const auto action_name = this->declare_parameter<std::string>("action_name", kDefaultActionName);
  const auto hypotheses_topic = this->declare_parameter<std::string>(
    "hypotheses_topic", "/ghost_mgg/hypotheses/mask_extrusion");
  this->declare_parameter<std::string>("mode", "success");
  this->declare_parameter<std::string>("preferred_hypothesis_id", "");
  this->declare_parameter<bool>("strict_preferred_hypothesis", false);
  this->declare_parameter<int>("response_delay_ms", 0);
  hypotheses_pub_ = this->create_publisher<ghost_mgg_interfaces::msg::GeometryHypothesisArray>(
    hypotheses_topic,
    rclcpp::QoS(1).transient_local().reliable());

  using namespace std::placeholders;
  action_server_ = rclcpp_action::create_server<RecoverGeometry>(
    this,
    action_name,
    std::bind(&MaskExtrusionRecoveryServer::handle_goal, this, _1, _2),
    std::bind(&MaskExtrusionRecoveryServer::handle_cancel, this, _1),
    std::bind(&MaskExtrusionRecoveryServer::handle_accepted, this, _1));
}

rclcpp_action::GoalResponse MaskExtrusionRecoveryServer::handle_goal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const RecoverGeometry::Goal> goal)
{
  (void)uuid;
  RCLCPP_INFO(
    this->get_logger(), "accepted mask extrusion recovery goal trial_id=%s observation_id=%s",
    goal->trial_id.c_str(), goal->observation_id.c_str());
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse MaskExtrusionRecoveryServer::handle_cancel(
  const std::shared_ptr<GoalHandleRecoverGeometry> goal_handle)
{
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "accepted mask extrusion recovery cancel request");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void MaskExtrusionRecoveryServer::handle_accepted(
  const std::shared_ptr<GoalHandleRecoverGeometry> goal_handle)
{
  std::thread{std::bind(&MaskExtrusionRecoveryServer::execute, this, goal_handle)}.detach();
}

void MaskExtrusionRecoveryServer::execute(
  const std::shared_ptr<GoalHandleRecoverGeometry> goal_handle)
{
  const auto start = this->now();
  const auto goal = goal_handle->get_goal();
  auto result = std::make_shared<RecoverGeometry::Result>();
  result->backend_name = kBackendName;

  publish_feedback(goal_handle, "accepted", 0.05, 0.0);

  auto mode = this->get_parameter("mode").as_string();
  auto response_delay_ms = static_cast<int>(this->get_parameter("response_delay_ms").as_int());
  if (mode == "delay" && response_delay_ms <= 0) {
    response_delay_ms = 1000;
  }

  if (!sleep_cancelable(goal_handle, response_delay_ms)) {
    result->status = kRecoverStatusCanceled;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "mask_extrusion baseline canceled";
    goal_handle->canceled(result);
    return;
  }

  publish_feedback(goal_handle, "generating_hypotheses", 0.35, 0.0);
  if (goal_handle->is_canceling()) {
    result->status = kRecoverStatusCanceled;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "mask_extrusion baseline canceled";
    goal_handle->canceled(result);
    return;
  }

  if (mode == "failure") {
    result->status = kRecoverStatusFailed;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "mask_extrusion baseline configured to fail";
    publish_feedback(goal_handle, "failed", 1.0, 0.0);
    goal_handle->succeed(result);
    return;
  }

  if (mode == "empty") {
    result->status = kRecoverStatusLowConfidence;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "mask_extrusion baseline configured to return no hypotheses";
    publish_feedback(goal_handle, "empty", 1.0, 0.0);
    goal_handle->succeed(result);
    return;
  }

  if (mode != "success" && mode != "delay") {
    result->status = kRecoverStatusFailed;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "unknown mask_extrusion recovery mode: " + mode;
    publish_feedback(goal_handle, "failed", 1.0, 0.0);
    goal_handle->succeed(result);
    return;
  }

  auto hypotheses = make_m2_mask_extrusion_hypotheses(100);
  const auto preferred_hypothesis_id =
    this->get_parameter("preferred_hypothesis_id").as_string();
  const auto strict_preferred_hypothesis =
    this->get_parameter("strict_preferred_hypothesis").as_bool();
  if (!preferred_hypothesis_id.empty()) {
    const auto preferred = std::find_if(
      hypotheses.begin(),
      hypotheses.end(),
      [&preferred_hypothesis_id](const auto & hypothesis) {
        return hypothesis.hypothesis_id == preferred_hypothesis_id;
      });
    if (preferred != hypotheses.end()) {
      if (strict_preferred_hypothesis) {
        const auto selected = *preferred;
        hypotheses.clear();
        hypotheses.push_back(selected);
      } else {
        std::rotate(hypotheses.begin(), preferred, std::next(preferred));
      }
    } else if (strict_preferred_hypothesis) {
      hypotheses.clear();
    }
  }

  const auto limit = std::min<std::size_t>(goal->max_hypotheses, hypotheses.size());
  result->hypotheses = std::vector<ghost_mgg_interfaces::msg::GeometryHypothesis>(
    hypotheses.begin(), hypotheses.begin() + limit);
  publish_hypotheses(*goal, result->hypotheses);
  const auto best_score = result->hypotheses.empty() ? 0.0 : result->hypotheses.front().score.total;
  publish_feedback(goal_handle, "ranked_hypotheses", 0.80, best_score);

  if (goal_handle->is_canceling()) {
    result->status = kRecoverStatusCanceled;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "mask_extrusion baseline canceled";
    goal_handle->canceled(result);
    return;
  }

  result->status = result->hypotheses.empty() ? kRecoverStatusLowConfidence : kRecoverStatusSucceeded;
  result->runtime_sec = elapsed_seconds(start, *this->get_clock());
  result->diagnostics = kSuccessDiagnostics;
  publish_feedback(goal_handle, "completed", 1.0, best_score);
  goal_handle->succeed(result);
}

bool MaskExtrusionRecoveryServer::sleep_cancelable(
  const std::shared_ptr<GoalHandleRecoverGeometry> & goal_handle,
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

void MaskExtrusionRecoveryServer::publish_feedback(
  const std::shared_ptr<GoalHandleRecoverGeometry> & goal_handle,
  const std::string & stage,
  double progress,
  double current_best_score) const
{
  auto feedback = std::make_shared<RecoverGeometry::Feedback>();
  feedback->backend_name = kBackendName;
  feedback->stage = stage;
  feedback->progress = progress;
  feedback->current_best_score = current_best_score;
  goal_handle->publish_feedback(feedback);
}

void MaskExtrusionRecoveryServer::publish_hypotheses(
  const RecoverGeometry::Goal & goal,
  const std::vector<ghost_mgg_interfaces::msg::GeometryHypothesis> & hypotheses) const
{
  ghost_mgg_interfaces::msg::GeometryHypothesisArray message;
  message.header.stamp = this->now();
  message.header.frame_id = "world";
  message.trial_id = goal.trial_id;
  message.observation_id = goal.observation_id;
  message.backend_name = kBackendName;
  message.hypotheses = hypotheses;
  hypotheses_pub_->publish(message);
}

}  // namespace ghost_mgg_backends
