#include "ghost_mgg_backends/dummy_recovery_server.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ghost_mgg_interfaces/msg/grasp_candidate.hpp"
#include "ghost_mgg_interfaces/msg/score_breakdown.hpp"

namespace ghost_mgg_backends
{
namespace
{
using GraspCandidate = ghost_mgg_interfaces::msg::GraspCandidate;
using ScoreBreakdown = ghost_mgg_interfaces::msg::ScoreBreakdown;

constexpr const char * kDefaultActionName = "/geometry_backends/dummy/recover";

ScoreBreakdown make_score(double total, double grasp)
{
  ScoreBreakdown score;
  score.visual = total - 0.05;
  score.failure = total - 0.10;
  score.depth = total - 0.15;
  score.physical = total - 0.08;
  score.grasp = grasp;
  score.prior = 0.50;
  score.total = total;
  return score;
}

GraspCandidate make_grasp(
  const std::string & hypothesis_id,
  std::uint8_t grasp_type,
  double score,
  double width_m)
{
  GraspCandidate grasp;
  grasp.grasp_id = "g_" + hypothesis_id + (grasp_type == GraspCandidate::GRASP_TYPE_TOP ? "_top" : "_side");
  grasp.grasp_pose.header.frame_id = "base_link";
  grasp.pregrasp_pose.header.frame_id = "base_link";
  grasp.grasp_pose.pose.orientation.w = 1.0;
  grasp.pregrasp_pose.pose.orientation.w = 1.0;
  grasp.pregrasp_pose.pose.position.z = 0.10;
  grasp.approach_vector.z = grasp_type == GraspCandidate::GRASP_TYPE_TOP ? -1.0 : 0.0;
  grasp.approach_vector.x = grasp_type == GraspCandidate::GRASP_TYPE_SIDE ? 1.0 : 0.0;
  grasp.gripper_width_m = width_m;
  grasp.grasp_type = grasp_type;
  grasp.score = score;
  grasp.validation_state = GraspCandidate::VALIDATION_VALID;
  return grasp;
}

GeometryHypothesis make_hypothesis(
  const std::string & hypothesis_id,
  std::uint8_t shape_type,
  double total_score,
  double confidence,
  double x,
  double y,
  double z,
  double width,
  double depth,
  double height,
  std::uint8_t grasp_type,
  const std::string & provenance)
{
  GeometryHypothesis hypothesis;
  hypothesis.hypothesis_id = hypothesis_id;
  hypothesis.shape_type = shape_type;
  hypothesis.pose_camera.header.frame_id = "camera_color_optical_frame";
  hypothesis.pose_base.header.frame_id = "base_link";
  hypothesis.pose_camera.pose.position.x = x;
  hypothesis.pose_camera.pose.position.y = y;
  hypothesis.pose_camera.pose.position.z = z;
  hypothesis.pose_camera.pose.orientation.w = 1.0;
  hypothesis.pose_base.pose.position.x = x;
  hypothesis.pose_base.pose.position.y = y;
  hypothesis.pose_base.pose.position.z = z;
  hypothesis.pose_base.pose.orientation.w = 1.0;
  hypothesis.dimensions_m.x = width;
  hypothesis.dimensions_m.y = depth;
  hypothesis.dimensions_m.z = height;
  hypothesis.score = make_score(total_score, total_score - 0.03);
  hypothesis.confidence = confidence;
  hypothesis.uncertainty = 1.0 - confidence;
  hypothesis.grasp_candidates.push_back(
    make_grasp(hypothesis_id, grasp_type, total_score - 0.02, std::max(width, depth) + 0.01));
  hypothesis.provenance = provenance;
  hypothesis.validation_state = GeometryHypothesis::VALIDATION_VALID;
  return hypothesis;
}

double elapsed_seconds(const rclcpp::Time & start, const rclcpp::Clock & clock)
{
  return (clock.now() - start).seconds();
}

}  // namespace

std::vector<GeometryHypothesis> make_dummy_hypotheses(
  const std::string & backend_name,
  const std::string & trial_id,
  const std::string & observation_id,
  std::size_t max_hypotheses)
{
  (void)trial_id;
  (void)observation_id;

  std::vector<GeometryHypothesis> hypotheses;
  hypotheses.reserve(3);
  hypotheses.push_back(make_hypothesis(
    "h1", GeometryHypothesis::SHAPE_BOX, 0.90, 0.90,
    0.30, 0.00, 0.08, 0.045, 0.050, 0.080,
    GraspCandidate::GRASP_TYPE_TOP, backend_name));
  hypotheses.push_back(make_hypothesis(
    "h2", GeometryHypothesis::SHAPE_CYLINDER, 0.75, 0.78,
    0.31, -0.02, 0.075, 0.040, 0.040, 0.075,
    GraspCandidate::GRASP_TYPE_SIDE, backend_name));
  hypotheses.push_back(make_hypothesis(
    "h3", GeometryHypothesis::SHAPE_CUP_LIKE, 0.60, 0.64,
    0.29, 0.02, 0.090, 0.055, 0.055, 0.090,
    GraspCandidate::GRASP_TYPE_SIDE, backend_name));

  if (max_hypotheses < hypotheses.size()) {
    hypotheses.resize(max_hypotheses);
  }
  return hypotheses;
}

DummyRecoveryServer::DummyRecoveryServer(const rclcpp::NodeOptions & options)
: rclcpp::Node("dummy_recovery_server", options)
{
  const auto action_name = this->declare_parameter<std::string>("action_name", kDefaultActionName);
  this->declare_parameter<std::string>("mode", "success");
  this->declare_parameter<int>("response_delay_ms", 0);

  using namespace std::placeholders;
  action_server_ = rclcpp_action::create_server<RecoverGeometry>(
    this,
    action_name,
    std::bind(&DummyRecoveryServer::handle_goal, this, _1, _2),
    std::bind(&DummyRecoveryServer::handle_cancel, this, _1),
    std::bind(&DummyRecoveryServer::handle_accepted, this, _1));
}

rclcpp_action::GoalResponse DummyRecoveryServer::handle_goal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const RecoverGeometry::Goal> goal)
{
  (void)uuid;
  RCLCPP_INFO(
    this->get_logger(), "accepted dummy recovery goal trial_id=%s observation_id=%s",
    goal->trial_id.c_str(), goal->observation_id.c_str());
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse DummyRecoveryServer::handle_cancel(
  const std::shared_ptr<GoalHandleRecoverGeometry> goal_handle)
{
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "accepted dummy recovery cancel request");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void DummyRecoveryServer::handle_accepted(
  const std::shared_ptr<GoalHandleRecoverGeometry> goal_handle)
{
  std::thread{std::bind(&DummyRecoveryServer::execute, this, goal_handle)}.detach();
}

void DummyRecoveryServer::execute(
  const std::shared_ptr<GoalHandleRecoverGeometry> goal_handle)
{
  const auto start = this->now();
  const auto goal = goal_handle->get_goal();
  auto result = std::make_shared<RecoverGeometry::Result>();
  const auto backend_name = goal->backend_name.empty() ? std::string{"dummy"} : goal->backend_name;
  result->backend_name = backend_name;

  publish_feedback(goal_handle, "accepted", 0.05, 0.0);

  auto mode = this->get_parameter("mode").as_string();
  auto response_delay_ms = static_cast<int>(this->get_parameter("response_delay_ms").as_int());
  if (mode == "delay" && response_delay_ms <= 0) {
    response_delay_ms = 1000;
  }

  if (!sleep_cancelable(goal_handle, response_delay_ms)) {
    result->status = kRecoverStatusCanceled;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "dummy recovery canceled";
    goal_handle->canceled(result);
    return;
  }

  publish_feedback(goal_handle, "generating_hypotheses", 0.35, 0.0);
  if (goal_handle->is_canceling()) {
    result->status = kRecoverStatusCanceled;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "dummy recovery canceled";
    goal_handle->canceled(result);
    return;
  }

  if (mode == "failure") {
    result->status = kRecoverStatusFailed;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "dummy recovery configured to fail";
    publish_feedback(goal_handle, "failed", 1.0, 0.0);
    goal_handle->succeed(result);
    return;
  }

  if (mode == "empty") {
    result->status = kRecoverStatusLowConfidence;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "dummy recovery configured to return no hypotheses";
    publish_feedback(goal_handle, "empty", 1.0, 0.0);
    goal_handle->succeed(result);
    return;
  }

  if (mode != "success" && mode != "delay") {
    result->status = kRecoverStatusFailed;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "unknown dummy recovery mode: " + mode;
    publish_feedback(goal_handle, "failed", 1.0, 0.0);
    goal_handle->succeed(result);
    return;
  }

  result->hypotheses = make_dummy_hypotheses(
    backend_name, goal->trial_id, goal->observation_id, goal->max_hypotheses);
  const auto best_score = result->hypotheses.empty() ? 0.0 : result->hypotheses.front().score.total;
  publish_feedback(goal_handle, "ranked_hypotheses", 0.80, best_score);

  if (goal_handle->is_canceling()) {
    result->status = kRecoverStatusCanceled;
    result->runtime_sec = elapsed_seconds(start, *this->get_clock());
    result->diagnostics = "dummy recovery canceled";
    goal_handle->canceled(result);
    return;
  }

  result->status = result->hypotheses.empty() ? kRecoverStatusLowConfidence : kRecoverStatusSucceeded;
  result->runtime_sec = elapsed_seconds(start, *this->get_clock());
  result->diagnostics = "dummy recovery completed";
  publish_feedback(goal_handle, "completed", 1.0, best_score);
  goal_handle->succeed(result);
}

bool DummyRecoveryServer::sleep_cancelable(
  const std::shared_ptr<GoalHandleRecoverGeometry> & goal_handle,
  int delay_ms)
{
  using namespace std::chrono_literals;
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

void DummyRecoveryServer::publish_feedback(
  const std::shared_ptr<GoalHandleRecoverGeometry> & goal_handle,
  const std::string & stage,
  double progress,
  double current_best_score) const
{
  auto feedback = std::make_shared<RecoverGeometry::Feedback>();
  feedback->backend_name = "dummy";
  feedback->stage = stage;
  feedback->progress = progress;
  feedback->current_best_score = current_best_score;
  goal_handle->publish_feedback(feedback);
}

}  // namespace ghost_mgg_backends
