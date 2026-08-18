#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "ghost_mgg_interfaces/action/recover_geometry.hpp"
#include "ghost_mgg_interfaces/msg/geometry_hypothesis.hpp"
#include "ghost_mgg_backends/recover_status.hpp"

namespace ghost_mgg_backends
{

using GeometryHypothesis = ghost_mgg_interfaces::msg::GeometryHypothesis;
using RecoverGeometry = ghost_mgg_interfaces::action::RecoverGeometry;

std::vector<GeometryHypothesis> make_dummy_hypotheses(
  const std::string & backend_name,
  const std::string & trial_id,
  const std::string & observation_id,
  std::size_t max_hypotheses);

class DummyRecoveryServer : public rclcpp::Node
{
public:
  using GoalHandleRecoverGeometry = rclcpp_action::ServerGoalHandle<RecoverGeometry>;

  explicit DummyRecoveryServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const RecoverGeometry::Goal> goal);

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleRecoverGeometry> goal_handle);

  void handle_accepted(const std::shared_ptr<GoalHandleRecoverGeometry> goal_handle);
  void execute(const std::shared_ptr<GoalHandleRecoverGeometry> goal_handle);

  bool sleep_cancelable(
    const std::shared_ptr<GoalHandleRecoverGeometry> & goal_handle,
    int delay_ms);

  void publish_feedback(
    const std::shared_ptr<GoalHandleRecoverGeometry> & goal_handle,
    const std::string & stage,
    double progress,
    double current_best_score) const;

  rclcpp_action::Server<RecoverGeometry>::SharedPtr action_server_;
};

}  // namespace ghost_mgg_backends
