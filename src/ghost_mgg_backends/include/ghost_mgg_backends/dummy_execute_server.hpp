#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "ghost_mgg_backends/execute_status.hpp"
#include "ghost_mgg_interfaces/action/execute_grasp.hpp"

namespace ghost_mgg_backends
{

using ExecuteGrasp = ghost_mgg_interfaces::action::ExecuteGrasp;

struct DummyExecuteOutcome
{
  bool succeeded = false;
  std::uint8_t status = kExecuteStatusFailed;
  std::string failure_reason;
};

DummyExecuteOutcome evaluate_dummy_execute(
  const std::string & mode,
  const std::string & hypothesis_id);

class DummyExecuteServer : public rclcpp::Node
{
public:
  using GoalHandleExecuteGrasp = rclcpp_action::ServerGoalHandle<ExecuteGrasp>;

  explicit DummyExecuteServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const ExecuteGrasp::Goal> goal);

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle);

  void handle_accepted(const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle);
  void execute(const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle);

  bool sleep_cancelable(
    const std::shared_ptr<GoalHandleExecuteGrasp> & goal_handle,
    int delay_ms);

  void publish_feedback(
    const std::shared_ptr<GoalHandleExecuteGrasp> & goal_handle,
    const std::string & stage,
    double progress) const;

  rclcpp_action::Server<ExecuteGrasp>::SharedPtr action_server_;
};

}  // namespace ghost_mgg_backends
