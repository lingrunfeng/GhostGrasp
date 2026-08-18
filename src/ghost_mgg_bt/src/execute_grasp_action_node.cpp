#include "ghost_mgg_bt/execute_grasp_action_node.hpp"

#include <algorithm>
#include <chrono>
#include <string>

namespace ghost_mgg_bt
{
namespace
{
template<typename FutureT>
bool ready(const FutureT & future)
{
  return future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

std::chrono::milliseconds action_server_wait_duration(double timeout_sec)
{
  const auto wait_sec = timeout_sec > 0.0 ? std::min(timeout_sec, 5.0) : 2.0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::duration<double>(wait_sec));
}
}  // namespace

ExecuteGraspActionNode::ExecuteGraspActionNode(
  const std::string & name,
  const BT::NodeConfig & config,
  rclcpp::Node::SharedPtr node,
  std::shared_ptr<BackendRegistry> registry,
  std::string executor_name)
: BT::StatefulActionNode(name, config),
  node_(std::move(node)),
  registry_(std::move(registry)),
  executor_name_(std::move(executor_name))
{
}

BT::PortsList ExecuteGraspActionNode::providedPorts()
{
  return {
    BT::InputPort<std::string>("trial_id"),
    BT::InputPort<ghost_mgg_interfaces::msg::GeometryHypothesis>("selected_hypothesis"),
    BT::InputPort<double>("max_runtime_sec"),
    BT::OutputPort<std::uint8_t>("execute_status"),
    BT::OutputPort<std::string>("execute_failure_reason"),
  };
}

BT::NodeStatus ExecuteGraspActionNode::onStart()
{
  std::string trial_id;
  ghost_mgg_interfaces::msg::GeometryHypothesis hypothesis;
  timeout_sec_ = 2.0;

  if (!getInput("trial_id", trial_id) ||
      !getInput("selected_hypothesis", hypothesis)) {
    setOutput("execute_status", static_cast<std::uint8_t>(2));
    setOutput("execute_failure_reason", std::string{"missing ExecuteGrasp input port"});
    return BT::NodeStatus::FAILURE;
  }
  getInput("max_runtime_sec", timeout_sec_);

  const auto action_name = registry_->executor(executor_name_).execute_action;
  client_ = rclcpp_action::create_client<ExecuteGrasp>(node_, action_name);
  if (!client_->wait_for_action_server(action_server_wait_duration(timeout_sec_))) {
    setOutput("execute_status", static_cast<std::uint8_t>(2));
    setOutput("execute_failure_reason", "execute action server unavailable: " + action_name);
    return BT::NodeStatus::FAILURE;
  }

  ExecuteGrasp::Goal goal;
  goal.trial_id = trial_id;
  goal.hypothesis = hypothesis;
  goal.max_runtime_sec = timeout_sec_;

  start_time_ = node_->now();
  goal_future_ = client_->async_send_goal(goal);
  goal_requested_ = true;
  result_requested_ = false;
  cancel_requested_ = false;
  goal_handle_.reset();
  return BT::NodeStatus::RUNNING;
}

BT::NodeStatus ExecuteGraspActionNode::onRunning()
{
  if (timeout_sec_ > 0.0 && (node_->now() - start_time_).seconds() > timeout_sec_) {
    onHalted();
    setOutput("execute_status", static_cast<std::uint8_t>(3));
    setOutput("execute_failure_reason", std::string{"execute action timed out"});
    return BT::NodeStatus::FAILURE;
  }

  if (goal_requested_ && !goal_handle_) {
    if (!ready(goal_future_)) {
      return BT::NodeStatus::RUNNING;
    }
    goal_handle_ = goal_future_.get();
    if (!goal_handle_) {
      setOutput("execute_status", static_cast<std::uint8_t>(2));
      setOutput("execute_failure_reason", std::string{"execute goal rejected"});
      return BT::NodeStatus::FAILURE;
    }
    result_future_ = client_->async_get_result(goal_handle_);
    result_requested_ = true;
  }

  if (!result_requested_ || !ready(result_future_)) {
    return BT::NodeStatus::RUNNING;
  }

  const auto wrapped = result_future_.get();
  if (!wrapped.result) {
    setOutput("execute_status", static_cast<std::uint8_t>(2));
    setOutput("execute_failure_reason", std::string{"execute result missing"});
    return BT::NodeStatus::FAILURE;
  }

  setOutput("execute_status", wrapped.result->status);
  setOutput("execute_failure_reason", wrapped.result->failure_reason);
  return wrapped.result->status == 1 ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

void ExecuteGraspActionNode::onHalted()
{
  cancel_requested_ = true;
  if (client_ && goal_handle_) {
    (void)client_->async_cancel_goal(goal_handle_);
  }
  goal_requested_ = false;
  result_requested_ = false;
  goal_handle_.reset();
}

}  // namespace ghost_mgg_bt
