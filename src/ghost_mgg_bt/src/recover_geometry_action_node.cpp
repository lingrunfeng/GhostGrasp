#include "ghost_mgg_bt/recover_geometry_action_node.hpp"

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "ghost_mgg_interfaces/msg/observation_ref.hpp"

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

RecoverGeometryActionNode::RecoverGeometryActionNode(
  const std::string & name,
  const BT::NodeConfig & config,
  rclcpp::Node::SharedPtr node,
  std::shared_ptr<BackendRegistry> registry)
: BT::StatefulActionNode(name, config),
  node_(std::move(node)),
  registry_(std::move(registry))
{
}

BT::PortsList RecoverGeometryActionNode::providedPorts()
{
  using Hypothesis = ghost_mgg_interfaces::msg::GeometryHypothesis;
  return {
    BT::InputPort<std::string>("backend_name"),
    BT::InputPort<std::string>("trial_id"),
    BT::InputPort<std::string>("observation_id"),
    BT::InputPort<ghost_mgg_interfaces::msg::ObservationRef>("observation_ref"),
    BT::InputPort<std::string>("target_label"),
    BT::InputPort<std::string>("shape_hint"),
    BT::InputPort<double>("max_runtime_sec"),
    BT::InputPort<unsigned>("max_hypotheses"),
    BT::OutputPort<std::vector<Hypothesis>>("hypotheses"),
    BT::OutputPort<std::uint8_t>("recover_status"),
    BT::OutputPort<std::string>("recover_diagnostics"),
  };
}

BT::NodeStatus RecoverGeometryActionNode::onStart()
{
  std::string backend_name;
  std::string trial_id;
  std::string observation_id;
  ghost_mgg_interfaces::msg::ObservationRef observation;
  std::string target_label;
  std::string shape_hint;
  unsigned max_hypotheses = 3;
  timeout_sec_ = 2.0;

  if (!getInput("backend_name", backend_name) ||
      !getInput("trial_id", trial_id) ||
      !getInput("observation_id", observation_id) ||
      !getInput("observation_ref", observation)) {
    setOutput("recover_status", static_cast<std::uint8_t>(2));
    setOutput("recover_diagnostics", std::string{"missing RecoverGeometry input port"});
    return BT::NodeStatus::FAILURE;
  }
  getInput("target_label", target_label);
  getInput("shape_hint", shape_hint);
  getInput("max_runtime_sec", timeout_sec_);
  getInput("max_hypotheses", max_hypotheses);

  const auto action_name = registry_->backend(backend_name).recover_action;
  client_ = rclcpp_action::create_client<RecoverGeometry>(node_, action_name);
  if (!client_->wait_for_action_server(action_server_wait_duration(timeout_sec_))) {
    setOutput("recover_status", static_cast<std::uint8_t>(2));
    setOutput("recover_diagnostics", "recover action server unavailable: " + action_name);
    return BT::NodeStatus::FAILURE;
  }

  RecoverGeometry::Goal goal;
  goal.trial_id = trial_id;
  goal.observation_id = observation_id;
  goal.backend_name = backend_name;
  goal.observation = observation;
  goal.target_label = target_label;
  goal.shape_hint = shape_hint;
  goal.max_runtime_sec = timeout_sec_;
  goal.max_hypotheses = max_hypotheses;

  start_time_ = node_->now();
  goal_future_ = client_->async_send_goal(goal);
  goal_requested_ = true;
  result_requested_ = false;
  cancel_requested_ = false;
  goal_handle_.reset();
  return BT::NodeStatus::RUNNING;
}

BT::NodeStatus RecoverGeometryActionNode::onRunning()
{
  if (timeout_sec_ > 0.0 && (node_->now() - start_time_).seconds() > timeout_sec_) {
    onHalted();
    setOutput("recover_status", static_cast<std::uint8_t>(4));
    setOutput("recover_diagnostics", std::string{"recover action timed out"});
    return BT::NodeStatus::FAILURE;
  }

  if (goal_requested_ && !goal_handle_) {
    if (!ready(goal_future_)) {
      return BT::NodeStatus::RUNNING;
    }
    goal_handle_ = goal_future_.get();
    if (!goal_handle_) {
      setOutput("recover_status", static_cast<std::uint8_t>(2));
      setOutput("recover_diagnostics", std::string{"recover goal rejected"});
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
    setOutput("recover_status", static_cast<std::uint8_t>(2));
    setOutput("recover_diagnostics", std::string{"recover result missing"});
    return BT::NodeStatus::FAILURE;
  }

  setOutput("recover_status", wrapped.result->status);
  setOutput("recover_diagnostics", wrapped.result->diagnostics);
  setOutput("hypotheses", wrapped.result->hypotheses);
  if (wrapped.result->status == 1 && !wrapped.result->hypotheses.empty()) {
    return BT::NodeStatus::SUCCESS;
  }
  return BT::NodeStatus::FAILURE;
}

void RecoverGeometryActionNode::onHalted()
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
