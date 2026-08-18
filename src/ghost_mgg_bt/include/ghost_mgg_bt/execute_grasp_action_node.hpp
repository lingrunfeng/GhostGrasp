#pragma once

#include <future>
#include <memory>
#include <string>

#include <behaviortree_cpp/action_node.h>
#include <behaviortree_cpp/basic_types.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "ghost_mgg_bt/backend_registry.hpp"
#include "ghost_mgg_interfaces/action/execute_grasp.hpp"

namespace ghost_mgg_bt
{

class ExecuteGraspActionNode : public BT::StatefulActionNode
{
public:
  using ExecuteGrasp = ghost_mgg_interfaces::action::ExecuteGrasp;
  using GoalHandle = rclcpp_action::ClientGoalHandle<ExecuteGrasp>;

  ExecuteGraspActionNode(
    const std::string & name,
    const BT::NodeConfig & config,
    rclcpp::Node::SharedPtr node,
    std::shared_ptr<BackendRegistry> registry,
    std::string executor_name);

  static BT::PortsList providedPorts();
  bool cancel_requested() const { return cancel_requested_; }

private:
  BT::NodeStatus onStart() override;
  BT::NodeStatus onRunning() override;
  void onHalted() override;

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<BackendRegistry> registry_;
  std::string executor_name_;
  rclcpp_action::Client<ExecuteGrasp>::SharedPtr client_;
  std::shared_future<typename GoalHandle::SharedPtr> goal_future_;
  std::shared_future<typename GoalHandle::WrappedResult> result_future_;
  typename GoalHandle::SharedPtr goal_handle_;
  rclcpp::Time start_time_;
  double timeout_sec_ = 0.0;
  bool goal_requested_ = false;
  bool result_requested_ = false;
  bool cancel_requested_ = false;
};

}  // namespace ghost_mgg_bt
