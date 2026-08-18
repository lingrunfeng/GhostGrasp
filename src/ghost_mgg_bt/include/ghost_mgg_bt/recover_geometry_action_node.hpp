#pragma once

#include <future>
#include <memory>
#include <string>
#include <vector>

#include <behaviortree_cpp/action_node.h>
#include <behaviortree_cpp/basic_types.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "ghost_mgg_bt/backend_registry.hpp"
#include "ghost_mgg_interfaces/action/recover_geometry.hpp"

namespace ghost_mgg_bt
{

class RecoverGeometryActionNode : public BT::StatefulActionNode
{
public:
  using RecoverGeometry = ghost_mgg_interfaces::action::RecoverGeometry;
  using GoalHandle = rclcpp_action::ClientGoalHandle<RecoverGeometry>;

  RecoverGeometryActionNode(
    const std::string & name,
    const BT::NodeConfig & config,
    rclcpp::Node::SharedPtr node,
    std::shared_ptr<BackendRegistry> registry);

  static BT::PortsList providedPorts();
  bool cancel_requested() const { return cancel_requested_; }

private:
  BT::NodeStatus onStart() override;
  BT::NodeStatus onRunning() override;
  void onHalted() override;

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<BackendRegistry> registry_;
  rclcpp_action::Client<RecoverGeometry>::SharedPtr client_;
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
