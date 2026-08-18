#pragma once

#include <memory>
#include <string>

#include <behaviortree_cpp/bt_factory.h>
#include <rclcpp/rclcpp.hpp>

#include "ghost_mgg_bt/backend_registry.hpp"
#include "ghost_mgg_bt/trial_logger.hpp"

namespace ghost_mgg_bt
{
void register_m0_nodes(
  BT::BehaviorTreeFactory & factory,
  rclcpp::Node::SharedPtr node,
  std::shared_ptr<BackendRegistry> registry,
  std::shared_ptr<TrialLogger> logger,
  const std::string & executor_name);
}  // namespace ghost_mgg_bt
