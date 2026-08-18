#pragma once

#include <string>
#include <vector>

#include <behaviortree_cpp/action_node.h>
#include <behaviortree_cpp/basic_types.h>

#include "ghost_mgg_interfaces/msg/geometry_hypothesis.hpp"

namespace ghost_mgg_bt
{

class SelectNextHypothesis : public BT::SyncActionNode
{
public:
  SelectNextHypothesis(const std::string & name, const BT::NodeConfig & config);

  static BT::PortsList providedPorts();

private:
  BT::NodeStatus tick() override;
};

}  // namespace ghost_mgg_bt
