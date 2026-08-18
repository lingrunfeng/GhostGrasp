#include "ghost_mgg_bt/select_next_hypothesis_node.hpp"

#include <string>

namespace ghost_mgg_bt
{

SelectNextHypothesis::SelectNextHypothesis(
  const std::string & name,
  const BT::NodeConfig & config)
: BT::SyncActionNode(name, config)
{
}

BT::PortsList SelectNextHypothesis::providedPorts()
{
  using Hypothesis = ghost_mgg_interfaces::msg::GeometryHypothesis;
  return {
    BT::InputPort<std::vector<Hypothesis>>("hypotheses"),
    BT::BidirectionalPort<int>("hypothesis_index"),
    BT::OutputPort<Hypothesis>("selected_hypothesis"),
    BT::OutputPort<std::string>("selected_hypothesis_id"),
    BT::BidirectionalPort<std::vector<std::string>>("attempted_hypothesis_ids"),
  };
}

BT::NodeStatus SelectNextHypothesis::tick()
{
  using Hypothesis = ghost_mgg_interfaces::msg::GeometryHypothesis;
  std::vector<Hypothesis> hypotheses;
  int hypothesis_index = 0;
  std::vector<std::string> attempted_ids;

  if (!getInput("hypotheses", hypotheses)) {
    return BT::NodeStatus::FAILURE;
  }
  getInput("hypothesis_index", hypothesis_index);
  getInput("attempted_hypothesis_ids", attempted_ids);

  if (hypothesis_index < 0 || static_cast<std::size_t>(hypothesis_index) >= hypotheses.size()) {
    return BT::NodeStatus::FAILURE;
  }

  const auto & selected = hypotheses[static_cast<std::size_t>(hypothesis_index)];
  attempted_ids.push_back(selected.hypothesis_id);

  setOutput("selected_hypothesis", selected);
  setOutput("selected_hypothesis_id", selected.hypothesis_id);
  setOutput("attempted_hypothesis_ids", attempted_ids);
  setOutput("hypothesis_index", hypothesis_index + 1);
  return BT::NodeStatus::SUCCESS;
}

}  // namespace ghost_mgg_bt
