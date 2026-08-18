#include "ghost_mgg_bt/report_trial_node.hpp"

#include <string>
#include <vector>

namespace ghost_mgg_bt
{
namespace
{
std::string final_status_from_execute(const std::uint8_t execute_status)
{
  return execute_status == 1 ? "succeeded" : "failed";
}
}  // namespace

ReportTrialNode::ReportTrialNode(
  const std::string & name,
  const BT::NodeConfig & config,
  std::shared_ptr<TrialLogger> logger)
: BT::SyncActionNode(name, config), logger_(std::move(logger))
{
}

BT::PortsList ReportTrialNode::providedPorts()
{
  using Hypothesis = ghost_mgg_interfaces::msg::GeometryHypothesis;
  return {
    BT::InputPort<std::string>("trial_id"),
    BT::InputPort<std::string>("observation_id"),
    BT::InputPort<std::string>("tree_name"),
    BT::InputPort<std::string>("backend_name"),
    BT::InputPort<std::uint8_t>("recover_status"),
    BT::InputPort<std::vector<Hypothesis>>("hypotheses"),
    BT::InputPort<std::vector<std::string>>("attempted_hypothesis_ids"),
    BT::InputPort<std::string>("selected_hypothesis_id"),
    BT::InputPort<std::uint8_t>("execute_status"),
    BT::BidirectionalPort<std::string>("final_status"),
    BT::InputPort<std::string>("failure_reason"),
  };
}

BT::NodeStatus ReportTrialNode::tick()
{
  if (!logger_) {
    return BT::NodeStatus::FAILURE;
  }

  TrialLogRecord record;
  getInput("trial_id", record.trial_id);
  getInput("observation_id", record.observation_id);
  getInput("tree_name", record.tree_name);
  getInput("backend_name", record.backend_name);
  getInput("recover_status", record.recover_status);
  getInput("hypotheses", record.hypotheses);
  getInput("attempted_hypothesis_ids", record.attempted_hypothesis_ids);
  getInput("selected_hypothesis_id", record.selected_hypothesis_id);
  getInput("execute_status", record.execute_status);
  getInput("failure_reason", record.failure_reason);

  record.final_status = final_status_from_execute(record.execute_status);
  setOutput("final_status", record.final_status);
  logger_->write(record);
  return BT::NodeStatus::SUCCESS;
}

}  // namespace ghost_mgg_bt
