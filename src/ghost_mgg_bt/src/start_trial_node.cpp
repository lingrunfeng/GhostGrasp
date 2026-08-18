#include "ghost_mgg_bt/start_trial_node.hpp"

#include <chrono>
#include <sstream>
#include <string>
#include <vector>

#include "ghost_mgg_interfaces/msg/observation_ref.hpp"

namespace ghost_mgg_bt
{

StartTrialNode::StartTrialNode(const std::string & name, const BT::NodeConfig & config)
: BT::SyncActionNode(name, config)
{
}

BT::PortsList StartTrialNode::providedPorts()
{
  return {
    BT::BidirectionalPort<std::string>("trial_id"),
    BT::BidirectionalPort<std::string>("observation_id"),
    BT::OutputPort<ghost_mgg_interfaces::msg::ObservationRef>("observation_ref"),
    BT::OutputPort<int>("hypothesis_index"),
    BT::OutputPort<std::vector<std::string>>("attempted_hypothesis_ids"),
  };
}

BT::NodeStatus StartTrialNode::tick()
{
  std::string trial_id;
  std::string observation_id;
  getInput("trial_id", trial_id);
  getInput("observation_id", observation_id);

  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  if (trial_id.empty()) {
    trial_id = "trial_m0_" + std::to_string(stamp);
  }
  if (observation_id.empty()) {
    observation_id = "obs_m0_" + std::to_string(stamp);
  }

  ghost_mgg_interfaces::msg::ObservationRef observation;
  observation.observation_id = observation_id;
  observation.cache_namespace = "m0_dummy";
  observation.rgb_frame_id = "camera_color_optical_frame";
  observation.depth_frame_id = "camera_depth_optical_frame";
  observation.mask_frame_id = "camera_color_optical_frame";
  observation.max_age_sec = 1.0;
  observation.has_rgb = false;
  observation.has_depth = false;
  observation.has_ir = false;
  observation.has_mask = false;
  observation.has_camera_info = false;

  setOutput("trial_id", trial_id);
  setOutput("observation_id", observation_id);
  setOutput("observation_ref", observation);
  setOutput("hypothesis_index", 0);
  setOutput("attempted_hypothesis_ids", std::vector<std::string>{});
  return BT::NodeStatus::SUCCESS;
}

}  // namespace ghost_mgg_bt
