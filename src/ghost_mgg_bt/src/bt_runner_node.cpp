#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <behaviortree_cpp/bt_factory.h>
#include <rclcpp/rclcpp.hpp>

#include "ghost_mgg_bt/backend_registry.hpp"
#include "ghost_mgg_bt/register_nodes.hpp"
#include "ghost_mgg_bt/trial_logger.hpp"
#include "ghost_mgg_interfaces/msg/geometry_hypothesis.hpp"

namespace
{
std::string default_share_file(
  const std::string & package_name,
  const std::string & relative_path)
{
  return (std::filesystem::path(ament_index_cpp::get_package_share_directory(package_name)) /
         relative_path).string();
}
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("m0_bt_runner");

  const auto registry_path = node->declare_parameter<std::string>(
    "backend_registry_path",
    default_share_file("ghost_mgg_bt", "config/backend_registry.yaml"));
  const auto tree_path = node->declare_parameter<std::string>(
    "tree_path",
    default_share_file("ghost_mgg_bt", "trees/m0_dummy_recovery.xml"));
  const auto log_dir = node->declare_parameter<std::string>(
    "trial_log_dir",
    (std::filesystem::current_path() / "log/ghost_mgg_trials").string());
  const auto backend_name = node->declare_parameter<std::string>("backend_name", "dummy");
  const auto executor_name = node->declare_parameter<std::string>("executor_name", "dummy");
  const auto target_label = node->declare_parameter<std::string>("target_label", "m0_dummy_target");
  const auto shape_hint = node->declare_parameter<std::string>("shape_hint", "unknown");
  const auto recover_timeout_sec = node->declare_parameter<double>("recover_timeout_sec", 2.0);
  const auto execute_timeout_sec = node->declare_parameter<double>("execute_timeout_sec", 2.0);
  const auto max_hypotheses = node->declare_parameter<int>("max_hypotheses", 3);

  const auto registry = std::make_shared<ghost_mgg_bt::BackendRegistry>(
    ghost_mgg_bt::BackendRegistry::from_file(registry_path));
  const auto logger = std::make_shared<ghost_mgg_bt::TrialLogger>(log_dir);

  BT::BehaviorTreeFactory factory;
  ghost_mgg_bt::register_m0_nodes(factory, node, registry, logger, executor_name);

  auto blackboard = BT::Blackboard::create();
  blackboard->set("backend_name", backend_name);
  blackboard->set("target_label", target_label);
  blackboard->set("shape_hint", shape_hint);
  blackboard->set("recover_timeout_sec", recover_timeout_sec);
  blackboard->set("execute_timeout_sec", execute_timeout_sec);
  blackboard->set("max_hypotheses", static_cast<unsigned int>(max_hypotheses));
  blackboard->set("trial_id", std::string{});
  blackboard->set("observation_id", std::string{});
  blackboard->set("hypothesis_index", 0);
  blackboard->set("attempted_hypothesis_ids", std::vector<std::string>{});
  blackboard->set(
    "hypotheses",
    std::vector<ghost_mgg_interfaces::msg::GeometryHypothesis>{});
  blackboard->set("final_status", std::string{});

  auto tree = factory.createTreeFromFile(tree_path, blackboard);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  BT::NodeStatus status = BT::NodeStatus::IDLE;
  const auto sleep_duration = std::chrono::milliseconds(20);
  while (rclcpp::ok()) {
    status = tree.tickOnce();
    executor.spin_some();
    if (status == BT::NodeStatus::SUCCESS || status == BT::NodeStatus::FAILURE) {
      break;
    }
    std::this_thread::sleep_for(sleep_duration);
  }

  RCLCPP_INFO(
    node->get_logger(),
    "M0 BT finished with status %s",
    BT::toStr(status, true).c_str());
  rclcpp::shutdown();
  return status == BT::NodeStatus::SUCCESS ? 0 : 1;
}
