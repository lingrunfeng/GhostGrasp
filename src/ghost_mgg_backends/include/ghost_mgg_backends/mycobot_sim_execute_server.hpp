#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "ghost_mgg_backends/execute_status.hpp"
#include "ghost_mgg_interfaces/action/execute_grasp.hpp"

namespace ghost_mgg_backends
{

using MyCobotFollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
using MyCobotExecuteGrasp = ghost_mgg_interfaces::action::ExecuteGrasp;

struct MyCobotTrajectoryPreset
{
  std::vector<double> pregrasp_positions;
  std::vector<double> grasp_positions;
  std::vector<double> retreat_positions;
  double segment_duration_sec = 1.0;
};

struct MyCobotPresetValidation
{
  bool valid = false;
  std::string failure_reason;
};

struct MyCobotExecuteOutcome
{
  bool succeeded = false;
  std::uint8_t status = kExecuteStatusFailed;
  std::string failure_reason;
};

std::vector<std::string> default_mycobot_joint_names();
MyCobotTrajectoryPreset default_mycobot_trajectory_preset();

MyCobotPresetValidation validate_mycobot_preset(
  const std::vector<std::string> & joint_names,
  const MyCobotTrajectoryPreset & preset);

std::vector<MyCobotFollowJointTrajectory::Goal> make_mycobot_trajectory_goals(
  const std::vector<std::string> & joint_names,
  const MyCobotTrajectoryPreset & preset);

MyCobotExecuteOutcome map_mycobot_trajectory_outcome(
  rclcpp_action::ResultCode result_code,
  const MyCobotFollowJointTrajectory::Result & trajectory_result,
  const std::string & stage);

class MyCobotSimExecuteServer : public rclcpp::Node
{
public:
  using GoalHandleExecuteGrasp = rclcpp_action::ServerGoalHandle<MyCobotExecuteGrasp>;
  using FollowGoalHandle = rclcpp_action::ClientGoalHandle<MyCobotFollowJointTrajectory>;

  explicit MyCobotSimExecuteServer(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const MyCobotExecuteGrasp::Goal> goal);

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle);

  void handle_accepted(const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle);
  void execute(const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle);

  MyCobotExecuteOutcome wait_for_trajectory_server(
    const std::shared_ptr<GoalHandleExecuteGrasp> & goal_handle,
    const rclcpp::Time & deadline);

  MyCobotExecuteOutcome execute_trajectory_stage(
    const std::shared_ptr<GoalHandleExecuteGrasp> & execute_goal_handle,
    const MyCobotFollowJointTrajectory::Goal & trajectory_goal,
    const std::string & stage,
    const rclcpp::Time & deadline);

  MyCobotTrajectoryPreset preset_from_parameters() const;
  std::vector<std::string> joint_names_from_parameters() const;

  void cancel_active_or_pending_trajectories();
  void clear_active_trajectory();

  void publish_feedback(
    const std::shared_ptr<GoalHandleExecuteGrasp> & goal_handle,
    const std::string & stage,
    double progress) const;

  rclcpp_action::Server<MyCobotExecuteGrasp>::SharedPtr action_server_;
  rclcpp_action::Client<MyCobotFollowJointTrajectory>::SharedPtr trajectory_client_;

  std::atomic_bool execute_active_{false};
  mutable std::mutex active_trajectory_mutex_;
  FollowGoalHandle::SharedPtr active_trajectory_goal_;
};

}  // namespace ghost_mgg_backends
