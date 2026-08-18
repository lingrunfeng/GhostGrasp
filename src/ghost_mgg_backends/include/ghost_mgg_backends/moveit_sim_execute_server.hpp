#pragma once

#include <atomic>
#include <optional>
#include <string>
#include <vector>

#include <control_msgs/action/gripper_command.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "ghost_mgg_backends/execute_status.hpp"
#include "ghost_mgg_interfaces/action/execute_grasp.hpp"
#include "ghost_mgg_interfaces/msg/geometry_hypothesis.hpp"
#include "ghost_mgg_interfaces/msg/grasp_candidate.hpp"

namespace ghost_mgg_backends
{

using MoveItExecuteGrasp = ghost_mgg_interfaces::action::ExecuteGrasp;
using GeometryHypothesis = ghost_mgg_interfaces::msg::GeometryHypothesis;
using GraspCandidate = ghost_mgg_interfaces::msg::GraspCandidate;
using GripperCommand = control_msgs::action::GripperCommand;
using GripperTrajectory = control_msgs::action::FollowJointTrajectory;

std::optional<GraspCandidate> select_first_valid_grasp_candidate(
  const GeometryHypothesis & hypothesis);

std::string pose_reference_frame_for_target(
  const geometry_msgs::msg::PoseStamped & target,
  const std::string & fallback_frame);

moveit_msgs::msg::CollisionObject make_proxy_collision_object(
  const GeometryHypothesis & hypothesis,
  const std::string & id_prefix);

moveit_msgs::msg::CollisionObject make_padded_proxy_collision_object(
  const GeometryHypothesis & hypothesis,
  const std::string & id_prefix,
  double padding_m);

std::vector<moveit_msgs::msg::CollisionObject> make_m2_scene_obstacle_collision_objects(
  const std::string & active_hypothesis_id,
  const std::string & id_prefix,
  double padding_m);

moveit_msgs::msg::CollisionObject make_table_collision_object(
  const std::string & frame_id,
  const std::string & object_id,
  double x,
  double y,
  double z,
  double size_x,
  double size_y,
  double size_z);

GripperCommand::Goal make_gripper_command_goal(double position, double max_effort);

GripperTrajectory::Goal make_gripper_trajectory_goal(
  double command_position,
  double duration_sec);

bool should_wait_for_gripper_command_result(
  const std::string & stage,
  bool wait_for_close_result);

std::optional<std::string> m2_model_name_for_hypothesis_id(
  const std::string & hypothesis_id);

std::optional<double> parse_gz_model_center_z(
  const std::string & gz_model_output);

bool target_lift_delta_satisfies(
  double initial_center_z,
  double observed_center_z,
  double min_lift_z_delta);

class MoveItSimExecuteServer : public rclcpp::Node
{
public:
  using GoalHandleExecuteGrasp = rclcpp_action::ServerGoalHandle<MoveItExecuteGrasp>;

  explicit MoveItSimExecuteServer(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const MoveItExecuteGrasp::Goal> goal);

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle);

  void handle_accepted(const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle);
  void execute(const std::shared_ptr<GoalHandleExecuteGrasp> goal_handle);

  void publish_feedback(
    const std::shared_ptr<GoalHandleExecuteGrasp> & goal_handle,
    const std::string & stage,
    double progress) const;

  rclcpp_action::Server<MoveItExecuteGrasp>::SharedPtr action_server_;
  rclcpp_action::Client<GripperTrajectory>::SharedPtr gripper_client_;
  std::atomic_bool execute_active_{false};
};

}  // namespace ghost_mgg_backends
