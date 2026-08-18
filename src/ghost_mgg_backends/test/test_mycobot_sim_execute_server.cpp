#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "ghost_mgg_backends/mycobot_sim_execute_server.hpp"

namespace
{
using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
}  // namespace

TEST(MyCobotSimExecuteServer, BuildsThreeNamedTrajectoryGoalsFromPreset)
{
  const ghost_mgg_backends::MyCobotTrajectoryPreset preset{
    {0.0, -0.6, 0.9, 0.0, -0.3, 0.0},
    {0.1, -0.5, 1.0, 0.0, -0.4, 0.1},
    {0.0, -0.7, 0.8, 0.0, -0.2, 0.0},
    1.25};

  const auto goals = ghost_mgg_backends::make_mycobot_trajectory_goals(
    ghost_mgg_backends::default_mycobot_joint_names(), preset);

  ASSERT_EQ(goals.size(), 3u);
  EXPECT_EQ(goals[0].trajectory.joint_names, ghost_mgg_backends::default_mycobot_joint_names());
  EXPECT_EQ(goals[1].trajectory.joint_names, ghost_mgg_backends::default_mycobot_joint_names());
  EXPECT_EQ(goals[2].trajectory.joint_names, ghost_mgg_backends::default_mycobot_joint_names());
  ASSERT_EQ(goals[0].trajectory.points.size(), 1u);
  ASSERT_EQ(goals[1].trajectory.points.size(), 1u);
  ASSERT_EQ(goals[2].trajectory.points.size(), 1u);
  EXPECT_EQ(goals[0].trajectory.points[0].positions, preset.pregrasp_positions);
  EXPECT_EQ(goals[1].trajectory.points[0].positions, preset.grasp_positions);
  EXPECT_EQ(goals[2].trajectory.points[0].positions, preset.retreat_positions);
  EXPECT_EQ(goals[0].trajectory.points[0].time_from_start.sec, 1);
  EXPECT_EQ(goals[0].trajectory.points[0].time_from_start.nanosec, 250000000u);
}

TEST(MyCobotSimExecuteServer, ValidatesPresetJointArrayLengths)
{
  const ghost_mgg_backends::MyCobotTrajectoryPreset valid{
    {0.0, -0.6, 0.9, 0.0, -0.3, 0.0},
    {0.1, -0.5, 1.0, 0.0, -0.4, 0.1},
    {0.0, -0.7, 0.8, 0.0, -0.2, 0.0},
    1.0};
  const ghost_mgg_backends::MyCobotTrajectoryPreset invalid{
    {0.0, -0.6, 0.9},
    valid.grasp_positions,
    valid.retreat_positions,
    1.0};

  const auto valid_result = ghost_mgg_backends::validate_mycobot_preset(
    ghost_mgg_backends::default_mycobot_joint_names(), valid);
  const auto invalid_result = ghost_mgg_backends::validate_mycobot_preset(
    ghost_mgg_backends::default_mycobot_joint_names(), invalid);

  EXPECT_TRUE(valid_result.valid);
  EXPECT_TRUE(valid_result.failure_reason.empty());
  EXPECT_FALSE(invalid_result.valid);
  EXPECT_NE(invalid_result.failure_reason.find("pregrasp_positions"), std::string::npos);
  EXPECT_NE(invalid_result.failure_reason.find("expected 6"), std::string::npos);
}

TEST(MyCobotSimExecuteServer, MapsFollowJointTrajectoryFailuresToExecuteOutcomes)
{
  FollowJointTrajectory::Result ok_result;
  ok_result.error_code = FollowJointTrajectory::Result::SUCCESSFUL;
  const auto ok = ghost_mgg_backends::map_mycobot_trajectory_outcome(
    rclcpp_action::ResultCode::SUCCEEDED, ok_result, "grasp");

  EXPECT_TRUE(ok.succeeded);
  EXPECT_EQ(ok.status, ghost_mgg_backends::kExecuteStatusSucceeded);
  EXPECT_TRUE(ok.failure_reason.empty());

  FollowJointTrajectory::Result tolerance_result;
  tolerance_result.error_code = FollowJointTrajectory::Result::GOAL_TOLERANCE_VIOLATED;
  tolerance_result.error_string = "joint tolerance violated";
  const auto tolerance_failure = ghost_mgg_backends::map_mycobot_trajectory_outcome(
    rclcpp_action::ResultCode::SUCCEEDED, tolerance_result, "retreat");

  EXPECT_FALSE(tolerance_failure.succeeded);
  EXPECT_EQ(tolerance_failure.status, ghost_mgg_backends::kExecuteStatusFailed);
  EXPECT_NE(tolerance_failure.failure_reason.find("retreat"), std::string::npos);
  EXPECT_NE(tolerance_failure.failure_reason.find("joint tolerance violated"), std::string::npos);

  const auto canceled = ghost_mgg_backends::map_mycobot_trajectory_outcome(
    rclcpp_action::ResultCode::CANCELED, ok_result, "pregrasp");
  EXPECT_FALSE(canceled.succeeded);
  EXPECT_EQ(canceled.status, ghost_mgg_backends::kExecuteStatusCanceled);
}

TEST(MyCobotSimExecuteServer, CancelsPendingTrajectoryGoalsWithoutSavedGoalHandle)
{
  const auto source_path =
    std::filesystem::path(GHOST_MGG_BACKENDS_SOURCE_DIR) / "src/mycobot_sim_execute_server.cpp";
  ASSERT_TRUE(std::filesystem::exists(source_path));

  std::ifstream in(source_path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const auto source = buffer.str();

  EXPECT_NE(source.find("cancel_active_or_pending_trajectories"), std::string::npos);
  EXPECT_NE(source.find("async_cancel_all_goals"), std::string::npos);
}
