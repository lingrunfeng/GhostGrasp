#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include <control_msgs/action/gripper_command.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include "ghost_mgg_backends/moveit_sim_execute_server.hpp"
#include "ghost_mgg_interfaces/msg/geometry_hypothesis.hpp"
#include "ghost_mgg_interfaces/msg/grasp_candidate.hpp"

namespace
{
using GeometryHypothesis = ghost_mgg_interfaces::msg::GeometryHypothesis;
using GraspCandidate = ghost_mgg_interfaces::msg::GraspCandidate;

GeometryHypothesis make_box_hypothesis()
{
  GeometryHypothesis hypothesis;
  hypothesis.hypothesis_id = "red_cube";
  hypothesis.shape_type = GeometryHypothesis::SHAPE_BOX;
  hypothesis.pose_base.header.frame_id = "world";
  hypothesis.pose_base.pose.position.x = 0.03;
  hypothesis.pose_base.pose.position.y = 0.065;
  hypothesis.pose_base.pose.position.z = 0.7525;
  hypothesis.pose_base.pose.orientation.w = 1.0;
  hypothesis.dimensions_m.x = 0.025;
  hypothesis.dimensions_m.y = 0.030;
  hypothesis.dimensions_m.z = 0.035;
  return hypothesis;
}
}  // namespace

TEST(MoveItSimExecuteServer, SelectsFirstValidGraspCandidate)
{
  auto hypothesis = make_box_hypothesis();

  GraspCandidate rejected;
  rejected.grasp_id = "rejected";
  rejected.validation_state = GraspCandidate::VALIDATION_REJECTED;

  GraspCandidate valid;
  valid.grasp_id = "valid";
  valid.validation_state = GraspCandidate::VALIDATION_VALID;
  valid.grasp_pose.header.frame_id = "world";
  valid.pregrasp_pose.header.frame_id = "world";

  hypothesis.grasp_candidates = {rejected, valid};

  const auto selected = ghost_mgg_backends::select_first_valid_grasp_candidate(hypothesis);

  ASSERT_TRUE(selected.has_value());
  EXPECT_EQ(selected->grasp_id, "valid");
}

TEST(MoveItSimExecuteServer, RejectsHypothesisWithoutValidGraspCandidate)
{
  auto hypothesis = make_box_hypothesis();

  GraspCandidate unknown;
  unknown.grasp_id = "unknown";
  unknown.validation_state = GraspCandidate::VALIDATION_UNKNOWN;
  hypothesis.grasp_candidates = {unknown};

  const auto selected = ghost_mgg_backends::select_first_valid_grasp_candidate(hypothesis);

  EXPECT_FALSE(selected.has_value());
}

TEST(MoveItSimExecuteServer, UsesTargetFrameForPositionOnlyTargets)
{
  geometry_msgs::msg::PoseStamped target;
  target.header.frame_id = "world";

  EXPECT_EQ(
    ghost_mgg_backends::pose_reference_frame_for_target(target, "base_link"),
    "world");

  target.header.frame_id.clear();
  EXPECT_EQ(
    ghost_mgg_backends::pose_reference_frame_for_target(target, "base_link"),
    "base_link");
}

TEST(MoveItSimExecuteServer, BuildsBoxProxyCollisionObject)
{
  const auto hypothesis = make_box_hypothesis();

  const auto object = ghost_mgg_backends::make_proxy_collision_object(hypothesis, "trial_123");

  EXPECT_EQ(object.header.frame_id, "world");
  EXPECT_EQ(object.id, "trial_123_red_cube_proxy");
  EXPECT_EQ(object.operation, object.ADD);
  ASSERT_EQ(object.primitives.size(), 1u);
  ASSERT_EQ(object.primitive_poses.size(), 1u);
  EXPECT_EQ(object.primitives[0].type, shape_msgs::msg::SolidPrimitive::BOX);
  ASSERT_EQ(object.primitives[0].dimensions.size(), 3u);
  EXPECT_DOUBLE_EQ(object.primitives[0].dimensions[0], 0.025);
  EXPECT_DOUBLE_EQ(object.primitives[0].dimensions[1], 0.030);
  EXPECT_DOUBLE_EQ(object.primitives[0].dimensions[2], 0.035);
  EXPECT_DOUBLE_EQ(object.primitive_poses[0].position.x, 0.03);
  EXPECT_DOUBLE_EQ(object.primitive_poses[0].position.y, 0.065);
  EXPECT_DOUBLE_EQ(object.primitive_poses[0].position.z, 0.7525);
}

TEST(MoveItSimExecuteServer, BuildsCylinderProxyCollisionObject)
{
  auto hypothesis = make_box_hypothesis();
  hypothesis.hypothesis_id = "blue_cylinder";
  hypothesis.shape_type = GeometryHypothesis::SHAPE_CYLINDER;
  hypothesis.dimensions_m.x = 0.026;
  hypothesis.dimensions_m.y = 0.030;
  hypothesis.dimensions_m.z = 0.040;

  const auto object = ghost_mgg_backends::make_proxy_collision_object(hypothesis, "trial_123");

  EXPECT_EQ(object.id, "trial_123_blue_cylinder_proxy");
  ASSERT_EQ(object.primitives.size(), 1u);
  EXPECT_EQ(object.primitives[0].type, shape_msgs::msg::SolidPrimitive::CYLINDER);
  ASSERT_EQ(object.primitives[0].dimensions.size(), 2u);
  EXPECT_DOUBLE_EQ(
    object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT],
    0.040);
  EXPECT_DOUBLE_EQ(
    object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS],
    0.015);
}

TEST(MoveItSimExecuteServer, BuildsPaddedProxyCollisionObject)
{
  const auto hypothesis = make_box_hypothesis();

  const auto object = ghost_mgg_backends::make_padded_proxy_collision_object(
    hypothesis, "trial_123", 0.006);

  ASSERT_EQ(object.primitives.size(), 1u);
  EXPECT_EQ(object.primitives[0].type, shape_msgs::msg::SolidPrimitive::BOX);
  ASSERT_EQ(object.primitives[0].dimensions.size(), 3u);
  EXPECT_DOUBLE_EQ(object.primitives[0].dimensions[0], 0.037);
  EXPECT_DOUBLE_EQ(object.primitives[0].dimensions[1], 0.042);
  EXPECT_DOUBLE_EQ(object.primitives[0].dimensions[2], 0.047);
}

TEST(MoveItSimExecuteServer, BuildsM2ObstacleObjectsExcludingActiveTarget)
{
  const auto objects = ghost_mgg_backends::make_m2_scene_obstacle_collision_objects(
    "mask_extrusion_red_cube", "trial_123_obstacle", 0.006);

  ASSERT_EQ(objects.size(), 3u);
  std::vector<std::string> object_ids;
  for (const auto & object : objects) {
    object_ids.push_back(object.id);
  }
  EXPECT_EQ(
    std::find(
      object_ids.begin(), object_ids.end(),
      "trial_123_obstacle_mask_extrusion_red_cube_proxy"),
    object_ids.end());
  EXPECT_NE(
    std::find(
      object_ids.begin(), object_ids.end(),
      "trial_123_obstacle_mask_extrusion_glass_block_proxy"),
    object_ids.end());
  EXPECT_NE(
    std::find(
      object_ids.begin(), object_ids.end(),
      "trial_123_obstacle_mask_extrusion_blue_cylinder_proxy"),
    object_ids.end());
  EXPECT_NE(
    std::find(
      object_ids.begin(), object_ids.end(),
      "trial_123_obstacle_mask_extrusion_green_cylinder_proxy"),
    object_ids.end());
}

TEST(MoveItSimExecuteServer, BuildsM2ObstacleObjectsExcludingShortM4TargetIds)
{
  const auto objects = ghost_mgg_backends::make_m2_scene_obstacle_collision_objects(
    "blue_cylinder", "trial_123_obstacle", 0.006);

  ASSERT_EQ(objects.size(), 3u);
  std::vector<std::string> object_ids;
  for (const auto & object : objects) {
    object_ids.push_back(object.id);
  }
  EXPECT_EQ(
    std::find(
      object_ids.begin(), object_ids.end(),
      "trial_123_obstacle_mask_extrusion_blue_cylinder_proxy"),
    object_ids.end());
  EXPECT_NE(
    std::find(
      object_ids.begin(), object_ids.end(),
      "trial_123_obstacle_mask_extrusion_red_cube_proxy"),
    object_ids.end());
}

TEST(MoveItSimExecuteServer, BuildsGripperOpenAndCloseCommands)
{
  const auto open_goal = ghost_mgg_backends::make_gripper_command_goal(0.15, 30.0);
  const auto close_goal = ghost_mgg_backends::make_gripper_command_goal(-0.70, 45.0);

  EXPECT_DOUBLE_EQ(open_goal.command.position, 0.15);
  EXPECT_DOUBLE_EQ(open_goal.command.max_effort, 30.0);
  EXPECT_DOUBLE_EQ(close_goal.command.position, -0.70);
  EXPECT_DOUBLE_EQ(close_goal.command.max_effort, 45.0);
}

TEST(MoveItSimExecuteServer, DoesNotBlockOnCloseGripperResultByDefault)
{
  EXPECT_TRUE(ghost_mgg_backends::should_wait_for_gripper_command_result("open_gripper", false));
  EXPECT_FALSE(ghost_mgg_backends::should_wait_for_gripper_command_result("close_gripper", false));
  EXPECT_TRUE(ghost_mgg_backends::should_wait_for_gripper_command_result("close_gripper", true));
}

TEST(MoveItSimExecuteServer, ExpandsScalarGripperCommandToGroupTrajectory)
{
  const auto open_goal = ghost_mgg_backends::make_gripper_trajectory_goal(0.15, 0.30);
  ASSERT_EQ(open_goal.trajectory.joint_names.size(), 6u);
  EXPECT_EQ(open_goal.trajectory.joint_names[0], "gripper_controller");
  EXPECT_EQ(open_goal.trajectory.joint_names[5], "gripper_right3_to_gripper_right1");
  ASSERT_EQ(open_goal.trajectory.points.size(), 1u);
  EXPECT_EQ(open_goal.trajectory.points[0].time_from_start.sec, 0);
  EXPECT_EQ(open_goal.trajectory.points[0].time_from_start.nanosec, 300000000);
  EXPECT_EQ(
    open_goal.trajectory.points[0].positions,
    (std::vector<double>{0.15, 0.15, -0.15, -0.15, -0.15, 0.15}));

  const auto close_goal = ghost_mgg_backends::make_gripper_trajectory_goal(-0.70, 0.30);
  EXPECT_EQ(
    close_goal.trajectory.points[0].positions,
    (std::vector<double>{-0.70, -0.70, 0.50, 0.70, 0.70, -0.50}));

  const auto m2_contact_goal = ghost_mgg_backends::make_gripper_trajectory_goal(-0.38, 0.30);
  EXPECT_EQ(
    m2_contact_goal.trajectory.points[0].positions,
    (std::vector<double>{-0.38, -0.38, 0.38, 0.38, 0.38, -0.38}));
}

TEST(MoveItSimExecuteServer, MapsM2HypothesesToGazeboTargetModels)
{
  EXPECT_EQ(
    ghost_mgg_backends::m2_model_name_for_hypothesis_id("mask_extrusion_glass_block"),
    std::optional<std::string>("glass_block"));
  EXPECT_EQ(
    ghost_mgg_backends::m2_model_name_for_hypothesis_id("glass_block"),
    std::optional<std::string>("glass_block"));
  EXPECT_EQ(
    ghost_mgg_backends::m2_model_name_for_hypothesis_id("mask_extrusion_red_cube"),
    std::optional<std::string>("red_cube"));
  EXPECT_EQ(
    ghost_mgg_backends::m2_model_name_for_hypothesis_id("red_cube"),
    std::optional<std::string>("red_cube"));
  EXPECT_EQ(
    ghost_mgg_backends::m2_model_name_for_hypothesis_id("mask_extrusion_blue_cylinder"),
    std::optional<std::string>("blue_cylinder"));
  EXPECT_EQ(
    ghost_mgg_backends::m2_model_name_for_hypothesis_id("blue_cylinder"),
    std::optional<std::string>("blue_cylinder"));
  EXPECT_EQ(
    ghost_mgg_backends::m2_model_name_for_hypothesis_id("mask_extrusion_green_cylinder"),
    std::optional<std::string>("green_cylinder"));
  EXPECT_EQ(
    ghost_mgg_backends::m2_model_name_for_hypothesis_id("green_cylinder"),
    std::optional<std::string>("green_cylinder"));
  EXPECT_FALSE(
    ghost_mgg_backends::m2_model_name_for_hypothesis_id("unknown_hypothesis").has_value());
}

TEST(MoveItSimExecuteServer, ParsesGazeboModelPoseAndChecksLiftDelta)
{
  const std::string output =
    "Requesting state for world [ghost_mgg_m2_visual]...\n"
    "Model: [42]\n"
    "  - Name: glass_block\n"
    "  - Pose [ XYZ (m) ] [ RPY (rad) ]:\n"
    "    [0.002000 0.100000 0.772900]\n"
    "    [0.000000 0.000000 -0.350000]\n";

  const auto z = ghost_mgg_backends::parse_gz_model_center_z(output);
  ASSERT_TRUE(z.has_value());
  EXPECT_DOUBLE_EQ(*z, 0.772900);
  EXPECT_TRUE(ghost_mgg_backends::target_lift_delta_satisfies(0.7525, *z, 0.010));
  EXPECT_FALSE(ghost_mgg_backends::target_lift_delta_satisfies(0.7525, 0.7580, 0.010));
  EXPECT_FALSE(ghost_mgg_backends::parse_gz_model_center_z("no pose here").has_value());
}
