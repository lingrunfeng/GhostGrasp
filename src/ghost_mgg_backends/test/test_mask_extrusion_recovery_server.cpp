#include <gtest/gtest.h>

#include <cstddef>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ghost_mgg_backends/m2_scene_primitives.hpp"
#include "ghost_mgg_backends/mask_extrusion_recovery_server.hpp"
#include "ghost_mgg_backends/recover_status.hpp"
#include "ghost_mgg_interfaces/action/recover_geometry.hpp"
#include "ghost_mgg_interfaces/msg/geometry_hypothesis.hpp"
#include "ghost_mgg_interfaces/msg/grasp_candidate.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

namespace
{
using GeometryHypothesis = ghost_mgg_interfaces::msg::GeometryHypothesis;
using GraspCandidate = ghost_mgg_interfaces::msg::GraspCandidate;
using RecoverGeometry = ghost_mgg_interfaces::action::RecoverGeometry;

struct SceneObject
{
  std::string hypothesis_id;
  std::uint8_t shape_type;
  std::string model_name;
  std::string visual_name;
  double score_total;
  double confidence;
};

constexpr double kExpectedTopGraspOffsetAboveObjectTop = 0.005;
constexpr double kExpectedTopPregraspLift = 0.080;

std::string read_visual_world_sdf()
{
  const auto path = std::filesystem::path(GHOST_MGG_BACKENDS_SOURCE_DIR)
    .parent_path() / "ghost_mgg_sim" / "worlds" / "m2_tabletop_visual.sdf";
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open M2 visual world SDF: " + path.string());
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::string read_model_sdf(const std::string & model_name)
{
  const auto path = std::filesystem::path(GHOST_MGG_BACKENDS_SOURCE_DIR)
    .parent_path() / "ghost_mgg_sim" / "models" / model_name / "model.sdf";
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open model SDF: " + path.string());
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::string include_block(const std::string & sdf, const std::string & model_name)
{
  const auto include_start = sdf.find("<name>" + model_name + "</name>");
  if (include_start == std::string::npos) {
    throw std::runtime_error("missing include block: " + model_name);
  }
  const auto include_end = sdf.find("</include>", include_start);
  if (include_end == std::string::npos) {
    throw std::runtime_error("unterminated include block: " + model_name);
  }
  return sdf.substr(include_start, include_end - include_start);
}

std::string visual_block(const std::string & sdf, const std::string & visual_name)
{
  const auto visual_start = sdf.find("<visual name=\"" + visual_name + "\">");
  if (visual_start == std::string::npos) {
    throw std::runtime_error("missing visual block: " + visual_name);
  }
  const auto visual_end = sdf.find("</visual>", visual_start);
  if (visual_end == std::string::npos) {
    throw std::runtime_error("unterminated visual block: " + visual_name);
  }
  return sdf.substr(visual_start, visual_end - visual_start);
}

std::vector<double> parse_pose_xyz(const std::string & block)
{
  const auto pose_start = block.find("<pose>");
  const auto pose_end = block.find("</pose>", pose_start);
  if (pose_start == std::string::npos || pose_end == std::string::npos) {
    throw std::runtime_error("visual block does not contain pose");
  }
  std::istringstream pose_stream(
    block.substr(pose_start + std::string{"<pose>"}.size(), pose_end - pose_start));
  std::vector<double> pose(3);
  pose_stream >> pose[0] >> pose[1] >> pose[2];
  return pose;
}

std::vector<double> parse_dimensions_xyz(const std::string & block, std::uint8_t shape_type)
{
  if (shape_type == GeometryHypothesis::SHAPE_CYLINDER) {
    const auto radius_start = block.find("<radius>");
    const auto radius_end = block.find("</radius>", radius_start);
    const auto length_start = block.find("<length>");
    const auto length_end = block.find("</length>", length_start);
    if (
      radius_start == std::string::npos || radius_end == std::string::npos ||
      length_start == std::string::npos || length_end == std::string::npos)
    {
      throw std::runtime_error("cylinder visual block lacks radius/length");
    }
    const auto radius_text = block.substr(
      radius_start + std::string{"<radius>"}.size(), radius_end - radius_start);
    const auto length_text = block.substr(
      length_start + std::string{"<length>"}.size(), length_end - length_start);
    const auto diameter = 2.0 * std::stod(radius_text);
    return {diameter, diameter, std::stod(length_text)};
  }

  const auto size_start = block.find("<size>");
  const auto size_end = block.find("</size>", size_start);
  if (size_start == std::string::npos || size_end == std::string::npos) {
    throw std::runtime_error("box visual block does not contain size");
  }
  std::istringstream size_stream(
    block.substr(size_start + std::string{"<size>"}.size(), size_end - size_start));
  std::vector<double> dimensions(3);
  size_stream >> dimensions[0] >> dimensions[1] >> dimensions[2];
  return dimensions;
}

rclcpp::ExecutorOptions executor_options_for(
  const std::shared_ptr<rclcpp::Context> & context)
{
  rclcpp::ExecutorOptions options;
  options.context = context;
  return options;
}

std::shared_ptr<rclcpp::Context> make_initialized_context()
{
  auto context = std::make_shared<rclcpp::Context>();
  context->init(0, nullptr);
  return context;
}

class ScopedActionServer
{
public:
  explicit ScopedActionServer(
    const std::string & action_name,
    const std::string & preferred_hypothesis_id = "",
    bool strict_preferred_hypothesis = false)
  : context_(make_initialized_context()),
    executor_(executor_options_for(context_))
  {
    rclcpp::NodeOptions server_options;
    server_options.context(context_);
    server_options.parameter_overrides({
      rclcpp::Parameter("action_name", action_name),
      rclcpp::Parameter("preferred_hypothesis_id", preferred_hypothesis_id),
      rclcpp::Parameter("strict_preferred_hypothesis", strict_preferred_hypothesis),
      rclcpp::Parameter("response_delay_ms", 1),
    });
    server_ = std::make_shared<ghost_mgg_backends::MaskExtrusionRecoveryServer>(server_options);

    rclcpp::NodeOptions client_options;
    client_options.context(context_);
    client_node_ = rclcpp::Node::make_shared(
      "mask_extrusion_recovery_client_test", client_options);

    executor_.add_node(server_);
    executor_.add_node(client_node_);
    spin_thread_ = std::thread([this]() { executor_.spin(); });
  }

  ~ScopedActionServer()
  {
    executor_.cancel();
    if (spin_thread_.joinable()) {
      spin_thread_.join();
    }
    executor_.remove_node(client_node_);
    executor_.remove_node(server_);
    if (context_->is_valid()) {
      context_->shutdown("mask extrusion action test finished");
    }
  }

  rclcpp::Node::SharedPtr client_node() const
  {
    return client_node_;
  }

private:
  std::shared_ptr<rclcpp::Context> context_;
  rclcpp::executors::MultiThreadedExecutor executor_;
  std::shared_ptr<ghost_mgg_backends::MaskExtrusionRecoveryServer> server_;
  rclcpp::Node::SharedPtr client_node_;
  std::thread spin_thread_;
};
}  // namespace

TEST(MaskExtrusionBaseline, BuildsDeterministicRankedM2Hypotheses)
{
  const auto hypotheses = ghost_mgg_backends::make_m2_mask_extrusion_hypotheses(10);
  const auto world_sdf = read_visual_world_sdf();
  const std::vector<SceneObject> scene_objects{
    {"mask_extrusion_glass_block", GeometryHypothesis::SHAPE_BOX, "glass_block", "glass_block_visual", 0.82, 0.80},
    {"mask_extrusion_red_cube", GeometryHypothesis::SHAPE_BOX, "red_cube", "red_cube_visual", 0.74, 0.72},
    {"mask_extrusion_blue_cylinder", GeometryHypothesis::SHAPE_CYLINDER, "blue_cylinder", "blue_cylinder_visual", 0.66, 0.65},
    {"mask_extrusion_green_cylinder", GeometryHypothesis::SHAPE_CYLINDER, "green_cylinder", "green_cylinder_visual", 0.58, 0.60}};

  ASSERT_EQ(hypotheses.size(), 4u);

  for (std::size_t i = 0; i < hypotheses.size(); ++i) {
    const auto pose = parse_pose_xyz(include_block(world_sdf, scene_objects[i].model_name));
    const auto model_sdf = read_model_sdf(scene_objects[i].model_name);
    const auto dimensions = parse_dimensions_xyz(
      visual_block(model_sdf, scene_objects[i].visual_name), scene_objects[i].shape_type);

    EXPECT_EQ(hypotheses[i].hypothesis_id, scene_objects[i].hypothesis_id);
    EXPECT_EQ(hypotheses[i].shape_type, scene_objects[i].shape_type);
    EXPECT_EQ(hypotheses[i].pose_base.header.frame_id, "world");
    EXPECT_EQ(hypotheses[i].pose_camera.header.frame_id, "world");
    EXPECT_NEAR(hypotheses[i].pose_base.pose.position.x, pose[0], 1e-9);
    EXPECT_NEAR(hypotheses[i].pose_base.pose.position.y, pose[1], 1e-9);
    EXPECT_NEAR(hypotheses[i].pose_base.pose.position.z, pose[2], 1e-9);
    EXPECT_NEAR(hypotheses[i].pose_camera.pose.position.x, pose[0], 1e-9);
    EXPECT_NEAR(hypotheses[i].pose_camera.pose.position.y, pose[1], 1e-9);
    EXPECT_NEAR(hypotheses[i].pose_camera.pose.position.z, pose[2], 1e-9);
    EXPECT_NEAR(hypotheses[i].dimensions_m.x, dimensions[0], 1e-9);
    EXPECT_NEAR(hypotheses[i].dimensions_m.y, dimensions[1], 1e-9);
    EXPECT_NEAR(hypotheses[i].dimensions_m.z, dimensions[2], 1e-9);
    EXPECT_NEAR(hypotheses[i].score.total, scene_objects[i].score_total, 1e-9);
    EXPECT_NEAR(hypotheses[i].confidence, scene_objects[i].confidence, 1e-9);
    EXPECT_NEAR(hypotheses[i].uncertainty, 1.0 - scene_objects[i].confidence, 1e-9);
    EXPECT_EQ(
      hypotheses[i].provenance,
      "mask_extrusion_baseline:m2_configured_scene_prior");
  }
  EXPECT_GT(hypotheses[0].score.total, hypotheses[1].score.total);
  EXPECT_GT(hypotheses[1].score.total, hypotheses[2].score.total);

  for (const auto & hypothesis : hypotheses) {
    EXPECT_EQ(hypothesis.validation_state, GeometryHypothesis::VALIDATION_VALID);
    EXPECT_GT(hypothesis.confidence, 0.0);
    EXPECT_LT(hypothesis.uncertainty, 1.0);
    ASSERT_EQ(hypothesis.grasp_candidates.size(), 4u);
    const std::vector<std::string> expected_yaw_suffixes{
      "top_yaw_000", "top_yaw_045", "top_yaw_090", "top_yaw_135"};
    for (std::size_t grasp_index = 0; grasp_index < hypothesis.grasp_candidates.size();
      ++grasp_index)
    {
      const auto & grasp = hypothesis.grasp_candidates[grasp_index];
      EXPECT_NE(grasp.grasp_id.find(expected_yaw_suffixes[grasp_index]), std::string::npos);
      EXPECT_EQ(grasp.validation_state, GraspCandidate::VALIDATION_VALID);
      EXPECT_EQ(grasp.grasp_type, GraspCandidate::GRASP_TYPE_TOP);
      EXPECT_EQ(grasp.grasp_pose.header.frame_id, "world");
      EXPECT_EQ(grasp.pregrasp_pose.header.frame_id, "world");
      EXPECT_NEAR(grasp.grasp_pose.pose.position.x, hypothesis.pose_base.pose.position.x, 1e-9);
      EXPECT_NEAR(grasp.grasp_pose.pose.position.y, hypothesis.pose_base.pose.position.y, 1e-9);
      EXPECT_NEAR(
        grasp.grasp_pose.pose.position.z,
        hypothesis.pose_base.pose.position.z + 0.5 * hypothesis.dimensions_m.z +
          kExpectedTopGraspOffsetAboveObjectTop,
        1e-9);
      EXPECT_NEAR(grasp.pregrasp_pose.pose.position.x, grasp.grasp_pose.pose.position.x, 1e-9);
      EXPECT_NEAR(grasp.pregrasp_pose.pose.position.y, grasp.grasp_pose.pose.position.y, 1e-9);
      EXPECT_NEAR(
        grasp.pregrasp_pose.pose.position.z - grasp.grasp_pose.pose.position.z,
        kExpectedTopPregraspLift,
        1e-9);
      EXPECT_NEAR(grasp.approach_vector.x, 0.0, 1e-9);
      EXPECT_NEAR(grasp.approach_vector.y, 0.0, 1e-9);
      EXPECT_NEAR(grasp.approach_vector.z, -1.0, 1e-9);
      EXPECT_GT(grasp.gripper_width_m, 0.0);
      const auto q_norm = std::sqrt(
        grasp.grasp_pose.pose.orientation.x * grasp.grasp_pose.pose.orientation.x +
        grasp.grasp_pose.pose.orientation.y * grasp.grasp_pose.pose.orientation.y +
        grasp.grasp_pose.pose.orientation.z * grasp.grasp_pose.pose.orientation.z +
        grasp.grasp_pose.pose.orientation.w * grasp.grasp_pose.pose.orientation.w);
      EXPECT_NEAR(q_norm, 1.0, 1e-9);
      EXPECT_NEAR(grasp.pregrasp_pose.pose.orientation.x, grasp.grasp_pose.pose.orientation.x, 1e-9);
      EXPECT_NEAR(grasp.pregrasp_pose.pose.orientation.y, grasp.grasp_pose.pose.orientation.y, 1e-9);
      EXPECT_NEAR(grasp.pregrasp_pose.pose.orientation.z, grasp.grasp_pose.pose.orientation.z, 1e-9);
      EXPECT_NEAR(grasp.pregrasp_pose.pose.orientation.w, grasp.grasp_pose.pose.orientation.w, 1e-9);
      if (grasp_index > 0) {
        EXPECT_LT(grasp.score, hypothesis.grasp_candidates[grasp_index - 1].score);
      }
    }
    EXPECT_NEAR(hypothesis.grasp_candidates.front().grasp_pose.pose.orientation.x, -std::sqrt(0.5), 1e-6);
    EXPECT_NEAR(hypothesis.grasp_candidates.front().grasp_pose.pose.orientation.y, 0.0, 1e-6);
    EXPECT_NEAR(hypothesis.grasp_candidates.front().grasp_pose.pose.orientation.z, 0.0, 1e-6);
    EXPECT_NEAR(hypothesis.grasp_candidates.front().grasp_pose.pose.orientation.w, std::sqrt(0.5), 1e-6);
  }
}

TEST(MaskExtrusionBaseline, RespectsMaxHypotheses)
{
  EXPECT_EQ(ghost_mgg_backends::make_m2_mask_extrusion_hypotheses(0).size(), 0u);
  EXPECT_EQ(ghost_mgg_backends::make_m2_mask_extrusion_hypotheses(1).size(), 1u);
  EXPECT_EQ(ghost_mgg_backends::make_m2_mask_extrusion_hypotheses(2).size(), 2u);
  EXPECT_EQ(ghost_mgg_backends::make_m2_mask_extrusion_hypotheses(99).size(), 4u);
}

TEST(MaskExtrusionBaseline, ExposesRecoverStatusConstants)
{
  EXPECT_EQ(ghost_mgg_backends::kRecoverStatusSucceeded, 1);
  EXPECT_EQ(ghost_mgg_backends::kRecoverStatusCanceled, 5);
}

TEST(MaskExtrusionRecoveryServer, ReturnsScenePriorHypothesesThroughActionContract)
{
  using namespace std::chrono_literals;

  const auto action_name = "/test/geometry_backends/mask_extrusion/recover";
  ScopedActionServer server(action_name);
  auto client = rclcpp_action::create_client<RecoverGeometry>(
    server.client_node(), action_name);

  ASSERT_TRUE(client->wait_for_action_server(2s));

  RecoverGeometry::Goal goal;
  goal.trial_id = "trial_m2_mask";
  goal.observation_id = "obs_m2_mask";
  goal.backend_name = "mask_extrusion";
  goal.max_hypotheses = 2;

  auto goal_handle_future = client->async_send_goal(goal);
  ASSERT_EQ(goal_handle_future.wait_for(2s), std::future_status::ready);
  auto goal_handle = goal_handle_future.get();
  ASSERT_NE(goal_handle, nullptr);

  auto result_future = client->async_get_result(goal_handle);
  ASSERT_EQ(result_future.wait_for(2s), std::future_status::ready);
  const auto wrapped_result = result_future.get();

  ASSERT_EQ(wrapped_result.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_NE(wrapped_result.result, nullptr);
  EXPECT_EQ(wrapped_result.result->status, ghost_mgg_backends::kRecoverStatusSucceeded);
  EXPECT_EQ(wrapped_result.result->backend_name, "mask_extrusion");
  ASSERT_EQ(wrapped_result.result->hypotheses.size(), 2u);
  EXPECT_EQ(wrapped_result.result->hypotheses.front().hypothesis_id, "mask_extrusion_glass_block");
  EXPECT_EQ(
    wrapped_result.result->diagnostics,
    "mask_extrusion baseline completed using configured M2 scene prior");
}

TEST(MaskExtrusionRecoveryServer, PrioritizesConfiguredTargetHypothesis)
{
  const std::string action_name =
    "/test/mask_extrusion/recover_preferred_green";
  ScopedActionServer server(action_name, "mask_extrusion_green_cylinder");
  auto client = rclcpp_action::create_client<RecoverGeometry>(
    server.client_node(),
    action_name);
  ASSERT_TRUE(client->wait_for_action_server(std::chrono::seconds(2)));

  RecoverGeometry::Goal goal;
  goal.trial_id = "trial_preferred_green";
  goal.observation_id = "obs_preferred_green";
  goal.max_hypotheses = 1;

  auto send_options = rclcpp_action::Client<RecoverGeometry>::SendGoalOptions();
  auto goal_future = client->async_send_goal(goal, send_options);
  ASSERT_EQ(goal_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  auto goal_handle = goal_future.get();
  ASSERT_NE(goal_handle, nullptr);

  auto result_future = client->async_get_result(goal_handle);
  ASSERT_EQ(result_future.wait_for(std::chrono::seconds(3)), std::future_status::ready);

  const auto wrapped_result = result_future.get();
  ASSERT_EQ(wrapped_result.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_EQ(wrapped_result.result->hypotheses.size(), 1u);
  EXPECT_EQ(
    wrapped_result.result->hypotheses.front().hypothesis_id,
    "mask_extrusion_green_cylinder");
}

TEST(MaskExtrusionRecoveryServer, StrictPreferredTargetReturnsOnlyConfiguredHypothesis)
{
  const std::string action_name =
    "/test/mask_extrusion/recover_strict_blue";
  ScopedActionServer server(action_name, "mask_extrusion_blue_cylinder", true);
  auto client = rclcpp_action::create_client<RecoverGeometry>(
    server.client_node(),
    action_name);
  ASSERT_TRUE(client->wait_for_action_server(std::chrono::seconds(2)));

  RecoverGeometry::Goal goal;
  goal.trial_id = "trial_strict_blue";
  goal.observation_id = "obs_strict_blue";
  goal.max_hypotheses = 4;

  auto goal_future = client->async_send_goal(goal);
  ASSERT_EQ(goal_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  auto goal_handle = goal_future.get();
  ASSERT_NE(goal_handle, nullptr);

  auto result_future = client->async_get_result(goal_handle);
  ASSERT_EQ(result_future.wait_for(std::chrono::seconds(3)), std::future_status::ready);

  const auto wrapped_result = result_future.get();
  ASSERT_EQ(wrapped_result.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_EQ(wrapped_result.result->hypotheses.size(), 1u);
  EXPECT_EQ(
    wrapped_result.result->hypotheses.front().hypothesis_id,
    "mask_extrusion_blue_cylinder");
}
