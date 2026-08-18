#include "ghost_mgg_backends/m2_scene_primitives.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <geometry_msgs/msg/quaternion.hpp>

#include "ghost_mgg_interfaces/msg/grasp_candidate.hpp"
#include "ghost_mgg_interfaces/msg/score_breakdown.hpp"

namespace ghost_mgg_backends
{
namespace
{
using GraspCandidate = ghost_mgg_interfaces::msg::GraspCandidate;
using ScoreBreakdown = ghost_mgg_interfaces::msg::ScoreBreakdown;

constexpr double kPi = 3.14159265358979323846;
constexpr double kTopGraspOffsetAboveObjectTop = 0.005;
constexpr double kTopPregraspLift = 0.080;

ScoreBreakdown make_score_breakdown(double total, double visual, double prior)
{
  ScoreBreakdown score;
  score.visual = visual;
  score.failure = 0.0;
  score.depth = 0.0;
  score.physical = 0.80;
  score.grasp = total - 0.05;
  score.prior = prior;
  score.total = total;
  return score;
}

geometry_msgs::msg::Quaternion top_down_orientation(double yaw_rad)
{
  geometry_msgs::msg::Quaternion orientation;
  const auto half_yaw = 0.5 * yaw_rad;
  const auto yaw_sin = std::sin(half_yaw);
  const auto yaw_cos = std::cos(half_yaw);
  const auto roll_sin = -std::sqrt(0.5);
  const auto roll_cos = std::sqrt(0.5);

  orientation.x = yaw_cos * roll_sin;
  orientation.y = yaw_sin * roll_sin;
  orientation.z = yaw_sin * roll_cos;
  orientation.w = yaw_cos * roll_cos;
  return orientation;
}

GraspCandidate make_top_grasp(
  const std::string & hypothesis_id,
  const std::string & yaw_suffix,
  double x,
  double y,
  double z,
  double height,
  double width,
  double yaw_rad,
  double score)
{
  GraspCandidate grasp;
  grasp.grasp_id = hypothesis_id + "_" + yaw_suffix;
  grasp.grasp_pose.header.frame_id = "world";
  grasp.pregrasp_pose.header.frame_id = "world";
  grasp.grasp_pose.pose.position.x = x;
  grasp.grasp_pose.pose.position.y = y;
  grasp.grasp_pose.pose.position.z = z + 0.5 * height + kTopGraspOffsetAboveObjectTop;
  grasp.pregrasp_pose.pose.position.x = x;
  grasp.pregrasp_pose.pose.position.y = y;
  grasp.pregrasp_pose.pose.position.z = grasp.grasp_pose.pose.position.z + kTopPregraspLift;
  grasp.grasp_pose.pose.orientation = top_down_orientation(yaw_rad);
  grasp.pregrasp_pose.pose.orientation = grasp.grasp_pose.pose.orientation;
  grasp.approach_vector.x = 0.0;
  grasp.approach_vector.y = 0.0;
  grasp.approach_vector.z = -1.0;
  grasp.gripper_width_m = width;
  grasp.grasp_type = GraspCandidate::GRASP_TYPE_TOP;
  grasp.score = score;
  grasp.validation_state = GraspCandidate::VALIDATION_VALID;
  return grasp;
}

std::vector<GraspCandidate> make_top_grasp_candidates(
  const std::string & hypothesis_id,
  double x,
  double y,
  double z,
  double height,
  double width)
{
  return {
    make_top_grasp(hypothesis_id, "top_yaw_000", x, y, z, height, width, 0.0, 0.86),
    make_top_grasp(hypothesis_id, "top_yaw_045", x, y, z, height, width, 0.25 * kPi, 0.82),
    make_top_grasp(hypothesis_id, "top_yaw_090", x, y, z, height, width, 0.50 * kPi, 0.78),
    make_top_grasp(hypothesis_id, "top_yaw_135", x, y, z, height, width, 0.75 * kPi, 0.74),
  };
}

GeometryHypothesis make_hypothesis(
  const std::string & id,
  std::uint8_t shape,
  double x,
  double y,
  double z,
  double sx,
  double sy,
  double sz,
  double total,
  double visual,
  double confidence)
{
  GeometryHypothesis hypothesis;
  hypothesis.hypothesis_id = id;
  hypothesis.shape_type = shape;
  hypothesis.pose_camera.header.frame_id = "world";
  hypothesis.pose_base.header.frame_id = "world";
  hypothesis.pose_camera.pose.orientation.w = 1.0;
  hypothesis.pose_base.pose.orientation.w = 1.0;
  hypothesis.pose_camera.pose.position.x = x;
  hypothesis.pose_camera.pose.position.y = y;
  hypothesis.pose_camera.pose.position.z = z;
  hypothesis.pose_base.pose.position.x = x;
  hypothesis.pose_base.pose.position.y = y;
  hypothesis.pose_base.pose.position.z = z;
  hypothesis.dimensions_m.x = sx;
  hypothesis.dimensions_m.y = sy;
  hypothesis.dimensions_m.z = sz;
  hypothesis.score = make_score_breakdown(total, visual, 0.55);
  hypothesis.confidence = confidence;
  hypothesis.uncertainty = 1.0 - confidence;
  hypothesis.grasp_candidates = make_top_grasp_candidates(
    id, x, y, z, sz, std::max(sx, sy) + 0.012);
  hypothesis.provenance = "mask_extrusion_baseline:m2_configured_scene_prior";
  hypothesis.validation_state = GeometryHypothesis::VALIDATION_VALID;
  return hypothesis;
}
}  // namespace

std::vector<GeometryHypothesis> make_m2_mask_extrusion_hypotheses(std::size_t max_hypotheses)
{
  const std::vector<GeometryHypothesis> all_hypotheses{
    make_hypothesis(
      "mask_extrusion_glass_block",
      GeometryHypothesis::SHAPE_BOX,
      0.002,
      0.100,
      0.7525,
      0.025,
      0.025,
      0.025,
      0.82,
      0.76,
      0.80),
    make_hypothesis(
      "mask_extrusion_red_cube",
      GeometryHypothesis::SHAPE_BOX,
      0.070172,
      0.012156,
      0.7525,
      0.025,
      0.025,
      0.025,
      0.74,
      0.70,
      0.72),
    make_hypothesis(
      "mask_extrusion_blue_cylinder",
      GeometryHypothesis::SHAPE_CYLINDER,
      0.115000,
      0.075000,
      0.7525,
      0.025,
      0.025,
      0.025,
      0.66,
      0.62,
      0.65),
    make_hypothesis(
      "mask_extrusion_green_cylinder",
      GeometryHypothesis::SHAPE_CYLINDER,
      -0.035000,
      -0.047344,
      0.7525,
      0.025,
      0.025,
      0.025,
      0.58,
      0.56,
      0.60),
  };

  const auto limit = std::min(max_hypotheses, all_hypotheses.size());
  return std::vector<GeometryHypothesis>(all_hypotheses.begin(), all_hypotheses.begin() + limit);
}

}  // namespace ghost_mgg_backends
