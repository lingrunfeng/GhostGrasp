#pragma once

#include <cstdint>
#include <string>

#include <sensor_msgs/msg/image.hpp>

namespace ghost_mgg_sim
{

enum class DepthFailureMode
{
  kDisabled,
  kHole,
  kTableLeakage,
  kMixed,
  kEdgeOnly,
  kFlyingPoints,
  kEdgeFlying,
  kReflective,
  kBiasedPatch,
};

struct DepthFailureInjectionConfig
{
  DepthFailureMode mode = DepthFailureMode::kMixed;
  double roi_center_u_ratio = 0.50;
  double roi_center_v_ratio = 0.58;
  double roi_width_ratio = 0.22;
  double roi_height_ratio = 0.22;
  float table_leak_depth_m = 1.20F;
  float flying_point_offset_m = 0.12F;
  float biased_depth_offset_m = -0.04F;
  uint32_t edge_band_pixels = 2u;
  uint32_t flying_point_stride = 5u;
  uint32_t pattern_seed = 0u;
};

struct DepthFailureSummary
{
  std::string failure_mode;
  std::string evidence_source = "roi";
  uint32_t total_pixels = 0u;
  uint32_t roi_pixels = 0u;
  uint32_t valid_depth_pixels = 0u;
  uint32_t hole_pixels = 0u;
  uint32_t table_leakage_pixels = 0u;
  uint32_t edge_pixels = 0u;
  uint32_t flying_point_pixels = 0u;
  uint32_t biased_depth_pixels = 0u;
  double valid_depth_ratio = 0.0;
  double hole_ratio = 0.0;
  double table_leakage_ratio = 0.0;
  double edge_ratio = 0.0;
  double flying_point_ratio = 0.0;
  double biased_depth_ratio = 0.0;
};

struct DepthFailureInjectionResult
{
  sensor_msgs::msg::Image corrupted_depth;
  sensor_msgs::msg::Image hole_mask;
  sensor_msgs::msg::Image table_leakage_mask;
  sensor_msgs::msg::Image edge_mask;
  sensor_msgs::msg::Image flying_point_mask;
  sensor_msgs::msg::Image biased_depth_mask;
  DepthFailureSummary summary;
};

DepthFailureMode depth_failure_mode_from_string(const std::string & mode);

DepthFailureInjectionResult inject_depth_failure(
  const sensor_msgs::msg::Image & depth,
  const DepthFailureInjectionConfig & config);

DepthFailureInjectionResult inject_depth_failure(
  const sensor_msgs::msg::Image & depth,
  const sensor_msgs::msg::Image & target_mask,
  const DepthFailureInjectionConfig & config);

std::string evidence_summary_to_json(const DepthFailureSummary & summary);

}  // namespace ghost_mgg_sim
