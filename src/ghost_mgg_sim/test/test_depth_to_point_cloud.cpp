#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "ghost_mgg_sim/depth_failure_injector.hpp"
#include "ghost_mgg_sim/depth_to_mono8.hpp"
#include "ghost_mgg_sim/depth_to_point_cloud.hpp"

namespace
{

sensor_msgs::msg::Image make_float_depth_image(uint32_t width, uint32_t height, float value)
{
  sensor_msgs::msg::Image depth;
  depth.header.frame_id = "d435_depth_optical_frame";
  depth.height = height;
  depth.width = width;
  depth.encoding = "32FC1";
  depth.step = width * sizeof(float);
  depth.data.resize(static_cast<size_t>(height) * depth.step);

  for (size_t offset = 0; offset < depth.data.size(); offset += sizeof(float)) {
    std::memcpy(depth.data.data() + offset, &value, sizeof(float));
  }
  return depth;
}

sensor_msgs::msg::Image make_mono8_mask(
  uint32_t width,
  uint32_t height,
  const std::vector<std::pair<uint32_t, uint32_t>> & active_pixels)
{
  sensor_msgs::msg::Image mask;
  mask.header.frame_id = "d435_depth_optical_frame";
  mask.height = height;
  mask.width = width;
  mask.encoding = "mono8";
  mask.step = width;
  mask.data.assign(static_cast<size_t>(height) * mask.step, 0u);

  for (const auto & [u, v] : active_pixels) {
    mask.data[static_cast<size_t>(v) * mask.step + u] = 255u;
  }
  return mask;
}

float read_depth_value(const sensor_msgs::msg::Image & depth, uint32_t u, uint32_t v)
{
  float value = 0.0F;
  const size_t offset = static_cast<size_t>(v) * depth.step + static_cast<size_t>(u) * sizeof(float);
  std::memcpy(&value, depth.data.data() + offset, sizeof(float));
  return value;
}

}  // namespace

TEST(DepthToPointCloud, ConvertsFiniteDepthPixelsWithPinholeIntrinsics)
{
  sensor_msgs::msg::Image depth;
  depth.header.frame_id = "d435_depth_optical_frame";
  depth.height = 2;
  depth.width = 2;
  depth.encoding = "32FC1";
  depth.step = depth.width * sizeof(float);
  depth.data.resize(depth.height * depth.step);
  auto * data = reinterpret_cast<float *>(depth.data.data());
  data[0] = 1.0F;
  data[1] = 0.0F;
  data[2] = std::numeric_limits<float>::quiet_NaN();
  data[3] = 2.0F;

  sensor_msgs::msg::CameraInfo info;
  info.header.frame_id = "d435_depth_optical_frame";
  info.width = 2;
  info.height = 2;
  info.k[0] = 2.0;
  info.k[2] = 0.0;
  info.k[4] = 2.0;
  info.k[5] = 0.0;

  const auto cloud = ghost_mgg_sim::depth_to_point_cloud(depth, info);

  EXPECT_EQ(cloud.header.frame_id, "d435_depth_optical_frame");
  EXPECT_EQ(cloud.width, 2u);
  EXPECT_EQ(cloud.height, 1u);
  EXPECT_EQ(cloud.fields.size(), 3u);
}

TEST(DepthToPointCloud, SupportsPixelStrideForLightweightRvizDisplay)
{
  auto depth = make_float_depth_image(4, 4, 1.0F);

  sensor_msgs::msg::CameraInfo info;
  info.header.frame_id = "d435_depth_optical_frame";
  info.width = 4;
  info.height = 4;
  info.k[0] = 2.0;
  info.k[2] = 0.0;
  info.k[4] = 2.0;
  info.k[5] = 0.0;

  const auto cloud = ghost_mgg_sim::depth_to_point_cloud(depth, info, 2u);

  EXPECT_EQ(cloud.width, 4u);
  EXPECT_EQ(cloud.height, 1u);
  EXPECT_EQ(cloud.row_step, cloud.point_step * cloud.width);
}

TEST(DepthToPointCloud, RejectsUnsupportedDepthEncoding)
{
  sensor_msgs::msg::Image depth;
  depth.encoding = "16UC1";
  sensor_msgs::msg::CameraInfo info;

  EXPECT_THROW(ghost_mgg_sim::depth_to_point_cloud(depth, info), std::invalid_argument);
}

TEST(DepthToMono8, ConvertsFloatDepthToStablePreviewImage)
{
  sensor_msgs::msg::Image depth;
  depth.header.frame_id = "d435_depth_optical_frame";
  depth.height = 1;
  depth.width = 4;
  depth.encoding = "32FC1";
  depth.step = depth.width * sizeof(float);
  depth.data.resize(depth.height * depth.step);
  auto * data = reinterpret_cast<float *>(depth.data.data());
  data[0] = 0.5F;
  data[1] = 1.0F;
  data[2] = 1.5F;
  data[3] = std::numeric_limits<float>::quiet_NaN();

  const auto preview = ghost_mgg_sim::depth_to_mono8(depth, 0.5F, 1.5F);

  ASSERT_EQ(preview.encoding, "mono8");
  ASSERT_EQ(preview.height, 1u);
  ASSERT_EQ(preview.width, 4u);
  ASSERT_EQ(preview.step, 4u);
  ASSERT_EQ(preview.data.size(), 4u);
  EXPECT_EQ(preview.data[0], 255u);
  EXPECT_EQ(preview.data[1], 128u);
  EXPECT_EQ(preview.data[2], 0u);
  EXPECT_EQ(preview.data[3], 0u);
}

TEST(DepthToMono8, RejectsInvalidDepthRange)
{
  sensor_msgs::msg::Image depth;
  depth.encoding = "32FC1";

  EXPECT_THROW(ghost_mgg_sim::depth_to_mono8(depth, 1.0F, 1.0F), std::invalid_argument);
}

TEST(DepthFailureInjector, HoleModeInvalidatesOnlyConfiguredRoi)
{
  auto depth = make_float_depth_image(4, 4, 1.0F);
  ghost_mgg_sim::DepthFailureInjectionConfig config;
  config.mode = ghost_mgg_sim::DepthFailureMode::kHole;
  config.roi_center_u_ratio = 0.5;
  config.roi_center_v_ratio = 0.5;
  config.roi_width_ratio = 0.5;
  config.roi_height_ratio = 0.5;

  const auto result = ghost_mgg_sim::inject_depth_failure(depth, config);

  EXPECT_EQ(result.corrupted_depth.encoding, "32FC1");
  EXPECT_EQ(result.hole_mask.encoding, "mono8");
  EXPECT_EQ(result.table_leakage_mask.encoding, "mono8");
  EXPECT_EQ(result.summary.roi_pixels, 4u);
  EXPECT_EQ(result.summary.hole_pixels, 4u);
  EXPECT_EQ(result.summary.table_leakage_pixels, 0u);
  EXPECT_TRUE(std::isnan(read_depth_value(result.corrupted_depth, 1, 1)));
  EXPECT_TRUE(std::isnan(read_depth_value(result.corrupted_depth, 2, 2)));
  EXPECT_FLOAT_EQ(read_depth_value(result.corrupted_depth, 0, 0), 1.0F);
  EXPECT_EQ(result.hole_mask.data[1 * result.hole_mask.step + 1], 255u);
  EXPECT_EQ(result.hole_mask.data[0], 0u);
}

TEST(DepthFailureInjector, MixedModeProducesHoleAndTableLeakageEvidence)
{
  auto depth = make_float_depth_image(4, 4, 1.0F);
  ghost_mgg_sim::DepthFailureInjectionConfig config;
  config.mode = ghost_mgg_sim::DepthFailureMode::kMixed;
  config.roi_center_u_ratio = 0.5;
  config.roi_center_v_ratio = 0.5;
  config.roi_width_ratio = 0.5;
  config.roi_height_ratio = 0.5;
  config.table_leak_depth_m = 1.25F;

  const auto result = ghost_mgg_sim::inject_depth_failure(depth, config);

  EXPECT_EQ(result.summary.roi_pixels, 4u);
  EXPECT_EQ(result.summary.hole_pixels, 2u);
  EXPECT_EQ(result.summary.table_leakage_pixels, 2u);
  EXPECT_GT(result.summary.hole_ratio, 0.0);
  EXPECT_GT(result.summary.table_leakage_ratio, 0.0);
  EXPECT_EQ(std::count(result.hole_mask.data.begin(), result.hole_mask.data.end(), 255u), 2);
  EXPECT_EQ(std::count(result.table_leakage_mask.data.begin(), result.table_leakage_mask.data.end(), 255u), 2);
  EXPECT_NE(
    ghost_mgg_sim::evidence_summary_to_json(result.summary).find("\"failure_mode\":\"mixed\""),
    std::string::npos);
}

TEST(DepthFailureInjector, TargetMaskModeUsesOnlyNonzeroMaskPixels)
{
  auto depth = make_float_depth_image(4, 4, 1.0F);
  auto mask = make_mono8_mask(4, 4, {{0u, 0u}, {3u, 1u}, {2u, 3u}});
  ghost_mgg_sim::DepthFailureInjectionConfig config;
  config.mode = ghost_mgg_sim::DepthFailureMode::kHole;

  const auto result = ghost_mgg_sim::inject_depth_failure(depth, mask, config);

  EXPECT_EQ(result.summary.evidence_source, "target_mask");
  EXPECT_EQ(result.summary.roi_pixels, 3u);
  EXPECT_EQ(result.summary.hole_pixels, 3u);
  EXPECT_EQ(result.summary.table_leakage_pixels, 0u);
  EXPECT_TRUE(std::isnan(read_depth_value(result.corrupted_depth, 0, 0)));
  EXPECT_TRUE(std::isnan(read_depth_value(result.corrupted_depth, 3, 1)));
  EXPECT_TRUE(std::isnan(read_depth_value(result.corrupted_depth, 2, 3)));
  EXPECT_FLOAT_EQ(read_depth_value(result.corrupted_depth, 1, 1), 1.0F);
  EXPECT_NE(
    ghost_mgg_sim::evidence_summary_to_json(result.summary).find("\"evidence_source\":\"target_mask\""),
    std::string::npos);
}

TEST(DepthFailureInjector, TargetMaskModeRejectsMismatchedMaskDimensions)
{
  auto depth = make_float_depth_image(4, 4, 1.0F);
  auto mask = make_mono8_mask(3, 4, {{0u, 0u}});
  ghost_mgg_sim::DepthFailureInjectionConfig config;

  EXPECT_THROW(
    ghost_mgg_sim::inject_depth_failure(depth, mask, config),
    std::invalid_argument);
}

TEST(DepthFailureInjector, EdgeOnlyModeKeepsBoundaryAndInvalidatesInterior)
{
  auto depth = make_float_depth_image(5, 5, 1.0F);
  ghost_mgg_sim::DepthFailureInjectionConfig config;
  config.mode = ghost_mgg_sim::DepthFailureMode::kEdgeOnly;
  config.roi_center_u_ratio = 0.5;
  config.roi_center_v_ratio = 0.5;
  config.roi_width_ratio = 0.6;
  config.roi_height_ratio = 0.6;
  config.edge_band_pixels = 1u;

  const auto result = ghost_mgg_sim::inject_depth_failure(depth, config);

  EXPECT_EQ(result.summary.roi_pixels, 9u);
  EXPECT_EQ(result.summary.edge_pixels, 8u);
  EXPECT_EQ(result.summary.hole_pixels, 1u);
  EXPECT_TRUE(std::isnan(read_depth_value(result.corrupted_depth, 3, 3)));
  EXPECT_FLOAT_EQ(read_depth_value(result.corrupted_depth, 2, 2), 1.0F);
  EXPECT_EQ(std::count(result.edge_mask.data.begin(), result.edge_mask.data.end(), 255u), 8);
}

TEST(DepthFailureInjector, FlyingPointModeCreatesOutliersAndHoles)
{
  auto depth = make_float_depth_image(5, 5, 1.0F);
  ghost_mgg_sim::DepthFailureInjectionConfig config;
  config.mode = ghost_mgg_sim::DepthFailureMode::kFlyingPoints;
  config.roi_center_u_ratio = 0.5;
  config.roi_center_v_ratio = 0.5;
  config.roi_width_ratio = 0.6;
  config.roi_height_ratio = 0.6;
  config.flying_point_stride = 2u;
  config.flying_point_offset_m = 0.25F;

  const auto result = ghost_mgg_sim::inject_depth_failure(depth, config);

  EXPECT_EQ(result.summary.roi_pixels, 9u);
  EXPECT_GT(result.summary.flying_point_pixels, 0u);
  EXPECT_GT(result.summary.hole_pixels, 0u);
  EXPECT_EQ(
    std::count(result.flying_point_mask.data.begin(), result.flying_point_mask.data.end(), 255u),
    static_cast<int>(result.summary.flying_point_pixels));
}

TEST(DepthFailureInjector, EdgeFlyingModeCombinesBoundaryReturnsAndFlyingOutliers)
{
  auto depth = make_float_depth_image(6, 6, 1.0F);
  ghost_mgg_sim::DepthFailureInjectionConfig config;
  config.mode = ghost_mgg_sim::DepthFailureMode::kEdgeFlying;
  config.roi_center_u_ratio = 0.5;
  config.roi_center_v_ratio = 0.5;
  config.roi_width_ratio = 0.67;
  config.roi_height_ratio = 0.67;
  config.edge_band_pixels = 1u;
  config.flying_point_stride = 2u;

  const auto result = ghost_mgg_sim::inject_depth_failure(depth, config);

  EXPECT_GT(result.summary.edge_pixels, 0u);
  EXPECT_GT(result.summary.flying_point_pixels, 0u);
  EXPECT_GT(result.summary.hole_pixels, 0u);
  EXPECT_NE(
    ghost_mgg_sim::evidence_summary_to_json(result.summary).find("\"failure_mode\":\"edge_flying\""),
    std::string::npos);
}

TEST(DepthFailureInjector, BiasedPatchModeOffsetsAllActiveTargetDepth)
{
  auto depth = make_float_depth_image(4, 4, 1.0F);
  auto mask = make_mono8_mask(4, 4, {{1u, 1u}, {2u, 2u}});
  ghost_mgg_sim::DepthFailureInjectionConfig config;
  config.mode = ghost_mgg_sim::DepthFailureMode::kBiasedPatch;
  config.biased_depth_offset_m = -0.10F;

  const auto result = ghost_mgg_sim::inject_depth_failure(depth, mask, config);

  EXPECT_EQ(result.summary.biased_depth_pixels, 2u);
  EXPECT_EQ(result.summary.hole_pixels, 0u);
  EXPECT_FLOAT_EQ(read_depth_value(result.corrupted_depth, 1, 1), 0.90F);
  EXPECT_FLOAT_EQ(read_depth_value(result.corrupted_depth, 2, 2), 0.90F);
  EXPECT_FLOAT_EQ(read_depth_value(result.corrupted_depth, 0, 0), 1.0F);
}

TEST(DepthFailureInjector, ReflectiveModeProducesMixedReflectiveEvidence)
{
  auto depth = make_float_depth_image(5, 5, 1.0F);
  ghost_mgg_sim::DepthFailureInjectionConfig config;
  config.mode = ghost_mgg_sim::DepthFailureMode::kReflective;
  config.roi_center_u_ratio = 0.5;
  config.roi_center_v_ratio = 0.5;
  config.roi_width_ratio = 0.6;
  config.roi_height_ratio = 0.6;

  const auto result = ghost_mgg_sim::inject_depth_failure(depth, config);

  EXPECT_GT(result.summary.hole_pixels, 0u);
  EXPECT_GT(result.summary.flying_point_pixels, 0u);
  EXPECT_GT(result.summary.biased_depth_pixels, 0u);
  EXPECT_NE(
    ghost_mgg_sim::evidence_summary_to_json(result.summary).find("\"failure_mode\":\"reflective\""),
    std::string::npos);
}
