#include "ghost_mgg_sim/depth_failure_injector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ghost_mgg_sim
{
namespace
{

constexpr const char * kDepth32FC1 = "32FC1";
constexpr const char * kMono8 = "mono8";
constexpr uint32_t kFloatBytes = sizeof(float);

struct PixelRoi
{
  uint32_t u_min = 0u;
  uint32_t u_max = 0u;
  uint32_t v_min = 0u;
  uint32_t v_max = 0u;
};

struct ActivePixelMask
{
  std::vector<uint8_t> active;
  uint32_t active_pixels = 0u;
  std::string evidence_source = "roi";
};

std::string mode_to_string(DepthFailureMode mode)
{
  switch (mode) {
    case DepthFailureMode::kDisabled:
      return "disabled";
    case DepthFailureMode::kHole:
      return "hole";
    case DepthFailureMode::kTableLeakage:
      return "table_leakage";
    case DepthFailureMode::kMixed:
      return "mixed";
    case DepthFailureMode::kEdgeOnly:
      return "edge_only";
    case DepthFailureMode::kFlyingPoints:
      return "flying_points";
    case DepthFailureMode::kEdgeFlying:
      return "edge_flying";
    case DepthFailureMode::kReflective:
      return "reflective";
    case DepthFailureMode::kBiasedPatch:
      return "biased_patch";
  }
  return "unknown";
}

void validate_depth_image(const sensor_msgs::msg::Image & depth)
{
  if (depth.encoding != kDepth32FC1) {
    throw std::invalid_argument("depth failure injector requires 32FC1 depth image");
  }
  if (depth.is_bigendian) {
    throw std::invalid_argument("depth failure injector does not support big-endian depth images");
  }
  if (depth.width == 0 || depth.height == 0) {
    throw std::invalid_argument("depth image width and height must be nonzero");
  }
  if (depth.step < depth.width * kFloatBytes) {
    throw std::invalid_argument("depth image row step is smaller than width * sizeof(float)");
  }
  if (depth.data.size() < static_cast<size_t>(depth.step) * depth.height) {
    throw std::invalid_argument("depth image data is smaller than step * height");
  }
}

void validate_target_mask(
  const sensor_msgs::msg::Image & mask,
  uint32_t expected_width,
  uint32_t expected_height)
{
  if (mask.encoding != kMono8) {
    throw std::invalid_argument("target mask must use mono8 encoding");
  }
  if (mask.is_bigendian) {
    throw std::invalid_argument("target mask must not be big-endian");
  }
  if (mask.width != expected_width || mask.height != expected_height) {
    throw std::invalid_argument("target mask dimensions must match depth image dimensions");
  }
  if (mask.step < mask.width) {
    throw std::invalid_argument("target mask row step is smaller than width");
  }
  if (mask.data.size() < static_cast<size_t>(mask.step) * mask.height) {
    throw std::invalid_argument("target mask data is smaller than step * height");
  }
}

sensor_msgs::msg::Image make_mask_like(const sensor_msgs::msg::Image & depth)
{
  sensor_msgs::msg::Image mask;
  mask.header = depth.header;
  mask.height = depth.height;
  mask.width = depth.width;
  mask.encoding = kMono8;
  mask.is_bigendian = false;
  mask.step = depth.width;
  mask.data.assign(static_cast<size_t>(mask.step) * mask.height, 0u);
  return mask;
}

uint32_t clamped_pixel_count(double ratio, uint32_t extent)
{
  const auto clamped_ratio = std::clamp(ratio, 0.0, 1.0);
  return std::max<uint32_t>(1u, static_cast<uint32_t>(std::lround(clamped_ratio * extent)));
}

PixelRoi compute_roi(const sensor_msgs::msg::Image & depth, const DepthFailureInjectionConfig & config)
{
  const uint32_t width_px = clamped_pixel_count(config.roi_width_ratio, depth.width);
  const uint32_t height_px = clamped_pixel_count(config.roi_height_ratio, depth.height);
  const int center_u = static_cast<int>(
    std::lround(std::clamp(config.roi_center_u_ratio, 0.0, 1.0) * depth.width));
  const int center_v = static_cast<int>(
    std::lround(std::clamp(config.roi_center_v_ratio, 0.0, 1.0) * depth.height));

  int u_min = center_u - static_cast<int>(width_px / 2u);
  int v_min = center_v - static_cast<int>(height_px / 2u);
  u_min = std::clamp(u_min, 0, std::max<int>(0, static_cast<int>(depth.width) - static_cast<int>(width_px)));
  v_min = std::clamp(v_min, 0, std::max<int>(0, static_cast<int>(depth.height) - static_cast<int>(height_px)));

  PixelRoi roi;
  roi.u_min = static_cast<uint32_t>(u_min);
  roi.v_min = static_cast<uint32_t>(v_min);
  roi.u_max = std::min(depth.width, roi.u_min + width_px);
  roi.v_max = std::min(depth.height, roi.v_min + height_px);
  return roi;
}

ActivePixelMask make_active_pixels_from_roi(
  const sensor_msgs::msg::Image & depth,
  const DepthFailureInjectionConfig & config)
{
  ActivePixelMask active_mask;
  active_mask.active.assign(static_cast<size_t>(depth.width) * depth.height, 0u);
  active_mask.evidence_source = "roi";

  const auto roi = compute_roi(depth, config);
  for (uint32_t v = roi.v_min; v < roi.v_max; ++v) {
    for (uint32_t u = roi.u_min; u < roi.u_max; ++u) {
      active_mask.active[static_cast<size_t>(v) * depth.width + u] = 1u;
      ++active_mask.active_pixels;
    }
  }

  return active_mask;
}

ActivePixelMask make_active_pixels_from_target_mask(
  const sensor_msgs::msg::Image & depth,
  const sensor_msgs::msg::Image & target_mask)
{
  validate_target_mask(target_mask, depth.width, depth.height);

  ActivePixelMask active_mask;
  active_mask.active.assign(static_cast<size_t>(depth.width) * depth.height, 0u);
  active_mask.evidence_source = "target_mask";

  for (uint32_t v = 0; v < target_mask.height; ++v) {
    for (uint32_t u = 0; u < target_mask.width; ++u) {
      const size_t mask_offset = static_cast<size_t>(v) * target_mask.step + u;
      if (target_mask.data[mask_offset] == 0u) {
        continue;
      }

      active_mask.active[static_cast<size_t>(v) * depth.width + u] = 1u;
      ++active_mask.active_pixels;
    }
  }

  return active_mask;
}

float read_float32(const std::vector<uint8_t> & data, size_t offset)
{
  float value = 0.0F;
  std::memcpy(&value, data.data() + offset, sizeof(float));
  return value;
}

void write_float32(std::vector<uint8_t> & data, size_t offset, float value)
{
  std::memcpy(data.data() + offset, &value, sizeof(float));
}

bool should_make_hole(DepthFailureMode mode, uint32_t u, uint32_t v, uint32_t seed)
{
  if (mode == DepthFailureMode::kHole) {
    return true;
  }
  if (mode != DepthFailureMode::kMixed) {
    return false;
  }
  return ((u + v + seed) % 2u) == 0u;
}

bool should_make_table_leakage(DepthFailureMode mode, uint32_t u, uint32_t v, uint32_t seed)
{
  if (mode == DepthFailureMode::kTableLeakage) {
    return true;
  }
  if (mode != DepthFailureMode::kMixed) {
    return false;
  }
  return !should_make_hole(mode, u, v, seed);
}

bool is_active(const ActivePixelMask & active_mask, uint32_t width, uint32_t u, uint32_t v)
{
  return active_mask.active[static_cast<size_t>(v) * width + u] != 0u;
}

bool is_edge_band_pixel(
  const ActivePixelMask & active_mask,
  uint32_t width,
  uint32_t height,
  uint32_t u,
  uint32_t v,
  uint32_t edge_band_pixels)
{
  if (!is_active(active_mask, width, u, v)) {
    return false;
  }

  const int band = std::max<int>(1, static_cast<int>(edge_band_pixels));
  for (int dv = -band; dv <= band; ++dv) {
    for (int du = -band; du <= band; ++du) {
      const int nu = static_cast<int>(u) + du;
      const int nv = static_cast<int>(v) + dv;
      if (nu < 0 || nv < 0 || nu >= static_cast<int>(width) || nv >= static_cast<int>(height)) {
        return true;
      }
      if (!is_active(active_mask, width, static_cast<uint32_t>(nu), static_cast<uint32_t>(nv))) {
        return true;
      }
    }
  }

  return false;
}

bool should_make_flying_point(uint32_t u, uint32_t v, const DepthFailureInjectionConfig & config)
{
  const uint32_t stride = std::max<uint32_t>(1u, config.flying_point_stride);
  const uint32_t hashed = (u * 73856093u) ^ (v * 19349663u) ^ (config.pattern_seed * 83492791u);
  return (hashed % stride) == 0u;
}

float finite_or_table_depth(float original_depth, const DepthFailureInjectionConfig & config)
{
  if (std::isfinite(original_depth) && original_depth > 0.0F) {
    return original_depth;
  }
  return config.table_leak_depth_m;
}

DepthFailureInjectionResult inject_depth_failure_for_active_pixels(
  const sensor_msgs::msg::Image & depth,
  const DepthFailureInjectionConfig & config,
  const ActivePixelMask & active_mask)
{
  DepthFailureInjectionResult result;
  result.corrupted_depth = depth;
  result.hole_mask = make_mask_like(depth);
  result.table_leakage_mask = make_mask_like(depth);
  result.edge_mask = make_mask_like(depth);
  result.flying_point_mask = make_mask_like(depth);
  result.biased_depth_mask = make_mask_like(depth);
  result.summary.failure_mode = mode_to_string(config.mode);
  result.summary.evidence_source = active_mask.evidence_source;
  result.summary.total_pixels = depth.width * depth.height;
  result.summary.roi_pixels = active_mask.active_pixels;

  for (uint32_t v = 0; v < depth.height; ++v) {
    for (uint32_t u = 0; u < depth.width; ++u) {
      const size_t active_offset = static_cast<size_t>(v) * depth.width + u;
      if (active_mask.active[active_offset] == 0u) {
        continue;
      }

      const size_t depth_offset =
        static_cast<size_t>(v) * result.corrupted_depth.step + static_cast<size_t>(u) * kFloatBytes;
      const size_t mask_offset =
        static_cast<size_t>(v) * result.hole_mask.step + static_cast<size_t>(u);
      const float original_depth = read_float32(result.corrupted_depth.data, depth_offset);

      if (should_make_hole(config.mode, u, v, config.pattern_seed)) {
        write_float32(
          result.corrupted_depth.data,
          depth_offset,
          std::numeric_limits<float>::quiet_NaN());
        result.hole_mask.data[mask_offset] = 255u;
        ++result.summary.hole_pixels;
      } else if (should_make_table_leakage(config.mode, u, v, config.pattern_seed)) {
        write_float32(result.corrupted_depth.data, depth_offset, config.table_leak_depth_m);
        result.table_leakage_mask.data[mask_offset] = 255u;
        ++result.summary.table_leakage_pixels;
      } else if (config.mode == DepthFailureMode::kEdgeOnly) {
        if (is_edge_band_pixel(
            active_mask,
            depth.width,
            depth.height,
            u,
            v,
            config.edge_band_pixels))
        {
          result.edge_mask.data[mask_offset] = 255u;
          ++result.summary.edge_pixels;
        } else {
          write_float32(
            result.corrupted_depth.data,
            depth_offset,
            std::numeric_limits<float>::quiet_NaN());
          result.hole_mask.data[mask_offset] = 255u;
          ++result.summary.hole_pixels;
        }
      } else if (config.mode == DepthFailureMode::kFlyingPoints) {
        if (should_make_flying_point(u, v, config)) {
          const float sign = ((u + v + config.pattern_seed) % 2u == 0u) ? 1.0F : -1.0F;
          write_float32(
            result.corrupted_depth.data,
            depth_offset,
            finite_or_table_depth(original_depth, config) + sign * config.flying_point_offset_m);
          result.flying_point_mask.data[mask_offset] = 255u;
          ++result.summary.flying_point_pixels;
        } else {
          write_float32(
            result.corrupted_depth.data,
            depth_offset,
            std::numeric_limits<float>::quiet_NaN());
          result.hole_mask.data[mask_offset] = 255u;
          ++result.summary.hole_pixels;
        }
      } else if (config.mode == DepthFailureMode::kEdgeFlying) {
        if (is_edge_band_pixel(
            active_mask,
            depth.width,
            depth.height,
            u,
            v,
            config.edge_band_pixels))
        {
          result.edge_mask.data[mask_offset] = 255u;
          ++result.summary.edge_pixels;
        } else if (should_make_flying_point(u, v, config)) {
          const float sign = ((u + v + config.pattern_seed) % 2u == 0u) ? 1.0F : -1.0F;
          write_float32(
            result.corrupted_depth.data,
            depth_offset,
            finite_or_table_depth(original_depth, config) + sign * config.flying_point_offset_m);
          result.flying_point_mask.data[mask_offset] = 255u;
          ++result.summary.flying_point_pixels;
        } else {
          write_float32(
            result.corrupted_depth.data,
            depth_offset,
            std::numeric_limits<float>::quiet_NaN());
          result.hole_mask.data[mask_offset] = 255u;
          ++result.summary.hole_pixels;
        }
      } else if (config.mode == DepthFailureMode::kBiasedPatch) {
        write_float32(
          result.corrupted_depth.data,
          depth_offset,
          finite_or_table_depth(original_depth, config) + config.biased_depth_offset_m);
        result.biased_depth_mask.data[mask_offset] = 255u;
        ++result.summary.biased_depth_pixels;
      } else if (config.mode == DepthFailureMode::kReflective) {
        const uint32_t selector = (u + 3u * v + config.pattern_seed) % 5u;
        if (selector == 0u) {
          write_float32(
            result.corrupted_depth.data,
            depth_offset,
            std::numeric_limits<float>::quiet_NaN());
          result.hole_mask.data[mask_offset] = 255u;
          ++result.summary.hole_pixels;
        } else if (selector == 1u || selector == 2u) {
          write_float32(
            result.corrupted_depth.data,
            depth_offset,
            finite_or_table_depth(original_depth, config) + config.biased_depth_offset_m);
          result.biased_depth_mask.data[mask_offset] = 255u;
          ++result.summary.biased_depth_pixels;
        } else if (selector == 3u) {
          write_float32(
            result.corrupted_depth.data,
            depth_offset,
            finite_or_table_depth(original_depth, config) + config.flying_point_offset_m);
          result.flying_point_mask.data[mask_offset] = 255u;
          ++result.summary.flying_point_pixels;
        } else if (is_edge_band_pixel(
            active_mask,
            depth.width,
            depth.height,
            u,
            v,
            config.edge_band_pixels))
        {
          result.edge_mask.data[mask_offset] = 255u;
          ++result.summary.edge_pixels;
        }
      }
    }
  }

  for (uint32_t v = 0; v < depth.height; ++v) {
    for (uint32_t u = 0; u < depth.width; ++u) {
      const size_t depth_offset =
        static_cast<size_t>(v) * result.corrupted_depth.step + static_cast<size_t>(u) * kFloatBytes;
      const float z = read_float32(result.corrupted_depth.data, depth_offset);
      if (std::isfinite(z) && z > 0.0F) {
        ++result.summary.valid_depth_pixels;
      }
    }
  }

  const double roi_pixels = static_cast<double>(std::max<uint32_t>(1u, result.summary.roi_pixels));
  const double total_pixels = static_cast<double>(std::max<uint32_t>(1u, result.summary.total_pixels));
  result.summary.hole_ratio = static_cast<double>(result.summary.hole_pixels) / roi_pixels;
  result.summary.table_leakage_ratio =
    static_cast<double>(result.summary.table_leakage_pixels) / roi_pixels;
  result.summary.edge_ratio = static_cast<double>(result.summary.edge_pixels) / roi_pixels;
  result.summary.flying_point_ratio =
    static_cast<double>(result.summary.flying_point_pixels) / roi_pixels;
  result.summary.biased_depth_ratio =
    static_cast<double>(result.summary.biased_depth_pixels) / roi_pixels;
  result.summary.valid_depth_ratio =
    static_cast<double>(result.summary.valid_depth_pixels) / total_pixels;

  return result;
}

}  // namespace

DepthFailureMode depth_failure_mode_from_string(const std::string & mode)
{
  if (mode == "disabled") {
    return DepthFailureMode::kDisabled;
  }
  if (mode == "hole" || mode == "holes") {
    return DepthFailureMode::kHole;
  }
  if (mode == "table_leakage" || mode == "leak" || mode == "leakage") {
    return DepthFailureMode::kTableLeakage;
  }
  if (mode == "mixed") {
    return DepthFailureMode::kMixed;
  }
  if (mode == "edge_only" || mode == "edge" || mode == "edges") {
    return DepthFailureMode::kEdgeOnly;
  }
  if (mode == "flying_points" || mode == "flying" || mode == "fly") {
    return DepthFailureMode::kFlyingPoints;
  }
  if (mode == "edge_flying" || mode == "edge_and_flying" || mode == "edge_fly") {
    return DepthFailureMode::kEdgeFlying;
  }
  if (mode == "reflective" || mode == "specular") {
    return DepthFailureMode::kReflective;
  }
  if (mode == "biased_patch" || mode == "biased" || mode == "bias") {
    return DepthFailureMode::kBiasedPatch;
  }
  throw std::invalid_argument("unknown depth failure mode: " + mode);
}

DepthFailureInjectionResult inject_depth_failure(
  const sensor_msgs::msg::Image & depth,
  const DepthFailureInjectionConfig & config)
{
  validate_depth_image(depth);
  return inject_depth_failure_for_active_pixels(
    depth,
    config,
    make_active_pixels_from_roi(depth, config));
}

DepthFailureInjectionResult inject_depth_failure(
  const sensor_msgs::msg::Image & depth,
  const sensor_msgs::msg::Image & target_mask,
  const DepthFailureInjectionConfig & config)
{
  validate_depth_image(depth);
  return inject_depth_failure_for_active_pixels(
    depth,
    config,
    make_active_pixels_from_target_mask(depth, target_mask));
}

std::string evidence_summary_to_json(const DepthFailureSummary & summary)
{
  std::ostringstream json;
  json << "{";
  json << "\"failure_mode\":\"" << summary.failure_mode << "\",";
  json << "\"evidence_source\":\"" << summary.evidence_source << "\",";
  json << "\"total_pixels\":" << summary.total_pixels << ",";
  json << "\"roi_pixels\":" << summary.roi_pixels << ",";
  json << "\"valid_depth_pixels\":" << summary.valid_depth_pixels << ",";
  json << "\"hole_pixels\":" << summary.hole_pixels << ",";
  json << "\"table_leakage_pixels\":" << summary.table_leakage_pixels << ",";
  json << "\"edge_pixels\":" << summary.edge_pixels << ",";
  json << "\"flying_point_pixels\":" << summary.flying_point_pixels << ",";
  json << "\"biased_depth_pixels\":" << summary.biased_depth_pixels << ",";
  json << "\"valid_depth_ratio\":" << summary.valid_depth_ratio << ",";
  json << "\"hole_ratio\":" << summary.hole_ratio << ",";
  json << "\"table_leakage_ratio\":" << summary.table_leakage_ratio << ",";
  json << "\"edge_ratio\":" << summary.edge_ratio << ",";
  json << "\"flying_point_ratio\":" << summary.flying_point_ratio << ",";
  json << "\"biased_depth_ratio\":" << summary.biased_depth_ratio;
  json << "}";
  return json.str();
}

}  // namespace ghost_mgg_sim
