#include "ghost_mgg_sim/depth_to_mono8.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ghost_mgg_sim
{
namespace
{

constexpr const char * kDepth32FC1 = "32FC1";
constexpr const char * kMono8 = "mono8";
constexpr uint32_t kFloatBytes = sizeof(float);

float read_float32(const std::vector<uint8_t> & data, size_t offset)
{
  float value = 0.0F;
  std::memcpy(&value, data.data() + offset, sizeof(float));
  return value;
}

void validate_depth_image(const sensor_msgs::msg::Image & depth)
{
  if (depth.encoding != kDepth32FC1) {
    throw std::invalid_argument("depth_to_mono8 requires 32FC1 depth image");
  }
  if (depth.is_bigendian) {
    throw std::invalid_argument("depth_to_mono8 does not support big-endian depth images");
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

}  // namespace

sensor_msgs::msg::Image depth_to_mono8(
  const sensor_msgs::msg::Image & depth,
  float min_depth_m,
  float max_depth_m)
{
  validate_depth_image(depth);
  if (!std::isfinite(min_depth_m) || !std::isfinite(max_depth_m) || max_depth_m <= min_depth_m) {
    throw std::invalid_argument("depth_to_mono8 requires max_depth_m > min_depth_m");
  }

  sensor_msgs::msg::Image preview;
  preview.header = depth.header;
  preview.height = depth.height;
  preview.width = depth.width;
  preview.encoding = kMono8;
  preview.is_bigendian = false;
  preview.step = depth.width;
  preview.data.resize(static_cast<size_t>(preview.step) * preview.height, 0u);

  const float range = max_depth_m - min_depth_m;
  for (uint32_t v = 0; v < depth.height; ++v) {
    for (uint32_t u = 0; u < depth.width; ++u) {
      const size_t input_offset =
        static_cast<size_t>(v) * depth.step + static_cast<size_t>(u) * kFloatBytes;
      const float z = read_float32(depth.data, input_offset);
      if (!std::isfinite(z) || z <= 0.0F) {
        continue;
      }

      const float normalized = std::clamp((z - min_depth_m) / range, 0.0F, 1.0F);
      const auto gray = static_cast<uint8_t>(std::lround(255.0F * (1.0F - normalized)));
      preview.data[static_cast<size_t>(v) * preview.step + u] = gray;
    }
  }

  return preview;
}

}  // namespace ghost_mgg_sim
