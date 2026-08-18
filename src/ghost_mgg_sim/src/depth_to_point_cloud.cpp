#include "ghost_mgg_sim/depth_to_point_cloud.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <sensor_msgs/msg/point_field.hpp>

namespace ghost_mgg_sim
{
namespace
{

constexpr const char * kDepth32FC1 = "32FC1";
constexpr uint32_t kFloatBytes = sizeof(float);
constexpr uint32_t kPointStep = 3 * kFloatBytes;

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

sensor_msgs::msg::PointField make_field(const std::string & name, uint32_t offset)
{
  sensor_msgs::msg::PointField field;
  field.name = name;
  field.offset = offset;
  field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  field.count = 1;
  return field;
}

void validate_inputs(
  const sensor_msgs::msg::Image & depth,
  const sensor_msgs::msg::CameraInfo & camera_info)
{
  if (depth.encoding != kDepth32FC1) {
    throw std::invalid_argument("depth_to_point_cloud requires 32FC1 depth image");
  }
  if (depth.is_bigendian) {
    throw std::invalid_argument("depth_to_point_cloud does not support big-endian depth images");
  }
  if (depth.width == 0 || depth.height == 0) {
    throw std::invalid_argument("depth image width and height must be nonzero");
  }
  if (camera_info.width != depth.width || camera_info.height != depth.height) {
    throw std::invalid_argument("CameraInfo dimensions must match depth image dimensions");
  }
  if (depth.step < depth.width * kFloatBytes) {
    throw std::invalid_argument("depth image row step is smaller than width * sizeof(float)");
  }
  if (depth.data.size() < static_cast<size_t>(depth.step) * depth.height) {
    throw std::invalid_argument("depth image data is smaller than step * height");
  }
  if (camera_info.k[0] == 0.0 || camera_info.k[4] == 0.0) {
    throw std::invalid_argument("CameraInfo fx and fy must be nonzero");
  }
}

}  // namespace

sensor_msgs::msg::PointCloud2 depth_to_point_cloud(
  const sensor_msgs::msg::Image & depth,
  const sensor_msgs::msg::CameraInfo & camera_info,
  uint32_t pixel_stride)
{
  validate_inputs(depth, camera_info);

  const double fx = camera_info.k[0];
  const double cx = camera_info.k[2];
  const double fy = camera_info.k[4];
  const double cy = camera_info.k[5];
  const uint32_t stride = std::max<uint32_t>(1u, pixel_stride);

  std::vector<float> xyz;
  xyz.reserve(
    static_cast<size_t>((depth.width + stride - 1u) / stride) *
    static_cast<size_t>((depth.height + stride - 1u) / stride) * 3u);

  for (uint32_t v = 0; v < depth.height; v += stride) {
    for (uint32_t u = 0; u < depth.width; u += stride) {
      const size_t offset = static_cast<size_t>(v) * depth.step + static_cast<size_t>(u) * kFloatBytes;
      const float z = read_float32(depth.data, offset);
      if (!std::isfinite(z) || z <= 0.0F) {
        continue;
      }

      xyz.push_back(static_cast<float>((static_cast<double>(u) - cx) * z / fx));
      xyz.push_back(static_cast<float>((static_cast<double>(v) - cy) * z / fy));
      xyz.push_back(z);
    }
  }

  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header = depth.header;
  if (cloud.header.frame_id.empty()) {
    cloud.header.frame_id = camera_info.header.frame_id;
  }
  cloud.height = 1;
  cloud.width = static_cast<uint32_t>(xyz.size() / 3);
  cloud.fields = {
    make_field("x", 0),
    make_field("y", kFloatBytes),
    make_field("z", 2 * kFloatBytes),
  };
  cloud.is_bigendian = false;
  cloud.point_step = kPointStep;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.is_dense = false;
  cloud.data.resize(cloud.row_step);

  for (size_t i = 0; i < xyz.size(); ++i) {
    write_float32(cloud.data, i * sizeof(float), xyz[i]);
  }

  return cloud;
}

}  // namespace ghost_mgg_sim
