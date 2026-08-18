#pragma once

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace ghost_mgg_sim
{

sensor_msgs::msg::PointCloud2 depth_to_point_cloud(
  const sensor_msgs::msg::Image & depth,
  const sensor_msgs::msg::CameraInfo & camera_info,
  uint32_t pixel_stride = 1u);

}  // namespace ghost_mgg_sim
