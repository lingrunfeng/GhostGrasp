#pragma once

#include <sensor_msgs/msg/image.hpp>

namespace ghost_mgg_sim
{

sensor_msgs::msg::Image depth_to_mono8(
  const sensor_msgs::msg::Image & depth,
  float min_depth_m,
  float max_depth_m);

}  // namespace ghost_mgg_sim
