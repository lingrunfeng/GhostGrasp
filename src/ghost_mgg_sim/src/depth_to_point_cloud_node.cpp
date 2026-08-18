#include <memory>
#include <algorithm>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "ghost_mgg_sim/depth_to_point_cloud.hpp"

namespace ghost_mgg_sim
{
namespace
{

class DepthToPointCloudNode : public rclcpp::Node
{
public:
  DepthToPointCloudNode()
  : Node("depth_to_point_cloud_node")
  {
    const auto depth_topic = declare_parameter<std::string>(
      "depth_topic", "/ghost_mgg/d435/depth/image_rect_raw");
    const auto camera_info_topic = declare_parameter<std::string>(
      "camera_info_topic", "/ghost_mgg/d435/depth/camera_info");
    const auto points_topic = declare_parameter<std::string>(
      "points_topic", "/ghost_mgg/d435/depth/points");
    output_frame_ = declare_parameter<std::string>(
      "output_frame", "d435_depth_optical_frame");
    pixel_stride_ = static_cast<uint32_t>(
      std::max<int>(1, declare_parameter<int>("pixel_stride", 1)));

    points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      points_topic, rclcpp::SensorDataQoS());
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic,
      rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
        latest_camera_info_ = msg;
      });
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic,
      rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        handle_depth(msg);
      });
  }

private:
  void handle_depth(const sensor_msgs::msg::Image::ConstSharedPtr & depth)
  {
    if (!latest_camera_info_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Waiting for CameraInfo before publishing depth point cloud");
      return;
    }

    try {
      auto cloud = depth_to_point_cloud(*depth, *latest_camera_info_, pixel_stride_);
      if (!output_frame_.empty()) {
        cloud.header.frame_id = output_frame_;
      }
      points_pub_->publish(cloud);
    } catch (const std::invalid_argument & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Failed to convert depth image to point cloud: %s", error.what());
    }
  }

  std::string output_frame_;
  uint32_t pixel_stride_ = 1u;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr latest_camera_info_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub_;
};

}  // namespace
}  // namespace ghost_mgg_sim

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ghost_mgg_sim::DepthToPointCloudNode>());
  rclcpp::shutdown();
  return 0;
}
