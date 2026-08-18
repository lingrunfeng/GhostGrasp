#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "ghost_mgg_sim/depth_to_mono8.hpp"

namespace ghost_mgg_sim
{
namespace
{

class DepthToMono8Node : public rclcpp::Node
{
public:
  DepthToMono8Node()
  : Node("depth_to_mono8_node")
  {
    const auto depth_topic = declare_parameter<std::string>(
      "depth_topic", "/ghost_mgg/d435/depth/image_rect_raw");
    const auto preview_topic = declare_parameter<std::string>(
      "preview_topic", "/ghost_mgg/d435/depth/image_viz");
    min_depth_m_ = declare_parameter<double>("min_depth_m", 0.2);
    max_depth_m_ = declare_parameter<double>("max_depth_m", 1.4);

    preview_pub_ = create_publisher<sensor_msgs::msg::Image>(
      preview_topic, rclcpp::SensorDataQoS());
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
    try {
      preview_pub_->publish(depth_to_mono8(
        *depth,
        static_cast<float>(min_depth_m_),
        static_cast<float>(max_depth_m_)));
    } catch (const std::invalid_argument & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Failed to convert depth image to mono8 preview: %s", error.what());
    }
  }

  double min_depth_m_;
  double max_depth_m_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr preview_pub_;
};

}  // namespace
}  // namespace ghost_mgg_sim

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ghost_mgg_sim::DepthToMono8Node>());
  rclcpp::shutdown();
  return 0;
}
