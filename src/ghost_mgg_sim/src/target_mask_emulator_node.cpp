#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace
{

uint32_t clamped_pixel_count(double ratio, uint32_t extent)
{
  const auto clamped_ratio = std::clamp(ratio, 0.0, 1.0);
  return std::max<uint32_t>(1u, static_cast<uint32_t>(std::lround(clamped_ratio * extent)));
}

class TargetMaskEmulatorNode : public rclcpp::Node
{
public:
  explicit TargetMaskEmulatorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("target_mask_emulator_node", options)
  {
    const auto input_image_topic = declare_parameter<std::string>(
      "input_image_topic", "/ghost_mgg/d435/depth/image_rect_raw");
    const auto mask_topic = declare_parameter<std::string>(
      "mask_topic", "/ghost_mgg/d435/target_mask");
    roi_center_u_ratio_ = declare_parameter<double>("roi_center_u_ratio", 0.50);
    roi_center_v_ratio_ = declare_parameter<double>("roi_center_v_ratio", 0.58);
    roi_width_ratio_ = declare_parameter<double>("roi_width_ratio", 0.22);
    roi_height_ratio_ = declare_parameter<double>("roi_height_ratio", 0.22);

    auto qos = rclcpp::SensorDataQoS();
    mask_pub_ = create_publisher<sensor_msgs::msg::Image>(mask_topic, qos);
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      input_image_topic,
      qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        publish_mask_for_image(*msg);
      });
  }

private:
  void publish_mask_for_image(const sensor_msgs::msg::Image & image)
  {
    if (image.width == 0u || image.height == 0u) {
      return;
    }

    sensor_msgs::msg::Image mask;
    mask.header = image.header;
    mask.height = image.height;
    mask.width = image.width;
    mask.encoding = "mono8";
    mask.is_bigendian = false;
    mask.step = image.width;
    mask.data.assign(static_cast<size_t>(mask.step) * mask.height, 0u);

    const uint32_t width_px = clamped_pixel_count(roi_width_ratio_, image.width);
    const uint32_t height_px = clamped_pixel_count(roi_height_ratio_, image.height);
    const int center_u = static_cast<int>(
      std::lround(std::clamp(roi_center_u_ratio_, 0.0, 1.0) * image.width));
    const int center_v = static_cast<int>(
      std::lround(std::clamp(roi_center_v_ratio_, 0.0, 1.0) * image.height));

    int u_min = center_u - static_cast<int>(width_px / 2u);
    int v_min = center_v - static_cast<int>(height_px / 2u);
    u_min = std::clamp(
      u_min,
      0,
      std::max<int>(0, static_cast<int>(image.width) - static_cast<int>(width_px)));
    v_min = std::clamp(
      v_min,
      0,
      std::max<int>(0, static_cast<int>(image.height) - static_cast<int>(height_px)));

    const auto u_max = std::min(image.width, static_cast<uint32_t>(u_min) + width_px);
    const auto v_max = std::min(image.height, static_cast<uint32_t>(v_min) + height_px);
    for (uint32_t v = static_cast<uint32_t>(v_min); v < v_max; ++v) {
      for (uint32_t u = static_cast<uint32_t>(u_min); u < u_max; ++u) {
        mask.data[static_cast<size_t>(v) * mask.step + u] = 255u;
      }
    }

    mask_pub_->publish(mask);
  }

  double roi_center_u_ratio_ = 0.50;
  double roi_center_v_ratio_ = 0.58;
  double roi_width_ratio_ = 0.22;
  double roi_height_ratio_ = 0.22;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TargetMaskEmulatorNode>());
  rclcpp::shutdown();
  return 0;
}
