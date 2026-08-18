#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

#include "ghost_mgg_sim/depth_failure_injector.hpp"

namespace ghost_mgg_sim
{
namespace
{

class DepthFailureInjectorNode : public rclcpp::Node
{
public:
  DepthFailureInjectorNode()
  : Node("depth_failure_injector_node")
  {
    const auto input_depth_topic = declare_parameter<std::string>(
      "input_depth_topic", "/ghost_mgg/d435/depth/image_rect_raw");
    const auto output_depth_topic = declare_parameter<std::string>(
      "output_depth_topic", "/ghost_mgg/d435/depth/m3_corrupted");
    const auto hole_mask_topic = declare_parameter<std::string>(
      "hole_mask_topic", "/ghost_mgg/d435/evidence/hole_mask");
    const auto table_leakage_mask_topic = declare_parameter<std::string>(
      "table_leakage_mask_topic", "/ghost_mgg/d435/evidence/table_leakage_mask");
    const auto edge_mask_topic = declare_parameter<std::string>(
      "edge_mask_topic", "/ghost_mgg/d435/evidence/edge_mask");
    const auto flying_point_mask_topic = declare_parameter<std::string>(
      "flying_point_mask_topic", "/ghost_mgg/d435/evidence/flying_point_mask");
    const auto biased_depth_mask_topic = declare_parameter<std::string>(
      "biased_depth_mask_topic", "/ghost_mgg/d435/evidence/biased_depth_mask");
    const auto summary_topic = declare_parameter<std::string>(
      "summary_topic", "/ghost_mgg/d435/evidence/summary");
    use_mask_topic_ = declare_parameter<bool>("use_mask_topic", false);
    const auto target_mask_topic = declare_parameter<std::string>(
      "target_mask_topic", "/ghost_mgg/d435/target_mask");

    config_.mode = depth_failure_mode_from_string(
      declare_parameter<std::string>("failure_mode", "mixed"));
    config_.roi_center_u_ratio = declare_parameter<double>("roi_center_u_ratio", 0.50);
    config_.roi_center_v_ratio = declare_parameter<double>("roi_center_v_ratio", 0.58);
    config_.roi_width_ratio = declare_parameter<double>("roi_width_ratio", 0.22);
    config_.roi_height_ratio = declare_parameter<double>("roi_height_ratio", 0.22);
    config_.table_leak_depth_m = static_cast<float>(
      declare_parameter<double>("table_leak_depth_m", 1.20));
    config_.flying_point_offset_m = static_cast<float>(
      declare_parameter<double>("flying_point_offset_m", 0.12));
    config_.biased_depth_offset_m = static_cast<float>(
      declare_parameter<double>("biased_depth_offset_m", -0.04));
    config_.edge_band_pixels = static_cast<uint32_t>(
      std::max<int>(1, declare_parameter<int>("edge_band_pixels", 2)));
    config_.flying_point_stride = static_cast<uint32_t>(
      std::max<int>(1, declare_parameter<int>("flying_point_stride", 5)));
    config_.pattern_seed = static_cast<uint32_t>(
      std::max<int>(0, declare_parameter<int>("pattern_seed", 0)));

    auto qos = rclcpp::SensorDataQoS();
    corrupted_depth_pub_ = create_publisher<sensor_msgs::msg::Image>(output_depth_topic, qos);
    hole_mask_pub_ = create_publisher<sensor_msgs::msg::Image>(hole_mask_topic, qos);
    table_leakage_mask_pub_ =
      create_publisher<sensor_msgs::msg::Image>(table_leakage_mask_topic, qos);
    edge_mask_pub_ = create_publisher<sensor_msgs::msg::Image>(edge_mask_topic, qos);
    flying_point_mask_pub_ =
      create_publisher<sensor_msgs::msg::Image>(flying_point_mask_topic, qos);
    biased_depth_mask_pub_ =
      create_publisher<sensor_msgs::msg::Image>(biased_depth_mask_topic, qos);
    summary_pub_ = create_publisher<std_msgs::msg::String>(
      summary_topic,
      rclcpp::QoS(1).transient_local().reliable());

    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      input_depth_topic,
      qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        handle_depth(msg);
      });

    if (use_mask_topic_) {
      target_mask_sub_ = create_subscription<sensor_msgs::msg::Image>(
        target_mask_topic,
        qos,
        [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          latest_target_mask_ = msg;
        });
    }
  }

private:
  void handle_depth(const sensor_msgs::msg::Image::ConstSharedPtr & depth)
  {
    try {
      if (use_mask_topic_ && latest_target_mask_ == nullptr) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Waiting for target mask before publishing depth failure evidence");
        return;
      }

      const auto result = use_mask_topic_ ?
        inject_depth_failure(*depth, *latest_target_mask_, config_) :
        inject_depth_failure(*depth, config_);
      corrupted_depth_pub_->publish(result.corrupted_depth);
      hole_mask_pub_->publish(result.hole_mask);
      table_leakage_mask_pub_->publish(result.table_leakage_mask);
      edge_mask_pub_->publish(result.edge_mask);
      flying_point_mask_pub_->publish(result.flying_point_mask);
      biased_depth_mask_pub_->publish(result.biased_depth_mask);

      std_msgs::msg::String summary;
      summary.data = evidence_summary_to_json(result.summary);
      summary_pub_->publish(summary);
    } catch (const std::invalid_argument & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Failed to inject depth failure: %s", error.what());
    }
  }

  DepthFailureInjectionConfig config_;
  bool use_mask_topic_ = false;
  sensor_msgs::msg::Image::ConstSharedPtr latest_target_mask_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr target_mask_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr corrupted_depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr hole_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr table_leakage_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr edge_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr flying_point_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr biased_depth_mask_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr summary_pub_;
};

}  // namespace
}  // namespace ghost_mgg_sim

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ghost_mgg_sim::DepthFailureInjectorNode>());
  rclcpp::shutdown();
  return 0;
}
