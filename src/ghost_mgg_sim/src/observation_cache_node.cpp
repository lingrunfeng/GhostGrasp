#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#include <ghost_mgg_interfaces/msg/observation_ref.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace
{
using ObservationRef = ghost_mgg_interfaces::msg::ObservationRef;
using CameraInfo = sensor_msgs::msg::CameraInfo;
using Image = sensor_msgs::msg::Image;

std::string make_observation_id(
  const std::string & prefix,
  const builtin_interfaces::msg::Time & stamp)
{
  std::ostringstream id;
  id << prefix << '_' << stamp.sec << '_' << stamp.nanosec;
  return id.str();
}

builtin_interfaces::msg::Time latest_stamp(
  const builtin_interfaces::msg::Time & a,
  const builtin_interfaces::msg::Time & b)
{
  if (b.sec > a.sec || (b.sec == a.sec && b.nanosec > a.nanosec)) {
    return b;
  }
  return a;
}

class ObservationCacheNode : public rclcpp::Node
{
public:
  explicit ObservationCacheNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("observation_cache_node", options)
  {
    cache_namespace_ = declare_parameter<std::string>("cache_namespace", "m2_d435");
    observation_prefix_ = declare_parameter<std::string>("observation_prefix", "m2_obs");
    max_age_sec_ = declare_parameter<double>("max_age_sec", 1.0);
    rgb_frame_id_ = declare_parameter<std::string>("rgb_frame_id", "d435_color_optical_frame");
    depth_frame_id_ = declare_parameter<std::string>("depth_frame_id", "d435_depth_optical_frame");
    mask_frame_id_ = declare_parameter<std::string>("mask_frame_id", "d435_color_optical_frame");

    const auto rgb_topic =
      declare_parameter<std::string>("rgb_topic", "/ghost_mgg/d435/color/image_raw");
    const auto depth_topic =
      declare_parameter<std::string>("depth_topic", "/ghost_mgg/d435/depth/image_rect_raw");
    const auto color_info_topic =
      declare_parameter<std::string>("color_camera_info_topic", "/ghost_mgg/d435/color/camera_info");
    const auto depth_info_topic =
      declare_parameter<std::string>("depth_camera_info_topic", "/ghost_mgg/d435/depth/camera_info");
    const auto target_mask_topic =
      declare_parameter<std::string>("target_mask_topic", "");
    const auto infra1_topic =
      declare_parameter<std::string>("infra1_topic", "");
    const auto infra2_topic =
      declare_parameter<std::string>("infra2_topic", "");
    const auto output_topic =
      declare_parameter<std::string>("observation_ref_topic", "/ghost_mgg/observations/latest");
    const auto publish_period_ms = declare_parameter<int>("publish_period_ms", 100);
    require_mask_ = !target_mask_topic.empty();
    require_ir_ = !infra1_topic.empty() && !infra2_topic.empty();

    observation_pub_ = create_publisher<ObservationRef>(
      output_topic,
      rclcpp::QoS(1).transient_local().reliable());

    auto qos = rclcpp::SensorDataQoS();
    rgb_sub_ = create_subscription<Image>(
      rgb_topic, qos, [this](const Image::SharedPtr msg) {
        rgb_stamp_ = msg->header.stamp;
        has_rgb_ = true;
      });
    depth_sub_ = create_subscription<Image>(
      depth_topic, qos, [this](const Image::SharedPtr msg) {
        depth_stamp_ = msg->header.stamp;
        has_depth_ = true;
      });
    color_info_sub_ = create_subscription<CameraInfo>(
      color_info_topic, qos, [this](const CameraInfo::SharedPtr msg) {
        color_info_stamp_ = msg->header.stamp;
        has_color_info_ = true;
      });
    depth_info_sub_ = create_subscription<CameraInfo>(
      depth_info_topic, qos, [this](const CameraInfo::SharedPtr msg) {
        depth_info_stamp_ = msg->header.stamp;
        has_depth_info_ = true;
      });
    if (require_mask_) {
      target_mask_sub_ = create_subscription<Image>(
        target_mask_topic, qos, [this](const Image::SharedPtr msg) {
          mask_stamp_ = msg->header.stamp;
          has_mask_ = true;
        });
    }
    if (require_ir_) {
      infra1_sub_ = create_subscription<Image>(
        infra1_topic, qos, [this](const Image::SharedPtr msg) {
          infra1_stamp_ = msg->header.stamp;
          has_infra1_ = true;
        });
      infra2_sub_ = create_subscription<Image>(
        infra2_topic, qos, [this](const Image::SharedPtr msg) {
          infra2_stamp_ = msg->header.stamp;
          has_infra2_ = true;
        });
    }

    const auto period = std::chrono::milliseconds(std::max<long>(20, publish_period_ms));
    timer_ = create_wall_timer(period, [this]() { publish_latest_ref(); });
  }

private:
  void publish_latest_ref()
  {
    if (!has_rgb_ || !has_depth_ || !has_color_info_ || !has_depth_info_) {
      return;
    }
    if (require_mask_ && !has_mask_) {
      return;
    }
    if (require_ir_ && (!has_infra1_ || !has_infra2_)) {
      return;
    }

    auto stamp = rgb_stamp_;
    stamp = latest_stamp(stamp, depth_stamp_);
    stamp = latest_stamp(stamp, color_info_stamp_);
    stamp = latest_stamp(stamp, depth_info_stamp_);
    if (has_mask_) {
      stamp = latest_stamp(stamp, mask_stamp_);
    }
    if (has_infra1_) {
      stamp = latest_stamp(stamp, infra1_stamp_);
    }
    if (has_infra2_) {
      stamp = latest_stamp(stamp, infra2_stamp_);
    }

    ObservationRef ref;
    ref.observation_id = make_observation_id(observation_prefix_, stamp);
    ref.cache_namespace = cache_namespace_;
    ref.rgb_frame_id = rgb_frame_id_;
    ref.depth_frame_id = depth_frame_id_;
    ref.mask_frame_id = mask_frame_id_;
    ref.stamp = stamp;
    ref.max_age_sec = max_age_sec_;
    ref.has_rgb = true;
    ref.has_depth = true;
    ref.has_ir = has_infra1_ && has_infra2_;
    ref.has_mask = has_mask_;
    ref.has_camera_info = true;
    observation_pub_->publish(ref);
  }

  std::string cache_namespace_;
  std::string observation_prefix_;
  double max_age_sec_ = 1.0;
  std::string rgb_frame_id_;
  std::string depth_frame_id_;
  std::string mask_frame_id_;

  bool has_rgb_ = false;
  bool has_depth_ = false;
  bool has_color_info_ = false;
  bool has_depth_info_ = false;
  bool has_mask_ = false;
  bool has_infra1_ = false;
  bool has_infra2_ = false;
  bool require_mask_ = false;
  bool require_ir_ = false;
  builtin_interfaces::msg::Time rgb_stamp_;
  builtin_interfaces::msg::Time depth_stamp_;
  builtin_interfaces::msg::Time color_info_stamp_;
  builtin_interfaces::msg::Time depth_info_stamp_;
  builtin_interfaces::msg::Time mask_stamp_;
  builtin_interfaces::msg::Time infra1_stamp_;
  builtin_interfaces::msg::Time infra2_stamp_;

  rclcpp::Subscription<Image>::SharedPtr rgb_sub_;
  rclcpp::Subscription<Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<Image>::SharedPtr target_mask_sub_;
  rclcpp::Subscription<Image>::SharedPtr infra1_sub_;
  rclcpp::Subscription<Image>::SharedPtr infra2_sub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr color_info_sub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr depth_info_sub_;
  rclcpp::Publisher<ObservationRef>::SharedPtr observation_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObservationCacheNode>());
  rclcpp::shutdown();
  return 0;
}
