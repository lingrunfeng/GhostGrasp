#include <chrono>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace
{

class RobotDescriptionPublisherNode : public rclcpp::Node
{
public:
  RobotDescriptionPublisherNode()
  : Node("robot_description_publisher_node")
  {
    robot_description_ = declare_parameter<std::string>("robot_description", "");
    publisher_ = create_publisher<std_msgs::msg::String>(
      "/robot_description",
      rclcpp::QoS(1).reliable().transient_local());

    timer_ = create_wall_timer(
      std::chrono::milliseconds(1000),
      [this]() {
        publish_description();
        ++publish_count_;
        if (publish_count_ >= kMaxPublishCount) {
          timer_->cancel();
        }
      });

    publish_description();
  }

private:
  void publish_description()
  {
    if (robot_description_.empty()) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "robot_description parameter is empty; RobotModel display will not load");
      return;
    }

    std_msgs::msg::String msg;
    msg.data = robot_description_;
    publisher_->publish(msg);
  }

  static constexpr int kMaxPublishCount = 12;
  std::string robot_description_;
  int publish_count_{0};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RobotDescriptionPublisherNode>());
  rclcpp::shutdown();
  return 0;
}
