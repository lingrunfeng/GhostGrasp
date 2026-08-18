#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace
{

std::string sanitize_id(std::string value)
{
  for (auto & ch : value) {
    const bool ok =
      (ch >= 'a' && ch <= 'z') ||
      (ch >= 'A' && ch <= 'Z') ||
      (ch >= '0' && ch <= '9') ||
      ch == '_' ||
      ch == '-';
    if (!ok) {
      ch = '_';
    }
  }
  return value;
}

class EvidenceSummaryLoggerNode : public rclcpp::Node
{
public:
  explicit EvidenceSummaryLoggerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("evidence_summary_logger_node", options)
  {
    const auto summary_topic = declare_parameter<std::string>(
      "summary_topic", "/ghost_mgg/d435/evidence/summary");
    const auto log_dir = declare_parameter<std::string>(
      "log_dir", "log/ghost_mgg_trials/m3_failure");
    scenario_id_ = sanitize_id(declare_parameter<std::string>("scenario_id", "custom"));

    std::filesystem::create_directories(log_dir);
    log_path_ = std::filesystem::path(log_dir) / (scenario_id_ + "_evidence.jsonl");
    log_file_.open(log_path_, std::ios::app);
    if (!log_file_.is_open()) {
      throw std::runtime_error("failed to open evidence summary log: " + log_path_.string());
    }

    summary_sub_ = create_subscription<std_msgs::msg::String>(
      summary_topic,
      rclcpp::QoS(1).transient_local().reliable(),
      [this](std_msgs::msg::String::ConstSharedPtr msg) {
        write_summary(*msg);
      });
  }

private:
  void write_summary(const std_msgs::msg::String & summary)
  {
    const auto now = get_clock()->now();
    log_file_ << "{";
    log_file_ << "\"scenario_id\":\"" << scenario_id_ << "\",";
    log_file_ << "\"stamp_sec\":" << now.seconds() << ",";
    log_file_ << "\"summary\":" << summary.data;
    log_file_ << "}\n";
    log_file_.flush();
  }

  std::string scenario_id_;
  std::filesystem::path log_path_;
  std::ofstream log_file_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr summary_sub_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EvidenceSummaryLoggerNode>());
  rclcpp::shutdown();
  return 0;
}
