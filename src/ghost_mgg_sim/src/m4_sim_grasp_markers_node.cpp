#include <chrono>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace
{
using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;

struct SimTarget
{
  std::string target_id;
  std::string shape_type;
  std::string grasp_type;
  double center_x_m = 0.0;
  double center_y_m = 0.0;
  double center_z_m = 0.0;
  double yaw_rad = 0.0;
  double size_x_m = 0.0;
  double size_y_m = 0.0;
  double size_z_m = 0.0;
  double radius_m = 0.0;
  double height_m = 0.0;
  double required_gripper_width_m = 0.0;
  double pregrasp_clearance_m = 0.09;
  bool valid = false;
  std::string failure_reason;
};

std::string read_file(const std::string & path)
{
  std::ifstream stream(path);
  if (!stream.is_open()) {
    throw std::runtime_error("failed to open M4_sim_grasp target config: " + path);
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

std::string extract_string(const std::string & object, const std::string & key)
{
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(object, match, pattern)) {
    return "";
  }
  return match[1].str();
}

double extract_number(const std::string & object, const std::string & key)
{
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
  std::smatch match;
  if (!std::regex_search(object, match, pattern)) {
    return 0.0;
  }
  return std::stod(match[1].str());
}

bool extract_bool(const std::string & object, const std::string & key)
{
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(object, match, pattern)) {
    return false;
  }
  return match[1].str() == "true";
}

std::vector<std::string> extract_row_objects(const std::string & json)
{
  const auto rows_pos = json.find("\"rows\"");
  if (rows_pos == std::string::npos) {
    return {};
  }
  const auto array_start = json.find('[', rows_pos);
  const auto array_end = json.rfind(']');
  if (array_start == std::string::npos || array_end == std::string::npos || array_end <= array_start) {
    return {};
  }

  std::vector<std::string> objects;
  int depth = 0;
  std::size_t object_start = std::string::npos;
  for (std::size_t index = array_start + 1; index < array_end; ++index) {
    const char ch = json[index];
    if (ch == '{') {
      if (depth == 0) {
        object_start = index;
      }
      ++depth;
    } else if (ch == '}') {
      --depth;
      if (depth == 0 && object_start != std::string::npos) {
        objects.push_back(json.substr(object_start, index - object_start + 1));
        object_start = std::string::npos;
      }
    }
  }
  return objects;
}

std::vector<SimTarget> load_targets(const std::string & path)
{
  const auto json = read_file(path);
  std::vector<SimTarget> targets;
  for (const auto & object : extract_row_objects(json)) {
    SimTarget target;
    target.target_id = extract_string(object, "target_id");
    target.shape_type = extract_string(object, "shape_type");
    target.grasp_type = extract_string(object, "grasp_type");
    target.center_x_m = extract_number(object, "center_x_m");
    target.center_y_m = extract_number(object, "center_y_m");
    target.center_z_m = extract_number(object, "center_z_m");
    target.yaw_rad = extract_number(object, "yaw_rad");
    target.size_x_m = extract_number(object, "size_x_m");
    target.size_y_m = extract_number(object, "size_y_m");
    target.size_z_m = extract_number(object, "size_z_m");
    target.radius_m = extract_number(object, "radius_m");
    target.height_m = extract_number(object, "height_m");
    target.required_gripper_width_m = extract_number(object, "required_gripper_width_m");
    target.pregrasp_clearance_m = extract_number(object, "pregrasp_clearance_m");
    target.valid = extract_bool(object, "valid");
    target.failure_reason = extract_string(object, "failure_reason");
    if (!target.target_id.empty() && !target.shape_type.empty()) {
      targets.push_back(target);
    }
  }
  return targets;
}

geometry_msgs::msg::Point point(double x, double y, double z)
{
  geometry_msgs::msg::Point output;
  output.x = x;
  output.y = y;
  output.z = z;
  return output;
}

void set_valid_color(Marker & marker, double alpha = 0.88)
{
  marker.color.a = alpha;
  marker.color.r = 0.03;
  marker.color.g = 0.96;
  marker.color.b = 0.26;
}

void set_proxy_color(Marker & marker)
{
  marker.color.a = 0.32;
  marker.color.r = 0.14;
  marker.color.g = 0.80;
  marker.color.b = 1.00;
}

void set_invalid_color(Marker & marker)
{
  marker.color.a = 0.80;
  marker.color.r = 0.95;
  marker.color.g = 0.12;
  marker.color.b = 0.08;
}

Marker make_delete_all_marker(const std::string & frame_id, const builtin_interfaces::msg::Time & stamp)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.action = Marker::DELETEALL;
  return marker;
}

double target_height(const SimTarget & target)
{
  if (target.shape_type == "cylinder") {
    return target.height_m;
  }
  return target.size_z_m;
}

double grasp_z(const SimTarget & target)
{
  return target.center_z_m + 0.5 * target_height(target) + 0.035;
}

Marker make_target_proxy(
  const SimTarget & target,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp,
  int id)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = "m4_sim_bound_target_proxy";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = target.shape_type == "cylinder" ? Marker::CYLINDER : Marker::CUBE;
  marker.pose.position = point(target.center_x_m, target.center_y_m, target.center_z_m);
  marker.pose.orientation.z = std::sin(target.yaw_rad * 0.5);
  marker.pose.orientation.w = std::cos(target.yaw_rad * 0.5);
  if (target.shape_type == "cylinder") {
    marker.scale.x = 2.0 * target.radius_m;
    marker.scale.y = 2.0 * target.radius_m;
    marker.scale.z = target.height_m;
  } else {
    marker.scale.x = target.size_x_m;
    marker.scale.y = target.size_y_m;
    marker.scale.z = target.size_z_m;
  }
  set_proxy_color(marker);
  return marker;
}

Marker make_top_grasp_arrow(
  const SimTarget & target,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp,
  int id)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = target.valid ? "m4_sim_bound_top_grasp" : "m4_sim_bound_reject";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = target.valid ? Marker::ARROW : Marker::CUBE;
  const double z_grasp = grasp_z(target);
  if (target.valid) {
    marker.points.push_back(point(target.center_x_m, target.center_y_m, z_grasp + target.pregrasp_clearance_m));
    marker.points.push_back(point(target.center_x_m, target.center_y_m, z_grasp));
    marker.scale.x = 0.008;
    marker.scale.y = 0.018;
    marker.scale.z = 0.030;
    set_valid_color(marker);
  } else {
    marker.pose.position = point(target.center_x_m, target.center_y_m, z_grasp);
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.030;
    marker.scale.y = 0.030;
    marker.scale.z = 0.010;
    set_invalid_color(marker);
  }
  return marker;
}

Marker make_width_bar(
  const SimTarget & target,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp,
  int id)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = "m4_sim_bound_width_bar";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = Marker::CUBE;
  marker.pose.position = point(target.center_x_m, target.center_y_m, grasp_z(target) + 0.006);
  marker.pose.orientation.z = std::sin(target.yaw_rad * 0.5);
  marker.pose.orientation.w = std::cos(target.yaw_rad * 0.5);
  marker.scale.x = std::max(0.006, target.required_gripper_width_m);
  marker.scale.y = 0.006;
  marker.scale.z = 0.004;
  if (target.valid) {
    set_valid_color(marker, 0.52);
  } else {
    set_invalid_color(marker);
    marker.color.a = 0.50;
  }
  return marker;
}

Marker make_summary_text(
  const std::vector<SimTarget> & targets,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp,
  int id)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = "m4_sim_bound_panel_text";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = Marker::TEXT_VIEW_FACING;
  marker.pose.position = point(-0.12, -0.18, 1.03);
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.018;
  marker.color.a = 0.96;
  marker.color.r = 0.94;
  marker.color.g = 0.98;
  marker.color.b = 1.0;

  int valid_count = 0;
  std::ostringstream label;
  label << "M4_sim_grasp\n";
  for (const auto & target : targets) {
    if (target.valid) {
      ++valid_count;
    }
    label << target.target_id << "=" << (target.valid ? target.grasp_type : "reject") << "\n";
  }
  label << "valid=" << valid_count << "/" << targets.size();
  marker.text = label.str();
  return marker;
}

class M4SimGraspMarkersNode : public rclcpp::Node
{
public:
  explicit M4SimGraspMarkersNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("m4_sim_grasp_markers_node", options)
  {
    targets_path_ = declare_parameter<std::string>(
      "targets_path", "config/m4_sim_grasp_targets.json");
    marker_topic_ = declare_parameter<std::string>(
      "marker_topic", "/ghost_mgg/m4_sim_grasp_markers");
    frame_id_ = declare_parameter<std::string>("frame_id", "world");

    marker_pub_ = create_publisher<MarkerArray>(
      marker_topic_,
      rclcpp::QoS(1).transient_local().reliable());
    timer_ = create_wall_timer(std::chrono::milliseconds(500), [this]() {
      publish_markers();
    });
  }

private:
  void publish_markers()
  {
    try {
      const auto targets = load_targets(targets_path_);
      MarkerArray markers;
      const auto stamp = now();
      markers.markers.push_back(make_delete_all_marker(frame_id_, stamp));

      int id = 1;
      for (const auto & target : targets) {
        markers.markers.push_back(make_target_proxy(target, frame_id_, stamp, id++));
        markers.markers.push_back(make_top_grasp_arrow(target, frame_id_, stamp, id++));
        markers.markers.push_back(make_width_bar(target, frame_id_, stamp, id++));
      }
      if (!targets.empty()) {
        markers.markers.push_back(make_summary_text(targets, frame_id_, stamp, id++));
      }
      marker_pub_->publish(markers);

      if (!reported_success_) {
        RCLCPP_INFO(
          get_logger(),
          "Published %zu M4_sim_grasp target-bound marker groups from %s",
          targets.size(),
          targets_path_.c_str());
        reported_success_ = true;
      }
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Cannot publish M4_sim_grasp markers yet: %s", error.what());
    }
  }

  std::string targets_path_;
  std::string marker_topic_;
  std::string frame_id_;
  bool reported_success_ = false;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<M4SimGraspMarkersNode>());
  rclcpp::shutdown();
  return 0;
}
