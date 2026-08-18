#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
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

struct GraspabilityRow
{
  std::string scene_id;
  std::string ranker;
  std::string hypothesis_id;
  std::string shape_type;
  std::string grasp_id;
  std::string grasp_type;
  double grasp_x_m = 0.0;
  double grasp_y_m = 0.0;
  double grasp_z_m = 0.0;
  double pregrasp_z_m = 0.0;
  double required_gripper_width_m = 0.0;
  double gripper_width_margin_m = 0.0;
  std::string grasp_width_axis;
  double grasp_width_base_m = 0.0;
  double source_center_u = 0.0;
  double source_center_v = 0.0;
  double table_depth_m = 0.0;
  double score = 0.0;
  bool valid = false;
  std::string failure_reason;
};

std::string read_file(const std::string & path)
{
  std::ifstream stream(path);
  if (!stream.is_open()) {
    throw std::runtime_error("failed to open M4_graspability report: " + path);
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

std::vector<GraspabilityRow> load_graspability_rows(const std::string & path)
{
  const auto json = read_file(path);
  std::vector<GraspabilityRow> rows;
  for (const auto & object : extract_row_objects(json)) {
    GraspabilityRow row;
    row.scene_id = extract_string(object, "scene_id");
    row.ranker = extract_string(object, "ranker");
    row.hypothesis_id = extract_string(object, "hypothesis_id");
    row.shape_type = extract_string(object, "shape_type");
    row.grasp_id = extract_string(object, "grasp_id");
    row.grasp_type = extract_string(object, "grasp_type");
    row.grasp_x_m = extract_number(object, "grasp_x_m");
    row.grasp_y_m = extract_number(object, "grasp_y_m");
    row.grasp_z_m = extract_number(object, "grasp_z_m");
    row.pregrasp_z_m = extract_number(object, "pregrasp_z_m");
    row.required_gripper_width_m = extract_number(object, "required_gripper_width_m");
    row.gripper_width_margin_m = extract_number(object, "gripper_width_margin_m");
    row.grasp_width_axis = extract_string(object, "grasp_width_axis");
    row.grasp_width_base_m = extract_number(object, "grasp_width_base_m");
    row.source_center_u = extract_number(object, "source_center_u");
    row.source_center_v = extract_number(object, "source_center_v");
    row.table_depth_m = extract_number(object, "table_depth_m");
    row.score = extract_number(object, "score");
    row.valid = extract_bool(object, "valid");
    row.failure_reason = extract_string(object, "failure_reason");
    if (!row.scene_id.empty() && !row.hypothesis_id.empty()) {
      rows.push_back(row);
    }
  }
  return rows;
}

Marker make_delete_all_marker(const std::string & frame_id, const builtin_interfaces::msg::Time & stamp)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.action = Marker::DELETEALL;
  return marker;
}

geometry_msgs::msg::Point point(double x, double y, double z)
{
  geometry_msgs::msg::Point output;
  output.x = x;
  output.y = y;
  output.z = z;
  return output;
}

void set_valid_color(Marker & marker)
{
  marker.color.a = 0.88;
  marker.color.r = 0.05;
  marker.color.g = 0.95;
  marker.color.b = 0.24;
}

void set_invalid_color(Marker & marker)
{
  marker.color.a = 0.78;
  marker.color.r = 0.95;
  marker.color.g = 0.14;
  marker.color.b = 0.10;
}

Marker make_top_grasp_arrow(
  const GraspabilityRow & row,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp,
  int id)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = "m4_grasp_valid_top_grasp";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = Marker::ARROW;
  marker.points.push_back(point(row.grasp_x_m, row.grasp_y_m, row.pregrasp_z_m));
  marker.points.push_back(point(row.grasp_x_m, row.grasp_y_m, row.grasp_z_m));
  marker.scale.x = 0.008;
  marker.scale.y = 0.018;
  marker.scale.z = 0.030;
  set_valid_color(marker);
  return marker;
}

Marker make_invalid_marker(
  const GraspabilityRow & row,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp,
  int id)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = "m4_grasp_invalid_reject";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = Marker::CUBE;
  marker.pose.position = point(row.grasp_x_m, row.grasp_y_m, row.grasp_z_m);
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.024;
  marker.scale.y = 0.024;
  marker.scale.z = 0.010;
  set_invalid_color(marker);
  return marker;
}

Marker make_width_bar(
  const GraspabilityRow & row,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp,
  int id)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = "m4_grasp_width_bar";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = Marker::CUBE;
  marker.pose.position = point(row.grasp_x_m, row.grasp_y_m, row.grasp_z_m + 0.006);
  marker.pose.orientation.w = 1.0;
  const double width = std::max(0.006, row.required_gripper_width_m);
  const double thickness = 0.006;
  marker.scale.x = row.grasp_width_axis == "x" ? width : thickness;
  marker.scale.y = row.grasp_width_axis == "y" ? width : thickness;
  marker.scale.z = 0.004;
  if (row.valid) {
    set_valid_color(marker);
    marker.color.a = 0.48;
  } else {
    set_invalid_color(marker);
    marker.color.a = 0.50;
  }
  return marker;
}

Marker make_summary_text_marker(
  const std::vector<GraspabilityRow> & rows,
  const std::string & scene_id,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp,
  int id)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = "m4_graspability_panel_text";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = Marker::TEXT_VIEW_FACING;
  marker.pose.position.x = 0.0;
  marker.pose.position.y = -0.25;
  marker.pose.position.z = 1.06;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.018;
  marker.color.a = 0.96;
  marker.color.r = 0.94;
  marker.color.g = 0.97;
  marker.color.b = 1.0;

  int valid_count = 0;
  std::map<std::string, int> reasons;
  for (const auto & row : rows) {
    if (row.valid) {
      ++valid_count;
    } else {
      ++reasons[row.failure_reason.empty() ? "unknown" : row.failure_reason];
    }
  }

  std::ostringstream label;
  label << "M4_graspability\n" << "scene=" << scene_id << "\n";
  label << "valid=" << valid_count << "/" << rows.size() << "\n";
  for (const auto & [reason, count] : reasons) {
    label << reason << "=" << count << "\n";
  }
  marker.text = label.str();
  return marker;
}

class M4GraspabilityMarkersNode : public rclcpp::Node
{
public:
  explicit M4GraspabilityMarkersNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("m4_graspability_markers_node", options)
  {
    graspability_report_path_ = declare_parameter<std::string>(
      "graspability_report_path", "reports/m4_graspability_dryrun/graspability.json");
    scene_id_ = declare_parameter<std::string>("scene_id", "daylight_transparent_jelly_cup_001");
    marker_topic_ = declare_parameter<std::string>(
      "marker_topic", "/ghost_mgg/m4_graspability_markers");
    frame_id_ = declare_parameter<std::string>("frame_id", "world");
    ranker_filter_ = declare_parameter<std::string>("ranker_filter", "");

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
      const auto rows = load_graspability_rows(graspability_report_path_);
      MarkerArray markers;
      const auto stamp = now();
      markers.markers.push_back(make_delete_all_marker(frame_id_, stamp));

      std::vector<GraspabilityRow> selected_rows;
      int id = 1;
      for (const auto & row : rows) {
        if (scene_id_ != "all" && row.scene_id != scene_id_) {
          continue;
        }
        if (!ranker_filter_.empty() && row.ranker != ranker_filter_) {
          continue;
        }
        selected_rows.push_back(row);
        if (row.valid) {
          markers.markers.push_back(make_top_grasp_arrow(row, frame_id_, stamp, id++));
        } else {
          markers.markers.push_back(make_invalid_marker(row, frame_id_, stamp, id++));
        }
        markers.markers.push_back(make_width_bar(row, frame_id_, stamp, id++));
      }

      if (!selected_rows.empty()) {
        markers.markers.push_back(
          make_summary_text_marker(selected_rows, scene_id_, frame_id_, stamp, id++));
      }
      marker_pub_->publish(markers);
      if (!reported_success_) {
        RCLCPP_INFO(
          get_logger(),
          "Published %zu M4_graspability markers for scene %s from %s",
          selected_rows.size(),
          scene_id_.c_str(),
          graspability_report_path_.c_str());
        reported_success_ = true;
      }
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Cannot publish M4_graspability markers yet: %s", error.what());
    }
  }

  std::string graspability_report_path_;
  std::string scene_id_;
  std::string marker_topic_;
  std::string frame_id_;
  std::string ranker_filter_;
  bool reported_success_ = false;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<M4GraspabilityMarkersNode>());
  rclcpp::shutdown();
  return 0;
}
