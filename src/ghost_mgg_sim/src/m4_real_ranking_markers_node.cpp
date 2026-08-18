#include <algorithm>
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

struct RankingRow
{
  std::string scene_id;
  std::string ranker;
  int rank = 0;
  std::string hypothesis_id;
  std::string shape_type;
  std::string target_label;
  double center_u = 0.0;
  double center_v = 0.0;
  double size_u_px = 0.0;
  double size_v_px = 0.0;
  double visual_score = 0.0;
  double failure_score = 0.0;
  double total_score = 0.0;
  double failure_inside_hole = 0.0;
  double failure_inside_table_leakage = 0.0;
  double failure_outside_hole_penalty = 0.0;
  double failure_outside_table_leakage_penalty = 0.0;
};

std::string read_file(const std::string & path)
{
  std::ifstream stream(path);
  if (!stream.is_open()) {
    throw std::runtime_error("failed to open M5 real D435 ranking report: " + path);
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

std::vector<RankingRow> load_ranking_rows(const std::string & path)
{
  const auto json = read_file(path);
  std::vector<RankingRow> rows;
  for (const auto & object : extract_row_objects(json)) {
    RankingRow row;
    row.scene_id = extract_string(object, "scene_id");
    row.ranker = extract_string(object, "ranker");
    row.rank = static_cast<int>(std::lround(extract_number(object, "rank")));
    row.hypothesis_id = extract_string(object, "hypothesis_id");
    row.shape_type = extract_string(object, "shape_type");
    row.target_label = extract_string(object, "target_label");
    row.center_u = extract_number(object, "center_u");
    row.center_v = extract_number(object, "center_v");
    row.size_u_px = extract_number(object, "size_u_px");
    row.size_v_px = extract_number(object, "size_v_px");
    row.visual_score = extract_number(object, "visual_score");
    row.failure_score = extract_number(object, "failure_score");
    row.total_score = extract_number(object, "total_score");
    row.failure_inside_hole = extract_number(object, "failure_inside_hole");
    row.failure_inside_table_leakage = extract_number(object, "failure_inside_table_leakage");
    row.failure_outside_hole_penalty = extract_number(object, "failure_outside_hole_penalty");
    row.failure_outside_table_leakage_penalty = extract_number(
      object, "failure_outside_table_leakage_penalty");
    if (!row.scene_id.empty() && !row.ranker.empty() && row.rank > 0) {
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

geometry_msgs::msg::Point projected_center(
  const RankingRow & row,
  double fx,
  double fy,
  double cx,
  double cy,
  double depth_m,
  double ranker_offset_m)
{
  geometry_msgs::msg::Point point;
  point.x = ((row.center_u - cx) / fx) * depth_m + ranker_offset_m;
  point.y = ((row.center_v - cy) / fy) * depth_m;
  point.z = depth_m;
  return point;
}

void set_row_color(Marker & marker, const RankingRow & row)
{
  marker.color.a = 0.74;
  if (row.ranker == "failure_aware") {
    marker.color.r = 0.0;
    marker.color.g = 0.92;
    marker.color.b = 0.28;
    return;
  }
  marker.color.r = 0.18;
  marker.color.g = 0.45;
  marker.color.b = 1.0;
}

Marker make_proxy_marker(
  const RankingRow & row,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp,
  int id,
  double fx,
  double fy,
  double cx,
  double cy,
  double depth_m,
  double marker_thickness_m)
{
  const bool failure_aware = row.ranker == "failure_aware";
  const double ranker_offset = failure_aware ? 0.018 : -0.018;

  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = failure_aware ? "m4_real_failure_aware_proxy" : "m4_real_silhouette_only_proxy";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = row.shape_type == "cylinder" ? Marker::CYLINDER : Marker::CUBE;
  marker.pose.position = projected_center(row, fx, fy, cx, cy, depth_m, ranker_offset);
  marker.pose.orientation.w = 1.0;
  marker.scale.x = std::max(0.003, row.size_u_px * depth_m / fx);
  marker.scale.y = std::max(0.003, row.size_v_px * depth_m / fy);
  marker.scale.z = std::max(0.002, marker_thickness_m);
  set_row_color(marker, row);
  return marker;
}

std::string short_ranker_name(const std::string & ranker)
{
  if (ranker == "silhouette_only") {
    return "sil";
  }
  if (ranker == "failure_aware") {
    return "fail";
  }
  return ranker;
}

Marker make_summary_text_marker(
  const std::vector<RankingRow> & rows,
  const std::string & scene_id,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp,
  int id,
  double depth_m)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = "m4_real_ranking_panel_text";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = Marker::TEXT_VIEW_FACING;
  marker.pose.position.x = 0.0;
  marker.pose.position.y = -0.25;
  marker.pose.position.z = depth_m - 0.03;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.012;
  marker.color.a = 0.96;
  marker.color.r = 0.94;
  marker.color.g = 0.97;
  marker.color.b = 1.0;

  std::ostringstream label;
  label.setf(std::ios::fixed);
  label.precision(2);
  label << "M4_real_ranking\n" << "scene=" << scene_id << "\n";
  for (const auto & row : rows) {
    label << short_ranker_name(row.ranker)
          << ":" << row.hypothesis_id << "\n"
          << "T=" << row.total_score
          << "/V=" << row.visual_score
          << "/F=" << row.failure_score << "\n";
    label << "ih=" << row.failure_inside_hole
          << "/leak=" << row.failure_inside_table_leakage
          << "/oh=" << row.failure_outside_hole_penalty
          << "/ol=" << row.failure_outside_table_leakage_penalty << "\n";
  }
  marker.text = label.str();
  return marker;
}

class M4RealRankingMarkersNode : public rclcpp::Node
{
public:
  explicit M4RealRankingMarkersNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("m4_real_ranking_markers_node", options)
  {
    ranking_report_path_ = declare_parameter<std::string>(
      "ranking_report_path", "reports/m5_real_d435_ranking/m5_real_ranking.json");
    scene_id_ = declare_parameter<std::string>("scene_id", "daylight_transparent_jelly_cup_001");
    marker_topic_ = declare_parameter<std::string>(
      "marker_topic", "/ghost_mgg/m4_real_ranking_markers");
    frame_id_ = declare_parameter<std::string>("frame_id", "d435_depth_optical_frame");
    top_k_ = declare_parameter<int>("top_k", 1);
    fx_ = declare_parameter<double>("fx", 554.0);
    fy_ = declare_parameter<double>("fy", 554.0);
    cx_ = declare_parameter<double>("cx", 320.0);
    cy_ = declare_parameter<double>("cy", 240.0);
    marker_depth_m_ = declare_parameter<double>("marker_depth_m", 1.12);
    marker_thickness_m_ = declare_parameter<double>("marker_thickness_m", 0.016);

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
      const auto rows = load_ranking_rows(ranking_report_path_);
      MarkerArray markers;
      const auto stamp = now();
      markers.markers.push_back(make_delete_all_marker(frame_id_, stamp));

      int id = 1;
      std::vector<RankingRow> selected_rows;
      for (const auto & row : rows) {
        if (row.scene_id != scene_id_ || row.rank > top_k_) {
          continue;
        }
        if (row.ranker != "silhouette_only" && row.ranker != "failure_aware") {
          continue;
        }
        markers.markers.push_back(
          make_proxy_marker(row, frame_id_, stamp, id++, fx_, fy_, cx_, cy_, marker_depth_m_, marker_thickness_m_));
        selected_rows.push_back(row);
      }

      if (!selected_rows.empty()) {
        markers.markers.push_back(
          make_summary_text_marker(selected_rows, scene_id_, frame_id_, stamp, id++, marker_depth_m_));
      }
      marker_pub_->publish(markers);
      if (!reported_success_) {
        RCLCPP_INFO(
          get_logger(),
          "Published %zu M4_real_ranking markers for scene %s from %s",
          selected_rows.size(),
          scene_id_.c_str(),
          ranking_report_path_.c_str());
        reported_success_ = true;
      }
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Cannot publish M4_real_ranking markers yet: %s", error.what());
    }
  }

  std::string ranking_report_path_;
  std::string scene_id_;
  std::string marker_topic_;
  std::string frame_id_;
  int top_k_ = 1;
  double fx_ = 554.0;
  double fy_ = 554.0;
  double cx_ = 320.0;
  double cy_ = 240.0;
  double marker_depth_m_ = 1.12;
  double marker_thickness_m_ = 0.016;
  bool reported_success_ = false;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<M4RealRankingMarkersNode>());
  rclcpp::shutdown();
  return 0;
}
