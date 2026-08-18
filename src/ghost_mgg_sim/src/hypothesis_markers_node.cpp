#include <cstdint>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>

#include <geometry_msgs/msg/point.hpp>
#include <ghost_mgg_interfaces/msg/geometry_hypothesis.hpp>
#include <ghost_mgg_interfaces/msg/geometry_hypothesis_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace
{
using GeometryHypothesis = ghost_mgg_interfaces::msg::GeometryHypothesis;
using GeometryHypothesisArray = ghost_mgg_interfaces::msg::GeometryHypothesisArray;
using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;
using String = std_msgs::msg::String;

std::string frame_for(const GeometryHypothesisArray & array, const GeometryHypothesis & hypothesis)
{
  if (!hypothesis.pose_base.header.frame_id.empty()) {
    return hypothesis.pose_base.header.frame_id;
  }
  if (!array.header.frame_id.empty()) {
    return array.header.frame_id;
  }
  return "world";
}

void set_rank_color(Marker & marker, std::size_t rank)
{
  marker.color.a = 0.55;
  if (rank == 0) {
    marker.color.r = 0.0;
    marker.color.g = 0.85;
    marker.color.b = 0.20;
    return;
  }
  if (rank == 1) {
    marker.color.r = 0.10;
    marker.color.g = 0.35;
    marker.color.b = 1.0;
    return;
  }
  marker.color.r = 1.0;
  marker.color.g = 0.68;
  marker.color.b = 0.05;
}

void set_executed_color(Marker & marker, double alpha = 0.22)
{
  marker.color.a = alpha;
  marker.color.r = 0.55;
  marker.color.g = 0.55;
  marker.color.b = 0.55;
}

Marker make_delete_all_marker(const builtin_interfaces::msg::Time & stamp)
{
  Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = stamp;
  marker.action = Marker::DELETEALL;
  return marker;
}

Marker make_proxy_marker(
  const GeometryHypothesisArray & array,
  const GeometryHypothesis & hypothesis,
  std::size_t rank,
  int id,
  bool executed)
{
  Marker marker;
  marker.header.frame_id = frame_for(array, hypothesis);
  marker.header.stamp = array.header.stamp;
  marker.ns = "m2_hypothesis_proxy";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = hypothesis.shape_type == GeometryHypothesis::SHAPE_CYLINDER ?
    Marker::CYLINDER : Marker::CUBE;
  marker.pose = hypothesis.pose_base.pose;
  marker.scale.x = hypothesis.dimensions_m.x;
  marker.scale.y = hypothesis.dimensions_m.y;
  marker.scale.z = hypothesis.dimensions_m.z;
  if (executed) {
    set_executed_color(marker, 0.16);
  } else {
    set_rank_color(marker, rank);
  }
  return marker;
}

Marker make_grasp_marker(
  const GeometryHypothesisArray & array,
  const GeometryHypothesis & hypothesis,
  std::size_t rank,
  int id,
  bool executed)
{
  Marker marker;
  marker.header.frame_id = frame_for(array, hypothesis);
  marker.header.stamp = array.header.stamp;
  marker.ns = "m2_hypothesis_grasp";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = Marker::SPHERE;
  if (!hypothesis.grasp_candidates.empty()) {
    marker.pose = hypothesis.grasp_candidates.front().grasp_pose.pose;
  } else {
    marker.pose = hypothesis.pose_base.pose;
  }
  marker.scale.x = 0.014;
  marker.scale.y = 0.014;
  marker.scale.z = 0.014;
  if (executed) {
    set_executed_color(marker, 0.25);
  } else {
    marker.color.a = 0.90;
    marker.color.r = rank == 0 ? 0.0 : 1.0;
    marker.color.g = rank == 0 ? 1.0 : 0.75;
    marker.color.b = 0.05;
  }
  return marker;
}

Marker make_grasp_approach_marker(
  const GeometryHypothesisArray & array,
  const GeometryHypothesis & hypothesis,
  int id,
  bool executed)
{
  Marker marker;
  marker.header.frame_id = frame_for(array, hypothesis);
  marker.header.stamp = array.header.stamp;
  marker.ns = "m2_hypothesis_approach";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = Marker::LINE_LIST;
  marker.scale.x = 0.003;
  if (executed) {
    set_executed_color(marker, 0.32);
  } else {
    marker.color.a = 0.95;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
  }

  geometry_msgs::msg::Point start;
  geometry_msgs::msg::Point end;
  if (!hypothesis.grasp_candidates.empty()) {
    start.x = hypothesis.grasp_candidates.front().pregrasp_pose.pose.position.x;
    start.y = hypothesis.grasp_candidates.front().pregrasp_pose.pose.position.y;
    start.z = hypothesis.grasp_candidates.front().pregrasp_pose.pose.position.z;
    end.x = hypothesis.grasp_candidates.front().grasp_pose.pose.position.x;
    end.y = hypothesis.grasp_candidates.front().grasp_pose.pose.position.y;
    end.z = hypothesis.grasp_candidates.front().grasp_pose.pose.position.z;
  } else {
    start.x = hypothesis.pose_base.pose.position.x;
    start.y = hypothesis.pose_base.pose.position.y;
    start.z = hypothesis.pose_base.pose.position.z + 0.05;
    end.x = hypothesis.pose_base.pose.position.x;
    end.y = hypothesis.pose_base.pose.position.y;
    end.z = hypothesis.pose_base.pose.position.z;
  }
  marker.points.push_back(start);
  marker.points.push_back(end);
  return marker;
}

std::string compact_shape_name(uint8_t shape_type)
{
  if (shape_type == GeometryHypothesis::SHAPE_CYLINDER) {
    return "cyl";
  }
  if (shape_type == GeometryHypothesis::SHAPE_BOX) {
    return "box";
  }
  if (shape_type == GeometryHypothesis::SHAPE_CUP_LIKE) {
    return "cup";
  }
  return "unk";
}

std::string stable_hypothesis_label(const GeometryHypothesis & hypothesis, std::size_t rank)
{
  const std::string & hypothesis_id = hypothesis.hypothesis_id;
  if (hypothesis_id.rfind("track_", 0) == 0) {
    const auto shape_separator = hypothesis_id.find('_', std::string("track_").size());
    if (shape_separator != std::string::npos) {
      return hypothesis_id.substr(0, shape_separator) + ":" +
             compact_shape_name(hypothesis.shape_type);
    }
    return hypothesis_id + ":" + compact_shape_name(hypothesis.shape_type);
  }

  std::ostringstream fallback;
  fallback << "rank_" << (rank + 1) << ":" << compact_shape_name(hypothesis.shape_type);
  return fallback.str();
}

Marker make_rank_label_marker(
  const GeometryHypothesisArray & array,
  const GeometryHypothesis & hypothesis,
  std::size_t rank,
  int id,
  bool executed)
{
  Marker marker;
  marker.header.frame_id = frame_for(array, hypothesis);
  marker.header.stamp = array.header.stamp;
  marker.ns = "m2_hypothesis_rank_label";
  marker.id = id;
  marker.action = Marker::ADD;
  marker.type = Marker::TEXT_VIEW_FACING;
  marker.pose = hypothesis.pose_base.pose;
  marker.pose.position.z += (hypothesis.dimensions_m.z * 0.5) + 0.030 + static_cast<double>(rank) * 0.016;
  marker.scale.z = 0.018;
  if (executed) {
    set_executed_color(marker, 0.42);
  } else {
    set_rank_color(marker, rank);
    marker.color.a = 0.92;
  }

  std::ostringstream label;
  label << stable_hypothesis_label(hypothesis, rank);
  if (!hypothesis.hypothesis_id.empty() && hypothesis.hypothesis_id.rfind("track_", 0) != 0) {
    label << "\n" << hypothesis.hypothesis_id;
  }
  label << std::fixed << std::setprecision(2)
        << "\nT=" << hypothesis.score.total
        << "/F=" << hypothesis.score.failure;
  marker.text = label.str();
  return marker;
}

std::string find_json_string_value(const std::string & text, const std::string & key)
{
  const auto quoted_key = "\"" + key + "\"";
  auto key_pos = text.find(quoted_key);
  if (key_pos == std::string::npos) {
    return "";
  }
  auto colon_pos = text.find(':', key_pos + quoted_key.size());
  if (colon_pos == std::string::npos) {
    return "";
  }
  auto value_start = text.find('"', colon_pos + 1);
  if (value_start == std::string::npos) {
    return "";
  }
  auto value_end = text.find('"', value_start + 1);
  if (value_end == std::string::npos) {
    return "";
  }
  return text.substr(value_start + 1, value_end - value_start - 1);
}

bool event_is_success(const std::string & text)
{
  if (text.find("\"executed_success\":true") != std::string::npos) {
    return true;
  }
  return find_json_string_value(text, "status_name") == "SUCCEEDED";
}

bool reset_executed_hypotheses_requested(const std::string & text)
{
  return text.find("\"reset_executed_hypotheses\":true") != std::string::npos ||
         text.find("\"reset_executed_hypotheses\": true") != std::string::npos;
}

std::string executed_hypothesis_id_from_event(const std::string & text)
{
  if (!event_is_success(text)) {
    return "";
  }
  const auto hypothesis_id = find_json_string_value(text, "hypothesis_id");
  if (!hypothesis_id.empty()) {
    return hypothesis_id;
  }
  return text;
}

class HypothesisMarkersNode : public rclcpp::Node
{
public:
  explicit HypothesisMarkersNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("hypothesis_markers_node", options)
  {
    const auto hypotheses_topic = declare_parameter<std::string>(
      "hypotheses_topic", "/ghost_mgg/hypotheses/mask_extrusion");
    const auto marker_topic = declare_parameter<std::string>(
      "marker_topic", "/ghost_mgg/hypothesis_markers");
    const auto executed_hypotheses_topic = declare_parameter<std::string>(
      "executed_hypotheses_topic", "/ghost_mgg/m4_executed_hypotheses");
    hide_executed_hypotheses_ = declare_parameter<bool>("hide_executed_hypotheses", false);
    hide_all_after_success_ = declare_parameter<bool>("hide_all_after_success", false);
    max_visible_hypotheses_ = declare_parameter<int>("max_visible_hypotheses", 0);
    marker_pub_ = create_publisher<MarkerArray>(
      marker_topic,
      rclcpp::QoS(1).transient_local().reliable());
    hypotheses_sub_ = create_subscription<GeometryHypothesisArray>(
      hypotheses_topic,
      rclcpp::QoS(1).transient_local().reliable(),
      [this](const GeometryHypothesisArray::SharedPtr msg) {
        last_hypotheses_ = *msg;
        has_last_hypotheses_ = true;
        publish_markers(*msg);
      });
    executed_sub_ = create_subscription<String>(
      executed_hypotheses_topic,
      rclcpp::QoS(1).transient_local().reliable(),
      [this](const String::SharedPtr msg) {
        if (reset_executed_hypotheses_requested(msg->data)) {
          executed_hypothesis_ids_.clear();
          successful_execution_seen_ = false;
          if (has_last_hypotheses_) {
            publish_markers(last_hypotheses_);
          }
          return;
        }
        const auto hypothesis_id = executed_hypothesis_id_from_event(msg->data);
        if (!hypothesis_id.empty()) {
          executed_hypothesis_ids_.insert(hypothesis_id);
          if (hide_all_after_success_) {
            successful_execution_seen_ = true;
          }
          if (has_last_hypotheses_) {
            publish_markers(last_hypotheses_);
          }
        }
      });
  }

private:
  void publish_markers(const GeometryHypothesisArray & hypotheses)
  {
    MarkerArray markers;
    markers.markers.push_back(make_delete_all_marker(hypotheses.header.stamp));
    if (hide_all_after_success_ && successful_execution_seen_) {
      marker_pub_->publish(markers);
      return;
    }
    int id = 1;
    std::size_t visible_count = 0;
    for (std::size_t rank = 0; rank < hypotheses.hypotheses.size(); ++rank) {
      const auto & hypothesis = hypotheses.hypotheses[rank];
      // existence proxies (transparent objects) sit at the end of the message
      // with low confidence: they never count against the visibility cap and
      // always render, otherwise a cluttered table silently hides them
      const bool existence_proxy =
        hypothesis.provenance.find("hole_existence_only") != std::string::npos;
      if (!existence_proxy && max_visible_hypotheses_ > 0 &&
        visible_count >= static_cast<std::size_t>(max_visible_hypotheses_))
      {
        continue;
      }
      const auto executed = executed_hypothesis_ids_.count(hypothesis.hypothesis_id) > 0;
      if (executed && hide_executed_hypotheses_) {
        continue;
      }
      markers.markers.push_back(make_proxy_marker(hypotheses, hypothesis, rank, id++, executed));
      markers.markers.push_back(make_grasp_marker(hypotheses, hypothesis, rank, id++, executed));
      markers.markers.push_back(make_grasp_approach_marker(hypotheses, hypothesis, id++, executed));
      markers.markers.push_back(make_rank_label_marker(hypotheses, hypothesis, rank, id++, executed));
      if (!existence_proxy) {
        ++visible_count;
      }
    }
    marker_pub_->publish(markers);
  }

  rclcpp::Subscription<GeometryHypothesisArray>::SharedPtr hypotheses_sub_;
  rclcpp::Subscription<String>::SharedPtr executed_sub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_pub_;
  std::set<std::string> executed_hypothesis_ids_;
  GeometryHypothesisArray last_hypotheses_;
  bool has_last_hypotheses_{false};
  bool hide_executed_hypotheses_{false};
  bool hide_all_after_success_{false};
  bool successful_execution_seen_{false};
  int max_visible_hypotheses_{0};
};
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HypothesisMarkersNode>());
  rclcpp::shutdown();
  return 0;
}
