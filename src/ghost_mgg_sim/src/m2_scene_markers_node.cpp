#include "ghost_mgg_sim/m2_scene_markers.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace ghost_mgg_sim
{
namespace
{

using Marker = visualization_msgs::msg::Marker;

constexpr char kWorldFrame[] = "world";
constexpr char kMarkerTopic[] = "/ghost_mgg/m2_scene_markers";

struct Color
{
  double r;
  double g;
  double b;
  double a;
};

struct Quaternion
{
  double x;
  double y;
  double z;
  double w;
};

Quaternion quaternion_from_rpy(double roll, double pitch, double yaw)
{
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);

  return Quaternion{
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
    cr * cp * cy + sr * sp * sy,
  };
}

Marker base_marker(
  int id,
  const std::string & name,
  int type,
  const Color & color)
{
  Marker marker;
  marker.header.frame_id = kWorldFrame;
  marker.ns = name;
  marker.id = id;
  marker.type = type;
  marker.action = Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.color.r = color.r;
  marker.color.g = color.g;
  marker.color.b = color.b;
  marker.color.a = color.a;
  return marker;
}

Marker make_solid_marker(
  int id,
  const std::string & name,
  int type,
  double x,
  double y,
  double z,
  double roll,
  double pitch,
  double yaw,
  double sx,
  double sy,
  double sz,
  const Color & color)
{
  Marker marker = base_marker(id, name, type, color);
  marker.pose.position.x = x;
  marker.pose.position.y = y;
  marker.pose.position.z = z;
  const Quaternion q = quaternion_from_rpy(roll, pitch, yaw);
  marker.pose.orientation.x = q.x;
  marker.pose.orientation.y = q.y;
  marker.pose.orientation.z = q.z;
  marker.pose.orientation.w = q.w;
  marker.scale.x = sx;
  marker.scale.y = sy;
  marker.scale.z = sz;
  return marker;
}

Marker make_camera_frustum()
{
  Marker marker = base_marker(
    5,
    "camera_frustum",
    Marker::LINE_LIST,
    Color{0.95, 0.78, 0.12, 1.0});
  marker.scale.x = 0.004;

  geometry_msgs::msg::Point origin;
  origin.x = 0.286775;
  origin.y = -0.079936;
  origin.z = 1.105590;

  std::vector<geometry_msgs::msg::Point> corners(4);
  corners[0].x = -0.090;
  corners[0].y = -0.140;
  corners[0].z = 0.7525;
  corners[1].x = -0.090;
  corners[1].y = 0.140;
  corners[1].z = 0.7525;
  corners[2].x = 0.190;
  corners[2].y = 0.140;
  corners[2].z = 0.7525;
  corners[3].x = 0.190;
  corners[3].y = -0.140;
  corners[3].z = 0.7525;

  for (const auto & corner : corners) {
    marker.points.push_back(origin);
    marker.points.push_back(corner);
  }

  for (std::size_t i = 0; i < corners.size(); ++i) {
    marker.points.push_back(corners[i]);
    marker.points.push_back(corners[(i + 1) % corners.size()]);
  }

  return marker;
}

geometry_msgs::msg::TransformStamped make_transform(
  const rclcpp::Time & stamp,
  const std::string & parent_frame,
  const std::string & child_frame,
  double x,
  double y,
  double z,
  double roll,
  double pitch,
  double yaw)
{
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = stamp;
  transform.header.frame_id = parent_frame;
  transform.child_frame_id = child_frame;
  transform.transform.translation.x = x;
  transform.transform.translation.y = y;
  transform.transform.translation.z = z;
  const Quaternion q = quaternion_from_rpy(roll, pitch, yaw);
  transform.transform.rotation.x = q.x;
  transform.transform.rotation.y = q.y;
  transform.transform.rotation.z = q.z;
  transform.transform.rotation.w = q.w;
  return transform;
}

}  // namespace

visualization_msgs::msg::MarkerArray make_m2_scene_markers()
{
  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(make_solid_marker(
    0,
    "table",
    Marker::CUBE,
    0.0,
    0.0,
    0.735,
    0.0,
    0.0,
    0.0,
    0.80,
    0.55,
    0.03,
    Color{0.82, 0.58, 0.34, 0.55}));
  markers.markers.push_back(make_solid_marker(
    1,
    "red_cube",
    Marker::CUBE,
    0.070172,
    0.012156,
    0.7525,
    0.0,
    0.0,
    0.65,
    0.025,
    0.025,
    0.025,
    Color{0.95, 0.04, 0.03, 1.0}));
  markers.markers.push_back(make_solid_marker(
    2,
    "blue_cylinder",
    Marker::CYLINDER,
    0.139726,
    -0.034920,
    0.7525,
    0.0,
    0.0,
    0.0,
    0.025,
    0.025,
    0.025,
    Color{0.05, 0.22, 0.95, 1.0}));
  markers.markers.push_back(make_solid_marker(
    3,
    "green_cylinder",
    Marker::CYLINDER,
    -0.035,
    -0.047344,
    0.7525,
    0.0,
    0.0,
    0.0,
    0.025,
    0.025,
    0.025,
    Color{0.04, 0.78, 0.18, 1.0}));
  markers.markers.push_back(make_solid_marker(
    4,
    "glass_block",
    Marker::CUBE,
    0.002,
    0.100,
    0.7525,
    0.0,
    0.0,
    -0.35,
    0.025,
    0.025,
    0.025,
    Color{0.70, 0.90, 1.00, 0.45}));
  markers.markers.push_back(make_camera_frustum());
  return markers;
}

class M2SceneMarkersNode : public rclcpp::Node
{
public:
  M2SceneMarkersNode()
  : Node("m2_scene_markers_node"),
    markers_(make_m2_scene_markers()),
    static_broadcaster_(std::make_shared<tf2_ros::StaticTransformBroadcaster>(this))
  {
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      kMarkerTopic,
      rclcpp::QoS(1).transient_local().reliable());
    publish_static_transforms();
    timer_ = create_wall_timer(
      std::chrono::milliseconds(500),
      [this]() {
        const auto stamp = now();
        for (auto & marker : markers_.markers) {
          marker.header.stamp = stamp;
        }
        marker_pub_->publish(markers_);
      });
  }

private:
  void publish_static_transforms()
  {
    const rclcpp::Time stamp = now();
    std::vector<geometry_msgs::msg::TransformStamped> transforms;
    transforms.push_back(make_transform(
      stamp,
      kWorldFrame,
      "base_link",
      -0.171207,
      0.228790,
      0.765000,
      0.0,
      0.0,
      -2.180640));
    transforms.push_back(make_transform(
      stamp,
      kWorldFrame,
      "table_frame",
      0.0,
      0.0,
      0.74,
      0.0,
      0.0,
      0.0));
    transforms.push_back(make_transform(
      stamp,
      kWorldFrame,
      "d435_link",
      0.286775,
      -0.079936,
      1.105590,
      0.0,
      0.950000,
      2.754700));
    transforms.push_back(make_transform(
      stamp,
      "d435_link",
      "d435_color_frame",
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0));
    transforms.push_back(make_transform(
      stamp,
      "d435_link",
      "d435_depth_frame",
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0));
    transforms.push_back(make_transform(
      stamp,
      "d435_color_frame",
      "d435_color_optical_frame",
      0.0,
      0.0,
      0.0,
      -1.5708,
      0.0,
      -1.5708));
    transforms.push_back(make_transform(
      stamp,
      "d435_depth_frame",
      "d435_depth_optical_frame",
      0.0,
      0.0,
      0.0,
      -1.5708,
      0.0,
      -1.5708));
    static_broadcaster_->sendTransform(transforms);
  }

  visualization_msgs::msg::MarkerArray markers_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
};

}  // namespace ghost_mgg_sim

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ghost_mgg_sim::M2SceneMarkersNode>());
  rclcpp::shutdown();
  return 0;
}
