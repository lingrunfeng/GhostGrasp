#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "ghost_mgg_backends/dummy_recovery_server.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ghost_mgg_backends::DummyRecoveryServer>());
  rclcpp::shutdown();
  return 0;
}
