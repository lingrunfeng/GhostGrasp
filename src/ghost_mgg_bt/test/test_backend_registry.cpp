#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ghost_mgg_bt/backend_registry.hpp"

TEST(BackendRegistry, ParsesBackendAndExecutorEndpoints)
{
  const auto path =
    std::filesystem::temp_directory_path() / "ghost_mgg_bt_registry_test.yaml";
  std::ofstream out(path);
  out << "backends:\n"
      << "  dummy:\n"
      << "    recover_action: /geometry_backends/dummy/recover\n"
      << "  mask_extrusion:\n"
      << "    recover_action: /geometry_backends/mask_extrusion/recover\n"
      << "executors:\n"
      << "  dummy:\n"
      << "    execute_action: /grasp_executors/dummy/execute\n";
  out.close();

  const auto registry = ghost_mgg_bt::BackendRegistry::from_file(path.string());

  EXPECT_EQ(
    registry.backend("dummy").recover_action,
    "/geometry_backends/dummy/recover");
  EXPECT_EQ(
    registry.backend("mask_extrusion").recover_action,
    "/geometry_backends/mask_extrusion/recover");
  EXPECT_EQ(
    registry.executor("dummy").execute_action,
    "/grasp_executors/dummy/execute");

  std::filesystem::remove(path);
}

TEST(BackendRegistry, ThrowsForMissingEndpoint)
{
  const auto path =
    std::filesystem::temp_directory_path() / "ghost_mgg_bt_registry_missing_test.yaml";
  std::ofstream out(path);
  out << "backends: {}\nexecutors: {}\n";
  out.close();

  const auto registry = ghost_mgg_bt::BackendRegistry::from_file(path.string());

  EXPECT_THROW(registry.backend("ghost_mgg"), std::out_of_range);
  EXPECT_THROW(registry.executor("hardware"), std::out_of_range);

  std::filesystem::remove(path);
}
