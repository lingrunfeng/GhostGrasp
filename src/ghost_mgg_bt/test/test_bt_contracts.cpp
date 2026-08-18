#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

TEST(BTContracts, M0TreeUsesStrictBehaviorTreeCppAndRankedFallback)
{
  const auto tree_path =
    std::filesystem::path(GHOST_MGG_BT_SOURCE_DIR) / "trees/m0_dummy_recovery.xml";
  ASSERT_TRUE(std::filesystem::exists(tree_path));

  std::ifstream in(tree_path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const auto xml = buffer.str();

  EXPECT_NE(xml.find("BTCPP_format=\"4\""), std::string::npos);
  EXPECT_NE(xml.find("<RecoverGeometry"), std::string::npos);
  EXPECT_NE(xml.find("observation_ref=\"{observation_ref}\""), std::string::npos);
  EXPECT_NE(xml.find("<RetryUntilSuccessful"), std::string::npos);
  EXPECT_NE(xml.find("<SelectNextHypothesis"), std::string::npos);
  EXPECT_NE(xml.find("<ExecuteGrasp"), std::string::npos);
  EXPECT_NE(xml.find("<ReportTrial"), std::string::npos);
  EXPECT_EQ(xml.find(std::string{"sensor_msgs/"} + "Image"), std::string::npos);
  EXPECT_EQ(xml.find(std::string{"Camera"} + "Info"), std::string::npos);
}

TEST(BTContracts, RegistryContainsDummyBackendAndExecutor)
{
  const auto registry_path =
    std::filesystem::path(GHOST_MGG_BT_SOURCE_DIR) / "config/backend_registry.yaml";
  ASSERT_TRUE(std::filesystem::exists(registry_path));

  std::ifstream in(registry_path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const auto yaml = buffer.str();

  EXPECT_NE(yaml.find("/geometry_backends/dummy/recover"), std::string::npos);
  EXPECT_NE(yaml.find("/grasp_executors/dummy/execute"), std::string::npos);
}

TEST(BTContracts, RegistryContainsMaskExtrusionBackend)
{
  const auto registry_path =
    std::filesystem::path(GHOST_MGG_BT_SOURCE_DIR) / "config/backend_registry.yaml";
  ASSERT_TRUE(std::filesystem::exists(registry_path));

  std::ifstream in(registry_path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const auto yaml = buffer.str();

  EXPECT_NE(yaml.find("mask_extrusion:"), std::string::npos);
  EXPECT_NE(yaml.find("/geometry_backends/mask_extrusion/recover"), std::string::npos);
}

TEST(BTContracts, M2TreeUsesSameActionContractForSimClosedLoop)
{
  const auto tree_path =
    std::filesystem::path(GHOST_MGG_BT_SOURCE_DIR) / "trees/m2_sim_closed_loop.xml";
  ASSERT_TRUE(std::filesystem::exists(tree_path));

  std::ifstream in(tree_path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const auto xml = buffer.str();

  EXPECT_NE(xml.find("BTCPP_format=\"4\""), std::string::npos);
  EXPECT_NE(xml.find("<RecoverGeometry"), std::string::npos);
  EXPECT_NE(xml.find("<RetryUntilSuccessful"), std::string::npos);
  EXPECT_NE(xml.find("num_attempts=\"4\""), std::string::npos);
  EXPECT_NE(xml.find("<ForceSuccess>"), std::string::npos);
  EXPECT_NE(xml.find("<ExecuteGrasp"), std::string::npos);
  EXPECT_NE(xml.find("tree_name=\"m2_sim_closed_loop\""), std::string::npos);
  EXPECT_NE(xml.find("observation_ref=\"{observation_ref}\""), std::string::npos);
  EXPECT_EQ(xml.find(std::string{"sensor_msgs/"} + "Image"), std::string::npos);
  EXPECT_EQ(xml.find(std::string{"Camera"} + "Info"), std::string::npos);
}

TEST(BTContracts, RegistryContainsM2MycobotSimExecutor)
{
  const auto registry_path =
    std::filesystem::path(GHOST_MGG_BT_SOURCE_DIR) / "config/backend_registry.yaml";
  ASSERT_TRUE(std::filesystem::exists(registry_path));

  std::ifstream in(registry_path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const auto yaml = buffer.str();

  EXPECT_NE(yaml.find("mycobot_sim:"), std::string::npos);
  EXPECT_NE(yaml.find("/grasp_executors/mycobot_sim/execute"), std::string::npos);
}

TEST(BTContracts, ActionNodesDoNotUseFixedShortServerWait)
{
  const auto recover_path =
    std::filesystem::path(GHOST_MGG_BT_SOURCE_DIR) / "src/recover_geometry_action_node.cpp";
  const auto execute_path =
    std::filesystem::path(GHOST_MGG_BT_SOURCE_DIR) / "src/execute_grasp_action_node.cpp";
  ASSERT_TRUE(std::filesystem::exists(recover_path));
  ASSERT_TRUE(std::filesystem::exists(execute_path));

  auto read = [](const std::filesystem::path & path) {
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
  };

  const auto recover_source = read(recover_path);
  const auto execute_source = read(execute_path);

  EXPECT_EQ(recover_source.find("milliseconds(250)"), std::string::npos);
  EXPECT_EQ(execute_source.find("milliseconds(250)"), std::string::npos);
  EXPECT_NE(recover_source.find("action_server_wait_duration"), std::string::npos);
  EXPECT_NE(execute_source.find("action_server_wait_duration"), std::string::npos);
}
