#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "ghost_mgg_bt/trial_logger.hpp"
#include "ghost_mgg_interfaces/msg/geometry_hypothesis.hpp"

TEST(TrialLogger, WritesOneJsonLineWithM0Fields)
{
  const auto dir = std::filesystem::temp_directory_path() / "ghost_mgg_bt_logger_test";
  std::filesystem::remove_all(dir);

  ghost_mgg_bt::TrialLogger logger(dir.string());
  ghost_mgg_bt::TrialLogRecord record;
  record.trial_id = "trial_m0_1";
  record.observation_id = "obs_m0_1";
  record.tree_name = "m0_dummy_recovery";
  record.backend_name = "dummy";
  record.recover_status = 1;
  record.execute_status = 1;
  record.final_status = "succeeded";
  record.selected_hypothesis_id = "h2";
  record.attempted_hypothesis_ids = {"h1", "h2"};

  ghost_mgg_interfaces::msg::GeometryHypothesis h1;
  h1.hypothesis_id = "h1";
  ghost_mgg_interfaces::msg::GeometryHypothesis h2;
  h2.hypothesis_id = "h2";
  record.hypotheses = {h1, h2};

  const auto path = logger.write(record);

  std::ifstream in(path);
  std::string line;
  std::getline(in, line);

  EXPECT_NE(line.find("\"trial_id\":\"trial_m0_1\""), std::string::npos);
  EXPECT_NE(line.find("\"observation_id\":\"obs_m0_1\""), std::string::npos);
  EXPECT_NE(line.find("\"backend_name\":\"dummy\""), std::string::npos);
  EXPECT_NE(line.find("\"selected_hypothesis_id\":\"h2\""), std::string::npos);
  EXPECT_NE(line.find("\"attempted_hypothesis_ids\":[\"h1\",\"h2\"]"), std::string::npos);
  EXPECT_NE(line.find("\"hypothesis_ids\":[\"h1\",\"h2\"]"), std::string::npos);

  std::filesystem::remove_all(dir);
}
