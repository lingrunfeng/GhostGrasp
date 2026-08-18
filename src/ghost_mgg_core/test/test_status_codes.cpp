#include "ghost_mgg_core/status_codes.hpp"
#include "ghost_mgg_core/trial_log_event.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(StatusCodes, ConvertsEveryRecoverStatusToStableString)
{
  EXPECT_EQ(ghost_mgg_core::to_string(ghost_mgg_core::RecoverStatus::kUnknown), "unknown");
  EXPECT_EQ(ghost_mgg_core::to_string(ghost_mgg_core::RecoverStatus::kSucceeded), "succeeded");
  EXPECT_EQ(ghost_mgg_core::to_string(ghost_mgg_core::RecoverStatus::kFailed), "failed");
  EXPECT_EQ(
    ghost_mgg_core::to_string(ghost_mgg_core::RecoverStatus::kLowConfidence),
    "low_confidence");
  EXPECT_EQ(ghost_mgg_core::to_string(ghost_mgg_core::RecoverStatus::kTimeout), "timeout");
  EXPECT_EQ(ghost_mgg_core::to_string(ghost_mgg_core::RecoverStatus::kCanceled), "canceled");
}

TEST(StatusCodes, ConvertsUnknownRecoverStatusValueToUnknown)
{
  const auto invalid_status = static_cast<ghost_mgg_core::RecoverStatus>(255);

  EXPECT_EQ(ghost_mgg_core::to_string(invalid_status), "unknown");
}

TEST(TrialLogEvent, SerializesRequiredFieldsAsDeterministicJsonLine)
{
  ghost_mgg_core::TrialLogEvent event;
  event.trial_id = "trial_001";
  event.observation_id = "obs_001";
  event.tree_name = "tabletop_grasp";
  event.backend_name = "dummy";
  event.recover_status = "succeeded";
  event.hypothesis_count = 2;
  event.attempted_hypothesis_ids = {"h1", "h2"};
  event.selected_hypothesis_id = "h2";
  event.execute_status = "succeeded";
  event.final_status = "succeeded";
  event.failure_reason = "";
  event.runtime_sec = 1.25;

  const auto json = ghost_mgg_core::to_json_line(event);

  EXPECT_EQ(
    json,
    "{\"trial_id\":\"trial_001\","
    "\"observation_id\":\"obs_001\","
    "\"tree_name\":\"tabletop_grasp\","
    "\"backend_name\":\"dummy\","
    "\"recover_status\":\"succeeded\","
    "\"hypothesis_count\":2,"
    "\"attempted_hypothesis_ids\":[\"h1\",\"h2\"],"
    "\"selected_hypothesis_id\":\"h2\","
    "\"execute_status\":\"succeeded\","
    "\"final_status\":\"succeeded\","
    "\"failure_reason\":\"\","
    "\"runtime_sec\":1.25}\n");
}

TEST(TrialLogEvent, EscapesStringsForJsonlSafety)
{
  ghost_mgg_core::TrialLogEvent event;
  event.trial_id = "trial\"quoted";
  event.observation_id = "obs\\slash";
  event.failure_reason = "line\nbreak\tand\rreturn";
  event.attempted_hypothesis_ids = {"h\"1", "h\\2"};

  const auto json = ghost_mgg_core::to_json_line(event);

  EXPECT_NE(json.find("\"trial_id\":\"trial\\\"quoted\""), std::string::npos);
  EXPECT_NE(json.find("\"observation_id\":\"obs\\\\slash\""), std::string::npos);
  EXPECT_NE(json.find("\"attempted_hypothesis_ids\":[\"h\\\"1\",\"h\\\\2\"]"), std::string::npos);
  EXPECT_NE(json.find("\"failure_reason\":\"line\\nbreak\\tand\\rreturn\""), std::string::npos);
  EXPECT_EQ(json.back(), '\n');
}
