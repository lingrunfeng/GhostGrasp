#include <gtest/gtest.h>

#include <string>

#include "ghost_mgg_backends/dummy_execute_server.hpp"
#include "ghost_mgg_backends/dummy_recovery_server.hpp"

TEST(DummyRecoveryServer, BuildsRankedHypotheses)
{
  const auto hypotheses =
    ghost_mgg_backends::make_dummy_hypotheses("dummy", "trial_1", "obs_1", 3);

  ASSERT_EQ(hypotheses.size(), 3u);
  EXPECT_EQ(hypotheses[0].hypothesis_id, "h1");
  EXPECT_EQ(hypotheses[1].hypothesis_id, "h2");
  EXPECT_EQ(hypotheses[2].hypothesis_id, "h3");
  EXPECT_GT(hypotheses[0].score.total, hypotheses[1].score.total);
  EXPECT_GT(hypotheses[1].score.total, hypotheses[2].score.total);
  ASSERT_EQ(hypotheses[0].grasp_candidates.size(), 1u);
  EXPECT_EQ(hypotheses[0].grasp_candidates[0].grasp_id, "g_h1_top");
  EXPECT_EQ(hypotheses[0].provenance, "dummy");
}

TEST(DummyRecoveryServer, RespectsMaxHypotheses)
{
  const auto hypotheses =
    ghost_mgg_backends::make_dummy_hypotheses("dummy", "trial_1", "obs_1", 2);

  ASSERT_EQ(hypotheses.size(), 2u);
  EXPECT_EQ(hypotheses[0].hypothesis_id, "h1");
  EXPECT_EQ(hypotheses[1].hypothesis_id, "h2");
}

TEST(DummyExecuteServer, FailsFirstHypothesisDeterministically)
{
  const auto h1 = ghost_mgg_backends::evaluate_dummy_execute(
    "fail_first_then_succeed", "h1");
  const auto h2 = ghost_mgg_backends::evaluate_dummy_execute(
    "fail_first_then_succeed", "h2");

  EXPECT_FALSE(h1.succeeded);
  EXPECT_EQ(h1.status, ghost_mgg_backends::kExecuteStatusFailed);
  EXPECT_EQ(h1.failure_reason, "dummy executor rejected first hypothesis");
  EXPECT_TRUE(h2.succeeded);
  EXPECT_EQ(h2.status, ghost_mgg_backends::kExecuteStatusSucceeded);
}

TEST(DummyExecuteServer, SupportsAlwaysSucceedAndFailAllModes)
{
  const auto ok = ghost_mgg_backends::evaluate_dummy_execute(
    "always_succeed", "h1");
  const auto fail = ghost_mgg_backends::evaluate_dummy_execute(
    "fail_all", "h2");

  EXPECT_TRUE(ok.succeeded);
  EXPECT_EQ(ok.status, ghost_mgg_backends::kExecuteStatusSucceeded);
  EXPECT_FALSE(fail.succeeded);
  EXPECT_EQ(fail.status, ghost_mgg_backends::kExecuteStatusFailed);
  EXPECT_EQ(fail.failure_reason, "dummy executor configured to fail all hypotheses");
}
