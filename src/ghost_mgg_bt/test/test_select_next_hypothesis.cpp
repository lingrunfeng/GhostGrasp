#include <string>
#include <vector>

#include <behaviortree_cpp/bt_factory.h>
#include <gtest/gtest.h>

#include "ghost_mgg_bt/select_next_hypothesis_node.hpp"
#include "ghost_mgg_interfaces/msg/geometry_hypothesis.hpp"

namespace
{
using ghost_mgg_interfaces::msg::GeometryHypothesis;

std::vector<GeometryHypothesis> make_hypotheses()
{
  std::vector<GeometryHypothesis> hypotheses(3);
  hypotheses[0].hypothesis_id = "h1";
  hypotheses[1].hypothesis_id = "h2";
  hypotheses[2].hypothesis_id = "h3";
  return hypotheses;
}
}  // namespace

TEST(SelectNextHypothesis, SelectsRankedHypothesesUntilExhausted)
{
  BT::BehaviorTreeFactory factory;
  factory.registerNodeType<ghost_mgg_bt::SelectNextHypothesis>("SelectNextHypothesis");

  auto blackboard = BT::Blackboard::create();
  blackboard->set("hypotheses", make_hypotheses());
  blackboard->set("hypothesis_index", 0);
  blackboard->set("attempted_hypothesis_ids", std::vector<std::string>{});

  const auto xml = R"(
    <root BTCPP_format="4">
      <BehaviorTree ID="TestTree">
        <SelectNextHypothesis hypotheses="{hypotheses}"
                              hypothesis_index="{hypothesis_index}"
                              selected_hypothesis="{selected_hypothesis}"
                              selected_hypothesis_id="{selected_hypothesis_id}"
                              attempted_hypothesis_ids="{attempted_hypothesis_ids}" />
      </BehaviorTree>
    </root>)";

  auto tree = factory.createTreeFromText(xml, blackboard);

  EXPECT_EQ(tree.tickOnce(), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(blackboard->get<std::string>("selected_hypothesis_id"), "h1");

  EXPECT_EQ(tree.tickOnce(), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(blackboard->get<std::string>("selected_hypothesis_id"), "h2");

  EXPECT_EQ(tree.tickOnce(), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(blackboard->get<std::string>("selected_hypothesis_id"), "h3");

  EXPECT_EQ(tree.tickOnce(), BT::NodeStatus::FAILURE);
  const auto attempted = blackboard->get<std::vector<std::string>>("attempted_hypothesis_ids");
  ASSERT_EQ(attempted.size(), 3U);
  EXPECT_EQ(attempted[0], "h1");
  EXPECT_EQ(attempted[1], "h2");
  EXPECT_EQ(attempted[2], "h3");
}
