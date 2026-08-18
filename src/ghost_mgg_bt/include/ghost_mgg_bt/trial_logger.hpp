#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "ghost_mgg_interfaces/msg/geometry_hypothesis.hpp"

namespace ghost_mgg_bt
{

struct TrialLogRecord
{
  std::string trial_id;
  std::string observation_id;
  std::string tree_name;
  std::string backend_name;
  std::uint8_t recover_status = 0;
  std::uint8_t execute_status = 0;
  std::string final_status;
  std::string selected_hypothesis_id;
  std::vector<std::string> attempted_hypothesis_ids;
  std::vector<ghost_mgg_interfaces::msg::GeometryHypothesis> hypotheses;
  std::string failure_reason;
  double runtime_sec = 0.0;
};

class TrialLogger
{
public:
  explicit TrialLogger(std::filesystem::path directory);

  std::filesystem::path write(const TrialLogRecord & record) const;

private:
  std::filesystem::path directory_;
};

}  // namespace ghost_mgg_bt
