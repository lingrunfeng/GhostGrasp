#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ghost_mgg_core
{
struct TrialLogEvent
{
  std::string trial_id;
  std::string observation_id;
  std::string tree_name;
  std::string backend_name;
  std::string recover_status;
  std::size_t hypothesis_count = 0;
  std::vector<std::string> attempted_hypothesis_ids;
  std::string selected_hypothesis_id;
  std::string execute_status;
  std::string final_status;
  std::string failure_reason;
  double runtime_sec = 0.0;
};

std::string to_json_line(const TrialLogEvent & event);
}  // namespace ghost_mgg_core
