#include "ghost_mgg_bt/trial_logger.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ghost_mgg_bt
{
namespace
{
std::string quote(const std::string & value)
{
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::uppercase << std::setw(4)
              << std::setfill('0') << static_cast<int>(c)
              << std::dec << std::nouppercase << std::setfill(' ');
        } else {
          out << static_cast<char>(c);
        }
        break;
    }
  }
  out << '"';
  return out.str();
}

void append_string_array(std::ostream & out, const std::vector<std::string> & values)
{
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << quote(values[i]);
  }
  out << ']';
}

std::vector<std::string> hypothesis_ids(
  const std::vector<ghost_mgg_interfaces::msg::GeometryHypothesis> & hypotheses)
{
  std::vector<std::string> ids;
  ids.reserve(hypotheses.size());
  for (const auto & hypothesis : hypotheses) {
    ids.push_back(hypothesis.hypothesis_id);
  }
  return ids;
}
}  // namespace

TrialLogger::TrialLogger(std::filesystem::path directory)
: directory_(std::move(directory))
{
}

std::filesystem::path TrialLogger::write(const TrialLogRecord & record) const
{
  std::filesystem::create_directories(directory_);
  const auto filename = record.trial_id.empty() ? "trial_unknown.jsonl" : record.trial_id + ".jsonl";
  const auto path = directory_ / filename;

  std::ofstream out(path, std::ios::app);
  if (!out) {
    throw std::runtime_error("failed to open trial log: " + path.string());
  }

  out << std::setprecision(15);
  out << '{';
  out << "\"trial_id\":" << quote(record.trial_id) << ',';
  out << "\"observation_id\":" << quote(record.observation_id) << ',';
  out << "\"tree_name\":" << quote(record.tree_name) << ',';
  out << "\"backend_name\":" << quote(record.backend_name) << ',';
  out << "\"recover_status\":" << static_cast<unsigned>(record.recover_status) << ',';
  out << "\"execute_status\":" << static_cast<unsigned>(record.execute_status) << ',';
  out << "\"final_status\":" << quote(record.final_status) << ',';
  out << "\"selected_hypothesis_id\":" << quote(record.selected_hypothesis_id) << ',';
  out << "\"attempted_hypothesis_ids\":";
  append_string_array(out, record.attempted_hypothesis_ids);
  out << ',';
  out << "\"hypothesis_ids\":";
  append_string_array(out, hypothesis_ids(record.hypotheses));
  out << ',';
  out << "\"hypothesis_count\":" << record.hypotheses.size() << ',';
  out << "\"failure_reason\":" << quote(record.failure_reason) << ',';
  out << "\"runtime_sec\":" << record.runtime_sec;
  out << "}\n";

  return path;
}

}  // namespace ghost_mgg_bt
