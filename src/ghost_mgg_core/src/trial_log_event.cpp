#include "ghost_mgg_core/trial_log_event.hpp"

#include <iomanip>
#include <ostream>
#include <sstream>

namespace ghost_mgg_core
{
namespace
{
void append_hex_escape(std::ostream & out, const unsigned char value)
{
  out << "\\u"
      << std::hex << std::uppercase << std::setw(4) << std::setfill('0')
      << static_cast<int>(value)
      << std::dec << std::nouppercase << std::setfill(' ');
}

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
          append_hex_escape(out, c);
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
}  // namespace

std::string to_json_line(const TrialLogEvent & event)
{
  std::ostringstream out;
  out << std::setprecision(15);
  out << '{';
  out << "\"trial_id\":" << quote(event.trial_id) << ',';
  out << "\"observation_id\":" << quote(event.observation_id) << ',';
  out << "\"tree_name\":" << quote(event.tree_name) << ',';
  out << "\"backend_name\":" << quote(event.backend_name) << ',';
  out << "\"recover_status\":" << quote(event.recover_status) << ',';
  out << "\"hypothesis_count\":" << event.hypothesis_count << ',';
  out << "\"attempted_hypothesis_ids\":";
  append_string_array(out, event.attempted_hypothesis_ids);
  out << ',';
  out << "\"selected_hypothesis_id\":" << quote(event.selected_hypothesis_id) << ',';
  out << "\"execute_status\":" << quote(event.execute_status) << ',';
  out << "\"final_status\":" << quote(event.final_status) << ',';
  out << "\"failure_reason\":" << quote(event.failure_reason) << ',';
  out << "\"runtime_sec\":" << event.runtime_sec;
  out << "}\n";
  return out.str();
}
}  // namespace ghost_mgg_core
