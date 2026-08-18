#pragma once

#include <cstdint>
#include <string_view>

namespace ghost_mgg_core
{
enum class RecoverStatus : std::uint8_t
{
  kUnknown = 0,
  kSucceeded = 1,
  kFailed = 2,
  kLowConfidence = 3,
  kTimeout = 4,
  kCanceled = 5
};

inline std::string_view to_string(const RecoverStatus status)
{
  switch (status) {
    case RecoverStatus::kSucceeded:
      return "succeeded";
    case RecoverStatus::kFailed:
      return "failed";
    case RecoverStatus::kLowConfidence:
      return "low_confidence";
    case RecoverStatus::kTimeout:
      return "timeout";
    case RecoverStatus::kCanceled:
      return "canceled";
    case RecoverStatus::kUnknown:
    default:
      return "unknown";
  }
}
}  // namespace ghost_mgg_core
