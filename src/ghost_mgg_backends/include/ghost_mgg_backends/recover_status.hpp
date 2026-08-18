#pragma once

#include <cstdint>

namespace ghost_mgg_backends
{

constexpr std::uint8_t kRecoverStatusSucceeded = 1;
constexpr std::uint8_t kRecoverStatusFailed = 2;
constexpr std::uint8_t kRecoverStatusLowConfidence = 3;
constexpr std::uint8_t kRecoverStatusTimeout = 4;
constexpr std::uint8_t kRecoverStatusCanceled = 5;

}  // namespace ghost_mgg_backends
