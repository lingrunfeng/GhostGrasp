#pragma once

#include <cstddef>
#include <vector>

#include "ghost_mgg_interfaces/msg/geometry_hypothesis.hpp"

namespace ghost_mgg_backends
{

using GeometryHypothesis = ghost_mgg_interfaces::msg::GeometryHypothesis;

std::vector<GeometryHypothesis> make_m2_mask_extrusion_hypotheses(std::size_t max_hypotheses);

}  // namespace ghost_mgg_backends
