/// @file
/// @brief A single atom: an element positioned at a point in space.

#pragma once

#include "element.h"
#include "vec.h"

namespace free_breakfast {

/// @brief One atom: a nucleus of a given element at a Cartesian position.
///
/// Plain aggregate — the position and the element are public with no invariant
/// to protect, so callers brace-initialise directly
/// (`Atom{{x, y, z}, Element::C}`). The position is the backend's own
/// @c double3 (@c Vec<double, 3>), in bohr, so a host-built atom and a
/// kernel-built one share a layout. A default-constructed @c Atom sits at the
/// origin carrying @c Element::Sentinel, i.e. "no element yet".
struct Atom {
  Vec<double, 3> position{0.0, 0.0, 0.0};
  Element element{Element{0}};
};

} // namespace free_breakfast
