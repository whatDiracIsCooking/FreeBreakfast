/// @file
/// @brief A single atom: an element positioned at a point in space.

#pragma once

#include "element.h"

namespace free_breakfast {

/// @brief One atom: a nucleus of a given element at a Cartesian position.
///
/// Plain aggregate — the coordinates and the element are public with no
/// invariant to protect, so callers brace-initialise directly
/// (`Atom<double>{x, y, z, Element::C}`). A default-constructed @c Atom sits at
/// the origin carrying @c Element::Sentinel, i.e. "no element yet".
///
/// @tparam T Scalar type of the coordinates (e.g. @c float or @c double), in
///           bohr.
template <typename T>
struct Atom {
  T x{T{0}};
  T y{T{0}};
  T z{T{0}};
  Element element{Element{0}};
};

} // namespace free_breakfast
