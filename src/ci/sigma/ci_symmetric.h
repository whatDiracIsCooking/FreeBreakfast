/// @file
/// @brief The symmetric layout's triangle (see CiLayout in sigma.cppm), for
/// device code: the GPU sigma's gathers, ci_unpack, and ci.solver's S².
///
/// Determinant (a, b), a ≥ b, of a vector over one StringSet for both spins is
/// triangle element a (a + 1) / 2 + b; one below the triangle, a < b, is read
/// from its mirror (b, a), and every off-diagonal element is stored scaled by
/// √2.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "runtime.h"

namespace free_breakfast::device {

/// @brief √2 and 1/√2, the off-diagonal scaling and its inverse.
inline constexpr double kCiSqrt2 = 1.4142135623730950488;
inline constexpr double kCiInvSqrt2 = 0.70710678118654752440;

/// @brief The triangle element of determinant (a, b) or its mirror.
__device__ inline size_t ci_triangle(const int32_t a, const int32_t b) {
  const auto hi = static_cast<size_t>(a > b ? a : b);
  const auto lo = static_cast<size_t>(a > b ? b : a);
  return (hi * (hi + 1) / 2) + lo;
}

/// @brief The determinant (a, b), a ≥ b, of triangle element @p element.
__device__ inline void ci_triangle_strings(const int32_t element, int32_t &a, int32_t &b) {
  // The root of a (a + 1) / 2 = element, then one step either way for the
  // double's rounding.
  a = static_cast<int32_t>((sqrt((8.0 * static_cast<double>(element)) + 1.0) - 1.0) * 0.5);
  while (static_cast<int64_t>(a) * (a + 1) / 2 > element) {
    --a;
  }
  while (static_cast<int64_t>(a + 1) * (a + 2) / 2 <= element) {
    ++a;
  }
  b = element - static_cast<int32_t>(static_cast<int64_t>(a) * (a + 1) / 2);
}

} // namespace free_breakfast::device
