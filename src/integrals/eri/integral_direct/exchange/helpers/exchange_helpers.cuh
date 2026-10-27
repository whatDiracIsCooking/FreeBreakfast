/// @file
/// @brief What the exchange helpers' device TUs share: launch shapes, the
/// argument checks, the bits of a non-negative double, and a distance.
///
/// Device-only: #include it into one of this directory's .cu files, after
/// block_reduce.cuh when that is wanted too. Each name here is used by more
/// than one of them; a helper only one kernel needs stays in that kernel's
/// anonymous namespace. Being a header, everything here is public (docs/style.md
/// "Headers shared with device code"), so it is in its own namespace, and every
/// function is `inline` so the TUs' copies are one at device link.

#pragma once

// Standard headers before the vendor runtime header, which primitives.h brings
// in last: under HIP it must not precede libc++.
#include "block_params.h"
#include "integral_types.h"
#include "primitives.h"

#include "exchange_helpers_bridge.h"

namespace free_breakfast::device::exchange {

/// @brief Warps in a kFourWarps block: the block table's and the bounds'
/// block_reduce width.
inline constexpr uint32_t kTableWarps = kFourWarps / kWarpSize;

/// @brief The most elements a launcher indexes, INT32_MAX: a buffer past it is
/// rejected rather than launched.
inline constexpr int64_t kMaxElements = 2147483647;

/// @brief The Cartesians of a shell of momentum @p l, (l + 1)(l + 2) / 2.
inline int32_t cartesian_count(const int32_t l) {
  return (l + 1) * (l + 2) / 2;
}

/// @brief Whether @p l is in [0, kExchangeMaxMomentum], K's limit.
inline bool valid_momentum(const int32_t l) {
  return l >= 0 && l <= kExchangeMaxMomentum;
}

/// @brief The kFourWarps blocks of a one-thread-per-element launch over @p n.
inline uint32_t element_blocks(const int64_t n) {
  return static_cast<uint32_t>(math::idivup(n, static_cast<int64_t>(kFourWarps)));
}

/// @brief A double's bits, as the vendors' atomicMax takes them: a non-negative
/// double orders as its bits, so a maximum of |x| or of a bound is an integer
/// one.
// int-type-ok: atomicMax and __double_as_longlong are declared on this type
using DoubleBits = unsigned long long;

/// @brief The bits of @p x, as DoubleBits.
__device__ inline DoubleBits bits_of(const double x) {
  return static_cast<DoubleBits>(__double_as_longlong(x));
}

/// @brief The distance between (@p ax, @p ay, @p az) and (@p bx, @p by, @p bz).
__device__ inline double distance(const double ax, const double ay, const double az,
                                  const double bx, const double by, const double bz) {
  const double dx = ax - bx;
  const double dy = ay - by;
  const double dz = az - bz;
  return math::sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace free_breakfast::device::exchange
