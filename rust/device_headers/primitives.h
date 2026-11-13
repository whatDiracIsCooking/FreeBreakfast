/// @file
/// @brief Integer rounding in free_breakfast::math, written once for host and
/// device code.
///
/// Forked from src/core/primitives/primitives.h and trimmed to what rust/'s
/// launch wrappers use: idivup. The element-wise functions, the combinatorics
/// and align_up come back with the first wrapper that names them; the host
/// side's are Rust's own (fb_core_primitives).

#pragma once

// Standard headers first, before wwr's runtime.h below pulls in the vendor
// runtime header: under HIP, host_defines.h poisons __noinline__ for libc++'s
// __config, so a libc++ header reached after it fails to compile.
#include <concepts>

// FBF_HOST_DEVICE: __host__ __device__ __forceinline__ in a device pass. The
// runtime header is needed explicitly because clang's HIP mode pre-includes
// only __host__/__device__, and __forceinline__ comes from hip_runtime.h; it is
// reached through WarpWraps' neutral runtime.h, never a vendor header by name.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include <runtime.h>
#define FBF_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define FBF_HOST_DEVICE
#endif

namespace free_breakfast::math {

/// @brief Ceiling of @p a / @p b: the number of @p b -sized chunks needed to
/// cover @p a
///
/// Assumes non-negative operands; undefined for @p b == 0. The quotient plus
/// one for a remainder, rather than src/'s (a + b - 1) / b: the same value
/// wherever that one does not overflow, and none near the top of T's range.
/// It also keeps devtools/check-ceil-div.py, which exempts src/'s copy alone,
/// meaning what it says here.
template <std::integral T>
[[nodiscard]] FBF_HOST_DEVICE constexpr T idivup(const T a, const T b) {
  return static_cast<T>(a / b + (a % b != 0 ? 1 : 0));
}

} // namespace free_breakfast::math

#undef FBF_HOST_DEVICE
