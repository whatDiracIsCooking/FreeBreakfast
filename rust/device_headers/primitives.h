/// @file
/// @brief Integer rounding and two element-wise functions in
/// free_breakfast::math, written once for host and device code.
///
/// Forked from src/core/primitives/primitives.h and trimmed to what rust/'s
/// launch wrappers use: idivup, pow and fma. The other element-wise
/// functions, the combinatorics and align_up come back with the first wrapper
/// that names them; the host side's are Rust's own (fb_core_primitives).
///
/// pow and fma call the C math library's global names, as src/'s do: both
/// vendors declare them __host__ __device__, so in a device pass they resolve
/// to the vendor's device implementation. Each accepts exactly float or
/// double, dispatching to the f-suffixed name for float, so an int argument
/// is rejected rather than promoted.

#pragma once

// Standard headers first, before wwr's runtime.h below pulls in the vendor
// runtime header: under HIP, host_defines.h poisons __noinline__ for libc++'s
// __config, so a libc++ header reached after it fails to compile.
// <math.h>, not <cmath>: the bodies call the global C names, which <math.h> is
// what declares.
#include <math.h>

#include <concepts>
#include <type_traits>

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

/// @brief @p x raised to the power @p y
template <typename T>
  requires(std::is_same_v<T, float> || std::is_same_v<T, double>)
[[nodiscard]] FBF_HOST_DEVICE T pow(const T x, const T y) {
  if constexpr (std::is_same_v<T, float>) {
    return ::powf(x, y);
  } else {
    return ::pow(x, y);
  }
}

/// @brief @p x * @p y + @p z, rounded once
template <typename T>
  requires(std::is_same_v<T, float> || std::is_same_v<T, double>)
[[nodiscard]] FBF_HOST_DEVICE T fma(const T x, const T y, const T z) {
  if constexpr (std::is_same_v<T, float>) {
    return ::fmaf(x, y, z);
  } else {
    return ::fma(x, y, z);
  }
}

} // namespace free_breakfast::math

#undef FBF_HOST_DEVICE
