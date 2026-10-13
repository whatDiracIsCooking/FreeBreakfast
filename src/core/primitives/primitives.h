/// @file
/// @brief Math primitives in free_breakfast::math -- sqrt, log, sin, cos, ...,
/// factorial, binomial, idivup, align_up -- written once for host and device
/// code.
///
/// Every function here is a single definition that compiles on both sides:
///   - a device `.cu` #includes this header, and each function is
///     __host__ __device__ __forceinline__;
///   - primitives.cppm GMF-includes it, where the macro below expands to
///     nothing, and re-exports the names, so host module code does
///     `import free_breakfast.core.primitives;` and runs the same source.
/// There is one list of functions, here. primitives.cppm only names them.
///
/// The element-wise functions call the C math library's global names (::sqrtf,
/// ::sqrt, ...), not std::. Both vendors declare those names __host__ __device__:
/// in a device pass they resolve to the vendor's device implementation (CUDA's
/// sqrt.rn.f32, HIP's ocml / llvm.sqrt), in a host compile to libm. That is
/// what lets one body serve both sides. They are not WarpWraps'
/// wrappers/math/math.cuh: those templates are __device__ only, and math.cuh
/// #errors outside a device pass, so a host+device definition cannot forward to
/// them.
///
/// Each element-wise function is a template accepting exactly float or double,
/// dispatching to the f-suffixed name for float. The constraint is a requires-
/// clause rather than a static_assert, so an int argument is rejected at overload
/// resolution instead of being promoted to double the way a bare ::sqrt(2) would
/// be. It is spelled inline, not as a concept, because free_breakfast::real_fp
/// is wwr's (re-exported by free_breakfast.core.types), and a second concept with
/// that meaning here would be a different entity that merely agrees today.
///
/// Everything is in namespace free_breakfast::math rather than free_breakfast
/// itself: free_breakfast::log is the logging facade's namespace, and a function
/// cannot share its name.
///
/// Link free_breakfast::core::primitives_header from device code; it brings
/// wwr::device for the vendor runtime header's include path.

#pragma once

// Standard headers first, before the vendor runtime header below: under HIP,
// host_defines.h poisons __noinline__ for libc++'s __config, so a libc++ header
// reached after it fails to compile (the same ordering wwr's math.cuh keeps).
// <math.h>, not <cmath>: the bodies call the global C names, which <math.h> is
// what declares.
#include <math.h>

#include <concepts>
#include <type_traits>

// free_breakfast::int32_t, the width the combinatorics take their counts in.
#include "integral_types.h"

// FBF_HOST_DEVICE: __host__ __device__ __forceinline__ in a device
// pass; nothing in a host compile, where none of the three exists.
// __forceinline__ matches wwr's device math: these are a few instructions,
// called from inner loops, and a call would cost more than the body.
//
// The runtime header is included explicitly because clang's HIP mode
// pre-includes only __host__/__device__; __forceinline__ comes from
// hip_runtime.h, and without it every definition below fails with "unknown type
// name '__forceinline__'" -- on HIP only, since nvcc pre-includes the whole CUDA
// runtime.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif
#define FBF_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define FBF_HOST_DEVICE
#endif

// The leading part of every element-wise declaration: a template over float or
// double only, host and device. #undef'd at the end of this file, with the
// generators below -- this header is #included as text, so anything left
// defined would leak into every includer.
#define FBF_MATH_FP_TEMPLATE                                                                       \
  template <typename T>                                                                            \
    requires(std::is_same_v<T, float> || std::is_same_v<T, double>)                                \
  [[nodiscard]] FBF_HOST_DEVICE

// T -> T, T x T -> T and T x T x T -> T: the f-suffixed C function for float,
// the bare one for double, chosen at compile time.
#define FBF_MATH_UNARY(fn, ffn, dfn)                                                               \
  FBF_MATH_FP_TEMPLATE T fn(const T x) {                                                           \
    if constexpr (std::is_same_v<T, float>) {                                                      \
      return ::ffn(x);                                                                             \
    } else {                                                                                       \
      return ::dfn(x);                                                                             \
    }                                                                                              \
  }

#define FBF_MATH_BINARY(fn, ffn, dfn)                                                              \
  FBF_MATH_FP_TEMPLATE T fn(const T x, const T y) {                                                \
    if constexpr (std::is_same_v<T, float>) {                                                      \
      return ::ffn(x, y);                                                                          \
    } else {                                                                                       \
      return ::dfn(x, y);                                                                          \
    }                                                                                              \
  }

#define FBF_MATH_TERNARY(fn, ffn, dfn)                                                             \
  FBF_MATH_FP_TEMPLATE T fn(const T x, const T y, const T z) {                                     \
    if constexpr (std::is_same_v<T, float>) {                                                      \
      return ::ffn(x, y, z);                                                                       \
    } else {                                                                                       \
      return ::dfn(x, y, z);                                                                       \
    }                                                                                              \
  }

// T -> bool. No f-suffixed spelling: C++'s <math.h> and both vendors provide a
// single overloaded ::isnan (etc.) for float and double alike.
#define FBF_MATH_PREDICATE(fn)                                                                     \
  template <typename T>                                                                            \
    requires(std::is_same_v<T, float> || std::is_same_v<T, double>)                                \
  [[nodiscard]] FBF_HOST_DEVICE bool fn(const T x) {                                               \
    return ::fn(x);                                                                                \
  }

namespace free_breakfast::math {

// ========================================================================
// Exponential / logarithmic
//
// expm1 and log1p keep precision for arguments near zero, which `exp(x) - 1`
// and `log(1 + x)` lose.
// ========================================================================

FBF_MATH_UNARY(exp, expf, exp)
FBF_MATH_UNARY(exp2, exp2f, exp2)
FBF_MATH_UNARY(expm1, expm1f, expm1)
FBF_MATH_UNARY(log, logf, log)
FBF_MATH_UNARY(log2, log2f, log2)
FBF_MATH_UNARY(log10, log10f, log10)
FBF_MATH_UNARY(log1p, log1pf, log1p)

// ========================================================================
// Powers and roots
// ========================================================================

FBF_MATH_BINARY(pow, powf, pow)
FBF_MATH_UNARY(sqrt, sqrtf, sqrt)

/// @brief Reciprocal square root of @p x
///
/// The one function whose two sides differ: the C library has no rsqrt, so the
/// host computes `1 / sqrt(x)` -- two roundings -- while a device pass uses the
/// vendor's rsqrt intrinsic. Results can differ in the last bit; do not compare
/// a host and a device result for exact equality. The branch is on the device
/// *sub*-pass macros, since a __host__ __device__ body is compiled twice within
/// one .cu and only the device half may name ::rsqrtf.
FBF_MATH_FP_TEMPLATE T rsqrt(const T x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if constexpr (std::is_same_v<T, float>) {
    return ::rsqrtf(x);
  } else {
    return ::rsqrt(x);
  }
#else
  return T{1} / free_breakfast::math::sqrt(x);
#endif
}

// ========================================================================
// Trigonometric, inverse trigonometric, hyperbolic
// ========================================================================

FBF_MATH_UNARY(sin, sinf, sin)
FBF_MATH_UNARY(cos, cosf, cos)
FBF_MATH_UNARY(tan, tanf, tan)
FBF_MATH_UNARY(asin, asinf, asin)
FBF_MATH_UNARY(acos, acosf, acos)
FBF_MATH_UNARY(atan, atanf, atan)
FBF_MATH_BINARY(atan2, atan2f, atan2)
FBF_MATH_UNARY(sinh, sinhf, sinh)
FBF_MATH_UNARY(cosh, coshf, cosh)
FBF_MATH_UNARY(tanh, tanhf, tanh)

// ========================================================================
// Rounding to integral values (still returning the floating type)
// ========================================================================

FBF_MATH_UNARY(floor, floorf, floor)
FBF_MATH_UNARY(ceil, ceilf, ceil)
FBF_MATH_UNARY(trunc, truncf, trunc)
FBF_MATH_UNARY(round, roundf, round)
FBF_MATH_UNARY(rint, rintf, rint)

// ========================================================================
// Magnitude, sign, remainder, min/max, fused multiply-add
//
// fmin/fmax return the non-NaN operand when exactly one is NaN, which
// `x < y ? x : y` does not. fma rounds x*y + z once; a bare `x * y + z` may be
// rounded twice under -ffp-contract=off.
// ========================================================================

FBF_MATH_UNARY(fabs, fabsf, fabs)
FBF_MATH_BINARY(hypot, hypotf, hypot)
FBF_MATH_BINARY(copysign, copysignf, copysign)
FBF_MATH_BINARY(fmod, fmodf, fmod)
FBF_MATH_BINARY(fmin, fminf, fmin)
FBF_MATH_BINARY(fmax, fmaxf, fmax)
FBF_MATH_TERNARY(fma, fmaf, fma)

// ========================================================================
// Classification
// ========================================================================

FBF_MATH_PREDICATE(isnan)
FBF_MATH_PREDICATE(isinf)
FBF_MATH_PREDICATE(isfinite)

// ========================================================================
// Small exact-in-double combinatorics
//
// Exact while the product fits a double's 53-bit significand -- far beyond the
// angular momenta a basis set reaches, which is what these serve. They take
// int32_t and return double because their callers immediately fold the result
// into floating-point arithmetic (a normalization, a harmonic coefficient).
// Plain loops, so unlike the functions above they are also constexpr.
// ========================================================================

/// @brief n!! over odd @p n: n * (n-2) * ... * 1; the empty product, 1.0, for
/// n <= 1
///
/// The (2l-1)!! of a Gaussian primitive's normalization and of the per-shell
/// spherical scale sqrt((2l-1)!!).
[[nodiscard]] FBF_HOST_DEVICE constexpr double double_factorial_odd(const int32_t n) {
  double result = 1.0;
  for (int32_t k = n; k > 1; k -= 2) {
    result *= static_cast<double>(k);
  }
  return result;
}

/// @brief n! as a double; 1.0 for n <= 1
[[nodiscard]] FBF_HOST_DEVICE constexpr double factorial(const int32_t n) {
  double result = 1.0;
  for (int32_t k = 2; k <= n; ++k) {
    result *= static_cast<double>(k);
  }
  return result;
}

/// @brief Binomial coefficient C(n, k) as a double; zero outside 0 <= k <= n
[[nodiscard]] FBF_HOST_DEVICE constexpr double binomial(const int32_t n, const int32_t k) {
  if (k < 0 || k > n) {
    return 0.0;
  }
  double result = 1.0;
  for (int32_t i = 0; i < k; ++i) {
    result = result * static_cast<double>(n - i) / static_cast<double>(i + 1);
  }
  return result;
}

// ========================================================================
// Integer rounding
//
// Grid and block sizing, and padding a count to a multiple: the host sizes a
// launch with the same arithmetic a kernel uses to bound its loop. Any integral
// type, so a size_t byte count and a uint32_t thread count both pass unconverted.
// ========================================================================

/// @brief Ceiling of @p a / @p b: the number of @p b -sized chunks needed to
/// cover @p a
///
/// Assumes non-negative operands; undefined for @p b == 0, and @p a + @p b may
/// overflow near the top of T's range.
template <std::integral T>
[[nodiscard]] FBF_HOST_DEVICE constexpr T idivup(const T a, const T b) {
  return (a + b - 1) / b;
}

/// @brief Round @p a up to the next multiple of @p b
///
/// Same domain as idivup: non-negative operands, undefined for @p b == 0.
template <std::integral T>
[[nodiscard]] FBF_HOST_DEVICE constexpr T align_up(const T a, const T b) {
  return idivup(a, b) * b;
}

} // namespace free_breakfast::math

#undef FBF_MATH_PREDICATE
#undef FBF_MATH_TERNARY
#undef FBF_MATH_BINARY
#undef FBF_MATH_UNARY
#undef FBF_MATH_FP_TEMPLATE
#undef FBF_HOST_DEVICE
