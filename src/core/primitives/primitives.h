/// @file
/// @brief Math primitives in free_breakfast::math -- sqrt, log, sin, cos, ...,
/// factorial, binomial -- written once for host and device code.
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

#include <type_traits>

// free_breakfast::int32_t, the width the combinatorics take their counts in.
#include "integral_types.h"

// FREE_BREAKFAST_HOST_DEVICE: __host__ __device__ __forceinline__ in a device
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
#define FREE_BREAKFAST_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define FREE_BREAKFAST_HOST_DEVICE
#endif

// The leading part of every element-wise declaration: a template over float or
// double only, host and device. #undef'd at the end of this file, with the
// generators below -- this header is #included as text, so anything left
// defined would leak into every includer.
#define FREE_BREAKFAST_MATH_FP_TEMPLATE                                                            \
  template <typename T>                                                                            \
    requires(std::is_same_v<T, float> || std::is_same_v<T, double>)                                \
  [[nodiscard]] FREE_BREAKFAST_HOST_DEVICE

// T -> T, T x T -> T and T x T x T -> T: the f-suffixed C function for float,
// the bare one for double, chosen at compile time.
#define FREE_BREAKFAST_MATH_UNARY(fn, ffn, dfn)                                                    \
  FREE_BREAKFAST_MATH_FP_TEMPLATE T fn(const T x) {                                                \
    if constexpr (std::is_same_v<T, float>) {                                                      \
      return ::ffn(x);                                                                             \
    } else {                                                                                       \
      return ::dfn(x);                                                                             \
    }                                                                                              \
  }

#define FREE_BREAKFAST_MATH_BINARY(fn, ffn, dfn)                                                   \
  FREE_BREAKFAST_MATH_FP_TEMPLATE T fn(const T x, const T y) {                                     \
    if constexpr (std::is_same_v<T, float>) {                                                      \
      return ::ffn(x, y);                                                                          \
    } else {                                                                                       \
      return ::dfn(x, y);                                                                          \
    }                                                                                              \
  }

#define FREE_BREAKFAST_MATH_TERNARY(fn, ffn, dfn)                                                  \
  FREE_BREAKFAST_MATH_FP_TEMPLATE T fn(const T x, const T y, const T z) {                          \
    if constexpr (std::is_same_v<T, float>) {                                                      \
      return ::ffn(x, y, z);                                                                       \
    } else {                                                                                       \
      return ::dfn(x, y, z);                                                                       \
    }                                                                                              \
  }

// T -> bool. No f-suffixed spelling: C++'s <math.h> and both vendors provide a
// single overloaded ::isnan (etc.) for float and double alike.
#define FREE_BREAKFAST_MATH_PREDICATE(fn)                                                          \
  template <typename T>                                                                            \
    requires(std::is_same_v<T, float> || std::is_same_v<T, double>)                                \
  [[nodiscard]] FREE_BREAKFAST_HOST_DEVICE bool fn(const T x) {                                    \
    return ::fn(x);                                                                                \
  }

namespace free_breakfast::math {

// ========================================================================
// Exponential / logarithmic
//
// expm1 and log1p keep precision for arguments near zero, which `exp(x) - 1`
// and `log(1 + x)` lose.
// ========================================================================

FREE_BREAKFAST_MATH_UNARY(exp, expf, exp)
FREE_BREAKFAST_MATH_UNARY(exp2, exp2f, exp2)
FREE_BREAKFAST_MATH_UNARY(expm1, expm1f, expm1)
FREE_BREAKFAST_MATH_UNARY(log, logf, log)
FREE_BREAKFAST_MATH_UNARY(log2, log2f, log2)
FREE_BREAKFAST_MATH_UNARY(log10, log10f, log10)
FREE_BREAKFAST_MATH_UNARY(log1p, log1pf, log1p)

// ========================================================================
// Powers and roots
// ========================================================================

FREE_BREAKFAST_MATH_BINARY(pow, powf, pow)
FREE_BREAKFAST_MATH_UNARY(sqrt, sqrtf, sqrt)

/// @brief Reciprocal square root of @p x
///
/// The one function whose two sides differ: the C library has no rsqrt, so the
/// host computes `1 / sqrt(x)` -- two roundings -- while a device pass uses the
/// vendor's rsqrt intrinsic. Results can differ in the last bit; do not compare
/// a host and a device result for exact equality. The branch is on the device
/// *sub*-pass macros, since a __host__ __device__ body is compiled twice within
/// one .cu and only the device half may name ::rsqrtf.
FREE_BREAKFAST_MATH_FP_TEMPLATE T rsqrt(const T x) {
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

FREE_BREAKFAST_MATH_UNARY(sin, sinf, sin)
FREE_BREAKFAST_MATH_UNARY(cos, cosf, cos)
FREE_BREAKFAST_MATH_UNARY(tan, tanf, tan)
FREE_BREAKFAST_MATH_UNARY(asin, asinf, asin)
FREE_BREAKFAST_MATH_UNARY(acos, acosf, acos)
FREE_BREAKFAST_MATH_UNARY(atan, atanf, atan)
FREE_BREAKFAST_MATH_BINARY(atan2, atan2f, atan2)
FREE_BREAKFAST_MATH_UNARY(sinh, sinhf, sinh)
FREE_BREAKFAST_MATH_UNARY(cosh, coshf, cosh)
FREE_BREAKFAST_MATH_UNARY(tanh, tanhf, tanh)

// ========================================================================
// Rounding to integral values (still returning the floating type)
// ========================================================================

FREE_BREAKFAST_MATH_UNARY(floor, floorf, floor)
FREE_BREAKFAST_MATH_UNARY(ceil, ceilf, ceil)
FREE_BREAKFAST_MATH_UNARY(trunc, truncf, trunc)
FREE_BREAKFAST_MATH_UNARY(round, roundf, round)
FREE_BREAKFAST_MATH_UNARY(rint, rintf, rint)

// ========================================================================
// Magnitude, sign, remainder, min/max, fused multiply-add
//
// fmin/fmax return the non-NaN operand when exactly one is NaN, which
// `x < y ? x : y` does not. fma rounds x*y + z once; a bare `x * y + z` may be
// rounded twice under -ffp-contract=off.
// ========================================================================

FREE_BREAKFAST_MATH_UNARY(fabs, fabsf, fabs)
FREE_BREAKFAST_MATH_BINARY(hypot, hypotf, hypot)
FREE_BREAKFAST_MATH_BINARY(copysign, copysignf, copysign)
FREE_BREAKFAST_MATH_BINARY(fmod, fmodf, fmod)
FREE_BREAKFAST_MATH_BINARY(fmin, fminf, fmin)
FREE_BREAKFAST_MATH_BINARY(fmax, fmaxf, fmax)
FREE_BREAKFAST_MATH_TERNARY(fma, fmaf, fma)

// ========================================================================
// Classification
// ========================================================================

FREE_BREAKFAST_MATH_PREDICATE(isnan)
FREE_BREAKFAST_MATH_PREDICATE(isinf)
FREE_BREAKFAST_MATH_PREDICATE(isfinite)

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
[[nodiscard]] FREE_BREAKFAST_HOST_DEVICE constexpr double double_factorial_odd(const int32_t n) {
  double result = 1.0;
  for (int32_t k = n; k > 1; k -= 2) {
    result *= static_cast<double>(k);
  }
  return result;
}

/// @brief n! as a double; 1.0 for n <= 1
[[nodiscard]] FREE_BREAKFAST_HOST_DEVICE constexpr double factorial(const int32_t n) {
  double result = 1.0;
  for (int32_t k = 2; k <= n; ++k) {
    result *= static_cast<double>(k);
  }
  return result;
}

/// @brief Binomial coefficient C(n, k) as a double; zero outside 0 <= k <= n
[[nodiscard]] FREE_BREAKFAST_HOST_DEVICE constexpr double binomial(const int32_t n,
                                                                   const int32_t k) {
  if (k < 0 || k > n) {
    return 0.0;
  }
  double result = 1.0;
  for (int32_t i = 0; i < k; ++i) {
    result = result * static_cast<double>(n - i) / static_cast<double>(i + 1);
  }
  return result;
}

} // namespace free_breakfast::math

#undef FREE_BREAKFAST_MATH_PREDICATE
#undef FREE_BREAKFAST_MATH_TERNARY
#undef FREE_BREAKFAST_MATH_BINARY
#undef FREE_BREAKFAST_MATH_UNARY
#undef FREE_BREAKFAST_MATH_FP_TEMPLATE
#undef FREE_BREAKFAST_HOST_DEVICE
