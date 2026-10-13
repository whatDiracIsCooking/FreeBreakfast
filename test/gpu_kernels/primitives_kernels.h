/// @file
/// @brief Host-callable launchers that evaluate free_breakfast::math on the
/// device, for test/gtest/primitives_gpu.cpp.
///
/// Plain C++ with no device code and no vendor header, so the gtest TU (a
/// module-importing .cpp, which cannot include a kernel) and primitives_kernels.cu
/// (which defines these) share one declaration. Each launcher takes HOST arrays:
/// it allocates device memory, copies in, launches, copies out and frees, so the
/// caller needs no runtime API of its own. Every launcher returns false if any
/// runtime call or the launch itself failed, and leaves @p out unspecified then.

#pragma once

#include "integral_types.h"

namespace free_breakfast::test_kernels {

/// @brief Which free_breakfast::math function a launch evaluates
///
/// Unary operations read @c x; binary ones @c x and @c y (in that argument
/// order, so atan2 is atan2(x, y)); fma reads all three. The predicates write
/// 1 or 0.
enum class MathOp : int32_t {
  Exp,
  Exp2,
  Expm1,
  Log,
  Log2,
  Log10,
  Log1p,
  Pow,
  Sqrt,
  Rsqrt,
  Sin,
  Cos,
  Tan,
  Asin,
  Acos,
  Atan,
  Atan2,
  Sinh,
  Cosh,
  Tanh,
  Floor,
  Ceil,
  Trunc,
  Round,
  Rint,
  Fabs,
  Hypot,
  Copysign,
  Fmod,
  Fmin,
  Fmax,
  Fma,
  IsNan,
  IsInf,
  IsFinite,
};

/// @brief Which free_breakfast::math combinatoric a launch evaluates; binomial
/// reads @c k, the others ignore it
enum class CombinatoricOp : int32_t {
  DoubleFactorialOdd,
  Factorial,
  Binomial,
};

/// @brief out[i] = op(x[i], y[i], z[i]) on the device, single precision
[[nodiscard]] bool evaluate_on_device(MathOp op, const float *x, const float *y, const float *z,
                                      float *out, size_t count);

/// @brief out[i] = op(x[i], y[i], z[i]) on the device, double precision
[[nodiscard]] bool evaluate_on_device(MathOp op, const double *x, const double *y, const double *z,
                                      double *out, size_t count);

/// @brief out[i] = op(n[i], k[i]) on the device
[[nodiscard]] bool combinatoric_on_device(CombinatoricOp op, const int32_t *n, const int32_t *k,
                                          double *out, size_t count);

} // namespace free_breakfast::test_kernels
