/// @file
/// @brief Math primitives (sqrt, log, sin, cos, ..., factorial, binomial,
/// idivup, align_up) for host module code.
///
/// Usage:
///   import free_breakfast.core.primitives;
///
///   double r = free_breakfast::math::sqrt(dx * dx + dy * dy + dz * dz);
///
/// Nothing is defined here. Every function is written once, in primitives.h,
/// as a single host+device definition; this unit GMF-includes that header --
/// where its __host__ __device__ __forceinline__ macro expands to nothing -- and
/// re-exports the names, so host module code and a device .cu run the same
/// source. See primitives.h for how one body serves both sides, and for the one
/// function (rsqrt) whose host and device results can differ in the last bit.
///
/// The list below must name everything primitives.h defines: a name missing here
/// is still callable from a .cu but not importable.

module;

#include "primitives.h"

export module free_breakfast.core.primitives;

export namespace free_breakfast::math {

// Exponential / logarithmic
using free_breakfast::math::exp;
using free_breakfast::math::exp2;
using free_breakfast::math::expm1;
using free_breakfast::math::log;
using free_breakfast::math::log10;
using free_breakfast::math::log1p;
using free_breakfast::math::log2;

// Powers and roots
using free_breakfast::math::pow;
using free_breakfast::math::rsqrt;
using free_breakfast::math::sqrt;

// Trigonometric, inverse trigonometric, hyperbolic
using free_breakfast::math::acos;
using free_breakfast::math::asin;
using free_breakfast::math::atan;
using free_breakfast::math::atan2;
using free_breakfast::math::cos;
using free_breakfast::math::cosh;
using free_breakfast::math::sin;
using free_breakfast::math::sinh;
using free_breakfast::math::tan;
using free_breakfast::math::tanh;

// Rounding to integral values
using free_breakfast::math::ceil;
using free_breakfast::math::floor;
using free_breakfast::math::rint;
using free_breakfast::math::round;
using free_breakfast::math::trunc;

// Magnitude, sign, remainder, min/max, fused multiply-add
using free_breakfast::math::copysign;
using free_breakfast::math::fabs;
using free_breakfast::math::fma;
using free_breakfast::math::fmax;
using free_breakfast::math::fmin;
using free_breakfast::math::fmod;
using free_breakfast::math::hypot;

// Classification
using free_breakfast::math::isfinite;
using free_breakfast::math::isinf;
using free_breakfast::math::isnan;

// Combinatorics
using free_breakfast::math::binomial;
using free_breakfast::math::double_factorial_odd;
using free_breakfast::math::factorial;

// Integer rounding
using free_breakfast::math::align_up;
using free_breakfast::math::idivup;

} // namespace free_breakfast::math
