/// @file
/// @brief The line searches' device launchers, declared once for the module
/// that calls them and the device TUs that define them.
///
/// Included by cg_unitary.cpp in its GLOBAL MODULE FRAGMENT, and by
/// line_search.cu and select_root.cu directly: a declaration
/// in a module's purview gets module linkage and can never bind to a
/// definition compiled in a plain TU, which a .cu is (see
/// src/recipes/complex_cast/complex_cast_bridge.h).
///
/// Every launcher is generic in the sample type T, its real type R and the
/// complex type Z of the same precision, so this header names no wwr complex
/// type; the .cu files instantiate the four (or two) combinations. Each is a
/// single-block launch on a few dozen device values: the point is not the
/// arithmetic, which is trivial, but keeping the samples off the host.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

// kCgMaxSamples, which the kernels size their arrays by.
#include "cg_unitary_limits.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see src/integrals/one_electron/overlap/overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief Table 1 step 7: the degree-@p order polynomial in s = mu / h through
/// the derivative samples at s = 0, 1, ..., @p order.
///
/// Sample i is @p sample_scale * Re dots[i], the derivative at mu = i h. The
/// variable is s, not mu, so the interpolation matrix is the integer
/// Vandermonde M[i][j] = i^j whatever h is, and the roots come out in units of
/// the sample spacing. @p coeffs receives order + 1 real coefficients,
/// ascending; a singular system (it is not, for distinct integer nodes, but a
/// NaN sample makes it so) leaves every coefficient but the constant zero.
/// 1 <= @p order <= 5.
///
/// @return The launch status.
template <typename T, typename R>
[[nodiscard]] wwr::wwrError_t table1_coefficients(wwr::wwrStream_t stream, const T *dots,
                                                  int32_t order, R sample_scale, R *coeffs);

/// @brief Table 2 steps 8-10: the Hann-windowed, centred DFT of the
/// @p num_samples derivative samples.
///
/// Coefficient c_k, k = -(N-1)/2 .. (N-1)/2, lands at index k + (N-1)/2, which
/// makes @p coeffs directly the ascending coefficients of the degree N-1
/// polynomial z^((N-1)/2) D(z), D(theta) = sum_k c_k e^{ik theta} being the
/// reconstructed derivative. @p num_samples is odd and at most kCgMaxSamples.
///
/// @return The launch status.
template <typename T, typename R, typename Z>
[[nodiscard]] wwr::wwrError_t hann_dft(wwr::wwrStream_t stream, const T *dots, int32_t num_samples,
                                       R sample_scale, Z *coeffs);

/// @brief Table 2 steps 12-13: choose among unit_circle_arguments' wanted
/// extrema in @p args (count in @p num_args) by the sampled cost
/// Re cost_dots[i].
///
/// The one nearest the best sample is taken; with no sample better than the
/// first by more than rounding, the one nearest 0. @p mu receives the step,
/// at most one sample interval below 0 for an argument mapped there, or 0
/// when there is no extremum.
///
/// @return The launch status.
template <typename T, typename R>
[[nodiscard]] wwr::wwrError_t select_dft_step(wwr::wwrStream_t stream, const R *args,
                                              const int32_t *num_args, const T *cost_dots,
                                              int32_t num_samples, R t_dft, bool maximizing, R *mu);

/// @brief Table 1 step 8: the smallest real root in (0, @p upper] of the real
/// degree-@p order polynomial @p coeffs, from polynomial_roots' @p roots and
/// @p iters.
///
/// A root is taken only when it is real to within a relative sqrt(eps) and the
/// polynomial changes sign across it: that rejects a complex pair, and a
/// double root, at which the derivative touches zero without crossing.
/// @p root receives it, or 0 when there is none or the roots did not settle.
///
/// @return The launch status.
template <typename R, typename Z>
[[nodiscard]] wwr::wwrError_t first_real_root(wwr::wwrStream_t stream, const R *coeffs,
                                              int32_t order, const Z *roots, const int32_t *iters,
                                              R upper, R *root);

/// @brief Table 2 step 11: the arguments of the roots on the unit circle of
/// hann_dft's degree-(@p num_coeffs - 1) polynomial at which J has the wanted
/// extremum, from polynomial_roots' @p roots and @p iters.
///
/// The test mirrors first_real_root's on theta: near the circle to within
/// sqrt(eps), and a sign change of the real D(theta) across it, in the wanted
/// direction: rising to minimize, falling (@p maximizing) to maximize. The
/// arguments are in (-2 pi / N, 2 pi - 2 pi / N]: the window is periodic, so
/// its last sample interval is the one just before 0. @p args receives them,
/// in no order, and @p num_args their count, 0 when the roots did not settle.
///
/// @return The launch status.
template <typename R, typename Z>
[[nodiscard]] wwr::wwrError_t
unit_circle_arguments(wwr::wwrStream_t stream, const Z *coeffs, int32_t num_coeffs, const Z *roots,
                      const int32_t *iters, bool maximizing, R *args, int32_t *num_args);

} // namespace free_breakfast::device
