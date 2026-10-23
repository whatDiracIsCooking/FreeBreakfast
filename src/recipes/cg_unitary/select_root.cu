// The device half of the line searches' root selection: from the roots
// polynomial_roots returns for Table 1's interpolant and Table 2's Fourier
// polynomial, the ones that are real zero crossings of the geodesic
// derivative. One thread each, on at most kCgMaxSamples roots. One source for
// both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// Both tests take the same shape. A root of the complex polynomial maps to a
// point x on a real line (Re z for Table 1, arg z for Table 2), with an
// off-line part y (Im z, or the departure of |z| from 1). It is a crossing
// when y is within delta = sqrt(eps) * scale of the line and the real
// function changes sign across [x - delta, x + delta]. The first half rejects
// a complex pair; the second rejects what is left of one near the line --
// a double root, where the derivative touches zero without crossing, which
// Aberth returns as a pair split by O(sqrt(eps)). A crossing pair closer than
// delta is rejected with it, which costs nothing: its extremum is that small.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h). <cfloat>
// for FLT/DBL_EPSILON: nvcc rejects std::numeric_limits in device code.
#include <cfloat>

#include "primitives.h"

#include "constants.h"

#include "cg_unitary_bridge.h"

#include "complex.h"

namespace free_breakfast::device {

namespace {

using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

__device__ float real_of(const wwrFloatComplex z) {
  return wwr::wwrCrealf(z);
}
__device__ double real_of(const wwrDoubleComplex z) {
  return wwr::wwrCreal(z);
}
__device__ float imag_of(const wwrFloatComplex z) {
  return wwr::wwrCimagf(z);
}
__device__ double imag_of(const wwrDoubleComplex z) {
  return wwr::wwrCimag(z);
}

template <typename R>
constexpr R kEpsilon = sizeof(R) == sizeof(float) ? R{FLT_EPSILON} : R{DBL_EPSILON};

// Whether f goes from non-positive at a to non-negative at b; an exact zero at
// either end counts.
template <typename R>
__device__ bool rises(const R fa, const R fb) {
  return fa <= R{0} && fb >= R{0};
}

// Whether f changes sign across [a, b], either way.
template <typename R>
__device__ bool straddles(const R fa, const R fb) {
  return rises(fa, fb) || rises(fb, fa);
}

template <typename R>
__device__ R horner(const R *const coeffs, const int32_t order, const R x) {
  R acc = coeffs[order];
  for (int32_t j = order - 1; j >= 0; --j) {
    acc = (acc * x) + coeffs[j];
  }
  return acc;
}

// D(theta) = sum_k c_k e^{ik theta}, coefficient k at index k + (N-1)/2. Real,
// because c_{-k} = conj(c_k), so only the real part of each term is summed.
template <typename R, typename Z>
__device__ R fourier(const Z *const coeffs, const int32_t num_coeffs, const R theta) {
  const int32_t half = (num_coeffs - 1) / 2;
  R acc{0};
  for (int32_t j = 0; j < num_coeffs; ++j) {
    const R angle = static_cast<R>(j - half) * theta;
    acc += (real_of(coeffs[j]) * math::cos(angle)) - (imag_of(coeffs[j]) * math::sin(angle));
  }
  return acc;
}

template <typename R, typename Z>
__global__ void first_real_root_kernel(const R *const coeffs, const int32_t order,
                                       const Z *const roots, const int32_t *const iters,
                                       const R upper, R *const root) {
  const R tolerance = math::sqrt(kEpsilon<R>);
  R best{0};
  if (*iters >= 0) {
    for (int32_t k = 0; k < order; ++k) {
      const R x = real_of(roots[k]);
      const R delta = tolerance * math::fmax(math::fabs(x), upper);
      // !(... <= ...) also drops a NaN estimate.
      if (!(math::fabs(imag_of(roots[k])) <= delta) || !(x > R{0} && x <= upper)) {
        continue;
      }
      if (straddles(horner(coeffs, order, x - delta), horner(coeffs, order, x + delta)) &&
          (best == R{0} || x < best)) {
        best = x;
      }
    }
  }
  *root = best;
}

template <typename R, typename Z>
__global__ void unit_circle_arguments_kernel(const Z *const coeffs, const int32_t num_coeffs,
                                             const Z *const roots, const int32_t *const iters,
                                             const bool maximizing, R *const args,
                                             int32_t *const num_args) {
  const R two_pi = kTwo<R> * kPi<R>;
  const R delta = math::sqrt(kEpsilon<R>) * two_pi;
  // One sample interval: the window's last one is mapped to just before 0.
  const R edge = two_pi / static_cast<R>(num_coeffs);
  int32_t count = 0;
  if (*iters >= 0) {
    for (int32_t k = 0; k < num_coeffs - 1; ++k) {
      const R re = real_of(roots[k]);
      const R im = imag_of(roots[k]);
      // |z| = e^{-Im theta} for z = e^{i theta}: |z| - 1 is theta's off-line
      // part to first order.
      if (!(math::fabs(math::hypot(re, im) - R{1}) <= delta)) {
        continue;
      }
      R theta = math::atan2(im, re);
      if (theta <= -edge) {
        theta += two_pi;
      }
      // The wanted extremum by the direction the derivative crosses in:
      // rising at a minimum, falling at a maximum. Not by a crossing's place
      // in the list, as the paper's odd-numbered rule has it: one crossing
      // missed, or the first one's argument rounded to just below 0 and so
      // wrapped to the far end, and every pick after it is the opposite kind.
      const R before = fourier(coeffs, num_coeffs, theta - delta);
      const R after = fourier(coeffs, num_coeffs, theta + delta);
      if (maximizing ? rises(after, before) : rises(before, after)) {
        args[count] = theta;
        ++count;
      }
    }
  }
  *num_args = count;
}

} // namespace

template <typename R, typename Z>
wwr::wwrError_t first_real_root(const wwr::wwrStream_t stream, const R *const coeffs,
                                const int32_t order, const Z *const roots,
                                const int32_t *const iters, const R upper, R *const root) {
  first_real_root_kernel<R, Z><<<1, 1, 0, stream>>>(coeffs, order, roots, iters, upper, root);
  return wwr::wwrGetLastError();
}

template <typename R, typename Z>
wwr::wwrError_t unit_circle_arguments(const wwr::wwrStream_t stream, const Z *const coeffs,
                                      const int32_t num_coeffs, const Z *const roots,
                                      const int32_t *const iters, const bool maximizing,
                                      R *const args, int32_t *const num_args) {
  unit_circle_arguments_kernel<R, Z>
      <<<1, 1, 0, stream>>>(coeffs, num_coeffs, roots, iters, maximizing, args, num_args);
  return wwr::wwrGetLastError();
}

// One per precision, matching cg_unitary.cpp's.
template wwr::wwrError_t first_real_root<float, wwrFloatComplex>(wwr::wwrStream_t, const float *,
                                                                 int32_t, const wwrFloatComplex *,
                                                                 const int32_t *, float, float *);
template wwr::wwrError_t first_real_root<double, wwrDoubleComplex>(wwr::wwrStream_t, const double *,
                                                                   int32_t,
                                                                   const wwrDoubleComplex *,
                                                                   const int32_t *, double,
                                                                   double *);

template wwr::wwrError_t
unit_circle_arguments<float, wwrFloatComplex>(wwr::wwrStream_t, const wwrFloatComplex *, int32_t,
                                              const wwrFloatComplex *, const int32_t *, bool,
                                              float *, int32_t *);
template wwr::wwrError_t
unit_circle_arguments<double, wwrDoubleComplex>(wwr::wwrStream_t, const wwrDoubleComplex *, int32_t,
                                                const wwrDoubleComplex *, const int32_t *, bool,
                                                double *, int32_t *);

} // namespace free_breakfast::device
