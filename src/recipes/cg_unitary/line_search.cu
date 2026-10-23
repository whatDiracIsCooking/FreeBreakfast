// The device half of the line searches' scalar stages around the root solve:
// Table 1's interpolation, Table 2's windowed DFT and its step selection. Each
// is a single-block launch on at most kCgMaxSamples device values. One source
// for both backends -- nvcc under CUDA, clang -x hip under HIP.

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

// The derivative and the cost are both real parts of Frobenius inner products,
// so a complex sample contributes its real part only.
__device__ float real_part(const float x) {
  return x;
}
__device__ double real_part(const double x) {
  return x;
}
__device__ float real_part(const wwrFloatComplex x) {
  return wwr::wwrCrealf(x);
}
__device__ double real_part(const wwrDoubleComplex x) {
  return wwr::wwrCreal(x);
}

__device__ wwrFloatComplex make_complex(const float re, const float im) {
  return wwr::make_wwrFloatComplex(re, im);
}
__device__ wwrDoubleComplex make_complex(const double re, const double im) {
  return wwr::make_wwrDoubleComplex(re, im);
}

template <typename R>
constexpr R kEpsilon = sizeof(R) == sizeof(float) ? R{FLT_EPSILON} : R{DBL_EPSILON};

// Table 1's largest order, which fixes the elimination's array bounds.
constexpr int32_t kMaxOrder = 5;

// Solve M b = rhs for M[i-1][j-1] = i^j, i, j = 1..order, rhs[i-1] = Jp(i) -
// Jp(0), by Gaussian elimination with partial pivoting: M is integer and well
// conditioned but not diagonally dominant. One thread; the system is at most
// 5 x 5.
template <typename T, typename R>
__global__ void table1_coefficients_kernel(const T *const dots, const int32_t order,
                                           const R sample_scale, R *const coeffs) {
  R jp[kMaxOrder + 1];
  for (int32_t i = 0; i <= order; ++i) {
    jp[i] = sample_scale * real_part(dots[i]);
  }
  coeffs[0] = jp[0];

  R m[kMaxOrder][kMaxOrder + 1];
  for (int32_t i = 1; i <= order; ++i) {
    R p{1};
    for (int32_t j = 1; j <= order; ++j) {
      p *= static_cast<R>(i);
      m[i - 1][j - 1] = p;
    }
    m[i - 1][order] = jp[i] - jp[0];
  }

  for (int32_t col = 0; col < order; ++col) {
    int32_t pivot = col;
    R best = math::fabs(m[col][col]);
    for (int32_t r = col + 1; r < order; ++r) {
      const R candidate = math::fabs(m[r][col]);
      if (candidate > best) {
        best = candidate;
        pivot = r;
      }
    }
    // !(best > 0) also catches the NaN a NaN sample spreads through the rhs:
    // a flat derivative, so no root and no step.
    if (!(best > R{0})) {
      for (int32_t j = 1; j <= order; ++j) {
        coeffs[j] = R{0};
      }
      return;
    }
    if (pivot != col) {
      for (int32_t j = col; j <= order; ++j) {
        const R t = m[col][j];
        m[col][j] = m[pivot][j];
        m[pivot][j] = t;
      }
    }
    for (int32_t r = col + 1; r < order; ++r) {
      const R factor = m[r][col] / m[col][col];
      for (int32_t j = col; j <= order; ++j) {
        m[r][j] -= factor * m[col][j];
      }
    }
  }

  for (int32_t i = order - 1; i >= 0; --i) {
    R acc = m[i][order];
    for (int32_t j = i + 1; j < order; ++j) {
      acc -= m[i][j] * coeffs[j + 1];
    }
    coeffs[i + 1] = acc / m[i][i];
  }
}

// One thread per coefficient, each an O(N) sum, N <= kCgMaxSamples.
template <typename T, typename R, typename Z>
__global__ void hann_dft_kernel(const T *const dots, const int32_t num_samples,
                                const R sample_scale, Z *const coeffs) {
  __shared__ R windowed[kCgMaxSamples];

  const auto t = static_cast<int32_t>(threadIdx.x);
  // The window 0.5 - 0.5 cos(2 pi (i + 1) / (N + 1)) is strictly positive over
  // i = 0..N-1, so it moves no zero of the derivative.
  if (t < num_samples) {
    const R window = R{0.5} - (R{0.5} * math::cos(kTwo<R> * kPi<R> * static_cast<R>(t + 1) /
                                                  static_cast<R>(num_samples + 1)));
    windowed[t] = window * sample_scale * real_part(dots[t]);
  }
  __syncthreads();

  if (t < num_samples) {
    const int32_t k = t - ((num_samples - 1) / 2);
    R re{0};
    R im{0};
    for (int32_t i = 0; i < num_samples; ++i) {
      const R angle =
          -kTwo<R> * kPi<R> * static_cast<R>(i) * static_cast<R>(k) / static_cast<R>(num_samples);
      re += windowed[i] * math::cos(angle);
      im += windowed[i] * math::sin(angle);
    }
    coeffs[t] = make_complex(re, im);
  }
}

// One thread.
template <typename T, typename R>
__global__ void select_dft_step_kernel(const R *const args, const int32_t *const num_args,
                                       const T *const cost_dots, const int32_t num_samples,
                                       const R t_dft, const bool maximizing, R *const mu) {
  const int32_t count = *num_args;

  // Step 13: the sample that best improves the cost. Anchoring on a measured
  // value rather than on the parity of a crossing alone keeps a saddle inside
  // the window from mislabelling every candidate. An improvement counts only
  // above sqrt(eps) of the costs' scale: near the optimum J varies across the
  // window by less than its own rounding, and the best sample is then noise
  // that can point at a crossing many periods away. Below the floor the
  // anchor stays at 0, which takes the first crossing.
  R scale{0};
  for (int32_t i = 0; i < num_samples; ++i) {
    scale = math::fmax(scale, math::fabs(real_part(cost_dots[i])));
  }
  const R floor = math::sqrt(kEpsilon<R>) * scale;
  int32_t best_i = 0;
  R best = real_part(cost_dots[0]);
  for (int32_t i = 1; i < num_samples; ++i) {
    const R c = real_part(cost_dots[i]);
    if (maximizing ? (c > best + floor) : (c < best - floor)) {
      best = c;
      best_i = i;
    }
  }
  // best_i == 0 is not a failure: the window is coarse by design (section
  // 3.2), so near a sampled optimum every other sample reads worse while a
  // shorter step still improves. mu_star = 0 then picks the first extremum,
  // Table 1's answer.
  const R mu_star = static_cast<R>(best_i) * t_dft / static_cast<R>(num_samples);

  // Step 12: every candidate is already the wanted kind of extremum.
  R chosen{0};
  R closest{-1};
  for (int32_t l = 0; l < count; ++l) {
    const R candidate = args[l] * t_dft / (kTwo<R> * kPi<R>);
    const R distance = math::fabs(candidate - mu_star);
    if (closest < R{0} || distance < closest) {
      closest = distance;
      chosen = candidate;
    }
  }
  *mu = chosen;
}

} // namespace

template <typename T, typename R>
wwr::wwrError_t table1_coefficients(const wwr::wwrStream_t stream, const T *const dots,
                                    const int32_t order, const R sample_scale, R *const coeffs) {
  table1_coefficients_kernel<T, R><<<1, 1, 0, stream>>>(dots, order, sample_scale, coeffs);
  return wwr::wwrGetLastError();
}

template <typename T, typename R, typename Z>
wwr::wwrError_t hann_dft(const wwr::wwrStream_t stream, const T *const dots,
                         const int32_t num_samples, const R sample_scale, Z *const coeffs) {
  hann_dft_kernel<T, R, Z>
      <<<1, kCgMaxSamples, 0, stream>>>(dots, num_samples, sample_scale, coeffs);
  return wwr::wwrGetLastError();
}

template <typename T, typename R>
wwr::wwrError_t select_dft_step(const wwr::wwrStream_t stream, const R *const args,
                                const int32_t *const num_args, const T *const cost_dots,
                                const int32_t num_samples, const R t_dft, const bool maximizing,
                                R *const mu) {
  select_dft_step_kernel<T, R>
      <<<1, 1, 0, stream>>>(args, num_args, cost_dots, num_samples, t_dft, maximizing, mu);
  return wwr::wwrGetLastError();
}

// One per element type, matching cg_unitary.cpp's.
template wwr::wwrError_t table1_coefficients<float, float>(wwr::wwrStream_t, const float *, int32_t,
                                                           float, float *);
template wwr::wwrError_t table1_coefficients<double, double>(wwr::wwrStream_t, const double *,
                                                             int32_t, double, double *);
template wwr::wwrError_t table1_coefficients<wwrFloatComplex, float>(wwr::wwrStream_t,
                                                                     const wwrFloatComplex *,
                                                                     int32_t, float, float *);
template wwr::wwrError_t table1_coefficients<wwrDoubleComplex, double>(wwr::wwrStream_t,
                                                                       const wwrDoubleComplex *,
                                                                       int32_t, double, double *);

template wwr::wwrError_t hann_dft<float, float, wwrFloatComplex>(wwr::wwrStream_t, const float *,
                                                                 int32_t, float, wwrFloatComplex *);
template wwr::wwrError_t hann_dft<double, double, wwrDoubleComplex>(wwr::wwrStream_t,
                                                                    const double *, int32_t, double,
                                                                    wwrDoubleComplex *);
template wwr::wwrError_t hann_dft<wwrFloatComplex, float, wwrFloatComplex>(wwr::wwrStream_t,
                                                                           const wwrFloatComplex *,
                                                                           int32_t, float,
                                                                           wwrFloatComplex *);
template wwr::wwrError_t
hann_dft<wwrDoubleComplex, double, wwrDoubleComplex>(wwr::wwrStream_t, const wwrDoubleComplex *,
                                                     int32_t, double, wwrDoubleComplex *);

template wwr::wwrError_t select_dft_step<float, float>(wwr::wwrStream_t, const float *,
                                                       const int32_t *, const float *, int32_t,
                                                       float, bool, float *);
template wwr::wwrError_t select_dft_step<double, double>(wwr::wwrStream_t, const double *,
                                                         const int32_t *, const double *, int32_t,
                                                         double, bool, double *);
template wwr::wwrError_t select_dft_step<wwrFloatComplex, float>(wwr::wwrStream_t, const float *,
                                                                 const int32_t *,
                                                                 const wwrFloatComplex *, int32_t,
                                                                 float, bool, float *);
template wwr::wwrError_t select_dft_step<wwrDoubleComplex, double>(wwr::wwrStream_t, const double *,
                                                                   const int32_t *,
                                                                   const wwrDoubleComplex *,
                                                                   int32_t, double, bool, double *);

} // namespace free_breakfast::device
