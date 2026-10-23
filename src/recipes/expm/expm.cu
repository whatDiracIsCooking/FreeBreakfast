// The device half of free_breakfast.recipes.expm: the phase_scale kernel, over
// wwr's parallel_for. One source for both backends -- nvcc under CUDA, clang
// -x hip under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "integral_types.h"

#include "expm_bridge.h"

#include "complex.h"
#include "extension/parallel_for/parallel_for.cuh"

namespace free_breakfast::device {

namespace {

using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

// e^{-i x}, overloaded per precision so the functor stays generic. The
// argument is reduced by the math library's own sin/cos, which is what keeps a
// large t * w (many turns of the phase) accurate.
__device__ wwrFloatComplex unit_phase(const float x) {
  float s = 0.0F;
  float c = 0.0F;
  sincosf(x, &s, &c);
  return wwr::make_wwrFloatComplex(c, -s);
}
__device__ wwrDoubleComplex unit_phase(const double x) {
  double s = 0.0;
  double c = 0.0;
  sincos(x, &s, &c);
  return wwr::make_wwrDoubleComplex(c, -s);
}
__device__ wwrFloatComplex multiply(const wwrFloatComplex a, const wwrFloatComplex b) {
  return wwr::wwrCmulf(a, b);
}
__device__ wwrDoubleComplex multiply(const wwrDoubleComplex a, const wwrDoubleComplex b) {
  return wwr::wwrCmul(a, b);
}

// One thread per element. The flat index k walks the matrix column-major, so
// consecutive threads share column j and its phase, and k is 64-bit: n * n
// passes 2^31 at n = 46341. Every member is a const scalar: parallel_for
// passes the functor as a grid constant and its device_functor concept
// requires it not be copy-assignable.
template <typename Z, typename R>
struct PhaseScaleFunctor {
  const Z *const u_;
  const R *const w_;
  Z *const v_;
  const R t_;
  const int64_t n_;

  __device__ void operator()(const int64_t k) const {
    const int64_t j = k / n_;
    v_[k] = multiply(u_[k], unit_phase(t_ * w_[j]));
  }
};

} // namespace

template <typename Z, typename R>
wwr::wwrError_t phase_scale(const wwr::wwrStream_t stream, const int32_t n, const Z *u, const R *w,
                            const R t, Z *v) {
  const PhaseScaleFunctor<Z, R> functor{u, w, v, t, n};
  return wwr::extension::parallel_for(stream, static_cast<int64_t>(n) * n, functor);
}

template wwr::wwrError_t phase_scale<wwrFloatComplex, float>(wwr::wwrStream_t, int32_t,
                                                             const wwrFloatComplex *, const float *,
                                                             float, wwrFloatComplex *);
template wwr::wwrError_t phase_scale<wwrDoubleComplex, double>(wwr::wwrStream_t, int32_t,
                                                               const wwrDoubleComplex *,
                                                               const double *, double,
                                                               wwrDoubleComplex *);

} // namespace free_breakfast::device
