// The device half of free_breakfast.recipes.complex_cast: the split_planes and
// merge_planes kernels, over wwr's parallel_for. One source for both backends
// -- nvcc under CUDA, clang -x hip under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "integral_types.h"

#include "complex_cast_bridge.h"

#include "complex.h"
#include "extension/parallel_for/parallel_for.cuh"

namespace free_breakfast::device {

namespace {

using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

// Components through complex.h's accessors, never .x/.y: hipFloatComplex is a
// class, not cuFloatComplex's float2, and the accessors are the one spelling
// both backends share. Overloaded per precision so the functors stay generic.
__device__ float real_part(const wwrFloatComplex z) {
  return wwr::wwrCrealf(z);
}
__device__ float imag_part(const wwrFloatComplex z) {
  return wwr::wwrCimagf(z);
}
__device__ double real_part(const wwrDoubleComplex z) {
  return wwr::wwrCreal(z);
}
__device__ double imag_part(const wwrDoubleComplex z) {
  return wwr::wwrCimag(z);
}
__device__ wwrFloatComplex make_complex(const float re, const float im) {
  return wwr::make_wwrFloatComplex(re, im);
}
__device__ wwrDoubleComplex make_complex(const double re, const double im) {
  return wwr::make_wwrDoubleComplex(re, im);
}

// One thread per element of the block. The flat index k walks it column-major,
// so consecutive threads touch consecutive rows of one column, and each side
// applies its own leading dimension. k is 64-bit: rows * cols passes 2^31 at
// n = 46341, where either 32-bit dimension alone is still fine.
//
// Every member is a const scalar: parallel_for passes the functor as a grid
// constant and its device_functor concept requires it not be copy-assignable.
template <typename Z, typename R>
struct SplitPlanesFunctor {
  const Z *const a_;
  R *const re_;
  R *const im_;
  const int64_t rows_;
  const int64_t lda_;
  const int64_t ldp_;

  __device__ void operator()(const int64_t k) const {
    const int64_t i = k % rows_;
    const int64_t j = k / rows_;
    const Z z = a_[i + (j * lda_)];
    re_[i + (j * ldp_)] = real_part(z);
    im_[i + (j * ldp_)] = imag_part(z);
  }
};

template <typename Z, typename R>
struct MergePlanesFunctor {
  const R *const re_;
  const R *const im_;
  Z *const c_;
  const int64_t rows_;
  const int64_t ldp_;
  const int64_t ldc_;

  __device__ void operator()(const int64_t k) const {
    const int64_t i = k % rows_;
    const int64_t j = k / rows_;
    c_[i + (j * ldc_)] = make_complex(re_[i + (j * ldp_)], im_[i + (j * ldp_)]);
  }
};

} // namespace

template <typename Z, typename R>
int32_t split_planes(const wwr::wwrStream_t stream, const int32_t rows, const int32_t cols,
                     const Z *a, const int32_t lda, R *re, R *im, const int32_t ldp) {
  const SplitPlanesFunctor<Z, R> functor{a, re, im, rows, lda, ldp};
  return static_cast<int32_t>(
      wwr::extension::parallel_for(stream, static_cast<int64_t>(rows) * cols, functor));
}

template <typename Z, typename R>
int32_t merge_planes(const wwr::wwrStream_t stream, const int32_t rows, const int32_t cols,
                     const R *re, const R *im, const int32_t ldp, Z *c, const int32_t ldc) {
  const MergePlanesFunctor<Z, R> functor{re, im, c, rows, ldp, ldc};
  return static_cast<int32_t>(
      wwr::extension::parallel_for(stream, static_cast<int64_t>(rows) * cols, functor));
}

template int32_t split_planes<wwrFloatComplex, float>(wwr::wwrStream_t, int32_t, int32_t,
                                                      const wwrFloatComplex *, int32_t, float *,
                                                      float *, int32_t);
template int32_t split_planes<wwrDoubleComplex, double>(wwr::wwrStream_t, int32_t, int32_t,
                                                        const wwrDoubleComplex *, int32_t, double *,
                                                        double *, int32_t);
template int32_t merge_planes<wwrFloatComplex, float>(wwr::wwrStream_t, int32_t, int32_t,
                                                      const float *, const float *, int32_t,
                                                      wwrFloatComplex *, int32_t);
template int32_t merge_planes<wwrDoubleComplex, double>(wwr::wwrStream_t, int32_t, int32_t,
                                                        const double *, const double *, int32_t,
                                                        wwrDoubleComplex *, int32_t);

} // namespace free_breakfast::device
