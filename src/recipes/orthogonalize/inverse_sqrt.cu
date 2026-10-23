// The device half of free_breakfast.recipes.orthogonalize: the inverse_sqrt
// kernel, over wwr's parallel_for. One source for both backends -- nvcc under
// CUDA, clang -x hip under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "primitives.h"

#include "inverse_sqrt_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"

namespace free_breakfast::device {

namespace {

// data[i] <- data[i] <= floor ? 0 : 1 / sqrt(data[i]).
//
// Both members are const scalars: parallel_for passes the functor as a grid
// constant and its device_functor concept requires it not be copy-assignable.
// A const member of class type would instead make it non-trivially-copyable
// under clang and break only the HIP build.
template <typename T>
struct InverseSqrtFunctor {
  T *const data_;
  const T floor_;

  __device__ void operator()(const int32_t i) const {
    const T value = data_[i];
    // The floor test comes first, so a zero or negative eigenvalue never
    // reaches the sqrt. 1 / sqrt rather than math::rsqrt: the device rsqrt
    // intrinsic is not correctly rounded, and this runs once per setup.
    data_[i] = value <= floor_ ? T{0} : T{1} / math::sqrt(value);
  }
};

} // namespace

template <typename T>
wwr::wwrError_t inverse_sqrt(const wwr::wwrStream_t stream, const int32_t count, T *data,
                             const T floor) {
  const InverseSqrtFunctor<T> functor{data, floor};
  return wwr::extension::parallel_for(stream, count, functor);
}

template wwr::wwrError_t inverse_sqrt<float>(wwr::wwrStream_t, int32_t, float *, float);
template wwr::wwrError_t inverse_sqrt<double>(wwr::wwrStream_t, int32_t, double *, double);

} // namespace free_breakfast::device
