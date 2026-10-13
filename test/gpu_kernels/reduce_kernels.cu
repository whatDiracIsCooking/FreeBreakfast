// Device side of test/gtest/reduce_gpu.cpp: a kernel that runs block_reduce
// (warp_reduce underneath), and the host launchers reduce_kernels.h declares.
// This is the first .cu in the tree to include src/core/kernel/block_reduce.cuh,
// so it is also what proves the kernel headers compile -- under nvcc in the
// CUDA tier, and under clang -x hip in devtools/cross-backend-check.sh.

// block_reduce.cuh first: it orders its standard headers ahead of the vendor
// runtime header, which under HIP must not precede libc++.
#include "block_reduce.cuh"

#include "device_array.cuh"
#include "reduce_kernels.h"

namespace free_breakfast::test_kernels {

namespace {

namespace cg = cooperative_groups;

using device::DeviceFunctor;

struct AddOp {
  template <typename R>
  __device__ R operator()(const R a, const R b) const {
    return a + b;
  }
};

// The larger of two, with NaN winning (`a != a`): a maximum must not drop one.
struct MaxNanOp {
  template <typename R>
  __device__ R operator()(const R a, const R b) const {
    return (a != a || a > b) ? a : b;
  }
};

// Associative, not commutative: a reduction that reorders operands, or folds in
// an inactive thread, returns the wrong input.
struct FirstOp {
  template <typename R>
  __device__ R operator()(const R a, const R /*b*/) const {
    return a;
  }
};

struct LastOp {
  template <typename R>
  __device__ R operator()(const R /*a*/, const R b) const {
    return b;
  }
};

static_assert(DeviceFunctor<AddOp> && DeviceFunctor<MaxNanOp> && DeviceFunctor<FirstOp> &&
              DeviceFunctor<LastOp>);
static_assert(DeviceFunctor<device::IdentityFunctor<double>>);
static_assert(!DeviceFunctor<int32_t>);

constexpr uint32_t kMaxWarps = 1024 / kWarpSize;

template <uint32_t kNumWarps, typename T, typename Op>
__global__ void block_reduce_kernel(const T *in, const uint32_t nactive, T *out, const Op op) {
  __shared__ T partials[kNumWarps];
  const uint32_t t = threadIdx.x;
  // IdentityFunctor is the seed of a transform-reduce; here it only has to
  // compile and pass the value through.
  const T v = device::IdentityFunctor<T>{}(in[t]);
  out[t] = device::block_reduce<kNumWarps>(cg::this_thread_block(), partials, v, op, nactive);
}

// Calls f(op) with the functor for @p op.
template <typename F>
bool with_op(const FoldOp op, F &&f) {
  switch (op) {
  case FoldOp::Add:
    return f(AddOp{});
  case FoldOp::MaxNan:
    return f(MaxNanOp{});
  case FoldOp::First:
    return f(FirstOp{});
  case FoldOp::Last:
    return f(LastOp{});
  }
  return false;
}

template <uint32_t kNumWarps, typename T, typename Op>
void launch_block_reduce(const T *in, const uint32_t nactive, T *out, const Op op) {
  block_reduce_kernel<kNumWarps><<<1, kNumWarps * kWarpSize>>>(in, nactive, out, op);
}

template <typename T>
bool block_reduce_launch(const FoldOp op, const uint32_t num_warps, const T *in,
                         const uint32_t nactive, T *out) {
  if (num_warps != 1 && num_warps != 2 && num_warps != 4 && num_warps != kMaxWarps) {
    return false;
  }
  const size_t threads = static_cast<size_t>(num_warps) * kWarpSize;
  const DeviceArray<T> din(threads);
  const DeviceArray<T> dout(threads);
  if (!din.ok() || !dout.ok() || !din.upload(in)) {
    return false;
  }
  return with_op(op,
                 [&](const auto f) {
                   switch (num_warps) {
                   case 1:
                     launch_block_reduce<1>(din.data(), nactive, dout.data(), f);
                     break;
                   case 2:
                     launch_block_reduce<2>(din.data(), nactive, dout.data(), f);
                     break;
                   case 4:
                     launch_block_reduce<4>(din.data(), nactive, dout.data(), f);
                     break;
                   default:
                     launch_block_reduce<kMaxWarps>(din.data(), nactive, dout.data(), f);
                     break;
                   }
                   return launched_cleanly();
                 }) &&
         dout.download(out);
}

} // namespace

bool block_reduce_on_device(const FoldOp op, const uint32_t num_warps, const double *in,
                            const uint32_t nactive, double *out) {
  return block_reduce_launch(op, num_warps, in, nactive, out);
}

bool block_reduce_on_device(const FoldOp op, const uint32_t num_warps, const int32_t *in,
                            const uint32_t nactive, int32_t *out) {
  return block_reduce_launch(op, num_warps, in, nactive, out);
}

} // namespace free_breakfast::test_kernels
