// The classes' geometry steps (exchange_helpers_bridge.h): the largest record
// bound, the AO centres, and each group's centre and reach. One source for
// both backends -- nvcc under CUDA, clang -x hip under HIP.

// block_reduce.cuh first: it orders its standard headers ahead of the vendor
// runtime header, which under HIP must not precede libc++.
#include "block_reduce.cuh"

#include "exchange_helpers.cuh"

#include "exchange_helpers_bridge.h"

namespace {

namespace fb = free_breakfast;
namespace cg = cooperative_groups;

using fb::int32_t;
using fb::device::exchange::bits_of;
using fb::device::exchange::DoubleBits;
using fb::device::exchange::kTableWarps;

// The larger of two.
struct MaxOp {
  __device__ double operator()(const double a, const double b) const { return a > b ? a : b; }
};

// A record vector's three segments, passed by value into a kernel.
struct Xyz {
  const double *x;
  const double *y;
  const double *z;
};

Xyz xyz(const double *const v[3]) {
  return {v[0], v[1], v[2]};
}

// One thread per record folds into its block, and the block's thread 0 into
// g_max as DoubleBits: a maximum, so the result does not depend on the order.
__global__ void __launch_bounds__(fb::kFourWarps)
    bound_max_kernel(const int32_t n_rec, const double *bound, double *g_max) {
  __shared__ double partials[kTableWarps];
  const auto r = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  double v = r < n_rec ? bound[r] : 0.0;
  v = fb::device::block_reduce<kTableWarps>(cg::this_thread_block(), partials, v, MaxOp{});
  if (threadIdx.x == 0) {
    atomicMax(reinterpret_cast<DoubleBits *>(g_max), bits_of(v));
  }
}

// One thread per group: see exchange_ao_centres.
__global__ void ao_centres_kernel(const int32_t n_groups, const int32_t *group_start,
                                  const int32_t nx, const int32_t *off_a, const Xyz p, const Xyz pa,
                                  const int32_t n_ao, double *centre) {
  const auto g = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (g >= n_groups) {
    return;
  }
  const int32_t r = group_start[g];
  const double x = p.x[r] - pa.x[r];
  const double y = p.y[r] - pa.y[r];
  const double z = p.z[r] - pa.z[r];
  for (int32_t i = 0; i < nx; ++i) {
    const int32_t m = off_a[r] + i;
    centre[m] = x;
    centre[n_ao + m] = y;
    centre[2 * n_ao + m] = z;
  }
}

// One block per group: its records' reach folded by the block, and its first
// record's centre. See exchange_group_geometry.
__global__ void __launch_bounds__(fb::kFourWarps)
    group_geometry_kernel(const int32_t n_groups, const int32_t *group_start, const double *bound,
                          const Xyz p, const Xyz pa, const Xyz pb, const double tau,
                          const double *g_max, const double *decay, double *geom) {
  __shared__ double partials[kTableWarps];
  const auto g = static_cast<int32_t>(blockIdx.x);
  const int32_t r0 = group_start[g];
  const int32_t r1 = group_start[g + 1];
  const double scale = *g_max * decay[0];
  const double cut = tau * (1.0 - fb::device::kExchangeDecayMargin);
  double reach = -1.0;
  for (int32_t r = r0 + static_cast<int32_t>(threadIdx.x); r < r1;
       r += static_cast<int32_t>(blockDim.x)) {
    if (bound[r] * scale >= cut) {
      // |AC| = |(P - C) - (P - A)|.
      const double dx = pb.x[r] - pa.x[r];
      const double dy = pb.y[r] - pa.y[r];
      const double dz = pb.z[r] - pa.z[r];
      const double d = fb::math::sqrt(dx * dx + dy * dy + dz * dz);
      reach = d > reach ? d : reach;
    }
  }
  reach = fb::device::block_reduce<kTableWarps>(cg::this_thread_block(), partials, reach, MaxOp{});
  if (threadIdx.x == 0) {
    geom[g] = p.x[r0] - pa.x[r0];
    geom[n_groups + g] = p.y[r0] - pa.y[r0];
    geom[2 * n_groups + g] = p.z[r0] - pa.z[r0];
    geom[3 * n_groups + g] = reach;
  }
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t exchange_bound_max(const wwr::wwrStream_t stream, const int32_t n_rec,
                                   const double *bound, double *g_max) {
  if (n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  bound_max_kernel<<<exchange::element_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, bound, g_max);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_ao_centres(const wwr::wwrStream_t stream, const int32_t n_groups,
                                    const int32_t *group_start, const int32_t nx,
                                    const int32_t *off_a, const double *const p[3],
                                    const double *const pa[3], const int32_t n_ao, double *centre) {
  if (n_groups < 1 || nx < 1 || n_ao < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  ao_centres_kernel<<<exchange::element_blocks(n_groups), kFourWarps, 0, stream>>>(
      n_groups, group_start, nx, off_a, xyz(p), xyz(pa), n_ao, centre);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_group_geometry(const wwr::wwrStream_t stream, const int32_t n_groups,
                                        const int32_t *group_start, const double *bound,
                                        const double *const p[3], const double *const pa[3],
                                        const double *const pb[3], const double tau,
                                        const double *g_max, const double *decay, double *geom) {
  if (n_groups < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  group_geometry_kernel<<<static_cast<uint32_t>(n_groups), kFourWarps, 0, stream>>>(
      n_groups, group_start, bound, xyz(p), xyz(pa), xyz(pb), tau, g_max, decay, geom);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
