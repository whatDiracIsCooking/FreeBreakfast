// The density's two steps (exchange_helpers_bridge.h): max|D|, and the decay
// fit's Λ and H_off. Each is one block striding the whole matrix. One source
// for both backends -- nvcc under CUDA, clang -x hip under HIP.

// block_reduce.cuh first: it orders its standard headers ahead of the vendor
// runtime header, which under HIP must not precede libc++.
#include "block_reduce.cuh"

#include "exchange_helpers.cuh"

#include "exchange_helpers_bridge.h"

namespace {

namespace fb = free_breakfast;
namespace cg = cooperative_groups;

using fb::int32_t;
using fb::int64_t;
using fb::kMaxThreads;
using fb::uint32_t;
using fb::device::exchange::bits_of;
using fb::device::exchange::distance;
using fb::device::exchange::DoubleBits;

// The warps of a kMaxThreads block.
constexpr uint32_t kMaxWarps = kMaxThreads / fb::kWarpSize;

// The larger of two, with NaN winning (`a != a`): a maximum must not drop one.
struct MaxNanOp {
  __device__ double operator()(const double a, const double b) const {
    return (a != a || a > b) ? a : b;
  }
};

// One block strides the whole matrix, each thread folding its elements from 0
// (every |D| is at least that), then the block folds the threads: one pass, one
// plain write, no atomics and no zeroing.
__global__ void __launch_bounds__(kMaxThreads)
    density_max_kernel(const int64_t n, const double *density, double *dmax) {
  __shared__ double partials[kMaxWarps];
  const MaxNanOp op{};
  double v = 0.0;
  for (int64_t e = threadIdx.x; e < n; e += blockDim.x) {
    v = op(v, fb::math::fabs(density[e]));
  }
  v = fb::device::block_reduce<kMaxWarps>(cg::this_thread_block(), partials, v, op);
  if (threadIdx.x == 0) {
    *dmax = v;
  }
}

__device__ double double_of(const DoubleBits b) {
  return __longlong_as_double(static_cast<int64_t>(b));
}

// The decay fit's bins: 1 bohr wide, the last taking everything past it.
constexpr int32_t kDecayBins = 1024;
constexpr double kDecayBinWidth = 1.0;

// The distance between the centres of AOs m and q, from the three segments.
__device__ double ao_distance(const int64_t n_ao, const double *centre, const int64_t m,
                              const int64_t q) {
  return distance(centre[m], centre[n_ao + m], centre[2 * n_ao + m], centre[q], centre[n_ao + q],
                  centre[2 * n_ao + q]);
}

// Λ from the bins' maxima, |D| as the bits of a non-negative double (0 for an
// empty bin): the least-squares slope of log(max) against the bin's centre,
// negated and clamped, two-pass about the means. One thread.
__device__ double decay_slope(const DoubleBits *bins) {
  int32_t n = 0;
  double sx = 0.0;
  double sy = 0.0;
  for (int32_t k = 0; k < kDecayBins; ++k) {
    const double v = double_of(bins[k]);
    if (v > 0.0) {
      ++n;
      sx += (k + 0.5) * kDecayBinWidth;
      sy += fb::math::log(v);
    }
  }
  if (n < 2) {
    return 0.0;
  }
  const double mx = sx / n;
  const double my = sy / n;
  double sxx = 0.0;
  double sxy = 0.0;
  for (int32_t k = 0; k < kDecayBins; ++k) {
    const double v = double_of(bins[k]);
    if (v > 0.0) {
      const double x = (k + 0.5) * kDecayBinWidth - mx;
      sxx += x * x;
      sxy += x * (fb::math::log(v) - my);
    }
  }
  const double slope = -sxy / sxx;
  return slope > 0.0 ? slope : 0.0;
}

// One block strides the whole matrix twice: the off-centre elements' maxima
// into shared bins (DoubleBits), the slope from them, then H_off as density_max_kernel
// folds max|D|. Every step a maximum or one thread's loop, so deterministic.
__global__ void __launch_bounds__(kMaxThreads)
    density_decay_kernel(const int64_t n_ao, const double *density, const double *centre,
                         double *decay) {
  __shared__ DoubleBits bins[kDecayBins];
  __shared__ double partials[kMaxWarps];
  __shared__ double slope;
  for (auto k = static_cast<int32_t>(threadIdx.x); k < kDecayBins;
       k += static_cast<int32_t>(blockDim.x)) {
    bins[k] = 0;
  }
  __syncthreads();
  const int64_t n = n_ao * n_ao;
  for (int64_t e = threadIdx.x; e < n; e += blockDim.x) {
    const double r = ao_distance(n_ao, centre, e / n_ao, e % n_ao);
    if (r >= fb::device::kExchangeSameCentre) {
      const double bin = r / kDecayBinWidth;
      const int32_t k = bin < kDecayBins - 1 ? static_cast<int32_t>(bin) : kDecayBins - 1;
      atomicMax(&bins[k], bits_of(fb::math::fabs(density[e])));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    slope = decay_slope(bins);
  }
  __syncthreads();
  const double lambda = slope;
  const MaxNanOp op{};
  double h = 0.0;
  for (int64_t e = threadIdx.x; e < n; e += blockDim.x) {
    const double r = ao_distance(n_ao, centre, e / n_ao, e % n_ao);
    if (r >= fb::device::kExchangeSameCentre) {
      h = op(h, fb::math::fabs(density[e]) * fb::math::exp(lambda * r));
    }
  }
  h = fb::device::block_reduce<kMaxWarps>(cg::this_thread_block(), partials, h, op);
  if (threadIdx.x == 0) {
    decay[1] = lambda;
    decay[2] = h;
  }
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t exchange_density_max(const wwr::wwrStream_t stream, const int32_t n_ao,
                                     const double *density, double *dmax) {
  if (n_ao < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const int64_t n = static_cast<int64_t>(n_ao) * n_ao;
  density_max_kernel<<<1, kMaxThreads, 0, stream>>>(n, density, dmax);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_density_decay(const wwr::wwrStream_t stream, const int32_t n_ao,
                                       const double *density, const double *centre, double *decay) {
  if (n_ao < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  density_decay_kernel<<<1, kMaxThreads, 0, stream>>>(n_ao, density, centre, decay);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
