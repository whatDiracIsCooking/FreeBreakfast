// The exchange build's hand-written device steps (exchange_helpers_bridge.h):
// max|D|, the Dcd gather, the block table, and the shell starts and mirror,
// the things the generated kb leaves to its launcher. One source for both
// backends -- nvcc under CUDA, clang -x hip under HIP.

// block_reduce.cuh first: it orders its standard headers ahead of the vendor
// runtime header, which under HIP must not precede libc++.
#include "block_reduce.cuh"

#include "angular_momentum.h"
#include "block_params.h"
#include "primitives.h"

#include "exchange_helpers_bridge.h"

namespace {

namespace fb = free_breakfast;
namespace cg = cooperative_groups;

using fb::int32_t;
using fb::int64_t;
using fb::uint32_t;

// The max|D| block: the most threads a block holds.
constexpr uint32_t kMaxWarps = 1024 / fb::kWarpSize;
constexpr uint32_t kMaxThreads = kMaxWarps * fb::kWarpSize;

// The block-table block: kFourWarps threads.
constexpr uint32_t kTableWarps = fb::kFourWarps / fb::kWarpSize;

constexpr int64_t kMaxElements = 2147483647;

// The larger of two, with NaN winning (`a != a`): a maximum must not drop one.
struct MaxNanOp {
  __device__ double operator()(const double a, const double b) const {
    return (a != a || a > b) ? a : b;
  }
};

struct AddOp {
  __device__ int32_t operator()(const int32_t a, const int32_t b) const { return a + b; }
};

int32_t cartesian_count(const int32_t l) {
  return (l + 1) * (l + 2) / 2;
}

bool valid_momentum(const int32_t l) {
  return l >= 0 && l <= fb::kMaxAngularMomentum;
}

// One thread per element, kFourWarps to a block.
uint32_t element_blocks(const int64_t n) {
  return static_cast<uint32_t>(fb::math::idivup(n, static_cast<int64_t>(fb::kFourWarps)));
}

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

// A double's bits, as the vendors' atomicMax takes them: a non-negative double
// orders as its bits, so a maximum of |x| or of a bound is an integer one.
// int-type-ok: atomicMax and __double_as_longlong are declared on this type
using DoubleBits = unsigned long long;

__device__ DoubleBits bits_of(const double x) {
  return static_cast<DoubleBits>(__double_as_longlong(x));
}

__device__ double double_of(const DoubleBits b) {
  return __longlong_as_double(static_cast<int64_t>(b));
}

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

// The decay fit's bins: 1 bohr wide, the last taking everything past it.
constexpr int32_t kDecayBins = 1024;
constexpr double kDecayBinWidth = 1.0;

__device__ double distance(const double ax, const double ay, const double az, const double bx,
                           const double by, const double bz) {
  const double dx = ax - bx;
  const double dy = ay - by;
  const double dz = az - bz;
  return fb::math::sqrt(dx * dx + dy * dy + dz * dz);
}

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

// One thread per element of Dcd, in its row-major order, so the writes
// coalesce.
__global__ void gather_dcd_kernel(const int64_t n, const int32_t nc, const int32_t nd,
                                  const int32_t n_d, const int64_t n_ao, const double *density,
                                  const int32_t *off_c, const int32_t *off_d, double *dcd) {
  const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= n) {
    return;
  }
  const int64_t width = static_cast<int64_t>(nc) * nd;
  const int64_t row = e / width;
  const auto col = static_cast<int32_t>(e % width);
  const auto rank_c = static_cast<int32_t>(row / n_d);
  const auto rank_d = static_cast<int32_t>(row % n_d);
  const int64_t m = off_c[rank_c] + col / nd;
  const int64_t q = off_d[rank_d] + col % nd;
  dcd[e] = density[m * n_ao + q];
}

// The first ket group whose first AO is at least off: with the symmetric
// sweep, a bra group at off takes the ket groups from it on. The groups are in
// basis order, so their first AOs rise and a binary search finds it.
__device__ int32_t first_ket_group(const int32_t n_b, const int32_t *b_start, const int32_t *b_off,
                                   const int32_t off) {
  int32_t lo = 0;
  int32_t hi = n_b;
  while (lo < hi) {
    const int32_t mid = lo + (hi - lo) / 2;
    if (b_off[b_start[mid]] < off) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// F_ab of bra group ga against ket group gb: see exchange_blocks.
__device__ double block_factor(const int32_t n_a, const double *a_geom, const int32_t ga,
                               const int32_t n_b, const double *b_geom, const int32_t gb,
                               const double *decay) {
  const double dm = decay[0];
  const double ra = a_geom[3 * n_a + ga];
  const double rb = b_geom[3 * n_b + gb];
  if (ra < 0.0 || rb < 0.0) {
    return dm;
  }
  const double d = distance(a_geom[ga], a_geom[n_a + ga], a_geom[2 * n_a + ga], b_geom[gb],
                            b_geom[n_b + gb], b_geom[2 * n_b + gb]) -
                   ra - rb;
  if (!(d > 2.0 * fb::device::kExchangeSameCentre)) {
    return dm;
  }
  return fb::math::fmin(dm, decay[2] * fb::math::exp(-decay[1] * d) *
                                (1.0 + fb::device::kExchangeDecayMargin));
}

// One block per bra group ga. Its blocks start after every earlier bra group's,
// n_b - first(g) each, which the block's threads sum, and its threads then
// write its own, one per ket group from first(ga) on, and each one's factor.
__global__ void __launch_bounds__(fb::kFourWarps)
    blocks_kernel(const int32_t n_a, const int32_t *a_start, const int32_t *a_off,
                  const double *a_geom, const int32_t n_b, const int32_t *b_start,
                  const int32_t *b_off, const double *b_geom, const double *decay,
                  const bool symmetric, const int32_t n_blk, int32_t *blocks, double *factor) {
  __shared__ int32_t partials[kTableWarps];
  const auto ga = static_cast<int32_t>(blockIdx.x);
  const auto first = [&](const int32_t g) {
    return symmetric ? first_ket_group(n_b, b_start, b_off, a_off[a_start[g]]) : 0;
  };
  int32_t before = 0;
  for (auto g = static_cast<int32_t>(threadIdx.x); g < ga; g += static_cast<int32_t>(blockDim.x)) {
    before += n_b - first(g);
  }
  const int32_t base =
      fb::device::block_reduce<kTableWarps>(cg::this_thread_block(), partials, before, AddOp{});
  const int32_t f = first(ga);
  const int32_t bra0 = a_start[ga];
  for (int32_t gb = f + static_cast<int32_t>(threadIdx.x); gb < n_b;
       gb += static_cast<int32_t>(blockDim.x)) {
    const int32_t blk = base + gb - f;
    const int32_t ket0 = b_start[gb];
    blocks[blk] = bra0;
    blocks[n_blk + blk] = a_start[ga + 1];
    blocks[2 * n_blk + blk] = ket0;
    blocks[3 * n_blk + blk] = b_start[gb + 1];
    blocks[4 * n_blk + blk] = a_off[bra0];
    blocks[5 * n_blk + blk] = b_off[ket0];
    factor[blk] = block_factor(n_a, a_geom, ga, n_b, b_geom, gb, decay);
  }
}

// One thread per AO of the momentum's shells, by rank.
__global__ void shell_starts_kernel(const int32_t n, const int32_t nx, const int32_t *off,
                                    int32_t *start) {
  const auto e = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (e < n) {
    const int32_t o = off[e / nx];
    start[o + e % nx] = o;
  }
}

// One thread per element of K; those of a lower shell block read their
// transpose, which is in an upper one, so no element is both read and written.
__global__ void mirror_kernel(const int64_t n, const int64_t n_ao, const int32_t *start,
                              double *k) {
  const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= n) {
    return;
  }
  const int64_t m = e / n_ao;
  const int64_t q = e % n_ao;
  if (q < start[m]) {
    k[e] = k[q * n_ao + m];
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

wwr::wwrError_t exchange_gather_dcd(const wwr::wwrStream_t stream, const int32_t lc,
                                    const int32_t ld, const int32_t n_ao, const double *density,
                                    const int32_t n_c, const int32_t *off_c, const int32_t n_d,
                                    const int32_t *off_d, double *dcd) {
  if (!valid_momentum(lc) || !valid_momentum(ld) || n_ao < 1 || n_c < 1 || n_d < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const int32_t nc = cartesian_count(lc);
  const int32_t nd = cartesian_count(ld);
  const int64_t n = static_cast<int64_t>(n_c) * n_d * nc * nd;
  if (n > kMaxElements) {
    return wwr::wwrErrorInvalidValue;
  }
  gather_dcd_kernel<<<element_blocks(n), kFourWarps, 0, stream>>>(n, nc, nd, n_d, n_ao, density,
                                                                  off_c, off_d, dcd);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_bound_max(const wwr::wwrStream_t stream, const int32_t n_rec,
                                   const double *bound, double *g_max) {
  if (n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  bound_max_kernel<<<element_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, bound, g_max);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_ao_centres(const wwr::wwrStream_t stream, const int32_t n_groups,
                                    const int32_t *group_start, const int32_t nx,
                                    const int32_t *off_a, const double *const p[3],
                                    const double *const pa[3], const int32_t n_ao, double *centre) {
  if (n_groups < 1 || nx < 1 || n_ao < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  ao_centres_kernel<<<element_blocks(n_groups), kFourWarps, 0, stream>>>(
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

wwr::wwrError_t exchange_density_decay(const wwr::wwrStream_t stream, const int32_t n_ao,
                                       const double *density, const double *centre, double *decay) {
  if (n_ao < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  density_decay_kernel<<<1, kMaxThreads, 0, stream>>>(n_ao, density, centre, decay);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_blocks(const wwr::wwrStream_t stream, const int32_t n_a,
                                const int32_t *a_start, const int32_t *a_off, const double *a_geom,
                                const int32_t n_b, const int32_t *b_start, const int32_t *b_off,
                                const double *b_geom, const double *decay, const bool symmetric,
                                const int32_t n_blk, int32_t *blocks, double *factor) {
  if (n_a < 1 || n_b < 1 || n_blk < 1 ||
      static_cast<int64_t>(n_blk) * kExchangeBlockFields > kMaxElements) {
    return wwr::wwrErrorInvalidValue;
  }
  blocks_kernel<<<static_cast<uint32_t>(n_a), kFourWarps, 0, stream>>>(
      n_a, a_start, a_off, a_geom, n_b, b_start, b_off, b_geom, decay, symmetric, n_blk, blocks,
      factor);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_shell_starts(const wwr::wwrStream_t stream, const int32_t l,
                                      const int32_t n_s, const int32_t *off, int32_t *start) {
  if (!valid_momentum(l) || n_s < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const int32_t nx = cartesian_count(l);
  const int64_t n = static_cast<int64_t>(n_s) * nx;
  if (n > kMaxElements) {
    return wwr::wwrErrorInvalidValue;
  }
  shell_starts_kernel<<<element_blocks(n), kFourWarps, 0, stream>>>(static_cast<int32_t>(n), nx,
                                                                    off, start);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_mirror(const wwr::wwrStream_t stream, const int32_t n_ao,
                                const int32_t *start, double *k) {
  if (n_ao < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const int64_t n = static_cast<int64_t>(n_ao) * n_ao;
  mirror_kernel<<<element_blocks(n), kFourWarps, 0, stream>>>(n, n_ao, start, k);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
