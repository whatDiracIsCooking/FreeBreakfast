// The device half of
// free_breakfast.integrals.eri.integral_direct.coulomb.pair_class:
// coulomb_pair_classes' prune (the bound maxima, the keep flag and its scan,
// the scatter). One source for both backends -- nvcc under CUDA, clang -x hip
// under HIP.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_reduce.cuh,
// block_params.h and primitives.h include theirs first, as the bridge does
// integral_types.h.
#include "block_reduce.cuh"

#include "block_params.h"
#include "primitives.h"

#include "basis.h"
#include "pair_class_bridge.h"

namespace {

namespace cg = cooperative_groups;

// The density cap coulomb_pair_classes assumes, Dc: |D_cd| <= this for every
// density its classes are used with (pair_class.cppm says why 2, and that it
// is an assumption).
constexpr double kPruneDensityCap = 2.0;

// The maxima's block: kFourWarps threads.
constexpr uint32_t kPruneWarps = free_breakfast::kFourWarps / free_breakfast::kWarpSize;

// The keep scan's one block: the most threads a block holds.
constexpr int32_t kScanThreads = 1024;

// A double's bits, as the vendors' atomicMax takes them: a non-negative double
// orders as its bits, so a maximum of a bound is an integer one.
// int-type-ok: atomicMax and __double_as_longlong are declared on this type
using DoubleBits = unsigned long long;

__device__ DoubleBits bits_of(const double x) {
  return static_cast<DoubleBits>(__double_as_longlong(x));
}

// The larger of two.
struct MaxOp {
  __device__ double operator()(const double a, const double b) const { return a > b ? a : b; }
};

// Record r's K = w ncart(la) B as a ket, w as dq's.
__device__ double prune_ket(const int32_t r, const double ncart, const double *bound_ab,
                            const int32_t *off_a, const int32_t *off_b) {
  const double w = off_a[r] == off_b[r] ? 1.0 : 2.0;
  return w * ncart * bound_ab[r];
}

// One thread per record folds B and K into its block, and the block's thread
// 0 into maxima as DoubleBits: a maximum, so the result does not depend on
// the order.
__global__ void __launch_bounds__(free_breakfast::kFourWarps)
    coulomb_prune_maxima_kernel(const int32_t n_rec, const double ncart, const double *bound_ab,
                                const int32_t *off_a, const int32_t *off_b, double *maxima) {
  __shared__ double bra_partials[kPruneWarps];
  __shared__ double ket_partials[kPruneWarps];
  const auto r = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  double bra = 0.0;
  double ket = 0.0;
  if (r < n_rec) {
    bra = bound_ab[r];
    ket = prune_ket(r, ncart, bound_ab, off_a, off_b);
  }
  const cg::thread_block block = cg::this_thread_block();
  bra = free_breakfast::device::block_reduce<kPruneWarps>(block, bra_partials, bra, MaxOp{});
  ket = free_breakfast::device::block_reduce<kPruneWarps>(block, ket_partials, ket, MaxOp{});
  if (threadIdx.x == 0) {
    atomicMax(reinterpret_cast<DoubleBits *>(maxima), bits_of(bra));
    atomicMax(reinterpret_cast<DoubleBits *>(maxima + 1), bits_of(ket));
  }
}

// One block walks the class kScanThreads records at a time: each thread its
// record's keep flag, an inclusive scan of the flags in shared memory
// (Hillis-Steele), and the kept records before the chunk carried in every
// thread's register. Once per basis, so the simple scan does.
__global__ void __launch_bounds__(kScanThreads)
    coulomb_prune_keep_kernel(const int32_t n_rec, const double ncart, const double tau,
                              const double *bound_ab, const int32_t *off_a, const int32_t *off_b,
                              const double *maxima, int32_t *dest, int32_t *count) {
  __shared__ int32_t scan[kScanThreads];
  const auto t = static_cast<int32_t>(threadIdx.x);
  const double max_bra = maxima[0];
  const double max_ket = maxima[1];
  int32_t before = 0;
  for (int64_t base = 0; base < n_rec; base += kScanThreads) {
    const int64_t r = base + t;
    bool keep = false;
    if (r < n_rec) {
      const auto k = static_cast<int32_t>(r);
      const double bra = bound_ab[k];
      const double ket = prune_ket(k, ncart, bound_ab, off_a, off_b);
      // Pruned only when under tau both as a bra and as a ket; tau = 0 keeps
      // every record.
      keep = bra * kPruneDensityCap * max_ket >= tau || ket * kPruneDensityCap * max_bra >= tau;
    }
    scan[t] = keep ? 1 : 0;
    __syncthreads();
    for (int32_t step = 1; step < kScanThreads; step *= 2) {
      const int32_t add = t >= step ? scan[t - step] : 0;
      __syncthreads();
      scan[t] += add;
      __syncthreads();
    }
    if (r < n_rec) {
      dest[r] = keep ? before + scan[t] - 1 : -1;
    }
    before += scan[kScanThreads - 1];
    // Every thread has read the chunk's total before the next chunk writes.
    __syncthreads();
  }
  if (t == 0) {
    *count = before;
  }
}

// to[dest[r]] = from[r] for each kept r: see coulomb_prune_scatter.
template <typename T>
__global__ void coulomb_prune_scatter_kernel(const int32_t n_rec, const int32_t *dest,
                                             const T *from, T *to) {
  const auto r = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r < n_rec && dest[r] >= 0) {
    to[dest[r]] = from[r];
  }
}

} // namespace

namespace free_breakfast::device {

namespace {

// One thread per record, kFourWarps to a block; a whole number of warps keeps
// every block's lanes in step.
uint32_t record_blocks(const int32_t n_rec) {
  return math::idivup(static_cast<uint32_t>(n_rec), kFourWarps);
}

// ncart(la), for K.
double cart_count(const int32_t la) {
  return static_cast<double>(function_count_cart(la));
}

template <typename T>
wwr::wwrError_t prune_scatter(const wwr::wwrStream_t stream, const int32_t n_rec,
                              const int32_t *dest, const T *from, T *to) {
  if (n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  coulomb_prune_scatter_kernel<T>
      <<<record_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, dest, from, to);
  return wwr::wwrGetLastError();
}

} // namespace

wwr::wwrError_t coulomb_prune_maxima(const wwr::wwrStream_t stream,
                                     const PrimitivePairClassView &cls, double *maxima) {
  if (cls.n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  coulomb_prune_maxima_kernel<<<record_blocks(cls.n_rec), kFourWarps, 0, stream>>>(
      cls.n_rec, cart_count(cls.la), cls.bound_ab, cls.off_a, cls.off_b, maxima);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t coulomb_prune_keep(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                   const double *maxima, const double tau, int32_t *dest,
                                   int32_t *count) {
  if (cls.n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  coulomb_prune_keep_kernel<<<1, kScanThreads, 0, stream>>>(
      cls.n_rec, cart_count(cls.la), tau, cls.bound_ab, cls.off_a, cls.off_b, maxima, dest, count);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t coulomb_prune_scatter(const wwr::wwrStream_t stream, const int32_t n_rec,
                                      const int32_t *dest, const double *from, double *to) {
  return prune_scatter(stream, n_rec, dest, from, to);
}

wwr::wwrError_t coulomb_prune_scatter(const wwr::wwrStream_t stream, const int32_t n_rec,
                                      const int32_t *dest, const int32_t *from, int32_t *to) {
  return prune_scatter(stream, n_rec, dest, from, to);
}

} // namespace free_breakfast::device
