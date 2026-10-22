// The device half of free_breakfast.integrals.coulomb: the launchers over the
// generated dq<la, lb>, e_hermite<lp, lq> and jab<la, lb> kernels, the two small kernels that make the inputs the generated ones
// leave to their caller (each record's weight, and jab's row map), and
// coulomb_pair_classes' prune (the bound maxima, the keep flag and its scan,
// the scatter). One source for both backends -- nvcc under CUDA, clang -x hip
// under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of its family's template. Each family's
// header declares the template, every specialization and its table, so nothing
// here names a class: a class the table names but no unit defines fails to
// link. Launching a specialization another TU defines wants relocatable device
// code under nvcc (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_reduce.cuh,
// block_params.h and primitives.h include theirs first, as the bridge does
// integral_types.h; the kernels' headers bring the runtime in through
// felizjueves/runtime.h.
#include <type_traits>

#include "block_reduce.cuh"

#include "block_params.h"
#include "primitives.h"

#include "basis.h"
#include "coulomb_bridge.h"

#include "kernels/dq/dq.h"
#include "kernels/e_hermite/e_hermite.h"
#include "kernels/jab/jab.h"

namespace {

namespace cg = cooperative_groups;

// The density cap coulomb_pair_classes assumes, Dc: |D_cd| <= this for every
// density its classes are used with (coulomb.cppm says why 2, and that it is
// an assumption).
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

// Record r's K = w ncart(la) B as a ket, w as coulomb_weights_kernel's.
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

// rows[k] = k: see coulomb_rows.
__global__ void coulomb_rows_kernel(const int32_t n, int32_t *rows) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n) {
    rows[k] = k;
  }
}

// A record's weight in the sum over unique shell pairs: a shell against itself
// (its two offsets equal, as mmd records it) is counted once, every other pair
// once for (c, d) and once for (d, c).
__global__ void coulomb_weights_kernel(const int32_t n_rec, const int32_t *off_a,
                                       const int32_t *off_b, double *w) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n_rec) {
    w[k] = off_a[k] == off_b[k] ? 1.0 : 2.0;
  }
}

} // namespace

namespace free_breakfast::device {

// One pair class per la <= lb <= kMaxAngularMomentum, and one Hermite class per
// lp, lq <= 2 kMaxAngularMomentum: a d-shell pair's E reaches |t| = 4.
constexpr auto kPairExtent = static_cast<size_t>(kMaxAngularMomentum + 1);
constexpr auto kHermiteExtent = static_cast<size_t>(2 * kMaxAngularMomentum + 1);

static_assert(std::extent_v<decltype(dq_kernels), 0> == kPairExtent &&
                  std::extent_v<decltype(dq_kernels), 1> == kPairExtent,
              "kernels/dq was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_coulomb_kernels.py");
static_assert(std::extent_v<decltype(jab_kernels), 0> == kPairExtent &&
                  std::extent_v<decltype(jab_kernels), 1> == kPairExtent,
              "kernels/jab was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_coulomb_kernels.py");
static_assert(std::extent_v<decltype(e_hermite_kernels), 0> == kHermiteExtent &&
                  std::extent_v<decltype(e_hermite_kernels), 1> == kHermiteExtent,
              "kernels/e_hermite was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_coulomb_kernels.py");

// One warp per bra record: blockDim = (1, warp width), the block e_hermite's
// Jsh[n_p * FJ_WAVEFRONT] was sized for. FJ_WAVEFRONT comes from WWR_WARP_SIZE
// (deps/CMakeLists.txt), so this holds unless the two have been split.
static_assert(e_hermite_block_dim[0] == 1 && e_hermite_block_dim[1] == kWarpSize,
              "e_hermite's block is not (1, kWarpSize): FJ_WAVEFRONT and "
              "WWR_WARP_SIZE disagree");

namespace {

bool has_pair_kernel(const PrimitivePairClassView &cls) {
  return cls.la >= 0 && cls.lb >= cls.la && cls.lb <= kMaxAngularMomentum && cls.n_rec >= 1;
}

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

wwr::wwrError_t coulomb_rows(const wwr::wwrStream_t stream, const int32_t n, int32_t *rows) {
  if (n < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  coulomb_rows_kernel<<<record_blocks(n), kFourWarps, 0, stream>>>(n, rows);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t coulomb_prepare(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                const int32_t n_ao, const double *density, double *w, double *dq,
                                double *dmax) {
  if (!has_pair_kernel(cls)) {
    return wwr::wwrErrorInvalidValue;
  }
  const uint32_t blocks = record_blocks(cls.n_rec);
  coulomb_weights_kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, cls.off_a, cls.off_b, w);
  if (const wwr::wwrError_t code = wwr::wwrGetLastError(); code != wwr::wwrSuccess) {
    return code;
  }
  const auto kernel = dq_kernels[cls.la][cls.lb];
  kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, n_ao, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z,
                                            cls.pb_x, cls.pb_y, cls.pb_z, cls.s, w, cls.off_a,
                                            cls.off_b, density, dq, dmax);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t coulomb_e_hermite(const wwr::wwrStream_t stream, const PrimitivePairClassView &bra,
                                  const PrimitivePairClassView &ket, const double tau,
                                  const double *ip, const double *iq, const double *dq,
                                  const double *dmax, double *jp) {
  if (!has_pair_kernel(bra) || !has_pair_kernel(ket)) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto kernel = e_hermite_kernels[bra.la + bra.lb][ket.la + ket.lb];
  // One warp per bra record, the kets strip-mined across its lanes; no dynamic
  // shared memory, J_p's accumulator is sized for this block in the text.
  const dim3 block(e_hermite_block_dim[0], e_hermite_block_dim[1]);
  kernel<<<static_cast<uint32_t>(bra.n_rec), block, 0, stream>>>(
      bra.n_rec, ket.n_rec, tau, bra.p_x, bra.p_y, bra.p_z, bra.zeta, ip, ket.p_x, ket.p_y, ket.p_z,
      ket.zeta, iq, dmax, dq, jp);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t coulomb_jab(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                            const int32_t n_ao, const int32_t *rows, const double *jp, double *j) {
  if (!has_pair_kernel(cls)) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto kernel = jab_kernels[cls.la][cls.lb];
  kernel<<<record_blocks(cls.n_rec), kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, cls.n_rec, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x, cls.pb_y,
      cls.pb_z, cls.s, cls.off_a, cls.off_b, rows, jp, j);
  return wwr::wwrGetLastError();
}

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
