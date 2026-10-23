// The device half of free_breakfast.integrals.mmd: each class's records
// formed on the device from the basis's shells and the class's shell pairs,
// once per basis, in descending zeta order (the order sorted from zeta alone,
// then every field written once, already in it), then each record's Schwarz
// bound through the generated pair_bound<la, lb> (kernels/pair_bound/,
// written by tools/emit_one_electron_kernels.py) and its dressed bound, as
// stored and swapped, through the generated record_bound<la, lb>
// (kernels/record_bound/, written by tools/emit_exchange_kernels.py), each
// class its own unit of this library: launching a specialization another TU
// defines wants relocatable device code under nvcc. One source for both
// backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridges do integral_types.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "angular_momentum.h"
#include "constants.h"
#include "merge_sort_bridge.h"
#include "mmd_bridge.h"

#include "kernels/pair_bound/pair_bound.h"
#include "kernels/record_bound/record_bound.h"

namespace {

using free_breakfast::device::PrimitivePairRecords;
using free_breakfast::device::PrimitivePairSource;

// Host record r's shell pair and its two primitives' indices.
struct RecordPrimitives {
  int32_t pair;
  int32_t ia;
  int32_t ib;
};

// Host record r: the shell pair whose [rec_begin[p], rec_begin[p + 1]) holds
// it (rec_begin[0] = 0 <= r < rec_begin[n_pairs]), then its offset there,
// bra-primitive-major.
__device__ RecordPrimitives record_primitives(const PrimitivePairSource &src, const int32_t r) {
  int32_t lo = 0;
  int32_t hi = src.n_pairs;
  while (hi - lo > 1) {
    const int32_t mid = lo + ((hi - lo) / 2);
    if (src.rec_begin[mid] <= r) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const uint32_t a = src.bra[lo];
  const uint32_t b = src.ket[lo];
  const int32_t kb = src.prim_begin[b + 1] - src.prim_begin[b];
  const int32_t local = r - src.rec_begin[lo];
  return {lo, src.prim_begin[a] + (local / kb), src.prim_begin[b] + (local % kb)};
}

// One axis of P = (alpha A + beta B) / zeta, each product rounded on its own:
// never contracted into an FMA, so P is the correctly rounded IEEE value of
// this expression, bit for bit what a host computes. Two things lean on it.
// The dressed bound's pair factor exp(zeta X_PA X_PB) turns one ulp of P into
// hundreds of ulps of bound on a distant pair, and FelizJueves's goldens
// (exchange_pairs_gpu.cpp) are formed this way. And with the products rounded
// separately the sum commutes exactly, so a diagonal shell pair's record and
// its mirror carry the same P. nvcc's _rn intrinsics are never fused; clang
// (HIP) honours the pragma under its default -ffp-contract=fast-honor-pragmas.
__device__ double product_centre(const double alpha, const double a, const double beta,
                                 const double b, const double zeta) {
#if defined(__NVCC__)
  return __ddiv_rn(__dadd_rn(__dmul_rn(alpha, a), __dmul_rn(beta, b)), zeta);
#else
#pragma clang fp contract(off)
  return ((alpha * a) + (beta * b)) / zeta;
#endif
}

// key[r] = -zeta and perm[r] = r for host record r: the ascending sort of key
// is the descending order of zeta, carrying each record's host index.
__global__ void mmd_order_key_kernel(const int32_t n_rec, const PrimitivePairSource src,
                                     double *key, uint32_t *perm) {
  const auto r = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r < n_rec) {
    const RecordPrimitives rp = record_primitives(src, r);
    key[r] = -(src.exponent[rp.ia] + src.exponent[rp.ib]);
    perm[r] = static_cast<uint32_t>(r);
  }
}

// Record k from host record perm[k]: one thread per record, the stores
// coalesced.
__global__ void mmd_records_kernel(const int32_t n_rec, const PrimitivePairSource src,
                                   const uint32_t *perm, const PrimitivePairRecords out) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k >= n_rec) {
    return;
  }
  const RecordPrimitives rp = record_primitives(src, static_cast<int32_t>(perm[k]));
  const uint32_t a = src.bra[rp.pair];
  const uint32_t b = src.ket[rp.pair];
  const double alpha = src.exponent[rp.ia];
  const double beta = src.exponent[rp.ib];
  const double zeta = alpha + beta;
  const double px = product_centre(alpha, src.c_x[a], beta, src.c_x[b], zeta);
  const double py = product_centre(alpha, src.c_y[a], beta, src.c_y[b], zeta);
  const double pz = product_centre(alpha, src.c_z[a], beta, src.c_z[b], zeta);
  out.zeta[k] = zeta;
  out.p_x[k] = px;
  out.p_y[k] = py;
  out.p_z[k] = pz;
  out.pa_x[k] = px - src.c_x[a];
  out.pa_y[k] = py - src.c_y[a];
  out.pa_z[k] = pz - src.c_z[a];
  out.pb_x[k] = px - src.c_x[b];
  out.pb_y[k] = py - src.c_y[b];
  out.pb_z[k] = pz - src.c_z[b];
  out.s[k] = src.coefficient[rp.ia] * src.coefficient[rp.ib] *
             free_breakfast::math::pow(free_breakfast::kPi<double> / zeta, 1.5);
  out.ea[k] = alpha;
  out.eb[k] = beta;
  out.off_a[k] = src.ao_offset[a];
  out.off_b[k] = src.ao_offset[b];
  out.rank_a[k] = src.rank[a];
  out.rank_b[k] = src.rank[b];
}

} // namespace

namespace free_breakfast::device {

// One pair class per la <= lb <= kMaxAngularMomentum.
constexpr auto kPairExtent = static_cast<size_t>(kMaxAngularMomentum + 1);

static_assert(std::extent_v<decltype(pair_bound_kernels), 0> == kPairExtent &&
                  std::extent_v<decltype(pair_bound_kernels), 1> == kPairExtent,
              "kernels/pair_bound was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_one_electron_kernels.py");

static_assert(std::extent_v<decltype(record_bound_kernels), 0> == kPairExtent &&
                  std::extent_v<decltype(record_bound_kernels), 1> == kPairExtent,
              "kernels/record_bound was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_exchange_kernels.py");

namespace {

uint32_t mmd_record_blocks(const int32_t n_rec) {
  return math::idivup(static_cast<uint32_t>(n_rec), kFourWarps);
}

} // namespace

wwr::wwrError_t mmd_order(const wwr::wwrStream_t stream, const int32_t n_rec,
                          const PrimitivePairSource &src, double *key, double *key_alt,
                          uint32_t *perm, uint32_t *perm_alt) {
  if (n_rec < 1 || src.n_pairs < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  mmd_order_key_kernel<<<mmd_record_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, src, key, perm);
  if (const wwr::wwrError_t code = wwr::wwrGetLastError(); code != wwr::wwrSuccess) {
    return code;
  }
  return merge_sort_passes(stream, static_cast<uint32_t>(n_rec), key, key_alt, perm, perm_alt);
}

wwr::wwrError_t mmd_records(const wwr::wwrStream_t stream, const int32_t n_rec,
                            const PrimitivePairSource &src, const uint32_t *perm,
                            const PrimitivePairRecords &out) {
  if (n_rec < 1 || src.n_pairs < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  mmd_records_kernel<<<mmd_record_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, src, perm, out);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t mmd_bound(const wwr::wwrStream_t stream, const int32_t la, const int32_t lb,
                          const int32_t n_rec, const double *zeta, double *bound) {
  if (n_rec < 1 || la < 0 || lb < la || lb > kMaxAngularMomentum) {
    return wwr::wwrErrorInvalidValue;
  }
  pair_bound_kernels[la][lb]<<<mmd_record_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, zeta,
                                                                                  bound);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t ordered_pair_bound(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                   double *bound) {
  if (cls.n_rec < 1 || cls.la < 0 || cls.la > kMaxAngularMomentum || cls.lb < 0 ||
      cls.lb > kMaxAngularMomentum) {
    return wwr::wwrErrorInvalidValue;
  }
  // One thread per record in 32-thread blocks, as FelizJueves's launch_shape
  // states it; the mapping is not part of the numerical contract.
  constexpr uint32_t kThreads = 32;
  record_bound_kernels[cls.la][cls.lb]<<<math::idivup(static_cast<uint32_t>(cls.n_rec), kThreads),
                                         kThreads, 0, stream>>>(cls.n_rec, cls.zeta, cls.pa_x,
                                                                cls.pa_y, cls.pa_z, cls.pb_x,
                                                                cls.pb_y, cls.pb_z, cls.s, bound);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
