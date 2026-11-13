// The device half of mmd: each class's records formed on the device from the
// basis's shells and the class's shell pairs, in descending zeta order (the
// order sorted from zeta alone, then every field written once, already in
// it), then each record's Schwarz bound through the generated
// pair_bound<la, lb> (kernels/pair_bound/) and its dressed bound, as stored
// and swapped, through the generated record_bound<la, lb>
// (kernels/record_bound/). Each class is its own unit of this library:
// launching a specialization another TU defines wants relocatable device code
// under nvcc. One source for both backends -- nvcc under CUDA, clang -x hip
// under HIP.
//
// Forked from src/integrals/mmd/mmd_launch.cu: the kernels are unchanged, and
// the launchers are rewritten as mmd.h's extern "C", enqueue-only launch
// wrappers (rust/docs/gpu_layer.md, "Layering"). The order's two scratch
// arrays are one workspace, the forked merge sort's (fb_merge_sort, in
// libfb_recipes_kernels.so); the status is the C ABI's wwr_get_last_error(),
// so no vendor code crosses; a class outside a table is
// FB_LAUNCH_UNSUPPORTED, split from the invalid arguments.

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why).
#include <stddef.h>
#include <type_traits>

#include "block_params.h"
#include "constants.h"
#include "primitives.h"

#include "mmd/mmd.h"
#include "sort/merge_sort.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

#include <runtime.h>

#include "mmd/kernels/pair_bound/pair_bound.h"
#include "mmd/kernels/record_bound/record_bound.h"

// The C structs' layout, which fb-integrals-sys asserts with the same numbers:
// both sides follow the C rule over the same fields, and these catch a field
// added on one side only.
static_assert(sizeof(fb_primitive_pair_source) == 96 &&
                  offsetof(fb_primitive_pair_source, bra) == 8 &&
                  offsetof(fb_primitive_pair_source, rank) == 88,
              "fb_primitive_pair_source's layout changed: update fb-integrals-sys's assertions");
static_assert(sizeof(fb_primitive_pair_records) == 136 &&
                  offsetof(fb_primitive_pair_records, rank_b) == 128,
              "fb_primitive_pair_records's layout changed: update fb-integrals-sys's assertions");
static_assert(sizeof(fb_primitive_pair_class_view) == 176 &&
                  offsetof(fb_primitive_pair_class_view, zeta) == 16 &&
                  offsetof(fb_primitive_pair_class_view, rank_b) == 168,
              "fb_primitive_pair_class_view's layout changed: update fb-integrals-sys's "
              "assertions");

namespace {

// Host record r's shell pair and its two primitives' indices.
struct RecordPrimitives {
  int32_t pair;
  int32_t ia;
  int32_t ib;
};

// Host record r: the shell pair whose [rec_begin[p], rec_begin[p + 1]) holds
// it (rec_begin[0] = 0 <= r < rec_begin[n_pairs]), then its offset there,
// bra-primitive-major.
__device__ RecordPrimitives record_primitives(const fb_primitive_pair_source &src,
                                              const int32_t r) {
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

// One axis of P = (alpha A + beta B) / zeta, as one explicit FMA: the
// larger-exponent term fused, the other rounded first. The terms are put in
// exponent order before either is formed, so a record and its primitives
// swapped run the same operations on the same operands: a diagonal shell
// pair's record and its mirror carry the same P, bit for bit, which K's gather
// relies on. Explicit rather than left to contraction, so both backends fuse
// the same product. The FMA rounds once where the plain expression rounds
// twice, so P differs from a host's unfused value by an ulp at times; the
// dressed bound's pair factor exp(zeta X_PA X_PB) magnifies that on a distant
// pair.
__device__ double product_centre(const double alpha, const double a, const double beta,
                                 const double b, const double zeta) {
  const bool alpha_first = alpha <= beta;
  const double e_lo = alpha_first ? alpha : beta;
  const double c_lo = alpha_first ? a : b;
  const double e_hi = alpha_first ? beta : alpha;
  const double c_hi = alpha_first ? b : a;
  return free_breakfast::math::fma(e_hi, c_hi, e_lo * c_lo) / zeta;
}

// key[r] = -zeta and perm[r] = r for host record r: the ascending sort of key
// is the descending order of zeta, carrying each record's host index.
__global__ void mmd_order_key_kernel(const int32_t n_rec, const fb_primitive_pair_source src,
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
__global__ void mmd_records_kernel(const int32_t n_rec, const fb_primitive_pair_source src,
                                   const uint32_t *perm, const fb_primitive_pair_records out) {
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

// The largest momentum each emitted table covers, its own limit rather than the
// tree's kMaxAngularMomentum: the two come from different emitters, so each
// launcher reads its own, and the header exports it.
constexpr int32_t kPairBoundMaxL =
    static_cast<int32_t>(std::extent_v<decltype(pair_bound_kernels), 0>) - 1;
constexpr int32_t kRecordBoundMaxL =
    static_cast<int32_t>(std::extent_v<decltype(record_bound_kernels), 0>) - 1;

static_assert(std::extent_v<decltype(pair_bound_kernels), 1> == kPairBoundMaxL + 1,
              "kernels/pair_bound's [la][lb] table is not square: rerun "
              "tools/emit/one_electron_kernels.py --out-root");
static_assert(kPairBoundMaxL == FB_PAIR_BOUND_MAX_MOMENTUM,
              "kernels/pair_bound's table and FB_PAIR_BOUND_MAX_MOMENTUM disagree");

static_assert(std::extent_v<decltype(record_bound_kernels), 1> == kRecordBoundMaxL + 1,
              "kernels/record_bound's [la][lb] table is not square: rerun "
              "tools/emit/exchange_kernels.py --out-root");
static_assert(kRecordBoundMaxL == FB_RECORD_BOUND_MAX_MOMENTUM,
              "kernels/record_bound's table and FB_RECORD_BOUND_MAX_MOMENTUM disagree");

// The launch just enqueued, as a status: the C ABI's own wwr_get_last_error(),
// already mapped onto wwr_status.
fb_launch_status last_launch_status() {
  return static_cast<fb_launch_status>(wwr_get_last_error());
}

uint32_t mmd_record_blocks(const int32_t n_rec) {
  return free_breakfast::math::idivup(static_cast<uint32_t>(n_rec), free_breakfast::kFourWarps);
}

} // namespace

extern "C" size_t fb_mmd_order_workspace_bytes(const int32_t n_rec) {
  return n_rec < 1 ? 0 : fb_merge_sort_workspace_bytes(static_cast<uint32_t>(n_rec), true);
}

extern "C" fb_launch_status fb_mmd_order(const wwr_stream stream, const int32_t n_rec,
                                         const fb_primitive_pair_source *src, double *key,
                                         uint32_t *perm, void *workspace,
                                         const size_t workspace_bytes) {
  if (n_rec < 1 || src == nullptr || src->n_pairs < 1 || key == nullptr || perm == nullptr ||
      workspace_bytes < fb_mmd_order_workspace_bytes(n_rec)) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  // The C ABI's stream is the runtime's, behind a void *.
  mmd_order_key_kernel<<<mmd_record_blocks(n_rec), free_breakfast::kFourWarps, 0,
                         static_cast<wwr::wwrStream_t>(stream)>>>(n_rec, *src, key, perm);
  if (const fb_launch_status status = last_launch_status(); status != FB_LAUNCH_SUCCESS) {
    return status;
  }
  return fb_merge_sort(stream, static_cast<uint32_t>(n_rec), key, perm, workspace, workspace_bytes);
}

extern "C" fb_launch_status fb_mmd_records(const wwr_stream stream, const int32_t n_rec,
                                           const fb_primitive_pair_source *src,
                                           const uint32_t *perm,
                                           const fb_primitive_pair_records *out) {
  if (n_rec < 1 || src == nullptr || src->n_pairs < 1 || perm == nullptr || out == nullptr) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  mmd_records_kernel<<<mmd_record_blocks(n_rec), free_breakfast::kFourWarps, 0,
                       static_cast<wwr::wwrStream_t>(stream)>>>(n_rec, *src, perm, *out);
  return last_launch_status();
}

extern "C" fb_launch_status fb_mmd_bound(const wwr_stream stream, const int32_t la,
                                         const int32_t lb, const int32_t n_rec, const double *zeta,
                                         double *bound) {
  if (n_rec < 1 || zeta == nullptr || bound == nullptr) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  if (la < 0 || lb < la || lb > kPairBoundMaxL) {
    return FB_LAUNCH_UNSUPPORTED;
  }
  pair_bound_kernels[la][lb]<<<mmd_record_blocks(n_rec), free_breakfast::kFourWarps, 0,
                               static_cast<wwr::wwrStream_t>(stream)>>>(n_rec, zeta, bound);
  return last_launch_status();
}

extern "C" fb_launch_status fb_ordered_pair_bound(const wwr_stream stream,
                                                  const fb_primitive_pair_class_view *cls,
                                                  double *bound) {
  if (cls == nullptr || bound == nullptr || cls->n_rec < 1) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  if (cls->la < 0 || cls->la > kRecordBoundMaxL || cls->lb < 0 || cls->lb > kRecordBoundMaxL) {
    return FB_LAUNCH_UNSUPPORTED;
  }
  // One thread per record in 32-thread blocks, as FelizJueves's launch_shape
  // states it; the mapping is not part of the numerical contract.
  constexpr uint32_t kThreads = 32;
  const auto kernel = record_bound_kernels[cls->la][cls->lb];
  const uint32_t blocks = free_breakfast::math::idivup(static_cast<uint32_t>(cls->n_rec), kThreads);
  kernel<<<blocks, kThreads, 0, static_cast<wwr::wwrStream_t>(stream)>>>(
      cls->n_rec, cls->zeta, cls->pa_x, cls->pa_y, cls->pa_z, cls->pb_x, cls->pb_y, cls->pb_z,
      cls->s, bound);
  return last_launch_status();
}
