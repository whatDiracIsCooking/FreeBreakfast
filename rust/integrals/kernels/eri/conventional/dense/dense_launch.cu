// The dense conventional ERI's launch wrapper over the generated
// eri_dj<la, lb, lc, ld> and eri_dk<la, lb, lc, ld> kernels. One source for
// both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// rust/'s own, with no C++ launcher to fork (dense.h has why), in the shape
// of conventional_launch.cu (rust/docs/gpu_layer.md, "Layering",
// "Statuses"): extern "C", enqueue-only, a refusal split into
// FB_LAUNCH_INVALID_ARGUMENT and FB_LAUNCH_UNSUPPORTED, and the 32-bit
// extent the kernels index by refused rather than assumed. The records are
// read in place from each class's record rows (fb_primitive_pair_class_view).
// kernels/eri_dj.h and kernels/eri_dk.h declare each layout's template,
// every specialization and its [la][lb][lc][ld] table, so nothing here
// names a class. Launching a kernel another TU defines wants relocatable
// device code under nvcc (../../../CMakeLists.txt).

// Standard headers, and ours (which include only standard headers), before
// the wwr headers below: under HIP the vendor runtime header they pull in
// must not precede libc++ (primitives.h has why). The kernels' headers bring
// the runtime in through felizjueves/runtime.h.
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "eri/conventional/conventional.h"
#include "eri/conventional/dense/dense.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

#include <runtime.h>

#include "eri/conventional/dense/kernels/eri_dj.h"
#include "eri/conventional/dense/kernels/eri_dk.h"
#include "eri/conventional/kernels/eri_task_threads.h"

static_assert(std::extent_v<decltype(eri_dj_kernels), 0> == FB_CONVENTIONAL_MAX_MOMENTUM + 1 &&
                  std::extent_v<decltype(eri_dj_kernels), 1> == FB_CONVENTIONAL_MAX_MOMENTUM + 1 &&
                  std::extent_v<decltype(eri_dj_kernels), 2> == FB_CONVENTIONAL_MAX_MOMENTUM + 1 &&
                  std::extent_v<decltype(eri_dj_kernels), 3> == FB_CONVENTIONAL_MAX_MOMENTUM + 1 &&
                  std::is_same_v<decltype(eri_dj_kernels), decltype(eri_dk_kernels)>,
              "kernels/'s [la][lb][lc][ld] extents and FB_CONVENTIONAL_MAX_MOMENTUM disagree: "
              "rerun tools/emit/conventional_kernels.py");
static_assert(std::is_same_v<std::remove_cv_t<decltype(eri_dense_task_threads)>,
                             std::remove_cv_t<decltype(eri_task_threads)>>,
              "kernels/eri_task_threads.h's two tables disagree: "
              "rerun tools/emit/conventional_kernels.py");

namespace {

// The blocks for one thread each of `threads`, kFourWarps to a block.
uint32_t blocks_for(const int32_t threads) {
  return free_breakfast::math::idivup(static_cast<uint32_t>(threads), free_breakfast::kFourWarps);
}

// The row dot's block: a fixed count, so the summation order is n_col's
// alone.
constexpr uint32_t kRowDotThreads = free_breakfast::kFourWarps;

// One block per row r: thread t sums the products at columns t, t + B, ...
// in order, then the block's partials are summed by a fixed tree in shared
// memory. Nothing in the order depends on the row, the grid or the device.
__global__ void dense_row_dot_kernel(const int32_t n_col, const double *m, const double *x,
                                     double *y) {
  __shared__ double partial[kRowDotThreads];
  const int32_t r = static_cast<int32_t>(blockIdx.x);
  const double *row = m + (static_cast<size_t>(r) * static_cast<size_t>(n_col));
  double sum = 0.0;
  for (int32_t c = static_cast<int32_t>(threadIdx.x); c < n_col;
       c += static_cast<int32_t>(kRowDotThreads)) {
    sum = fma(row[c], x[c], sum);
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t half = kRowDotThreads / 2; half > 0; half /= 2) {
    if (threadIdx.x < half) {
      partial[threadIdx.x] += partial[threadIdx.x + half];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    y[r] = partial[0];
  }
}

static_assert((kRowDotThreads & (kRowDotThreads - 1)) == 0, "the tree halves a power of two");

} // namespace

extern "C" fb_launch_status fb_dense_eri_class(const wwr_stream stream, const int32_t layout,
                                               const fb_primitive_pair_class_view *bra,
                                               const fb_primitive_pair_class_view *ket,
                                               const int32_t n_bpair, const int32_t bra_stride,
                                               const int32_t *bra_pairs, const int32_t n_kpair,
                                               const int32_t *ket_pairs, const int32_t task_stride,
                                               const int32_t *tasks, const int32_t n_ao,
                                               const int32_t m0, const int32_t n_row, double *eri) {
  if ((layout != FB_DENSE_LAYOUT_J && layout != FB_DENSE_LAYOUT_K) || bra == nullptr ||
      ket == nullptr || bra_pairs == nullptr || ket_pairs == nullptr || tasks == nullptr ||
      eri == nullptr || n_bpair < 1 || n_kpair < 1 || n_ao < 1 || n_row < 1 || m0 < 0 ||
      bra_stride < n_bpair) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  const int64_t n_task = static_cast<int64_t>(n_bpair) * n_kpair;
  const int64_t n_col = static_cast<int64_t>(n_ao) * n_ao;
  if (n_task > task_stride || n_col > INT32_MAX || n_row % n_ao != 0 ||
      static_cast<int64_t>(m0) * n_ao + n_row > n_col ||
      static_cast<int64_t>(n_row) * n_col > INT32_MAX) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  // Each side's fields the kernels read, in their parameter order.
  const double *const b[] = {bra->zeta, bra->p_x,  bra->p_y,  bra->p_z,  bra->pa_x, bra->pa_y,
                             bra->pa_z, bra->pb_x, bra->pb_y, bra->pb_z, bra->s};
  const double *const k[] = {ket->zeta, ket->p_x,  ket->p_y,  ket->p_z,  ket->pa_x, ket->pa_y,
                             ket->pa_z, ket->pb_x, ket->pb_y, ket->pb_z, ket->s};
  for (int32_t f = 0; f < 11; ++f) {
    if (b[f] == nullptr || k[f] == nullptr) {
      return FB_LAUNCH_INVALID_ARGUMENT;
    }
  }
  if (bra->n_rec < 1 || ket->n_rec < 1) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  const int32_t la = bra->la;
  const int32_t lb = bra->lb;
  const int32_t lc = ket->la;
  const int32_t ld = ket->lb;
  const auto in_table = [](const int32_t l) { return l >= 0 && l <= FB_CONVENTIONAL_MAX_MOMENTUM; };
  if (!in_table(la) || !in_table(lb) || !in_table(lc) || !in_table(ld)) {
    return FB_LAUNCH_UNSUPPORTED;
  }
  const auto kernel =
      layout == FB_DENSE_LAYOUT_J ? eri_dj_kernels[la][lb][lc][ld] : eri_dk_kernels[la][lb][lc][ld];
  // Threads per task of the form the class was emitted in (FelizJueves's
  // task_threads: 1 through (pp|pp); from a d shell the low-stack form's,
  // the n_b rows of one bra Cartesian a thread), the task the fast index.
  const int32_t per_task = eri_dense_task_threads[la][lb][lc][ld];
  if (kernel == nullptr || per_task < 1) {
    return FB_LAUNCH_UNSUPPORTED;
  }
  // The kernel's thread index is an int32: refused, not wrapped.
  const int64_t n_thread = n_task * per_task;
  if (n_thread > INT32_MAX) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  const auto bp = [&](const int32_t field) {
    return bra_pairs + (static_cast<size_t>(field) * static_cast<size_t>(bra_stride));
  };
  const auto kp = [&](const int32_t field) {
    return ket_pairs + (static_cast<size_t>(field) * static_cast<size_t>(n_kpair));
  };
  const auto t = [&](const int32_t field) {
    return tasks + (static_cast<size_t>(field) * static_cast<size_t>(task_stride));
  };
  const auto n_tasks = static_cast<int32_t>(n_task);
  // The C ABI's stream is the runtime's, behind a void *.
  kernel<<<blocks_for(static_cast<int32_t>(n_thread)), free_breakfast::kFourWarps, 0,
           static_cast<wwr::wwrStream_t>(stream)>>>(
      n_tasks, bra->n_rec, ket->n_rec, n_bpair, n_kpair, n_ao, m0, n_row,
      static_cast<int32_t>(n_col), b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9],
      b[10], k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9], k[10], bp(0), bp(1), bp(2),
      bp(3), kp(0), kp(1), kp(2), kp(3), t(0), t(1), eri);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}

extern "C" fb_launch_status fb_dense_row_dot(const wwr_stream stream, const int32_t n_row,
                                             const int32_t n_col, const double *m, const double *x,
                                             double *y) {
  if (m == nullptr || x == nullptr || y == nullptr || n_row < 1 || n_col < 1 ||
      static_cast<int64_t>(n_row) * n_col > INT32_MAX) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  dense_row_dot_kernel<<<static_cast<uint32_t>(n_row), kRowDotThreads, 0,
                         static_cast<wwr::wwrStream_t>(stream)>>>(n_col, m, x, y);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}
