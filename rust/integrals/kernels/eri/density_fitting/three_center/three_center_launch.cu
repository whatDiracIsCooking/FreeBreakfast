// The DF three-center integrals' launch wrappers over the generated
// eri3<la, lb, lP> kernels, and the (m, n) symmetrization of the spherical
// tensor. One source for both backends -- nvcc under CUDA, clang -x hip under
// HIP.
//
// Forked from
// src/integrals/eri/density_fitting/three_center/three_center_launch.cu,
// rewritten as three_center.h's extern "C", enqueue-only launch wrappers
// (rust/docs/gpu_layer.md, "Layering"), their refusals split into
// FB_LAUNCH_INVALID_ARGUMENT and FB_LAUNCH_UNSUPPORTED ("Statuses"), and the
// 32-bit extent the kernels index by, and the symmetrization's out-of-place
// rule, refused rather than assumed. The generated kernels are sources of this
// library of their own (kernels/), each one explicit specialization of the
// kernel template `eri3`. kernels/eri3.h declares the template, every
// specialization and eri3_kernels, their [la][lb][lP] table, nullptr where
// la > lb (only la <= lb is emitted), so nothing here names a class: a class
// the table names but no unit defines fails to link. Launching a
// specialization another TU defines wants relocatable device code under nvcc
// (CMakeLists.txt).

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why). The kernels' header brings the
// runtime in through felizjueves/runtime.h.
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "angular_momentum.h"
#include "block_params.h"
#include "constants.h"
#include "primitives.h"

#include "eri/density_fitting/three_center/three_center.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

#include <runtime.h>

#include "eri/density_fitting/three_center/kernels/eri3.h"

static_assert(std::extent_v<decltype(eri3_kernels), 0> == FB_THREE_CENTER_MAX_MOMENTUM + 1 &&
                  std::extent_v<decltype(eri3_kernels), 1> == FB_THREE_CENTER_MAX_MOMENTUM + 1,
              "kernels/'s [la][lb] extents and FB_THREE_CENTER_MAX_MOMENTUM disagree: rerun "
              "tools/emit/df_kernels.py");
static_assert(std::extent_v<decltype(eri3_kernels), 2> == FB_THREE_CENTER_AUX_MAX_MOMENTUM + 1,
              "kernels/'s [lP] extent and FB_THREE_CENTER_AUX_MAX_MOMENTUM disagree: rerun "
              "tools/emit/df_kernels.py");
static_assert(FB_THREE_CENTER_MAX_MOMENTUM == free_breakfast::kMaxAngularMomentum,
              "(mn|P)'s orbital limit and the orbital basis's kMaxAngularMomentum disagree");

namespace {

// n_ao * n_ao * n_aux, the element count the kernels index by a 32-bit int,
// or -1 when it does not fit one. Both counts positive.
int64_t tensor_count(const int32_t n_ao, const int32_t n_aux) {
  const int64_t count = int64_t{n_ao} * n_ao * n_aux;
  return count > INT32_MAX ? -1 : count;
}

// One thread per element (P, n, m): half of it plus half of its mirror
// (P, m, n), so the two threads of a mirrored pair write the same bits (the
// halving is exact, and a + b = b + a). Out of place: a thread reads its
// mirror, which another thread writes.
__global__ void three_center_symmetrize_kernel(const int32_t n_ao, const int32_t n_aux,
                                               const double *in, double *out) {
  const int32_t i = static_cast<int32_t>(threadIdx.x + (blockIdx.x * blockDim.x));
  if (i >= n_aux * n_ao * n_ao) {
    return;
  }
  const int32_t p = i % n_aux;
  const int32_t nm = i / n_aux;
  const int32_t n = nm % n_ao;
  const int32_t m = nm / n_ao;
  out[i] = free_breakfast::kHalf<double> * (in[i] + in[p + (n_aux * (m + (n_ao * n)))]);
}

} // namespace

extern "C" fb_launch_status
fb_three_center_class(const wwr_stream stream, const int32_t la, const int32_t lb, const int32_t lp,
                      const int32_t n_bra_rec, const double *bra, const int32_t n_aux_rec,
                      const double *aux, const int32_t n_cell, const int32_t *cells,
                      const int32_t n_ao, const int32_t n_aux, double *eri3) {
  if (bra == nullptr || aux == nullptr || cells == nullptr || eri3 == nullptr || n_bra_rec < 1 ||
      n_aux_rec < 1 || n_cell < 1 || n_ao < 1 || n_aux < 1 || tensor_count(n_ao, n_aux) < 0) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  if (la < 0 || lb < la || lb > FB_THREE_CENTER_MAX_MOMENTUM || lp < 0 ||
      lp > FB_THREE_CENTER_AUX_MAX_MOMENTUM) {
    return FB_LAUNCH_UNSUPPORTED;
  }
  const auto b = [&](const int32_t field) {
    return bra + (static_cast<size_t>(field) * static_cast<size_t>(n_bra_rec));
  };
  const auto a = [&](const int32_t field) {
    return aux + (static_cast<size_t>(field) * static_cast<size_t>(n_aux_rec));
  };
  const auto c = [&](const int32_t field) {
    return cells + (static_cast<size_t>(field) * static_cast<size_t>(n_cell));
  };
  const auto kernel = eri3_kernels[la][lb][lp];
  // One thread per cell, kFourWarps to a block.
  const uint32_t blocks =
      free_breakfast::math::idivup(static_cast<uint32_t>(n_cell), free_breakfast::kFourWarps);
  // The C ABI's stream is the runtime's, behind a void *.
  kernel<<<blocks, free_breakfast::kFourWarps, 0, static_cast<wwr::wwrStream_t>(stream)>>>(
      n_cell, n_bra_rec, n_aux_rec, n_ao, n_aux, b(0), b(1), b(2), b(3), b(4), b(5), b(6), b(7),
      b(8), b(9), b(10), a(0), a(1), a(2), a(3), a(4), c(0), c(1), c(2), c(3), c(4), c(5), c(6),
      eri3);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}

extern "C" fb_launch_status fb_three_center_symmetrize(const wwr_stream stream, const int32_t n_ao,
                                                       const int32_t n_aux, const double *input,
                                                       double *output) {
  if (input == nullptr || output == nullptr || n_ao < 1 || n_aux < 1) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  const int64_t count = tensor_count(n_ao, n_aux);
  // The two ranges as addresses, which compare across allocations where the
  // pointers themselves need not.
  const auto in_begin = reinterpret_cast<uintptr_t>(input);
  const auto out_begin = reinterpret_cast<uintptr_t>(output);
  const auto bytes = static_cast<uintptr_t>(count) * sizeof(double);
  if (count < 0 || (in_begin < out_begin + bytes && out_begin < in_begin + bytes)) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  const auto threads = static_cast<uint32_t>(count);
  three_center_symmetrize_kernel<<<
      free_breakfast::math::idivup(threads, free_breakfast::kFourWarps), free_breakfast::kFourWarps,
      0, static_cast<wwr::wwrStream_t>(stream)>>>(n_ao, n_aux, input, output);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}
