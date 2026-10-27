// The device half of free_breakfast.integrals.eri.density_fitting.three_center:
// the launcher over the generated eri3<la, lb, lP>, the check of their table
// against (mn|P)'s limits (three_center_bridge.h), and the (m, n)
// symmetrization of the spherical tensor. One source for both backends -- nvcc
// under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of the kernel template `eri3`.
// kernels/eri3.h declares the template, every specialization and eri3_kernels,
// their [la][lb][lP] table, nullptr where la > lb (only la <= lb is emitted),
// so nothing here names a class: a class the table names but no unit defines
// fails to link. Launching a specialization another TU defines wants
// relocatable device code under nvcc (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' header brings the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "three_center_bridge.h"

#include "kernels/eri3.h"

static_assert(std::extent_v<decltype(eri3_kernels), 0> ==
                      free_breakfast::device::kThreeCenterMaxMomentum + 1 &&
                  std::extent_v<decltype(eri3_kernels), 1> ==
                      free_breakfast::device::kThreeCenterMaxMomentum + 1,
              "kernels/'s [la][lb] extents and kThreeCenterMaxMomentum disagree: "
              "rerun tools/emit_df_kernels.py");

static_assert(std::extent_v<decltype(eri3_kernels), 2> ==
                  free_breakfast::device::kThreeCenterAuxMaxMomentum + 1,
              "kernels/'s [lP] extent and kThreeCenterAuxMaxMomentum disagree: "
              "rerun tools/emit_df_kernels.py");

namespace {

// The symmetrization's weight on each of the two mirrored elements.
constexpr double kThreeCenterHalf = 0.5;

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
  out[i] = kThreeCenterHalf * (in[i] + in[p + (n_aux * (m + (n_ao * n)))]);
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t three_center_class(const wwr::wwrStream_t stream, const int32_t la,
                                   const int32_t lb, const int32_t lp, const int32_t n_bra_rec,
                                   const double *bra, const int32_t n_aux_rec, const double *aux,
                                   const int32_t n_cell, const int32_t *cells, const int32_t n_ao,
                                   const int32_t n_aux, double *eri3) {
  if (la < 0 || lb < la || lb > kThreeCenterMaxMomentum || lp < 0 ||
      lp > kThreeCenterAuxMaxMomentum || n_bra_rec < 1 || n_aux_rec < 1 || n_cell < 1 || n_ao < 1 ||
      n_aux < 1) {
    return wwr::wwrErrorInvalidValue;
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
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(n_cell), kFourWarps);
  kernel<<<blocks, kFourWarps, 0, stream>>>(n_cell, n_bra_rec, n_aux_rec, n_ao, n_aux, b(0), b(1),
                                            b(2), b(3), b(4), b(5), b(6), b(7), b(8), b(9), b(10),
                                            a(0), a(1), a(2), a(3), a(4), c(0), c(1), c(2), c(3),
                                            c(4), c(5), c(6), eri3);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t three_center_symmetrize(const wwr::wwrStream_t stream, const int32_t n_ao,
                                        const int32_t n_aux, const double *in, double *out) {
  if (n_ao < 1 || n_aux < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto count = static_cast<uint32_t>(n_aux * n_ao * n_ao);
  three_center_symmetrize_kernel<<<math::idivup(count, kFourWarps), kFourWarps, 0, stream>>>(
      n_ao, n_aux, in, out);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
