/// @file
/// @brief orbital_pairs: the packed order's two blocks as 2D copies each way,
/// and unpack's two transposing geams for kappa's upper blocks.

module free_breakfast.response.orbital_pairs;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // geam, wwrMemcpy2DAsync, wwrMemsetAsync, gpu_status
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast::response {

// What both packing functions refuse; caller names the function.
Status check_orbital_pairs(const std::string_view caller, const int32_t nmo, const int32_t ncore,
                           const int32_t ncas) {
  if (ncore < 0 || ncas < 0) {
    return err_config("{}: ncore ({}) and ncas ({}) must not be negative", caller, ncore, ncas);
  }
  if (std::cmp_greater(int64_t{ncore} + ncas, nmo)) {
    return err_config("{}: ncore {} + ncas {} exceeds the {} orbitals", caller, ncore, ncas, nmo);
  }
  return {};
}

// The packed order's two blocks, as 2D copies between the nmo x nmo matrix and
// the packed vector: from the matrix @p from into the vector @p to when
// to_packed, otherwise from the vector into the matrix. A block with no rows
// or no columns is skipped, since a zero-extent copy is not portable across
// the two runtimes.
Status copy_orbital_pairs(DeviceResources &res, const int32_t nmo, const int32_t ncore,
                          const int32_t ncas, const double *const from, double *const to,
                          const bool to_packed) {
  const auto n = static_cast<size_t>(nmo);
  const auto core = static_cast<size_t>(ncore);
  const auto occ = core + static_cast<size_t>(ncas);
  const size_t n_rot = n - core; // the active and virtual rows of a core column
  const size_t n_vir = n - occ;  // the virtual rows of an active column
  const size_t pitch = n * sizeof(double);
  // (first row, row count, first column, column count, offset into the vector)
  const std::array<std::array<size_t, 5>, 2> blocks{
      {{core, n_rot, 0, core, 0}, {occ, n_vir, core, occ - core, core * n_rot}}};
  for (const auto &[row, rows, col, cols, at] : blocks) {
    if (rows == 0 || cols == 0) {
      continue;
    }
    const size_t width = rows * sizeof(double);
    const size_t offset = row + (n * col);
    FBF_TRY(gpu_status(
        wwr::wwrMemcpy2DAsync(to_packed ? to + at : to + offset, to_packed ? width : pitch,
                              to_packed ? from + offset : from + at, to_packed ? pitch : width,
                              width, cols, wwr::wwrMemcpyDeviceToDevice, res.stream())));
  }
  return {};
}

Status pack_orbital_pairs(DeviceResources &res, const int32_t nmo, const int32_t ncore,
                          const int32_t ncas, const double *const full, double *const packed) {
  FBF_TRY(check_orbital_pairs("pack_orbital_pairs", nmo, ncore, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  return copy_orbital_pairs(res, nmo, ncore, ncas, full, packed, true);
}

Status unpack_orbital_pairs(DeviceResources &res, const int32_t nmo, const int32_t ncore,
                            const int32_t ncas, const double *const packed, double *const kappa) {
  FBF_TRY(check_orbital_pairs("unpack_orbital_pairs", nmo, ncore, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto n = static_cast<size_t>(nmo);
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(kappa, 0, n * n * sizeof(double), res.stream())));
  FBF_TRY(copy_orbital_pairs(res, nmo, ncore, ncas, packed, kappa, false));

  // The upper blocks, kappa_qp = -kappa_pq: each the negated transpose of a
  // lower block it shares no element with. geam's B is C itself at beta = 0,
  // its in-place mode, so nothing past kappa is read.
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const int32_t nocc = ncore + ncas;
  // (rows of the upper block, its columns, the lower block's first row, the
  // shared first column of the lower block and row of the upper one)
  const std::array<std::array<int32_t, 4>, 2> blocks{
      {{ncore, nmo - ncore, ncore, 0}, {ncas, nmo - nocc, nocc, ncore}}};
  for (const auto &[rows, cols, lower_row, first] : blocks) {
    if (rows == 0 || cols == 0) {
      continue;
    }
    const double *const lower = kappa + lower_row + (n * static_cast<size_t>(first));
    double *const upper = kappa + first + (n * static_cast<size_t>(lower_row));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, rows,
                                                  cols, &kNegOne<double>, lower, nmo,
                                                  &kZero<double>, upper, nmo, upper, nmo)));
  }
  return {};
}

} // namespace free_breakfast::response
