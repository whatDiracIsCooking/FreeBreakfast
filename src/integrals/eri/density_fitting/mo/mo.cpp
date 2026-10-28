/// @file
/// @brief b_half (one batched gemm) and b_mo (that, then a gemm) over the
/// fitted B.

module free_breakfast.integrals.eri.density_fitting.mo;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm(StridedBatched), wwrMemGetInfo, wwrMemsetAsync, gpu_status
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

// The checks b_half and b_mo share: every count non-negative, each range
// inside C's columns, and X's rows, naux |rows|, in the int32_t dispatch (the
// second gemm's m and leading dimension).
Status check_mo_ranges(const char *caller, const int32_t naux, const int32_t nao, const int32_t nmo,
                       const std::span<const OrbitalRange> ranges) {
  if (naux < 0 || nao < 0 || nmo < 0) {
    return err_config("{}: naux = {}, nao = {} and nmo = {} must be non-negative", caller, naux,
                      nao, nmo);
  }
  for (const OrbitalRange r : ranges) {
    if (r.first < 0 || r.count < 0 || int64_t{r.first} + r.count > nmo) {
      return err_config("{}: columns [{}, {} + {}) are outside C's {}", caller, r.first, r.first,
                        r.count, nmo);
    }
  }
  const int64_t rows = int64_t{naux} * ranges.front().count;
  if (rows > std::numeric_limits<int32_t>::max()) {
    return err_config("{}: naux |rows| = {} exceeds int32_t", caller, rows);
  }
  return {};
}

// X_m = B_m C_rows for every m: naux x |rows| from naux x nao times
// nao x |rows|, C_rows shared by the batch. df_exchange's call exactly, for
// rows the occupied block.
Status half_transform(DeviceResources &res, const int32_t naux, const int32_t nao, const double *b,
                      const double *c, const OrbitalRange rows, double *x) {
  return gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, rows.count, nao, &kOne<double>,
      b, naux, int64_t{naux} * nao, c + (int64_t{nao} * rows.first), nao, 0, &kZero<double>, x,
      naux, int64_t{naux} * rows.count, nao));
}

Status b_half(DeviceResources &res, const int32_t naux, const int32_t nao, const int32_t nmo,
              const double *b, const double *c, const OrbitalRange rows, double *out) {
  FBF_TRY(check_mo_ranges("b_half", naux, nao, nmo, std::array{rows}));
  if (naux == 0 || nao == 0 || rows.count == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return half_transform(res, naux, nao, b, c, rows, out);
}

Status b_mo(DeviceResources &res, const int32_t naux, const int32_t nao, const int32_t nmo,
            const double *b, const double *c, const OrbitalRange rows, const OrbitalRange cols,
            double *out) {
  FBF_TRY(check_mo_ranges("b_mo", naux, nao, nmo, std::array{rows, cols}));
  if (naux == 0 || rows.count == 0 || cols.count == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto x_rows = static_cast<int32_t>(int64_t{naux} * rows.count);
  if (nao == 0) {
    // No AO to sum over: B_Q,pq = 0, which a gemm of inner dimension 0 is not
    // promised to write on both backends.
    const auto count = static_cast<size_t>(x_rows) * static_cast<size_t>(cols.count);
    return gpu_status(wwr::wwrMemsetAsync(out, 0, sizeof(double) * count, res.stream()));
  }

  // X past the device's memory would abort in device_workspace (CreatePolicy),
  // not return. A workspace already that large is not grown, so is not
  // checked. X's doubles are below 2^62, but its bytes need not fit size_t.
  const size_t x_count = static_cast<size_t>(x_rows) * static_cast<size_t>(nao);
  if (x_count > std::numeric_limits<size_t>::max() / sizeof(double)) {
    return err_unsupported("b_mo: the half transform's {} doubles overflow size_t bytes", x_count);
  }
  const size_t x_bytes = sizeof(double) * x_count;
  if (x_bytes > res.device_workspace(0).num_elements()) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
    if (x_bytes > free_bytes) {
      return err_unsupported("b_mo: the half transform over {} fitting functions, {} orbital "
                             "functions and {} rows needs {} bytes of device memory, and device "
                             "{} has {} free",
                             naux, nao, rows.count, x_bytes, res.dev_idx(), free_bytes);
    }
  }
  const auto [x] = res.device_regions<double>(x_count);
  FBF_TRY(half_transform(res, naux, nao, b, c, rows, x));

  // B_Q,pq = sum_m X_Q,pm C_m,q: X as the (naux |rows|) x nao matrix, times
  // C_cols.
  return gpu_status(wwr::gemm<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, x_rows, cols.count, nao,
      &kOne<double>, x, x_rows, c + (int64_t{nao} * cols.first), nao, &kZero<double>, out, x_rows));
}

} // namespace free_breakfast
