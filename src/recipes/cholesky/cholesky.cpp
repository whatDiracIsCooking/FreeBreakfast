/// @file
/// @brief cholesky: the diagonal blocks by cholesky.cu's kernel, the panel
/// below each by a trsm and the trailing matrix by a syrk.

module;

// The launcher, defined in cholesky.cu: declared in the GMF so it keeps
// external linkage (see the header).
#include "cholesky_bridge.h"

module free_breakfast.recipes.cholesky;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // trsm, syrk, wwrMemsetAsync, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

Result<int32_t> cholesky(DeviceResources &res, const int32_t n, double *a) {
  if (n < 0) {
    return err_config("cholesky: n = {} is negative", n);
  }
  if (n == 0) {
    return 0;
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  int32_t *const info = res.info().data();
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(info, 0, sizeof(int32_t), stream)));

  // Right-looking: once block j is factored, the panel under it is
  // A21 L11^-T, and the trailing matrix loses L21 L21^T. After a minor that is
  // not positive the later blocks leave themselves alone (the kernel reads the
  // flag), and the panels carry on over a matrix that no longer matters.
  const auto ld = static_cast<int64_t>(n);
  for (int32_t j = 0; j < n; j += device::kCholeskyBlock) {
    const int32_t jb = std::min(device::kCholeskyBlock, n - j);
    double *const diagonal = a + j + (j * ld);
    FBF_TRY(gpu_status(device::cholesky_diagonal_block(stream, jb, j, diagonal, ld, info)));
    const int32_t rest = n - j - jb;
    if (rest == 0) {
      break;
    }
    double *const panel = diagonal + jb;
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        res.blas().get(), wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, rest, jb, &kOne<double>, diagonal, n, panel, n)));
    FBF_TRY(gpu_status(wwr::syrk<double, int32_t>(res.blas().get(), wwr::WWRBLAS_FILL_MODE_LOWER,
                                                  wwr::WWRBLAS_OP_N, rest, jb, &kNegOne<double>,
                                                  panel, n, &kOne<double>, panel + (jb * ld), n)));
  }
  return res.fetch_info();
}

} // namespace free_breakfast
