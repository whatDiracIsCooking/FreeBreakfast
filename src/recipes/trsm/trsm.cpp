/// @file
/// @brief trsm: the diagonal blocks by trsm.cu's kernel, the rest of B updated
/// after each by a gemm.

module;

// The launcher, defined in trsm.cu: declared in the GMF so it keeps external
// linkage (see the header).
#include "trsm_bridge.h"

module free_breakfast.recipes.trsm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.primitives; // math::idivup
import free_breakfast.core.wwr;        // gemm, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

// Every case is solved in one frame: M Y = alpha C, M lower or upper and of
// order k, C k x ncols. On the left, M is op(A) and C is B. On the right,
// X op(A) = alpha B is op(A)^T X^T = alpha B^T, so M is op(A)^T and C is B^T.
// A lower M is solved block by block from the top, an upper one from the
// bottom; after each diagonal block the rows of C still to come lose M's
// off-diagonal block times the rows just solved (a gemm, written below in
// terms of A and B as stored).
//
// alpha is applied once per row of C: the first diagonal block takes s = alpha,
// and the first gemm scales every other row by beta = alpha, so the later blocks
// and gemms take 1.
template <real_fp T>
Status trsm(DeviceResources &res, const wwr::wwrblasSideMode_t side,
            const wwr::wwrblasFillMode_t fill, const wwr::wwrblasOperation_t op,
            const wwr::wwrblasDiagType_t diag, const int32_t m, const int32_t n, const T alpha,
            const T *a, const int32_t lda, T *b, const int32_t ldb) {
  const bool left = side == wwr::WWRBLAS_SIDE_LEFT;
  if (!left && side != wwr::WWRBLAS_SIDE_RIGHT) {
    return err_config("trsm: side must be WWRBLAS_SIDE_LEFT or _RIGHT");
  }
  if (fill != wwr::WWRBLAS_FILL_MODE_LOWER && fill != wwr::WWRBLAS_FILL_MODE_UPPER) {
    return err_config("trsm: fill must be WWRBLAS_FILL_MODE_LOWER or _UPPER");
  }
  if (op != wwr::WWRBLAS_OP_N && op != wwr::WWRBLAS_OP_T && op != wwr::WWRBLAS_OP_C) {
    return err_config("trsm: op must be WWRBLAS_OP_N, _T or _C");
  }
  if (diag != wwr::WWRBLAS_DIAG_NON_UNIT && diag != wwr::WWRBLAS_DIAG_UNIT) {
    return err_config("trsm: diag must be WWRBLAS_DIAG_NON_UNIT or _UNIT");
  }
  if (m < 0 || n < 0) {
    return err_config("trsm: m = {} and n = {} must be non-negative", m, n);
  }
  const int32_t k = left ? m : n;
  if (lda < std::max(1, k) || ldb < std::max(1, m)) {
    return err_config("trsm: lda = {} must be at least max(1, {}) and ldb = {} at least max(1, {})",
                      lda, k, ldb, m);
  }
  if (m == 0 || n == 0) {
    return {};
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const bool trans = op != wwr::WWRBLAS_OP_N;
  // One right-hand side is a vector: the library's trsv, which takes no
  // workspace on either backend, in place of a launch and a gemm per block.
  // On the right, x op(A) = alpha b is op(A)^T x^T = alpha b^T, over b's
  // stride. alpha = 0 stays below, where A and B are not read.
  if ((left ? n : m) == 1 && alpha != T{0}) {
    const int32_t inc = left ? 1 : ldb;
    if (alpha != kOne<T>) {
      FBF_TRY(gpu_status(wwr::scal<T, int32_t>(res.blas().get(), k, &alpha, b, inc)));
    }
    const wwr::wwrblasOperation_t vector_op = left == trans ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;
    return gpu_status(
        wwr::trsv<T, int32_t>(res.blas().get(), fill, vector_op, diag, k, a, lda, b, inc));
  }
  const bool lower_fill = fill == wwr::WWRBLAS_FILL_MODE_LOWER;
  // op(A) is lower for (lower, N) and (upper, T); M is op(A) on the left and
  // its transpose on the right.
  const bool lower = left ? lower_fill != trans : lower_fill == trans;
  // M(p, q) is A(q, p) where M is op(A)^T on the right or A^T on the left.
  const bool a_trans = left ? trans : !trans;
  const bool unit = diag == wwr::WWRBLAS_DIAG_UNIT;
  const int32_t ncols = left ? n : m;
  const auto la = static_cast<int64_t>(lda);
  const auto lb = static_cast<int64_t>(ldb);
  // Row r of C: row r of B on the left, column r on the right.
  const auto c_row = [&](const int32_t r) { return b + (left ? r : r * lb); };

  const int32_t blocks = math::idivup(k, device::kTrsmBlock);
  for (int32_t step = 0; step < blocks; ++step) {
    const int32_t block = lower ? step : blocks - 1 - step;
    const int32_t k0 = block * device::kTrsmBlock;
    const int32_t jb = std::min(device::kTrsmBlock, k - k0);
    // alpha = 0 keeps s = 0 on every block, which zeroes it unread.
    const T scale = step == 0 || alpha == T{0} ? alpha : kOne<T>;
    FBF_TRY(gpu_status(device::trsm_diagonal_block(stream, jb, ncols, scale, a + k0 + (k0 * la), la,
                                                   a_trans, !lower, unit, c_row(k0), lb, !left)));
    // alpha = 0 zeroed this block without reading it, and zeroes the rest
    // block by block the same way: no gemm, which would read A.
    if (alpha == T{0}) {
      continue;
    }
    // The rows still to come: below the block for a lower M, above it for an
    // upper one.
    const int32_t r0 = lower ? k0 + jb : 0;
    const int32_t rows = lower ? k - r0 : k0;
    if (rows == 0) {
      continue;
    }
    // M(rest, block) in A's storage. On the left it is op(A)(rest, block): A at
    // (rest, block) under N, A at (block, rest) transposed under T. On the
    // right it is op(A)(block, rest)^T, so the gemm multiplies by
    // op(A)(block, rest): A at (block, rest) under N, A at (rest, block)
    // transposed under T.
    const bool rest_first = left != trans;
    const T *const off = rest_first ? a + r0 + (k0 * la) : a + k0 + (r0 * la);
    const wwr::wwrblasOperation_t a_op = trans ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;
    const T *const beta = step == 0 ? &alpha : &kOne<T>;
    if (left) {
      // B(rest, :) = beta B(rest, :) - op(A)(rest, block) B(block, :)
      FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(res.blas().get(), a_op, wwr::WWRBLAS_OP_N, rows, n,
                                               jb, &kNegOne<T>, off, lda, c_row(k0), ldb, beta,
                                               c_row(r0), ldb)));
    } else {
      // B(:, rest) = beta B(:, rest) - B(:, block) op(A)(block, rest)
      FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(res.blas().get(), wwr::WWRBLAS_OP_N, a_op, m, rows,
                                               jb, &kNegOne<T>, c_row(k0), ldb, off, lda, beta,
                                               c_row(r0), ldb)));
    }
  }
  return {};
}

template Status trsm<float>(DeviceResources &, wwr::wwrblasSideMode_t, wwr::wwrblasFillMode_t,
                            wwr::wwrblasOperation_t, wwr::wwrblasDiagType_t, int32_t, int32_t,
                            float, const float *, int32_t, float *, int32_t);
template Status trsm<double>(DeviceResources &, wwr::wwrblasSideMode_t, wwr::wwrblasFillMode_t,
                             wwr::wwrblasOperation_t, wwr::wwrblasDiagType_t, int32_t, int32_t,
                             double, const double *, int32_t, double *, int32_t);

} // namespace free_breakfast
