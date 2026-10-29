/// @file
/// @brief gemm_lt: the descriptor, the two operand layouts, the one-algorithm
/// heuristic and wwrblasLtMatmul.

module free_breakfast.experimental.blas_lt;

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import wwr.extension.blaslt;

namespace free_breakfast {

namespace ext = wwr::extension;

// Not exported, so module-private; named for this module alone, as a module
// without anonymous namespaces requires (docs/style.md "Module-private code").
using LtCreate = CreatePolicy<wwr::wwrblasLtStatus_t>;
using LtDestroy = DestroyPolicy<wwr::wwrblasLtStatus_t>;
using LtMatmulDesc = ext::MatmulDesc<LtCreate, LtDestroy>;
using LtMatrixLayout = ext::MatrixLayout<LtCreate, LtDestroy>;
using LtMatmulPreference = ext::MatmulPreference<LtCreate, LtDestroy>;

// The layout of an operand as it sits in memory. Lt describes the stored
// matrix, not the logical one, so a transposed operand's extents are swapped
// against wwr::gemm's m/n/k.
LtMatrixLayout lt_layout(const wwr::wwrblasOperation_t op, const int64_t rows_untransposed,
                         const int64_t cols_untransposed, const int64_t ld) {
  const bool transposed = op != wwr::WWRBLAS_OP_N;
  const auto rows = static_cast<uint64_t>(transposed ? cols_untransposed : rows_untransposed);
  const auto cols = static_cast<uint64_t>(transposed ? rows_untransposed : cols_untransposed);
  return LtMatrixLayout{wwr::WWRBLAS_R_64F, rows, cols, ld};
}

LtResult gemm_lt(DeviceResources &res, BlasLtHandle &lt, const wwr::wwrblasOperation_t op_a,
                 const wwr::wwrblasOperation_t op_b, const int64_t m, const int64_t n,
                 const int64_t k, const double *const a, const int64_t lda, const double *const b,
                 const int64_t ldb, double *const c, const int64_t ldc, void *const workspace,
                 const size_t workspace_bytes) {
  LtMatmulDesc desc{wwr::WWRBLAS_COMPUTE_64F, wwr::WWRBLAS_R_64F};
  for (const auto [attribute, op] : {std::pair{wwr::WWRBLASLT_MATMUL_DESC_TRANSA, op_a},
                                     std::pair{wwr::WWRBLASLT_MATMUL_DESC_TRANSB, op_b}}) {
    const wwr::wwrblasLtStatus_t status =
        wwr::wwrblasLtMatmulDescSetAttribute(desc.get(), attribute, &op, sizeof(op));
    if (status != wwr::WWRBLAS_STATUS_SUCCESS) {
      return {0, status};
    }
  }

  const LtMatrixLayout a_layout = lt_layout(op_a, m, k, lda);
  const LtMatrixLayout b_layout = lt_layout(op_b, k, n, ldb);
  const LtMatrixLayout c_layout = lt_layout(wwr::WWRBLAS_OP_N, m, n, ldc);

  LtMatmulPreference preference{};
  {
    const wwr::wwrblasLtStatus_t status = wwr::wwrblasLtMatmulPreferenceSetAttribute(
        preference.get(), wwr::WWRBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
        sizeof(workspace_bytes));
    if (status != wwr::WWRBLAS_STATUS_SUCCESS) {
      return {0, status};
    }
  }

  wwr::wwrblasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0; // int-type-ok: the vendor's out-parameter type
  const wwr::wwrblasLtStatus_t search = wwr::wwrblasLtMatmulAlgoGetHeuristic(
      lt.get(), desc.get(), a_layout.get(), b_layout.get(), c_layout.get(), c_layout.get(),
      preference.get(), 1, &heuristic, &returned);
  if (search != wwr::WWRBLAS_STATUS_SUCCESS || returned == 0) {
    return {static_cast<int32_t>(returned), search};
  }

  // beta = 1: the digest accumulates into C, and D is C in place, which Lt
  // allows while Cdesc and Ddesc agree.
  const double one = 1.0;
  const wwr::wwrblasLtStatus_t status = wwr::wwrblasLtMatmul(
      lt.get(), desc.get(), &one, a, a_layout.get(), b, b_layout.get(), &one, c, c_layout.get(), c,
      c_layout.get(), &heuristic.algo, workspace, workspace_bytes, res.stream());
  return {static_cast<int32_t>(returned), status};
}

} // namespace free_breakfast
