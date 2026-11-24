/// @file
/// @brief The triangular solve op(A) X = alpha B, or X op(A) = alpha B, on the
/// GPU, in place: blocked, its diagonal blocks in shared memory and the
/// updates between them over gemm.
///
/// The README has why this is not the BLAS library's trsm.

export module free_breakfast.recipes.trsm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief BLAS's trsm: B <- alpha op(A)^-1 B (@p side WWRBLAS_SIDE_LEFT) or
/// B <- alpha B op(A)^-1 (WWRBLAS_SIDE_RIGHT), in place, for a triangular A.
///
/// B is m x n with leading dimension @p ldb. A is triangular of order k, m on
/// the left and n on the right, with leading dimension @p lda; @p fill says
/// which triangle holds it, and only that triangle is read. @p op is
/// WWRBLAS_OP_N, _T or _C (on a real T, _C is _T). With WWRBLAS_DIAG_UNIT the
/// diagonal is taken as 1 and not read. Every matrix is column-major; A and B
/// must not overlap. With @p alpha zero, B becomes zero and neither A nor B's
/// old values are read, as in the reference BLAS.
///
/// A singular A is not detected: as in BLAS, a zero pivot gives Inf or NaN.
///
/// Everything is queued on res.stream(), with res.blas() in host pointer mode
/// (its default), and nothing is synchronized. The library's trsm workspace is
/// never touched, and nothing is drawn from res.device_workspace(), so A or B
/// may point into it. The device res.dev_idx() names is made current for the
/// call and the previous one restored.
///
/// @return Success, with nothing launched when m or n is 0; InvalidConfig for
///         a negative m or n, an @p lda below max(1, k) or an @p ldb below
///         max(1, m), or a @p side, @p fill, @p op or @p diag that is none of
///         its values; or the GpuRuntime error of the first launch or BLAS call
///         that failed.
template <real_fp T>
[[nodiscard]] Status trsm(DeviceResources &res, wwr::wwrblasSideMode_t side,
                          wwr::wwrblasFillMode_t fill, wwr::wwrblasOperation_t op,
                          wwr::wwrblasDiagType_t diag, int32_t m, int32_t n, T alpha, const T *a,
                          int32_t lda, T *b, int32_t ldb);

} // namespace free_breakfast
