/// @file
/// @brief The density-fitting factor B = L^-1 (mn|P), M = L L^T the Coulomb
/// metric, on the GPU: one potrf and one trsm, in place.
///
/// With B, (mn|ls) ~ sum_Q B_Q,mn B_Q,ls = (mn|P) M^-1 (P|ls), so J and K are
/// products over B alone. The README has the layout and its size.

export module free_breakfast.integrals.eri.density_fitting.fit;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief Fit the three-center tensor over the metric in place: @p metric
/// becomes its Cholesky factor L (M = L L^T) and @p b becomes B = L^-1 (mn|P).
///
/// @p metric is the naux x naux Coulomb metric (P|Q), column-major and packed;
/// only its lower triangle is read, and on success that triangle holds L (the
/// strict upper triangle is left as it was). @p b is the naux x nao^2 matrix
/// (P|mn), column-major with leading dimension naux: element (P, n + nao m)
/// at P + naux (n + nao m), so P is fastest, then n, then m ([P][n][m]). On
/// success it holds B in the same layout, [Q][n][m]. (mn|P) is symmetric in m
/// and n, so which of the two is slower is the caller's to choose; B keeps it.
///
/// The metric is not regularized: a factorization that meets a leading minor
/// that is not positive is an error, not an eigen-decomposition fallback.
///
/// potrf's work comes from res.device_workspace(), which grows if it is
/// smaller; res.info() takes potrf's flag. Neither @p metric nor @p b may
/// point into res.device_workspace(). Everything is queued on res.stream()
/// with res.blas() in host pointer mode (its default), and the call waits on
/// the stream once, to read potrf's flag, before the trsm is queued; it does
/// not wait for the trsm. The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// @param res     Supplies the stream, the BLAS and solver handles, the
///                scratch workspace and potrf's flag.
/// @param nao     Orbital basis functions: @p b has nao^2 columns.
/// @param naux    Fitting functions: the order of @p metric and the rows of
///                @p b. nao = 0 or naux = 0 is an empty fit and succeeds.
/// @param metric  In: (P|Q). Out: L in the lower triangle.
/// @param b       In: (P|mn). Out: B = L^-1 (P|mn).
/// @return Success; InvalidConfig for a negative count, or nao^2 past
///         int32_t; Numerical if the metric is not positive definite, naming
///         the leading minor potrf stopped at (its pivot, 1-based); or the
///         GpuRuntime error of the first runtime, solver or BLAS call that
///         failed.
[[nodiscard]] Status density_fit(DeviceResources &res, int32_t nao, int32_t naux, double *metric,
                                 double *b);

} // namespace free_breakfast
