/// @file
/// @brief The density-fitting factor B = L^-1 (mn|P), M = L L^T the Coulomb
/// metric, on the GPU: one Cholesky factorization and one trsm, in place; or,
/// for a near-singular metric and only when asked, B = M^-1/2 (mn|P) over the
/// eigenmodes above a threshold.
///
/// With B, (mn|ls) ~ sum_Q B_Q,mn B_Q,ls = (mn|P) M^-1 (P|ls), so J and K are
/// products over B alone. The README has the layout and its size.

export module free_breakfast.integrals.eri.density_fitting.fit;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.core.footprint;

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
/// By default (@p lindep = 0) the metric is not regularized: a factorization
/// that meets a leading minor that is not positive is an error. A positive
/// @p lindep instead fits over M's eigenmodes above it (PySCF's DF lindep):
/// @p metric becomes X = M^-1/2 = U Lambda^-1/2 U^T, a mode with eigenvalue at
/// or below @p lindep contributing 0 (recipes/orthogonalize), and @p b becomes
/// B = X (P|mn), the same shape, so B^T B is (mn|P) M^+ (P|ls) over the kept
/// modes. Its B is a rotation of the Cholesky path's on a metric that drops
/// nothing, and X is not L: a consumer that reads L (the DF gradients and
/// Hessians) cannot take this fit. The README has which consumers do.
///
/// The factorization is recipes/cholesky's, not the dense solver's potrf
/// (its README has why); it takes no scratch, and res.info() takes its flag.
/// Everything is queued on res.stream() with res.blas() in host pointer mode
/// (its default), and the call waits on the stream once, to read the flag,
/// before the trsm is queued; it does not wait for the trsm. With @p lindep,
/// orthogonalize's waits replace that one, its scratch and an naux x nao
/// product are res.device_workspace()'s, and a copy of M is a pool buffer.
/// The device res.dev_idx() names is made current for the call and the
/// previous one restored.
///
/// @param res     Supplies the stream, the BLAS handle and the
///                factorization's flag.
/// @param nao     Orbital basis functions: @p b has nao^2 columns.
/// @param naux    Fitting functions: the order of @p metric and the rows of
///                @p b. nao = 0 or naux = 0 is an empty fit and succeeds.
/// @param metric  In: (P|Q). Out: L in the lower triangle; with @p lindep,
///                X, the whole matrix.
/// @param b       In: (P|mn). Out: B = L^-1 (P|mn); with @p lindep, X (P|mn).
/// @param lindep  0, the default: the Cholesky path. Positive: the eigenvalue
///                at or below which a mode of M is dropped.
/// @return Success; InvalidConfig for a negative count, nao^2 past int32_t, or
///         a negative or NaN @p lindep; Numerical if the metric is not
///         positive definite, naming the leading minor the factorization
///         stopped at (its pivot, 1-based), or with @p lindep if its
///         eigendecomposition did not converge; or the GpuRuntime error of the
///         first runtime, launch, solver or BLAS call that failed.
[[nodiscard]] Status density_fit(DeviceResources &res, int32_t nao, int32_t naux, double *metric,
                                 double *b, double lindep = 0.0);

/// @brief What @ref density_fit takes: nothing on the Cholesky path; with
/// @p lindep, the metric's copy and @ref orthogonalize_footprint's scratch,
/// and the device workspace, the larger of orthogonalize's and one slab's.
/// Empty for nao or naux 0. @p syevd_lwork is syevd's work at order naux
/// (@ref syevd_lwork), read only with @p lindep; the Cholesky path passes 0.
[[nodiscard]] MemoryFootprint density_fit_footprint(int32_t nao, int32_t naux, size_t syevd_lwork,
                                                    double lindep = 0.0);

} // namespace free_breakfast
