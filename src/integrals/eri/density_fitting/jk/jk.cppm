/// @file
/// @brief Density-fitted J and K from the fitted factor B on the GPU: J from
/// the density by two gemvs, K from a factor of it by one batched gemm and one
/// syrk, or from two factors of it by two batched gemms and one gemm.
///
/// B is @ref density_fit's output, [Q][n][m] with Q fastest, so that
/// (mn|ls) ~ sum_Q B_Q,mn B_Q,ls. The README has the layouts and the scratch.

export module free_breakfast.integrals.eri.density_fitting.jk;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief The density-fitted Coulomb matrix J_mn = sum_Q B_Q,mn gamma_Q,
/// gamma_Q = sum_ls B_Q,ls D_ls, into @p j: two gemvs over B.
///
/// @p b is B as @ref density_fit leaves it, the naux x nao^2 matrix with
/// element (Q, n + nao m) at Q + naux (n + nao m). @p density and @p j are
/// nao x nao, column-major and packed; D is read as the vector of its nao^2
/// elements, so any D gives sum_ls (mn|ls) D_ls, symmetric or not. @p j is
/// overwritten. gamma, naux doubles, comes from res.device_workspace(), which
/// grows if it is smaller; none of @p b, @p density and @p j may point into
/// it, and @p j must not overlap @p b or @p density.
///
/// Queued on res.stream() with res.blas() in host pointer mode (its default),
/// with no host synchronization. The device res.dev_idx() names is made
/// current for the call and the previous one restored.
///
/// @param res      Supplies the stream, the BLAS handle and the scratch.
/// @param nao      Orbital basis functions. nao = 0 succeeds and writes nothing.
/// @param naux     Fitting functions: the rows of @p b. naux = 0 gives J = 0.
/// @param b        B, [Q][n][m].
/// @param density  D.
/// @param j        Out: J.
/// @return Success; InvalidConfig for a negative count, or nao^2 past int32_t;
///         or the GpuRuntime error of the first runtime or BLAS call that
///         failed, which leaves @p j unspecified.
[[nodiscard]] Status df_coulomb(DeviceResources &res, int32_t nao, int32_t naux, const double *b,
                                const double *density, double *j);

/// @brief The density-fitted exchange matrix K_mn = sum_ls (ml|ns) D_ls of the
/// density D = 2 C C^T, from the factor @p c and never from D, into @p k:
/// X_m = B_m C, one batched gemm over m, then K = 2 X^T X, one syrk.
///
/// @p b is B as for @ref df_coulomb; its block B_m, the naux x nao matrix of
/// columns n + nao m, sits at naux nao m. @p c is nao x @p nocc,
/// column-major and packed: an SCF's occupied orbitals, or any factor of a
/// positive semidefinite D. Two factors of one D give the same K to rounding.
/// X is stored [Q][i][m], element (Q, i, m) at Q + naux (i + nocc m), so it
/// reads as the (naux nocc) x nao matrix whose syrk is K / 2. @p k is nao x
/// nao, column-major and packed, overwritten, and exactly symmetric: syrk
/// writes the lower triangle, and the upper one is its transpose bit for bit.
///
/// The scratch is X, naux nocc nao doubles, from res.device_workspace(),
/// which grows if it is smaller; none of @p b, @p c and @p k may point into it, and @p k must
/// not overlap @p b or @p c.
///
/// Queued on res.stream() with res.blas() in host pointer mode (its default),
/// with no host synchronization. The device res.dev_idx() names is made
/// current for the call and the previous one restored.
///
/// @param res   Supplies the stream, the BLAS handle and the scratch.
/// @param nao   Orbital basis functions. nao = 0 succeeds and writes nothing.
/// @param naux  Fitting functions: the rows of @p b. naux = 0 gives K = 0.
/// @param nocc  Columns of @p c. nocc = 0 gives K = 0.
/// @param b     B, [Q][n][m].
/// @param c     C, with D = 2 C C^T.
/// @param k     Out: K.
/// @return Success; InvalidConfig for a negative count, or naux nocc past
///         int32_t; or the GpuRuntime error of the first runtime or BLAS call
///         that failed, which leaves @p k unspecified.
[[nodiscard]] Status df_exchange(DeviceResources &res, int32_t nao, int32_t naux, int32_t nocc,
                                 const double *b, const double *c, double *k);

/// @brief The density-fitted exchange matrix K_mn = sum_ls (ml|ns) D_ls of the
/// two-factor density D = L R^T, from @p left and @p right and never from D,
/// into @p k: X_m = B_m L and Y_m = B_m R, two batched gemms over m, then
/// K = X^T Y, one gemm.
///
/// For a D the one-factor form cannot express: indefinite, or not symmetric
/// (a response or transition density). L R^T is not symmetrized, and neither
/// is K: K[L R^T]^T = K[R L^T]. So a symmetric D = L R^T + R L^T has
/// K = K[L R^T] + K[L R^T]^T, and a sum of two-factor terms is one call over
/// the factors laid side by side, L = [L_1 L_2], R = [R_1 R_2]. K[C C^T] here
/// is half the one-factor @c df_exchange of C, which is K[2 C C^T].
///
/// @p b is B as for @ref df_coulomb. @p left and @p right are nao x
/// @p rank, column-major and packed, and may be the same pointer. X and Y are
/// stored as the one-factor form's X, [Q][i][m], element (Q, i, m) at
/// Q + naux (i + rank m), so each reads as an (naux rank) x nao matrix and K is
/// their product. @p k is nao x nao, column-major and packed, overwritten.
///
/// The scratch is X and Y, naux rank nao doubles each, from
/// res.device_workspace(), which grows if it is smaller; none of @p b,
/// @p left, @p right and @p k may point into it, and @p k must not overlap
/// @p b, @p left or @p right.
///
/// Queued on res.stream() with res.blas() in host pointer mode (its default),
/// with no host synchronization. The device res.dev_idx() names is made
/// current for the call and the previous one restored.
///
/// @param res    Supplies the stream, the BLAS handle and the scratch.
/// @param nao    Orbital basis functions. nao = 0 succeeds and writes nothing.
/// @param naux   Fitting functions: the rows of @p b. naux = 0 gives K = 0.
/// @param rank   Columns of @p left and of @p right. rank = 0 gives K = 0.
/// @param b      B, [Q][n][m].
/// @param left   L, with D = L R^T.
/// @param right  R.
/// @param k      Out: K.
/// @return Success; InvalidConfig for a negative count, or naux rank past
///         int32_t; or the GpuRuntime error of the first runtime or BLAS call
///         that failed, which leaves @p k unspecified.
[[nodiscard]] Status df_exchange(DeviceResources &res, int32_t nao, int32_t naux, int32_t rank,
                                 const double *b, const double *left, const double *right,
                                 double *k);

} // namespace free_breakfast
