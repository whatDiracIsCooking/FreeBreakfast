/// @file
/// @brief The fitted factor B in the MO basis, over arbitrary column ranges of
/// the orbitals C: the half transform B_P,mp = sum_n B_P,mn C_np, one batched
/// gemm, and the full one B_P,pq = sum_mn B_P,mn C_mp C_nq, that and a gemm.
///
/// B is @ref density_fit's output, [Q][n][m] with Q fastest. L^-1 touches only
/// P and C only m and n, so fitting and transforming commute, and
/// (pq|rs) ~ sum_Q B_Q,pq B_Q,rs. The README has the layouts.

export module free_breakfast.integrals.eri.density_fitting.mo;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief The columns [first, first + count) of an nao x nmo C.
struct OrbitalRange {
  int32_t first = 0;
  int32_t count = 0;
};

/// @brief The half transform X_Q,pm = sum_n B_Q,nm C_n,p over p in @p rows,
/// into @p out: X_m = B_m C_rows, one batched gemm over m.
///
/// @p b is B as @ref density_fit leaves it, the naux x nao^2 matrix with
/// element (Q, n + nao m) at Q + naux (n + nao m). @p c is nao x @p nmo,
/// column-major and packed. @p out is X, [Q][p][m]: element (Q, p, m) at
/// Q + naux (p + |rows| m), the (naux |rows|) x nao matrix that is
/// @c df_exchange's scratch X for @p rows the occupied block. It is
/// overwritten, and must not overlap @p b or @p c. The fast AO index is the
/// one contracted; (mn|P) is symmetric in m and n, so either may be.
///
/// Queued on res.stream() with res.blas() in host pointer mode (its default),
/// with no host synchronization and no scratch. The device res.dev_idx()
/// names is made current for the call and the previous one restored.
///
/// @param res   Supplies the stream and the BLAS handle.
/// @param naux  Fitting functions: the rows of @p b.
/// @param nao   Orbital basis functions: @p c's rows.
/// @param nmo   @p c's columns.
/// @param b     B, [Q][n][m].
/// @param c     C.
/// @param rows  The columns of C transformed onto: p.
/// @param out   Out: X, naux |rows| nao doubles.
/// @return Success, writing nothing when @p out is empty (naux, nao or
///         |rows| zero); InvalidConfig for a negative count, @p rows outside
///         [0, nmo), or naux |rows| past int32_t; or the GpuRuntime error of
///         the BLAS call, which leaves @p out unspecified.
[[nodiscard]] Status b_half(DeviceResources &res, int32_t naux, int32_t nao, int32_t nmo,
                            const double *b, const double *c, OrbitalRange rows, double *out);

/// @brief B in the MO basis, B_Q,pq = sum_mn B_Q,mn C_m,p C_n,q over p in
/// @p rows and q in @p cols, into @p out: @ref b_half over @p rows, then its
/// slow AO index contracted with C_cols, one gemm.
///
/// @p b, @p c, @p nao and @p nmo are as for @ref b_half. @p out is [Q][p][q]:
/// element (Q, p, q) at Q + naux (p + |rows| q), the (naux |rows|) x |cols|
/// matrix with leading dimension naux |rows|. So at fixed p it is the
/// naux x |cols| matrix at Q + naux p, leading dimension naux |rows|, and
/// consecutive p are naux apart. It is overwritten, and must not overlap @p b
/// or @p c. (pq|rs) ~ sum_Q B_Q,pq B_Q,rs.
///
/// The scratch is @ref b_half's X, naux |rows| nao doubles, from
/// res.device_workspace(), which grows if it is smaller; none of @p b, @p c
/// and @p out may point into it. A workspace that would have to grow past the
/// device's free memory is refused up front, since the allocation would abort
/// (CreatePolicy) rather than return.
///
/// Queued on res.stream() with res.blas() in host pointer mode (its default),
/// with no host synchronization. The device res.dev_idx() names is made
/// current for the call and the previous one restored.
///
/// @param res   Supplies the stream, the BLAS handle and the scratch.
/// @param naux  Fitting functions: the rows of @p b.
/// @param nao   Orbital basis functions: @p c's rows.
/// @param nmo   @p c's columns.
/// @param b     B, [Q][n][m].
/// @param c     C.
/// @param rows  The columns of C on the fast index: p.
/// @param cols  The columns of C on the slow index: q.
/// @param out   Out: B in the MO basis, naux |rows| |cols| doubles.
/// @return Success, writing nothing when @p out is empty (naux, |rows| or
///         |cols| zero); InvalidConfig for a negative count, a range outside
///         [0, nmo), or naux |rows| past int32_t; Unsupported for scratch
///         past the device's free memory, naming the bytes; or the GpuRuntime
///         error of the first runtime or BLAS call that failed, which leaves
///         @p out unspecified.
[[nodiscard]] Status b_mo(DeviceResources &res, int32_t naux, int32_t nao, int32_t nmo,
                          const double *b, const double *c, OrbitalRange rows, OrbitalRange cols,
                          double *out);

} // namespace free_breakfast
