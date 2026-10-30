/// @file
/// @brief nevpt2::rdms' and nevpt2::rdms_adjoint's launchers, declared once
/// for the module that calls them and the device TU that defines them: the
/// eri's conversion, the per-tile R and L2, fdm2 and the wedge after the tile
/// loop, and the transposes of each.
///
/// Included by rdm.cpp in its GLOBAL MODULE FRAGMENT, and by nevpt2_rdm.cu
/// directly: a declaration in a module's purview gets module linkage and can
/// never bind to a definition compiled in a plain TU, which a .cu is (see
/// src/integrals/one_electron/overlap/overlap_bridge.h).
///
/// A determinant (K_α, K_β) is element K_α · n_beta + K_β of the CI vector,
/// and a link table is a DeviceStringSet's replacement lists, `per`
/// Replacements per string (replacement.h). Every n⁴ and n⁶ tensor is
/// row-major. Each returns the launch status and does not synchronize.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"
#include "replacement.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see src/integrals/one_electron/overlap/overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::nevpt2::device {

/// @brief From @p in, (tu|vw) at t + n (u + n (v + n w)) (ActiveSpaceHamiltonian::eri),
/// write @p eri[a, b, c, d] = (ab|cd), row-major, and the consume GEMMs'
/// copy @p consume[x, a, p, q] = (ax|qp), row-major.
[[nodiscard]] wwr::wwrError_t rdm_convert_eri(wwr::wwrStream_t stream, int32_t n, const double *in,
                                              double *eri, double *consume);

/// @brief R and L2 for the @p width determinants K in [@p begin, @p begin +
/// @p width) of the CI vector @p c: R[tu, k] = ⟨K| E_tu |c⟩ and
/// L2[pqrs, k] = ⟨c| E_pq E_rs |K⟩, k = K − @p begin, each row-major with
/// leading dimension @p width. Overwrites both.
[[nodiscard]] wwr::wwrError_t rdm_produce(wwr::wwrStream_t stream, int32_t n, int32_t n_beta,
                                          const ci::Replacement *alpha_links, int32_t alpha_per,
                                          const ci::Replacement *beta_links, int32_t beta_per,
                                          const double *c, int32_t begin, int32_t width, double *r,
                                          double *l2);

/// @brief The ca order's fdm2[i, j, k, l] = Σ_ABC eri[l, C, B, A] dm3[A, B, C, k, j, i].
[[nodiscard]] wwr::wwrError_t rdm_fdm2_ca(wwr::wwrStream_t stream, int32_t n, const double *eri,
                                          const double *dm3, double *fdm2);

/// @brief The ac order's fdm2[i, j, k, l] = Σ_CEF eri[k, C, E, F] dm3[i, j, C, l, E, F].
[[nodiscard]] wwr::wwrError_t rdm_fdm2_ac(wwr::wwrStream_t stream, int32_t n, const double *eri,
                                          const double *dm3, double *fdm2);

/// @brief PySCF's wedge on a fully summed @p f3, in place, for either order:
/// for every j < i, f3[j, b, i, c, d, e] = f3[i, c, j, b, d, e], then
/// f3[j, i, i, c, d, e] += fdm2[j, c, d, e] and f3[j, b, i, j, d, e] −=
/// fdm2[i, b, d, e].
[[nodiscard]] wwr::wwrError_t rdm_wedge(wwr::wwrStream_t stream, int32_t n, const double *fdm2,
                                        double *f3);

/// @brief The wedge's transpose, for either order: from @p f3_bar, the bar
/// on the wedged f3, write @p raw_bar, the bar on f3 before the wedge
/// (zero where the first index is below the third, which the wedge
/// overwrites; f3̄[i, c, j, b, d, e] + f3̄[j, b, i, c, d, e] where it is
/// above), and @p fdm2_bar[i, b, d, e] = Σ_{j>i} f3̄[i, j, j, b, d, e] −
/// Σ_{j<i} f3̄[j, b, i, j, d, e]. Overwrites both.
[[nodiscard]] wwr::wwrError_t rdm_wedge_adjoint(wwr::wwrStream_t stream, int32_t n,
                                                const double *f3_bar, double *raw_bar,
                                                double *fdm2_bar);

/// @brief The two fdm2s' transpose onto dm3: @p dm3_bar_out = @p dm3_bar_in
/// plus Σ_l eri[l, C, B, A] fdm2̄_ca[i, j, k, l] at dm3[A, B, C, k, j, i] and
/// Σ_k eri[k, C, E, F] fdm2̄_ac[i, j, k, l] at dm3[i, j, C, l, E, F].
[[nodiscard]] wwr::wwrError_t rdm_fdm2_dm3_adjoint(wwr::wwrStream_t stream, int32_t n,
                                                   const double *eri, const double *fdm2_ca_bar,
                                                   const double *fdm2_ac_bar,
                                                   const double *dm3_bar_in, double *dm3_bar_out);

/// @brief The bar on @p in's layout, (tu|vw) at t + n (u + n (v + n w)), of
/// the conversion and the two fdm2s: @p eri_bar = the bar on eri[t, u, v, w]
/// (Σ_ijk fdm2̄_ca[i, j, k, t] dm3[w, v, u, k, j, i] + Σ_ijl fdm2̄_ac[i, j, t,
/// l] dm3[i, j, u, l, v, w]) plus @p consume_bar[u, t, w, v]. Overwrites it.
[[nodiscard]] wwr::wwrError_t rdm_eri_adjoint(wwr::wwrStream_t stream, int32_t n, const double *dm3,
                                              const double *fdm2_ca_bar, const double *fdm2_ac_bar,
                                              const double *consume_bar, double *eri_bar);

/// @brief rdm_produce's transpose over the same tile: for every determinant J
/// of the CI vector, @p y[J] += Σ_K Σ_tu R̄[tu, k] ⟨K| E_tu |J⟩ + Σ_K Σ_pqrs
/// L2̄[pqrs, k] ⟨J| E_pq E_rs |K⟩ over K in [@p begin, @p begin + @p width),
/// k = K − @p begin, gathered over J's own replacement lists. One thread per
/// J, so nothing is atomic.
[[nodiscard]] wwr::wwrError_t
rdm_produce_adjoint(wwr::wwrStream_t stream, int32_t n, int32_t n_beta, int32_t ndet,
                    const ci::Replacement *alpha_links, int32_t alpha_per,
                    const ci::Replacement *beta_links, int32_t beta_per, int32_t begin,
                    int32_t width, const double *r_bar, const double *l2_bar, double *y);

} // namespace free_breakfast::nevpt2::device
