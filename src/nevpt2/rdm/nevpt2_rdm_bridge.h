/// @file
/// @brief nevpt2::rdms' launchers, declared once for the module that calls
/// them and the device TU that defines them: the eri's conversion, the
/// per-tile R and L2, and fdm2 and the wedge after the tile loop.
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

} // namespace free_breakfast::nevpt2::device
