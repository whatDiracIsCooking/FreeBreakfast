/// @file
/// @brief rdm12's launchers, declared once for the module that calls them and
/// the device TU that defines them: the gather of the one-replacement
/// intermediate D, and Γ's assembly.
///
/// Included by rdm.cpp in its GLOBAL MODULE FRAGMENT, and by ci_rdm.cu
/// directly: a declaration in a module's purview gets module linkage and can
/// never bind to a definition compiled in a plain TU, which a .cu is (see
/// src/integrals/one_electron/overlap/overlap_bridge.h).
///
/// The layout is the GPU sigma's (see ../sigma/ci_sigma_bridge.h): a
/// determinant (K_α, K_β) is element K_α · n_beta + K_β of a CI vector. A link
/// table is a DeviceStringSet's replacement lists, `per` Replacements per
/// string (replacement.h), of which D's column is the *unpacked* pair index
/// p · n + q (D is not symmetric in its pair, as sigma's is).
///
/// It returns the launch status and does not synchronize.

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

namespace free_breakfast::ci::device {

/// @brief D for the determinants K in [@p begin, @p begin + @p rows) of the
/// one CI vector @p c: D_{pq,K} = ⟨c| E_pq |K⟩ = Σ s · c_J over K's α and β
/// links (p, q, J, s).
///
/// With @p symmetric, @p c is in ci::Layout::symmetric (see ../sigma/
/// ci_symmetric.h), one set for both spins (n_beta = n_alpha), and K runs
/// over its triangle elements: D_K is of the unpacked c, and each off-diagonal
/// row (K_α > K_β) is scaled by √2, so that Σ over the triangle of
/// D_K D_Kᵀ, and of D_K times the packed c, is the full layout's sum over
/// every determinant.
///
/// @p d is rows × n², column-major with leading dimension @p rows (column
/// p · n + q), and is accumulated into: it must be zero on entry.
[[nodiscard]] wwr::wwrError_t rdm_gather_d(wwr::wwrStream_t stream, bool symmetric, int32_t begin,
                                           int32_t rows, int32_t n, int32_t n_beta,
                                           const Replacement *alpha_links, int32_t alpha_per,
                                           const Replacement *beta_links, int32_t beta_per,
                                           const double *c, double *d);

/// @brief Γ_tuvw = M_{tu,wv} − δ_uv γ_tw into @p gamma2, n⁴ row-major (t, u,
/// v, w), from @p m, n² × n² column-major with only its lower triangle
/// written (a syrk's), and @p gamma, n × n.
[[nodiscard]] wwr::wwrError_t rdm_assemble(wwr::wwrStream_t stream, int32_t n, const double *m,
                                           const double *gamma, double *gamma2);

} // namespace free_breakfast::ci::device
