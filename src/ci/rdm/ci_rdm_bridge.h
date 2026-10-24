/// @file
/// @brief rdm12's one launcher, declared once for the module that calls it and
/// the device TU that defines it: the gather of the one-replacement
/// intermediate D.
///
/// Included by rdm.cpp in its GLOBAL MODULE FRAGMENT, and by ci_rdm.cu
/// directly: a declaration in a module's purview gets module linkage and can
/// never bind to a definition compiled in a plain TU, which a .cu is (see
/// src/integrals/one_electron/overlap/overlap_bridge.h).
///
/// The layout is the GPU sigma's (see ../sigma/ci_sigma_bridge.h): a
/// determinant (K_α, K_β) is element K_α · n_beta + K_β of a CI vector. A link
/// table holds a StringSet's replacement lists, `per` links per string, three
/// int32_t per link a†_p a_q: the target string's address, the *unpacked* pair
/// index p · n + q (D is not symmetric in its pair, as sigma's is), and the
/// sign.
///
/// It returns the launch status and does not synchronize.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see src/integrals/one_electron/overlap/overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief D for the determinants K in [@p begin, @p begin + @p rows) of the
/// one CI vector @p c: D_{pq,K} = ⟨c| E_pq |K⟩ = Σ s · c_J over K's α and β
/// links (p, q, J, s).
///
/// @p d is rows × n², column-major with leading dimension @p rows (column
/// p · n + q), and is accumulated into: it must be zero on entry.
[[nodiscard]] wwr::wwrError_t ci_rdm_gather_d(wwr::wwrStream_t stream, int32_t begin, int32_t rows,
                                              int32_t n_beta, const int32_t *alpha_links,
                                              int32_t alpha_per, const int32_t *beta_links,
                                              int32_t beta_per, const double *c, double *d);

} // namespace free_breakfast::device
