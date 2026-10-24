/// @file
/// @brief The GPU sigma's launchers, declared once for the module that calls
/// them and the device TU that defines them.
///
/// Included by sigma.cpp in its GLOBAL MODULE FRAGMENT, and by ci_sigma.cu
/// directly: a declaration in a module's purview gets module linkage and can
/// never bind to a definition compiled in a plain TU, which a .cu is (see
/// src/integrals/one_electron/overlap/overlap_bridge.h).
///
/// The layout every launcher shares:
///
/// - A determinant (I_α, I_β) is element I_α · n_beta + I_β of a CI vector,
///   n_det = n_alpha · n_beta long; a block of vectors is column-major with
///   leading dimension n_det. A *row* is one (vector j, determinant I) pair,
///   row j · n_det + I, so a block of k vectors has k · n_det rows.
/// - An orbital pair (t, u) is packed as hi (hi + 1) / 2 + lo, lo = min(t, u),
///   hi = max(t, u): n_pairs = n (n + 1) / 2.
/// - A link table holds a StringSet's replacement lists, `per` links per
///   string, three int32_t per link: the target string's address, the packed
///   pair (p, q), and the sign of a†_p a_q.
///
/// Each returns the launch status and does not synchronize.

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

/// @brief The packed D of rows [@p row_begin, @p row_begin + @p rows):
/// D⁺_K[pair(v, w)] = (E_vw c)_K + (E_wv c)_K for v ≠ w, (E_vv c)_K on the
/// diagonal, gathered over K's α and β links.
///
/// @p d is rows × n_pairs, column-major with leading dimension @p rows, and is
/// accumulated into: it must be zero on entry. @p c is the block of vectors.
[[nodiscard]] wwr::wwrError_t ci_gather_d(wwr::wwrStream_t stream, int32_t row_begin, int32_t rows,
                                          int32_t n_det, int32_t n_beta, const int32_t *alpha_links,
                                          int32_t alpha_per, const int32_t *beta_links,
                                          int32_t beta_per, const double *c, double *d);

/// @brief σ for every one of the @p rows rows, from G:
/// σ_I = e_core c_I + G_I[n_pairs] + Σ s · G_J[pair(p, q)] over I's α and β
/// links (p, q, J, s).
///
/// @p g is rows × (n_pairs + 1), column-major with leading dimension @p rows:
/// column pair(t, u) is G_K[tu] = ½ Σ_vw (tu|vw) D_K[vw], and the last column
/// the one-electron term Σ_vw k_vw D_K[vw]. @p sigma has @p c's layout.
[[nodiscard]] wwr::wwrError_t ci_sigma_from_g(wwr::wwrStream_t stream, int32_t rows, int32_t n_det,
                                              int32_t n_beta, int32_t n_pairs,
                                              const int32_t *alpha_links, int32_t alpha_per,
                                              const int32_t *beta_links, int32_t beta_per,
                                              double e_core, const double *c, const double *g,
                                              double *sigma);

/// @brief diag(H) for the @p n_alpha · @p n_beta determinants, by host_diagonal's
/// rule and in its order.
///
/// @p strings_alpha and @p strings_beta are the two sets' bit patterns, by
/// address. @p terms is n + 2 n² doubles: h_ii, then (ii|jj) and (ij|ji), each
/// n × n row-major over (i, j).
[[nodiscard]] wwr::wwrError_t ci_diagonal(wwr::wwrStream_t stream, int32_t n, int32_t n_alpha,
                                          int32_t n_beta, const uint64_t *strings_alpha,
                                          const uint64_t *strings_beta, double e_core,
                                          const double *terms, double *diagonal);

} // namespace free_breakfast::device
