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
/// - A link table is a DeviceStringSet's replacement lists, `per` Replacements
///   per string (replacement.h): the target string's address, the sign, and
///   the (p, q) of a†_p a_q, whose packed pair the kernels form.
/// - With @p symmetric, the vectors are in ci::Layout::symmetric (see
///   ci_symmetric.h): n_alpha = n_beta strings, n_det the triangle's
///   n_alpha (n_alpha + 1) / 2, and a row a triangle element.
///
/// Each returns the launch status and does not synchronize.

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

/// @brief The packed D of rows [@p row_begin, @p row_begin + @p rows):
/// D⁺_K[pair(v, w)] = (E_vw c)_K + (E_wv c)_K for v ≠ w, (E_vv c)_K on the
/// diagonal, gathered over K's α and β links.
///
/// @p d is rows × n_pairs, column-major with leading dimension @p rows, and is
/// accumulated into: it must be zero on entry. @p c is the block of vectors.
/// A float @p d takes each term rounded to float: an entry has at most two,
/// one link of each spin. With @p symmetric, D is of the unpacked c, and
/// with @p scale_off_diagonal too, each off-diagonal row (I_α > I_β) is
/// scaled by √2: the triangle's rows then carry the weight 2 of a determinant
/// and its mirror in a sum over the packed c (ci::Sigma::rdm1's). Without
/// @p symmetric, @p scale_off_diagonal is ignored.
[[nodiscard]] wwr::wwrError_t gather_d(wwr::wwrStream_t stream, bool symmetric, int32_t row_begin,
                                       int32_t rows, int32_t n_det, int32_t n_beta,
                                       const Replacement *alpha_links, int32_t alpha_per,
                                       const Replacement *beta_links, int32_t beta_per,
                                       bool scale_off_diagonal, const double *c, double *d);
[[nodiscard]] wwr::wwrError_t gather_d(wwr::wwrStream_t stream, bool symmetric, int32_t row_begin,
                                       int32_t rows, int32_t n_det, int32_t n_beta,
                                       const Replacement *alpha_links, int32_t alpha_per,
                                       const Replacement *beta_links, int32_t beta_per,
                                       bool scale_off_diagonal, const double *c, float *d);

/// @brief gather_d over rows [@p row_begin, @p row_begin + @p rows) of a
/// string block rather than of the vector: block row r is string
/// @p first + r / n_minor of the block's spin, the β with @p beta_major and
/// else the α, against string r % n_minor of the other, n_minor being the
/// other spin's count (@p n_alpha with @p beta_major, else @p n_beta). @p d is
/// as gather_d's. With @p symmetric (@p beta_major false, @p n_alpha =
/// @p n_beta the one set's m) a block row is a determinant of the unpacked
/// vector, of either order, its links read through the mirror; D is of the
/// unpacked c, unscaled.
[[nodiscard]] wwr::wwrError_t gather_d_block(wwr::wwrStream_t stream, bool symmetric,
                                             bool beta_major, int32_t first, int32_t row_begin,
                                             int32_t rows, int32_t n_alpha, int32_t n_beta,
                                             const Replacement *alpha_links, int32_t alpha_per,
                                             const Replacement *beta_links, int32_t beta_per,
                                             const double *c, double *d);
[[nodiscard]] wwr::wwrError_t gather_d_block(wwr::wwrStream_t stream, bool symmetric,
                                             bool beta_major, int32_t first, int32_t row_begin,
                                             int32_t rows, int32_t n_alpha, int32_t n_beta,
                                             const Replacement *alpha_links, int32_t alpha_per,
                                             const Replacement *beta_links, int32_t beta_per,
                                             const double *c, float *d);

/// @brief σ for every one of the @p rows rows, from G:
/// σ_I = e_core c_I + G_I[n_pairs] + Σ s · G_J[pair(p, q)] over I's α and β
/// links (p, q, J, s).
///
/// @p g is rows × (n_pairs + 1), column-major with leading dimension @p rows:
/// column pair(t, u) is G_K[tu] = ½ Σ_vw (tu|vw) D_K[vw], and the last column
/// the one-electron term Σ_vw k_vw D_K[vw]. @p sigma has @p c's layout. A
/// float @p g is summed in double. With @p symmetric, G_J of a J below the
/// triangle is read from its mirror's row.
[[nodiscard]] wwr::wwrError_t sigma_from_g(wwr::wwrStream_t stream, bool symmetric, int32_t rows,
                                           int32_t n_det, int32_t n_beta, int32_t n_pairs,
                                           const Replacement *alpha_links, int32_t alpha_per,
                                           const Replacement *beta_links, int32_t beta_per,
                                           double e_core, const double *c, const double *g,
                                           double *sigma);
[[nodiscard]] wwr::wwrError_t sigma_from_g(wwr::wwrStream_t stream, bool symmetric, int32_t rows,
                                           int32_t n_det, int32_t n_beta, int32_t n_pairs,
                                           const Replacement *alpha_links, int32_t alpha_per,
                                           const Replacement *beta_links, int32_t beta_per,
                                           double e_core, const double *c, const float *g,
                                           double *sigma);

/// @brief Add to @p sigma what the G of one string block (see
/// gather_d_block: strings [@p first, @p first + @p count)) serves.
///
/// @p g is the block's rows × (n_pairs + 1), column-major with leading
/// dimension count · n_minor, as sigma_from_g's. A determinant (a, b)'s β
/// links reach (a, b′), so the block holding a serves them, and with them
/// E_core c and the one-electron term; its α links reach (a′, b), served by the
/// block holding b. In the full layout an α block (@p beta_major false) adds
/// the β-link part to its own rows and a β block the α-link part. With
/// @p symmetric G of (a′, b) is G of its mirror, so a block of the one set
/// serves both parts, each to the triangle rows it reaches (a or b in the
/// block), scaled by √2 off the diagonal. @p sigma must be zero before the
/// first block; once every block has added, it is sigma_from_g's σ.
[[nodiscard]] wwr::wwrError_t sigma_from_g_block(wwr::wwrStream_t stream, bool symmetric,
                                                 bool beta_major, int32_t first, int32_t count,
                                                 int32_t n_alpha, int32_t n_beta, int32_t n_pairs,
                                                 const Replacement *alpha_links, int32_t alpha_per,
                                                 const Replacement *beta_links, int32_t beta_per,
                                                 double e_core, const double *c, const double *g,
                                                 double *sigma);
[[nodiscard]] wwr::wwrError_t
sigma_from_g_block(wwr::wwrStream_t stream, bool symmetric, bool beta_major, int32_t first,
                   int32_t count, int32_t n_alpha, int32_t n_beta, int32_t n_pairs,
                   const Replacement *alpha_links, int32_t alpha_per, const Replacement *beta_links,
                   int32_t beta_per, double e_core, const double *c, const float *g, double *sigma);

/// @brief diag(H) for the @p n_alpha · @p n_beta determinants, by ci::Sigma::diagonal's
/// rule and in its order.
///
/// @p strings_alpha and @p strings_beta are the two sets' bit patterns, by
/// address. @p terms is n + 2 n² doubles: h_ii, then (ii|jj) and (ij|ji), each
/// n × n row-major over (i, j).
[[nodiscard]] wwr::wwrError_t diagonal(wwr::wwrStream_t stream, int32_t n, int32_t n_alpha,
                                       int32_t n_beta, const uint64_t *strings_alpha,
                                       const uint64_t *strings_beta, double e_core,
                                       const double *terms, double *diagonal);

/// @brief γ, n × n, from its packed form: @p packed[pair(t, u)] is
/// Σ_K c_K D_K[pair(t, u)], which is γ_tt on the diagonal and γ_tu + γ_ut =
/// 2 γ_tu off it. γ is symmetric, so row- and column-major agree.
[[nodiscard]] wwr::wwrError_t unpack_rdm1(wwr::wwrStream_t stream, int32_t n, const double *packed,
                                          double *gamma);

/// @brief The @p k ci::Layout::symmetric vectors @p x, n_strings (n_strings + 1)
/// / 2 each, unpacked into the @p k full ones @p y, n_strings² each.
[[nodiscard]] wwr::wwrError_t unpack(wwr::wwrStream_t stream, int32_t n_strings, int32_t k,
                                     const double *x, double *y);

/// @brief The @p k full vectors @p x, n_strings² each, packed into the @p k
/// ci::Layout::symmetric ones @p y, n_strings (n_strings + 1) / 2 each:
/// unpack's adjoint.
[[nodiscard]] wwr::wwrError_t pack(wwr::wwrStream_t stream, int32_t n_strings, int32_t k,
                                   const double *x, double *y);

} // namespace free_breakfast::ci::device
