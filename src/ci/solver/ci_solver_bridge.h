/// @file
/// @brief The solver's launchers, declared once for the module that calls them
/// and the device TU that defines them: S² on a block of CI vectors, the
/// diagonal preconditioner with Olsen's correction, the lowest-value select,
/// and the P-space's Hamiltonian, gather and scatter.
///
/// Included by solver.cpp in its GLOBAL MODULE FRAGMENT, and by ci_solver.cu
/// directly: a declaration in a module's purview gets module linkage and can
/// never bind to a definition compiled in a plain TU, which a .cu is (see
/// src/integrals/one_electron/overlap/overlap_bridge.h).
///
/// The layout is the GPU sigma's (see ../sigma/ci_sigma_bridge.h): a
/// determinant (I_α, I_β) is element I_α · n_beta + I_β, a block of vectors is
/// column-major with leading dimension n_det, and a *row* is one (vector j,
/// determinant I) pair, row j · n_det + I. The tables are S²'s own:
///
/// - An α link table holds the α set's replacement lists, `alpha_per` links per
///   string, three int32_t per link a†_p a_q: the target string's address, the
///   index q · n + p of its β partner a†_q a_p in a β slot table, and the sign.
/// - A β slot table holds, per β string and n × n over (p, q) row-major, the
///   replacement a†_p a_q of that string as sign · (target + 1), or 0 when it
///   annihilates the string.
/// - With @p symmetric, the vectors are in CiLayout::symmetric (see
///   ../sigma/ci_symmetric.h): n_det is the triangle's, and a row a triangle
///   element.
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

/// @brief For every one of the @p rows rows of the block @p x,
/// y = (accumulate ? y : 0) + scale · ((S² − shift) x), with
/// S² = constant − Σ_pq E^β_pq E^α_qp and @p constant = S_z (S_z + 1) + N_β.
///
/// One thread per row gathers over its own α links and their β partners, since
/// S² is real symmetric. @p y has @p x's layout and must not overlap it.
[[nodiscard]] wwr::wwrError_t ci_spin_square(wwr::wwrStream_t stream, bool symmetric, int32_t rows,
                                             int32_t n_det, int32_t n_beta, int32_t n,
                                             const int32_t *alpha_links, int32_t alpha_per,
                                             const int32_t *beta_slots, double constant,
                                             double shift, double scale, bool accumulate,
                                             const double *x, double *y);

/// @brief shifted_J = diagonal_J + scale · (diag(S²)_J − shift) for the
/// @p n_det determinants, diag(S²)_J = constant − the orbitals doubly occupied
/// in J. With @p symmetric, J runs over the triangle and @p diagonal is still
/// the full layout's.
///
/// With @p exchange (device, n × n, (ij|ji)) not null, diagonal_J is
/// spin-averaged first: + Σ_{i<j open} (δ_{σ_i σ_j} − p) K_ij over J's open
/// shells, p = [n_α(n_α − 1) + n_β(n_β − 1)] / [n_o(n_o − 1)] with n_α and n_β
/// of the n_o open shells α and β: the average of diagonal over the
/// determinants of J's configuration.
///
/// One thread per determinant counts its α links a†_p a_p whose β partner
/// a†_p a_p does not annihilate J's β string: the doubly occupied orbitals.
/// The same links give the α occupations, and the slot table's (p, p)
/// entries the β ones.
[[nodiscard]] wwr::wwrError_t ci_spin_diagonal(wwr::wwrStream_t stream, bool symmetric,
                                               int32_t n_det, int32_t n_beta, int32_t n,
                                               const int32_t *alpha_links, int32_t alpha_per,
                                               const int32_t *beta_slots, double constant,
                                               double shift, double scale, const double *exchange,
                                               const double *diagonal, double *shifted);

/// @brief out_i = in_i / (theta − shifted_i) for the @p n elements of one
/// column, the denominator held off zero at ±1e-3: M⁻¹ in, M = theta − shifted.
/// @p out may be @p in.
[[nodiscard]] wwr::wwrError_t ci_precondition(wwr::wwrStream_t stream, int32_t n, double theta,
                                              const double *shifted, const double *in, double *out);

/// @brief Olsen's correction for one column, correction = M⁻¹ (residual − ε x)
/// with M as ci_precondition's and ε = dots[1] / dots[0], read on the device:
/// @p dots (device, 2) holds x^T M⁻¹ x and x^T M⁻¹ r. A zero dots[0] gives
/// ε = 0. With @p shifted null, M⁻¹ is left out: correction = residual − ε x,
/// for an M that is not diagonal to apply after. @p correction must overlap
/// neither @p x nor @p residual.
[[nodiscard]] wwr::wwrError_t ci_olsen(wwr::wwrStream_t stream, int32_t n, double theta,
                                       const double *shifted, const double *x,
                                       const double *residual, const double *dots,
                                       double *correction);

/// @brief The 64-bit words of ci_lowest's @p scratch, besides its survivors:
/// a 256-bin histogram, the threshold found so far and the counters.
inline constexpr int32_t kCiLowestScratch = 136;

/// @brief The addresses of the @p count lowest of the @p n @p values into
/// @p picks, ascending in value and ties to the lower address: the first
/// @p count of a stable sort by value, without the sort.
///
/// A radix select over (value, address), a byte at a time from the top, the
/// value as its order-preserving 64 bits (−0 as +0): each of the twelve passes
/// histograms the next byte of the elements that match the prefix so far, and
/// one block picks the byte under which the count-th lies. That leaves the
/// count-th element exactly, so the elements at or below it are the @p count
/// lowest: they are compacted into @p survivors (device, @p count) in any
/// order, then each written to @p picks at its rank among them. Twelve passes
/// over @p values whatever @p count is. @p scratch is kCiLowestScratch device
/// words. Requires 0 < count <= n.
[[nodiscard]] wwr::wwrError_t ci_lowest(wwr::wwrStream_t stream, int32_t n, const double *values,
                                        int32_t count, uint64_t *scratch, int32_t *survivors,
                                        int32_t *picks);

/// @brief ci_solve's @p count guess columns, n long, column-major: column g is
/// a 1e-3 spread (a splitmix64 hash of the element's position g · n + i,
/// uniform in [−1e-3, 1e-3)) plus 1 on determinant @p picks[g], or the spread
/// alone with @p picks null. Not yet orthonormal.
[[nodiscard]] wwr::wwrError_t ci_guess(wwr::wwrStream_t stream, int32_t n, int32_t count,
                                       const int32_t *picks, double *guess);

/// @brief to[j + g · p] = from[g · n + picks[j]] for the @p p elements
/// @p picks of each of the @p count n-long columns of @p from: their P part,
/// packed p × count.
[[nodiscard]] wwr::wwrError_t ci_gather(wwr::wwrStream_t stream, int32_t n, int32_t count,
                                        int32_t p, const int32_t *picks, const double *from,
                                        double *to);

/// @brief to[g · n + picks[j]] = from[j + g · p]: ci_gather's inverse, which
/// writes the P elements of the @p count n-long columns of @p to and leaves
/// the rest. @p picks must be distinct.
[[nodiscard]] wwr::wwrError_t ci_scatter(wwr::wwrStream_t stream, int32_t n, int32_t count,
                                         int32_t p, const int32_t *picks, const double *from,
                                         double *to);

/// @brief The P-space Hamiltonian, h[i + j · p] = <φ_i| H + λ (S² − target)
/// |φ_j> over the @p p elements @p picks of a vector in the layout: in the
/// full one a determinant (a, b) is element a · n_beta + b; with
/// @p symmetric a triangle element (see ../sigma/ci_symmetric.h), φ the
/// determinant (a, a) or (|ab> + |ba>) / √2, so h combines up to four
/// determinant elements.
///
/// A determinant element is Slater–Condon's over the strings' bit patterns
/// (@p alpha_strings and @p beta_strings by address; one table for both in
/// the symmetric layout), with the integrals read from CiSigma's
/// @p contraction (n_pairs × (n_pairs + 1), column-major: column pair(t, u)
/// ½ (tu|vw) down the rows pair(v, w), the last k_vw =
/// h_vw − ½ Σ_u (vu|uw)), and the diagonal from @p diagonal (the full layout's
/// diag(H)) plus λ (@p spin_shift − the doubly occupied orbitals),
/// @p spin_shift = S_z (S_z + 1) + N_β − target. S² off the diagonal joins
/// only the two determinants of one αβ swap a†_pα a_qα a†_qβ a_pβ, at −λ times
/// their signs. One thread per (i, j).
[[nodiscard]] wwr::wwrError_t ci_pspace(wwr::wwrStream_t stream, bool symmetric, int32_t p,
                                        const int32_t *picks, int32_t n_beta, int32_t n,
                                        const uint64_t *alpha_strings, const uint64_t *beta_strings,
                                        const double *contraction, const double *diagonal,
                                        double penalty, double spin_shift, double *h);

} // namespace free_breakfast::device
