/// @file
/// @brief The solver's launchers, declared once for the module that calls them
/// and the device TU that defines them: S² on a block of CI vectors, and the
/// diagonal preconditioner with Olsen's correction.
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
/// ε = 0. @p correction must overlap neither @p x nor @p residual.
[[nodiscard]] wwr::wwrError_t ci_olsen(wwr::wwrStream_t stream, int32_t n, double theta,
                                       const double *shifted, const double *x,
                                       const double *residual, const double *dots,
                                       double *correction);

/// @brief The blocks ci_lowest's first pass runs, and so the length of its
/// @p partial_values and @p partial_index: at most kCiLowestBlocks.
inline constexpr int32_t kCiLowestBlocks = 1024;

/// @brief The addresses of the @p count lowest of the @p n @p values into
/// @p picks, ascending in value and ties to the lower address: the first
/// @p count of a stable sort by value, without the sort.
///
/// Pick j is the least (value, address) above pick j − 1, in two passes: each
/// block of the first reduces its stride of @p values into @p partial_values
/// and @p partial_index, and one block reduces those. @p count passes over
/// @p values, for the handful of guesses a Davidson wants. Requires
/// count <= n.
[[nodiscard]] wwr::wwrError_t ci_lowest(wwr::wwrStream_t stream, int32_t n, const double *values,
                                        int32_t count, double *partial_values,
                                        int32_t *partial_index, int32_t *picks);

/// @brief ci_solve's @p count guess columns, n long, column-major: column g is
/// a 1e-3 spread (a splitmix64 hash of the element's position g · n + i,
/// uniform in [−1e-3, 1e-3)) plus 1 on determinant @p picks[g]. Not yet
/// orthonormal.
[[nodiscard]] wwr::wwrError_t ci_guess(wwr::wwrStream_t stream, int32_t n, int32_t count,
                                       const int32_t *picks, double *guess);

} // namespace free_breakfast::device
