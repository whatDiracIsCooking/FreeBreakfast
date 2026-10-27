/// @file
/// @brief The Pipek-Mezey launchers, declared once for the module that calls
/// them and the device TU that defines them.
///
/// Included by pipek_mezey.cpp in its GLOBAL MODULE FRAGMENT, and by pipek_mezey_kernels.cu
/// directly: a declaration in a module's purview gets module linkage and can
/// never bind to a definition compiled in a plain TU, which a .cu is (see
/// src/integrals/one_electron/overlap/overlap_bridge.h).
///
/// Every matrix is double, is column-major with a packed leading
/// dimension, and the population tensor is laid out by pm_pop_index. Each
/// returns the launch status and does not synchronize.

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

namespace free_breakfast::localization::device {

/// @brief The largest n_occ the launches can address: the population build and
/// the rotation put the orbital pair on the y and z grid axes, which CUDA caps
/// at 65535 blocks.
inline constexpr int32_t kMaxOccupied = 65535;

/// @brief Build Q^A_ij for every atom and every occupied pair.
///
/// @p c is the n_ao x n_occ occupied block and @p w = S C, both with leading
/// dimension @p ld. Atom A owns the AOs [atom_ao_offsets[A],
/// atom_ao_offsets[A + 1]). @p pops receives n_atoms * n_occ * n_occ elements.
[[nodiscard]] wwr::wwrError_t populations(wwr::wwrStream_t stream, int32_t n_atoms, int32_t n_occ,
                                          int32_t ld, const double *c, const double *w,
                                          const int32_t *atom_ao_offsets, double *pops);

/// @brief The rotation of each of one round's @p n_pairs disjoint pairs, for
/// the functional L_@p exponent (2 or 4).
///
/// One block per pair, reducing over the atoms. The outputs are per orbital:
/// each orbital is in exactly one pair of a round, so one launch writes all
/// n_occ entries of @p cos_out, @p sin_out and @p partner_out. A pair's two
/// members share the cosine and carry opposite sines; an orbital drawn against
/// the padding index n_occ gets the identity and is its own partner.
[[nodiscard]] wwr::wwrError_t pair_angles(wwr::wwrStream_t stream, int32_t exponent,
                                          int32_t n_atoms, int32_t n_occ, const double *pops,
                                          const int32_t *pairs, int32_t n_pairs,
                                          double gain_threshold, double *cos_out, double *sin_out,
                                          int32_t *partner_out);

/// @brief @p pops_out = R^T @p pops_in R for the round pair_angles wrote.
/// Out-of-place: each output element reads four input elements.
[[nodiscard]] wwr::wwrError_t rotate_populations(wwr::wwrStream_t stream, int32_t n_atoms,
                                                 int32_t n_occ, const double *pops_in,
                                                 const double *cos_v, const double *sin_v,
                                                 const int32_t *partner, double *pops_out);

/// @brief @p out = @p in R for the same round, on an n_occ x n_occ matrix.
/// Out-of-place, as above.
[[nodiscard]] wwr::wwrError_t rotate_columns(wwr::wwrStream_t stream, int32_t n_occ,
                                             const double *in, const double *cos_v,
                                             const double *sin_v, const int32_t *partner,
                                             double *out);

/// @brief L_@p exponent = sum_i sum_A (Q^A_ii)^exponent into the device scalar
/// @p out. One block, so the summation order, and so the value, is the same
/// on every run.
[[nodiscard]] wwr::wwrError_t functional(wwr::wwrStream_t stream, int32_t exponent, int32_t n_atoms,
                                         int32_t n_occ, const double *pops, double *out);

/// @brief Write the @p n x @p n identity into @p matrix.
[[nodiscard]] wwr::wwrError_t identity(wwr::wwrStream_t stream, int32_t n, double *matrix);

} // namespace free_breakfast::localization::device
