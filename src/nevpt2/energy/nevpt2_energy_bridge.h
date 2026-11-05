/// @file
/// @brief The class energies' seven launchers -- the three diagonal-slice
/// accumulations, the Sijrs reduction, its adjoint, its tangent and its
/// adjoint's tangent -- declared once for the module that
/// calls them and the device TU that defines them.
///
/// Included by energy.cpp and reverse.cpp in their GLOBAL MODULE FRAGMENTs, and by
/// nevpt2_energy.cu directly: a declaration in a module's purview gets module
/// linkage and can never bind to a definition compiled in a plain TU, which a
/// .cu is (see src/integrals/one_electron/overlap/overlap_bridge.h).

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

namespace free_breakfast::nevpt2::device {

/// @brief a16[:, i, :, :, :, i] += fdm2 for every i: the NumPy reference's
/// slice assignment in make_a16. a16 is n⁶ and fdm2 (p, r, a, b) n⁴, both
/// row-major. Launch status only; it does not synchronize.
[[nodiscard]] wwr::wwrError_t diag_a16(wwr::wwrStream_t stream, const double *fdm2, double *a16,
                                       int32_t n);

/// @brief a22[:, i, :, i, :, :] -= fdm2 for every i (make_a22's first slice):
/// a22 n⁶, fdm2 (I, k, b, c) n⁴, both row-major.
[[nodiscard]] wwr::wwrError_t diag_a22a(wwr::wwrStream_t stream, const double *fdm2, double *a22,
                                        int32_t n);

/// @brief a22[:, i, :, :, i, :] += 2 fdm2 for every i (make_a22's second
/// slice): a22 n⁶, fdm2 (I, k, a, c) n⁴, both row-major.
[[nodiscard]] wwr::wwrError_t diag_a22b(wwr::wwrStream_t stream, const double *fdm2, double *a22,
                                        int32_t n);

/// @brief The Sijrs class over one slab of core orbitals [i0, i0 + ni):
/// acc[0] += Σ g_iajb θ and acc[1] += Σ g_iajb θ / (e_i + e_j - e_a - e_b),
/// θ = 2 g_iajb - g_ibja, PySCF's Sijrs with no vanishing-norm guard.
///
/// @p g is the slab (ia|jb), column-major (ni, nvirt, ncore, nvirt), as
/// nevpt2::slab writes ExtBlock::Sijrs; @p e_core, @p e_virt the whole
/// orbital-energy vectors. The sum is reassociated (a block reduction, then
/// an atomicAdd per block).
[[nodiscard]] wwr::wwrError_t sijrs(wwr::wwrStream_t stream, const double *g, int32_t ni,
                                    int32_t i0, int32_t ncore, int32_t nvirt, const double *e_core,
                                    const double *e_virt, double *acc);

/// @brief The adjoint of @ref sijrs over the same slab, by hand, of
/// @p weight E: with D = e_i + e_j - e_a - e_b and E = Σ g_iajb θ / D,
/// g_bar[iajb] += weight 2 θ / D (g_iajb's own term and its partner
/// g_ibja's), and weight g_iajb θ / D² taken off e_core_bar[i] and
/// e_core_bar[j] and added to e_virt_bar[a] and e_virt_bar[b]. The SC reverse
/// pass's weight is 1; the SC-QD one's is Σ_J H̄_JJ, since every state's own
/// Sijrs is the one E.
///
/// @p g_bar is the slab's layout. Either orbital-energy bar may be null, and
/// is then not formed; they are whole vectors, added to with atomics. Launch
/// status only; it does not synchronize.
[[nodiscard]] wwr::wwrError_t sijrs_adjoint(wwr::wwrStream_t stream, const double *g, int32_t ni,
                                            int32_t i0, int32_t ncore, int32_t nvirt,
                                            const double *e_core, const double *e_virt,
                                            double *g_bar, double *e_core_bar, double *e_virt_bar,
                                            double weight);

/// @brief The forward tangent of @ref sijrs's energy over the same slab, by
/// hand: *acc += Σ (ġ θ + g θ̇) / D - g θ Ḋ / D² along the slab's tangent
/// @p g_dot (the slab's layout) and the orbital energies' @p e_core_dot and
/// @p e_virt_dot (whole vectors; either may be null, and is then zero), with
/// θ = 2 g_iajb - g_ibja and D = e_i + e_j - e_a - e_b. Reassociated as
/// @ref sijrs is. Launch status only; it does not synchronize.
[[nodiscard]] wwr::wwrError_t sijrs_tangent(wwr::wwrStream_t stream, const double *g,
                                            const double *g_dot, int32_t ni, int32_t i0,
                                            int32_t ncore, int32_t nvirt, const double *e_core,
                                            const double *e_virt, const double *e_core_dot,
                                            const double *e_virt_dot, double *acc);

/// @brief The forward tangent of @ref sijrs_adjoint's bars (weight 1) over
/// the same slab, by hand, along the slab's tangent @p g_dot and the orbital
/// energies' @p e_core_dot and @p e_virt_dot (either may be null, and is then
/// zero): g_bar_dot += 2 θ̇ / D - 2 θ Ḋ / D², and the orbital energies' share
/// w = g θ / D² moves by ẇ = (ġ θ + g θ̇) / D² - 2 g θ Ḋ / D³, taken off
/// e_core_bar_dot[i] and [j] and added to e_virt_bar_dot[a] and [b] (either
/// may be null, and is then not formed). The SC Hessian-vector product's
/// Sijrs share. Launch status only; it does not synchronize.
[[nodiscard]] wwr::wwrError_t
sijrs_adjoint_tangent(wwr::wwrStream_t stream, const double *g, const double *g_dot, int32_t ni,
                      int32_t i0, int32_t ncore, int32_t nvirt, const double *e_core,
                      const double *e_virt, const double *e_core_dot, const double *e_virt_dot,
                      double *g_bar_dot, double *e_core_bar_dot, double *e_virt_bar_dot);

} // namespace free_breakfast::nevpt2::device
