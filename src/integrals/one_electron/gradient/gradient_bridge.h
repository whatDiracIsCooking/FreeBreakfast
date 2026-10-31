/// @file
/// @brief The one-electron gradient and derivative-matrix launchers, declared
/// once for the module that calls them and the device TU that defines them.
///
/// Included by gradient.cpp in its GLOBAL MODULE FRAGMENT, and by
/// gradient_launch.cu directly. It has to be the GMF: a declaration in a
/// module's purview gets module linkage and can never bind to a definition
/// compiled in a plain TU, which a .cu is (wwr docs/architecture.md, section
/// 14).
///
/// wwrStream_t and wwrError_t come from a different wwr header in each pass, as
/// overlap_bridge.h explains, so the module passes its stream in and takes
/// the status back with no cast.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "nuclei_view.h"
#include "primitive_pair_class_view.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief Add one class's overlap gradient into @p g, on @p stream:
/// G[A, k] += Sum_mn M_mn dS_mn/dR_{A,k}, the generated kernel gs_{la}{lb}, one
/// thread per record.
///
/// First writes each record's two atoms into @p atoms (the bra's at
/// [0, n_rec), the ket's at [n_rec, 2 n_rec)): @p ao_atom[off] is the atom of
/// Cartesian AO off. @p m is the n_ao x n_ao Cartesian matrix; the contraction
/// is the same for it and its transpose, so its layout does not matter. @p g
/// is [n_atom, 3], row-major, and is only added into.
///
/// cls.n_rec and n_atom must be positive, cls.la <= cls.lb <= the largest
/// momentum kernels/overlap_gradient.h's table covers, and n_ao * n_ao must fit
/// an int32_t.
///
/// @return The first launch status that failed (wwrErrorInvalidValue for a
///         class with no kernel). Launch errors only: the kernels' own faults
///         surface at the next synchronization.
[[nodiscard]] wwr::wwrError_t overlap_gradient_class(wwr::wwrStream_t stream,
                                                     const PrimitivePairClassView &cls,
                                                     const int32_t *ao_atom, int32_t *atoms,
                                                     int32_t n_ao, int32_t n_atom, const double *m,
                                                     double *g);

/// @brief As @ref overlap_gradient_class, for the kinetic energy T: the
/// generated kernel gt_{la}{lb}, its table kernels/kinetic_gradient.h.
[[nodiscard]] wwr::wwrError_t kinetic_gradient_class(wwr::wwrStream_t stream,
                                                     const PrimitivePairClassView &cls,
                                                     const int32_t *ao_atom, int32_t *atoms,
                                                     int32_t n_ao, int32_t n_atom, const double *m,
                                                     double *g);

/// @brief As @ref overlap_gradient_class, for the nuclear attraction V: the
/// generated kernel gv_{la}{lb}, its table kernels/potential_gradient.h, which
/// also adds the Hellmann-Feynman term into every nucleus's row of @p g.
///
/// The atoms are @p nuclei, so @p g is [nuclei.n_nuc, 3] and @p ao_atom
/// indexes them.
[[nodiscard]] wwr::wwrError_t potential_gradient_class(wwr::wwrStream_t stream,
                                                       const PrimitivePairClassView &cls,
                                                       const int32_t *ao_atom, int32_t *atoms,
                                                       const NucleiView &nuclei, int32_t n_ao,
                                                       const double *m, double *g);

/// @brief Add one class's overlap derivative matrices into @p x1, on
/// @p stream: x1[k, m, n] += dS_mn/dA_k, A the centre of m, the generated
/// kernel ips_{la}{lb}, one thread per record.
///
/// @p x1 is [3, n_ao, n_ao] over Cartesian AOs, row-major, and is only added
/// into. cls.n_rec must be positive, cls.la <= cls.lb <= the largest momentum
/// kernels/overlap_ip1.h's table covers, and 3 n_ao^2 must fit an int32_t.
///
/// @return As @ref overlap_gradient_class.
[[nodiscard]] wwr::wwrError_t overlap_ip1_class(wwr::wwrStream_t stream,
                                                const PrimitivePairClassView &cls, int32_t n_ao,
                                                double *x1);

/// @brief As @ref overlap_ip1_class, for the kinetic energy T: ipt_{la}{lb},
/// its table kernels/kinetic_ip1.h.
[[nodiscard]] wwr::wwrError_t kinetic_ip1_class(wwr::wwrStream_t stream,
                                                const PrimitivePairClassView &cls, int32_t n_ao,
                                                double *x1);

/// @brief As @ref overlap_ip1_class, for the nuclear attraction V with every
/// nucleus of @p nuclei held fixed: ipv_{la}{lb}, its table
/// kernels/potential_ip1.h. nuclei.n_nuc must be positive.
[[nodiscard]] wwr::wwrError_t potential_ip1_class(wwr::wwrStream_t stream,
                                                  const PrimitivePairClassView &cls,
                                                  const NucleiView &nuclei, int32_t n_ao,
                                                  double *x1);

/// @brief Add one class's term of V's operator at nucleus @p nuc into @p xc,
/// on @p stream: xc[k, m, n] += dV^C_mn/dC_k, V^C = -Z_C / |r - C|, the
/// generated kernel iprv_{la}{lb}. @p nuc must be in [0, nuclei.n_nuc). As
/// @ref potential_ip1_class in every other respect.
[[nodiscard]] wwr::wwrError_t potential_rinv_class(wwr::wwrStream_t stream,
                                                   const PrimitivePairClassView &cls,
                                                   const NucleiView &nuclei, int32_t nuc,
                                                   int32_t n_ao, double *xc);

/// @brief The centres' terms of every atom's derivative matrices, on
/// @p stream: for every atom A in [0, n_atom) and axis k,
///   out[A, k, m, n] += [ao_atom[m] == A] y[k, m, n] + [ao_atom[n] == A] y[k, n, m]
/// with @p y the [3, n, n] row-major first-index derivative (an ip1 matrix
/// carried onto the layout @p ao_atom indexes) and @p out [n_atom, 3, n, n].
///
/// n_atom and n must be positive and 3 n_atom no more than a grid's y extent.
///
/// @return The launch status.
[[nodiscard]] wwr::wwrError_t add_centre_derivatives(wwr::wwrStream_t stream, int32_t n_atom,
                                                     int32_t n, const int32_t *ao_atom,
                                                     const double *y, double *out);

} // namespace free_breakfast::device
