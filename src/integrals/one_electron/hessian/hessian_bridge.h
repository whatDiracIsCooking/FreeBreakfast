/// @file
/// @brief The one-electron Hessian launchers, declared once for the module
/// that calls them and the device TU that defines them.
///
/// Included by hessian.cpp in its GLOBAL MODULE FRAGMENT, and by
/// hessian_launch.cu directly. It has to be the GMF: a declaration in a
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

/// @brief Add one class's overlap Hessian into @p h, on @p stream:
/// H[A, k, B, l] += Sum_mn M_mn d2S_mn/dR_{A,k}dR_{B,l}, the generated kernel
/// d2s_{la}{lb}, one thread per record.
///
/// First writes each record's two atoms into @p atoms (the bra's at
/// [0, n_rec), the ket's at [n_rec, 2 n_rec)): @p ao_atom[off] is the atom of
/// Cartesian AO off. @p m is the n_ao x n_ao Cartesian matrix; the contraction
/// is the same for it and its transpose, so its layout does not matter. @p h
/// is [n_atom, 3, n_atom, 3], row-major, and is only added into.
///
/// cls.n_rec and n_atom must be positive, cls.la <= cls.lb <= the largest
/// momentum kernels/overlap_hessian.h's table covers, and n_ao * n_ao must fit
/// an int32_t.
///
/// @return The first launch status that failed (wwrErrorInvalidValue for a
///         class with no kernel). Launch errors only: the kernels' own faults
///         surface at the next synchronization.
[[nodiscard]] wwr::wwrError_t overlap_hessian_class(wwr::wwrStream_t stream,
                                                    const PrimitivePairClassView &cls,
                                                    const int32_t *ao_atom, int32_t *atoms,
                                                    int32_t n_ao, int32_t n_atom, const double *m,
                                                    double *h);

/// @brief As @ref overlap_hessian_class, for the kinetic energy T: the
/// generated kernel d2t_{la}{lb}, its table kernels/kinetic_hessian.h.
[[nodiscard]] wwr::wwrError_t kinetic_hessian_class(wwr::wwrStream_t stream,
                                                    const PrimitivePairClassView &cls,
                                                    const int32_t *ao_atom, int32_t *atoms,
                                                    int32_t n_ao, int32_t n_atom, const double *m,
                                                    double *h);

/// @brief As @ref overlap_hessian_class, for the nuclear attraction V: the
/// generated kernel d2v_{la}{lb}, its table kernels/potential_hessian.h,
/// which also adds the centre-nucleus and nucleus-nucleus terms into every
/// nucleus's rows and columns of @p h.
///
/// The atoms are @p nuclei, so @p h is [nuclei.n_nuc, 3, nuclei.n_nuc, 3] and
/// @p ao_atom indexes them.
[[nodiscard]] wwr::wwrError_t potential_hessian_class(wwr::wwrStream_t stream,
                                                      const PrimitivePairClassView &cls,
                                                      const int32_t *ao_atom, int32_t *atoms,
                                                      const NucleiView &nuclei, int32_t n_ao,
                                                      const double *m, double *h);

} // namespace free_breakfast::device
