/// @file
/// @brief The dipole-gradient launcher, declared once for the module that calls
/// it and the device TU that defines it.
///
/// Included by dipole_gradient.cpp in its GLOBAL MODULE FRAGMENT, and by
/// dipole_gradient_launch.cu directly. It has to be the GMF: a declaration in a
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

#include "primitive_pair_class_view.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief Add one class's dipole gradient into @p g, on @p stream:
/// G[c, A, k] += Sum_mn M_mn d mu^c_mn / dR_{A,k}, mu^c = <m| r_c |n> about
/// the origin, the generated kernel gmu_{la}{lb}, one thread per record.
///
/// First writes each record's two atoms into @p atoms (the bra's at
/// [0, n_rec), the ket's at [n_rec, 2 n_rec)): @p ao_atom[off] is the atom of
/// Cartesian AO off. @p m is the n_ao x n_ao Cartesian matrix; the contraction
/// is the same for it and its transpose, so its layout does not matter. @p g
/// is [3, n_atom, 3], row-major, and is only added into.
///
/// cls.n_rec and n_atom must be positive, cls.la <= cls.lb <= the largest
/// momentum kernels/dipole_gradient.h's table covers, and n_ao * n_ao must fit
/// an int32_t.
///
/// @return The first launch status that failed (wwrErrorInvalidValue for a
///         class with no kernel). Launch errors only: the kernels' own faults
///         surface at the next synchronization.
[[nodiscard]] wwr::wwrError_t dipole_gradient_class(wwr::wwrStream_t stream,
                                                    const PrimitivePairClassView &cls,
                                                    const int32_t *ao_atom, int32_t *atoms,
                                                    int32_t n_ao, int32_t n_atom, const double *m,
                                                    double *g);

} // namespace free_breakfast::device
