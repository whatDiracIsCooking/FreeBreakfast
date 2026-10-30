/// @file
/// @brief The J gradient's launcher, declared once for the module that calls
/// it and the device TU that defines it.
///
/// Included by coulomb_gradient.cpp in its GLOBAL MODULE FRAGMENT, and by
/// coulomb_gradient_launch.cu directly, for the reason overlap_bridge.h gives:
/// a declaration in a module's purview gets module linkage and can never bind
/// to a definition compiled in a plain TU. wwrStream_t and wwrError_t come from
/// a different wwr header in each pass, as there.

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

/// @brief Add one bra class's term of dE_J/dR into @p g, on @p stream:
/// G[A, k] += Sum_ab M_ab Sum_t (dA_k E^ab)_t J_t, the generated gjab<la, lb>,
/// one thread per record.
///
/// First writes each record's two atoms into @p atoms (the bra's at
/// [0, n_rec), the ket's at [n_rec, 2 n_rec)): @p ao_atom[off] is the atom of
/// Cartesian AO off. @p cls's records are read in record order, ea and eb
/// included; @p rows is the record-to-row map of @p jp, which is cls.n_rec x
/// n_hermite(cls.la + cls.lb + 1), record-major: the raised [p|q]'s J_p
/// (e_hermite_class at raise 1). @p m is the n_ao x n_ao Cartesian density,
/// symmetric; @p g is [n_atom, 3], row-major, and is only added into.
///
/// cls.n_rec and n_atom must be positive, cls.la <= cls.lb <= the largest
/// momentum kernels/gjab.h's table covers, and n_ao * n_ao must fit an int32_t.
///
/// @return The first launch status that failed (wwrErrorInvalidValue for a
///         class with no kernel). Launch errors only.
[[nodiscard]] wwr::wwrError_t coulomb_gradient_class(wwr::wwrStream_t stream,
                                                     const PrimitivePairClassView &cls,
                                                     const int32_t *rows, const int32_t *ao_atom,
                                                     int32_t *atoms, int32_t n_ao, int32_t n_atom,
                                                     const double *jp, const double *m, double *g);

} // namespace free_breakfast::device
