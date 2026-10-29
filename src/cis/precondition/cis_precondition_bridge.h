/// @file
/// @brief The CIS preconditioner's launchers, declared once for the module
/// that calls them and the device TU that defines them: the orbital-energy
/// diagonal Δε and the diagonal correction r / (θ − Δε).
///
/// Included by precondition.cpp in its GLOBAL MODULE FRAGMENT, and by
/// cis_precondition.cu directly: a declaration in a module's purview gets
/// module linkage and can never bind to a definition compiled in a plain TU,
/// which a .cu is (see src/integrals/one_electron/overlap/overlap_bridge.h).
///
/// An element ia of an n_ov vector is i · n_virt + a, i over the occupied
/// orbitals and a over the virtual ones: row-major (i, a). A block of k such
/// vectors is column-major with leading dimension n_ov.
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

namespace free_breakfast::cis::device {

/// @brief delta[i · n_virt + a] = orbital_energies[n_occ + a] −
/// orbital_energies[i] for the @p n_occ × @p n_virt pairs: Δε_ia = ε_a − ε_i,
/// the occupied energies first in @p orbital_energies (device,
/// n_occ + n_virt). One thread per element.
[[nodiscard]] wwr::wwrError_t diagonal(wwr::wwrStream_t stream, int32_t n_occ, int32_t n_virt,
                                       const double *orbital_energies, double *delta);

/// @brief correction[c · n + j] = residual[c · n + j] / (theta[c] − delta[j])
/// over the @p n × @p columns block, the denominator's magnitude held at or
/// above 1e-3 with its sign kept (a zero one taken as +1e-3). @p theta is
/// device, one per column. @p correction may be @p residual. One thread per
/// element.
[[nodiscard]] wwr::wwrError_t precondition(wwr::wwrStream_t stream, int32_t n, int32_t columns,
                                           const double *theta, const double *delta,
                                           const double *residual, double *correction);

} // namespace free_breakfast::cis::device
