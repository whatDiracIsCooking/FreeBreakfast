/// @file
/// @brief The diis_solve launcher, declared once for the module that calls it
/// and the device TU that defines it.
///
/// Included by diis.cpp in its GLOBAL MODULE FRAGMENT, and by diis_solve.cu
/// directly. It has to be the GMF: a declaration in a module's purview gets
/// module linkage and can never bind to a definition compiled in a plain TU,
/// which a .cu is (wwr docs/architecture.md, section 14).
///
/// wwrStream_t and wwrError_t come from a different wwr header in each pass, as
/// src/integrals/one_electron/overlap/overlap_bridge.h explains, so the module
/// passes its stream in and takes the status back with no cast.

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

/// @brief The largest bordered system diis_solve takes: kDiisMaxHistoryLimit
/// plus the Lagrange row. The kernel's shared memory is sized statically from
/// it, (dim+1)*dim doubles = 34,320 B, under the 48 KiB static limit; diis.cpp
/// asserts it agrees with the module's constant.
inline constexpr int32_t kDiisSolveMaxDim = 65;

/// @brief Solve the bordered DIIS system for the @p m extrapolation
/// coefficients, on @p stream, in one block using shared memory.
///
/// @p gram is the device-resident residual Gram, physical-column indexed with
/// a fixed leading dimension @p ld (>= @p m). Only its upper triangle over the
/// valid columns, `gram[a*ld + c]` for a <= c < m, is read; the kernel
/// symmetrizes it. The bordered system
///
///     [ <e_i|e_j>  -1 ] [ c ]   [ 0  ]
///     [   -1^T      0 ] [ l ] = [ -1 ]
///
/// is built in shared memory and solved by Gaussian elimination with partial
/// pivoting, and the @p m coefficients are written to @p coeff.
///
/// If a pivot falls below 1e-14 the subspace is singular, and @p coeff is set
/// to the unit selector at @p newest (1 there, 0 elsewhere), so the caller's
/// extrapolation reproduces the newest Fock. @p singular_out (one device int)
/// is set to 1 in that case and 0 otherwise. @p m must be in 1..ld, @p ld
/// below kDiisSolveMaxDim, and @p newest in 0..m-1.
///
/// @return The launch status. Launch errors only: the kernel's own faults
///         surface at the next synchronization.
[[nodiscard]] wwr::wwrError_t diis_solve(wwr::wwrStream_t stream, int32_t m, int32_t ld,
                                         int32_t newest, const double *gram, double *coeff,
                                         int32_t *singular_out);

} // namespace free_breakfast::device
