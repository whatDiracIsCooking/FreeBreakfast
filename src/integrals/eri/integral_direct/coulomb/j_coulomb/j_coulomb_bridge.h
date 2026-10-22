/// @file
/// @brief The bra back-transform's launchers, declared once for the module that
/// calls them and the device TU that defines them.
///
/// Included by j_coulomb.cpp in its GLOBAL MODULE FRAGMENT, and by
/// j_coulomb_launch.cu directly, for the reason overlap_bridge.h gives: a
/// declaration in a module's purview gets module linkage and can never bind to
/// a definition compiled in a plain TU. wwrStream_t and wwrError_t come from a
/// different wwr header in each pass, as there.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "angular_momentum.h"
#include "primitive_pair_class_view.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief Fill @p rows[k] = k for k < @p n, on @p stream: jab's record-to-row
/// map of J_p, the identity with one batch per class.
///
/// @return The launch status (wwrErrorInvalidValue if @p n is not positive).
[[nodiscard]] wwr::wwrError_t j_coulomb_rows(wwr::wwrStream_t stream, int32_t n, int32_t *rows);

/// @brief Add one bra class's back-transform of @p jp into the n_ao x n_ao
/// matrix @p j, on @p stream: the generated jab<la, lb>, one thread per record.
///
/// @p jp is complete (every ket class added, e_hermite_class); @p rows holds
/// at least cls.n_rec entries, rows[k] = k (@ref j_coulomb_rows). Each value
/// is atomically added at (m, n) and (n, m), a shell against itself writing
/// its block's lower triangle and mirror once, so @p j must be zeroed before
/// the first class and is symmetric after the last. The kernels index @p j as
/// a 32-bit `m * n_ao + n`.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no kernel).
///         Launch errors only.
[[nodiscard]] wwr::wwrError_t j_coulomb_class(wwr::wwrStream_t stream,
                                              const PrimitivePairClassView &cls, int32_t n_ao,
                                              const int32_t *rows, const double *jp, double *j);

} // namespace free_breakfast::device
