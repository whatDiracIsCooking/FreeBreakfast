/// @file
/// @brief The mixed overlap launcher, declared once for the module that calls
/// it and the device TU that defines it.
///
/// Included by overlap_mixed.cpp in its GLOBAL MODULE FRAGMENT, and by
/// overlap_mixed_launch.cu directly, for the reason overlap_bridge.h gives: a
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

/// @brief Add one mixed class's overlap contributions into the column-major
/// n_rows x n_cols matrix @p s, on @p stream: one thread per record.
///
/// @p cls is a class of mixed_primitive_pair_classes(rows, cols): off_a a
/// Cartesian AO of the rows' basis, off_b one of the columns'. Each value
/// <m|n> is atomically added once, at element (m, n), i.e. s[m + n * n_rows];
/// nothing is mirrored. So @p s must be zeroed before the first class of a
/// build.
///
/// The generated overlap_mixed<la, lb> writes its row basis row-major,
/// S[m * n_b + n], so the launcher hands it the class swapped -- the columns'
/// shell as its row: overlap_mixed<cls.lb, cls.la>, P - A and P - B, off_a and
/// off_b, n_rows and n_cols exchanged. <n|m> = <m|n>, and its row-major
/// n_cols x n_rows is this column-major n_rows x n_cols.
///
/// cls.n_rec must be positive, and cls.la and cls.lb at most
/// kMaxAngularMomentum. The kernels index @p s as a 32-bit int, so
/// n_rows * n_cols must fit an int32_t.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no kernel).
///         Launch errors only: the kernel's own faults surface at the next
///         synchronization.
[[nodiscard]] wwr::wwrError_t overlap_mixed_class(wwr::wwrStream_t stream,
                                                  const PrimitivePairClassView &cls, int32_t n_rows,
                                                  int32_t n_cols, double *s);

} // namespace free_breakfast::device
