/// @file
/// @brief The overlap launcher, declared once for the module that calls it and
/// the device TU that defines it.
///
/// Included by one_electron.cpp in its GLOBAL MODULE FRAGMENT, and by
/// overlap_launch.cu directly. It has to be the GMF: a declaration in a
/// module's purview gets module linkage and can never bind to a definition
/// compiled in a plain TU, which a .cu is (wwr docs/architecture.md, section
/// 14).
///
/// wwrStream_t comes from wwr's runtime.h, whose always-on part is that one type
/// and is host-safe; it is the same type wwr.runtime_api exports, so the module
/// passes its stream straight through.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "runtime.h"

namespace free_breakfast::device {

/// @brief The largest angular momentum, per shell, the generated overlap
/// kernels cover: classes (la, lb) with la <= lb <= this.
inline constexpr int32_t kOverlapMaxAngularMomentum = 2;

/// @brief Add one class's overlap contributions into the n_ao x n_ao matrix
/// @p s, on @p stream: the generated kernel s_{la}{lb}, one thread per record.
///
/// The record arrays are @c PrimitivePairClass's fields on the device, @p n_rec
/// entries each (three per record for @p xpa and @p xpb). Each value is
/// atomically added at (m, n) and at (n, m), so @p s must be zeroed before the
/// first class of a build and is symmetric after the last.
///
/// @p n_rec must be positive, and @p la <= @p lb <= kOverlapMaxAngularMomentum.
/// The kernels index @p s as a 32-bit `m * n_ao + n`, so n_ao * n_ao must fit an
/// int32_t.
///
/// @return The launch status, as the integer value of `wwrError_t`
///         (wwrErrorInvalidValue for a class with no kernel). The enum itself
///         is only in runtime.h's device section, which a host GMF does not
///         see; the caller casts it back. Launch errors only: the kernel's own
///         faults surface at the next synchronization.
[[nodiscard]] int32_t overlap_class(wwr::wwrStream_t stream, int32_t la, int32_t lb, int32_t n_rec,
                                    int32_t n_ao, const double *zeta, const double *xpa,
                                    const double *xpb, const double *s_rec, const int32_t *off_a,
                                    const int32_t *off_b, double *s);

} // namespace free_breakfast::device
