/// @file
/// @brief The ket pre-contraction's launcher, declared once for the module that
/// calls it and the device TU that defines it.
///
/// Included by dq.cpp in its GLOBAL MODULE FRAGMENT, and by dq_launch.cu
/// directly, for the reason overlap_bridge.h gives: a declaration in a
/// module's purview gets module linkage and can never bind to a definition
/// compiled in a plain TU. wwrStream_t and wwrError_t come from a different wwr
/// header in each pass, as there.

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

/// @brief One class's inputs to every later step, on @p stream: each record's
/// weight w (2 for a shell pair of two shells, 1 for a shell against itself,
/// read off its offsets) into @p w, then the pre-contraction of the AO
/// @p density into @p dq and @p dmax (the generated dq<la, lb>). The record
/// bound, e_hermite's Ip and Iq, is the class's own (cls.bound, mmd's).
///
/// @p density is n_ao x n_ao and symmetric (it is read at one orientation of
/// each block); @p dq is q-major, n_hermite(la + lb) x n_rec (so e_hermite's
/// lanes, consecutive kets, read it contiguously), and @p w and @p dmax
/// n_rec, all written whole. cls.n_rec must be positive, cls.la <=
/// cls.lb <= kMaxAngularMomentum, and n_ao * n_ao must fit an int32_t.
///
/// @return The first launch status (wwrErrorInvalidValue for a class with no
///         kernel). Launch errors only, as overlap_class.
[[nodiscard]] wwr::wwrError_t dq_class(wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                       int32_t n_ao, const double *density, double *w, double *dq,
                                       double *dmax);

} // namespace free_breakfast::device
