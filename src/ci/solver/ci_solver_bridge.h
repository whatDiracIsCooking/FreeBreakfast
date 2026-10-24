/// @file
/// @brief The solver's launchers, declared once for the module that calls them
/// and the device TU that defines them: S² on a block of CI vectors, and the
/// diagonal preconditioner.
///
/// Included by solver.cpp in its GLOBAL MODULE FRAGMENT, and by ci_solver.cu
/// directly: a declaration in a module's purview gets module linkage and can
/// never bind to a definition compiled in a plain TU, which a .cu is (see
/// src/integrals/one_electron/overlap/overlap_bridge.h).
///
/// The layout is the GPU sigma's (see ../sigma/ci_sigma_bridge.h): a
/// determinant (I_α, I_β) is element I_α · n_beta + I_β, a block of vectors is
/// column-major with leading dimension n_det, and a *row* is one (vector j,
/// determinant I) pair, row j · n_det + I. The tables are S²'s own:
///
/// - An α link table holds the α set's replacement lists, `alpha_per` links per
///   string, three int32_t per link a†_p a_q: the target string's address, the
///   index q · n + p of its β partner a†_q a_p in a β slot table, and the sign.
/// - A β slot table holds, per β string and n × n over (p, q) row-major, the
///   replacement a†_p a_q of that string as sign · (target + 1), or 0 when it
///   annihilates the string.
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

namespace free_breakfast::device {

/// @brief For every one of the @p rows rows of the block @p x,
/// y = (accumulate ? y : 0) + scale · ((S² − shift) x), with
/// S² = constant − Σ_pq E^β_pq E^α_qp and @p constant = S_z (S_z + 1) + N_β.
///
/// One thread per row gathers over its own α links and their β partners, since
/// S² is real symmetric. @p y has @p x's layout and must not overlap it.
[[nodiscard]] wwr::wwrError_t ci_spin_square(wwr::wwrStream_t stream, int32_t rows, int32_t n_det,
                                             int32_t n_beta, int32_t n, const int32_t *alpha_links,
                                             int32_t alpha_per, const int32_t *beta_slots,
                                             double constant, double shift, double scale,
                                             bool accumulate, const double *x, double *y);

/// @brief correction_i = residual_i / (theta − shifted_i) for the @p n
/// elements of one column, the denominator held off zero at ±1e-3.
[[nodiscard]] wwr::wwrError_t ci_precondition(wwr::wwrStream_t stream, int32_t n, double theta,
                                              const double *shifted, const double *residual,
                                              double *correction);

} // namespace free_breakfast::device
