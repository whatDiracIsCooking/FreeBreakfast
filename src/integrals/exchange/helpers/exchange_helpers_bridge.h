/// @file
/// @brief The three hand-written device steps the generated kernel_k leaves to
/// its launcher: max|D|, the Dcd gather and the block scatter, declared once
/// for the module that calls them and the device TU that defines them.
///
/// Included by the exchange module's implementation unit in its GLOBAL MODULE
/// FRAGMENT, and by exchange_helpers.cu directly, for the reason
/// overlap_bridge.h gives: a declaration in a module's purview gets module
/// linkage and can never bind to a definition compiled in a plain TU.
/// wwrStream_t and wwrError_t come from a different wwr header in each pass, as
/// there. test/gtest/exchange_helpers_gpu.cpp includes it too, and so imports
/// nothing that carries wwr.runtime_api (its header says why).
///
/// Every pointer is device memory, every launch is asynchronous on @p stream,
/// and every matrix is n_ao x n_ao Cartesian, indexed `m * n_ao + n`. The
/// launchers return launch errors only, as overlap_class.
///
/// A shell's RANK is its index among the basis's shells of the same angular
/// momentum, in basis order: the p shells of a basis are ranks 0, 1, ... in the
/// order they appear, whatever s or d shells lie between them. It depends only
/// on the basis, so the shell offsets by momentum the gather takes are built
/// once per basis, and a record's `dc` / `dd` (kernel_k's row of Dcd) must be
/// formed from the same ranks.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief Write max |D| over the n_ao x n_ao @p density into the one double
/// @p dmax, on @p stream: kernel_k's `maxD`, the global density factor of its
/// screen `Ip * Iq * maxD < tau`.
///
/// One block folds the whole matrix, so the result is exact and needs no
/// zeroing first; a NaN anywhere is the result.
///
/// @return The launch status (wwrErrorInvalidValue if @p n_ao is not
///         positive).
[[nodiscard]] wwr::wwrError_t exchange_density_max(wwr::wwrStream_t stream, int32_t n_ao,
                                                   const double *density, double *dmax);

/// @brief Gather the (lc, ld) shell blocks of the n_ao x n_ao @p density into
/// kernel_k's @p dcd, on @p stream.
///
/// @p off_c holds the AO offset of each of the @p n_c shells of momentum @p lc,
/// by rank, and @p off_d those of the @p n_d shells of momentum @p ld. With
/// n_x = (l_x + 1)(l_x + 2) / 2 Cartesians per shell, @p dcd is
/// (n_c n_d) x (n_x(lc) n_x(ld)), row-major and written whole:
///
///     dcd[rank_c * n_d + rank_d][i * n_x(ld) + j] = D[off_c[rank_c] + i][off_d[rank_d] + j]
///
/// so the kernel's `dc[bra] + dd[ket]` names the (c, d) block when
/// dc = rank_c * n_d and dd = rank_d. Nothing is mirrored: (lc, ld) and
/// (ld, lc) are two gathers.
///
/// @return The launch status (wwrErrorInvalidValue if @p lc or @p ld is
///         outside [0, kMaxAngularMomentum], @p n_ao, @p n_c or @p n_d is not
///         positive, or @p dcd has more than INT32_MAX elements).
[[nodiscard]] wwr::wwrError_t exchange_gather_dcd(wwr::wwrStream_t stream, int32_t lc, int32_t ld,
                                                  int32_t n_ao, const double *density, int32_t n_c,
                                                  const int32_t *off_c, int32_t n_d,
                                                  const int32_t *off_d, double *dcd);

/// @brief Add one (bra class, b shell) block of kernel_k's output into the
/// n_ao x n_ao matrix @p k, on @p stream, atomically.
///
/// @p block is @p n_bra rows of n_x(la) n_x(lb), row-major (the kernel's
/// `K[bra, a n_b + b]`); row r is the bra record whose first shell starts at
/// AO @p off_a[r], and every row's second index is the b shell at AO @p off_b:
///
///     k[off_a[r] + i][off_b + j] += block[r][i * n_x(lb) + j]
///
/// No mirroring: the ordered sweep computes K[a][b] and K[b][a] each directly.
/// Each element is added atomically, so blocks may be scattered from several
/// streams at once, and @p k must be zeroed before the first.
///
/// @return The launch status (wwrErrorInvalidValue if @p la or @p lb is
///         outside [0, kMaxAngularMomentum], @p n_ao or @p n_bra is not
///         positive, @p off_b is negative, or @p block has more than INT32_MAX
///         elements).
[[nodiscard]] wwr::wwrError_t exchange_scatter_block(wwr::wwrStream_t stream, int32_t la,
                                                     int32_t lb, int32_t n_ao, int32_t n_bra,
                                                     const int32_t *off_a, int32_t off_b,
                                                     const double *block, double *k);

} // namespace free_breakfast::device
