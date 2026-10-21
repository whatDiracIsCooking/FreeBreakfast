/// @file
/// @brief The hand-written device steps the generated kb kernels leave to
/// their launcher: max|D|, the Dcd gather, the block table, and the shell
/// starts and mirror that complete the symmetric sweep's K, declared once for
/// the module that calls them and the device TU that defines them.
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
/// once per basis, and a record's `dc` / `dd` (kb's row of Dcd) must be
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

/// @brief The int32s per block of a kb launch's table: bra0, bra1, ket0, ket1,
/// ko_a, ko_b, each a segment of n_blk.
inline constexpr int32_t kExchangeBlockFields = 6;

/// @brief Write max |D| over the n_ao x n_ao @p density into the one double
/// @p dmax, on @p stream: kb's `maxD`, the global density factor of its
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
/// kb's @p dcd, on @p stream.
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

/// @brief Write the block table of one kb launch, a bra class against a ket
/// class, into @p blocks, on @p stream: one block per (bra group, ket group),
/// or with @p symmetric only those whose bra shell is at most the ket shell.
///
/// Each side is given by its groups by first shell: @p n_a bra groups, group g
/// the bra records [@p a_start[g], @p a_start[g + 1]), and @p a_off the bra
/// records' first-shell AO offset (OrderedPairClass::off_a); likewise @p n_b,
/// @p b_start and @p b_off for the kets. The groups of each side must be in
/// basis order, so AO offsets rise with them and `a <= b` is
/// `a_off[a_start[ga]] <= b_off[b_start[gb]]`.
///
/// @p blocks is @ref kExchangeBlockFields segments of @p n_blk int32s, kb's
/// bra0, bra1, ket0, ket1, ko_a and ko_b in that order, written whole: block
/// blk is bra group ga against ket group gb, bra-major, the ket groups of one
/// bra group ascending:
///
///     bra0 = a_start[ga], bra1 = a_start[ga + 1], ko_a = a_off[bra0]
///     ket0 = b_start[gb], ket1 = b_start[gb + 1], ko_b = b_off[ket0]
///
/// @p n_blk must be the number of blocks so chosen, n_a n_b when not
/// @p symmetric (the host counts it from the classes' groups; nothing here
/// checks it, and a smaller one is written out of bounds).
///
/// @return The launch status (wwrErrorInvalidValue if @p n_a, @p n_b or
///         @p n_blk is not positive, or the table has more than INT32_MAX
///         elements).
[[nodiscard]] wwr::wwrError_t exchange_blocks(wwr::wwrStream_t stream, int32_t n_a,
                                              const int32_t *a_start, const int32_t *a_off,
                                              int32_t n_b, const int32_t *b_start,
                                              const int32_t *b_off, bool symmetric, int32_t n_blk,
                                              int32_t *blocks);

/// @brief Write the first AO of its shell for every AO of the @p n_s shells of
/// momentum @p l, `start[off[r] + i] = off[r]` for i < n_x(l), on @p stream.
///
/// @p off is the momentum's AO offsets by rank (the gather's `off_c`); one
/// launch per momentum the basis reaches writes the n_ao entries of @p start
/// whole, for @ref exchange_mirror.
///
/// @return The launch status (wwrErrorInvalidValue if @p l is outside
///         [0, kMaxAngularMomentum] or @p n_s is not positive).
[[nodiscard]] wwr::wwrError_t exchange_shell_starts(wwr::wwrStream_t stream, int32_t l, int32_t n_s,
                                                    const int32_t *off, int32_t *start);

/// @brief Complete the symmetric sweep's @p k from its a <= b shell blocks, on
/// @p stream: every element of a lower shell block, whose column is in an
/// earlier shell than its row, is set to its transpose,
///
///     k[m][q] = k[q][m]   for q < start[m]
///
/// @p start the first AO of each AO's shell (@ref exchange_shell_starts). The
/// a == b blocks, computed whole, are left as they are. It sets, never adds:
/// the lower blocks' contents are overwritten, and every kb launch into the
/// upper ones must be ordered before it.
///
/// @return The launch status (wwrErrorInvalidValue if @p n_ao is not
///         positive).
[[nodiscard]] wwr::wwrError_t exchange_mirror(wwr::wwrStream_t stream, int32_t n_ao,
                                              const int32_t *start, double *k);

} // namespace free_breakfast::device
