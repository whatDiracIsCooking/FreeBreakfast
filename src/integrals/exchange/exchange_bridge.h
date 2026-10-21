/// @file
/// @brief The K build's launcher over the generated kb<la, lc, lb, ld>, and the
/// two small kernels that make its per-record inputs, declared once for the
/// module that calls them and the device TU that defines them.
///
/// Included by exchange.cpp in its GLOBAL MODULE FRAGMENT, and by
/// exchange_launch.cu directly, for the reason overlap_bridge.h gives: a
/// declaration in a module's purview gets module linkage and can never bind to
/// a definition compiled in a plain TU. wwrStream_t and wwrError_t come from a
/// different wwr header in each pass, as there.
///
/// Every pointer is device memory, every launch is asynchronous on @p stream,
/// and the launchers return launch errors only, as overlap_class. A shell's
/// RANK is exchange_helpers_bridge.h's.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include "ordered_pair_class_view.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief Write the AO offset of every second shell of @p cls by its rank,
/// `off[cls.rank_b[r]] = cls.off_b[r]`, on @p stream: the gather's `off_c` /
/// `off_d` for momentum cls.lb.
///
/// @p cls should be a class (l, l): it holds every shell of momentum l as a
/// second shell (each paired with itself at least), so @p off, one entry per
/// shell of momentum l, is written whole. Records of one shell write the same
/// value, so their order is free.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no
///         records).
[[nodiscard]] wwr::wwrError_t exchange_rank_offsets(wwr::wwrStream_t stream,
                                                    const OrderedPairClassView &cls, int32_t *off);

/// @brief One bra class's rows of the (bra.lb, ld) Dcd, `dc[r] =
/// bra.rank_b[r] * n_d`, on @p stream, @p n_d the number of shells of
/// momentum ld; @p dc is bra.n_rec int32_ts, written whole. It depends on the
/// ket's ld, so a bra class has one per ld.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no
///         records or a negative @p n_d).
[[nodiscard]] wwr::wwrError_t exchange_bra_rows(wwr::wwrStream_t stream,
                                                const OrderedPairClassView &bra, int32_t n_d,
                                                int32_t *dc);

/// @brief Add kb<bra.la, bra.lb, ket.la, ket.lb> of @p bra against @p ket into
/// the n_ao x n_ao @p k, on @p stream: one warp per block of @p blocks, the
/// block's kets strip-mined across its lanes, every lane walking its bras, and
/// lane 0 adding the contracted K_ab block into @p k at the block's AO offsets.
///
/// @p blocks is the launch's table, @p n_blk blocks (exchange_blocks): each a
/// range of bra records of one first shell a and a range of ket records of one
/// first shell b, and their AO offsets; @p factor, the table's, is each block's
/// density factor F_ab. Every range must be non-increasing in record_bound, as
/// each of the classes' groups is: the kernel breaks out of its bra loop at the
/// first bra with `Ip * Iq[ket0] * F_ab < tau`, and each lane out of its ket
/// loop at its first ket with `Ip * Iq * F_ab < tau`, Ip and Iq the records'
/// dressed bounds.
/// @p dc is @p bra's, from @ref exchange_bra_rows (for ld = ket.lb); the kets'
/// `dd` is ket.rank_b. @p dcd is the (bra.lb, ket.lb) gather
/// (exchange_gather_dcd), @p n_dc its row count.
///
/// The adds are atomic, so launches of several class pairs into one @p k need
/// no ordering among them; zero @p k before the first.
///
/// @return The launch status (wwrErrorInvalidValue for a class pair with no
///         kernel, a non-positive @p n_blk or a non-positive @p n_ao).
[[nodiscard]] wwr::wwrError_t exchange_k(wwr::wwrStream_t stream, const OrderedPairClassView &bra,
                                         const int32_t *dc, const OrderedPairClassView &ket,
                                         int32_t n_dc, double tau, const double *dcd, int32_t n_blk,
                                         const int32_t *blocks, const double *factor, int32_t n_ao,
                                         double *k);

} // namespace free_breakfast::device
