/// @file
/// @brief The K build's launchers over the generated k<la, lc, lb, ld>, and the
/// three small kernels that make its per-record inputs, declared once for the
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

/// @brief One bra class's screen bound with the density factor folded in,
/// `ip[r] = bra.bound[r] * *dmax`, on @p stream; @p ip is bra.n_rec doubles,
/// written whole.
///
/// kernel_k takes maxD by value, a host double, but max|D| is on the device
/// (exchange_density_max), and reading it back would synchronize the host; so
/// k is launched with maxD = 1 and the bra's Ip carries it. The screen
/// `Ip * Iq * maxD < tau` is the same product.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no
///         records).
[[nodiscard]] wwr::wwrError_t exchange_bra_bound(wwr::wwrStream_t stream,
                                                 const OrderedPairClassView &bra,
                                                 const double *dmax, double *ip);

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

/// @brief Add k<bra.la, bra.lb, ket.la, ket.lb> of the first @p n_bra records
/// of @p bra against the ket records [@p start, @p start + @p count) of @p ket
/// into @p block, on @p stream: one warp per bra record, the kets
/// strip-mined across its lanes.
///
/// @p n_bra is a prefix of the bra class: the symmetric sweep passes the
/// records whose first shell a is at most the kets' b (a prefix, the groups
/// being in basis order), and the block's later rows are left untouched.
///
/// The kets must all have one first shell b (one of ket's groups) and be
/// non-increasing in bound: each lane breaks at its first ket with
/// `Ip * Iq < tau`. @p ip and @p dc are @p bra's, from
/// @ref exchange_bra_bound and @ref exchange_bra_rows (for ld = ket.lb); @p dcd is the
/// (bra.lb, ket.lb) gather (exchange_gather_dcd), @p n_dc its row count.
/// @p block is the (bra class, b) block, bra.n_rec x n_x(bra.la) n_x(ket.la),
/// row-major, added to atomically: zero it before the first launch into it.
///
/// @return The launch status (wwrErrorInvalidValue for a class pair with no
///         kernel, an @p n_bra outside [1, bra.n_rec], or an empty or
///         out-of-range ket range).
[[nodiscard]] wwr::wwrError_t exchange_k(wwr::wwrStream_t stream, const OrderedPairClassView &bra,
                                         int32_t n_bra, const double *ip, const int32_t *dc,
                                         const OrderedPairClassView &ket, int32_t start,
                                         int32_t count, int32_t n_dc, double tau, const double *dcd,
                                         double *block);

} // namespace free_breakfast::device
