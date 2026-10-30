/// @file
/// @brief The K build's launcher over the generated kb<la, lc, lb, ld> and
/// kbr<la, lc, lb, ld>, the small kernel that makes its rank-to-AO table, and
/// the K gradient's launcher over gk<la, lc, lb, ld> with its record-atom
/// gather, declared once for the module that calls them and the device TU
/// that defines them.
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

/// @brief Add K class (bra.la, bra.lb | ket.la, ket.lb) of @p bra against @p ket
/// into the n_ao x n_ao @p k, on @p stream: one warp per block of @p blocks,
/// the block's kets strip-mined across its lanes, every lane walking its bras,
/// and lane 0 adding the contracted K_ab block into @p k at the block's AO
/// offsets.
///
/// A class whose every shell is at most kExchangeKbMaxMomentum (d) launches
/// kb<bra.la, bra.lb, ket.la, ket.lb>, which forms E inline; any other, through
/// kExchangeMaxMomentum (f), the rolled kbr<...>, which reads @p bra's e_bra
/// and @p ket's e_ket (exchange_e_table) and its class tables, and is the same
/// K to rounding.
///
/// @p blocks is the launch's table, @p n_blk blocks (exchange_blocks, the
/// bra's OrderedPairClass::blocks): each a
/// range of bra records of one first shell a and a range of ket records of one
/// first shell b, and their AO offsets; @p factor, the table's
/// (exchange_factors), is each block's density factor F_ab. Every range must be non-increasing in record_bound, as
/// each of the classes' groups is: the kernel breaks out of its bra loop at the
/// first bra with `Ip * Iq[ket0] * F_ab < tau`, and each lane out of its ket
/// loop at its first ket with `Ip * Iq * F_ab < tau`, Ip and Iq the records'
/// dressed bounds.
/// @p dc is @p bra's Dcd rows for ld = ket.lb (OrderedPairClass::dc,
/// ordered_pair_bra_rows); the kets' `dd` is ket.rank_b. @p dcd is the
/// (bra.lb, ket.lb) gather (exchange_gather_dcd), @p n_dc its row count.
///
/// The adds are atomic, so launches of several class pairs into one @p k need
/// no ordering among them; zero @p k before the first.
///
/// @return The launch status (wwrErrorInvalidValue for a class pair with no
///         kernel, a kbr pair whose bra or ket has no E table, a non-positive
///         @p n_blk or a non-positive @p n_ao).
[[nodiscard]] wwr::wwrError_t exchange_k(wwr::wwrStream_t stream, const OrderedPairClassView &bra,
                                         const int32_t *dc, const OrderedPairClassView &ket,
                                         int32_t n_dc, double tau, const double *dcd, int32_t n_blk,
                                         const int32_t *blocks, const double *factor, int32_t n_ao,
                                         double *k);

/// @brief Write each record's atom from its AO offset, `atom[r] =
/// ao_atom[off[r]]` for r below @p n_rec, on @p stream: gk's atom_a and
/// atom_c, which the ordered classes do not carry.
///
/// @return The launch status (wwrErrorInvalidValue for a non-positive
///         @p n_rec).
[[nodiscard]] wwr::wwrError_t exchange_record_atoms(wwr::wwrStream_t stream, int32_t n_rec,
                                                    const int32_t *off, const int32_t *ao_atom,
                                                    int32_t *atom);

/// @brief Add the K gradient class (bra.la, bra.lb | ket.la, ket.lb) of @p bra
/// against @p ket into the n_atom x 3 @p g, on @p stream: gk<bra.la, bra.lb,
/// ket.la, ket.lb>, one warp per bra record (a, c), every ket record (b, d)
/// strip-mined across its lanes, unscreened, and lane 0 adding the record's
/// derivative on its two centres into G[atom_a] and G[atom_c].
///
/// That is the bra derivative of `E_K = 1/2 SUM D_ab D_cd (ac|bd)` over the
/// class pair's ordered quartets; for a symmetric @p m it equals the ket's, so
/// the launches of every (bra class, ket class) of a set sum to the whole
/// dE_K/dR. @p m is the n_ao x n_ao Cartesian density, read at the records' AO
/// offsets. @p atom_a and @p atom_c are the bra's records' two atoms
/// (exchange_record_atoms of its off_a and off_b).
///
/// The adds are atomic, so launches into one @p g need no ordering among
/// them; zero @p g before the first.
///
/// @return The launch status (wwrErrorInvalidValue for a class pair with no
///         kernel, either class with no records, or a non-positive @p n_ao or
///         @p n_atom).
[[nodiscard]] wwr::wwrError_t exchange_gradient_class(wwr::wwrStream_t stream,
                                                      const OrderedPairClassView &bra,
                                                      const int32_t *atom_a, const int32_t *atom_c,
                                                      const OrderedPairClassView &ket, int32_t n_ao,
                                                      int32_t n_atom, const double *m, double *g);

} // namespace free_breakfast::device
