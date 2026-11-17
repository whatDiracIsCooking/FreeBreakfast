/* The conventional ERI's launch wrappers: the packed four-index tensor, its
 * exchange-ordered gather, and the packing of a density and unpacking of a
 * packed result around the GEMVs. No C++ bridge to fork: the C++ tree has no
 * conventional mode, so this header and conventional_launch.cu are rust/'s
 * own, over FelizJueves's eri/full_eri units. fb_integrals.h includes this,
 * and devtools/rust-bindings.py reads it into fb-integrals-sys. Plain C: no
 * templates, no vendor types.
 *
 * THE PACKING. Cartesian AO pairs m >= n are numbered
 * pidx(m, n) = m (m + 1) / 2 + n, symmetric in its arguments, and
 * n_pp = n_ao (n_ao + 1) / 2 of them. The packed ERI is the n_pp x n_pp
 * matrix ERI_pp[pidx(m, n)][pidx(l, s)] = (mn|ls), exactly symmetric, and the
 * exchange-ordered one X_pp[pidx(l, s)][pidx(m, n)] = ((ml|ns) + (ms|nl)) / 2,
 * exactly symmetric too. For a symmetric D, J_pp = ERI_pp Dp and
 * K_pp = X_pp Dp, Dp[pidx(l, s)] = D_ls + D_sl off the diagonal and D_ll on
 * it, and J_mn = J_pp[pidx(m, n)]. Each matrix is row-major, element (i, j)
 * at i n_pp + j; being symmetric, it is the same matrix column-major. */
#pragma once

#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"
#include "primitive_pair_class_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The largest momentum of a shell fb_conventional_eri_class has a kernel
/// for, d: the extents of the eri table tools/emit/conventional_kernels.py
/// writes (FelizJueves's ERI_L_MAX), which conventional_launch.cu
/// static_asserts against it.
#define FB_CONVENTIONAL_MAX_MOMENTUM ((int32_t)2)

/// The largest Cartesian AO count any wrapper here takes: FelizJueves's
/// N_AO_MAX, where the packed ERI is about 1 GB of doubles and the pair of
/// it and X_pp about 2 GB. Its n_pp^2 is far inside the kernels' 32-bit
/// index, which holds to n_ao = 304.
#define FB_CONVENTIONAL_MAX_AO ((int32_t)150)

/// The fields of one shell pair, each a segment of n_pair int32s, in this
/// order: the pair's record range [rec0, rec1) among its class's records,
/// and the first Cartesian AO of its two shells, off_a and off_b.
#define FB_CONVENTIONAL_PAIR_FIELDS ((int32_t)4)

/// The fields of one task (a bra shell pair crossed with a ket shell pair),
/// each a segment of n_task int32s, in this order: the bra pair's index
/// among bra_pairs and the ket pair's among ket_pairs.
#define FB_CONVENTIONAL_TASK_FIELDS ((int32_t)2)

/// Writes class (bra->la, bra->lb | ket->la, ket->lb) of the packed ERI into
/// eri, n_pp x n_pp doubles, on stream: the generated eri<la, lb, lc, ld>,
/// kernels/eri_task_threads.h's threads per task (one through (pp|pp); from
/// a d shell the low-stack form's, one bra row a thread).
///
/// The bra and ket records are bra's and ket's, read in place: each one's
/// zeta, p_*, pa_*, pb_* and s, n_rec long; ket's P, P - A and P - B are the
/// kernel's Q, Q - C and Q - D. bra_pairs is FB_CONVENTIONAL_PAIR_FIELDS
/// segments of n_bpair int32s, each pair's record range inside
/// [0, bra->n_rec) and its two offsets opening blocks inside n_ao; ket_pairs
/// likewise over ket. tasks is FB_CONVENTIONAL_TASK_FIELDS segments of
/// n_task int32s, indices into the two pair tables. pidx is n_ao x n_ao
/// int32s, element m n_ao + n holding pidx(m, n). Each element is a plain store,
/// with its bra-ket mirror; a diagonal pair (off_a == off_b) writes its
/// lower triangle alone. A set of tasks that names every unordered pair of
/// shell pairs once owns every element once, so classes launched into one
/// eri need no ordering among them and a run is bit-reproducible; the
/// elements no task owns are untouched. Every array is device memory.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null bra, ket or array (one of
/// the eleven record fields each side reads among them), a count not
/// positive (each side's n_rec among them), an n_ao above
/// FB_CONVENTIONAL_MAX_AO or n_task times the class's threads per task past
/// INT32_MAX (the kernels' thread index), and FB_LAUNCH_UNSUPPORTED for a class with no
/// kernel (0 <= la <= lb <= FB_CONVENTIONAL_MAX_MOMENTUM and
/// 0 <= lc <= ld <= FB_CONVENTIONAL_MAX_MOMENTUM with (la, lb) not after
/// (lc, ld) has one), each enqueueing nothing; otherwise the launch status.
/// Launch errors only: the kernel's own faults surface at the next
/// synchronization.
fb_launch_status fb_conventional_eri_class(wwr_stream stream,
                                           const fb_primitive_pair_class_view *bra,
                                           const fb_primitive_pair_class_view *ket, int32_t n_bpair,
                                           const int32_t *bra_pairs, int32_t n_kpair,
                                           const int32_t *ket_pairs, int32_t n_task,
                                           const int32_t *tasks, int32_t n_ao, const int32_t *pidx,
                                           double *eri);

/// Writes X_pp from the packed ERI on stream: the generated eri_kperm, one
/// thread per packed column, every element of x a plain store, exactly
/// symmetric. pm and pn are n_pp int32s, the AO pair (m, n), m >= n,
/// of each packed index, and pidx as fb_conventional_eri_class takes it; eri and
/// x are n_pp x n_pp doubles each. Every array is device memory.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null array, an n_ao not in
/// 1..=FB_CONVENTIONAL_MAX_AO, or an eri and x that overlap, each enqueueing
/// nothing; otherwise the launch status.
fb_launch_status fb_conventional_exchange(wwr_stream stream, int32_t n_ao, const int32_t *pm,
                                          const int32_t *pn, const int32_t *pidx, const double *eri,
                                          double *x);

/// Packs the n_ao x n_ao density d into dp, n_pp doubles, on stream:
/// element pidx(l, s) of dp is d_ls + d_sl for l > s and d_ll on the
/// diagonal, one thread per packed pair. d is read as stored; its two
/// layouts give the same dp. Both arrays are device memory.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null array or an n_ao not in
/// 1..=FB_CONVENTIONAL_MAX_AO, enqueueing nothing; otherwise the launch
/// status.
fb_launch_status fb_conventional_pack(wwr_stream stream, int32_t n_ao, const double *d, double *dp);

/// Unpacks vp, n_pp doubles, into the symmetric n_ao x n_ao v on stream:
/// v_mn is element pidx(m, n) of vp, one thread per element, so v_mn and v_nm
/// are the same bits. Both arrays are device memory.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null array or an n_ao not in
/// 1..=FB_CONVENTIONAL_MAX_AO, enqueueing nothing; otherwise the launch
/// status.
fb_launch_status fb_conventional_unpack(wwr_stream stream, int32_t n_ao, const double *vp,
                                        double *v);

#ifdef __cplusplus
}
#endif
