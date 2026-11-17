/* The dense conventional ERI's launch wrapper: a row block of the dense
 * J-layout or K-layout matrix of the Cartesian ERI. No C++ bridge to fork:
 * the C++ tree has no conventional mode, so this header and dense_launch.cu
 * are rust/'s own, over FelizJueves's eri/full_eri dense units.
 * fb_integrals.h includes this, and devtools/rust-bindings.py reads it into
 * fb-integrals-sys. Plain C: no templates, no vendor types.
 *
 * THE MATRICES. With every Cartesian AO pair unpacked, (m, n) -> m n_ao + n,
 * the J layout is M_J[(m,n)][(l,s)] = (mn|ls) and the K layout
 * M_K[(m,n)][(l,s)] = (ml|ns), each dense n_ao^2 x n_ao^2 and row-major, so
 * J = M_J vec(D) and K = M_K vec(D) for any D, vec(D)[(l,s)] = D_ls. A row's
 * m is the bra's first AO in both layouts, so the shells of m own the
 * contiguous rows [m0 n_ao, (m0 + n_m) n_ao) of either matrix: a launch
 * writes that LOCAL row block, n_row = n_m n_ao rows of n_col = n_ao^2,
 * element (row, col) at (row - m0 n_ao) n_col + col. */
#pragma once

#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"
#include "primitive_pair_class_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The J layout: row (m, n), column (l, s) of M_J is (mn|ls). One of
/// fb_dense_eri_class's two layouts.
#define FB_DENSE_LAYOUT_J ((int32_t)0)

/// The K layout: row (m, n), column (l, s) of M_K is (ml|ns).
#define FB_DENSE_LAYOUT_K ((int32_t)1)

/// Writes class (bra->la, bra->lb | ket->la, ket->lb) of the local row block
/// eri of the dense matrix in layout (FB_DENSE_LAYOUT_J or
/// FB_DENSE_LAYOUT_K) on stream: the generated eri_dj<la, lb, lc, ld> or
/// eri_dk<la, lb, lc, ld>, one thread per task, a task an ORDERED quartet of
/// shells (a bra shell pair crossed with a ket shell pair) whose (ab|cd)
/// block it stores, element by element, plainly, with no mirror.
///
/// The bra and ket records are bra's and ket's, read in place: each one's
/// zeta, p_*, pa_*, pb_* and s, n_rec long, every ORDERED shell pair's (the
/// records between a basis and itself, la > lb among them); ket's P, P - A
/// and P - B are the kernel's Q, Q - C and Q - D. bra_pairs is
/// FB_CONVENTIONAL_PAIR_FIELDS segments, field f's starting at
/// bra_pairs + f bra_stride, of which the first n_bpair pairs are
/// launched: each pair's record range inside [0, bra->n_rec) and its two
/// AO offsets, the first's block of AOs inside [m0, m0 + n_row / n_ao) and
/// the second's inside [0, n_ao). ket_pairs is FB_CONVENTIONAL_PAIR_FIELDS
/// segments of n_kpair int32s likewise over ket, both offsets' blocks
/// inside [0, n_ao). tasks is FB_CONVENTIONAL_TASK_FIELDS segments, field
/// f's starting at tasks + f task_stride, of which the first
/// n_bpair n_kpair are launched: task t the bra pair t / n_kpair and the
/// ket pair t % n_kpair. eri is the n_row x n_ao^2 block. Every element a
/// launch's tasks name is written once, so launches into one eri need no
/// ordering among them and a run is bit-reproducible; the elements no task
/// names are untouched. Every array is device memory.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a layout not one of the two, a
/// null bra, ket or array (one of the eleven record fields each side reads
/// among them), a count not positive (each side's n_rec among them), an
/// n_bpair past bra_stride, an n_bpair n_kpair past task_stride, an n_row
/// that is not a multiple of n_ao, rows past the matrix
/// (m0 n_ao + n_row > n_ao^2), or an n_ao^2 or n_row n_ao^2 past INT32_MAX
/// (the kernels' flat index); and FB_LAUNCH_UNSUPPORTED for a class with no
/// kernel (each momentum in 0..=FB_CONVENTIONAL_MAX_MOMENTUM has one), each
/// enqueueing nothing; otherwise the launch status. Launch errors only: the
/// kernel's own faults surface at the next synchronization.
fb_launch_status fb_dense_eri_class(wwr_stream stream, int32_t layout,
                                    const fb_primitive_pair_class_view *bra,
                                    const fb_primitive_pair_class_view *ket, int32_t n_bpair,
                                    int32_t bra_stride, const int32_t *bra_pairs, int32_t n_kpair,
                                    const int32_t *ket_pairs, int32_t task_stride,
                                    const int32_t *tasks, int32_t n_ao, int32_t m0, int32_t n_row,
                                    double *eri);

/// Writes y = M x on stream, M the n_row x n_col row-major matrix m and x
/// n_col long: one block per row, each element of y the row's products
/// summed in an order fixed by n_col alone, so a row's y is the same bits
/// whatever n_row, whatever the rows beside it, and whichever device, of a
/// kind, runs it. What the dense J and K are read by, a row block per
/// device, in place of the BLAS's GEMV, whose order may depend on the
/// call's shape. Every array is device memory; y is apart from m and x.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null array, a count not
/// positive, or an n_row n_col past INT32_MAX (the flat index), enqueueing
/// nothing; otherwise the launch status.
fb_launch_status fb_dense_row_dot(wwr_stream stream, int32_t n_row, int32_t n_col, const double *m,
                                  const double *x, double *y);

#ifdef __cplusplus
}
#endif
