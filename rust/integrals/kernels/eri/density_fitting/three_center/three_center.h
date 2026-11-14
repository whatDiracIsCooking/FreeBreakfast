/* The DF three-center integrals' launch wrappers: the C header forked from
 * src/integrals/eri/density_fitting/three_center/three_center_bridge.h.
 * three_center_launch.cu defines them; fb_integrals.h includes this, and
 * devtools/rust-bindings.py reads it into fb-integrals-sys. Plain C: no
 * templates, no vendor types. */
#pragma once

#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The largest momentum of an orbital shell m or n fb_three_center_class has
/// a kernel for, f: the la and lb extents of the eri3 table
/// tools/emit/df_kernels.py writes, which three_center_launch.cu
/// static_asserts against it and against the orbital basis's limit.
#define FB_THREE_CENTER_MAX_MOMENTUM ((int32_t)3)

/// The largest momentum of an auxiliary shell P fb_three_center_class has a
/// kernel for, i: the lP extent of the eri3 table, as
/// FB_METRIC_MAX_MOMENTUM is the metric's. It is above the orbital limit on
/// purpose.
#define FB_THREE_CENTER_AUX_MAX_MOMENTUM ((int32_t)6)

/// The fields of one bra record, each a segment of n_bra_rec doubles, in this
/// order: zeta, P's x, y and z, P - A's, P - B's, and s = c_a c_b
/// (pi / zeta)^{3/2}: the primitive-pair records' fields of the same names.
#define FB_THREE_CENTER_BRA_FIELDS ((int32_t)11)

/// The fields of one aux record, each a segment of n_aux_rec doubles, in this
/// order: the centre's x, y and z, the exponent alpha, and the weight
/// s = c (pi / alpha)^{3/2}: the metric's aux record.
#define FB_THREE_CENTER_AUX_FIELDS ((int32_t)5)

/// The fields of one cell (an AO shell pair crossed with an aux shell), each
/// a segment of n_cell int32s, in this order: the pair's bra record range
/// [ab_rec0, ab_rec1), the aux shell's record range [p_rec0, p_rec1), and the
/// first Cartesian AO of the pair's two shells and of the aux shell, off_a,
/// off_b and off_p.
#define FB_THREE_CENTER_CELL_FIELDS ((int32_t)7)

/// Writes class (la, lb, lp) of the Cartesian (mn|P) into the n_ao x n_ao x
/// n_aux eri3, element (m, n, P) at (m n_ao + n) n_aux + P, on stream: the
/// generated eri3<la, lb, lp>, one thread per cell.
///
/// bra is FB_THREE_CENTER_BRA_FIELDS segments of n_bra_rec doubles, aux
/// FB_THREE_CENTER_AUX_FIELDS segments of n_aux_rec, and cells
/// FB_THREE_CENTER_CELL_FIELDS segments of n_cell int32s, each cell's pair of
/// momenta (la, lb) and its aux shell of lp. Each element is a plain store,
/// with its (m, n) mirror: every element belongs to exactly one cell, so
/// classes launched into one eri3 need no ordering among them and no zeroing
/// before the first, and a run is bit-reproducible. A diagonal pair
/// (off_a == off_b) writes its whole block. Every array is device memory.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null array, a count not positive
/// or an n_ao and n_aux whose n_ao * n_ao * n_aux does not fit the kernels'
/// 32-bit index, and FB_LAUNCH_UNSUPPORTED for a class with no kernel
/// (0 <= la <= lb <= FB_THREE_CENTER_MAX_MOMENTUM with
/// 0 <= lp <= FB_THREE_CENTER_AUX_MAX_MOMENTUM has one), each enqueueing
/// nothing; otherwise the launch status. Launch errors only: the kernel's own
/// faults surface at the next synchronization.
fb_launch_status fb_three_center_class(wwr_stream stream, int32_t la, int32_t lb, int32_t lp,
                                       int32_t n_bra_rec, const double *bra, int32_t n_aux_rec,
                                       const double *aux, int32_t n_cell, const int32_t *cells,
                                       int32_t n_ao, int32_t n_aux, double *eri3);

/// output = half of input plus half of its (m, n) transpose, on stream:
/// element (P, n, m) at P + n_aux (n + n_ao m) of each, so output is exactly
/// symmetric in m and n. input and output are device arrays of
/// n_ao * n_ao * n_aux doubles.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null array, a count not positive,
/// an n_ao and n_aux whose n_ao * n_ao * n_aux does not fit a 32-bit index,
/// or an input and output that overlap, each enqueueing nothing; otherwise the
/// launch status.
fb_launch_status fb_three_center_symmetrize(wwr_stream stream, int32_t n_ao, int32_t n_aux,
                                            const double *input, double *output);

#ifdef __cplusplus
}
#endif
