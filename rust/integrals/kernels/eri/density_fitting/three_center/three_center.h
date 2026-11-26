/* The DF three-center integrals' packed launch wrapper, its shape the C
 * header forked from
 * src/integrals/eri/density_fitting/three_center/three_center_bridge.h.
 * three_center_launch.cu defines it; fb_integrals.h includes this, and
 * devtools/rust-bindings.py reads it into fb-integrals-sys. Plain C: no
 * templates, no vendor types. */
#pragma once

#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"
#include "primitive_pair_class_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The largest momentum of an orbital shell m or n
/// fb_three_center_packed_class and fb_three_center_spherical_class have a
/// kernel for, f: the la and lb extents of the speri3 and ssperi3 tables
/// tools/emit/packed_three_center_kernels.py writes, which
/// three_center_launch.cu static_asserts against it and against the orbital
/// basis's limit.
#define FB_THREE_CENTER_MAX_MOMENTUM ((int32_t)3)

/// The largest momentum of an auxiliary shell P fb_three_center_packed_class
/// and fb_three_center_spherical_class have a kernel for, i: the lP extent of
/// the speri3 and ssperi3 tables, as
/// FB_METRIC_MAX_MOMENTUM is the metric's. It is above the orbital limit on
/// purpose.
#define FB_THREE_CENTER_AUX_MAX_MOMENTUM ((int32_t)6)

/// The fields of one aux record, each a segment of n_aux_rec doubles, in this
/// order: the centre's x, y and z, the exponent alpha, and the weight
/// s = c (pi / alpha)^{3/2}: the metric's aux record.
#define FB_THREE_CENTER_AUX_FIELDS ((int32_t)5)

/// The fields of one packed cell (an AO shell pair crossed with an aux
/// shell), each a segment of n_cell int32s, in this order: the pair's bra
/// record range [ab_rec0, ab_rec1) among its class's records, the aux shell's
/// record range [p_rec0, p_rec1), the cell's destination offset dst, and its
/// diagonal flag diag, 1 for a shell's pair with itself and 0 otherwise.
#define FB_THREE_CENTER_PACKED_CELL_FIELDS ((int32_t)6)

/// Writes class (cls->la, cls->lb, lp) of the (mn|Q) into the packed eri3,
/// m and n Cartesian and Q the aux shell's real solid harmonics, n_col columns
/// of n_aux doubles, n_aux the aux basis's spherical functions, on stream: the
/// generated speri3<la, lb, lp>, one thread per cell, each cell's whole block
/// at its destination, element (a, b, q) at dst + n_aux (a nB + b) + q, a the
/// bra shell's Cartesian (momentum la), b the ket's (lb, nB of them) and q
/// the aux shell's solid harmonic, 2 lp + 1 of them in the order m = -lp ..=
/// lp (fb_basis::spherical_transform_matrix's block of lp, transposed). A
/// pair whose block opens at column c, crossed with an aux shell whose first
/// spherical function is off_p, has dst = c n_aux + off_p. A diagonal cell
/// (diag = 1, read only when la == lb) stores its full square block; any
/// other stores its block once, with no (m, n) mirror.
///
/// The bra records are cls's, read in place: its zeta, p_*, pa_*, pb_* and
/// s, each cls->n_rec long. aux is FB_THREE_CENTER_AUX_FIELDS segments of
/// n_aux_rec doubles, and cells FB_THREE_CENTER_PACKED_CELL_FIELDS segments
/// of n_cell int32s, each cell a shell pair of cls's class, its bra record range inside
/// [0, cls->n_rec), with an aux shell of lp, its block inside eri3. Each
/// element is a plain store: cells whose blocks do not overlap need no
/// ordering among them and no zeroing before the first, and a run is
/// bit-reproducible. Every array is device memory.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null cls, a null array (one of
/// cls's eleven it reads among them), a count not positive (cls->n_rec among
/// them) or an n_col and n_aux whose n_col * n_aux does not fit the kernels'
/// 32-bit index, and FB_LAUNCH_UNSUPPORTED for a class with no kernel
/// (0 <= la <= lb <= FB_THREE_CENTER_MAX_MOMENTUM with
/// 0 <= lp <= FB_THREE_CENTER_AUX_MAX_MOMENTUM has one), each enqueueing
/// nothing; otherwise the launch status. Launch errors only: the kernel's own faults surface at the
/// next synchronization.
fb_launch_status fb_three_center_packed_class(wwr_stream stream,
                                              const fb_primitive_pair_class_view *cls, int32_t lp,
                                              int32_t n_aux_rec, const double *aux, int32_t n_cell,
                                              const int32_t *cells, int32_t n_col, int32_t n_aux,
                                              double *eri3);

/// Writes class (cls->la, cls->lb, lp) of the (mu nu|Q) into the packed eri3,
/// all three indices real solid harmonics, on stream: the generated
/// ssperi3<la, lb, lp> (FelizJueves#745), fb_three_center_packed_class's
/// arguments, cells, limits, refusals and stores with the bra transformed in
/// the kernel too, so a cell's block is (2 la + 1)(2 lb + 1)(2 lp + 1)
/// elements, element (mu, nu, q) at dst + n_aux (mu (2 lb + 1) + nu) + q, mu
/// the bra shell's solid harmonic and nu the ket's, each in the order
/// m = -l ..= l. n_col counts spherical bra columns, so a pair whose block
/// opens at spherical column c, crossed with an aux shell whose first
/// spherical function is off_p, has dst = c n_aux + off_p.
fb_launch_status fb_three_center_spherical_class(wwr_stream stream,
                                                 const fb_primitive_pair_class_view *cls,
                                                 int32_t lp, int32_t n_aux_rec, const double *aux,
                                                 int32_t n_cell, const int32_t *cells,
                                                 int32_t n_col, int32_t n_aux, double *eri3);

#ifdef __cplusplus
}
#endif
