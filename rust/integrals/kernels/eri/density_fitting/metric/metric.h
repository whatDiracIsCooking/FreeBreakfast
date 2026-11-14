/* The DF Coulomb metric's launch wrapper: the C header forked from
 * src/integrals/eri/density_fitting/metric/metric_bridge.h. metric_launch.cu
 * defines it; fb_integrals.h includes this, and devtools/rust-bindings.py
 * reads it into fb-integrals-sys. Plain C: no templates, no vendor types. */
#pragma once

#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The largest momentum of an auxiliary shell fb_metric_class has a kernel
/// for, i: the extent of the metric table tools/emit/df_kernels.py writes,
/// which metric_launch.cu static_asserts against it. It is the aux basis's
/// limit, above the orbital one on purpose.
#define FB_METRIC_MAX_MOMENTUM ((int32_t)6)

/// The fields of one aux record, each a segment of n_rec doubles, in this
/// order: the centre's x, y and z, the exponent alpha, and the weight
/// s = c (pi / alpha)^{3/2}.
#define FB_METRIC_RECORD_FIELDS ((int32_t)5)

/// The fields of one aux shell pair (P, Q), each a segment of n_pair int32s,
/// in this order: P's record range [p_rec0, p_rec1), Q's [q_rec0, q_rec1),
/// and the two shells' first Cartesian AOs off_p and off_q.
#define FB_METRIC_PAIR_FIELDS ((int32_t)6)

/// Writes class (lp, lq) of the Cartesian metric (P|Q) into the n_aux x n_aux
/// metric, on stream: the generated metric<lp, lq>, one thread per aux shell
/// pair.
///
/// records is FB_METRIC_RECORD_FIELDS segments of n_rec doubles, every record
/// the pairs name; pairs is FB_METRIC_PAIR_FIELDS segments of n_pair int32s,
/// each pair's P of momentum lp and Q of lq. Each element is a plain store,
/// with its mirror: every element belongs to exactly one aux shell pair, so
/// classes launched into one metric need no ordering among them and no
/// zeroing before the first, and a run is bit-reproducible. A diagonal pair
/// (P == Q, only when lp == lq) writes its whole block too. Every array is
/// device memory.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null array, a count not positive
/// or an n_aux whose n_aux * n_aux does not fit the kernels' 32-bit index,
/// and FB_LAUNCH_UNSUPPORTED for a class with no kernel
/// (0 <= lp <= lq <= FB_METRIC_MAX_MOMENTUM has one), each enqueueing
/// nothing; otherwise the launch status. Launch errors only: the kernel's own
/// faults surface at the next synchronization.
fb_launch_status fb_metric_class(wwr_stream stream, int32_t lp, int32_t lq, int32_t n_rec,
                                 const double *records, int32_t n_pair, const int32_t *pairs,
                                 int32_t n_aux, double *metric);

#ifdef __cplusplus
}
#endif
