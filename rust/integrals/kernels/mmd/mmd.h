/* The primitive-pair records' launch wrappers -- the records, the bound and
 * the dressed bound -- and the structs they read and write: the C header
 * forked from src/integrals/mmd/mmd_bridge.h, without its order (mmd_order):
 * the records stay in the order the host enumerates them, each shell pair's
 * one contiguous run (mmd_launch.cu has why). mmd_launch.cu defines them;
 * fb_integrals.h includes this, and devtools/rust-bindings.py reads it into
 * fb-integrals-sys. Plain C: no templates, no vendor types. */
#pragma once

#include <stddef.h>
#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"
#include "primitive_pair_class_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The largest momentum fb_mmd_bound has a kernel for, f: the extent of the
/// pair_bound table tools/emit/one_electron_kernels.py writes, which
/// mmd_launch.cu static_asserts against it.
#define FB_PAIR_BOUND_MAX_MOMENTUM ((int32_t)3)
/// The largest momentum, of either shell, fb_ordered_pair_bound has a kernel
/// for, f: the extent of the record_bound table
/// tools/emit/exchange_kernels.py writes, which mmd_launch.cu static_asserts
/// against it.
#define FB_RECORD_BOUND_MAX_MOMENTUM ((int32_t)3)

/// What a class's records are formed from, on the device: the basis's shells
/// and primitives, and the class's shell pairs. The struct is host memory,
/// handed to a wrapper by address; every array is device memory.
///
/// A record is named by its index in the order the host would have written it
/// (shell pair, then bra primitive, then ket primitive): the shell pair p
/// whose `[rec_begin[p], rec_begin[p + 1])` holds it, then its offset in that
/// range, bra-primitive-major.
typedef struct fb_primitive_pair_source {
  /// The class's shell pairs, positive.
  int32_t n_pairs;
  /// Each shell pair's bra shell, already in la <= lb order.
  const uint32_t *bra;
  /// Each shell pair's ket shell.
  const uint32_t *ket;
  /// Each shell pair's first record, n_pairs + 1 long: `rec_begin[0]` is 0,
  /// `rec_begin[n_pairs]` the class's record count.
  const int32_t *rec_begin;
  /// Each shell's first primitive in exponent and coefficient, n_shells + 1
  /// long.
  const int32_t *prim_begin;
  /// Every primitive's exponent, shell by shell.
  const double *exponent;
  /// Every primitive's contraction coefficient, as the basis gives it.
  const double *coefficient;
  /// Each shell's centre: x.
  const double *c_x;
  /// Each shell's centre: y.
  const double *c_y;
  /// Each shell's centre: z.
  const double *c_z;
  /// Each shell's first Cartesian AO.
  const int32_t *ao_offset;
  /// Each shell's rank among the basis's shells of its momentum.
  const int32_t *rank;
} fb_primitive_pair_source;

/// A class's record fields on the device, writable: what fb_mmd_records
/// fills. The bounds are not among them; fb_mmd_bound and
/// fb_ordered_pair_bound write those. The struct is host memory, handed to
/// fb_mmd_records by address; every array is device memory.
typedef struct fb_primitive_pair_records {
  /// zeta = alpha + beta.
  double *zeta;
  /// P: x.
  double *p_x;
  /// P: y.
  double *p_y;
  /// P: z.
  double *p_z;
  /// P - A: x.
  double *pa_x;
  /// P - A: y.
  double *pa_y;
  /// P - A: z.
  double *pa_z;
  /// P - B: x.
  double *pb_x;
  /// P - B: y.
  double *pb_y;
  /// P - B: z.
  double *pb_z;
  /// c_a c_b (pi / zeta)^{3/2}.
  double *s;
  /// alpha.
  double *ea;
  /// beta.
  double *eb;
  /// The bra shell's first Cartesian AO.
  int32_t *off_a;
  /// The ket shell's first Cartesian AO.
  int32_t *off_b;
  /// The bra shell's rank.
  int32_t *rank_a;
  /// The ket shell's rank.
  int32_t *rank_b;
} fb_primitive_pair_records;

/// Every record field but the bounds, on stream: record r is the record of
/// host index r, formed from src, one thread per record, so each shell pair's
/// records are the contiguous run `[rec_begin[p], rec_begin[p + 1])`.
///
/// P is (alpha A + beta B) / zeta, its numerator one explicit FMA with the
/// terms in exponent order: the same for a record and its primitives swapped,
/// bit for bit, though it may sit an ulp from a host's unfused value.
///
/// Every array of out is n_rec long, written whole.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for n_rec or
/// src->n_pairs not positive or a null src or out; otherwise the launch
/// status. Launch errors only.
fb_launch_status fb_mmd_records(wwr_stream stream, int32_t n_rec,
                                const fb_primitive_pair_source *src,
                                const fb_primitive_pair_records *out);

/// Each record's Schwarz bound I = pair_bound(zeta, la + lb) into bound, on
/// stream: the generated `pair_bound<la, lb>` (kernels/pair_bound/), one thread
/// per record.
///
/// zeta and bound are device arrays n_rec long, bound written whole; they
/// must not overlap.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for n_rec not positive or a null zeta
/// or bound, and FB_LAUNCH_UNSUPPORTED for a class with no kernel (la <= lb
/// <= FB_PAIR_BOUND_MAX_MOMENTUM has one), each enqueueing nothing; otherwise
/// the launch status. Launch errors only.
fb_launch_status fb_mmd_bound(wwr_stream stream, int32_t la, int32_t lb, int32_t n_rec,
                              const double *zeta, double *bound);

/// Each record's DRESSED Schwarz bound into bound, on stream: the generated
/// `record_bound<cls->la, cls->lb>` (kernels/record_bound/), one thread per
/// record, `I = max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p| I_p(zeta)`
/// from the view's zeta, P - A, P - B and s, "a" its records' first shell.
/// The bound is ORDERED: the max runs over the first shell's Cartesians, so
/// (la, lb) and (lb, la) are different kernels with different values, and cls
/// may be a swapped view, la > lb with P - A and P - B exchanged. Only la,
/// lb, n_rec, zeta, pa_*, pb_* and s are read. bound is a device array
/// cls->n_rec long, written whole, and must not overlap the view's arrays; a
/// pair whose pair factor underflows gets exactly 0.0.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null cls or bound or a view with
/// no records, and FB_LAUNCH_UNSUPPORTED for a class with no kernel (each l
/// at most FB_RECORD_BOUND_MAX_MOMENTUM has one), each enqueueing nothing;
/// otherwise the launch status. Launch errors only.
fb_launch_status fb_ordered_pair_bound(wwr_stream stream, const fb_primitive_pair_class_view *cls,
                                       double *bound);

#ifdef __cplusplus
}
#endif
