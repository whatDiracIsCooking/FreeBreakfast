/* The symmetrize launch wrappers: the C header forked from
 * src/recipes/symmetrize/symmetrize_bridge.h. symmetrize.cu defines both;
 * fb_recipes.h includes this, and devtools/rust_bindings.py reads it into
 * fb-recipes-sys. Plain C: no templates, no vendor types. */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// A <- (A + A^T) / 2 in place, on stream, over the column-major n x n a
/// with leading dimension lda; (i, j) and (j, i) come out the same bits.
///
/// Each pair is summed once in each order, which IEEE addition makes the
/// same value, and halved, so the result is exactly symmetric and a
/// symmetric input comes back unchanged. Rows past n are neither read nor
/// written. a is a device pointer.
///
/// Returns FB_LAUNCH_SUCCESS, enqueueing nothing, for n 0, whatever lda;
/// FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for a negative n, an lda
/// below n, a null a with n positive, or an n of 2,097,121 or more (past
/// the launch grid); otherwise the launch's status. Launch errors only: a
/// fault inside the kernel surfaces at the next synchronization.
fb_launch_status fb_symmetrize(wwr_stream stream, int32_t n, double *a, int32_t lda);

/// Overwrites the strict upper triangle of the column-major n x n a
/// (leading dimension lda) with the transpose of its strict lower one when
/// from_lower, or the strict lower with the transpose of the strict upper
/// when not, in place, on stream.
///
/// The source triangle and the diagonal are not written, and no value of
/// the destination reaches the result, so it may hold anything, NaN
/// included. Rows past n are neither read nor written. a is a device
/// pointer.
///
/// Returns as fb_symmetrize does.
fb_launch_status fb_mirror_triangle(wwr_stream stream, bool from_lower, int32_t n, double *a,
                                    int32_t lda);

#ifdef __cplusplus
}
#endif
