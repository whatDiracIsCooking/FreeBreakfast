/* The Cholesky diagonal tile's launch wrapper: the C header forked from
 * src/recipes/cholesky/cholesky_bridge.h. cholesky.cu defines it;
 * fb_recipes.h includes this, and devtools/rust_bindings.py reads it into
 * fb-recipes-sys. Plain C: no templates, no vendor types. */
#pragma once

#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The order of the diagonal tiles a blocked Cholesky factors one at a time
/// with fb_cholesky_diagonal, and so the width of its panels: the C++
/// kCholeskyBlock. A tile is 64^2 doubles, 32 KiB of shared memory.
#define FB_CHOLESKY_TILE ((int32_t)64)

/// Factors the jb x jb diagonal tile at a (column-major, leading dimension
/// lda) in place, A = L L^T in its lower triangle, on stream, and writes the
/// same lower triangle into l11, packed (leading dimension jb).
///
/// first is the tile's first column in the whole matrix. If *info is
/// already non-zero when the kernel runs (an earlier tile met a minor that
/// is not positive), nothing is written. Otherwise, at the tile's first
/// pivot that is not positive (or is NaN), column k, *info becomes
/// first + k + 1, the 1-based leading minor of the whole matrix, and the
/// tile is left part-factored. Only the lower triangles are read or
/// written: the strict upper triangle of the tile at a, and of l11, are
/// left as they were. a, l11 and info are device pointers.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for a jb outside
/// 1..FB_CHOLESKY_TILE, a negative first, an lda below jb, or a null a, l11
/// or info; otherwise the launch's status. Launch errors only: a fault
/// inside the kernel surfaces at the next synchronization.
fb_launch_status fb_cholesky_diagonal(wwr_stream stream, int32_t jb, int32_t first, double *a,
                                      int32_t lda, double *l11, int32_t *info);

#ifdef __cplusplus
}
#endif
