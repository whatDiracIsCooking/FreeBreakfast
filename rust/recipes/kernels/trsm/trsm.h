/* The triangular solve's diagonal-block launch wrapper: the C header forked
 * from src/recipes/trsm/trsm_bridge.h. trsm.cu defines them; fb_recipes.h
 * includes this, and devtools/rust-bindings.py reads it into fb-recipes-sys.
 * Plain C: no templates, no vendor types. */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The order of the diagonal blocks a blocked triangular solve hands to
/// fb_trsm_diagonal_f64 one at a time, and so the inner dimension
/// of its gemm updates: the C++ kTrsmBlock. A block is 64^2 doubles, 32 KiB.
#define FB_TRSM_TILE ((int32_t)64)

/// Solves M X = s B in place for one jb x jb triangular diagonal block M
/// and ncols right-hand sides, on stream, in doubles.
///
/// M(p, q) is a[q + p lda] when a_trans, else a[p + q lda]; it is upper
/// triangular when upper, else lower, and only that triangle is read. With
/// unit its diagonal is taken as 1 and not read. B(r, c) is b[c + r ldb]
/// when b_trans, else b[r + c ldb], for r below jb and c below ncols, and X
/// overwrites it; nothing else of b is read or written. With s zero, X is
/// zero and neither M nor B is read (BLAS's alpha = 0). A singular M is not
/// detected: X comes back Inf or NaN. a and b are device pointers, and must
/// not overlap.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for a jb outside
/// 1..FB_TRSM_TILE, an ncols below 1, an lda below jb, an ldb below jb (or
/// below ncols when b_trans), or a null a or b; otherwise the launch's
/// status. Launch errors only: a fault inside the kernel surfaces at the
/// next synchronization.
fb_launch_status fb_trsm_diagonal_f64(wwr_stream stream, int32_t jb, int32_t ncols, double s,
                                      const double *a, int32_t lda, bool a_trans, bool upper,
                                      bool unit, double *b, int32_t ldb, bool b_trans);

#ifdef __cplusplus
}
#endif
