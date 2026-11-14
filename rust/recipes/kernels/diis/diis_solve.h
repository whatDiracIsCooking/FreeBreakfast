/* The DIIS coefficient solve's launch wrapper: the C header forked from
 * src/recipes/diis/diis_solve_bridge.h. diis_solve.cu defines it;
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

/// The order of the largest bordered system fb_diis_solve takes, a history
/// of 64 plus the Lagrange row: the C++ kDiisSolveMaxDim. The kernel's shared memory is sized statically from
/// it, (dim + 1) dim doubles = 34,320 B, under CUDA's 48 KiB static limit.
#define FB_DIIS_SOLVE_MAX_DIM ((int32_t)65)

/// Solves the bordered DIIS system for the m extrapolation coefficients, on
/// stream, in one block, in shared memory: rows [ <e_i|e_j> | -1 ] for
/// i < m and [ -1^T | 0 ] times [ c ; l ] equal to [ 0 ; -1 ], so c
/// minimizes |sum_i c_i e_i| subject to sum_i c_i = 1. gram is the residual Gram, column-indexed with leading dimension ld;
/// only its upper triangle over the first m columns, gram[a ld + c] for
/// a <= c < m, is read, and the kernel symmetrizes it. The system is solved
/// by Gauss-Jordan elimination with partial pivoting, and the m
/// coefficients are written to coeff. If a pivot falls below 1e-14 the
/// subspace is singular: coeff becomes the unit selector at newest (1
/// there, 0 elsewhere), so an extrapolation over it reproduces the newest
/// iterate, and *singular is set to 1; otherwise *singular is 0. gram,
/// coeff and singular are device pointers.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for an m below 1,
/// an ld below m or not below FB_DIIS_SOLVE_MAX_DIM, a newest outside
/// 0..m-1, or a null gram, coeff or singular; otherwise the launch's
/// status. Launch errors only: a fault inside the kernel surfaces at the
/// next synchronization.
fb_launch_status fb_diis_solve(wwr_stream stream, int32_t m, int32_t ld, int32_t newest,
                               const double *gram, double *coeff, int32_t *singular);

#ifdef __cplusplus
}
#endif
