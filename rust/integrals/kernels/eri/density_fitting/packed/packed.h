/* Density fitting's packed-map kernel: the columns of the packed shell-pair
 * map gathered back into the square, for one vector or a batch of aux
 * functions. No C++ bridge to fork: the C++ B is square and resident, so
 * this header and packed_launch.cu are rust/'s own. fb_integrals.h includes
 * this, and devtools/rust-bindings.py reads it into fb-integrals-sys. Plain
 * C: no templates, no vendor types.
 *
 * THE TABLES. The map's (fb-steps' eri step writes them, from its
 * pair_columns), n_columns entries each: column c stands at row
 * rows[c] and column cols[c] of the n_ao x n_ao matrix, and has the weight
 * weights[c], 1 on a diagonal shell pair's block, whose (m, n) and (n, m)
 * are columns of their own, and 2 on any other, whose one column stands for
 * both. A packed batch of b aux functions is b rows of n_columns, element
 * (c, q) at c + n_columns q; its square slab is b matrices of n_ao x n_ao,
 * element (m, n) of aux function q at n + n_ao (m + n_ao q). Every index is
 * computed in 64 bits, so no product of these extents is held to an int32. */
#pragma once

#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Gathers the packed batch packed, b x n_columns doubles, into the square
/// slab, b x n_ao x n_ao doubles, on stream: for each column c and aux
/// function q, (m, n) and (n, m) of matrix q both set to element (c, q),
/// (m, n) column c's entries of rows and cols, one thread per column looping over q, except a column of
/// weight 1 (a diagonal block's) with m < n, which is not read: a diagonal
/// block is read from its lower triangle, so (m, n) and (n, m) are the same
/// bits. With b = 1 it unpacks one packed vector. The elements no column
/// covers are untouched: zero the slab first for the map's unpack. rows,
/// cols and weights are the map's tables (above). Every array is device
/// memory.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null array or an n_ao,
/// n_columns or b not positive, enqueueing nothing; otherwise the launch
/// status.
fb_launch_status fb_packed_gather(wwr_stream stream, int32_t n_ao, int32_t n_columns, int32_t b,
                                  const int32_t *rows, const int32_t *cols, const double *weights,
                                  const double *packed, double *slab);

#ifdef __cplusplus
}
#endif
