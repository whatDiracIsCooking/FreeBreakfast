/* The inverse square root's launch wrapper: the C header forked from
 * src/recipes/orthogonalize/inverse_sqrt_bridge.h. inverse_sqrt.cu defines
 * it; fb_recipes.h includes this, and devtools/rust_bindings.py reads it
 * into fb-recipes-sys. Plain C: no templates, no vendor types. */
#pragma once

#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Replaces each of the first count of data with 1 / sqrt(x), in place, on
/// stream, except an element at or below floor, which becomes 0.
///
/// The floor test comes before the square root, so a zero or negative
/// element never reaches it and the result is never Inf, nor a NaN that was
/// not one already. floor should be non-negative. data is a device pointer.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for a null data
/// with count positive; FB_LAUNCH_SUCCESS, enqueueing nothing, for count 0;
/// otherwise the launch's status. Launch errors only: a fault inside the
/// kernel surfaces at the next synchronization.
fb_launch_status fb_inverse_sqrt(wwr_stream stream, uint32_t count, double *data, double floor);

#ifdef __cplusplus
}
#endif
