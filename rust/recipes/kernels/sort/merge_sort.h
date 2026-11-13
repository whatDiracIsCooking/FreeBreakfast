/* The merge sort's launch wrapper and its workspace query: the C header
 * forked from src/recipes/sort/merge_sort_bridge.h. merge_sort.cu defines
 * both; fb_recipes.h includes this, and devtools/rust_bindings.py reads it
 * into fb-recipes-sys. Plain C: no templates, no vendor types. */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The bytes of the one scratch buffer fb_merge_sort needs for n keys, with
/// or without values: zero up to one tile (2816 keys), where the scratch is
/// never touched. A plain host function, pure in its arguments: it calls no
/// runtime and needs no device.
size_t fb_merge_sort_workspace_bytes(uint32_t n, bool with_values);

/// Sorts the first n of keys ascending in IEEE-754 total order, stably, on
/// stream, and applies the same permutation to values unless it is null.
///
/// The keys are moved as raw bits: -0.0 sorts before +0.0, and NaNs sort by
/// sign and payload past the infinities. The result always ends in keys and
/// values. Every pointer is a device pointer, none of the arrays may overlap,
/// and workspace is FB_DEVICE_ALIGNMENT-aligned and at least
/// fb_merge_sort_workspace_bytes(n, values != NULL) long; it may be null when
/// that is zero.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT, enqueueing nothing, for a null keys
/// with n positive or a workspace shorter than the query; otherwise the
/// status of the first failed launch, or FB_LAUNCH_SUCCESS. Launch errors
/// only: a fault inside a kernel surfaces at the next synchronization.
fb_launch_status fb_merge_sort(wwr_stream stream, uint32_t n, double *keys, uint32_t *values,
                               void *workspace, size_t workspace_bytes);

#ifdef __cplusplus
}
#endif
