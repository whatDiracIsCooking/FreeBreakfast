/* The launch status and the device alignment every launch wrapper under rust/
 * shares (rust/docs/gpu_layer.md, "Statuses" and "Scratch: the workspace
 * query"). Plain C: devtools/rust_bindings.py reads it, through each module's
 * header, into that module's -sys bindings. */
#pragma once

#include <stddef.h>
#include <stdint.h>

/// A launch wrapper's result: 0 is success; a positive value is the runtime's
/// launch error, as the C ABI's wwr_status; a negative one is the wrapper's
/// own refusal, before anything is enqueued.
typedef int32_t fb_launch_status;

/// Success: every launch was enqueued.
#define FB_LAUNCH_SUCCESS ((fb_launch_status)0)
/// No kernel for these parameters (a momentum outside the dispatch table).
#define FB_LAUNCH_UNSUPPORTED ((fb_launch_status) - 1)
/// A count, pointer or workspace the wrapper refuses: a null array, or a
/// workspace shorter than its query asked for.
#define FB_LAUNCH_INVALID_ARGUMENT ((fb_launch_status) - 2)

/// The alignment, in bytes, a launch wrapper assumes of every buffer it is
/// handed, scratch and output alike, and the one a workspace is partitioned
/// to: the pool's allocation granularity on both vendors.
#define FB_DEVICE_ALIGNMENT ((size_t)256)
