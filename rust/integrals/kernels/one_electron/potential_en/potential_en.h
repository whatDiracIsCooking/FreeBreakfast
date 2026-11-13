/* The nuclear attraction's launch wrapper: the C header forked from
 * src/integrals/one_electron/potential_en/potential_en_bridge.h.
 * potential_en_launch.cu defines it; fb_integrals.h includes this, and
 * devtools/rust-bindings.py reads it into fb-integrals-sys. Plain C: no
 * templates, no vendor types. */
#pragma once

#include <stdint.h>

/* wwr_stream, the C ABI's stream handle (deps/WarpWraps/src/capi/gen). */
#include "common.h"

#include "fb_launch.h"
#include "nuclei_view.h"
#include "primitive_pair_class_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The largest momentum fb_potential_en_class has a kernel for, f: the
/// extent of the potential_en table tools/emit/one_electron_kernels.py
/// writes, which potential_en_launch.cu static_asserts against it.
#define FB_POTENTIAL_EN_MAX_MOMENTUM ((int32_t)3)

/// Adds one class's nuclear-attraction contributions into the n_ao x n_ao
/// matrix v, on stream: the generated kernel v_{la}{lb}, one thread per
/// record, the nuclei serial inside it.
///
/// Reads cls's zeta, p_* (the product centre), pa_*, pb_*, s and offsets,
/// and nuclei's charges and positions. Each value is atomically added at
/// (m, n) and at (n, m), so v must be zeroed before the first class of a
/// build and is symmetric after the last. v is a device array; the kernels
/// index it as a 32-bit m * n_ao + n, so n_ao * n_ao must fit an int32_t.
///
/// Returns FB_LAUNCH_INVALID_ARGUMENT for a null cls, nuclei or v, a view
/// with no records, no nuclei or n_ao not positive, and
/// FB_LAUNCH_UNSUPPORTED for a class with no kernel
/// (la <= lb <= FB_POTENTIAL_EN_MAX_MOMENTUM has one), each enqueueing
/// nothing; otherwise the launch status. Launch errors only: the kernel's own
/// faults surface at the next synchronization.
fb_launch_status fb_potential_en_class(wwr_stream stream, const fb_primitive_pair_class_view *cls,
                                       const fb_nuclei_view *nuclei, int32_t n_ao, double *v);

#ifdef __cplusplus
}
#endif
