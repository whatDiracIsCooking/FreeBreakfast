/* libfb_integrals_kernels.so's one C header: every integrals launch wrapper
 * rust/ has forked, and the header devtools/rust-bindings.py reads into
 * fb-integrals-sys's committed bindings (rust/docs/gpu_layer.md, "From bridge
 * to C ABI"). */
#pragma once

#include "mmd/mmd.h"
#include "one_electron/kinetic/kinetic.h"
#include "one_electron/overlap/overlap.h"
#include "one_electron/potential_en/potential_en.h"

#include <stdint.h>

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The version of this header's ABI. Bump it with any change to a
/// declaration here or in a header this one includes, so that fb-integrals-sys
/// refuses a library built from another version of them.
#define FB_INTEGRALS_ABI_VERSION ((uint32_t)2)

/// FB_INTEGRALS_ABI_VERSION as this library was compiled. A plain host
/// function: it calls no runtime and needs no device.
uint32_t fb_integrals_abi_version(void);

/// The backend this library was built for, FB_BACKEND_CUDA or
/// FB_BACKEND_HIP. A plain host function: it calls no runtime and needs no
/// device.
fb_backend fb_integrals_backend(void);

#ifdef __cplusplus
}
#endif
