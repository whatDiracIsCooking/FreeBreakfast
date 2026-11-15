/* libfb_localization_kernels.so's one C header: every localization launch
 * wrapper rust/ has forked, and the header devtools/rust_bindings.py reads
 * into fb-localization-sys's committed bindings (rust/docs/gpu_layer.md,
 * "From bridge to C ABI"). */
#pragma once

#include "pipek_mezey/pipek_mezey.h"

#include <stdint.h>

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The version of this header's ABI. Bump it with any change to a
/// declaration here or in a header this one includes, so that
/// fb-localization-sys refuses a library built from another version of them.
#define FB_LOCALIZATION_ABI_VERSION ((uint32_t)1)

/// FB_LOCALIZATION_ABI_VERSION as this library was compiled. A plain host
/// function: it calls no runtime and needs no device.
uint32_t fb_localization_abi_version(void);

/// The backend this library was built for, FB_BACKEND_CUDA or
/// FB_BACKEND_HIP. A plain host function: it calls no runtime and needs no
/// device.
fb_backend fb_localization_backend(void);

#ifdef __cplusplus
}
#endif
