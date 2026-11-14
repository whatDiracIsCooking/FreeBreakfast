/* libfb_recipes_kernels.so's one C header: every recipes launch wrapper
 * rust/ has forked, and the header devtools/rust_bindings.py reads into
 * fb-recipes-sys's committed bindings (rust/docs/gpu_layer.md, "From bridge
 * to C ABI"). */
#pragma once

#include "orthogonalize/inverse_sqrt.h"
#include "sort/merge_sort.h"

#include <stdint.h>

#include "fb_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The version of this header's ABI. Bump it with any change to a
/// declaration here or in a header this one includes, so that fb-recipes-sys
/// refuses a library built from another version of them.
#define FB_RECIPES_ABI_VERSION ((uint32_t)2)

/// FB_RECIPES_ABI_VERSION as this library was compiled. A plain host
/// function: it calls no runtime and needs no device.
uint32_t fb_recipes_abi_version(void);

/// The backend this library was built for, FB_BACKEND_CUDA or
/// FB_BACKEND_HIP. A plain host function: it calls no runtime and needs no
/// device.
fb_backend fb_recipes_backend(void);

#ifdef __cplusplus
}
#endif
