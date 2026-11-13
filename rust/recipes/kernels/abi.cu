// libfb_recipes_kernels.so's ABI version and backend, which fb-recipes-sys checks at load
// (rust/docs/gpu_layer.md, "What the port changes"). Host functions alone:
// nothing here launches.

#include "fb_recipes.h"

// The compiler says which backend this is, not a definition the build passes,
// so a library compiled for one cannot report the other. clang's HIP mode
// defines __HIP__; nvcc defines __CUDACC__.
#if defined(__HIP__)
#define FB_BACKEND_BUILT FB_BACKEND_HIP
#elif defined(__CUDACC__)
#define FB_BACKEND_BUILT FB_BACKEND_CUDA
#else
#error "abi.cu must be compiled as CUDA (nvcc) or as HIP (clang -x hip)"
#endif

extern "C" uint32_t fb_recipes_abi_version(void) {
  return FB_RECIPES_ABI_VERSION;
}

extern "C" fb_backend fb_recipes_backend(void) {
  return FB_BACKEND_BUILT;
}
