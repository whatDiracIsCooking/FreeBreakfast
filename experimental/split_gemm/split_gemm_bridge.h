/// @file
/// @brief The two elementwise launchers of the split GEMM, declared once for
/// the module that calls them and the device TU that defines them.
///
/// Included by split_gemm.cpp in its GLOBAL MODULE FRAGMENT, and by
/// split_gemm_kernels.cu directly: a declaration in a module's purview would
/// get module linkage and never bind to the definition in the .cu (see
/// ../sort/radix/radix_sort_bridge.h). wwrError_t comes from a different wwr
/// header in each pass, as that header explains.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief hi[i] = (float)x[i] and lo[i] = (float)(x[i] - (double)hi[i])
///
/// The pair represents x[i] to about 2^-48: hi carries fp32's 24 bits and lo
/// the next 24. Every step is exact in fp64, so the split itself loses nothing
/// beyond the bits lo cannot hold.
[[nodiscard]] wwr::wwrError_t split_hi_lo(wwr::wwrStream_t stream, const double *x, float *hi,
                                          float *lo, int64_t count);

/// @brief dst[i] += (double)src[i]
///
/// The fp64 accumulation across K chunks. This is what the chunking buys: the
/// sum inside a chunk is Sgemm's fp32 one, the sum across them is this.
[[nodiscard]] wwr::wwrError_t accumulate_f32(wwr::wwrStream_t stream, const float *src, double *dst,
                                             int64_t count);

} // namespace free_breakfast::device
