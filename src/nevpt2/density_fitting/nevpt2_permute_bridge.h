/// @file
/// @brief The slab permute's launcher, declared once for the module that calls
/// it and the device TU that defines it.
///
/// Included by density_fitting.cpp in its GLOBAL MODULE FRAGMENT, and by
/// nevpt2_permute.cu directly: a declaration in a module's purview gets module
/// linkage and can never bind to a definition compiled in a plain TU, which a
/// .cu is (see src/integrals/one_electron/overlap/overlap_bridge.h).

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see src/integrals/one_electron/overlap/overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::nevpt2::device {

/// @brief out[o0, o1, o2, o3] = in[o0 s0 + o1 s1 + o2 s2 + o3 s3] for every
/// element of the column-major @p n0 x @p n1 x @p n2 x @p n3 @p out, on
/// @p stream: a four-index permute, one thread per output element.
///
/// s_k is the input stride of the input axis that output axis k reads, so a
/// permute by axes a (output axis k is input axis a[k]) passes
/// n_k = in_dims[a[k]] and s_k = in_strides[a[k]]. @p out must not overlap
/// @p in. Launch status only; it does not synchronize.
[[nodiscard]] wwr::wwrError_t permute(wwr::wwrStream_t stream, int64_t n0, int64_t n1, int64_t n2,
                                      int64_t n3, int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                                      const double *in, double *out);

} // namespace free_breakfast::nevpt2::device
