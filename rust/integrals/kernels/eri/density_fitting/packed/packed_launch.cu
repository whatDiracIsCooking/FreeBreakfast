// Density fitting's packed-map kernels and their launch wrappers: the pack
// of a density into the map's columns and the gather of packed columns back
// into the square. One source for both backends -- nvcc under CUDA, clang -x
// hip under HIP.
//
// rust/'s own, with no C++ launcher to fork (packed.h has why), in the shape
// of the forked ones (rust/docs/gpu_layer.md, "Layering", "Statuses"):
// extern "C", enqueue-only, a refusal FB_LAUNCH_INVALID_ARGUMENT. One thread
// per column, the column count an int32 by the ABI; every element index is
// 64 bits wide, so no extent past the column count is held to an int32.

// Standard headers, and ours (which include only standard headers), before
// the wwr headers below: under HIP the vendor runtime header they pull in
// must not precede libc++ (primitives.h has why).
#include <cstdint>

#include "block_params.h"
#include "primitives.h"

#include "eri/density_fitting/packed/packed.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

#include <runtime.h>

namespace {

// One thread per column c: dp[c] = w_c d[m n_ao + n], the host's
// Packing::pack and one multiply by the weight, so the two agree bitwise.
__global__ void packed_pack_kernel(const int32_t n_ao, const int32_t n_columns, const int32_t *rows,
                                   const int32_t *cols, const double *weights, const double *d,
                                   double *dp) {
  const int32_t c = static_cast<int32_t>(threadIdx.x + (blockIdx.x * blockDim.x));
  if (c >= n_columns) {
    return;
  }
  const int64_t at = (static_cast<int64_t>(rows[c]) * n_ao) + cols[c];
  dp[c] = weights[c] * d[at];
}

// One thread per column c, over the b aux functions: element (c, q) to
// (m, n) and (n, m) of matrix q, the host's Packing::unpack per function. A
// diagonal block's column (weight 1) above its diagonal is not read, so no
// two columns write one element and the mirror is the same bits.
__global__ void packed_gather_kernel(const int32_t n_ao, const int32_t n_columns, const int32_t b,
                                     const int32_t *rows, const int32_t *cols,
                                     const double *weights, const double *packed, double *slab) {
  const int32_t c = static_cast<int32_t>(threadIdx.x + (blockIdx.x * blockDim.x));
  if (c >= n_columns) {
    return;
  }
  const int64_t m = rows[c];
  const int64_t n = cols[c];
  if (weights[c] == 1.0 && m < n) {
    return;
  }
  const int64_t square = static_cast<int64_t>(n_ao) * n_ao;
  const int64_t mn = (m * n_ao) + n;
  const int64_t nm = (n * n_ao) + m;
  for (int64_t q = 0; q < b; ++q) {
    const double v = packed[c + (q * n_columns)];
    slab[(q * square) + mn] = v;
    slab[(q * square) + nm] = v;
  }
}

// One thread per column, kFourWarps to a block.
uint32_t blocks_for(const int32_t n_columns) {
  return free_breakfast::math::idivup(static_cast<uint32_t>(n_columns), free_breakfast::kFourWarps);
}

} // namespace

extern "C" fb_launch_status fb_packed_pack(const wwr_stream stream, const int32_t n_ao,
                                           const int32_t n_columns, const int32_t *rows,
                                           const int32_t *cols, const double *weights,
                                           const double *d, double *dp) {
  if (rows == nullptr || cols == nullptr || weights == nullptr || d == nullptr || dp == nullptr ||
      n_ao < 1 || n_columns < 1) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  // The C ABI's stream is the runtime's, behind a void *.
  packed_pack_kernel<<<blocks_for(n_columns), free_breakfast::kFourWarps, 0,
                       static_cast<wwr::wwrStream_t>(stream)>>>(n_ao, n_columns, rows, cols,
                                                                weights, d, dp);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}

extern "C" fb_launch_status fb_packed_gather(const wwr_stream stream, const int32_t n_ao,
                                             const int32_t n_columns, const int32_t b,
                                             const int32_t *rows, const int32_t *cols,
                                             const double *weights, const double *packed,
                                             double *slab) {
  if (rows == nullptr || cols == nullptr || weights == nullptr || packed == nullptr ||
      slab == nullptr || n_ao < 1 || n_columns < 1 || b < 1) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  packed_gather_kernel<<<blocks_for(n_columns), free_breakfast::kFourWarps, 0,
                         static_cast<wwr::wwrStream_t>(stream)>>>(n_ao, n_columns, b, rows, cols,
                                                                  weights, packed, slab);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}
