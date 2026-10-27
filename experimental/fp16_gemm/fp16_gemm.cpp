/// @file
/// @brief gemm_fp16 over cublasHgemm / cublasGemmEx or hipblasHgemm /
/// hipblasGemmEx.

module;

// wwr's vendor-include point for the blas layer: cublas_v2.h or hipblas.h,
// plus WWR_SELECTED_CUDA / WWR_SELECTED_HIP. In the GMF, so the vendor
// declarations keep their linkage; the implementation unit only, so no
// importer sees them.
#include <blas.h>

module free_breakfast.experimental.fp16_gemm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import wwr.fp16;

namespace free_breakfast {

wwr::wwrblasStatus_t gemm_fp16(DeviceResources &res, const Fp16Accumulate accumulate,
                               const int32_t m, const int32_t n, const int32_t k,
                               const wwr::wwrHalf *a, const wwr::wwrHalf *b, wwr::wwrHalf *c) {
  const wwr::wwrblasHandle_t handle = res.blas().get();
  // Borrowed for the call, as recipes' gemmwich does: alpha and beta are on the host.
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{handle, wwr::WWRBLAS_POINTER_MODE_HOST};

  if (accumulate == Fp16Accumulate::kHalf) {
    const wwr::wwrHalf one = wwr::wwrFloat2Half(1.0F);
    const wwr::wwrHalf zero = wwr::wwrFloat2Half(0.0F);
#if defined(WWR_SELECTED_CUDA)
    return ::cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, a, m, b, k, &zero, c, m);
#else
    // hipblasHalf is uint16_t unless HIPBLAS_USE_HIP_HALF; the bits are __half's either way.
    using H = hipblasHalf;
    return ::hipblasHgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k,
                          reinterpret_cast<const H *>(&one), reinterpret_cast<const H *>(a), m,
                          reinterpret_cast<const H *>(b), k, reinterpret_cast<const H *>(&zero),
                          reinterpret_cast<H *>(c), m);
#endif
  }

  const float one = 1.0F;
  const float zero = 0.0F;
#if defined(WWR_SELECTED_CUDA)
  return ::cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, a, CUDA_R_16F, m, b,
                        CUDA_R_16F, k, &zero, c, CUDA_R_16F, m, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT);
#else
  return ::hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &one, a, HIP_R_16F, m, b,
                         HIP_R_16F, k, &zero, c, HIP_R_16F, m, HIPBLAS_COMPUTE_32F,
                         HIPBLAS_GEMM_DEFAULT);
#endif
}

} // namespace free_breakfast
