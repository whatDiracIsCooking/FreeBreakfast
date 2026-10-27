/// @file
/// @brief gemm_fp16 over wwrblasHgemm / wwrblasGemmEx.

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
    return wwr::wwrblasHgemm(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, n, k, &one, a, m, b,
                             k, &zero, c, m);
  }

  const float one = 1.0F;
  const float zero = 0.0F;
  return wwr::wwrblasGemmEx(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, n, k, &one, a,
                            wwr::WWRBLAS_R_16F, m, b, wwr::WWRBLAS_R_16F, k, &zero, c,
                            wwr::WWRBLAS_R_16F, m, wwr::WWRBLAS_COMPUTE_32F,
                            wwr::WWRBLAS_GEMM_DEFAULT);
}

} // namespace free_breakfast
