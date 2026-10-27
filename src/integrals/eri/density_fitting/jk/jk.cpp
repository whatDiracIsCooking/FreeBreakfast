/// @file
/// @brief df_coulomb (two gemvs) and df_exchange over the fitted B: of one
/// factor, a batched gemm, a syrk and its mirror; of two, two batched gemms and
/// a gemm.

module free_breakfast.integrals.eri.density_fitting.jk;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemv, gemm(StridedBatched), syrk, geam, copy, gpu_status
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

Status df_coulomb(DeviceResources &res, const int32_t nao, const int32_t naux, const double *b,
                  const double *density, double *j) {
  if (nao < 0 || naux < 0) {
    return err_config("df_coulomb: nao = {} and naux = {} must be non-negative", nao, naux);
  }
  // Both gemvs run over the nao^2 columns of B in the int32_t dispatch.
  const int64_t pairs = int64_t{nao} * nao;
  if (pairs > std::numeric_limits<int32_t>::max()) {
    return err_config("df_coulomb: nao^2 = {} exceeds int32_t", pairs);
  }
  if (nao == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  if (naux == 0) {
    return gpu_status(
        wwr::wwrMemsetAsync(j, 0, sizeof(double) * static_cast<size_t>(pairs), res.stream()));
  }
  const auto pairs32 = static_cast<int32_t>(pairs);
  const auto [gamma] = res.device_regions<double>(static_cast<size_t>(naux));
  // gamma = B vec(D), then vec(J) = B^T gamma.
  FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(res.blas().get(), wwr::WWRBLAS_OP_N, naux, pairs32,
                                                &kOne<double>, b, naux, density, 1, &kZero<double>,
                                                gamma, 1)));
  return gpu_status(wwr::gemv<double, int32_t>(res.blas().get(), wwr::WWRBLAS_OP_T, naux, pairs32,
                                               &kOne<double>, b, naux, gamma, 1, &kZero<double>, j,
                                               1));
}

Status df_exchange(DeviceResources &res, const int32_t nao, const int32_t naux, const int32_t nocc,
                   const double *b, const double *c, double *k) {
  if (nao < 0 || naux < 0 || nocc < 0) {
    return err_config("df_exchange: nao = {}, naux = {} and nocc = {} must be non-negative", nao,
                      naux, nocc);
  }
  // X's rows, syrk's k and leading dimension, in the int32_t dispatch.
  const int64_t rows = int64_t{naux} * nocc;
  if (rows > std::numeric_limits<int32_t>::max()) {
    return err_config("df_exchange: naux * nocc = {} exceeds int32_t", rows);
  }
  if (nao == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const auto nn = static_cast<size_t>(nao) * static_cast<size_t>(nao);
  if (rows == 0) {
    return gpu_status(wwr::wwrMemsetAsync(k, 0, sizeof(double) * nn, stream));
  }
  const auto rows32 = static_cast<int32_t>(rows);
  const auto block = int64_t{naux} * nao; // B_m's stride
  const auto [x, lower] = res.device_regions<double>(static_cast<size_t>(rows) * nao, nn);
  const wwr::wwrblasHandle_t blas = res.blas().get();

  // X_m = B_m C for every m: naux x nocc from naux x nao times nao x nocc, C
  // shared by the batch.
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, nocc, nao, &kOne<double>, b, naux, block, c,
      nao, 0, &kZero<double>, x, naux, rows, nao)));

  // K = 2 X^T X in the lower triangle of a zeroed scratch, so its strict upper
  // triangle stays exactly 0.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(lower, 0, sizeof(double) * nn, stream)));
  FBF_TRY(gpu_status(wwr::syrk<double, int32_t>(blas, wwr::WWRBLAS_FILL_MODE_LOWER,
                                                wwr::WWRBLAS_OP_T, nao, rows32, &kTwo<double>, x,
                                                rows32, &kZero<double>, lower, nao)));
  // K = L + L^T: off the diagonal each element is one of L's plus an exact 0,
  // so (m, n) and (n, m) are the same bits. The diagonal comes out doubled and
  // is copied back from L.
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nao,
                                                nao, &kOne<double>, lower, nao, &kOne<double>,
                                                lower, nao, k, nao)));
  return gpu_status(wwr::copy<double, int32_t>(blas, nao, lower, nao + 1, k, nao + 1));
}

Status df_exchange(DeviceResources &res, const int32_t nao, const int32_t naux, const int32_t rank,
                   const double *b, const double *left, const double *right, double *k) {
  if (nao < 0 || naux < 0 || rank < 0) {
    return err_config("df_exchange: nao = {}, naux = {} and rank = {} must be non-negative", nao,
                      naux, rank);
  }
  // X's and Y's rows, the product's inner dimension and leading dimensions, in
  // the int32_t dispatch.
  const int64_t rows = int64_t{naux} * rank;
  if (rows > std::numeric_limits<int32_t>::max()) {
    return err_config("df_exchange: naux * rank = {} exceeds int32_t", rows);
  }
  if (nao == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  if (rows == 0) {
    const auto nn = static_cast<size_t>(nao) * static_cast<size_t>(nao);
    return gpu_status(wwr::wwrMemsetAsync(k, 0, sizeof(double) * nn, stream));
  }
  const auto rows32 = static_cast<int32_t>(rows);
  const auto block = int64_t{naux} * nao; // B_m's stride
  const auto scratch = static_cast<size_t>(rows) * nao;
  const auto [x, y] = res.device_regions<double>(scratch, scratch);
  const wwr::wwrblasHandle_t blas = res.blas().get();

  // X_m = B_m L and Y_m = B_m R for every m: naux x rank from naux x nao times
  // nao x rank, the factor shared by the batch.
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, rank, nao, &kOne<double>, b, naux, block,
      left, nao, 0, &kZero<double>, x, naux, rows, nao)));
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, rank, nao, &kOne<double>, b, naux, block,
      right, nao, 0, &kZero<double>, y, naux, rows, nao)));

  // K = X^T Y: K_mn = sum_Qi (B_m L)_Qi (B_n R)_Qi = sum_ls (ml|ns) (L R^T)_ls.
  return gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nao, nao,
                                               rows32, &kOne<double>, x, rows32, y, rows32,
                                               &kZero<double>, k, nao));
}

} // namespace free_breakfast
