/// @file
/// @brief df_coulomb (two gemvs) and df_exchange over the fitted B: of one
/// factor, a batched gemm, a syrk and its mirror; of two, two batched gemms and
/// a gemm; and both of a block of densities, J by two gemms and K by two
/// batched gemms and a gemm batched over the densities; and K of a block of
/// symmetrized densities sharing one left factor, X built once.

module free_breakfast.integrals.eri.density_fitting.jk;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemv, gemm(StridedBatched), syrk, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.symmetrize;

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
  const auto [x] = res.device_regions<double>(static_cast<size_t>(rows) * nao);
  const wwr::wwrblasHandle_t blas = res.blas().get();

  // X_m = B_m C for every m: naux x nocc from naux x nao times nao x nocc, C
  // shared by the batch.
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, nocc, nao, &kOne<double>, b, naux, block, c,
      nao, 0, &kZero<double>, x, naux, rows, nao)));

  // K = 2 X^T X in the lower triangle, then its mirror onto the upper, so
  // (m, n) and (n, m) are the same bits.
  FBF_TRY(gpu_status(wwr::syrk<double, int32_t>(blas, wwr::WWRBLAS_FILL_MODE_LOWER,
                                                wwr::WWRBLAS_OP_T, nao, rows32, &kTwo<double>, x,
                                                rows32, &kZero<double>, k, nao)));
  return symmetrize(res, wwr::WWRBLAS_FILL_MODE_LOWER, nao, k, nao);
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

Status df_coulomb(DeviceResources &res, const int32_t nao, const int32_t naux, const int32_t count,
                  const double *b, const double *density, double *j) {
  if (nao < 0 || naux < 0 || count < 0) {
    return err_config("df_coulomb: nao = {}, naux = {} and count = {} must be non-negative", nao,
                      naux, count);
  }
  // Both gemms run over the nao^2 columns of B in the int32_t dispatch.
  const int64_t pairs = int64_t{nao} * nao;
  if (pairs > std::numeric_limits<int32_t>::max()) {
    return err_config("df_coulomb: nao^2 = {} exceeds int32_t", pairs);
  }
  if (nao == 0 || count == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  if (naux == 0) {
    return gpu_status(wwr::wwrMemsetAsync(
        j, 0, sizeof(double) * static_cast<size_t>(pairs) * static_cast<size_t>(count),
        res.stream()));
  }
  const auto pairs32 = static_cast<int32_t>(pairs);
  const auto [gamma] =
      res.device_regions<double>(static_cast<size_t>(naux) * static_cast<size_t>(count));
  const wwr::wwrblasHandle_t blas = res.blas().get();
  // Gamma = B [vec(D_d)], naux x count, then [vec(J_d)] = B^T Gamma.
  FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux,
                                                count, pairs32, &kOne<double>, b, naux, density,
                                                pairs32, &kZero<double>, gamma, naux)));
  return gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, pairs32,
                                               count, naux, &kOne<double>, b, naux, gamma, naux,
                                               &kZero<double>, j, pairs32));
}

Status df_exchange(DeviceResources &res, const int32_t nao, const int32_t naux, const int32_t rank,
                   const int32_t count, const double *b, const double *left, const double *right,
                   double *k) {
  if (nao < 0 || naux < 0 || rank < 0 || count < 0) {
    return err_config(
        "df_exchange: nao = {}, naux = {}, rank = {} and count = {} must be non-negative", nao,
        naux, rank, count);
  }
  // X's and Y's rows (their leading dimension), and K's nao^2 stride, in the
  // int32_t dispatch.
  const int64_t rows = int64_t{naux} * rank * count;
  const int64_t nn = int64_t{nao} * nao;
  if (rows > std::numeric_limits<int32_t>::max() || nn > std::numeric_limits<int32_t>::max()) {
    return err_config("df_exchange: naux * rank * count = {} or nao^2 = {} exceeds int32_t", rows,
                      nn);
  }
  if (nao == 0 || count == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  if (rows == 0) {
    return gpu_status(wwr::wwrMemsetAsync(
        k, 0, sizeof(double) * static_cast<size_t>(nn) * static_cast<size_t>(count), stream));
  }
  const auto rows32 = static_cast<int32_t>(rows);
  const int32_t density_rows = naux * rank; // X_d's rows; it divides rows
  const int32_t width = rank * count;       // the factors' columns, side by side
  const auto block = int64_t{naux} * nao;   // B_m's stride
  const auto scratch = static_cast<size_t>(rows) * nao;
  const auto [x, y] = res.device_regions<double>(scratch, scratch);
  const wwr::wwrblasHandle_t blas = res.blas().get();

  // X_m = B_m [L_d] and Y_m = B_m [R_d] for every m: naux x (count rank) from
  // naux x nao times nao x (count rank), the factors shared by the batch.
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, width, nao, &kOne<double>, b, naux, block,
      left, nao, 0, &kZero<double>, x, naux, rows, nao)));
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, width, nao, &kOne<double>, b, naux, block,
      right, nao, 0, &kZero<double>, y, naux, rows, nao)));

  // K_d = X_d^T Y_d for every d: X_d and Y_d the (naux rank) x nao blocks of
  // rows at naux rank d.
  return gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nao, nao, density_rows, &kOne<double>, x, rows32,
      density_rows, y, rows32, density_rows, &kZero<double>, k, nao, nn, count));
}

Status df_exchange_symmetrized(DeviceResources &res, const int32_t nao, const int32_t naux,
                               const int32_t rank, const int32_t count, const double *b,
                               const double *left, const double *right, double *k) {
  if (nao < 0 || naux < 0 || rank < 0 || count < 0) {
    return err_config("df_exchange_symmetrized: nao = {}, naux = {}, rank = {} and count = {} "
                      "must be non-negative",
                      nao, naux, rank, count);
  }
  // Y's rows (its leading dimension), and K's nao^2 stride, in the int32_t
  // dispatch; X's rows divide Y's.
  const int64_t rows = int64_t{naux} * rank * count;
  const int64_t nn = int64_t{nao} * nao;
  if (rows > std::numeric_limits<int32_t>::max() || nn > std::numeric_limits<int32_t>::max()) {
    return err_config(
        "df_exchange_symmetrized: naux * rank * count = {} or nao^2 = {} exceeds int32_t", rows,
        nn);
  }
  if (nao == 0 || count == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  if (rows == 0) {
    return gpu_status(wwr::wwrMemsetAsync(
        k, 0, sizeof(double) * static_cast<size_t>(nn) * static_cast<size_t>(count), stream));
  }
  const auto rows32 = static_cast<int32_t>(rows);
  const int32_t density_rows = naux * rank; // X's rows and Y_d's
  const int32_t width = rank * count;       // the right factors' columns, side by side
  const auto block = int64_t{naux} * nao;   // B_m's stride
  const auto [x, y] = res.device_regions<double>(static_cast<size_t>(density_rows) * nao,
                                                 static_cast<size_t>(rows) * nao);
  const wwr::wwrblasHandle_t blas = res.blas().get();

  // X_m = B_m L and Y_m = B_m [R_d] for every m, the factors shared by the
  // batch.
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, rank, nao, &kOne<double>, b, naux, block,
      left, nao, 0, &kZero<double>, x, naux, density_rows, nao)));
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, width, nao, &kOne<double>, b, naux, block,
      right, nao, 0, &kZero<double>, y, naux, rows, nao)));

  // K_d = X^T Y_d for every d, X shared: Y_d the (naux rank) x nao block of
  // rows at naux rank d. Then each K_d <- (K_d + K_d^T) / 2.
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nao, nao, density_rows, &kOne<double>, x,
      density_rows, 0, y, rows32, density_rows, &kZero<double>, k, nao, nn, count)));
  for (int32_t d = 0; d < count; ++d) {
    FBF_TRY(symmetrize(res, nao, k + (nn * d), nao));
  }
  return {};
}

} // namespace free_breakfast
