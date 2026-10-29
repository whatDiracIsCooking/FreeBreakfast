/// @file
/// @brief nonsymmetric_jk: D's two parts by geam, K of each, J of the
/// symmetric one, and G = J - K/2.

module free_breakfast.integrals.eri.integral_direct.nonsymmetric_jk;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kHalf, kNegHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // geam, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;

#include "diagnostics.h"

namespace free_breakfast {

Status nonsymmetric_jk(DeviceResources &res, const std::span<CoulombPairClass> j_classes,
                       const std::span<OrderedPairClass> k_classes,
                       const CartesianMatrixAO &density, CartesianMatrixAO &g, CartesianMatrixAO &k,
                       const double tau_j, const double tau_k) {
  // Written so that NaN fails too. coulomb and exchange check their own, but
  // only once the geams have written g.
  if (!(tau_j >= 0.0) || !(tau_k >= 0.0)) {
    return err_config("nonsymmetric_jk: tau_j ({}) and tau_k ({}) must be 0 or above", tau_j,
                      tau_k);
  }
  for (const auto &[name, dev] :
       {std::pair{"the density", density.dev_idx()}, {"G", g.dev_idx()}, {"K", k.dev_idx()}}) {
    if (dev != res.dev_idx()) {
      return err_config("nonsymmetric_jk: {} is on device {}, but the call passed device {}", name,
                        dev, res.dev_idx());
    }
  }
  if (g.n() != density.n() || k.n() != density.n()) {
    return err_config("nonsymmetric_jk: the density is {0} x {0}, but G is {1} x {1} and K "
                      "{2} x {2}",
                      density.n(), g.n(), k.n());
  }
  // Each step below reads one of the three while it writes another.
  if (&g == &k || &g == &density || &k == &density) {
    return err_config("nonsymmetric_jk: G, K and the density must be three matrices");
  }
  if (density.n() == 0) {
    return {};
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(density.n());
  const size_t bytes = sizeof(double) * density.num_elements();
  const double *const d = density.data();
  // K[A], then K[D]. Neither build takes the workspace.
  const auto [w] = res.device_regions<double>(density.num_elements());

  // g = A = (D - D^T)/2; k = K[A], held in w.
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                                &kHalf<double>, d, n, &kNegHalf<double>, d, n,
                                                g.data(), n)));
  FBF_TRY(exchange(res, k_classes, g, k, tau_k, DensitySymmetry::Antisymmetric));
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(w, k.data(), bytes, wwr::wwrMemcpyDeviceToDevice, res.stream())));
  // g = S = (D + D^T)/2; k = K[S], and w = K[A] + K[S] = K[D].
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                                &kHalf<double>, d, n, &kHalf<double>, d, n,
                                                g.data(), n)));
  FBF_TRY(exchange(res, k_classes, g, k, tau_k, DensitySymmetry::Symmetric));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n,
                                                &kOne<double>, w, n, &kOne<double>, k.data(), n, w,
                                                n)));
  // k = J[S] = J[D] from g, which still holds S; then g = J[D] - K[D]/2 and
  // k = K[D].
  FBF_TRY(coulomb(res, j_classes, g, k, tau_j));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n,
                                                &kOne<double>, k.data(), n, &kNegHalf<double>, w, n,
                                                g.data(), n)));
  return gpu_status(
      wwr::wwrMemcpyAsync(k.data(), w, bytes, wwr::wwrMemcpyDeviceToDevice, res.stream()));
}

} // namespace free_breakfast
