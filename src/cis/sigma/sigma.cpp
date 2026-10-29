/// @file
/// @brief Sigma's checks, its buffers, and the five steps of a column.

module free_breakfast.cis.sigma;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kTwo, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // dgmm, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.linear_operator;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.integrals.eri.integral_direct.nonsymmetric_jk;

#include "diagnostics.h"

namespace free_breakfast::cis {

static_assert(LinearOperator<Sigma, double>);

Result<Sigma> Sigma::create(DeviceResources &res, const BasisSet &basis,
                            std::shared_ptr<const Buffers::device<double>> u,
                            const std::span<CoulombPairClass> j_classes,
                            const std::span<OrderedPairClass> k_classes, const SphericalMatrixAO &c,
                            const int32_t n_occ, const double *const delta, const Spin spin) {
  if (c.dev_idx() != res.dev_idx()) {
    return err_config("cis::Sigma::create: the MO coefficients are on device {}, but the call "
                      "passed device {}",
                      c.dev_idx(), res.dev_idx());
  }
  if (c.n() != basis.nao_sphe()) {
    return err_config("cis::Sigma::create: the MO coefficients are {0} x {0}, but the basis has "
                      "{1} spherical AOs",
                      c.n(), basis.nao_sphe());
  }
  const auto nao = static_cast<int64_t>(c.n());
  if (n_occ < 1 || n_occ >= nao || n_occ * (nao - n_occ) > std::numeric_limits<int32_t>::max()) {
    return err_config("cis::Sigma::create: needs 1 <= n_occ < nao and n_occ * (nao - n_occ) to "
                      "fit an int32_t (n_occ = {}, nao = {})",
                      n_occ, nao);
  }
  if (delta == nullptr) {
    return err_config("cis::Sigma::create: delta is null");
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  // nonsymmetric_jk's own scratch, one nao_cart^2 block of the workspace,
  // grown here so that no apply allocates it. The workspace only grows, so a
  // later, larger user leaves it big enough.
  const auto nao_cart = static_cast<size_t>(basis.nao_cart());
  static_cast<void>(res.device_workspace(nao_cart * nao_cart * sizeof(double)));
  return Sigma{res, basis, std::move(u), j_classes, k_classes, c, n_occ, delta, spin};
}

Sigma::Sigma(DeviceResources &res, const BasisSet &basis,
             std::shared_ptr<const Buffers::device<double>> u,
             const std::span<CoulombPairClass> j_classes,
             const std::span<OrderedPairClass> k_classes, const SphericalMatrixAO &c,
             const int32_t n_occ, const double *const delta, const Spin spin)
    : dev_idx_{res.dev_idx()}, nao_{static_cast<int32_t>(c.n())}, n_occ_{n_occ},
      n_virt_{static_cast<int32_t>(c.n()) - n_occ}, spin_{spin}, c_{c.data()}, delta_{delta},
      j_classes_{j_classes}, k_classes_{k_classes}, density_{res, basis, u},
      response_{res, basis, std::move(u)}, other_{res, basis},
      half_(static_cast<size_t>(c.n()) * static_cast<size_t>(n_occ), res.device()) {}

Status Sigma::apply(DeviceResources &res, const int32_t k, const double *const x, double *const y) {
  if (res.dev_idx() != dev_idx_ || k < 0) {
    return err_config("cis::Sigma::apply: needs the sigma's device {} and k >= 0 (device {}, "
                      "k = {})",
                      dev_idx_, res.dev_idx(), k);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const int32_t n_ov = size();
  const double *const c_virt = c_ + (static_cast<size_t>(nao_) * static_cast<size_t>(n_occ_));
  // σ's two-electron weight, and which of nonsymmetric_jk's outputs is R:
  // the one response_ carries back to the spherical layout.
  const bool singlet = spin_ == Spin::Singlet;
  const double *const weight = singlet ? &kTwo<double> : &kNegOne<double>;
  CartesianMatrixAO &g = singlet ? response_.cartesian() : other_;
  CartesianMatrixAO &kd = singlet ? other_ : response_.cartesian();

  for (size_t col = 0; col < static_cast<size_t>(k); ++col) {
    // b as the n_virt x n_occ column-major matrix B = X^T: element (a, i) at
    // a + i * n_virt is b_ia.
    const double *const b = x + (col * static_cast<size_t>(n_ov));
    double *const s = y + (col * static_cast<size_t>(n_ov));

    // 1. half = C_virt B (nao x n_occ), then D = C_occ half^T = C_occ X C_virt^T.
    FBF_TRY(gpu_status(
        gemm(blas, nao_, n_occ_, n_virt_, &kOne<double>, c_virt, b, &kZero<double>, half_.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, nao_, nao_, n_occ_, &kOne<double>, c_, half_.data(), &kZero<double>,
        density_.spherical().data())));
    // 2.-4. D onto the Cartesian layout, its response, R back.
    FBF_TRY(density_.spherical_to_cartesian(res));
    FBF_TRY(nonsymmetric_jk(res, j_classes_, k_classes_, density_.cartesian(), g, kd));
    FBF_TRY(response_.cartesian_to_spherical(res));
    // 5. σ = Δε ∘ b; half = R^T C_occ; σ += w C_virt^T half, which is
    //    element (a, i) of w (C_occ^T R C_virt)^T. R is not symmetric, and
    //    σ_ia contracts C_occ with its row index: (ij|ab) X_jb is
    //    sum_mn C_mi K[D]_mn C_na, with K_mn = sum_ls D_ls (ml|ns).
    FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, n_ov, 1, b, n_ov,
                                                  delta_, 1, s, n_ov)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nao_, n_occ_, nao_, &kOne<double>,
                                               response_.spherical().data(), c_, &kZero<double>,
                                               half_.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_virt_, n_occ_, nao_, weight, c_virt,
                                               half_.data(), &kOne<double>, s)));
  }
  return {};
}

} // namespace free_breakfast::cis
