/// @file
/// @brief df_rhf: the shared one-electron terms, then DF-J (a gemv, a trsv and
/// two rank-one gemms) and DF-K (b_mo, a trsm, a gemm, a strided-batched gemm
/// and a gemm), each against the metric's and the three-center integrals'
/// derivatives.

module free_breakfast.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kNegTwo, kNegHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemv, trsv, trsm, gemmStridedBatched, wwrMemGetInfo
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.gradient;
import free_breakfast.integrals.eri.density_fitting.mo;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast::gradient {

// a + b, element by element: one term out of its two derivative pieces.
std::vector<double> sum_of(std::vector<double> a, const std::vector<double> &b) {
  std::ranges::transform(a, b, a.begin(), std::plus{});
  return a;
}

Result<RhfTerms> df_rhf(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                        const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                        const double *b, const double *l, const SphericalMatrixAO &c,
                        const std::span<const double> mo_energies, const int32_t nocc) {
  RhfTerms g;
  const SphericalMatrixAO d =
      FBF_TRY(one_electron_terms(res, molecule, basis, classes, c, mo_energies, nocc, g));
  const size_t n_rows = g.nuclear_repulsion.size();
  const int64_t nao = basis.nao_sphe();
  const int64_t naux = aux.nao_sphe();
  if (naux * nao * nao > std::numeric_limits<int32_t>::max()) {
    return err_unsupported("gradient::df_rhf: naux nao^2 = {} exceeds int32_t", naux * nao * nao);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  if (nocc == 0 || naux == 0) {
    // No density, or nothing to fit it with: no two-electron energy.
    g.coulomb.assign(n_rows, 0.0);
    g.exchange.assign(n_rows, 0.0);
  } else {
    // Gam, its two halves' fitted pairs c and their back-transform Y, the
    // metric weights, and d. Past the device's memory an allocation would
    // abort (CreatePolicy), not return.
    const auto pairs = static_cast<size_t>(nao * nao);
    const size_t gamma_count = static_cast<size_t>(naux) * pairs;
    const size_t c_count = static_cast<size_t>(naux) * nocc * nocc;
    const size_t y_count = static_cast<size_t>(naux) * nocc * static_cast<size_t>(nao);
    const size_t w_count = static_cast<size_t>(naux) * static_cast<size_t>(naux);
    const size_t bytes =
        sizeof(double) * (gamma_count + c_count + y_count + w_count + static_cast<size_t>(naux));
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
    if (bytes > free_bytes) {
      return err_unsupported("gradient::df_rhf: Gam over {} orbital and {} fitting functions and "
                             "its scratch need {} bytes of device memory, and device {} has {} "
                             "free",
                             nao, naux, bytes, res.dev_idx(), free_bytes);
    }
    Buffers::device<double> gamma(gamma_count, res.device());
    Buffers::device<double> fitted(c_count, res.device());
    Buffers::device<double> y(y_count, res.device());
    Buffers::device<double> w(w_count, res.device());
    Buffers::device<double> fitted_density(static_cast<size_t>(naux), res.device());
    const wwr::wwrblasHandle_t blas = res.blas().get();
    const auto n = static_cast<int32_t>(nao);
    const auto q = static_cast<int32_t>(naux);
    const auto nn = static_cast<int32_t>(pairs);

    // Coulomb. d = L^-T (B vec D) = M^-1 (P|mn) D_mn; then Gam = d vec(D)^T,
    // element (P, n + nao m) = d_P D_nm, and W = -1/2 d d^T.
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, nn, &kOne<double>, b,
                                                  q, d.data(), 1, &kZero<double>,
                                                  fitted_density.data(), 1)));
    FBF_TRY(gpu_status(wwr::trsv<double, int32_t>(blas, wwr::WWRBLAS_FILL_MODE_LOWER,
                                                  wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_NON_UNIT, q,
                                                  l, q, fitted_density.data(), 1)));
    FBF_TRY(gpu_status(gemm(blas, q, nn, 1, &kOne<double>, fitted_density.data(), d.data(),
                            &kZero<double>, gamma.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, 1, &kNegHalf<double>, fitted_density.data(), fitted_density.data(),
        &kZero<double>, w.data())));
    g.coulomb = sum_of(FBF_TRY(three_center_gradient(res, basis, classes, aux, gamma.data())),
                       FBF_TRY(metric_gradient(res, aux, w.data())));

    // Exchange. c_P,ij = (L^-T B_ij)_P, [P][i][j]: the naux x nocc^2 matrix
    // at P + naux (i + nocc j).
    const OrbitalRange occupied{.first = 0, .count = nocc};
    FBF_TRY(b_mo(res, q, n, n, b, c.data(), occupied, occupied, fitted.data()));
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, nocc * nocc, &kOne<double>, l, q, fitted.data(), q)));
    // Y_P,in = Sum_j c_P,ij C_nj, at P + naux (i + nocc n); then for each n,
    // Gam_(P, n, m) = -2 Sum_i Y_P,in C_mi, the naux x nao block at
    // P + naux (n + nao m): leading dimension naux nao, consecutive n naux
    // apart.
    const int32_t rows = q * nocc;
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, rows, n, nocc, &kOne<double>, fitted.data(), c.data(), &kZero<double>, y.data())));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, nocc, &kNegTwo<double>, y.data(), q,
        int64_t{rows}, c.data(), n, 0, &kZero<double>, gamma.data(), q * n, naux, n)));
    // W_PQ = Sum_ij c_P,ij c_Q,ij.
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, nocc * nocc, &kOne<double>, fitted.data(), fitted.data(), &kZero<double>,
        w.data())));
    g.exchange = sum_of(FBF_TRY(three_center_gradient(res, basis, classes, aux, gamma.data())),
                        FBF_TRY(metric_gradient(res, aux, w.data())));
  }

  g.total = g.nuclear_repulsion;
  for (const std::vector<double> *term :
       {&g.kinetic, &g.potential, &g.overlap, &g.coulomb, &g.exchange}) {
    g.total = sum_of(std::move(g.total), *term);
  }
  return g;
}

} // namespace free_breakfast::gradient
