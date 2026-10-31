/// @file
/// @brief df_rhf_response: S^x and F^(x) carried to their MO blocks, one J/K
/// build per coordinate for G(D_S^x), the CPHF right-hand sides, the lockstep
/// CPHF, and the symmetric form's four terms as gemms over the blocks.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kTwo, kNegOne, kNegTwo, kNegHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, gemmStridedBatched, geam, dgmm, axpy, wwrMemGetInfo
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.gradient;
import free_breakfast.integrals.eri.density_fitting.gradient;
import free_breakfast.integrals.eri.density_fitting.jk;
import free_breakfast.recipes.blas;
import free_breakfast.response.cphf;

#include "diagnostics.h"

namespace free_breakfast::hessian {

Result<RhfResponse> df_rhf_response(DeviceResources &res, const Molecule &molecule,
                                    const BasisSet &basis,
                                    const std::span<const PrimitivePairClass> classes,
                                    const BasisSet &aux, const Buffers::device<double> &b,
                                    const double *l, const Buffers::device<double> &c,
                                    const std::span<const double> mo_energies, const int32_t nocc,
                                    const double tolerance, const int32_t max_iterations) {
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const size_t nmo = mo_energies.size();
  const size_t n_x = 3 * static_cast<size_t>(basis.natoms());
  if (molecule.natoms() != basis.natoms() || nocc < 1 || static_cast<size_t>(nocc) >= nmo ||
      c.num_elements() != nao * nmo || b.num_elements() != naux * nao * nao) {
    return err_config("hessian::df_rhf_response: needs a molecule of the basis's {} atoms, "
                      "1 <= nocc < nmo, C nao nmo and B naux nao^2 long ({} atoms, nocc {}, nmo "
                      "{}, nao {}, naux {}, C {} doubles, B {} doubles)",
                      basis.natoms(), molecule.natoms(), nocc, nmo, nao, naux, c.num_elements(),
                      b.num_elements());
  }
  const auto occ = static_cast<size_t>(nocc);
  const size_t nvir = nmo - occ;
  const size_t nn = nao * nao;
  // One MO stack: C^T X C_o for every coordinate, slab x (nmo x nocc,
  // column-major) at x nmo nocc.
  const size_t stack = nmo * occ * n_x;
  const size_t pairs = nvir * occ;
  constexpr auto kInt32Max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (n_x * nn > kInt32Max || stack > kInt32Max || occ * occ * n_x > kInt32Max) {
    return err_unsupported("hessian::df_rhf_response: 3 n_atom nao^2 = {} or the MO stack {} "
                           "exceeds int32_t",
                           n_x * nn, stack);
  }
  // The call's own buffers at their peak (the MO stacks, the right-hand sides,
  // one coordinate's J/K build) beside the three AO stacks F^(x), J^(x) and
  // K^(x): past the device's memory an allocation would abort (CreatePolicy),
  // not return.
  const size_t held_count = (3 * stack) + (nao * occ) + (3 * nn);
  const size_t bytes = sizeof(double) * (held_count + (pairs * n_x) + (3 * n_x * nn));
  const ScopedDeviceIndex on{res.dev_idx()};
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (bytes > free_bytes) {
    return err_unsupported("hessian::df_rhf_response: the derivative matrices and MO blocks of "
                           "{} coordinates over {} orbital functions need {} bytes of device "
                           "memory, and device {} has {} free",
                           n_x, nao, bytes, res.dev_idx(), free_bytes);
  }

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto n_ao = static_cast<int32_t>(nao);
  const auto n_aux = static_cast<int32_t>(naux);
  const auto n_mo = static_cast<int32_t>(nmo);
  const auto n_vir = static_cast<int32_t>(nvir);
  const auto n_coord = static_cast<int32_t>(n_x);
  const auto occ_cols = static_cast<int32_t>(occ * n_x);

  // S, F^(x) and G(D_S^x)'s MO stacks, and one coordinate's C_o S^x_oo, D_S,
  // J and K. Held across df_jk_derivative, df_coulomb and df_exchange, which
  // take the workspace, so the call's own (docs/style.md "Scratch memory").
  Buffers::device<double> held(held_count, res.device());
  double *const s_mo = held.data();
  double *const f_mo = s_mo + stack;
  double *const g_mo = f_mo + stack;
  double *const factor = g_mo + stack;
  double *const density = factor + (nao * occ);
  double *const j = density + nn;
  double *const k = j + nn;
  const double *const c_occ = c.data();

  // out = C^T X_x C_o for every slab X_x of @p x: X_x C_o, one batched gemm
  // into the workspace, then C^T over all of them at once, the products
  // standing side by side as one nao x (nocc 3 n_atom) matrix.
  const auto to_mo = [&](const double *x, double *out) -> Status {
    auto [y] = res.device_regions<double>(nao * occ * n_x);
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_ao, nocc, n_ao, &kOne<double>, x, n_ao,
        static_cast<int64_t>(nn), c_occ, n_ao, 0, &kZero<double>, y, n_ao,
        static_cast<int64_t>(nao * occ), n_coord)));
    return gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_mo, occ_cols, n_ao, &kOne<double>, c.data(),
                                              y, &kZero<double>, out));
  };

  {
    const Buffers::device<double> sx = FBF_TRY(overlap_derivative(res, basis, classes));
    FBF_TRY(to_mo(sx.data(), s_mo));
  }
  {
    // F^(x) = T^x + V^x + J^(x) - 1/2 K^(x).
    const auto count = static_cast<int32_t>(n_x * nn);
    Buffers::device<double> fx = FBF_TRY(kinetic_derivative(res, basis, classes));
    {
      const Buffers::device<double> vx =
          FBF_TRY(potential_derivative(res, basis, classes, molecule));
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, count, &kOne<double>, vx.data(), 1, fx.data(), 1)));
    }
    Buffers::device<double> jx(n_x * nn, res.device());
    Buffers::device<double> kx(n_x * nn, res.device());
    FBF_TRY(df_jk_derivative(res, basis, classes, aux, b.data(), l, c_occ, nocc, jx.data(),
                             kx.data(), nullptr));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, count, &kOne<double>, jx.data(), 1, fx.data(), 1)));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, count, &kNegHalf<double>, kx.data(), 1, fx.data(), 1)));
    FBF_TRY(to_mo(fx.data(), f_mo));
  }

  // G(D_S^x) = J[D_S^x] - K[D_S^x] / 2, D_S^x = 2 C_o S^x_oo C_o^T: with
  // R = C_o S^x_oo, D_S^x / 2 = C_o R^T (S^x_oo symmetric), so K[D_S^x] / 2 is
  // the two-factor K[C_o R^T]. One J/K build per coordinate, carried to its
  // MO slab as S and F^(x) are.
  for (size_t x = 0; x < n_x; ++x) {
    const size_t at = x * nmo * occ;
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_ao,
                                                  nocc, nocc, &kOne<double>, c_occ, n_ao, s_mo + at,
                                                  n_mo, &kZero<double>, factor, n_ao)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, n_ao, n_ao, nocc, &kTwo<double>, c_occ, factor, &kZero<double>, density)));
    FBF_TRY(df_coulomb(res, n_ao, n_aux, b.data(), density, j));
    FBF_TRY(df_exchange(res, n_ao, n_aux, nocc, b.data(), c_occ, factor, k));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(nn), &kNegOne<double>, k, 1, j, 1)));
    // G C_o into K's space, then C^T (G C_o).
    FBF_TRY(gpu_status(gemm(blas, n_ao, nocc, n_ao, &kOne<double>, j, c_occ, &kZero<double>, k)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_mo, nocc, n_ao, &kOne<double>, c.data(), k,
                                               &kZero<double>, g_mo + at)));
  }

  // B = -F^(x)_vo + G(D_S^x)_vo + S^x_vo diag(eps_occ): every coordinate's vo
  // block is rows nocc.. of its slab, so the blocks of a stack are one
  // nvir x (nocc 3 n_atom) matrix of leading dimension nmo, and packed they
  // are response::cphf's (n_vir n_occ) x 3 n_atom.
  Buffers::device<double> rhs(pairs * n_x, res.device());
  // eps_i at i + nocc x, the diagonal S_vo is scaled by; read by the copy
  // below, which the stream has finished by the sync at the end.
  std::vector<double> eps_tiled(occ * n_x);
  for (size_t x = 0; x < n_x; ++x) {
    std::ranges::copy(mo_energies.first(occ),
                      eps_tiled.begin() + static_cast<std::ptrdiff_t>(occ * x));
  }
  {
    auto [eps, scaled] = res.device_regions<double>(occ * n_x, pairs * n_x);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(eps, eps_tiled.data(), sizeof(double) * occ * n_x,
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_vir, occ_cols, &kNegOne<double>, f_mo + occ,
        n_mo, &kOne<double>, g_mo + occ, n_mo, rhs.data(), n_vir)));
    FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, n_vir, occ_cols,
                                                  s_mo + occ, n_mo, eps, 1, scaled, n_vir)));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(pairs * n_x),
                                                  &kOne<double>, scaled, 1, rhs.data(), 1)));
  }

  RhfResponse out{
      std::vector<double>(n_x * n_x),
      FBF_TRY(response::cphf(res, b, n_aux, c, mo_energies, nocc, rhs, tolerance, max_iterations))};

  // The terms, over the oo blocks packed to nocc^2 x 3 n_atom (column x the
  // block of coordinate x):
  //   P = B^T U + S_oo^T F_oo,   H = -2 (P + P^T),
  //   H += 2 S_oo^T (G_oo + S_oo o E),   E_ij = eps_i + eps_j,
  // so the two terms symmetric only in exact arithmetic are made so by one
  // geam, and Omega and Theta are symmetric to rounding.
  std::vector<double> weights(occ * occ);
  for (size_t col = 0; col < occ; ++col) {
    for (size_t row = 0; row < occ; ++row) {
      weights[row + (occ * col)] = mo_energies[row] + mo_energies[col];
    }
  }
  const size_t oo = occ * occ * n_x;
  auto [s_oo, f_oo, q, e, p, h] =
      res.device_regions<double>(oo, oo, oo, occ * occ, n_x * n_x, n_x * n_x);
  const auto n_oo = static_cast<int32_t>(occ * occ);
  const auto n_pairs = static_cast<int32_t>(pairs);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(e, weights.data(), sizeof(double) * occ * occ,
                                         wwr::wwrMemcpyHostToDevice, stream)));
  // The packing copies: geam with beta zero, B never read.
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nocc,
                                                occ_cols, &kOne<double>, s_mo, n_mo, &kZero<double>,
                                                s_mo, n_mo, s_oo, nocc)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nocc,
                                                occ_cols, &kOne<double>, f_mo, n_mo, &kZero<double>,
                                                f_mo, n_mo, f_oo, nocc)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nocc,
                                                occ_cols, &kOne<double>, g_mo, n_mo, &kZero<double>,
                                                g_mo, n_mo, q, nocc)));
  FBF_TRY(
      gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_coord, n_coord, n_pairs, &kOne<double>, rhs.data(),
                                         out.cphf.solution.data(), &kZero<double>, p)));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_coord, n_coord, n_oo, &kOne<double>, s_oo,
                                             f_oo, &kOne<double>, p)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n_coord,
                                                n_coord, &kNegTwo<double>, p, n_coord,
                                                &kNegTwo<double>, p, n_coord, h, n_coord)));
  // S_oo o E into F_oo's space, which P has read, then G_oo + S_oo o E.
  FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, n_oo, n_coord, s_oo,
                                                n_oo, e, 1, f_oo, n_oo)));
  FBF_TRY(gpu_status(
      wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(oo), &kOne<double>, f_oo, 1, q, 1)));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_coord, n_coord, n_oo, &kTwo<double>, s_oo, q,
                                             &kOne<double>, h)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.hessian.data(), h, sizeof(double) * n_x * n_x,
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  return out;
}

} // namespace free_breakfast::hessian
