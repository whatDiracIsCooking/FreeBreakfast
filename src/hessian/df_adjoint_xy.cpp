/// @file
/// @brief df_adjoint_xy: Sum h_bar h^xy + Sum Lambda B^xy, the second
/// derivative of a linear functional of the density-fitted MO integrals at a
/// general symmetric adjoint (h_bar, Lambda), along the orthonormal path (the
/// Lowdin connection to second order) and the fitted index's symmetric gauge
/// to second order. Every piece is the second-derivative integrals at the
/// adjoint's AO forms, or a Gram matrix of per-coordinate stacks built over
/// mcscf::df_integral_derivatives' first derivatives (README.md "The DF
/// second derivative at an MO adjoint").

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kHalf, kNegHalf, kTwo, kNegTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, gemv, geam, trsm, gemmStridedBatched
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.hessian;
import free_breakfast.integrals.eri.density_fitting.gradient;
import free_breakfast.integrals.eri.density_fitting.hessian;
import free_breakfast.integrals.eri.density_fitting.mo;
import free_breakfast.mcscf.third_derivative;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;
import :atom_block;
import :state_specific; // Ledger

#include "diagnostics.h"

namespace free_breakfast::hessian {

Result<std::vector<double>> df_adjoint_xy(DeviceResources &res, const Molecule &molecule,
                                          const BasisSet &basis,
                                          const std::span<const PrimitivePairClass> classes,
                                          const BasisSet &aux, const SphericalMatrixAO &h,
                                          const Buffers::device<double> &b, const double *const l,
                                          const SphericalMatrixAO &c, const double *const h_bar,
                                          const double *const lambda, const size_t max_bytes) {
  constexpr std::string_view kWho = "hessian::df_adjoint_xy";
  const size_t nmo = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  if (n_atom == 0 || molecule.natoms() != n_atom || aux.natoms() != n_atom || nmo == 0 ||
      naux == 0 || c.n() != nmo || h.n() != nmo || b.num_elements() != naux * nmo * nmo) {
    return err_config("{}: needs a molecule and two bases on the same atoms, C and h nao x nao "
                      "and B naux nao^2 long ({}, {} and {} atoms, nao {}, naux {}, C {} x {}, h "
                      "{} x {}, B {} doubles)",
                      kWho, molecule.natoms(), n_atom, aux.natoms(), nmo, naux, c.n(), c.n(), h.n(),
                      h.n(), b.num_elements());
  }
  if (l == nullptr || h_bar == nullptr || lambda == nullptr) {
    return err_config("{}: L, h_bar and Lambda must be given", kWho);
  }
  const size_t nn = nmo * nmo;
  const size_t n_x = 3 * static_cast<size_t>(n_atom);
  const size_t factor = naux * nn;
  const size_t sq = naux * naux;
  constexpr auto kInt32Max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (factor > kInt32Max || sq > kInt32Max || n_x * nn > kInt32Max) {
    return err_unsupported("{}: naux nmo^2 = {}, naux^2 = {} or 3 n_atom nmo^2 = {} exceeds "
                           "int32_t",
                           kWho, factor, sq, n_x * nn);
  }
  size_t max_blk = 0;
  for (uint32_t a = 0; a < n_atom; ++a) {
    const auto [p0, p1] = atom_function_range(aux, a);
    max_blk = std::max(max_blk, p1 - p0);
  }

  // Held for the whole call, across b_mo, metric_derivative,
  // df_integral_derivatives and the second-derivative integrals, which all
  // take the workspace, so the call's own (docs/style.md "Scratch memory"):
  // B over every MO pair (Q fastest), Lambda~ (Q slowest; Gam at the end), one
  // naux nmo^2 scratch (Y, Y', Gam's half transform), the tau^x stack, one
  // more nmo^2 stack (N^y, then V^y, then sym(X) tau^x), M1, G, two naux^2
  // scratches, h_MO, sym(X), the gemmwich work, -1/2 sym(X) and the five
  // Gram matrices. No per-coordinate naux^2 stack: mu^x meets the products
  // only through M^x's atom block of M1 (against_mu below).
  // df_integral_derivatives' own buffers beside its first chunk (its
  // integral_derivative.cpp's count) are checked here too, so that nothing
  // is allocated before a refusal.
  const size_t own = (3 * factor) + (2 * n_x * nn) + (6 * sq) + (4 * nn) + (5 * n_x * n_x);
  const size_t derivative_one = (2 * factor) + nn + (4 * n_x * nn) + (3 * sq) + (3 * nn * max_blk);
  const ScopedDeviceIndex on{res.dev_idx()};
  const size_t memory = FBF_TRY(res.memory_budget());
  const size_t budget = max_bytes > 0 ? std::min(max_bytes, memory) : memory;
  // The AO matrix beside them is one more nmo^2.
  const size_t need = sizeof(double) * (own + nn + derivative_one);
  if (need > budget) {
    return err_unsupported("{}: {} orbitals, {} fitting functions and {} coordinates (B and Lambda "
                           "over every MO pair, naux nmo^2 = {} doubles, every coordinate's "
                           "tau^x, and one coordinate's W^x) need {} bytes of device memory, and "
                           "device {} has {} to give",
                           kWho, nmo, naux, n_x, factor, need, res.dev_idx(), budget);
  }

  Buffers::device<double> held(own, res.device());
  double *const b_q = held.data();
  double *const lam_t = b_q + factor;
  double *const y_buf = lam_t + factor;
  double *const tau_s = y_buf + factor;
  double *const stack = tau_s + (n_x * nn);
  double *const m1 = stack + (n_x * nn);
  double *const g = m1 + (3 * sq);
  double *const sq_a = g + sq;
  double *const sq_b = sq_a + sq;
  double *const h_mo = sq_b + sq;
  double *const x_sym = h_mo + nn;
  double *const work = x_sym + nn;
  double *const ao_mat = work + nn;
  double *const products = ao_mat + nn;
  double *const p1 = products;
  double *const p2 = p1 + (n_x * n_x);
  double *const p4 = p2 + (n_x * n_x);
  double *const p5 = p4 + (n_x * n_x);
  double *const p6 = p5 + (n_x * n_x);
  SphericalMatrixAO ao(res, basis);

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto n = static_cast<int32_t>(nmo);
  const auto q = static_cast<int32_t>(naux);
  const auto nx = static_cast<int32_t>(n_x);
  const auto n_nn = static_cast<int32_t>(nn);
  const auto n_sq = static_cast<int32_t>(sq);

  // h_MO = C^T h C; B_Q,pq at Q + naux (p + nmo q); Lambda~, Lambda in W^x's
  // layout, (p, q, Q) at p + nmo (q + nmo Q): naux x nmo^2 transposed.
  FBF_TRY(gpu_status(gemmwich(blas, n, n, c.data(), h.data(), work, h_mo)));
  const OrbitalRange all{.first = 0, .count = n};
  FBF_TRY(b_mo(res, q, n, n, b.data(), c.data(), all, all, b_q));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n_nn, q,
                                                &kOne<double>, lambda, q, &kZero<double>, lambda,
                                                n_nn, lam_t, n_nn)));

  // sym(X), X = 2 h_MO h_bar + 2 Sum_Q B_Q Lambda_Q: B and Lambda as the
  // (naux nmo) x nmo matrices of rows (Q, r), each B_Q symmetric.
  FBF_TRY(gpu_status(gemm(blas, n, &kTwo<double>, h_mo, h_bar, &kZero<double>, work)));
  FBF_TRY(gpu_status(
      gemm<wwr::WWRBLAS_OP_T>(blas, n, n, q * n, &kTwo<double>, b_q, lambda, &kOne<double>, work)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                                &kHalf<double>, work, n, &kHalf<double>, work, n,
                                                x_sym, n)));
  // sym(G), G_QP = Sum_pq Lambda_Q,pq B_P,pq.
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      blas, q, q, n_nn, &kOne<double>, lambda, b_q, &kZero<double>, sq_a)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, q,
                                                &kHalf<double>, sq_a, q, &kHalf<double>, sq_a, q, g,
                                                q)));

  // <mu^x, H> for every x at once, for any naux x naux @p h, by M^x's atom
  // block rather than mu^x: with mu^x = -1/2 L^-1 M^x L^-T and Z = -1/2
  // L^-T H L^-1 (made in @p h), <mu^x, H> = <M^x, Z>, and M^x = S^ + S^^T,
  // S^ M1's slab k at its columns P on A (x = 3 A + k; M1 below), so
  // <mu^x, H> = <S^, Z + Z^T>: a dot over the columns on A, contiguous in
  // both, the three k of an atom one gemv (lda naux^2). Z + Z^T goes in
  // sq_b; the column @p column, 3 n_atom long, gets every x but those of an
  // atom without fitting functions, whose mu^x is zero and whose rows keep
  // the zero the products start at.
  const auto against_mu = [&](double *const h_in, double *const column) -> Status {
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, q, &kNegHalf<double>, l, q, h_in, q)));
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_N,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, q, &kOne<double>, l, q, h_in, q)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, q,
                                                  &kOne<double>, h_in, q, &kOne<double>, h_in, q,
                                                  sq_b, q)));
    for (uint32_t a = 0; a < n_atom; ++a) {
      const auto [p0, p1_end] = atom_function_range(aux, a);
      if (p1_end == p0) {
        continue;
      }
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
          blas, wwr::WWRBLAS_OP_T, static_cast<int32_t>(naux * (p1_end - p0)), 3, &kOne<double>,
          m1 + (naux * p0), n_sq, sq_b + (naux * p0), 1, &kZero<double>,
          column + (3 * static_cast<size_t>(a)), 1)));
    }
    return {};
  };

  // <sym(G) mu^x, mu^y>, column x by against_mu at sym(G) mu^x, mu^x made
  // in sq_b one coordinate at a time. M1's slab k holds d(P|Q)/dA_k (A P's
  // centre) at row Q, column P, so with S^ that slab's columns P on A (zero
  // elsewhere), M^x = S^ + S^^T, and mu^x = -1/2 L^-1 M^x L^-T two trsms.
  FBF_TRY(metric_derivative(res, aux, m1));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(products, 0, sizeof(double) * 5 * n_x * n_x, stream)));
  for (uint32_t a = 0; a < n_atom; ++a) {
    const auto [p0, p1_end] = atom_function_range(aux, a);
    const size_t nblk = p1_end - p0;
    if (nblk == 0) {
      continue;
    }
    for (size_t k = 0; k < 3; ++k) {
      const size_t x = (3 * static_cast<size_t>(a)) + k;
      FBF_TRY(gpu_status(wwr::wwrMemsetAsync(sq_a, 0, sizeof(double) * sq, stream)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(sq_a + (naux * p0), m1 + (k * sq) + (naux * p0),
                                             sizeof(double) * naux * nblk,
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q,
                                                    q, &kOne<double>, sq_a, q, &kOne<double>, sq_a,
                                                    q, sq_b, q)));
      FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
          blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_N,
          wwr::WWRBLAS_DIAG_NON_UNIT, q, q, &kNegHalf<double>, l, q, sq_b, q)));
      FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
          blas, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
          wwr::WWRBLAS_DIAG_NON_UNIT, q, q, &kOne<double>, l, q, sq_b, q)));
      FBF_TRY(gpu_status(gemm(blas, q, &kOne<double>, g, sq_b, &kZero<double>, sq_a)));
      FBF_TRY(against_mu(sq_a, p6 + (x * n_x)));
    }
  }

  // Per coordinate y of each chunk of (h1^y, tau^y, W^y): tau^y kept; N^y =
  // Sum_Q Lambda_Q W^y_Q + h_bar h1^y (Lambda~ and W^y side by side as
  // nmo x (nmo naux), each slab symmetric); and column y of
  // <mu^x, H^y - 2 Lambda (B tau^y)^T>, H^y = Lambda W^y over the pairs,
  // by against_mu.
  const auto consume = [&](const int32_t x0, const int32_t count, const double *const h1,
                           const double *const tau, const double *const w1) -> Status {
    for (size_t j = 0; j < static_cast<size_t>(count); ++j) {
      const size_t y = static_cast<size_t>(x0) + j;
      const double *const w = w1 + (j * factor);
      const double *const tau_y = tau + (j * nn);
      double *const n_y = stack + (y * nn);
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(tau_s + (y * nn), tau_y, sizeof(double) * nn,
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, n, n, n * q, &kOne<double>, lam_t, w, &kZero<double>, n_y)));
      FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, h_bar, h1 + (j * nn), &kOne<double>, n_y)));
      FBF_TRY(gpu_status(gemm(blas, q, q, n_nn, &kOne<double>, lambda, w, &kZero<double>, sq_a)));
      FBF_TRY(
          gpu_status(gemm(blas, q * n, n, n, &kOne<double>, b_q, tau_y, &kZero<double>, y_buf)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, q, q, n_nn, &kNegTwo<double>, lambda, y_buf, &kOne<double>, sq_a)));
      FBF_TRY(against_mu(sq_a, p2 + (y * n_x)));
    }
    return {};
  };
  // The derivative integrals read the budget at their call, past the held
  // buffers; a caller's max_bytes bounds them to what it leaves.
  FBF_TRY(
      mcscf::df_integral_derivatives(res, molecule, basis, classes, aux, h, b, l, c, consume,
                                     max_bytes > 0 ? budget - (sizeof(double) * (own + nn)) : 0));

  // <tau^x, N^y>; then V^y = Sum_Q Lambda_Q tau^y B_Q + h_bar tau^y h_MO
  // (Y' = tau^y B_Q at Q + naux (r + nmo q), one strided-batched gemm over
  // the naux x nmo slab at each q, B_Q and tau^y symmetric) and
  // <tau^x, V^y>; then <tau^x, sym(X) tau^y>.
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nx, nx, n_nn, &kOne<double>, tau_s, stack,
                                             &kZero<double>, p1)));
  for (size_t y = 0; y < n_x; ++y) {
    const double *const tau_y = tau_s + (y * nn);
    double *const v_y = stack + (y * nn);
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, n, &kOne<double>, b_q, q,
        static_cast<int64_t>(naux * nmo), tau_y, n, 0, &kZero<double>, y_buf, q,
        static_cast<int64_t>(naux * nmo), n)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n, n, q * n, &kOne<double>, lambda, y_buf,
                                               &kZero<double>, v_y)));
    FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, tau_y, h_mo, &kZero<double>, work)));
    FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, h_bar, work, &kOne<double>, v_y)));
  }
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nx, nx, n_nn, &kOne<double>, tau_s, stack,
                                             &kZero<double>, p4)));
  FBF_TRY(gpu_status(gemm(blas, n, n * nx, n, &kOne<double>, x_sym, tau_s, &kZero<double>, stack)));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nx, nx, n_nn, &kOne<double>, tau_s, stack,
                                             &kZero<double>, p5)));

  // The second-derivative integrals. One-electron at C h_bar C^T; the
  // overlap at -1/2 C sym(X) C^T; the metric at -1/2 L^-T sym(G) L^-1; the
  // three-center integrals at Gam = L^-T C Lambda C^T (mcscf::df_gradient's
  // extra adjoint's: a gemm over (Q, p), a strided-batched gemm over n, a
  // trsm), into Lambda~'s buffer, which the products above are done with.
  FBF_TRY(gpu_status(gemmwich<wwr::WWRBLAS_OP_T>(blas, n, n, c.data(), h_bar, work, ao.data())));
  const std::vector<double> one = FBF_TRY(kinetic_hessian(res, basis, classes, ao));
  const std::vector<double> potential =
      FBF_TRY(potential_hessian(res, basis, classes, molecule, ao));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n,
                                                &kNegHalf<double>, x_sym, n, &kZero<double>, x_sym,
                                                n, ao_mat, n)));
  FBF_TRY(gpu_status(gemmwich<wwr::WWRBLAS_OP_T>(blas, n, n, c.data(), ao_mat, work, ao.data())));
  const std::vector<double> overlap = FBF_TRY(overlap_hessian(res, basis, classes, ao));
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(sq_a, g, sizeof(double) * sq, wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
      blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
      wwr::WWRBLAS_DIAG_NON_UNIT, q, q, &kNegHalf<double>, l, q, sq_a, q)));
  FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
      blas, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_N,
      wwr::WWRBLAS_DIAG_NON_UNIT, q, q, &kOne<double>, l, q, sq_a, q)));
  const std::vector<double> metric = FBF_TRY(metric_hessian(res, aux, sq_a));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      blas, q * n, n, n, &kOne<double>, lambda, c.data(), &kZero<double>, y_buf)));
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, n, &kOne<double>, y_buf, q,
      static_cast<int64_t>(naux * nmo), c.data(), n, 0, &kZero<double>, lam_t, q * n,
      static_cast<int64_t>(naux), n)));
  FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
      blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
      wwr::WWRBLAS_DIAG_NON_UNIT, q, n_nn, &kOne<double>, l, q, lam_t, q)));
  const std::vector<double> three_center =
      FBF_TRY(three_center_hessian(res, basis, classes, aux, lam_t));

  std::vector<double> host(5 * n_x * n_x);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), products, sizeof(double) * host.size(),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());

  // The kernels' Hessians are row-major (x, y) at x 3 n_atom + y; the Gram
  // matrices column-major at x + 3 n_atom y.
  const size_t nn_x = n_x * n_x;
  std::vector<double> out(nn_x);
  for (size_t y = 0; y < n_x; ++y) {
    for (size_t x = 0; x < n_x; ++x) {
      const size_t xy = x + (n_x * y);
      const size_t yx = y + (n_x * x);
      const size_t row = (x * n_x) + y;
      out[xy] = one[row] + potential[row] + overlap[row] + metric[row] + three_center[row] +
                host[(4 * nn_x) + xy] + host[(3 * nn_x) + xy] + host[nn_x + xy] + host[nn_x + yx] +
                (2.0 * (host[xy] + host[yx])) - (host[(2 * nn_x) + xy] + host[(2 * nn_x) + yx]);
    }
  }
  return out;
}

MemoryFootprint df_adjoint_xy_footprint(const BasisSet &basis, const BasisSet &aux) {
  const size_t nmo = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  if (n_atom == 0 || aux.natoms() != n_atom || nmo == 0 || naux == 0) {
    return {};
  }
  const size_t nn = nmo * nmo;
  const size_t n_x = 3 * static_cast<size_t>(n_atom);
  const size_t factor = naux * nn;
  const size_t sq = naux * naux;
  const auto n = static_cast<int32_t>(nmo);
  const OrbitalRange all{.first = 0, .count = n};
  const MemoryFootprint one = one_electron_hessian_footprint(basis);
  Ledger l;
  // The held buffers and the AO matrix, then each call in turn beside them.
  l.take(sizeof(double) * ((3 * factor) + (2 * n_x * nn) + (6 * sq) + (4 * nn) + (5 * n_x * n_x)));
  l.take(sizeof(double) * nn);
  l.borrow(b_mo_footprint(static_cast<int32_t>(naux), n, all, all));
  l.run(metric_derivative_footprint(aux));
  l.run(mcscf::df_integral_derivatives_footprint(basis, aux));
  l.run(one);
  l.run(one);
  l.run(one);
  l.run(metric_hessian_footprint(aux));
  l.run(three_center_hessian_footprint(basis, aux));
  return l.footprint(0);
}

} // namespace free_breakfast::hessian
