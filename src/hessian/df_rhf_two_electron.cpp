/// @file
/// @brief df_rhf_two_electron: the fitted densities' second-derivative
/// contractions, then the fit's cross terms as Gram matrices of L^-1 R^x, one
/// atom block of R^x built per walk over the aux atoms and contracted one
/// pair of atom blocks at a time.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kTwo, kNegOne, kNegTwo, kNegHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, gemv, trsm, gemmStridedBatched, wwrMemGetInfo
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.gradient;
import free_breakfast.integrals.eri.density_fitting.hessian;
import free_breakfast.integrals.eri.density_fitting.mo;
import free_breakfast.recipes.blas;

#include "diagnostics.h"

namespace free_breakfast::hessian {

// [begin, end) of the spherical functions on @p atom of a placed basis: its
// shells are contiguous, so their functions are.
std::pair<size_t, size_t> atom_function_range(const BasisSet &basis, const uint32_t atom) {
  const auto start_of = [&basis](const uint32_t shell) -> size_t {
    return shell < basis.nshells() ? basis.ao_offset_sphe(shell) : basis.nao_sphe();
  };
  const uint32_t first = basis.atom_shell_offset(atom);
  return {start_of(first), start_of(first + basis.atom_shell_count(atom))};
}

Result<RhfTwoElectron> df_rhf_two_electron(DeviceResources &res, const BasisSet &basis,
                                           const std::span<const PrimitivePairClass> classes,
                                           const BasisSet &aux, const Buffers::device<double> &b,
                                           const double *l, const Buffers::device<double> &c,
                                           const int32_t nocc) {
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  if (n_atom == 0 || aux.natoms() != n_atom || nocc < 1 || static_cast<size_t>(nocc) > nao ||
      c.num_elements() < nao * static_cast<size_t>(nocc) || b.num_elements() != naux * nao * nao) {
    return err_config("hessian::df_rhf_two_electron: needs two bases placed on the same atoms, "
                      "1 <= nocc <= nao, C at least nao nocc and B naux nao^2 long ({} and {} "
                      "atoms, nocc {}, nao {}, naux {}, C {} doubles, B {} doubles)",
                      n_atom, aux.natoms(), nocc, nao, naux, c.num_elements(), b.num_elements());
  }
  const auto occ = static_cast<size_t>(nocc);
  const size_t oo = occ * occ;
  const size_t nn = nao * nao;
  const size_t n_x = 3 * static_cast<size_t>(n_atom);
  size_t nblk_max = 0;
  for (uint32_t a = 0; a < n_atom; ++a) {
    const auto [p0, p1] = atom_function_range(aux, a);
    nblk_max = std::max(nblk_max, p1 - p0);
  }
  // R^x for one atom's three coordinates, naux x nocc^2 each: one atom block.
  const size_t block = 3 * naux * oo;
  constexpr auto kInt32Max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (naux * nn > kInt32Max || block > kInt32Max || naux * occ * nao > kInt32Max) {
    return err_unsupported("hessian::df_rhf_two_electron: naux nao^2 = {}, an atom block 3 naux "
                           "nocc^2 = {} or naux nocc nao = {} exceeds int32_t",
                           naux * nn, block, naux * occ * nao);
  }
  // Held for the whole call: the fitted pairs c and a vector of twos. The
  // explicit part's: Gam, its half back-transform Y, W, d and D. The cross
  // terms': M1, one aux atom's derivative block X, two atom blocks, their
  // Coulomb vectors and two 3 x 3 products. All are held across calls that
  // take the workspace (b_mo, the Hessian and derivative integrals), so they
  // are the call's own (docs/style.md "Scratch memory"); past the device's
  // memory an allocation would abort (CreatePolicy), not return.
  const size_t held_count = (naux * oo) + occ;
  const size_t explicit_count = (naux * nn) + (naux * occ * nao) + (naux * naux) + naux + nn;
  const size_t cross_count =
      (3 * naux * naux) + (3 * nn * nblk_max) + (2 * block) + (2 * 3 * naux) + (2 * 9);
  const size_t bytes = sizeof(double) * (held_count + std::max(explicit_count, cross_count));
  const ScopedDeviceIndex on{res.dev_idx()};
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (bytes > free_bytes) {
    return err_unsupported("hessian::df_rhf_two_electron: the fitted densities and a pair of "
                           "atom blocks of the cross term over {} orbital and {} fitting "
                           "functions and {} occupied orbitals need {} bytes of device memory, "
                           "and device {} has {} free",
                           nao, naux, occ, bytes, res.dev_idx(), free_bytes);
  }

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto n = static_cast<int32_t>(nao);
  const auto q = static_cast<int32_t>(naux);
  const auto n_oo = static_cast<int32_t>(oo);
  const double *const c_occ = c.data();

  Buffers::device<double> held(held_count, res.device());
  double *const fitted = held.data();
  double *const twos = fitted + (naux * oo);
  // Read by the copy below, which the stream has finished by the first sync.
  const std::vector<double> twos_host(occ, 2.0);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(twos, twos_host.data(), sizeof(double) * occ,
                                         wwr::wwrMemcpyHostToDevice, stream)));
  // c_P,ij = (L^-T B_ij)_P, [P][i][j]: the naux x nocc^2 matrix at
  // P + naux (i + nocc j), symmetric in i and j.
  const OrbitalRange occupied{.first = 0, .count = nocc};
  FBF_TRY(b_mo(res, q, n, nocc, b.data(), c_occ, occupied, occupied, fitted));
  FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
      blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
      wwr::WWRBLAS_DIAG_NON_UNIT, q, n_oo, &kOne<double>, l, q, fitted, q)));
  // The pairs' diagonal c_P,ii is the naux x nocc matrix of leading dimension
  // naux (nocc + 1), so 2 Sum_i of it is one gemv against the twos.
  const auto diagonal_ld = static_cast<int32_t>(naux * (occ + 1));

  RhfTwoElectron out;
  {
    // The explicit part, gradient::df_rhf's weights: E_J's Gam = D (x) d and
    // W = -1/2 d d^T, d = M^-1 u = 2 Sum_i c_ii; E_x's Gam_mnP = -2 Sum_ij
    // C_mi C_nj c_P,ij and W = Sum_ij c_ij c_ij^T.
    Buffers::device<double> scratch(explicit_count, res.device());
    double *const gamma = scratch.data();
    double *const y = gamma + (naux * nn);
    double *const w = y + (naux * occ * nao);
    double *const d = w + (naux * naux);
    double *const density = d + naux;
    FBF_TRY(
        gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, nocc, &kOne<double>,
                                              fitted, diagonal_ld, twos, 1, &kZero<double>, d, 1)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, n, n, nocc, &kTwo<double>, c_occ, c_occ, &kZero<double>, density)));
    // Gam = d vec(D)^T, element (P, n + nao m) = d_P D_nm.
    FBF_TRY(gpu_status(gemm(blas, q, static_cast<int32_t>(nn), 1, &kOne<double>, d, density,
                            &kZero<double>, gamma)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, q, q, 1, &kNegHalf<double>,
                                                                  d, d, &kZero<double>, w)));
    out.coulomb = FBF_TRY(three_center_hessian(res, basis, classes, aux, gamma));
    const std::vector<double> coulomb_metric = FBF_TRY(metric_hessian(res, aux, w));
    std::ranges::transform(out.coulomb, coulomb_metric, out.coulomb.begin(), std::plus{});

    // Y_P,in = Sum_j c_P,ij C_nj at P + naux (i + nocc n); then for each n,
    // Gam_(P, n, m) = -2 Sum_i Y_P,in C_mi (gradient::df_rhf's two products).
    const auto rows = static_cast<int32_t>(naux * occ);
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, rows, n, nocc, &kOne<double>, fitted, c_occ, &kZero<double>, y)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, nocc, &kNegTwo<double>, y, q,
        int64_t{rows}, c_occ, n, 0, &kZero<double>, gamma, q * n, static_cast<int64_t>(naux), n)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, n_oo, &kOne<double>, fitted, fitted, &kZero<double>, w)));
    out.exchange = FBF_TRY(three_center_hessian(res, basis, classes, aux, gamma));
    const std::vector<double> exchange_metric = FBF_TRY(metric_hessian(res, aux, w));
    std::ranges::transform(out.exchange, exchange_metric, out.exchange.begin(), std::plus{});
  }

  Buffers::device<double> walk(cross_count, res.device());
  double *const m1 = walk.data();
  double *const x = m1 + (3 * naux * naux);
  double *const r_a = x + (3 * nn * nblk_max);
  double *const r_b = r_a + block;
  double *const t_a = r_b + block;
  double *const t_b = t_a + (3 * naux);
  double *const h_k = t_b + (3 * naux);
  double *const h_j = h_k + 9;
  FBF_TRY(metric_derivative(res, aux, m1));

  // One atom block: for x = 3 A + k, r = L^-1 R^x into @p r (slab k, naux x
  // nocc^2 at P + naux (j + nocc i)) and t = L^-1 g^x = 2 Sum_i r_ii into
  // @p t (naux at k naux). With X (three_center_derivative) the derivative on
  // m's centre, d(mn|P)/dA_k = -[m on A] X_mn - [n on A] X_nm + [P on A]
  // (X_mn + X_nm), so with Y_ij = Sum_{m on A, n} C_mi C_nj X_mn,P and Z the
  // same over every m,
  //   T^x_ij = -(Y_ij + Y_ji) + [P on A] (Z_ij + Z_ji),
  // each aux atom's rows from its own block of X; and with M1[k] the metric's
  // derivative on P's centre, (M^x c)_P = [P on A] Sum_Q M1[k][P][Q] c_Q +
  // Sum_{Q on A} M1[k][Q][P] c_Q.
  const auto build = [&](const uint32_t atom, double *r, double *t) -> Status {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(r, 0, sizeof(double) * block, stream)));
    const auto [a0, a1] = atom_function_range(basis, atom);
    for (uint32_t aux_atom = 0; aux_atom < n_atom; ++aux_atom) {
      const auto [p0, p1] = atom_function_range(aux, aux_atom);
      const size_t nblk = p1 - p0;
      if (nblk == 0) {
        continue;
      }
      FBF_TRY(three_center_derivative(res, basis, classes, aux, aux_atom, x));
      // W[k][m][P][j] = Sum_n X[k][m][n][P] C_nj, at P + nblk (j + nocc (m +
      // nao k)): one gemm per (k, m), each X block nblk x nao.
      const auto blk = static_cast<int32_t>(nblk);
      auto [w] = res.device_regions<double>(3 * nao * nblk * occ);
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, blk, nocc, n, &kOne<double>, x, blk,
          static_cast<int64_t>(nblk * nao), c_occ, n, 0, &kZero<double>, w, blk,
          static_cast<int64_t>(nblk * occ), 3 * n)));
      for (size_t k = 0; k < 3; ++k) {
        const double *const w_k = w + (k * nao * nblk * occ);
        double *const r_k = r + (k * naux * oo) + p0;
        // alpha Sum_{m in [m0, m0 + count)} C_m. W[k][m][P][.], into rows P of
        // R^x as both (i, j) and (j, i): one gemm per j, then one per i.
        const auto add = [&](const size_t m0, const size_t count, const double *alpha) -> Status {
          const double *const w_m = w_k + (nblk * occ * m0);
          const auto nm = static_cast<int32_t>(count);
          const auto ld_w = static_cast<int32_t>(nblk * occ);
          FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
              blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, blk, nocc, nm, alpha, w_m, ld_w,
              static_cast<int64_t>(nblk), c_occ + m0, n, 0, &kOne<double>, r_k,
              static_cast<int32_t>(naux * occ), static_cast<int64_t>(naux), nocc)));
          return gpu_status(wwr::gemmStridedBatched<double, int32_t>(
              blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, blk, nocc, nm, alpha, w_m, ld_w,
              static_cast<int64_t>(nblk), c_occ + m0, n, 0, &kOne<double>, r_k, q,
              static_cast<int64_t>(naux * occ), nocc));
        };
        if (a1 > a0) {
          FBF_TRY(add(a0, a1 - a0, &kNegOne<double>));
        }
        if (aux_atom == atom) {
          FBF_TRY(add(0, nao, &kOne<double>));
        }
      }
    }
    const auto [q0, q1] = atom_function_range(aux, atom);
    const auto nq = static_cast<int32_t>(q1 - q0);
    for (size_t k = 0; k < 3; ++k) {
      const double *const m1_k = m1 + (k * naux * naux);
      double *const r_k = r + (k * naux * oo);
      if (nq > 0) {
        // M1[k] is column-major with M1[k][P][Q] at (Q, P): its columns on A,
        // transposed, against c for the rows on A, and the same columns as
        // they stand against c's rows on A for every row.
        FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
            blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nq, n_oo, q, &kNegOne<double>,
            m1_k + (naux * q0), q, fitted, q, &kOne<double>, r_k + q0, q)));
        FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q, n_oo, nq, &kNegOne<double>,
            m1_k + (naux * q0), q, fitted + q0, q, &kOne<double>, r_k, q)));
      }
    }
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_N,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, 3 * n_oo, &kOne<double>, l, q, r, q)));
    for (size_t k = 0; k < 3; ++k) {
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, nocc, &kOne<double>,
                                                    r + (k * naux * oo), diagonal_ld, twos, 1,
                                                    &kZero<double>, t + (k * naux), 1)));
    }
    return {};
  };

  // H_AB, both orders, from the blocks of atoms A and B: -2 r_A^T r_B for the
  // exchange and t_A^T t_B for the Coulomb cross term, added to the explicit
  // part.
  std::array<double, 9> cross_k{};
  std::array<double, 9> cross_j{};
  const auto contract = [&](const uint32_t atom_a, const uint32_t atom_b, const double *r_other,
                            const double *t_other) -> Status {
    FBF_TRY(
        gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, 3, 3, static_cast<int32_t>(naux * oo),
                                           &kNegTwo<double>, r_a, r_other, &kZero<double>, h_k)));
    FBF_TRY(gpu_status(
        gemm<wwr::WWRBLAS_OP_T>(blas, 3, 3, q, &kOne<double>, t_a, t_other, &kZero<double>, h_j)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(cross_k.data(), h_k, sizeof(double) * 9,
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(cross_j.data(), h_j, sizeof(double) * 9,
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    for (size_t k = 0; k < 3; ++k) {
      for (size_t m = 0; m < 3; ++m) {
        // Element (k, m) of the 3 x 3 product is coordinate 3 A + k with 3 B + m.
        const size_t xa = (3 * size_t{atom_a}) + k;
        const size_t xb = (3 * size_t{atom_b}) + m;
        const size_t at = (xa * n_x) + xb;
        out.exchange[at] += cross_k[k + (3 * m)];
        out.coulomb[at] += cross_j[k + (3 * m)];
        if (atom_a != atom_b) {
          const size_t mirror = (xb * n_x) + xa;
          out.exchange[mirror] += cross_k[k + (3 * m)];
          out.coulomb[mirror] += cross_j[k + (3 * m)];
        }
      }
    }
    return {};
  };

  for (uint32_t atom_a = 0; atom_a < n_atom; ++atom_a) {
    FBF_TRY(build(atom_a, r_a, t_a));
    FBF_TRY(contract(atom_a, atom_a, r_a, t_a));
    for (uint32_t atom_b = 0; atom_b < atom_a; ++atom_b) {
      FBF_TRY(build(atom_b, r_b, t_b));
      FBF_TRY(contract(atom_a, atom_b, r_b, t_b));
    }
  }
  return out;
}

} // namespace free_breakfast::hessian
