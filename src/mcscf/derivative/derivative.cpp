/// @file
/// @brief mcscf::df_derivatives: F^[x] from h^x, two df_jk_derivative calls
/// (the core density and the active one's natural factor) and the active
/// 2-RDM term from a walk over the aux atoms' derivative blocks; F[tau^x] and
/// H[tau^x] from the undifferentiated Fock matrices, ppaa and papa and the J
/// and K of the two response densities, batched over the coordinates; then
/// the orbital right-hand side by a geam per coordinate and the CI one by one
/// sigma per coordinate.

module free_breakfast.mcscf.derivative;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kNegOne, kTwo, kNegTwo, kHalf, kNegHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, gemmStridedBatched, gemv, geam, axpy, scal, dgmm, trsm
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.active_space;
import free_breakfast.integrals.one_electron.gradient;
import free_breakfast.integrals.eri.density_fitting.gradient;
import free_breakfast.integrals.eri.density_fitting.jk;
import free_breakfast.integrals.eri.density_fitting.mo;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.diagonalize;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.response.fock;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// A natural occupation of the weighted 1-RDM below this is a bad RDM, not
// rounding; between it and zero it is taken as zero.
constexpr double kNaturalOccupationFloor = -1e-8;

// The spherical functions of @p basis on @p atom, [first, last).
std::pair<size_t, size_t> derivative_atom_functions(const BasisSet &basis, const uint32_t atom) {
  const uint32_t count = basis.atom_shell_count(atom);
  if (count == 0) {
    return {0, 0};
  }
  const uint32_t first = basis.atom_shell_offset(atom);
  const uint32_t last = first + count - 1;
  return {basis.ao_offset_sphe(first),
          basis.ao_offset_sphe(last) + basis.shells()[last].function_count_sphe()};
}

// The 8-fold symmetric part of an n^4 array (pq|rs)-like, in place: either
// index order, since the result is the same array in both.
void derivative_symmetrize8(std::vector<double> &g, const size_t n) {
  const auto at = [n](const size_t p, const size_t q, const size_t r, const size_t s) {
    return p + (n * (q + (n * (r + (n * s)))));
  };
  std::vector<double> out(g.size());
  for (size_t s = 0; s < n; ++s) {
    for (size_t r = 0; r < n; ++r) {
      for (size_t q = 0; q < n; ++q) {
        for (size_t p = 0; p < n; ++p) {
          out[at(p, q, r, s)] = 0.125 * (g[at(p, q, r, s)] + g[at(q, p, r, s)] + g[at(p, q, s, r)] +
                                         g[at(q, p, s, r)] + g[at(r, s, p, q)] + g[at(s, r, p, q)] +
                                         g[at(r, s, q, p)] + g[at(s, r, q, p)]);
        }
      }
    }
  }
  g = std::move(out);
}

Result<Derivatives>
df_derivatives(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
               const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *const l,
               const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas,
               const int32_t nalpha, const int32_t nbeta, const double *const roots,
               const std::span<const double> weights) {
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const size_t n_atom = basis.natoms();
  const size_t n_x = 3 * n_atom;
  if (molecule.natoms() != basis.natoms() || aux.natoms() != basis.natoms() || n_atom == 0) {
    return err_config("mcscf::df_derivatives: the molecule ({} atoms) and both bases ({} and {}) "
                      "must be one placement",
                      molecule.natoms(), basis.natoms(), aux.natoms());
  }
  if (c.n() != nao || h.n() != nao) {
    return err_config("mcscf::df_derivatives: C is {0} x {0} and h {1} x {1}, but the basis has "
                      "{2} functions",
                      c.n(), h.n(), nao);
  }
  if (ncore < 0 || ncas < 1 || static_cast<size_t>(ncore) + static_cast<size_t>(ncas) > nao) {
    return err_config("mcscf::df_derivatives: ncore {} and ncas {} do not fit {} orbitals", ncore,
                      ncas, nao);
  }
  if (naux == 0 || b.num_elements() != naux * nao * nao) {
    return err_config("mcscf::df_derivatives: B holds {} doubles, but naux nao^2 is {} x {}^2",
                      b.num_elements(), naux, nao);
  }
  if (weights.empty()) {
    return err_config("mcscf::df_derivatives: no roots");
  }
  const auto n_core = static_cast<size_t>(ncore);
  const auto n_act = static_cast<size_t>(ncas);
  const size_t n_occ = n_core + n_act;
  const size_t na2 = n_act * n_act;
  const size_t na3 = na2 * n_act;
  const size_t na4 = na2 * na2;
  const size_t nn = nao * nao;
  const size_t stack = n_x * nn;
  const size_t nroots = weights.size();
  size_t max_blk = 0;
  for (uint32_t a = 0; a < n_atom; ++a) {
    const auto [first, last] = derivative_atom_functions(aux, a);
    max_blk = std::max(max_blk, last - first);
  }
  constexpr auto kInt32Max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (stack > kInt32Max || naux * nn > kInt32Max || nao * na3 * n_x > kInt32Max ||
      naux * na2 * n_x > kInt32Max || 3 * nn * max_blk > kInt32Max) {
    return err_unsupported("mcscf::df_derivatives: {} orbital and {} fitting functions over {} "
                           "coordinates pass the BLAS calls' 32-bit indices",
                           nao, naux, n_x);
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();

  // The call's own buffers at their peak (the response-density phase: the
  // five MO stacks held throughout, the two outputs, eight stacks of the
  // response densities' J and K and their MO blocks, the 2-RDM products and
  // the active-pair intermediates), past which an allocation would abort
  // (CreatePolicy) rather than return. The CI right-hand side is counted
  // from the strings below, once their size is known.
  const size_t held = (15 * stack) + (nao * na3 * n_x) + (3 * naux * na2 * n_x) +
                      (naux * n_act * nao) + (3 * naux * naux) + (3 * nn * max_blk) +
                      (3 * nn * n_act) + (2 * nao * n_occ * n_x);
  {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
    if (sizeof(double) * held > free_bytes) {
      return err_unsupported("mcscf::df_derivatives: the derivative stacks of {} coordinates "
                             "over {} orbital and {} fitting functions need {} bytes of device "
                             "memory, and device {} has {} free",
                             n_x, nao, naux, sizeof(double) * held, res.dev_idx(), free_bytes);
    }
  }

  // The roots' weighted RDMs, Gamma 8-fold symmetrized: F's formula assumes
  // it, and the energy sees only that part.
  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  std::vector<double> gamma(na2);
  std::vector<double> gamma2(na4);
  ci::Rdm rdm_builder = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto n_det = static_cast<size_t>(rdm_builder.size());
  {
    Buffers::device<double> rdm(na2 + na4, res.device());
    FBF_TRY(rdm_builder.rdm12(res, weights, roots, rdm.data(), rdm.data() + na2));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma.data(), rdm.data(), sizeof(double) * na2,
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma2.data(), rdm.data() + na2, sizeof(double) * na4,
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
  }
  for (size_t t = 0; t < n_act; ++t) {
    for (size_t u = 0; u < t; ++u) {
      const double mean = 0.5 * (gamma[(t * n_act) + u] + gamma[(u * n_act) + t]);
      gamma[(t * n_act) + u] = mean;
      gamma[(u * n_act) + t] = mean;
    }
  }
  derivative_symmetrize8(gamma2, n_act);

  const auto n = static_cast<int32_t>(nao);
  const auto q = static_cast<int32_t>(naux);
  const auto nc = ncore;
  const auto na = ncas;
  const auto nx = static_cast<int32_t>(n_x);
  const auto nr = static_cast<int32_t>(nroots);
  const auto s_nn = static_cast<int64_t>(nn);
  const double *const cm = c.data();
  const double *const c_act = cm + (nao * n_core);
  const auto slab = [&](double *base, const size_t x) { return base + (x * nn); };
  const auto zero = [&](double *p, const size_t count) {
    return gpu_status(wwr::wwrMemsetAsync(p, 0, sizeof(double) * count, stream));
  };
  // out_x = C^T X_x C for every slab of a stack: X_x C, one batched gemm into
  // the workspace, then C^T over all of them side by side.
  const auto to_mo = [&](const double *x, double *out) -> Status {
    auto [y] = res.device_regions<double>(stack);
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<double>, x, n, s_nn, cm, n, 0,
        &kZero<double>, y, n, s_nn, nx)));
    return gpu_status(
        gemm<wwr::WWRBLAS_OP_T>(blas, n, n * nx, n, &kOne<double>, cm, y, &kZero<double>, out));
  };

  // The integrals and the generalized Fock at C and the RDMs.
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, q, c, ncore, ncas));
  const response::Fock f0 = FBF_TRY(
      response::fock(res, molecule, h, b, q, c, ncore, ncas, ints.ppaa, ints.papa, gamma, gamma2));
  const double *const gamma_d = f0.gamma.data();
  const double *const gamma2_d = f0.gamma2.data();

  // Held throughout, across calls that take the workspace, so the call's own
  // (docs/style.md "Scratch memory"): tau^x = -1/2 C^T S^x C, h^x's MO block,
  // F^I^x = C^T (h^x + J^x[D_c] - 1/2 K^x[D_c]) C, F^A^x = C^T (J^x[D_a] -
  // 1/2 K^x[D_a]) C, each a stack of nmo x nmo slabs.
  Buffers::device<double> tau(stack, res.device());
  Buffers::device<double> hx(stack, res.device());
  Buffers::device<double> fix(stack, res.device());
  Buffers::device<double> fax(stack, res.device());
  const auto count = static_cast<int32_t>(stack);
  {
    const Buffers::device<double> sx = FBF_TRY(overlap_derivative(res, basis, classes));
    FBF_TRY(to_mo(sx.data(), tau.data()));
    FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, count, &kNegHalf<double>, tau.data(), 1)));
  }
  {
    Buffers::device<double> fi_ao = FBF_TRY(kinetic_derivative(res, basis, classes));
    {
      const Buffers::device<double> vx =
          FBF_TRY(potential_derivative(res, basis, classes, molecule));
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, count, &kOne<double>, vx.data(), 1, fi_ao.data(), 1)));
    }
    FBF_TRY(to_mo(fi_ao.data(), hx.data()));
    if (ncore > 0) {
      Buffers::device<double> jx(stack, res.device());
      Buffers::device<double> kx(stack, res.device());
      FBF_TRY(df_jk_derivative(res, basis, classes, aux, b.data(), l, cm, ncore, jx.data(),
                               kx.data(), nullptr));
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, count, &kOne<double>, jx.data(), 1, fi_ao.data(), 1)));
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, count, &kNegHalf<double>, kx.data(), 1,
                                                    fi_ao.data(), 1)));
    }
    FBF_TRY(to_mo(fi_ao.data(), fix.data()));
  }
  {
    // D_a = C_a gamma C_a^T = 2 X X^T, X = C_a U diag(sqrt(n / 2)) over
    // gamma's natural orbitals, the factor df_jk_derivative takes.
    Buffers::device<double> natural(na2 + n_act, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(natural.data(), gamma_d, sizeof(double) * na2,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(diagonalize(res, na, natural.data(), natural.data() + na2));
    std::vector<double> u(na2 + n_act);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(u.data(), natural.data(), sizeof(double) * u.size(),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    for (size_t k = 0; k < n_act; ++k) {
      const double occupation = u[na2 + k];
      if (occupation < kNaturalOccupationFloor) {
        return err_config("mcscf::df_derivatives: the weighted 1-RDM has a natural occupation "
                          "{:.3e} < 0",
                          occupation);
      }
      const double scale = std::sqrt(std::max(occupation, 0.0) / 2.0);
      for (size_t t = 0; t < n_act; ++t) {
        u[t + (n_act * k)] *= scale;
      }
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(natural.data(), u.data(), sizeof(double) * na2,
                                           wwr::wwrMemcpyHostToDevice, stream)));
    Buffers::device<double> factor(nao * n_act, res.device());
    FBF_TRY(gpu_status(gemm(blas, n, na, na, &kOne<double>, c_act, natural.data(), &kZero<double>,
                            factor.data())));
    Buffers::device<double> jx(stack, res.device());
    Buffers::device<double> kx(stack, res.device());
    FBF_TRY(df_jk_derivative(res, basis, classes, aux, b.data(), l, factor.data(), ncas, jx.data(),
                             kx.data(), nullptr));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, count, &kNegHalf<double>, kx.data(), 1, jx.data(), 1)));
    FBF_TRY(to_mo(jx.data(), fax.data()));
  }

  // The active 2-RDM term of F^[x], A^x_pt = Sum_uvw (pu|vw)^x Gamma_tuvw over
  // the fit (pu|vw)^x = T^x N T + T N T^x - T N M^x N T (N = M^-1), and the
  // active block (tu|vw)^x. With c = N T the fitted pairs (L^-T over b_mo's B):
  //   A^x = C^T V^x + Sum_uP c_P,pu R^x_P,tu,
  //   V^x_mt = Sum_nP d(mn|P)/dx Yt_n,P,t,  Yt_n,P,t = Sum_u C_nu Y_P,tu,
  //   Y = c_aa Gamma,  R^x = (T^x_aa - M^x c_aa) Gamma,
  // T^x_aa the active pairs' derivative three-center integrals.
  Buffers::device<double> ax(nao * n_act * n_x, res.device());
  Buffers::device<double> eri(na4 * n_x, res.device());
  {
    // c_P,up at P + naux (u + ncas p), u active, p every orbital; its active
    // columns are c_aa, (P, u, v) at P + naux (u + ncas v).
    Buffers::device<double> cpa(naux * n_act * nao, res.device());
    FBF_TRY(b_mo(res, q, n, n, b.data(), cm, OrbitalRange{ncore, ncas}, OrbitalRange{0, n},
                 cpa.data()));
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, na * n, &kOne<double>, l, q, cpa.data(), q)));
    const double *const c_aa = cpa.data() + (naux * n_act * n_core);
    const auto qa2 = static_cast<int32_t>(naux * na2);
    const auto a2 = static_cast<int32_t>(na2);
    // Y_P,tu at P + naux (t + ncas u); Gamma as the symmetric ncas^2 square.
    Buffers::device<double> y(naux * na2, res.device());
    FBF_TRY(
        gpu_status(gemm(blas, q, a2, a2, &kOne<double>, c_aa, gamma2_d, &kZero<double>, y.data())));
    Buffers::device<double> m1(3 * naux * naux, res.device());
    FBF_TRY(metric_derivative(res, aux, m1.data()));

    // V^x (nao x ncas per x); T'^x_aa (naux x ncas^2 per x), whose symmetric
    // part in the pair is T^x_aa (README.md); the sums over the aux atoms of
    // Q1 and O per axis, read after the walk; and a vector of ones.
    Buffers::device<double> vx(nao * n_act * n_x, res.device());
    Buffers::device<double> taa(naux * na2 * n_x, res.device());
    Buffers::device<double> q1sum(3 * nao * n_act, res.device());
    Buffers::device<double> osum(3 * nn * n_act, res.device());
    Buffers::device<double> ones(nao, res.device());
    Buffers::device<double> block(3 * nn * max_blk, res.device());
    FBF_TRY(zero(vx.data(), vx.num_elements()));
    FBF_TRY(zero(taa.data(), taa.num_elements()));
    FBF_TRY(zero(q1sum.data(), q1sum.num_elements()));
    FBF_TRY(zero(osum.data(), osum.num_elements()));
    const std::vector<double> ones_host(nao, 1.0);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ones.data(), ones_host.data(), sizeof(double) * nao,
                                           wwr::wwrMemcpyHostToDevice, stream)));
    const auto na_n = static_cast<int32_t>(nao * n_act);
    const auto nn_a = static_cast<int32_t>(nn * n_act);
    for (uint32_t atom = 0; atom < n_atom; ++atom) {
      const auto [p0, p1] = derivative_atom_functions(aux, atom);
      const auto nb = static_cast<int32_t>(p1 - p0);
      if (nb == 0) {
        continue;
      }
      // X[k][m][n][P'] = (nabla_k m n|P) for P on this atom.
      FBF_TRY(three_center_derivative(res, basis, classes, aux, atom, block.data()));
      const auto nbs = static_cast<size_t>(nb);
      auto [yt, q1, o, hh, ex] = res.device_regions<double>(
          nbs * nao * n_act, nao * n_act, nn * n_act, nbs * n_act * nao, nbs * na2);
      // Yt_A[P' + nb (n + nao t)] = Sum_u C_nu Y[(p0 + P') + naux (t + ncas u)].
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nb, n, na, &kOne<double>, y.data() + p0,
          q * na, int64_t{q}, c_act, n, 0, &kZero<double>, yt, nb, int64_t{nb} * n, na)));
      for (size_t k = 0; k < 3; ++k) {
        const double *const xk = block.data() + (k * nn * nbs);
        const size_t on_a = (3 * atom) + k;
        // Q1_mt = Sum_nP' X[k][m][n][P'] Yt[P', n, t]: X as (nb nao) x nao.
        FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n, na, nb * n, &kOne<double>, xk, yt,
                                                   &kZero<double>, q1)));
        FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, na_n, &kOne<double>, q1, 1,
                                                      vx.data() + (nao * n_act * on_a), 1)));
        FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, na_n, &kOne<double>, q1, 1,
                                                      q1sum.data() + (nao * n_act * k), 1)));
        // O[n][m][t] = Sum_P' X[k][n][m][P'] Yt[P', n, t], one n at a time.
        FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
            blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n, na, nb, &kOne<double>, xk, nb,
            int64_t{nb} * n, yt, nb * n, int64_t{nb}, &kZero<double>, o, n, int64_t{n} * na, n)));
        FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, na_n, n, &kOne<double>, o, na_n, ones.data(), 1, &kOne<double>,
            vx.data() + (nao * n_act * on_a), 1)));
        FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, nn_a, &kOne<double>, o, 1,
                                                      osum.data() + (nn * n_act * k), 1)));
        // H[P' + nb (w + ncas m)] = Sum_n X[k][m][n][P'] C_nw, one m at a time.
        FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nb, na, n, &kOne<double>, xk, nb,
            int64_t{nb} * n, c_act, n, 0, &kZero<double>, hh, nb, int64_t{nb} * na, n)));
        for (uint32_t other = 0; other < n_atom; ++other) {
          const auto [m0, m1] = derivative_atom_functions(basis, other);
          if (m1 == m0) {
            continue;
          }
          // E_X = Sum_{m on X} H[., m] C_m., (P', w, v) at P' + nb (w + ncas v).
          FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
              blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nb * na, na,
              static_cast<int32_t>(m1 - m0), &kOne<double>, hh + (nbs * n_act * m0), nb * na,
              c_act + m0, n, &kZero<double>, ex, nb * na)));
          double *const on_x = taa.data() + (naux * na2 * ((3 * other) + k)) + p0;
          double *const on_p = taa.data() + (naux * na2 * on_a) + p0;
          FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                                        nb, a2, &kNegTwo<double>, ex, nb,
                                                        &kOne<double>, on_x, q, on_x, q)));
          FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                                        nb, a2, &kTwo<double>, ex, nb,
                                                        &kOne<double>, on_p, q, on_p, q)));
        }
      }
    }
    // The orbital centres' terms: -[m on X] Q1 and -Sum_{n on X} O.
    for (uint32_t atom = 0; atom < n_atom; ++atom) {
      const auto [m0, m1] = derivative_atom_functions(basis, atom);
      if (m1 == m0) {
        continue;
      }
      const auto rows = static_cast<int32_t>(m1 - m0);
      for (size_t k = 0; k < 3; ++k) {
        double *const v_slab = vx.data() + (nao * n_act * ((3 * atom) + k));
        FBF_TRY(gpu_status(
            wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, rows, na,
                                       &kNegOne<double>, q1sum.data() + (nao * n_act * k) + m0, n,
                                       &kOne<double>, v_slab + m0, n, v_slab + m0, n)));
        FBF_TRY(gpu_status(
            wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, na_n, rows, &kNegOne<double>,
                                       osum.data() + (nn * n_act * k) + (nao * n_act * m0), na_n,
                                       ones.data(), 1, &kOne<double>, v_slab, 1)));
      }
    }
    // G^x = M^x c_aa: M^x_PQ = [P on X] M1[k][P][Q] + [Q on X] M1[k][Q][P],
    // M1[k] read as the naux x naux Mk with Mk(Q, P) = M1[k][P][Q].
    Buffers::device<double> gx(naux * na2 * n_x, res.device());
    FBF_TRY(zero(gx.data(), gx.num_elements()));
    for (uint32_t atom = 0; atom < n_atom; ++atom) {
      const auto [p0, p1] = derivative_atom_functions(aux, atom);
      if (p1 == p0) {
        continue;
      }
      const auto nb = static_cast<int32_t>(p1 - p0);
      for (size_t k = 0; k < 3; ++k) {
        const double *const mk = m1.data() + (k * naux * naux) + (naux * p0);
        double *const g_slab = gx.data() + (naux * na2 * ((3 * atom) + k));
        FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N,
                                                      nb, a2, q, &kOne<double>, mk, q, c_aa, q,
                                                      &kOne<double>, g_slab + p0, q)));
        FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q,
                                                      a2, nb, &kOne<double>, mk, q, c_aa + p0, q,
                                                      &kOne<double>, g_slab, q)));
      }
    }
    // (tu|vw)^x = T'^T c + c^T T' - c^T G, unsymmetrized (the host
    // symmetrizes it with the tau part, below).
    const auto s_qa2 = static_cast<int64_t>(qa2);
    const auto s_a4 = static_cast<int64_t>(na4);
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, a2, a2, q, &kOne<double>, taa.data(), q, s_qa2,
        c_aa, q, 0, &kZero<double>, eri.data(), a2, s_a4, nx)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, a2, a2, q, &kOne<double>, c_aa, q, 0,
        taa.data(), q, s_qa2, &kOne<double>, eri.data(), a2, s_a4, nx)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, a2, a2, q, &kNegOne<double>, c_aa, q, 0,
        gx.data(), q, s_qa2, &kOne<double>, eri.data(), a2, s_a4, nx)));
    // R^x = (T'^x - G^x) Gamma, (P, u, t) at P + naux (u + ncas t), into gx.
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, qa2 * nx, &kNegOne<double>, gx.data(), 1, taa.data(), 1)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q, a2, a2, &kOne<double>, taa.data(), q, s_qa2,
        gamma2_d, a2, 0, &kZero<double>, gx.data(), q, s_qa2, nx)));
    // A^x = C^T V^x + c_pa^T R^x, (p, t) at p + nmo (t + ncas x).
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n, na * nx, n, &kOne<double>, cm, vx.data(),
                                               &kZero<double>, ax.data())));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n, na, q * na, &kOne<double>, cpa.data(),
        q * na, 0, gx.data(), q * na, s_qa2, &kOne<double>, ax.data(), n, int64_t{n} * na, nx)));
  }

  // F^[x]: columns core 2 (F^I^x + F^A^x), columns active F^I^x gamma + A^x.
  Derivatives out{.fock = Buffers::device<double>(stack, res.device()),
                  .rhs_orbital = Buffers::device<double>(stack, res.device()),
                  .rhs_ci = Buffers::device<double>(n_det * nroots * n_x, res.device()),
                  .rhs_theta = std::vector<double>(nroots * nroots * n_x),
                  .hamiltonian = std::vector<double>(nroots * nroots * n_x)};
  // F~^x = F^[x] + F[tau^x], and the inner Fock F^I at I^x, F^I~^x.
  Buffers::device<double> ft(stack, res.device());
  Buffers::device<double> fit(stack, res.device());
  FBF_TRY(zero(out.fock.data(), stack));
  FBF_TRY(zero(ft.data(), stack));
  const auto assemble = [&](const double *fi, const double *fa, double *f) -> Status {
    for (size_t x = 0; x < n_x; ++x) {
      if (ncore > 0) {
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, nc, &kTwo<double>, fi + (x * nn), n,
            &kTwo<double>, fa + (x * nn), n, f + (x * nn), n)));
      }
    }
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, na, na, &kOne<double>, fi + (nao * n_core),
        n, s_nn, gamma_d, na, 0, &kOne<double>, f + (nao * n_core), n, s_nn, nx)));
    for (size_t x = 0; x < n_x; ++x) {
      double *const act = f + (x * nn) + (nao * n_core);
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n,
                                                    na, &kOne<double>, act, n, &kOne<double>,
                                                    ax.data() + (x * nao * n_act), n, act, n)));
    }
    return {};
  };
  FBF_TRY(assemble(fix.data(), fax.data(), out.fock.data()));

  {
    // The response densities' factors: L_c = C tau[:, core] beside R_c =
    // 2 C_c, L_a = C tau[:, act] beside R_a = C_a gamma, each R repeated per
    // coordinate for the block K; dD_c = L_c R_c^T + R_c L_c^T and dD_a the
    // same, the derivatives of D_c and D_a along C -> C (1 + e tau).
    Buffers::device<double> lc(std::max<size_t>(nao * n_core * n_x, 1), res.device());
    Buffers::device<double> rc(std::max<size_t>(nao * n_core * n_x, 1), res.device());
    Buffers::device<double> la(nao * n_act * n_x, res.device());
    Buffers::device<double> ra(nao * n_act * n_x, res.device());
    Buffers::device<double> dens(2 * stack, res.device());
    Buffers::device<double> jd(2 * stack, res.device());
    Buffers::device<double> kd(2 * stack, res.device());
    const auto rep = [&](const double *from, double *to, const size_t len) -> Status {
      for (size_t x = 0; x < n_x; ++x) {
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to + (x * len), from, sizeof(double) * len,
                                               wwr::wwrMemcpyDeviceToDevice, stream)));
      }
      return {};
    };
    const auto response_density = [&](const int32_t first, const int32_t rank, double *left,
                                      double *right, double *d) -> Status {
      const int64_t s_f = int64_t{n} * rank;
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, rank, n, &kOne<double>, cm, n, 0,
          tau.data() + (nao * static_cast<size_t>(first)), n, s_nn, &kZero<double>, left, n, s_f,
          nx)));
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n, rank, &kOne<double>, left, n, s_f,
          right, n, s_f, &kZero<double>, d, n, s_nn, nx)));
      return gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n, rank, &kOne<double>, right, n, s_f,
          left, n, s_f, &kOne<double>, d, n, s_nn, nx));
    };
    FBF_TRY(zero(dens.data(), 2 * stack));
    FBF_TRY(zero(kd.data(), 2 * stack));
    if (ncore > 0) {
      FBF_TRY(rep(cm, rc.data(), nao * n_core));
      FBF_TRY(
          gpu_status(wwr::scal<double, int32_t>(blas, n * nc * nx, &kTwo<double>, rc.data(), 1)));
      FBF_TRY(response_density(0, nc, lc.data(), rc.data(), dens.data()));
    }
    FBF_TRY(gpu_status(
        gemm(blas, n, na, na, &kOne<double>, c_act, gamma_d, &kZero<double>, ra.data())));
    for (size_t x = 1; x < n_x; ++x) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ra.data() + (x * nao * n_act), ra.data(),
                                             sizeof(double) * nao * n_act,
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
    }
    FBF_TRY(response_density(ncore, na, la.data(), ra.data(), dens.data() + stack));
    // G[dD] = J[dD] - 1/2 (K[L R^T] + K[L R^T]^T), core then active.
    FBF_TRY(df_coulomb(res, n, q, 2 * nx, b.data(), dens.data(), jd.data()));
    if (ncore > 0) {
      FBF_TRY(df_exchange(res, n, q, nc, nx, b.data(), lc.data(), rc.data(), kd.data()));
    }
    FBF_TRY(df_exchange(res, n, q, na, nx, b.data(), la.data(), ra.data(), kd.data() + stack));
    for (size_t d = 0; d < 2 * n_x; ++d) {
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n, &kNegHalf<double>, kd.data() + (d * nn),
          n, &kNegHalf<double>, kd.data() + (d * nn), n, dens.data() + (d * nn), n)));
    }
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, 2 * count, &kOne<double>, jd.data(), 1, dens.data(), 1)));
    // Their MO blocks, into jd.
    FBF_TRY(to_mo(dens.data(), jd.data()));
    FBF_TRY(to_mo(dens.data() + stack, jd.data() + stack));

    // F^I~ = F^I^x + tau F^I + F^I tau + C^T G[dD_c] C, and F^A~ the same
    // over F^A and dD_a (into fax, whose F^A^x is no longer needed).
    const auto tilde = [&](const double *f, const double *g_mo, double *inout) -> Status {
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<double>, tau.data(), n, s_nn,
          f, n, 0, &kOne<double>, inout, n, s_nn, nx)));
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<double>, f, n, 0, tau.data(),
          n, s_nn, &kOne<double>, inout, n, s_nn, nx)));
      return gpu_status(wwr::axpy<double, int32_t>(blas, count, &kOne<double>, g_mo, 1, inout, 1));
    };
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(fit.data(), fix.data(), sizeof(double) * stack,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(tilde(f0.fock_inactive.data(), jd.data(), fit.data()));
    FBF_TRY(tilde(f0.fock_active.data(), jd.data() + stack, fax.data()));
  }
  {
    // The active 2-RDM term's tau part: tau Q0 (the outer index), Q0 =
    // F[:, act] - F^I[:, act] gamma the undifferentiated term, and the inner
    // ones, ppaa R1 + 2 papa R2 with R1[a, (v, w, t)] = Sum_u tau_au
    // Gamma_tuvw and R2[a, (u, w, t)] = Sum_v tau_av Gamma_tuvw (Gamma's
    // (v, w) symmetry folds the fourth index into the third's factor 2).
    FBF_TRY(assemble(fit.data(), fax.data(), ft.data()));
    std::vector<double> g12(2 * na4);
    for (size_t t = 0; t < n_act; ++t) {
      for (size_t w = 0; w < n_act; ++w) {
        for (size_t v = 0; v < n_act; ++v) {
          for (size_t u = 0; u < n_act; ++u) {
            const double value = gamma2[(((((t * n_act) + u) * n_act) + v) * n_act) + w];
            g12[u + (n_act * (v + (n_act * (w + (n_act * t)))))] = value;
            g12[na4 + v + (n_act * (u + (n_act * (w + (n_act * t)))))] = value;
          }
        }
      }
    }
    auto [g12_d, q0, r, t23] =
        res.device_regions<double>(2 * na4, nao * n_act, nao * na3 * n_x, nao * n_act * n_x);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(g12_d, g12.data(), sizeof(double) * g12.size(),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(q0, f0.fock.data() + (nao * n_core),
                                           sizeof(double) * nao * n_act,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(gemm(blas, n, na, na, &kNegOne<double>,
                            f0.fock_inactive.data() + (nao * n_core), gamma_d, &kOne<double>, q0)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, na, n, &kOne<double>, tau.data(), n, s_nn,
        q0, n, 0, &kOne<double>, ft.data() + (nao * n_core), n, s_nn, nx)));
    const auto a3 = static_cast<int32_t>(na3);
    const auto rows = static_cast<int32_t>(nao * na2);
    const int64_t s_r = int64_t{n} * a3;
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, a3, na, &kOne<double>,
        tau.data() + (nao * n_core), n, s_nn, g12_d, na, 0, &kZero<double>, r, n, s_r, nx)));
    FBF_TRY(gpu_status(
        gemm(blas, n, na * nx, rows, &kOne<double>, ints.ppaa.data(), r, &kZero<double>, t23)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, a3, na, &kOne<double>,
        tau.data() + (nao * n_core), n, s_nn, g12_d + na4, na, 0, &kZero<double>, r, n, s_r, nx)));
    FBF_TRY(gpu_status(
        gemm(blas, n, na * nx, rows, &kTwo<double>, ints.papa.data(), r, &kOne<double>, t23)));
    for (size_t x = 0; x < n_x; ++x) {
      double *const act = ft.data() + (x * nn) + (nao * n_core);
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n,
                                                    na, &kOne<double>, act, n, &kOne<double>,
                                                    t23 + (x * nao * n_act), n, act, n)));
    }
  }

  // G^x = 2 (F~ - F~^T), zero on the three diagonal blocks.
  {
    const std::array<std::pair<size_t, size_t>, 3> spaces{
        {{0, n_core}, {n_core, n_occ}, {n_occ, nao}}};
    for (size_t x = 0; x < n_x; ++x) {
      double *const g = slab(out.rhs_orbital.data(), x);
      const double *const f = ft.data() + (x * nn);
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n,
                                                    n, &kTwo<double>, f, n, &kNegTwo<double>, f, n,
                                                    g, n)));
      for (const auto &[lo, hi] : spaces) {
        if (hi == lo) {
          continue;
        }
        double *const blk = g + lo + (nao * lo);
        const auto m = static_cast<int32_t>(hi - lo);
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m,
                                                      m, &kZero<double>, blk, n, &kZero<double>,
                                                      blk, n, blk, n)));
      }
    }
  }

  // The CI part: per coordinate H~^x over the active space, E_core~ =
  // Sum_i (h~_ii + F^I~_ii) with h~ = h^x + tau h + h tau, h_eff~ = F^I~'s
  // active block, (tu|vw)~ = (tu|vw)^x + the tau one-index transform from
  // ppaa, T1[t, u, (v, w)] = Sum_a tau_at (au|vw) on each index in turn.
  Buffers::device<double> t1(na4 * n_x, res.device());
  for (size_t x = 0; x < n_x; ++x) {
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, na, na, n, &kOne<double>,
        slab(tau.data(), x) + (nao * n_core), n, 0, ints.ppaa.data() + (nao * n_core), n, s_nn,
        &kZero<double>, t1.data() + (x * na4), na, static_cast<int64_t>(na2),
        static_cast<int32_t>(na2))));
  }
  std::vector<double> h_mo(nn);
  {
    auto [work, mo] = res.device_regions<double>(nn, nn);
    FBF_TRY(gpu_status(gemmwich(blas, n, cm, h.data(), work, mo)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(h_mo.data(), mo, sizeof(double) * nn,
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
  }
  const auto download = [&](const double *from, const size_t len) -> Result<std::vector<double>> {
    std::vector<double> to(len);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to.data(), from, sizeof(double) * len,
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    return to;
  };
  const std::vector<double> tau_h = FBF_TRY(download(tau.data(), stack));
  const std::vector<double> hx_h = FBF_TRY(download(hx.data(), stack));
  const std::vector<double> fit_h = FBF_TRY(download(fit.data(), stack));
  const std::vector<double> eri_h = FBF_TRY(download(eri.data(), na4 * n_x));
  const std::vector<double> t1_h = FBF_TRY(download(t1.data(), na4 * n_x));

  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  const auto nd = static_cast<int32_t>(n_det);
  Buffers::device<double> scale(nroots, res.device());
  {
    std::vector<double> two_w(nroots);
    std::ranges::transform(weights, two_w.begin(), [](const double w) { return 2.0 * w; });
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(scale.data(), two_w.data(), sizeof(double) * nroots,
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(res.sync());
  }
  std::vector<double> h_eff(na2);
  std::vector<double> g_act(na4);
  std::vector<double> ham(nroots * nroots);
  const auto t1_at = [&](const size_t x, const size_t t, const size_t u, const size_t v,
                         const size_t w) {
    return t1_h[(x * na4) + t + (n_act * (u + (n_act * (v + (n_act * w)))))];
  };
  for (size_t x = 0; x < n_x; ++x) {
    const double *const f = fit_h.data() + (x * nn);
    const double *const tx = tau_h.data() + (x * nn);
    double e_core = 0.0;
    for (size_t i = 0; i < n_core; ++i) {
      double tau_h_ii = 0.0;
      for (size_t a = 0; a < nao; ++a) {
        tau_h_ii += tx[a + (nao * i)] * h_mo[a + (nao * i)];
      }
      e_core += hx_h[(x * nn) + i + (nao * i)] + (2.0 * tau_h_ii) + f[i + (nao * i)];
    }
    for (size_t t = 0; t < n_act; ++t) {
      for (size_t u = 0; u < n_act; ++u) {
        h_eff[(t * n_act) + u] =
            0.5 * (f[(n_core + t) + (nao * (n_core + u))] + f[(n_core + u) + (nao * (n_core + t))]);
      }
    }
    for (size_t w = 0; w < n_act; ++w) {
      for (size_t v = 0; v < n_act; ++v) {
        for (size_t u = 0; u < n_act; ++u) {
          for (size_t t = 0; t < n_act; ++t) {
            g_act[t + (n_act * (u + (n_act * (v + (n_act * w)))))] =
                eri_h[(x * na4) + t + (n_act * u) + (na2 * (v + (n_act * w)))] +
                t1_at(x, t, u, v, w) + t1_at(x, u, t, v, w) + t1_at(x, v, w, t, u) +
                t1_at(x, w, v, t, u);
          }
        }
      }
    }
    derivative_symmetrize8(g_act, n_act);
    FBF_TRY(sigma.set_integrals(res, e_core, h_eff, g_act));
    double *const y = out.rhs_ci.data() + (x * n_det * nroots);
    FBF_TRY(sigma.apply(res, nr, roots, y));
    auto [ham_d] = res.device_regions<double>(nroots * nroots);
    FBF_TRY(gpu_status(
        gemm<wwr::WWRBLAS_OP_T>(blas, nr, nr, nd, &kOne<double>, roots, y, &kZero<double>, ham_d)));
    FBF_TRY(gpu_status(gemm(blas, nd, nr, nr, &kNegOne<double>, roots, ham_d, &kOne<double>, y)));
    FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, nd, nr, y, nd,
                                                  scale.data(), 1, y, nd)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ham.data(), ham_d, sizeof(double) * nroots * nroots,
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    for (size_t s = 0; s < nroots; ++s) {
      for (size_t r = 0; r < nroots; ++r) {
        const double value = 0.5 * (ham[r + (nroots * s)] + ham[s + (nroots * r)]);
        const size_t at = r + (nroots * (s + (nroots * x)));
        out.hamiltonian[at] = value;
        // gamma_rs = 2 (w_s - w_r) H_rs: element (r, s) of the theta slab.
        out.rhs_theta[at] = 2.0 * (weights[s] - weights[r]) * value;
      }
    }
  }
  return out;
}

} // namespace free_breakfast::mcscf
