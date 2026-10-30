/// @file
/// @brief mcscf::df_gradient: the fitted integrals and generalized Fock at the
/// solution, the AO densities and -W (gemms and two gemmwiches), the shared
/// one-electron terms, then one Gam and one W built from J (gemvs, trsvs and
/// rank-one gemms), the core and core-active K and the active 2-RDM term
/// (b_mo, trsms, gemms and strided-batched gemms), against one
/// three_center_gradient and one metric_gradient.

module free_breakfast.mcscf.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kNegOne, kTwo, kNegTwo, kHalf, kNegHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemv, trsv, trsm, geam, gemmStridedBatched, wwrMemcpyAsync
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.active_space;
import free_breakfast.integrals.eri.density_fitting.gradient;
import free_breakfast.integrals.eri.density_fitting.mo;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.response.fock;
import free_breakfast.gradient;
import free_breakfast.mcscf.solve;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

Result<GradientTerms>
df_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const Solution &solution, const int32_t ncore, const int32_t ncas, const int32_t root) {
  const SphericalMatrixAO &c = solution.c;
  const auto nroots = static_cast<int32_t>(solution.roots.energies.size());
  if (nroots > 1) {
    return err_unsupported("mcscf::df_gradient: the solution averages {} roots; a state-averaged "
                           "gradient needs the orbital and CI multipliers, which this call does "
                           "not solve for",
                           nroots);
  }
  if (root < 0 || root >= nroots) {
    return err_config("mcscf::df_gradient: root {} of a solution with {} roots", root, nroots);
  }
  if (c.dev_idx() != res.dev_idx()) {
    return err_config("mcscf::df_gradient: C is on device {}, but the call passed device {}",
                      c.dev_idx(), res.dev_idx());
  }
  const int64_t nao = basis.nao_sphe();
  if (c.n() != nao) {
    return err_config("mcscf::df_gradient: C is {} square, but the basis has {} spherical AOs",
                      c.n(), nao);
  }
  if (ncore < 0 || ncas < 1 || ncore + ncas > nao) {
    return err_config("mcscf::df_gradient: ncore {} and ncas {} over {} orbitals", ncore, ncas,
                      nao);
  }
  const auto n_act = static_cast<size_t>(ncas);
  if (solution.gamma.size() != n_act * n_act ||
      solution.gamma2.size() != n_act * n_act * n_act * n_act) {
    return err_config("mcscf::df_gradient: RDMs of {} and {} elements for ncas {}",
                      solution.gamma.size(), solution.gamma2.size(), ncas);
  }
  const int64_t naux = aux.nao_sphe();
  if (naux < 1 || b.num_elements() != static_cast<size_t>(naux * nao * nao)) {
    return err_config("mcscf::df_gradient: B holds {} doubles, but naux nao^2 is {} x {}^2",
                      b.num_elements(), naux, nao);
  }
  if (naux * nao * nao > std::numeric_limits<int32_t>::max()) {
    return err_unsupported("mcscf::df_gradient: naux nao^2 = {} exceeds int32_t", naux * nao * nao);
  }
  const int32_t nocc = ncore + ncas;
  const ScopedDeviceIndex on{res.dev_idx()};

  // Gam, W, the two fitted densities, G, the core pairs c_P,ip and V, the
  // back-transform Y (core then active), and the active pairs and Z. Each is
  // held across a call that takes the workspace (b_mo, the derivative
  // calls), so each is the call's own. Past the device's memory an
  // allocation would abort (CreatePolicy), not return.
  const auto n_ao = static_cast<size_t>(nao);
  const auto n_aux = static_cast<size_t>(naux);
  const auto n_core = static_cast<size_t>(ncore);
  const auto n_occ = static_cast<size_t>(nocc);
  const size_t pairs = n_ao * n_ao;
  const size_t gamma_count = n_aux * pairs;
  const size_t w_count = n_aux * n_aux;
  const size_t core_count = n_aux * n_core * n_occ;
  const size_t y_count = n_aux * std::max(n_core, n_act) * n_ao;
  const size_t active_count = n_aux * n_act * n_act;
  const size_t g_count = n_occ * n_occ;
  const size_t bytes = sizeof(double) * (gamma_count + w_count + (2 * n_aux) + g_count +
                                         (2 * core_count) + y_count + (2 * active_count));
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (bytes > free_bytes) {
    return err_unsupported("mcscf::df_gradient: Gam over {} orbital and {} fitting functions and "
                           "its scratch need {} bytes of device memory, and device {} has {} "
                           "free",
                           nao, naux, bytes, res.dev_idx(), free_bytes);
  }

  // The fitted integrals and the generalized Fock at the solution's orbitals
  // and RDMs: F, and gamma and Gamma on the device, Gamma in (tu|vw)'s order.
  const CasscfIntegrals ints = FBF_TRY(
      casscf_integrals(res, basis, molecule, h, b, static_cast<int32_t>(naux), c, ncore, ncas));
  const response::Fock f =
      FBF_TRY(response::fock(res, molecule, h, b, static_cast<int32_t>(naux), c, ncore, ncas,
                             ints.ppaa, ints.papa, solution.gamma, solution.gamma2));

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(nao);
  const auto q = static_cast<int32_t>(naux);
  const auto nn = static_cast<int32_t>(pairs);
  const double *const c_act = c.data() + (n_ao * n_core);

  // D_a = C_a gamma C_a^T, D = 2 C_c C_c^T + D_a, and -W = -1/2 (C F C^T +
  // (C F C^T)^T): F is symmetric only to the solve's convergence, and S' is
  // symmetric, so its symmetric part is what -Sum W S' reads.
  SphericalMatrixAO d(res, basis);
  SphericalMatrixAO d_active(res, basis);
  SphericalMatrixAO minus_w(res, basis);
  {
    const auto [work, cfc] = res.device_regions<double>(pairs, pairs);
    FBF_TRY(gpu_status(
        gemmwich<wwr::WWRBLAS_OP_T>(blas, n, ncas, c_act, f.gamma.data(), work, d_active.data())));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d.data(), d_active.data(), sizeof(double) * pairs,
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    if (ncore > 0) {
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, n, n, ncore, &kTwo<double>, c.data(), c.data(), &kOne<double>, d.data())));
    }
    FBF_TRY(
        gpu_status(gemmwich<wwr::WWRBLAS_OP_T>(blas, n, n, c.data(), f.fock.data(), work, cfc)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                                  &kNegHalf<double>, cfc, n, &kNegHalf<double>, cfc,
                                                  n, minus_w.data(), n)));
  }

  gradient::RhfTerms one;
  FBF_TRY(gradient::one_electron_terms(res, molecule, basis, classes, d, minus_w, one));

  Buffers::device<double> gamma(gamma_count, res.device());
  Buffers::device<double> w(w_count, res.device());
  Buffers::device<double> fitted_density(2 * n_aux, res.device());
  double *const d_t = fitted_density.data();
  double *const d_a = d_t + n_aux;

  // Coulomb, E_J = 1/2 J[D].D - 1/2 J[D_a].D_a: the core-core half and the
  // core-active whole of it, and no active-active J (Gamma's). Each density
  // X fits to d_X = L^-T (B vec X); then Gam = d_t vec(D)^T - d_a vec(D_a)^T
  // at (P, n + nao m), and W = -1/2 d_t d_t^T + 1/2 d_a d_a^T.
  for (const auto &[density, fit] : {std::pair{d.data(), d_t}, std::pair{d_active.data(), d_a}}) {
    FBF_TRY(
        gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, nn, &kOne<double>,
                                              b.data(), q, density, 1, &kZero<double>, fit, 1)));
    FBF_TRY(
        gpu_status(wwr::trsv<double, int32_t>(blas, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
                                              wwr::WWRBLAS_DIAG_NON_UNIT, q, l, q, fit, 1)));
  }
  FBF_TRY(
      gpu_status(gemm(blas, q, nn, 1, &kOne<double>, d_t, d.data(), &kZero<double>, gamma.data())));
  FBF_TRY(gpu_status(
      gemm(blas, q, nn, 1, &kNegOne<double>, d_a, d_active.data(), &kOne<double>, gamma.data())));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      blas, q, q, 1, &kNegHalf<double>, d_t, d_t, &kZero<double>, w.data())));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, q, q, 1, &kHalf<double>, d_a,
                                                                d_a, &kOne<double>, w.data())));

  Buffers::device<double> y(y_count, res.device());
  Buffers::device<double> g_occ(g_count, res.device());
  Buffers::device<double> fitted_core(core_count, res.device());
  Buffers::device<double> v(core_count, res.device());
  Buffers::device<double> fitted_active(active_count, res.device());
  Buffers::device<double> z(active_count, res.device());
  // Exchange of the core with itself and with the active space,
  // E_K = -Sum_i Sum_pq G_pq (ip|iq), i core, p and q occupied, G =
  // diag(1_core, gamma): -Sum_ij (ij|ij) and -Sum_i,tu gamma_tu (it|iu).
  // Over the fitted pairs c_P,ip = (L^-T B_ip)_P, [P][i][p] at
  // P + naux (i + ncore p), and V_P,iq = Sum_p c_P,ip G_pq,
  // Gam += -2 Sum_i C_mi Sum_q V_P,iq C_nq and W += Sum_iq V_P,iq c_Q,iq.
  // The host G is read by its copy before the first derivative call waits
  // on the stream, and it outlives that call.
  std::vector<double> occupation(g_count, 0.0);
  if (ncore > 0) {
    for (size_t i = 0; i < n_core; ++i) {
      occupation[i * (n_occ + 1)] = 1.0;
    }
    for (size_t t = 0; t < n_act; ++t) {
      for (size_t u = 0; u < n_act; ++u) {
        occupation[(n_core + t) + (n_occ * (n_core + u))] = solution.gamma[(t * n_act) + u];
      }
    }
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(g_occ.data(), occupation.data(), sizeof(double) * g_count,
                                       wwr::wwrMemcpyHostToDevice, res.stream())));
    const OrbitalRange core{.first = 0, .count = ncore};
    const OrbitalRange occupied{.first = 0, .count = nocc};
    FBF_TRY(b_mo(res, q, n, n, b.data(), c.data(), core, occupied, fitted_core.data()));
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, ncore * nocc, &kOne<double>, l, q, fitted_core.data(), q)));
    const int32_t rows = q * ncore;
    FBF_TRY(gpu_status(gemm(blas, rows, nocc, nocc, &kOne<double>, fitted_core.data(), g_occ.data(),
                            &kZero<double>, v.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, ncore * nocc, &kOne<double>, v.data(), fitted_core.data(), &kOne<double>,
        w.data())));
    // Y_P,in = Sum_q V_P,iq C_nq at P + naux (i + ncore n); then for each n
    // the naux x nao block of Gam at P + naux (n + nao m), -2 Y_n C_core^T,
    // as gradient::df_rhf builds its exchange.
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, rows, n, nocc, &kOne<double>, v.data(), c.data(), &kZero<double>, y.data())));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, ncore, &kNegTwo<double>, y.data(), q,
        int64_t{rows}, c.data(), n, 0, &kOne<double>, gamma.data(), q * n, naux, n)));
  }

  // The active 2-RDM, E_aa = 1/2 Sum Gamma_tuvw (tu|vw): over the fitted
  // active pairs c_P,tu at P + naux (t + ncas u), Z = c Gamma (Gamma as the
  // ncas^2 square matrix it is in F's order), Gam += Sum_tu C_mt C_nu Z_P,tu
  // and W += -1/2 Z c^T.
  {
    const OrbitalRange active{.first = ncore, .count = ncas};
    FBF_TRY(b_mo(res, q, n, n, b.data(), c.data(), active, active, fitted_active.data()));
    const int32_t pairs_act = ncas * ncas;
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, pairs_act, &kOne<double>, l, q, fitted_active.data(), q)));
    FBF_TRY(gpu_status(gemm(blas, q, pairs_act, pairs_act, &kOne<double>, fitted_active.data(),
                            f.gamma2.data(), &kZero<double>, z.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, pairs_act, &kNegHalf<double>, z.data(), fitted_active.data(), &kOne<double>,
        w.data())));
    const int32_t rows = q * ncas;
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, rows, n, ncas, &kOne<double>, z.data(), c_act, &kZero<double>, y.data())));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, ncas, &kOne<double>, y.data(), q,
        int64_t{rows}, c_act, n, 0, &kOne<double>, gamma.data(), q * n, naux, n)));
  }

  GradientTerms g;
  g.two_electron = FBF_TRY(three_center_gradient(res, basis, classes, aux, gamma.data()));
  const std::vector<double> metric_part = FBF_TRY(metric_gradient(res, aux, w.data()));
  std::ranges::transform(g.two_electron, metric_part, g.two_electron.begin(), std::plus{});
  g.kinetic = std::move(one.kinetic);
  g.potential = std::move(one.potential);
  g.overlap = std::move(one.overlap);
  g.nuclear_repulsion = std::move(one.nuclear_repulsion);
  g.total = g.nuclear_repulsion;
  for (const std::vector<double> *term : {&g.kinetic, &g.potential, &g.overlap, &g.two_electron}) {
    std::ranges::transform(g.total, *term, g.total.begin(), std::plus{});
  }
  return g;
}

} // namespace free_breakfast::mcscf
