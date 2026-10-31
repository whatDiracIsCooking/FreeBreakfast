/// @file
/// @brief mcscf::df_gradient: the fitted integrals and generalized Fock at the
/// solution, the AO densities and -W (gemms and two gemmwiches), the shared
/// one-electron terms, then one Gam and one W built from J (gemvs, trsvs and
/// rank-one gemms), the core and core-active K and the active 2-RDM term
/// (b_mo, trsms, gemms and strided-batched gemms), against one
/// three_center_gradient and one metric_gradient. For one root of a state
/// average, the Z-vector first: its CI part enters as transition RDMs, its
/// orbital part as the same terms differentiated along C -> C (1 + kappa).

module free_breakfast.mcscf.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kNegOne, kTwo, kNegTwo, kHalf, kNegHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemv, trsv, trsm, geam, axpy, gemmStridedBatched, wwrMemcpyAsync
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.active_space;
import free_breakfast.integrals.eri.density_fitting.gradient;
import free_breakfast.integrals.eri.density_fitting.jk;
import free_breakfast.integrals.eri.density_fitting.mo;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.ci.solver;
import free_breakfast.response.fock;
import free_breakfast.response.hessian;
import free_breakfast.response.orbital_pairs;
import free_breakfast.gradient;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.relaxed_hessian;
import free_breakfast.mcscf.zvector;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// The weights are recovered from the solution's RDMs (sa_gradient_weights):
// a pivot of their least-squares system below this, relative to its largest
// element, means the roots' 1-RDMs do not determine them; and the weighted
// gamma must be their weighted sum to this, absolute.
constexpr double kWeightsPivotFloor = 1e-10;
constexpr double kWeightsResidual = 1e-10;
// trace(gamma) is the active electron count to this.
constexpr double kElectronCountTolerance = 1e-6;

// @p count doubles from the device into a host vector, waited for.
Result<std::vector<double>> gradient_download(DeviceResources &res, const double *const from,
                                              const size_t count) {
  std::vector<double> to(count);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to.data(), from, count * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return to;
}

// w_r of the average @p solution is over. A Solution carries the weighted
// gamma and each root's gamma but not the weights, so they are the
// least-squares solution of sum_r w_r gamma_r = gamma with sum_r w_r = 1: its
// (nroots + 1)^2 KKT system, by Gaussian elimination with partial pivoting.
Result<std::vector<double>> sa_gradient_weights(const Solution &solution, const size_t n_act) {
  const size_t nroots = solution.roots.energies.size();
  const size_t m = n_act * n_act;
  const size_t k = nroots + 1;
  const auto root_gamma = [&](const size_t r, const size_t i) {
    return solution.root_gammas[(r * m) + i];
  };
  // Row-major [A | rhs], A = [[G^T G, 1], [1^T, 0]].
  std::vector<double> a(k * (k + 1), 0.0);
  const auto at = [&](const size_t i, const size_t j) -> double & { return a[(i * (k + 1)) + j]; };
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = 0; s < nroots; ++s) {
      for (size_t i = 0; i < m; ++i) {
        at(r, s) += root_gamma(r, i) * root_gamma(s, i);
      }
    }
    for (size_t i = 0; i < m; ++i) {
      at(r, k) += root_gamma(r, i) * solution.gamma[i];
    }
    at(r, nroots) = 1.0;
    at(nroots, r) = 1.0;
  }
  at(nroots, k) = 1.0;
  double scale = 0.0;
  for (size_t i = 0; i < k; ++i) {
    for (size_t j = 0; j < k; ++j) {
      scale = std::max(scale, std::abs(at(i, j)));
    }
  }
  for (size_t col = 0; col < k; ++col) {
    size_t pivot = col;
    for (size_t i = col + 1; i < k; ++i) {
      if (std::abs(at(i, col)) > std::abs(at(pivot, col))) {
        pivot = i;
      }
    }
    if (std::abs(at(pivot, col)) < kWeightsPivotFloor * scale) {
      return err_unsupported("mcscf::df_gradient: the {} roots' 1-RDMs are linearly dependent, so "
                             "the solution's weights cannot be recovered from them",
                             nroots);
    }
    for (size_t j = 0; j <= k; ++j) {
      std::swap(at(col, j), at(pivot, j));
    }
    for (size_t i = col + 1; i < k; ++i) {
      const double factor = at(i, col) / at(col, col);
      for (size_t j = col; j <= k; ++j) {
        at(i, j) -= factor * at(col, j);
      }
    }
  }
  std::vector<double> x(k);
  for (size_t i = k; i-- > 0;) {
    double sum = at(i, k);
    for (size_t j = i + 1; j < k; ++j) {
      sum -= at(i, j) * x[j];
    }
    x[i] = sum / at(i, i);
  }
  std::vector<double> w(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(nroots));
  double residual = 0.0;
  for (size_t i = 0; i < m; ++i) {
    double sum = 0.0;
    for (size_t r = 0; r < nroots; ++r) {
      sum += w[r] * root_gamma(r, i);
    }
    residual = std::max(residual, std::abs(sum - solution.gamma[i]));
  }
  if (residual > kWeightsResidual ||
      std::ranges::any_of(w, [](const double v) { return v < -kWeightsResidual; })) {
    return err_config("mcscf::df_gradient: the weighted gamma is not a weighted average of the "
                      "roots' (residual {:.3e})",
                      residual);
  }
  for (double &v : w) {
    v = std::max(v, 0.0);
  }
  const double sum = std::accumulate(w.begin(), w.end(), 0.0);
  for (double &v : w) {
    v /= sum;
  }
  return w;
}

// Root @p root's RDMs plus z_c's transition RDMs (README "The state-averaged
// Lagrangian"), row-major as ci::Rdm::rdm12's, and kappa(z_o) into the
// nmo x nmo @p kappa: the multipliers @p given, or when it is null the
// Z-vector of root @p root over the CI-relaxed Hessian at the solution's
// orbitals and roots. @p averaged is the Fock at the solution's weighted
// RDMs, @p ints the fitted integrals at its orbitals.
Result<std::pair<std::vector<double>, std::vector<double>>> sa_gradient_lagrangian(
    DeviceResources &res, const Molecule &molecule, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, const int32_t naux, const Solution &solution,
    const int32_t ncore, const int32_t ncas, const int32_t nelecas, const CasscfIntegrals &ints,
    const response::Fock &averaged, const int32_t root, const std::span<const double> weights,
    const ZVector *const given, double *const kappa) {
  const SphericalMatrixAO &c = solution.c;
  const auto n = static_cast<size_t>(ncas);
  const auto mo = static_cast<size_t>(c.n());
  const size_t nroots = weights.size();
  const int32_t n_beta = (nelecas - (molecule.multiplicity() - 1)) / 2;
  const ci::DeviceStringSet alpha =
      FBF_TRY(ci::DeviceStringSet::create(res, ncas, nelecas - n_beta));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, n_beta));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  if (solution.roots.vectors.num_elements() != dim * nroots) {
    return err_config("mcscf::df_gradient: {} CI doubles for {} roots of {} determinants",
                      solution.roots.vectors.num_elements(), nroots, dim);
  }
  const size_t n_core = static_cast<size_t>(ncore);
  const size_t packed = (n_core * (mo - n_core)) + (n * (mo - n_core - n));
  if (given != nullptr &&
      (given->orbital.num_elements() != packed || given->ci.num_elements() != dim * nroots)) {
    return err_config("mcscf::df_gradient: multipliers of {} orbital and {} CI doubles, but the "
                      "solution has {} pairs and {} roots of {} determinants",
                      given->orbital.num_elements(), given->ci.num_elements(), packed, nroots, dim);
  }

  // Root I's RDMs.
  Buffers::device<double> rdms((n * n) + (n * n * n * n), res.device());
  double *const gamma_device = rdms.data();
  double *const gamma2_device = gamma_device + (n * n);
  std::vector<double> one_hot(nroots, 0.0);
  one_hot[static_cast<size_t>(root)] = 1.0;
  FBF_TRY(rdm.rdm12(res, one_hot, solution.roots.vectors.data(), gamma_device, gamma2_device));
  std::vector<double> gamma = FBF_TRY(gradient_download(res, gamma_device, n * n));
  std::vector<double> gamma2 = FBF_TRY(gradient_download(res, gamma2_device, n * n * n * n));

  // No multipliers given: the Z-vector over H_0 at the solution's orbitals
  // and g_I, root I's orbital gradient there. The sigma takes host arrays
  // row-major, which needs no reorder, as mcscf::solve sets them: h_eff is
  // symmetric, and (tu|vw) column-major is (wv|ut) row-major.
  std::optional<ZVector> solved;
  if (given == nullptr) {
    ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
    FBF_TRY(sigma.set_integrals(
        res, ints.hamiltonian.e_core,
        FBF_TRY(gradient_download(res, ints.hamiltonian.h_eff.data(), n * n)),
        FBF_TRY(gradient_download(res, ints.hamiltonian.eri.data(), n * n * n * n))));
    response::Hessian hessian = FBF_TRY(
        response::Hessian::create(res, h, b, naux, c, ncore, ncas, ints.ppaa, ints.papa, averaged));
    ResponseIntegrals response_ints{
        mo,
        n_core,
        n,
        FBF_TRY(gradient_download(res, averaged.fock_inactive.data(), mo * mo)),
        FBF_TRY(gradient_download(res, ints.ppaa.data(), mo * mo * n * n)),
        FBF_TRY(gradient_download(res, ints.papa.data(), mo * mo * n * n))};
    RelaxedHessian relaxed = FBF_TRY(RelaxedHessian::create(
        res, hessian, sigma, rdm, alpha, beta, solution.roots, weights, std::move(response_ints)));
    const response::Fock own = FBF_TRY(response::fock(res, molecule, h, b, naux, c, ncore, ncas,
                                                      ints.ppaa, ints.papa, gamma, gamma2));
    solved.emplace(FBF_TRY(zvector(res, relaxed, hessian, solution.roots, own.gradient.data())));
  }
  const ZVector &z = given != nullptr ? *given : *solved;

  // z_c's transition RDMs, sum_r w_r (<z_c,r|.|c_r> + <c_r|.|z_c,r>): half
  // the difference of the RDMs of c + z_c and c - z_c at the weights, the
  // polarization mcscf::solve takes them by. z_c is zvector's, unweighted.
  const size_t block = dim * nroots;
  const auto block32 = static_cast<int32_t>(block);
  const wwr::wwrblasHandle_t blas = res.blas().get();
  Buffers::device<double> shifted(2 * block, res.device());
  for (const auto &[to, sign] : {std::pair{shifted.data(), &kOne<double>},
                                 std::pair{shifted.data() + block, &kNegOne<double>}}) {
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(to, solution.roots.vectors.data(), block * sizeof(double),
                                       wwr::wwrMemcpyDeviceToDevice, res.stream())));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, block32, sign, z.ci.data(), 1, to, 1)));
  }
  for (const auto &[from, half] :
       {std::pair{shifted.data(), 0.5}, std::pair{shifted.data() + block, -0.5}}) {
    FBF_TRY(rdm.rdm12(res, weights, from, gamma_device, gamma2_device));
    const std::vector<double> part = FBF_TRY(gradient_download(res, gamma_device, n * n));
    const std::vector<double> part2 = FBF_TRY(gradient_download(res, gamma2_device, n * n * n * n));
    for (size_t i = 0; i < part.size(); ++i) {
      gamma[i] += half * part[i];
    }
    for (size_t i = 0; i < part2.size(); ++i) {
      gamma2[i] += half * part2[i];
    }
  }
  FBF_TRY(response::unpack_orbital_pairs(res, static_cast<int32_t>(mo), ncore, ncas,
                                         z.orbital.data(), kappa));
  return std::pair{std::move(gamma), std::move(gamma2)};
}

// The orbital response's one-electron densities and generalized Fock: along
// C -> C (1 + kappa) at the averaged RDMs, D_cL = 2 (C~_c C_c^T + C_c C~_c^T)
// into @p d_core, D_aL = C~_a gamma C_a^T + C_a gamma C~_a^T into
// @p d_active (C~ = C kappa, @p c_rot), and F_L into @p f_l: the change of
// F's inner orbitals, its outer index held (C C^T = S^-1 is stationary to
// first order), so W's response is C sym(F_L) C^T (README "The orbital
// response"). @p gamma2 is the averaged Gamma, row-major.
Status sa_gradient_orbital_response(DeviceResources &res, const Buffers::device<double> &b,
                                    const int32_t naux, const int32_t n, const int32_t ncore,
                                    const int32_t ncas, const double *const c,
                                    const double *const c_rot, const double *const kappa,
                                    const response::Fock &averaged, const CasscfIntegrals &ints,
                                    const std::span<const double> gamma2, double *const d_core,
                                    double *const d_active, double *const f_l) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n_ao = static_cast<size_t>(n);
  const auto n_act = static_cast<size_t>(ncas);
  const size_t pairs = n_ao * n_ao;
  const size_t act4 = n_act * n_act * n_act * n_act;
  const size_t act3 = n_act * n_act * n_act;
  const double *const c_act = c + (n_ao * static_cast<size_t>(ncore));
  const double *const c_rot_act = c_rot + (n_ao * static_cast<size_t>(ncore));
  const double *const kappa_act = kappa + (n_ao * static_cast<size_t>(ncore));
  double *const f_l_act = f_l + (n_ao * static_cast<size_t>(ncore));

  // Held across df_coulomb and df_exchange, which take the workspace, so the
  // call's own: J, K, a work matrix, V_cL and V_cL + V_aL in the AO basis
  // and the MO one, C~_a gamma, T (nmo x ncas), G1, G2, R1 and R2.
  Buffers::device<double> j(pairs, res.device());
  Buffers::device<double> k(pairs, res.device());
  Buffers::device<double> work(pairs, res.device());
  Buffers::device<double> v_core_ao(pairs, res.device());
  Buffers::device<double> v_sum_ao(pairs, res.device());
  Buffers::device<double> v_core(pairs, res.device());
  Buffers::device<double> v_sum(pairs, res.device());
  Buffers::device<double> rotated_gamma(n_ao * n_act, res.device());
  Buffers::device<double> t(n_ao * n_act, res.device());
  Buffers::device<double> g12(2 * act4, res.device());
  Buffers::device<double> r12(2 * n_ao * act3, res.device());

  // V_cL = J[D_cL] - 1/2 K[D_cL]: K[D_cL] = 2 (K[C_c C~_c^T] + its transpose),
  // K[L R^T]^T being K[R L^T].
  if (ncore > 0) {
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, n, n, ncore, &kTwo<double>, c_rot, c, &kZero<double>, work.data())));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                                  &kOne<double>, work.data(), n, &kOne<double>,
                                                  work.data(), n, d_core, n)));
    FBF_TRY(df_coulomb(res, n, naux, b.data(), d_core, j.data()));
    FBF_TRY(df_exchange(res, n, naux, ncore, b.data(), c, c_rot, k.data()));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                                  &kOne<double>, k.data(), n, &kOne<double>,
                                                  k.data(), n, work.data(), n)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n,
                                                  &kOne<double>, j.data(), n, &kNegOne<double>,
                                                  work.data(), n, v_core_ao.data(), n)));
  } else {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(d_core, 0, pairs * sizeof(double), res.stream())));
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(v_core_ao.data(), 0, pairs * sizeof(double), res.stream())));
  }

  // V_aL = J[D_aL] - 1/2 K[D_aL], K[D_aL] = K[C_a (C~_a gamma)^T] + its
  // transpose; then V_cL + V_aL.
  FBF_TRY(gpu_status(gemm(blas, n, ncas, ncas, &kOne<double>, c_rot_act, averaged.gamma.data(),
                          &kZero<double>, rotated_gamma.data())));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      blas, n, n, ncas, &kOne<double>, rotated_gamma.data(), c_act, &kZero<double>, work.data())));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                                &kOne<double>, work.data(), n, &kOne<double>,
                                                work.data(), n, d_active, n)));
  FBF_TRY(df_coulomb(res, n, naux, b.data(), d_active, j.data()));
  FBF_TRY(df_exchange(res, n, naux, ncas, b.data(), c_act, rotated_gamma.data(), k.data()));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                                &kOne<double>, k.data(), n, &kOne<double>, k.data(),
                                                n, work.data(), n)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n,
                                                &kOne<double>, j.data(), n, &kNegHalf<double>,
                                                work.data(), n, v_sum_ao.data(), n)));
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(pairs), &kOne<double>,
                                                v_core_ao.data(), 1, v_sum_ao.data(), 1)));
  FBF_TRY(gpu_status(gemmwich(blas, n, c, v_core_ao.data(), work.data(), v_core.data())));
  FBF_TRY(gpu_status(gemmwich(blas, n, c, v_sum_ao.data(), work.data(), v_sum.data())));

  // F_L = F kappa^T (the outer C of W's second index) + [(F^I + F^A) kappa +
  // V_cL + V_aL] gamma_c + [F^I kappa + V_cL] gamma_a + the active 2-RDM
  // term's inner rotations; gamma_c = 2 on the core.
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      blas, n, n, n, &kOne<double>, averaged.fock.data(), kappa, &kZero<double>, f_l)));
  if (ncore > 0) {
    // work = F^I + F^A.
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n,
                                                  &kOne<double>, averaged.fock_inactive.data(), n,
                                                  &kOne<double>, averaged.fock_active.data(), n,
                                                  work.data(), n)));
    FBF_TRY(
        gpu_status(gemm(blas, n, ncore, n, &kTwo<double>, work.data(), kappa, &kOne<double>, f_l)));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, n * ncore, &kTwo<double>, v_sum.data(), 1, f_l, 1)));
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(
      t.data(), v_core.data() + (n_ao * static_cast<size_t>(ncore)), n_ao * n_act * sizeof(double),
      wwr::wwrMemcpyDeviceToDevice, res.stream())));
  FBF_TRY(gpu_status(gemm(blas, n, ncas, n, &kOne<double>, averaged.fock_inactive.data(), kappa_act,
                          &kOne<double>, t.data())));
  FBF_TRY(gpu_status(gemm(blas, n, ncas, ncas, &kOne<double>, t.data(), averaged.gamma.data(),
                          &kOne<double>, f_l_act)));

  // The active 2-RDM term sum_uvw (pu|vw) Gamma_tuvw with u, v and w rotated:
  // sum_svw (ps|vw) R1_svw,t + sum_suw (pu|sw) R2_suw,t, R1 = kappa_act G1 and
  // R2 = kappa_act G2 over G1[u, (v, w, t)] = Gamma_tuvw and
  // G2[v, (u, w, t)] = Gamma_tuvw + Gamma_tuwv, so that each R lies as ppaa's
  // and papa's columns (s, ., .) do. G1 and G2 are permuted on the host.
  std::vector<double> g12_host(2 * act4);
  for (size_t tt = 0; tt < n_act; ++tt) {
    for (size_t u = 0; u < n_act; ++u) {
      for (size_t v = 0; v < n_act; ++v) {
        for (size_t w = 0; w < n_act; ++w) {
          const double tuvw = gamma2[(((((tt * n_act) + u) * n_act) + v) * n_act) + w];
          const double tuwv = gamma2[(((((tt * n_act) + u) * n_act) + w) * n_act) + v];
          g12_host[u + (n_act * (v + (n_act * (w + (n_act * tt)))))] = tuvw;
          g12_host[act4 + v + (n_act * (u + (n_act * (w + (n_act * tt)))))] = tuvw + tuwv;
        }
      }
    }
  }
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(g12.data(), g12_host.data(), g12_host.size() * sizeof(double),
                                     wwr::wwrMemcpyHostToDevice, res.stream())));
  const auto cubed = static_cast<int32_t>(act3);
  const int32_t columns = n * ncas * ncas;
  for (const auto &[ints_block, which] :
       {std::pair{ints.ppaa.data(), size_t{0}}, std::pair{ints.papa.data(), size_t{1}}}) {
    double *const r = r12.data() + (which * n_ao * act3);
    FBF_TRY(gpu_status(gemm(blas, n, cubed, ncas, &kOne<double>, kappa_act,
                            g12.data() + (which * act4), &kZero<double>, r)));
    FBF_TRY(gpu_status(
        gemm(blas, n, ncas, columns, &kOne<double>, ints_block, r, &kOne<double>, f_l_act)));
  }
  // The host G1 and G2 outlive their upload.
  return res.sync();
}

Result<GradientTerms>
df_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const Solution &solution, const int32_t ncore, const int32_t ncas, const int32_t root,
            const ZVector *const multipliers, const MoAdjoint *const extra) {
  const SphericalMatrixAO &c = solution.c;
  const auto nroots = static_cast<int32_t>(solution.roots.energies.size());
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
  // A state average, or caller-supplied multipliers, takes the terms at the
  // Lagrangian and adds the orbital response; one root without them is the
  // solution's own energy.
  const bool averaged = nroots > 1;
  const bool responds = averaged || multipliers != nullptr;
  if (extra != nullptr &&
      (extra->h_bar == nullptr || extra->lambda == nullptr || extra->x == nullptr)) {
    return err_config("mcscf::df_gradient: an extra adjoint with a null h_bar, lambda or x");
  }
  if (averaged && solution.root_gammas.size() != static_cast<size_t>(nroots) * n_act * n_act) {
    return err_config("mcscf::df_gradient: {} root gamma elements for {} roots and ncas {}",
                      solution.root_gammas.size(), nroots, ncas);
  }
  const int64_t naux = aux.nao_sphe();
  if (naux < 1 || b.num_elements() != static_cast<size_t>(naux * nao * nao)) {
    return err_config("mcscf::df_gradient: B holds {} doubles, but naux nao^2 is {} x {}^2",
                      b.num_elements(), naux, nao);
  }
  if (naux * nao * nao > std::numeric_limits<int32_t>::max()) {
    return err_unsupported("mcscf::df_gradient: naux nao^2 = {} exceeds int32_t", naux * nao * nao);
  }
  // The active electrons, for the CI strings of the Z-vector: trace(gamma).
  double trace = 0.0;
  for (size_t t = 0; t < n_act; ++t) {
    trace += solution.gamma[t * (n_act + 1)];
  }
  const auto nelecas = static_cast<int32_t>(std::lround(trace));
  const int32_t two_s = molecule.multiplicity() - 1;
  if (responds && (std::abs(trace - nelecas) > kElectronCountTolerance || nelecas < two_s ||
                   (nelecas - two_s) % 2 != 0 || nelecas - ((nelecas - two_s) / 2) > ncas)) {
    return err_config("mcscf::df_gradient: trace(gamma) {} is no electron count of multiplicity "
                      "{} in {} orbitals",
                      trace, molecule.multiplicity(), ncas);
  }
  const int32_t nocc = ncore + ncas;
  const ScopedDeviceIndex on{res.dev_idx()};

  // Gam, W, the fitted densities, G, the core pairs c_P,ip and V, the
  // back-transform Y (core then active), and the active pairs and Z; for a
  // state average also the response's fitted densities, core pairs c'_P,ip,
  // their scratch and V at the averaged RDMs, and the same three of the
  // active pairs; for an extra adjoint, Lambda C^T and G^aux. Each is held
  // across a call that takes the workspace (b_mo, the derivative calls), so
  // each is the call's own. Past the device's memory an allocation would
  // abort (CreatePolicy), not return.
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
  const size_t fit_count = (responds ? 6 : 2) * n_aux;
  const size_t pieces = responds ? 6 : 2;
  const size_t bytes =
      sizeof(double) *
      (gamma_count + w_count + fit_count + (responds ? 2 : 1) * g_count + (pieces * core_count) +
       y_count + (pieces * active_count) + (extra != nullptr ? gamma_count + w_count : 0));
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
  // and (weighted) RDMs.
  const CasscfIntegrals ints = FBF_TRY(
      casscf_integrals(res, basis, molecule, h, b, static_cast<int32_t>(naux), c, ncore, ncas));
  response::Fock solution_fock =
      FBF_TRY(response::fock(res, molecule, h, b, static_cast<int32_t>(naux), c, ncore, ncas,
                             ints.ppaa, ints.papa, solution.gamma, solution.gamma2));

  // For a state average or given multipliers, the Lagrangian: the RDMs the
  // terms below take are root I's plus z_c's transition RDMs, F is their
  // generalized Fock, and kappa(z_o) drives the orbital response; z is the
  // caller's, or else the Z-vector. For one root without multipliers they are
  // the solution's own, and there is no response.
  std::vector<double> gamma = solution.gamma;
  std::vector<double> gamma2 = solution.gamma2;
  Buffers::device<double> kappa(responds ? pairs : 1, res.device());
  std::optional<response::Fock> lagrangian_fock;
  if (responds) {
    std::vector<double> weights{1.0};
    if (averaged) {
      weights = FBF_TRY(sa_gradient_weights(solution, n_act));
    }
    std::tie(gamma, gamma2) = FBF_TRY(sa_gradient_lagrangian(
        res, molecule, h, b, static_cast<int32_t>(naux), solution, ncore, ncas, nelecas, ints,
        solution_fock, root, weights, multipliers, kappa.data()));
    lagrangian_fock.emplace(
        FBF_TRY(response::fock(res, molecule, h, b, static_cast<int32_t>(naux), c, ncore, ncas,
                               ints.ppaa, ints.papa, gamma, gamma2)));
  }
  const response::Fock &f = responds ? *lagrangian_fock : solution_fock;

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(nao);
  const auto q = static_cast<int32_t>(naux);
  const auto nn = static_cast<int32_t>(pairs);
  const double *const c_act = c.data() + (n_ao * n_core);

  // The orbital response's C~ = C kappa, beside C so that b_mo transforms
  // over either ([C | C~], 2 nmo columns); its D_cL and D_aL, their sum D_L;
  // the averaged D and D_a it is the derivative of; and F_L.
  Buffers::device<double> c_both(responds ? 2 * pairs : 1, res.device());
  const double *const c_rot = responds ? c_both.data() + pairs : nullptr;
  SphericalMatrixAO d_core_l(res, basis);
  SphericalMatrixAO d_active_l(res, basis);
  SphericalMatrixAO d_averaged(res, basis);
  SphericalMatrixAO d_active_averaged(res, basis);
  SphericalMatrixAO f_l(res, basis);
  if (responds) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c_both.data(), c.data(), pairs * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, c.data(), kappa.data(), &kZero<double>,
                            c_both.data() + pairs)));
    FBF_TRY(sa_gradient_orbital_response(res, b, q, n, ncore, ncas, c.data(), c_rot, kappa.data(),
                                         solution_fock, ints, solution.gamma2, d_core_l.data(),
                                         d_active_l.data(), f_l.data()));
    // d_core_l <- D_L = D_cL + D_aL.
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, nn, &kOne<double>, d_active_l.data(), 1,
                                                  d_core_l.data(), 1)));
  }
  const double *const d_l = d_core_l.data();

  // D_a = C_a gamma C_a^T, D = 2 C_c C_c^T + D_a, and -W = -1/2 (C F C^T +
  // (C F C^T)^T): F is symmetric only to the solve's convergence, and S' is
  // symmetric, so its symmetric part is what -Sum W S' reads. For a state
  // average F is F + F_L, the one-electron D is D + D_L, and the averaged D
  // and D_a are built too, for J's response. An extra adjoint adds
  // P = C h_bar C^T to the one-electron D and 1/2 X to F, so W gains
  // 1/2 C sym(X) C^T.
  SphericalMatrixAO d(res, basis);
  SphericalMatrixAO d_active(res, basis);
  SphericalMatrixAO d_one(res, basis);
  SphericalMatrixAO minus_w(res, basis);
  {
    const auto [work, cfc] = res.device_regions<double>(pairs, pairs);
    std::vector<std::pair<const double *, std::pair<double *, double *>>> densities{
        {f.gamma.data(), {d.data(), d_active.data()}}};
    if (responds) {
      densities.push_back(
          {solution_fock.gamma.data(), {d_averaged.data(), d_active_averaged.data()}});
    }
    for (const auto &[gamma_device, out] : densities) {
      FBF_TRY(gpu_status(
          gemmwich<wwr::WWRBLAS_OP_T>(blas, n, ncas, c_act, gamma_device, work, out.second)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.first, out.second, sizeof(double) * pairs,
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
      if (ncore > 0) {
        FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
            blas, n, n, ncore, &kTwo<double>, c.data(), c.data(), &kOne<double>, out.first)));
      }
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_one.data(), d.data(), sizeof(double) * pairs,
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    const double *fock_total = f.fock.data();
    if (responds) {
      FBF_TRY(
          gpu_status(wwr::axpy<double, int32_t>(blas, nn, &kOne<double>, d_l, 1, d_one.data(), 1)));
      // f_l <- F + F_L.
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, nn, &kOne<double>, f.fock.data(), 1, f_l.data(), 1)));
      fock_total = f_l.data();
    }
    if (extra != nullptr) {
      FBF_TRY(
          gpu_status(gemmwich<wwr::WWRBLAS_OP_T>(blas, n, n, c.data(), extra->h_bar, work, cfc)));
      FBF_TRY(
          gpu_status(wwr::axpy<double, int32_t>(blas, nn, &kOne<double>, cfc, 1, d_one.data(), 1)));
      if (!responds) {
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(f_l.data(), f.fock.data(), sizeof(double) * pairs,
                                               wwr::wwrMemcpyDeviceToDevice, res.stream())));
      }
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, nn, &kHalf<double>, extra->x, 1, f_l.data(), 1)));
      fock_total = f_l.data();
    }
    FBF_TRY(gpu_status(gemmwich<wwr::WWRBLAS_OP_T>(blas, n, n, c.data(), fock_total, work, cfc)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                                  &kNegHalf<double>, cfc, n, &kNegHalf<double>, cfc,
                                                  n, minus_w.data(), n)));
  }

  gradient::RhfTerms one;
  FBF_TRY(gradient::one_electron_terms(res, molecule, basis, classes, d_one, minus_w, one));

  Buffers::device<double> gamma_mnp(gamma_count, res.device());
  Buffers::device<double> w(w_count, res.device());
  Buffers::device<double> fitted_density(fit_count, res.device());
  double *const d_t = fitted_density.data();
  double *const d_a = d_t + n_aux;

  // Coulomb, E_J = 1/2 J[D].D - 1/2 J[D_a].D_a: the core-core half and the
  // core-active whole of it, and no active-active J (Gamma's). Each density
  // X fits to d_X = L^-T (B vec X); then Gam = d_t vec(D)^T - d_a vec(D_a)^T
  // at (P, n + nao m), and W = -1/2 d_t d_t^T + 1/2 d_a d_a^T. Its response
  // to the orbitals at the averaged D and D_a, by the product rule, adds
  // d vec(D_L)^T + d_L vec(D)^T - d_a vec(D_aL)^T - d_aL vec(D_a)^T to Gam
  // and -1/2 (d_L d^T + d d_L^T) + 1/2 (d_aL d_a^T + d_a d_aL^T) to W.
  double *const d_sa = d_a + n_aux;
  double *const d_asa = d_sa + n_aux;
  double *const d_lf = d_asa + n_aux;
  double *const d_alf = d_lf + n_aux;
  std::vector<std::pair<const double *, double *>> fits{{d.data(), d_t}, {d_active.data(), d_a}};
  if (responds) {
    fits.insert(fits.end(), {{d_averaged.data(), d_sa},
                             {d_active_averaged.data(), d_asa},
                             {d_l, d_lf},
                             {d_active_l.data(), d_alf}});
  }
  for (const auto &[density, fit] : fits) {
    FBF_TRY(
        gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, nn, &kOne<double>,
                                              b.data(), q, density, 1, &kZero<double>, fit, 1)));
    FBF_TRY(
        gpu_status(wwr::trsv<double, int32_t>(blas, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
                                              wwr::WWRBLAS_DIAG_NON_UNIT, q, l, q, fit, 1)));
  }
  std::vector<std::tuple<const double *, const double *, const double *>> gam_terms{
      {d_t, d.data(), &kOne<double>}, {d_a, d_active.data(), &kNegOne<double>}};
  std::vector<std::tuple<const double *, const double *, const double *>> w_terms{
      {d_t, d_t, &kNegHalf<double>}, {d_a, d_a, &kHalf<double>}};
  if (responds) {
    gam_terms.insert(gam_terms.end(), {{d_sa, d_l, &kOne<double>},
                                       {d_lf, d_averaged.data(), &kOne<double>},
                                       {d_asa, d_active_l.data(), &kNegOne<double>},
                                       {d_alf, d_active_averaged.data(), &kNegOne<double>}});
    w_terms.insert(w_terms.end(), {{d_lf, d_sa, &kNegHalf<double>},
                                   {d_sa, d_lf, &kNegHalf<double>},
                                   {d_alf, d_asa, &kHalf<double>},
                                   {d_asa, d_alf, &kHalf<double>}});
  }
  FBF_TRY(gpu_status(
      wwr::wwrMemsetAsync(gamma_mnp.data(), 0, gamma_count * sizeof(double), res.stream())));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(w.data(), 0, w_count * sizeof(double), res.stream())));
  // An extra adjoint first, into the zeroed Gam: Lambda_AO = C Lambda^L C^T
  // (a gemm over q and a strided-batched gemm over n, df_rhf's exchange
  // shape), G^aux = Lambda_AO B^T (Sum_pq Lambda^L_pq B^L'_pq, in the AO
  // basis), W += -1/2 L^-T G^aux L^-1 (two trsms), and Gam = L^-T Lambda_AO
  // (one trsm in place) before any other term is added to it.
  Buffers::device<double> half(extra != nullptr ? gamma_count : 1, res.device());
  Buffers::device<double> g_aux(extra != nullptr ? w_count : 1, res.device());
  if (extra != nullptr) {
    const int32_t rows = q * n;
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, rows, n, n, &kOne<double>, extra->lambda, c.data(), &kZero<double>, half.data())));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, n, &kOne<double>, half.data(), q,
        int64_t{rows}, c.data(), n, 0, &kZero<double>, gamma_mnp.data(), q * n, naux, n)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, nn, &kOne<double>, gamma_mnp.data(), b.data(), &kZero<double>, g_aux.data())));
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, q, &kOne<double>, l, q, g_aux.data(), q)));
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_N,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, q, &kOne<double>, l, q, g_aux.data(), q)));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, q * q, &kNegHalf<double>, g_aux.data(), 1, w.data(), 1)));
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, nn, &kOne<double>, l, q, gamma_mnp.data(), q)));
  }
  for (const auto &[fit, density, scale] : gam_terms) {
    FBF_TRY(gpu_status(gemm(blas, q, nn, 1, scale, fit, density, &kOne<double>, gamma_mnp.data())));
  }
  for (const auto &[left, right, scale] : w_terms) {
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, q, q, 1, scale, left, right,
                                                                  &kOne<double>, w.data())));
  }

  Buffers::device<double> y(y_count, res.device());
  Buffers::device<double> g_occ((responds ? 2 : 1) * g_count, res.device());
  Buffers::device<double> fitted_core(pieces / 2 * core_count, res.device());
  Buffers::device<double> v(pieces / 2 * core_count, res.device());
  Buffers::device<double> fitted_active(pieces / 2 * active_count, res.device());
  Buffers::device<double> z(pieces / 2 * active_count, res.device());
  // For a state average, the second and third of each: the pairs' response
  // c'_P,pq = c_P,p~q + c_P,pq~ (p~ = C~'s column p) and its scratch, and V
  // (or Z) at the averaged RDMs.
  double *const fitted_core_l = responds ? fitted_core.data() + core_count : nullptr;
  double *const core_scratch = responds ? fitted_core_l + core_count : nullptr;
  double *const v_averaged = responds ? v.data() + core_count : nullptr;
  double *const fitted_active_l = responds ? fitted_active.data() + active_count : nullptr;
  double *const active_scratch = responds ? fitted_active_l + active_count : nullptr;
  double *const z_averaged = responds ? z.data() + active_count : nullptr;
  const auto mo2 = static_cast<int32_t>(2 * nao);
  // c'_P over [C | C~]: b_mo of the rotated rows plus b_mo of the rotated
  // columns, then L^-T.
  const auto rotated_pairs = [&](const OrbitalRange rows, const OrbitalRange cols,
                                 double *const out, double *const scratch) -> Status {
    FBF_TRY(b_mo(res, q, n, mo2, b.data(), c_both.data(),
                 {.first = n + rows.first, .count = rows.count}, cols, out));
    FBF_TRY(b_mo(res, q, n, mo2, b.data(), c_both.data(), rows,
                 {.first = n + cols.first, .count = cols.count}, scratch));
    const int32_t count = q * rows.count * cols.count;
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, count, &kOne<double>, scratch, 1, out, 1)));
    return gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, rows.count * cols.count, &kOne<double>, l, q, out, q));
  };
  // Exchange of the core with itself and with the active space,
  // E_K = -Sum_i Sum_pq G_pq (ip|iq), i core, p and q occupied, G =
  // diag(1_core, gamma): -Sum_ij (ij|ij) and -Sum_i,tu gamma_tu (it|iu).
  // Over the fitted pairs c_P,ip = (L^-T B_ip)_P, [P][i][p] at
  // P + naux (i + ncore p), and V_P,iq = Sum_p c_P,ip G_pq,
  // Gam += -2 Sum_i C_mi Sum_q V_P,iq C_nq and W += Sum_iq V_P,iq c_Q,iq.
  // The orbital response, every C of those in turn C~ at the averaged G_a:
  // V += c' G_a, W += V_a c'^T, Gam += -2 Sum_i (C_mi Sum_q V_a,iq C~_nq +
  // C~_mi Sum_q V_a,iq C_nq). The host G is read by its copy before the first
  // derivative call waits on the stream, and it outlives that call.
  std::vector<double> occupation((responds ? 2 : 1) * g_count, 0.0);
  if (ncore > 0) {
    // G of the terms' RDMs, then (for a state average) of the averaged ones.
    const auto fill = [&](const std::vector<double> &from, const size_t offset) {
      for (size_t i = 0; i < n_core; ++i) {
        occupation[offset + (i * (n_occ + 1))] = 1.0;
      }
      for (size_t t = 0; t < n_act; ++t) {
        for (size_t u = 0; u < n_act; ++u) {
          occupation[offset + (n_core + t) + (n_occ * (n_core + u))] = from[(t * n_act) + u];
        }
      }
    };
    fill(gamma, 0);
    if (responds) {
      fill(solution.gamma, g_count);
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(g_occ.data(), occupation.data(),
                                           sizeof(double) * occupation.size(),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    const double *const g_averaged = responds ? g_occ.data() + g_count : nullptr;
    const OrbitalRange core{.first = 0, .count = ncore};
    const OrbitalRange occupied{.first = 0, .count = nocc};
    FBF_TRY(b_mo(res, q, n, n, b.data(), c.data(), core, occupied, fitted_core.data()));
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, ncore * nocc, &kOne<double>, l, q, fitted_core.data(), q)));
    const int32_t rows = q * ncore;
    FBF_TRY(gpu_status(gemm(blas, rows, nocc, nocc, &kOne<double>, fitted_core.data(), g_occ.data(),
                            &kZero<double>, v.data())));
    if (responds) {
      FBF_TRY(rotated_pairs(core, occupied, fitted_core_l, core_scratch));
      FBF_TRY(gpu_status(gemm(blas, rows, nocc, nocc, &kOne<double>, fitted_core_l, g_averaged,
                              &kOne<double>, v.data())));
      FBF_TRY(gpu_status(gemm(blas, rows, nocc, nocc, &kOne<double>, fitted_core.data(), g_averaged,
                              &kZero<double>, v_averaged)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, q, q, ncore * nocc, &kOne<double>, v_averaged, fitted_core_l, &kOne<double>,
          w.data())));
    }
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, ncore * nocc, &kOne<double>, v.data(), fitted_core.data(), &kOne<double>,
        w.data())));
    // Y_P,in = Sum_q V_P,iq C_nq at P + naux (i + ncore n); then for each n
    // the naux x nao block of Gam at P + naux (n + nao m), -2 Y_n C_core^T,
    // as gradient::df_rhf builds its exchange.
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, rows, n, nocc, &kOne<double>, v.data(), c.data(), &kZero<double>, y.data())));
    if (responds) {
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, rows, n, nocc, &kOne<double>, v_averaged, c_rot, &kOne<double>, y.data())));
    }
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, ncore, &kNegTwo<double>, y.data(), q,
        int64_t{rows}, c.data(), n, 0, &kOne<double>, gamma_mnp.data(), q * n, naux, n)));
    if (responds) {
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, rows, n, nocc, &kOne<double>, v_averaged, c.data(), &kZero<double>, y.data())));
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, ncore, &kNegTwo<double>, y.data(), q,
          int64_t{rows}, c_rot, n, 0, &kOne<double>, gamma_mnp.data(), q * n, naux, n)));
    }
  }

  // The active 2-RDM, E_aa = 1/2 Sum Gamma_tuvw (tu|vw): over the fitted
  // active pairs c_P,tu at P + naux (t + ncas u), Z = c Gamma (Gamma as the
  // ncas^2 square matrix it is in F's order), Gam += Sum_tu C_mt C_nu Z_P,tu
  // and W += -1/2 Z c^T. The orbital response, at the averaged Gamma_a:
  // Z += c' Gamma_a, W += -1/2 Z_a c'^T, Gam += Sum_tu (C_mt C~_nu +
  // C~_mt C_nu) Z_a,P,tu.
  {
    const OrbitalRange active{.first = ncore, .count = ncas};
    FBF_TRY(b_mo(res, q, n, n, b.data(), c.data(), active, active, fitted_active.data()));
    const int32_t pairs_act = ncas * ncas;
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, pairs_act, &kOne<double>, l, q, fitted_active.data(), q)));
    FBF_TRY(gpu_status(gemm(blas, q, pairs_act, pairs_act, &kOne<double>, fitted_active.data(),
                            f.gamma2.data(), &kZero<double>, z.data())));
    if (responds) {
      FBF_TRY(rotated_pairs(active, active, fitted_active_l, active_scratch));
      FBF_TRY(gpu_status(gemm(blas, q, pairs_act, pairs_act, &kOne<double>, fitted_active_l,
                              solution_fock.gamma2.data(), &kOne<double>, z.data())));
      FBF_TRY(gpu_status(gemm(blas, q, pairs_act, pairs_act, &kOne<double>, fitted_active.data(),
                              solution_fock.gamma2.data(), &kZero<double>, z_averaged)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, q, q, pairs_act, &kNegHalf<double>, z_averaged, fitted_active_l, &kOne<double>,
          w.data())));
    }
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, pairs_act, &kNegHalf<double>, z.data(), fitted_active.data(), &kOne<double>,
        w.data())));
    const int32_t rows = q * ncas;
    const double *const c_rot_act = responds ? c_rot + (n_ao * n_core) : nullptr;
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, rows, n, ncas, &kOne<double>, z.data(), c_act, &kZero<double>, y.data())));
    if (responds) {
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, rows, n, ncas, &kOne<double>, z_averaged, c_rot_act, &kOne<double>, y.data())));
    }
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, ncas, &kOne<double>, y.data(), q,
        int64_t{rows}, c_act, n, 0, &kOne<double>, gamma_mnp.data(), q * n, naux, n)));
    if (responds) {
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, rows, n, ncas, &kOne<double>, z_averaged, c_act, &kZero<double>, y.data())));
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, ncas, &kOne<double>, y.data(), q,
          int64_t{rows}, c_rot_act, n, 0, &kOne<double>, gamma_mnp.data(), q * n, naux, n)));
    }
  }

  GradientTerms g;
  g.two_electron = FBF_TRY(three_center_gradient(res, basis, classes, aux, gamma_mnp.data()));
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
