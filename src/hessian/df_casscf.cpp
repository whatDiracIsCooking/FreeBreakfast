/// @file
/// @brief df_casscf: the state-specific DF-CASSCF nuclear Hessian, E_xy
/// (df_casscf_exy over mcscf::df_derivatives' F^[x]) plus the response term,
/// lambda^x by mcscf::zvectors on df_derivatives' right-hand sides, assembled
/// as Lambda^T E_lambda. by two gemms; the solution checked by one product of
/// the unreduced operator; the sum symmetrized.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // axpy, nrm2, scal, wwrMemcpyAsync, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.active_space;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.response.fock;
import free_breakfast.response.hessian;
import free_breakfast.response.orbital_pairs;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.derivative;
import free_breakfast.mcscf.relaxed_hessian;
import free_breakfast.mcscf.zvector;

#include "diagnostics.h"

namespace free_breakfast::hessian {

// trace(gamma) is the active electron count to this.
constexpr double kCasscfElectronCountTolerance = 1e-6;
// A CP column whose unreduced residual, ||E_ll l^x + E_lx|| with the CI rows
// halved, is above this was not solved: at a saddle in the parameters the
// Schur complement's inner CI solves stop on negative curvature and leave it
// there. A converged solve sits near 1e-9 (the inner solves' 1e-8, relative).
constexpr double kCasscfCpResidualCeiling = 1e-6;

Result<CasscfTerms> df_casscf(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                              const std::span<const PrimitivePairClass> classes,
                              const BasisSet &aux, const SphericalMatrixAO &h,
                              const Buffers::device<double> &b, const double *l,
                              const mcscf::Solution &solution, const int32_t ncore,
                              const int32_t ncas, const double tolerance,
                              const int32_t max_iterations) {
  const SphericalMatrixAO &c = solution.c;
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  if (solution.roots.energies.size() != 1) {
    return err_config("hessian::df_casscf: a state-specific solution has one root, this one {}",
                      solution.roots.energies.size());
  }
  if (c.dev_idx() != res.dev_idx() || c.n() != nao || molecule.natoms() != n_atom ||
      aux.natoms() != n_atom || ncore < 0 || ncas < 1 ||
      int64_t{ncore} + int64_t{ncas} > static_cast<int64_t>(nao) ||
      b.num_elements() != naux * nao * nao) {
    return err_config("hessian::df_casscf: needs C nao square on device {}, a molecule and two "
                      "bases on the same atoms, ncore >= 0, ncas >= 1, ncore + ncas <= nao and B "
                      "naux nao^2 long (C {} square on device {}, {}, {} and {} atoms, ncore {}, "
                      "ncas {}, nao {}, naux {}, B {} doubles)",
                      res.dev_idx(), c.n(), c.dev_idx(), molecule.natoms(), n_atom, aux.natoms(),
                      ncore, ncas, nao, naux, b.num_elements());
  }
  const auto n_act = static_cast<size_t>(ncas);
  if (solution.gamma.size() != n_act * n_act ||
      solution.gamma2.size() != n_act * n_act * n_act * n_act) {
    return err_config("hessian::df_casscf: RDMs of {} and {} elements for ncas {}",
                      solution.gamma.size(), solution.gamma2.size(), ncas);
  }
  // The active electrons, for the CI strings: trace(gamma).
  double trace = 0.0;
  for (size_t t = 0; t < n_act; ++t) {
    trace += solution.gamma[t * (n_act + 1)];
  }
  const auto nelecas = static_cast<int32_t>(std::lround(trace));
  const int32_t two_s = molecule.multiplicity() - 1;
  if (std::abs(trace - nelecas) > kCasscfElectronCountTolerance || nelecas < two_s ||
      (nelecas - two_s) % 2 != 0 || nelecas - ((nelecas - two_s) / 2) > ncas) {
    return err_config("hessian::df_casscf: trace(gamma) {} is no electron count of multiplicity "
                      "{} in {} orbitals",
                      trace, molecule.multiplicity(), ncas);
  }
  const int32_t nbeta = (nelecas - two_s) / 2;
  const int32_t nalpha = nelecas - nbeta;
  const ScopedDeviceIndex on{res.dev_idx()};

  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  if (solution.roots.vectors.num_elements() != dim) {
    return err_config("hessian::df_casscf: a root of {} doubles over {} determinants",
                      solution.roots.vectors.num_elements(), dim);
  }
  const auto n_x = static_cast<size_t>(3) * n_atom;
  const auto n_x32 = static_cast<int32_t>(n_x);
  const size_t mm = nao * nao;
  const auto naux32 = static_cast<int32_t>(naux);
  const auto nao32 = static_cast<int32_t>(nao);
  const auto dim32 = static_cast<int32_t>(dim);
  const std::vector<double> weight{1.0};
  const auto download = [&res](const double *const from,
                               const size_t count) -> Result<std::vector<double>> {
    std::vector<double> to(count);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to.data(), from, count * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
    return to;
  };

  // F^[x] and the right-hand sides E_lambda x, then E_xy at the root's RDMs.
  mcscf::Derivatives derivatives =
      FBF_TRY(mcscf::df_derivatives(res, molecule, basis, classes, aux, h, b, l, c, ncore, ncas,
                                    nalpha, nbeta, solution.roots.vectors.data(), weight));
  CasscfExy exy;
  {
    Buffers::device<double> c_columns(mm, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c_columns.data(), c.data(), mm * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    exy = FBF_TRY(df_casscf_exy(res, molecule, basis, classes, aux, h, b, l, c_columns, ncore, ncas,
                                solution.gamma, solution.gamma2, derivatives.fock));
  }

  // H~ at the solution, as mcscf::df_gradient builds it for a Z-vector: the
  // sigma over H_0 (row-major host arrays need no reorder: h_eff is
  // symmetric, and (tu|vw) column-major is (wv|ut) row-major), the orbital
  // Hessian over the root's Fock, and the CI response.
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, naux32, c, ncore, ncas));
  const response::Fock fock =
      FBF_TRY(response::fock(res, molecule, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa,
                             solution.gamma, solution.gamma2));
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  FBF_TRY(sigma.set_integrals(
      res, ints.hamiltonian.e_core, FBF_TRY(download(ints.hamiltonian.h_eff.data(), n_act * n_act)),
      FBF_TRY(download(ints.hamiltonian.eri.data(), n_act * n_act * n_act * n_act))));
  response::Hessian hessian = FBF_TRY(
      response::Hessian::create(res, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa, fock));
  mcscf::ResponseIntegrals response_ints{nao,
                                         static_cast<size_t>(ncore),
                                         n_act,
                                         FBF_TRY(download(fock.fock_inactive.data(), mm)),
                                         FBF_TRY(download(ints.ppaa.data(), mm * n_act * n_act)),
                                         FBF_TRY(download(ints.papa.data(), mm * n_act * n_act))};
  mcscf::RelaxedHessian relaxed = FBF_TRY(mcscf::RelaxedHessian::create(
      res, hessian, sigma, rdm, alpha, beta, solution.roots, weight, std::move(response_ints)));

  // The right-hand sides in zvectors' coordinates: the orbital rows packed
  // (rhs_orbital holds G_pq at p > q, the order the packing reads), the CI
  // rows halved (2 Q H~^x c at weight 1, against its unscaled P H' c).
  const auto pairs = static_cast<size_t>(relaxed.size());
  const auto pairs32 = static_cast<int32_t>(pairs);
  Buffers::device<double> gradients(pairs * n_x, res.device());
  for (size_t x = 0; x < n_x; ++x) {
    FBF_TRY(response::pack_orbital_pairs(res, nao32, ncore, ncas,
                                         derivatives.rhs_orbital.data() + (x * mm),
                                         gradients.data() + (x * pairs)));
  }
  const size_t ci_count = dim * n_x;
  const auto ci_count32 = static_cast<int32_t>(ci_count);
  const wwr::wwrblasHandle_t blas = res.blas().get();
  Buffers::device<double> ci_rhs(ci_count, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ci_rhs.data(), derivatives.rhs_ci.data(),
                                         ci_count * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                         res.stream())));
  FBF_TRY(
      gpu_status(wwr::scal<double, int32_t>(blas, ci_count32, &kHalf<double>, ci_rhs.data(), 1)));
  mcscf::ZVectors cp =
      FBF_TRY(mcscf::zvectors(res, relaxed, hessian, solution.roots, n_x32, gradients.data(),
                              ci_rhs.data(), mcscf::CpSolver::Schur, tolerance, max_iterations));

  // The solution checked on the unreduced equations, which no inner solve
  // stands in for: r = H z_o + H_oc z_c + g, r_c = P (H' c + (H_0 - E) z_c)
  // + y (y is off the root already).
  std::vector<double> residuals(n_x);
  {
    Buffers::device<double> r(pairs * n_x, res.device());
    Buffers::device<double> r_c(ci_count, res.device());
    FBF_TRY(
        relaxed.apply_coupled(res, n_x32, cp.orbital.data(), cp.ci.data(), r.data(), r_c.data()));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, pairs32 * n_x32, &kOne<double>,
                                                  gradients.data(), 1, r.data(), 1)));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, ci_count32, &kOne<double>, ci_rhs.data(), 1,
                                                  r_c.data(), 1)));
    for (size_t x = 0; x < n_x; ++x) {
      double orbital = 0.0;
      double ci = 0.0;
      FBF_TRY(gpu_status(
          wwr::nrm2<double, int32_t>(blas, pairs32, r.data() + (x * pairs), 1, &orbital)));
      FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, dim32, r_c.data() + (x * dim), 1, &ci)));
      residuals[x] = std::hypot(orbital, ci);
    }
    const auto worst = std::ranges::max_element(residuals);
    if (*worst > kCasscfCpResidualCeiling) {
      return err_divergence("hessian::df_casscf: the CP-CASSCF solve left a residual of {:.3e} in "
                            "coordinate {} (above {:.0e}): the parameters' Hessian is not "
                            "positive there, as at a state-specific excited root (a saddle)",
                            *worst, std::distance(residuals.begin(), worst),
                            kCasscfCpResidualCeiling);
    }
  }

  // Lambda^T E_lambda.: element (x, y) is g^x . z_o^y + rhs_ci^x . z_c^y, the
  // CI rows the energy's own (2 Q H~^x c) against z_c, the response itself.
  std::vector<double> response;
  {
    Buffers::device<double> response_device(n_x * n_x, res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_x32, n_x32, pairs32, &kOne<double>,
                                               gradients.data(), cp.orbital.data(), &kZero<double>,
                                               response_device.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_x32, n_x32, dim32, &kOne<double>,
                                               derivatives.rhs_ci.data(), cp.ci.data(),
                                               &kOne<double>, response_device.data())));
    response = FBF_TRY(download(response_device.data(), n_x * n_x));
  }

  // Summed and symmetrized, the antisymmetric part measured as it is
  // discarded.
  std::vector<double> total(n_x * n_x);
  double asymmetry = 0.0;
  for (size_t y = 0; y < n_x; ++y) {
    for (size_t x = 0; x < n_x; ++x) {
      const double upper = exy.exy[x + (n_x * y)] + response[x + (n_x * y)];
      const double lower = exy.exy[y + (n_x * x)] + response[y + (n_x * x)];
      asymmetry = std::max(asymmetry, std::abs(upper - lower));
      total[x + (n_x * y)] = 0.5 * (upper + lower);
    }
  }
  return CasscfTerms{.hessian = std::move(total),
                     .asymmetry = asymmetry,
                     .exy = std::move(exy),
                     .response = std::move(response),
                     .cp = std::move(cp),
                     .residuals = std::move(residuals)};
}

} // namespace free_breakfast::hessian
