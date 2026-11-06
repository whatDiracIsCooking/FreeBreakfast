/// @file
/// @brief df_casscf: the state-specific DF-CASSCF nuclear Hessian, E_xy
/// (df_casscf_exy over mcscf::df_derivatives' F^[x]) plus the response term,
/// lambda^x by mcscf::zvectors on df_derivatives' right-hand sides, assembled
/// as Lambda^T E_lambda. by two gemms; the solution checked by one product of
/// the unreduced operator; the sum symmetrized. The CP step on its own,
/// df_casscf_cp, and state_specific_electrons, the inputs' check
/// df_nevpt2_tangents shares.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // axpy, dot, nrm2, scal, wwrMemcpyAsync, ScopedDeviceIndex, gpu_status
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
import :state_specific; // state_specific_electrons

#include "diagnostics.h"

namespace free_breakfast::hessian {

// trace(gamma) is the active electron count to this.
constexpr double kCasscfElectronCountTolerance = 1e-6;
// A CP column whose unreduced residual, ||E_ll l^x + E_lx|| with the CI rows
// halved, is above this was not solved: at a saddle in the parameters the
// Schur complement's inner CI solves stop on negative curvature and leave it
// there. A converged solve sits near 1e-9 (the inner solves' 1e-8, relative).
constexpr double kCasscfCpResidualCeiling = 1e-6;
// The probe column's seed: a fixed pseudo-random right-hand side, so that a
// refusal repeats run to run (README.md "A singular H~").
constexpr uint64_t kCasscfCpProbeSeed = 978;

// The probe's right-hand side: @p pairs pseudo-random doubles in [-1, 1) from
// a fixed seed, by the generator's bits (no distribution, whose output the
// standard leaves to the library), normalized.
std::vector<double> probe_column(const size_t pairs) {
  std::mt19937_64 bits(kCasscfCpProbeSeed);
  std::vector<double> s(pairs);
  double norm = 0.0;
  for (double &v : s) {
    v = (2.0 * std::ldexp(static_cast<double>(bits() >> 11), -53)) - 1.0;
    norm += v * v;
  }
  norm = std::sqrt(norm);
  for (double &v : s) {
    v /= norm;
  }
  return s;
}

Result<std::pair<int32_t, int32_t>>
state_specific_electrons(const std::string_view who, const DeviceResources &res,
                         const Molecule &molecule, const BasisSet &basis, const BasisSet &aux,
                         const Buffers::device<double> &b, const mcscf::Solution &solution,
                         const int32_t ncore, const int32_t ncas) {
  const SphericalMatrixAO &c = solution.c;
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  if (solution.roots.energies.size() != 1) {
    return err_config("{}: a state-specific solution has one root, this one {}", who,
                      solution.roots.energies.size());
  }
  if (c.dev_idx() != res.dev_idx() || c.n() != nao || molecule.natoms() != n_atom ||
      aux.natoms() != n_atom || ncore < 0 || ncas < 1 ||
      int64_t{ncore} + int64_t{ncas} > static_cast<int64_t>(nao) ||
      b.num_elements() != naux * nao * nao) {
    return err_config(
        "{}: needs C nao square on device {}, a molecule and two bases on the same "
        "atoms, ncore >= 0, ncas >= 1, ncore + ncas <= nao and B naux nao^2 long (C "
        "{} square on device {}, {}, {} and {} atoms, ncore {}, ncas {}, nao {}, naux "
        "{}, B {} doubles)",
        who, res.dev_idx(), c.n(), c.dev_idx(), molecule.natoms(), n_atom, aux.natoms(), ncore,
        ncas, nao, naux, b.num_elements());
  }
  const auto n_act = static_cast<size_t>(ncas);
  if (solution.gamma.size() != n_act * n_act ||
      solution.gamma2.size() != n_act * n_act * n_act * n_act) {
    return err_config("{}: RDMs of {} and {} elements for ncas {}", who, solution.gamma.size(),
                      solution.gamma2.size(), ncas);
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
    return err_config("{}: trace(gamma) {} is no electron count of multiplicity {} in {} orbitals",
                      who, trace, molecule.multiplicity(), ncas);
  }
  const int32_t nbeta = (nelecas - two_s) / 2;
  return std::pair{nelecas - nbeta, nbeta};
}

// The probe's column is appended to the coordinates' as zvectors takes
// them, its CI and rotation rows zero, solved with them, and dropped.
Result<mcscf::ZVectors> probed_zvectors(DeviceResources &res, const std::string_view who,
                                        mcscf::RelaxedHessian &relaxed,
                                        const response::Hessian &hessian, const ci::Roots &roots,
                                        const size_t n_x, const Buffers::device<double> &gradients,
                                        const Buffers::device<double> &ci_rhs,
                                        const std::span<const double> theta_rhs,
                                        const double tolerance, const int32_t max_iterations) {
  const auto pairs = static_cast<size_t>(relaxed.size());
  const auto pairs32 = static_cast<int32_t>(pairs);
  const size_t dim = roots.vectors.num_elements();
  const size_t n_col = n_x + 1;
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const std::vector<double> s = probe_column(pairs);
  Buffers::device<double> rhs(pairs * n_col, res.device());
  Buffers::device<double> rhs_ci(dim * n_col, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(rhs.data(), gradients.data(), pairs * n_x * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(rhs.data() + (pairs * n_x), s.data(), pairs * sizeof(double),
                                     wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(rhs_ci.data(), ci_rhs.data(), dim * n_x * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(gpu_status(
      wwr::wwrMemsetAsync(rhs_ci.data() + (dim * n_x), 0, dim * sizeof(double), stream)));
  // The host s goes out of scope only after the copy has left it.
  FBF_TRY(res.sync());
  // The rotations' right-hand sides, if any, one more column of zeros.
  std::vector<double> theta(theta_rhs.begin(), theta_rhs.end());
  theta.resize(theta_rhs.size() + (theta_rhs.size() / n_x), 0.0);

  const auto singular = [&](const std::string &why) {
    return err_divergence("{}: the CP-CASSCF's H~ is singular or not positive ({}): the CASSCF "
                          "is not an isolated minimum (a saddle, or one point of a continuous "
                          "family of solutions, as when the active space holds part of a "
                          "degenerate set), and its Hessian depends on which point it reached",
                          who, why);
  };
  Result<mcscf::ZVectors> solved =
      mcscf::zvectors(res, relaxed, hessian, roots, static_cast<int32_t>(n_col), rhs.data(),
                      rhs_ci.data(), mcscf::CpSolver::Schur, tolerance, max_iterations, theta);
  if (!solved.has_value()) {
    const ErrorKind kind = solved.error().kind;
    if (kind == ErrorKind::Divergence || kind == ErrorKind::MaxIterationsExceeded) {
      return singular(solved.error().message);
    }
    return err(std::move(solved.error()));
  }
  mcscf::ZVectors &all = *solved;

  // rho against the products' resolution: the CI response leaves each
  // product's R x wrong by up to kCasscfCpResponseTolerance of R x, and at a
  // minimum 0 <= -R <= H, so H~'s eigenvalues are known to that times ||H||,
  // which max diag(H) bounds from below.
  double us = 0.0;
  double uu = 0.0;
  const double *const u = all.orbital.data() + (pairs * n_x);
  FBF_TRY(gpu_status(
      wwr::dot<double, int32_t>(blas, pairs32, u, 1, rhs.data() + (pairs * n_x), 1, &us)));
  FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, pairs32, u, 1, u, 1, &uu)));
  std::vector<double> diagonal(pairs);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(diagonal.data(), hessian.diagonal().data(),
                                     pairs * sizeof(double), wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  const double rho = uu > 0.0 ? -us / uu : 0.0;
  const double bar = kCasscfCpResponseTolerance * std::ranges::max(diagonal);
  if (!(rho > bar)) {
    return singular(std::format("a generic direction's Rayleigh quotient {:.3e}, not above "
                                "the {:.3e} its products resolve, {:.0e} of max diag(H)",
                                rho, bar, kCasscfCpResponseTolerance));
  }

  // The coordinates' columns alone.
  mcscf::ZVectors cp{.orbital = Buffers::device<double>(pairs * n_x, res.device()),
                     .ci = Buffers::device<double>(dim * n_x, res.device()),
                     .theta = std::move(all.theta),
                     .iterations = std::move(all.iterations),
                     .residuals = std::move(all.residuals)};
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(cp.orbital.data(), all.orbital.data(),
                                         pairs * n_x * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                         stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(cp.ci.data(), all.ci.data(), dim * n_x * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  cp.theta.resize(cp.theta.size() / n_col * n_x);
  cp.iterations.resize(n_x);
  cp.residuals.resize(n_x);
  FBF_TRY(res.sync());
  return std::move(cp);
}

// The CP-CASSCF step of a state-specific DF-CASSCF Hessian, and what the
// Hessian reads beside lambda^x: F^[x] and the right-hand sides, the orbital
// ones packed, and each column's residual on the unreduced equations.
struct CasscfCp {
  mcscf::Derivatives derivatives;
  Buffers::device<double> gradients;
  mcscf::ZVectors cp;
  std::vector<double> residuals;
};

// lambda^x by mcscf::zvectors on df_derivatives' right-hand sides, checked by
// one product of the unreduced operator (df_casscf's steps 1, 3 and 4,
// README.md); @p who names the caller in every refusal.
Result<CasscfCp> casscf_cp(DeviceResources &res, const std::string_view who,
                           const Molecule &molecule, const BasisSet &basis,
                           const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                           const SphericalMatrixAO &h, const Buffers::device<double> &b,
                           const double *l, const mcscf::Solution &solution, const int32_t ncore,
                           const int32_t ncas, const double tolerance,
                           const int32_t max_iterations) {
  const auto [nalpha, nbeta] =
      FBF_TRY(state_specific_electrons(who, res, molecule, basis, aux, b, solution, ncore, ncas));
  const SphericalMatrixAO &c = solution.c;
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  const auto n_act = static_cast<size_t>(ncas);
  const ScopedDeviceIndex on{res.dev_idx()};

  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  if (solution.roots.vectors.num_elements() != dim) {
    return err_config("{}: a root of {} doubles over {} determinants", who,
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

  // F^[x] and the right-hand sides E_lambda x.
  mcscf::Derivatives derivatives =
      FBF_TRY(mcscf::df_derivatives(res, molecule, basis, classes, aux, h, b, l, c, ncore, ncas,
                                    nalpha, nbeta, solution.roots.vectors.data(), weight));
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
  mcscf::RelaxedHessian relaxed = FBF_TRY(
      mcscf::RelaxedHessian::create(res, hessian, sigma, rdm, alpha, beta, solution.roots, weight,
                                    std::move(response_ints), kCasscfCpResponseTolerance));

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
  mcscf::ZVectors cp = FBF_TRY(probed_zvectors(res, who, relaxed, hessian, solution.roots, n_x,
                                               gradients, ci_rhs, {}, tolerance, max_iterations));

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
      return err_divergence("{}: the CP-CASSCF solve left a residual of {:.3e} in coordinate {} "
                            "(above {:.0e}): the parameters' Hessian is not positive there, as "
                            "at a state-specific excited root (a saddle)",
                            who, *worst, std::distance(residuals.begin(), worst),
                            kCasscfCpResidualCeiling);
    }
  }

  return CasscfCp{.derivatives = std::move(derivatives),
                  .gradients = std::move(gradients),
                  .cp = std::move(cp),
                  .residuals = std::move(residuals)};
}

Result<mcscf::ZVectors>
df_casscf_cp(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
             const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
             const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
             const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
             const double tolerance, const int32_t max_iterations) {
  CasscfCp step = FBF_TRY(casscf_cp(res, "hessian::df_casscf_cp", molecule, basis, classes, aux, h,
                                    b, l, solution, ncore, ncas, tolerance, max_iterations));
  return std::move(step.cp);
}

Result<CasscfTerms> df_casscf(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                              const std::span<const PrimitivePairClass> classes,
                              const BasisSet &aux, const SphericalMatrixAO &h,
                              const Buffers::device<double> &b, const double *l,
                              const mcscf::Solution &solution, const int32_t ncore,
                              const int32_t ncas, const double tolerance,
                              const int32_t max_iterations) {
  const ScopedDeviceIndex on{res.dev_idx()};
  // E_lambda x and lambda^x, then E_xy at the root's RDMs and F^[x].
  CasscfCp step = FBF_TRY(casscf_cp(res, "hessian::df_casscf", molecule, basis, classes, aux, h, b,
                                    l, solution, ncore, ncas, tolerance, max_iterations));
  const size_t nao = basis.nao_sphe();
  const size_t mm = nao * nao;
  const auto n_x = static_cast<size_t>(3) * basis.natoms();
  const auto n_x32 = static_cast<int32_t>(n_x);
  const auto pairs32 = static_cast<int32_t>(step.gradients.num_elements() / n_x);
  const auto dim32 = static_cast<int32_t>(solution.roots.vectors.num_elements());
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto download = [&res](const double *const from,
                               const size_t count) -> Result<std::vector<double>> {
    std::vector<double> to(count);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to.data(), from, count * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
    return to;
  };
  CasscfExy exy;
  {
    Buffers::device<double> c_columns(mm, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c_columns.data(), solution.c.data(), mm * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    exy = FBF_TRY(df_casscf_exy(res, molecule, basis, classes, aux, h, b, l, c_columns, ncore, ncas,
                                solution.gamma, solution.gamma2, step.derivatives.fock));
  }
  const mcscf::Derivatives &derivatives = step.derivatives;
  const Buffers::device<double> &gradients = step.gradients;
  const mcscf::ZVectors &cp = step.cp;

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
                     .cp = std::move(step.cp),
                     .residuals = std::move(step.residuals)};
}

} // namespace free_breakfast::hessian
