/// @file
/// @brief df_sacasscf: a state-averaged DF-CASSCF root's nuclear Hessian,
/// eq. sa assembled from the merged parts: z by mcscf::zvector, lambda^x by
/// mcscf::zvectors over the SA Hessian, E_I,xy (df_casscf_exy), the z terms
/// (df_sacasscf_z), root I's parameter Hessian (RelaxedHessian's unreduced
/// product at one-hot weights), the third-derivative action
/// (mcscf::ThirdDerivative) and, at equal weights, eq. rot
/// (mcscf::root_rotation); the solution checked, the sum symmetrized.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne
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
import free_breakfast.mcscf.root_rotation;
import free_breakfast.mcscf.third_derivative;
import free_breakfast.mcscf.zvector;

#include "diagnostics.h"

namespace free_breakfast::hessian {

// trace(gamma) is the active electron count to this (df_casscf's rule).
constexpr double kSaCasscfElectronCountTolerance = 1e-6;
// A lambda^x column whose unreduced residual is above this was not solved
// (df_casscf's ceiling, and its reason: a saddle in the parameters).
constexpr double kSaCasscfCpResidualCeiling = 1e-6;
// The averaged gamma at the weights given against the solution's, absolute:
// the solve's own RDMs, so a match is to rounding; a wrong weight misses by
// the roots' difference times it.
constexpr double kSaCasscfWeightsTolerance = 1e-8;
// Roots closer than this (hartree) are degenerate: mcscf::root_rotation's
// floor, checked here before anything runs, at any weights.
constexpr double kSaCasscfRootGapFloor = 1e-8;
// The largest rotation rate between two roots, per bohr, at which the
// Hessian still describes E_I: mcscf::root_rotation's default max_rotation,
// put on lambda^x's rotations at unequal weights (README.md).
constexpr double kSaCasscfMaxRotation = 10.0;

Result<SaCasscfTerms>
df_sacasscf(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const mcscf::Solution &solution, const std::span<const double> weights,
            const int32_t root, const int32_t ncore, const int32_t ncas, const double tolerance,
            const int32_t max_iterations) {
  const SphericalMatrixAO &c = solution.c;
  const ci::Roots &roots = solution.roots;
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  const size_t nroots = roots.energies.size();
  if (nroots < 2 || weights.size() != nroots ||
      std::ranges::any_of(
          weights, [](const double w) { return !(w > 0.0); }) ||
      root < 0 || std::cmp_greater_equal(root, nroots)) {
    return err_config("hessian::df_sacasscf: needs a state average of two roots or more, one "
                      "positive weight per root and a root among them ({} roots, {} weights, "
                      "root {})",
                      nroots, weights.size(), root);
  }
  if (c.dev_idx() != res.dev_idx() || c.n() != nao || molecule.natoms() != n_atom ||
      aux.natoms() != n_atom || ncore < 0 || ncas < 1 ||
      int64_t{ncore} + int64_t{ncas} > static_cast<int64_t>(nao) ||
      b.num_elements() != naux * nao * nao) {
    return err_config("hessian::df_sacasscf: needs C nao square on device {}, a molecule and two "
                      "bases on the same atoms, ncore >= 0, ncas >= 1, ncore + ncas <= nao and B "
                      "naux nao^2 long (C {} square on device {}, {}, {} and {} atoms, ncore {}, "
                      "ncas {}, nao {}, naux {}, B {} doubles)",
                      res.dev_idx(), c.n(), c.dev_idx(), molecule.natoms(), n_atom, aux.natoms(),
                      ncore, ncas, nao, naux, b.num_elements());
  }
  const auto n_act = static_cast<size_t>(ncas);
  if (solution.gamma.size() != n_act * n_act ||
      solution.gamma2.size() != n_act * n_act * n_act * n_act) {
    return err_config("hessian::df_sacasscf: RDMs of {} and {} elements for ncas {}",
                      solution.gamma.size(), solution.gamma2.size(), ncas);
  }
  double trace = 0.0;
  for (size_t t = 0; t < n_act; ++t) {
    trace += solution.gamma[t * (n_act + 1)];
  }
  const auto nelecas = static_cast<int32_t>(std::lround(trace));
  const int32_t two_s = molecule.multiplicity() - 1;
  if (std::abs(trace - nelecas) > kSaCasscfElectronCountTolerance || nelecas < two_s ||
      (nelecas - two_s) % 2 != 0 || nelecas - ((nelecas - two_s) / 2) > ncas) {
    return err_config("hessian::df_sacasscf: trace(gamma) {} is no electron count of "
                      "multiplicity {} in {} orbitals",
                      trace, molecule.multiplicity(), ncas);
  }
  // A degeneracy: the adiabatic energy has no Hessian there.
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = r + 1; s < nroots; ++s) {
      const double gap = std::abs(roots.energies[r] - roots.energies[s]);
      if (!(gap >= kSaCasscfRootGapFloor)) {
        return err_numerical("hessian::df_sacasscf: roots {} and {} are {:.3e} hartree apart "
                             "(under {:.0e}): a degeneracy, where E_I has no Hessian",
                             r, s, gap, kSaCasscfRootGapFloor);
      }
    }
  }
  const int32_t nbeta = (nelecas - two_s) / 2;
  const int32_t nalpha = nelecas - nbeta;
  const ScopedDeviceIndex on{res.dev_idx()};

  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  const size_t n_ci = dim * nroots;
  if (roots.vectors.num_elements() != n_ci) {
    return err_config("hessian::df_sacasscf: {} roots of {} determinants, got {} doubles", nroots,
                      dim, roots.vectors.num_elements());
  }
  const auto n_x = static_cast<size_t>(3) * n_atom;
  const auto n_x32 = static_cast<int32_t>(n_x);
  const size_t mm = nao * nao;
  const auto naux32 = static_cast<int32_t>(naux);
  const auto nao32 = static_cast<int32_t>(nao);
  const auto n_ci32 = static_cast<int32_t>(n_ci);
  const size_t all_pairs = nroots * (nroots - 1) / 2;
  const auto own = static_cast<size_t>(root);
  std::vector<double> one_hot(nroots, 0.0);
  one_hot[own] = 1.0;
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto copy = [&res](double *to, const double *from, const size_t count,
                           const wwr::wwrMemcpyKind kind) -> Status {
    return gpu_status(wwr::wwrMemcpyAsync(to, from, count * sizeof(double), kind, res.stream()));
  };
  const auto download = [&](const double *const from,
                            const size_t count) -> Result<std::vector<double>> {
    std::vector<double> to(count);
    FBF_TRY(copy(to.data(), from, count, wwr::wwrMemcpyDeviceToHost));
    FBF_TRY(res.sync());
    return to;
  };
  // gamma and Gamma of the roots at @p at, row-major, on the host.
  const auto rdms = [&](std::span<const double> at)
      -> Result<std::pair<std::vector<double>, std::vector<double>>> {
    Buffers::device<double> gamma(n_act * n_act, res.device());
    Buffers::device<double> gamma2(n_act * n_act * n_act * n_act, res.device());
    FBF_TRY(rdm.rdm12(res, at, roots.vectors.data(), gamma.data(), gamma2.data()));
    return std::pair{FBF_TRY(download(gamma.data(), gamma.num_elements())),
                     FBF_TRY(download(gamma2.data(), gamma2.num_elements()))};
  };
  // Lambda^T Y for two size x n_x blocks on the device, (x, y) at x + n_x y.
  const auto pair_columns = [&](const int32_t rows, const double *lambda,
                                const double *y) -> Result<std::vector<double>> {
    Buffers::device<double> out(n_x * n_x, res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_x32, n_x32, rows, &kOne<double>, lambda, y,
                                               &kZero<double>, out.data())));
    return download(out.data(), n_x * n_x);
  };

  // The SA Hessian A at the solution, as mcscf::df_gradient builds H~ for a
  // Z-vector (df_casscf's construction at the averaged RDMs and weights).
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, naux32, c, ncore, ncas));
  const auto [gamma_sa, gamma2_sa] = FBF_TRY(rdms(weights));
  // A Solution does not carry its weights: these must be the ones its RDMs
  // are the average at.
  for (size_t i = 0; i < gamma_sa.size(); ++i) {
    if (!(std::abs(gamma_sa[i] - solution.gamma[i]) <= kSaCasscfWeightsTolerance)) {
      return err_config("hessian::df_sacasscf: the roots' gamma at the weights given differs from "
                        "the solution's by {:.3e} at element {}: not the weights it averaged",
                        std::abs(gamma_sa[i] - solution.gamma[i]), i);
    }
  }
  const response::Fock fock_sa = FBF_TRY(response::fock(res, molecule, h, b, naux32, c, ncore, ncas,
                                                        ints.ppaa, ints.papa, gamma_sa, gamma2_sa));
  // F^I, ppaa and papa on the host for each H~ (F^I is the core's alone, the
  // same at any RDMs).
  const auto response_integrals = [&]() -> Result<mcscf::ResponseIntegrals> {
    return mcscf::ResponseIntegrals{nao,
                                    static_cast<size_t>(ncore),
                                    n_act,
                                    FBF_TRY(download(fock_sa.fock_inactive.data(), mm)),
                                    FBF_TRY(download(ints.ppaa.data(), mm * n_act * n_act)),
                                    FBF_TRY(download(ints.papa.data(), mm * n_act * n_act))};
  };
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  FBF_TRY(sigma.set_integrals(
      res, ints.hamiltonian.e_core, FBF_TRY(download(ints.hamiltonian.h_eff.data(), n_act * n_act)),
      FBF_TRY(download(ints.hamiltonian.eri.data(), n_act * n_act * n_act * n_act))));
  response::Hessian hessian_sa = FBF_TRY(
      response::Hessian::create(res, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa, fock_sa));
  mcscf::RelaxedHessian relaxed_sa = FBF_TRY(mcscf::RelaxedHessian::create(
      res, hessian_sa, sigma, rdm, alpha, beta, roots, weights, FBF_TRY(response_integrals())));
  const auto pairs = static_cast<size_t>(relaxed_sa.size());
  const auto pairs32 = static_cast<int32_t>(pairs);

  // z: A z = -(g_I, 0), g_I root I's orbital gradient at the SA orbitals. Its
  // CI rows' scaling leaves the solution the energy's.
  const auto [gamma_i, gamma2_i] = FBF_TRY(rdms(one_hot));
  const response::Fock fock_i = FBF_TRY(response::fock(res, molecule, h, b, naux32, c, ncore, ncas,
                                                       ints.ppaa, ints.papa, gamma_i, gamma2_i));
  mcscf::ZVector z =
      FBF_TRY(mcscf::zvector(res, relaxed_sa, hessian_sa, roots, fock_i.gradient.data(), nullptr,
                             tolerance, max_iterations));

  // The third-derivative operator along z. Whether the rotations are
  // parameters is its rule (all weights equal, or all pairwise different), so
  // z is laid out with every rotation and cut to its size().
  Buffers::device<double> z_packed(pairs + n_ci + all_pairs, res.device());
  FBF_TRY(copy(z_packed.data(), z.orbital.data(), pairs, wwr::wwrMemcpyDeviceToDevice));
  FBF_TRY(copy(z_packed.data() + pairs, z.ci.data(), n_ci, wwr::wwrMemcpyDeviceToDevice));
  if (z.theta.size() != all_pairs) {
    return err_config("hessian::df_sacasscf: zvector returned {} rotations for {} roots",
                      z.theta.size(), nroots);
  }
  FBF_TRY(
      copy(z_packed.data() + pairs + n_ci, z.theta.data(), all_pairs, wwr::wwrMemcpyHostToDevice));
  FBF_TRY(res.sync());
  mcscf::ThirdDerivative third_op =
      FBF_TRY(mcscf::ThirdDerivative::create(res, h, b, naux32, c, ncore, ncas, alpha, beta,
                                             roots.vectors.data(), weights, z_packed.data()));
  const auto size = static_cast<size_t>(third_op.size());
  const size_t n_theta = size - pairs - n_ci;
  const bool rotating = n_theta != 0;
  Buffers::device<double> z_column(size, res.device());
  FBF_TRY(copy(z_column.data(), z_packed.data(), size, wwr::wwrMemcpyDeviceToDevice));

  // lambda^x over df_derivatives' SA right-hand sides in zvector's
  // convention: the orbital rows packed, the CI rows over 2 w_r, the
  // rotation rows over 2 (w_r - w_s), which is H~^x_sr itself.
  mcscf::Derivatives derivatives_sa =
      FBF_TRY(mcscf::df_derivatives(res, molecule, basis, classes, aux, h, b, l, c, ncore, ncas,
                                    nalpha, nbeta, roots.vectors.data(), weights));
  Buffers::device<double> gradients(pairs * n_x, res.device());
  for (size_t x = 0; x < n_x; ++x) {
    FBF_TRY(response::pack_orbital_pairs(res, nao32, ncore, ncas,
                                         derivatives_sa.rhs_orbital.data() + (x * mm),
                                         gradients.data() + (x * pairs)));
  }
  Buffers::device<double> ci_rhs(n_ci * n_x, res.device());
  FBF_TRY(
      copy(ci_rhs.data(), derivatives_sa.rhs_ci.data(), n_ci * n_x, wwr::wwrMemcpyDeviceToDevice));
  const auto dim32 = static_cast<int32_t>(dim);
  for (size_t x = 0; x < n_x; ++x) {
    for (size_t r = 0; r < nroots; ++r) {
      const double scale = 0.5 / weights[r];
      FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, dim32, &scale,
                                                    ci_rhs.data() + (x * n_ci) + (r * dim), 1)));
    }
  }
  std::vector<double> theta_rhs;
  if (rotating) {
    for (size_t x = 0; x < n_x; ++x) {
      for (size_t r = 0; r < nroots; ++r) {
        for (size_t s = r + 1; s < nroots; ++s) {
          theta_rhs.push_back(derivatives_sa.hamiltonian[s + (nroots * (r + (nroots * x)))]);
        }
      }
    }
  }
  mcscf::ZVectors cp = FBF_TRY(
      mcscf::zvectors(res, relaxed_sa, hessian_sa, roots, n_x32, gradients.data(), ci_rhs.data(),
                      mcscf::CpSolver::Schur, tolerance, max_iterations, theta_rhs));

  // The solution checked on the unreduced equations (df_casscf's check), the
  // rotation rows y_theta + t at unequal weights.
  std::vector<double> residuals(n_x);
  {
    Buffers::device<double> r(pairs * n_x, res.device());
    Buffers::device<double> r_c(n_ci * n_x, res.device());
    std::vector<double> r_theta(rotating ? cp.theta.size() : 0);
    FBF_TRY(relaxed_sa.apply_coupled(
        res, n_x32, cp.orbital.data(), cp.ci.data(), r.data(), r_c.data(),
        rotating ? std::span<const double>(cp.theta) : std::span<const double>{}, r_theta));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, pairs32 * n_x32, &kOne<double>,
                                                  gradients.data(), 1, r.data(), 1)));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n_ci32 * n_x32, &kOne<double>,
                                                  ci_rhs.data(), 1, r_c.data(), 1)));
    for (size_t x = 0; x < n_x; ++x) {
      double orbital = 0.0;
      double ci = 0.0;
      FBF_TRY(gpu_status(
          wwr::nrm2<double, int32_t>(blas, pairs32, r.data() + (x * pairs), 1, &orbital)));
      FBF_TRY(
          gpu_status(wwr::nrm2<double, int32_t>(blas, n_ci32, r_c.data() + (x * n_ci), 1, &ci)));
      double rotations = 0.0;
      for (size_t k = 0; k < n_theta; ++k) {
        const double row = r_theta[(x * n_theta) + k] + theta_rhs[(x * n_theta) + k];
        rotations += row * row;
      }
      residuals[x] = std::hypot(orbital, ci, std::sqrt(rotations));
    }
    const auto worst = std::ranges::max_element(residuals);
    if (*worst > kSaCasscfCpResidualCeiling) {
      return err_divergence("hessian::df_sacasscf: the CP-CASSCF solve left a residual of "
                            "{:.3e} in coordinate {} (above {:.0e}): the SA Hessian is not "
                            "positive there (a saddle)",
                            *worst, std::distance(residuals.begin(), worst),
                            kSaCasscfCpResidualCeiling);
    }
  }
  // At unequal weights lambda^x's rotations are the roots' mixing rate, the
  // quantity mcscf::root_rotation bounds at equal weights.
  if (rotating) {
    const auto fastest =
        std::ranges::max_element(cp.theta, {}, [](const double t) { return std::abs(t); });
    if (std::abs(*fastest) > kSaCasscfMaxRotation) {
      const auto at = static_cast<size_t>(std::distance(cp.theta.begin(), fastest));
      return err_numerical("hessian::df_sacasscf: the roots rotate into each other at {:.3e} per "
                           "bohr along coordinate {} (above {:.0f}): too near a degeneracy for "
                           "E_I to have a Hessian",
                           std::abs(*fastest), at / n_theta, kSaCasscfMaxRotation);
    }
  }

  // Lambda in ThirdDerivative's packed layout: pairs, d_r per root, Theta.
  Buffers::device<double> lambda(size * n_x, res.device());
  for (size_t x = 0; x < n_x; ++x) {
    double *column = lambda.data() + (x * size);
    FBF_TRY(copy(column, cp.orbital.data() + (x * pairs), pairs, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(copy(column + pairs, cp.ci.data() + (x * n_ci), n_ci, wwr::wwrMemcpyDeviceToDevice));
    if (rotating) {
      FBF_TRY(copy(column + pairs + n_ci, cp.theta.data() + (x * n_theta), n_theta,
                   wwr::wwrMemcpyHostToDevice));
    }
  }
  FBF_TRY(res.sync());
  const auto size32 = static_cast<int32_t>(size);

  // The third-derivative action, Lambda^T T(., Lambda, z).
  std::vector<double> third;
  {
    Buffers::device<double> y(size * n_x, res.device());
    FBF_TRY(third_op.apply(res, n_x32, lambda.data(), y.data()));
    third = FBF_TRY(pair_columns(size32, lambda.data(), y.data()));
  }

  // Root I's parameter Hessian between the responses: the unreduced product
  // at weights one-hot on I (H at root I's Fock), its CI rows (unweighted,
  // as zvector's) times 2 w_r and its rotation rows times 2 (w_r - w_s),
  // back to the energy's, then paired with Lambda.
  std::vector<double> root_hessian;
  {
    response::Hessian hessian_i = FBF_TRY(
        response::Hessian::create(res, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa, fock_i));
    mcscf::RelaxedHessian relaxed_i = FBF_TRY(mcscf::RelaxedHessian::create(
        res, hessian_i, sigma, rdm, alpha, beta, roots, one_hot, FBF_TRY(response_integrals())));
    Buffers::device<double> y_o(pairs * n_x, res.device());
    Buffers::device<double> y_c(n_ci * n_x, res.device());
    std::vector<double> y_theta(rotating ? cp.theta.size() : 0);
    FBF_TRY(relaxed_i.apply_coupled(
        res, n_x32, cp.orbital.data(), cp.ci.data(), y_o.data(), y_c.data(),
        rotating ? std::span<const double>(cp.theta) : std::span<const double>{}, y_theta));
    Buffers::device<double> y(size * n_x, res.device());
    for (size_t x = 0; x < n_x; ++x) {
      FBF_TRY(copy(y.data() + (x * size), y_o.data() + (x * pairs), pairs,
                   wwr::wwrMemcpyDeviceToDevice));
      for (size_t r = 0; r < nroots; ++r) {
        const double scale = 2.0 * one_hot[r];
        FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, dim32, &scale,
                                                      y_c.data() + (x * n_ci) + (r * dim), 1)));
      }
      FBF_TRY(copy(y.data() + (x * size) + pairs, y_c.data() + (x * n_ci), n_ci,
                   wwr::wwrMemcpyDeviceToDevice));
    }
    if (rotating) {
      for (size_t x = 0; x < n_x; ++x) {
        size_t at = x * n_theta;
        for (size_t r = 0; r < nroots; ++r) {
          for (size_t s = r + 1; s < nroots; ++s, ++at) {
            y_theta[at] *= 2.0 * (one_hot[r] - one_hot[s]);
          }
        }
        FBF_TRY(copy(y.data() + (x * size) + pairs + n_ci, y_theta.data() + (x * n_theta), n_theta,
                     wwr::wwrMemcpyHostToDevice));
      }
    }
    root_hessian = FBF_TRY(pair_columns(size32, lambda.data(), y.data()));
  }

  // L_xy's z part and L_xlambda at z, and L_xlambda lambda^y.
  SaCasscfZTerms z_terms =
      FBF_TRY(df_sacasscf_z(res, molecule, basis, classes, aux, h, b, l, c, ncore, ncas, nalpha,
                            nbeta, roots.vectors, weights, root, z_column));
  std::vector<double> lxlambda_lambda =
      FBF_TRY(pair_columns(size32, z_terms.lxlambda.data(), lambda.data()));

  // E_I,xy at root I's RDMs and F^[x] (df_derivatives at one-hot weights).
  CasscfExy exy;
  {
    const mcscf::Derivatives derivatives_i =
        FBF_TRY(mcscf::df_derivatives(res, molecule, basis, classes, aux, h, b, l, c, ncore, ncas,
                                      nalpha, nbeta, roots.vectors.data(), one_hot));
    Buffers::device<double> c_columns(mm, res.device());
    FBF_TRY(copy(c_columns.data(), c.data(), mm, wwr::wwrMemcpyDeviceToDevice));
    exy = FBF_TRY(df_casscf_exy(res, molecule, basis, classes, aux, h, b, l, c_columns, ncore, ncas,
                                gamma_i, gamma2_i, derivatives_i.fock));
  }

  // Eq. rot at equal weights (it refuses a degeneracy); none at unequal.
  std::vector<double> rotation(n_x * n_x, 0.0);
  if (!rotating) {
    const mcscf::RootRotation rot = FBF_TRY(mcscf::root_rotation(
        res, molecule, basis, h, b, c, ncore, ncas, nalpha, nbeta, roots.vectors.data(), weights,
        roots.energies, derivatives_sa, cp.orbital.data()));
    std::copy_n(rot.hessian.begin() + static_cast<std::ptrdiff_t>(own * n_x * n_x), n_x * n_x,
                rotation.begin());
  }

  // Summed and symmetrized, the antisymmetric part measured as it is
  // discarded: L_xlambda lambda^y enters with its transpose.
  std::vector<double> total(n_x * n_x);
  double asymmetry = 0.0;
  const auto element = [&](const size_t x, const size_t y) {
    const size_t at = x + (n_x * y);
    return exy.exy[at] + z_terms.lxy.exy[at] + lxlambda_lambda[at] +
           lxlambda_lambda[y + (n_x * x)] + root_hessian[at] + third[at] + rotation[at];
  };
  for (size_t y = 0; y < n_x; ++y) {
    for (size_t x = 0; x < n_x; ++x) {
      const double upper = element(x, y);
      const double lower = element(y, x);
      asymmetry = std::max(asymmetry, std::abs(upper - lower));
      total[x + (n_x * y)] = 0.5 * (upper + lower);
    }
  }
  return SaCasscfTerms{.hessian = std::move(total),
                       .asymmetry = asymmetry,
                       .exy = std::move(exy),
                       .z_terms = std::move(z_terms),
                       .lxlambda_lambda = std::move(lxlambda_lambda),
                       .root_hessian = std::move(root_hessian),
                       .third = std::move(third),
                       .rotation = std::move(rotation),
                       .z = std::move(z),
                       .cp = std::move(cp),
                       .residuals = std::move(residuals)};
}

} // namespace free_breakfast::hessian
