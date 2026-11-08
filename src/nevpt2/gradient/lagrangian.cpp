/// @file
/// @brief lagrangian_gradient, sc_gradient and pc_gradient: the Z-vector of
/// the SC- or PC-NEVPT2 Lagrangian on a single-state DF-CASSCF over the CI-relaxed
/// Hessian, the gate that its orbital source term is symmetric in every
/// block, and the nuclear term as one mcscf::df_gradient pass over the
/// multipliers and the correlation's MO adjoint. And qd_sc_lagrangian: the
/// SC-QD-NEVPT2 Lagrangian's multipliers on a DF-SA-CASSCF, through the same
/// Z-vector and gate at the average's weights, at any symmetric W (the
/// state form a wrapper at u_K u_K^T); qd_sc_gradient, its two
/// derivative passes (df_gradient at the mix rho, df_transition_gradient at
/// zeta's folded RDMs); qd_sc_coupling, the same at an interstate pair's
/// W^PQ; and qd_pc_gradient and qd_pc_coupling, the same Lagrangian and
/// passes over the PC-QD adjoint, at a state's W, any symmetric W or W^PQ.

module free_breakfast.nevpt2.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, axpy, scal, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.active_space; // casscf_integrals
import free_breakfast.recipes.blas;           // gemm
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.response.fock;
import free_breakfast.response.hessian;
import free_breakfast.response.orbital_pairs;
import free_breakfast.mcscf.solve;
import free_breakfast.response.relaxed_hessian;
import free_breakfast.response.zvector;
import free_breakfast.mcscf.root_rotation; // kRootGapFloor
import free_breakfast.mcscf.gradient;
import :operands; // to_host, to_device
import :builders; // with_qd_inputs

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// The Lagrangian's orbital source term X^L must be symmetric in every block
// at the solution (docs/nevpt2/nevpt2.tex, "The gate: a free, exact check of
// the whole solve"). Its antisymmetric part is the Z-vector's residual on the
// non-redundant pairs (to its 1e-10), z^c's division on the inactive ones
// (round-off), and on the active-active ones the CI equation's residual
// (the CI response's conjugate gradients, 1e-10 of their right-hand side,
// times y). Past this, a block's term is wrong.
constexpr double kAsymmetryTolerance = 1e-8;
// trace(gamma) is the active electron count to this, as mcscf::df_gradient
// holds it.
constexpr double kNelecasTolerance = 1e-6;
// The five-point stencil's offsets and weights: d/de F(e) =
// sum_k w_k F(s_k e) / (12 e), exact for F a polynomial of degree 4 or less.
constexpr std::array<double, 4> kStencilOffsets{-2.0, -1.0, 1.0, 2.0};
constexpr std::array<double, 4> kStencilWeights{1.0, -8.0, 8.0, -1.0};
// The stencil's step: max |e kappa| this, so C (1 + 2 e kappa) stays within
// a factor of order one of C and the round-off of the quotient is ~1e-15.
constexpr double kStencilReach = 0.1;
// The QD Lagrangians' Z-vector is solved to this (the single-state one's to
// its zvector_tolerance, by default the same 1e-10), in at most this many H~
// products, and its true residual (one more product) is held to it, one
// column or several. The CI response inside each product is converged to
// response::response_tolerance_for it, relative to its right-hand side: at
// RelaxedHessian's default 1e-8 the products' error, about 0.04 ||z|| 1e-8,
// was a floor under the true residual that crossed 1e-10 at PSB3 (2.5e-10 at
// cc-pVDZ, 3.8e-10 at cc-pVTZ, #864; mcscf/intersection/README.md "The floor
// under the true residual").
constexpr double kZvectorTolerance = 1e-10;
constexpr int32_t kZvectorProducts = 200;

// The six blocks the gate reports, each (row space, column space): 0 core,
// 1 active, 2 virtual, as ScGradient::asymmetry orders them.
constexpr std::array<std::pair<int32_t, int32_t>, 6> kAsymmetryBlocks{
    {{0, 0}, {1, 1}, {2, 2}, {1, 0}, {2, 0}, {2, 1}}};
constexpr std::array<std::string_view, 6> kAsymmetryBlockNames{"core-core",       "active-active",
                                                               "virtual-virtual", "active-core",
                                                               "virtual-core",    "virtual-active"};

// The generalized Fock F (nmo x nmo, column-major) of the orbitals @p c at
// the RDMs @p gamma and @p gamma2 (row-major, as ci::Rdm::rdm12's), on the
// host: response::fock over casscf_integrals of @p c.
Result<std::vector<double>> lagrangian_fock(DeviceResources &res, const Molecule &molecule,
                                            const BasisSet &basis, const SphericalMatrixAO &h,
                                            const Buffers::device<double> &b, const int32_t naux,
                                            const SphericalMatrixAO &c, const int32_t ncore,
                                            const int32_t ncas, std::span<const double> gamma,
                                            std::span<const double> gamma2) {
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, naux, c, ncore, ncas));
  const response::Fock f = FBF_TRY(response::fock(res, molecule, h, b, naux, c, ncore, ncas,
                                                  ints.ppaa, ints.papa, gamma, gamma2));
  const auto nmo = static_cast<size_t>(c.n());
  return to_host(res, f.fock.data(), nmo * nmo);
}

// max |A^L_pq| over p > q per block (kAsymmetryBlocks), A^L the Lagrangian's
// orbital source term's antisymmetric part:
//   A^L = 2 (F - F^T)(gamma + gamma_T) + A^Z + @p a,
// F the generalized Fock at the Lagrangian's CAS RDMs @p gamma and @p gamma2
// (the root's, or the rho-weighted ones plus zeta's) plus z's transition RDMs
// sum_r w_r (<z_r|.|c_r> + <c_r|.|z_r>) at the solution's @p weights, z_r
// carrying the rotations among the roots as its part along them (z_r +=
// Theta_sr c_s, as mcscf::df_gradient takes them): E_cas's, z . g_ci's and
// the root-pair terms. A^Z = 2 (F' - F'^T), F' the derivative of the
// solution's own F (its weighted RDMs) along C (1 + e kappa(Z)) (Z . g_orb's
// term: by the symmetry of the orbital Hessian at the solution, whose full
// gradient vanishes, the derivative of the gradient along Z), and @p a the
// correlation's and z^c's A. F is a polynomial of degree 4 in C, so F' is
// the five-point stencil's, exactly. F is affine in the RDMs, its zero-RDM
// part F_0 the core's; @p core_shift adds that many more F_0 to F, so that a
// Lagrangian whose RDMs carry the core at a weight other than 1 (the QD W
// form's tr W) has it at that weight: 0 for a weight of 1.
Result<std::array<double, 6>>
lagrangian_asymmetry(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                     const SphericalMatrixAO &h, const Buffers::device<double> &b,
                     const int32_t naux, const mcscf::Solution &solution, const int32_t ncore,
                     const int32_t ncas, ci::Rdm &rdm, const std::span<const double> weights,
                     std::vector<double> gamma, std::vector<double> gamma2,
                     const response::ZVector &z, const double *a, const double core_shift) {
  const SphericalMatrixAO &c = solution.c;
  const auto nmo = static_cast<size_t>(c.n());
  const auto n = static_cast<size_t>(ncas);
  const auto n32 = static_cast<int32_t>(nmo);
  const wwr::wwrblasHandle_t blas = res.blas().get();

  // z with Theta folded in, then half the difference of the weighted RDMs of
  // c + z and c - z.
  const size_t nroots = weights.size();
  const auto dim = static_cast<size_t>(rdm.size());
  const size_t block = dim * nroots;
  std::vector<double> z_ci = FBF_TRY(to_host(res, z.ci.data(), block));
  const std::vector<double> roots = FBF_TRY(to_host(res, solution.roots.vectors.data(), block));
  size_t pair = 0;
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = r + 1; s < nroots && pair < z.theta.size(); ++s, ++pair) {
      for (size_t i = 0; i < dim; ++i) {
        z_ci[(r * dim) + i] += z.theta[pair] * roots[(s * dim) + i];
        z_ci[(s * dim) + i] -= z.theta[pair] * roots[(r * dim) + i];
      }
    }
  }
  Buffers::device<double> gamma_d(n * n, res.device());
  Buffers::device<double> gamma2_d(n * n * n * n, res.device());
  std::vector<double> shifted(block);
  for (const double sign : {1.0, -1.0}) {
    for (size_t i = 0; i < block; ++i) {
      shifted[i] = roots[i] + (sign * z_ci[i]);
    }
    const Buffers::device<double> shifted_d = FBF_TRY(to_device(res, shifted));
    FBF_TRY(rdm.rdm12(res, weights, shifted_d.data(), gamma_d.data(), gamma2_d.data()));
    const std::vector<double> part = FBF_TRY(to_host(res, gamma_d.data(), n * n));
    const std::vector<double> part2 = FBF_TRY(to_host(res, gamma2_d.data(), n * n * n * n));
    for (size_t i = 0; i < part.size(); ++i) {
      gamma[i] += 0.5 * sign * part[i];
    }
    for (size_t i = 0; i < part2.size(); ++i) {
      gamma2[i] += 0.5 * sign * part2[i];
    }
  }
  std::vector<double> f_lagrangian =
      FBF_TRY(lagrangian_fock(res, molecule, basis, h, b, naux, c, ncore, ncas, gamma, gamma2));
  if (core_shift != 0.0) {
    const std::vector<double> f0 = FBF_TRY(
        lagrangian_fock(res, molecule, basis, h, b, naux, c, ncore, ncas,
                        std::vector<double>(n * n, 0.0), std::vector<double>(n * n * n * n, 0.0)));
    for (size_t i = 0; i < f_lagrangian.size(); ++i) {
      f_lagrangian[i] += core_shift * f0[i];
    }
  }

  // F' along kappa(Z) by the stencil, at the solution's weighted RDMs.
  Buffers::device<double> kappa(nmo * nmo, res.device());
  FBF_TRY(response::unpack_orbital_pairs(res, n32, ncore, ncas, z.orbital.data(), kappa.data()));
  const std::vector<double> kappa_host = FBF_TRY(to_host(res, kappa.data(), nmo * nmo));
  double largest = 0.0;
  for (const double v : kappa_host) {
    largest = std::max(largest, std::abs(v));
  }
  std::vector<double> f_prime(nmo * nmo, 0.0);
  if (largest > 0.0) {
    const double step = kStencilReach / largest;
    SphericalMatrixAO moved(res, basis);
    for (size_t k = 0; k < kStencilOffsets.size(); ++k) {
      const double e = kStencilOffsets[k] * step;
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(moved.data(), c.data(), nmo * nmo * sizeof(double),
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
      FBF_TRY(gpu_status(gemm(blas, n32, &e, c.data(), kappa.data(), &kOne<double>, moved.data())));
      const std::vector<double> f = FBF_TRY(lagrangian_fock(
          res, molecule, basis, h, b, naux, moved, ncore, ncas, solution.gamma, solution.gamma2));
      for (size_t i = 0; i < f.size(); ++i) {
        f_prime[i] += kStencilWeights[k] * f[i] / (12.0 * step);
      }
    }
  }

  const std::vector<double> source = FBF_TRY(to_host(res, a, nmo * nmo));
  const std::array<size_t, 4> first{0, static_cast<size_t>(ncore),
                                    static_cast<size_t>(ncore + ncas), nmo};
  std::array<double, 6> worst{};
  for (size_t k = 0; k < kAsymmetryBlocks.size(); ++k) {
    const auto [row, col] = kAsymmetryBlocks[k];
    for (size_t q = first[col]; q < first[col + 1]; ++q) {
      for (size_t p = std::max(first[row], q + 1); p < first[row + 1]; ++p) {
        const size_t pq = p + (nmo * q);
        const size_t qp = q + (nmo * p);
        const double value = (2.0 * (f_lagrangian[pq] - f_lagrangian[qp])) +
                             (2.0 * (f_prime[pq] - f_prime[qp])) + source[pq];
        worst[k] = std::max(worst[k], std::abs(value));
      }
    }
  }
  return worst;
}

// @p phi's h_bar, Lambda, y, X and A added onto @p h_bar, @p lambda, @p y,
// @p x and @p a in place, field by field: a correlation energy's adjoint
// plus Phi's, as one Lagrangian's.
Status add_adjoint(DeviceResources &res, Buffers::device<double> &h_bar,
                   Buffers::device<double> &lambda, Buffers::device<double> &y,
                   Buffers::device<double> &x, Buffers::device<double> &a, const ZcAdjoint &phi) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  for (const auto &[to, from] :
       {std::pair{&h_bar, &phi.h_bar}, std::pair{&lambda, &phi.lambda}, std::pair{&y, &phi.y},
        std::pair{&x, &phi.x}, std::pair{&a, &phi.a}}) {
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(to->num_elements()),
                                                  &kOne<double>, from->data(), 1, to->data(), 1)));
  }
  return {};
}

// The active strings of the solution's spin over @p ncas orbitals, the
// active electrons trace(@p gamma): InvalidConfig, naming @p who, when that is
// no electron count of the molecule's multiplicity in ncas orbitals.
Result<std::pair<ci::DeviceStringSet, ci::DeviceStringSet>>
active_strings(DeviceResources &res, const std::string_view who, const Molecule &molecule,
               const std::span<const double> gamma, const int32_t ncas) {
  const auto n = static_cast<size_t>(ncas);
  double trace = 0.0;
  for (size_t t = 0; t < n; ++t) {
    trace += gamma[t * (n + 1)];
  }
  const auto nelecas = static_cast<int32_t>(std::lround(trace));
  const int32_t two_s = molecule.multiplicity() - 1;
  if (std::abs(trace - nelecas) > kNelecasTolerance || nelecas < two_s ||
      (nelecas - two_s) % 2 != 0 || nelecas - ((nelecas - two_s) / 2) > ncas) {
    return err_config("{}: trace(gamma) {} is no electron count of multiplicity {} in {} orbitals",
                      who, trace, molecule.multiplicity(), ncas);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const int32_t n_beta = (nelecas - two_s) / 2;
  ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nelecas - n_beta));
  ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, n_beta));
  return std::pair{std::move(alpha), std::move(beta)};
}

// The Z-vector of a Lagrangian on @p solution at its @p weights:
// response::zvector over the CI-relaxed Hessian at the solution, as
// mcscf::df_gradient builds it for its own (H_0's sigma, row-major with no
// reorder: h_eff is symmetric, and (tu|vw) column-major is (wv|ut)
// row-major; the orbital Hessian over @p averaged, the Fock at the
// solution's weighted RDMs; the response integrals), for the right-hand
// sides @p orbital_rhs (the packed pairs), @p ci_rhs (dim x nroots) and
// @p theta_rhs (zvector's), @p ncol columns of each stacked, solved to
// @p tolerance with the CI response inside H~ converged to
// response::response_tolerance_for(@p tolerance). One column is response::zvector's
// solve; more are one response::zvectors (Schur) over the one Hessian, split
// into a ZVector per column. Either way each column's true residual (one
// more product) is held to @p tolerance (Numerical naming @p who and the
// column): the conjugate gradients stop on their recurrence (#786), and a
// column whose true residual then misses is restarted from it once, to half
// the tolerance (response/zvector/README.md "The restart", #1108), so a
// refusal here is a floor, not a recurrence that landed on the bar.
Result<std::vector<response::ZVector>>
lagrangian_zvector(DeviceResources &res, const std::string_view who, const SphericalMatrixAO &h,
                   const Buffers::device<double> &b, const int32_t naux,
                   const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                   const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta, ci::Rdm &rdm,
                   const CasscfIntegrals &ints, const response::Fock &averaged,
                   const std::span<const double> weights, const int32_t ncol,
                   const double *orbital_rhs, const double *ci_rhs,
                   const std::span<const double> theta_rhs, const double tolerance) {
  const SphericalMatrixAO &c = solution.c;
  const auto nmo = static_cast<size_t>(c.n());
  const auto n = static_cast<size_t>(ncas);
  const size_t mo2 = nmo * nmo;
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  FBF_TRY(sigma.set_integrals(res, ints.hamiltonian.e_core,
                              FBF_TRY(to_host(res, ints.hamiltonian.h_eff.data(), n * n)),
                              FBF_TRY(to_host(res, ints.hamiltonian.eri.data(), n * n * n * n))));
  response::Hessian hessian = FBF_TRY(
      response::Hessian::create(res, h, b, naux, c, ncore, ncas, ints.ppaa, ints.papa, averaged));
  response::ResponseIntegrals response_ints{
      nmo,
      static_cast<size_t>(ncore),
      n,
      FBF_TRY(to_host(res, averaged.fock_inactive.data(), mo2)),
      FBF_TRY(to_host(res, ints.ppaa.data(), mo2 * n * n)),
      FBF_TRY(to_host(res, ints.papa.data(), mo2 * n * n))};
  response::RelaxedHessian relaxed = FBF_TRY(response::RelaxedHessian::create(
      res, hessian, sigma, rdm, alpha, beta, solution.roots, weights, std::move(response_ints),
      response::response_tolerance_for(tolerance)));
  const auto check = [&](const size_t col, const size_t cols, const double residual,
                         const int32_t products) -> Status {
    if (!(residual <= tolerance)) {
      return err_numerical("{}: Z-vector column {} of {} has true residual {:.3e} after {} "
                           "products, above {:g}",
                           who, col, cols, residual, products, tolerance);
    }
    return {};
  };
  std::vector<response::ZVector> out;
  if (ncol == 1) {
    out.push_back(FBF_TRY(response::zvector(res, relaxed, hessian, solution.roots, orbital_rhs,
                                            ci_rhs, tolerance, kZvectorProducts, theta_rhs)));
    FBF_TRY(check(0, 1, out.front().residual, out.front().iterations));
    return out;
  }
  const response::ZVectors z = FBF_TRY(
      response::zvectors(res, relaxed, hessian, solution.roots, ncol, orbital_rhs, ci_rhs,
                         response::CpSolver::Schur, tolerance, kZvectorProducts, theta_rhs));
  const auto cols = static_cast<size_t>(ncol);
  const size_t orbital = z.orbital.num_elements() / cols;
  const size_t ci = z.ci.num_elements() / cols;
  const size_t theta = z.theta.size() / cols;
  for (size_t col = 0; col < cols; ++col) {
    FBF_TRY(check(col, cols, z.residuals[col], z.iterations[col]));
    response::ZVector one{.orbital = Buffers::device<double>(orbital, res.device()),
                          .ci = Buffers::device<double>(ci, res.device()),
                          .theta = std::vector<double>(
                              z.theta.begin() + static_cast<std::ptrdiff_t>(col * theta),
                              z.theta.begin() + static_cast<std::ptrdiff_t>((col + 1) * theta)),
                          .iterations = z.iterations[col],
                          .residual = z.residuals[col]};
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(one.orbital.data(), z.orbital.data() + (col * orbital),
                                           orbital * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                           res.stream())));
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(one.ci.data(), z.ci.data() + (col * ci), ci * sizeof(double),
                                       wwr::wwrMemcpyDeviceToDevice, res.stream())));
    out.push_back(std::move(one));
  }
  // z is freed on return, while the copies read it.
  FBF_TRY(res.sync());
  return out;
}

// Refuse an asymmetric block of the gate, naming @p who.
Status check_asymmetry(const std::string_view who, const std::array<double, 6> &asymmetry,
                       const response::ZVector &z) {
  for (size_t k = 0; k < asymmetry.size(); ++k) {
    if (!(asymmetry[k] <= kAsymmetryTolerance)) {
      return err_numerical("{}: the Lagrangian's orbital source term is not symmetric on its {} "
                           "block: max |X^L - X^L^T| {:.3e} (gate {:g}; Z-vector {} iterations, "
                           "residual {:.3e})",
                           who, kAsymmetryBlockNames[k], asymmetry[k], kAsymmetryTolerance,
                           z.iterations, z.residual);
    }
  }
  return {};
}

Result<ScGradient> lagrangian_gradient(DeviceResources &res, const Molecule &molecule,
                                       const BasisSet &basis,
                                       const std::span<const PrimitivePairClass> classes,
                                       const BasisSet &aux, const SphericalMatrixAO &h,
                                       const Buffers::device<double> &b, const double *l,
                                       const mcscf::Solution &solution, const int32_t ncore,
                                       const int32_t ncas, ScAdjoint corr, ZcMultipliers zc,
                                       const ZcAdjoint &phi, const double zvector_tolerance) {
  const size_t nroots = solution.roots.energies.size();
  if (nroots != 1) {
    return err_unsupported("nevpt2::lagrangian_gradient: the solution averages {} roots, and the "
                           "NEVPT2 gradient is of a single-state CASSCF only",
                           nroots);
  }
  const SphericalMatrixAO &c = solution.c;
  const int64_t nao = basis.nao_sphe();
  const int64_t naux = aux.nao_sphe();
  if (c.n() != nao || ncore < 0 || ncas < 1 || ncore + ncas > nao) {
    return err_config("nevpt2::lagrangian_gradient: C over {} orbitals, ncore {} and ncas {}, "
                      "but the basis has {} spherical AOs",
                      c.n(), ncore, ncas, nao);
  }
  const auto nmo = static_cast<size_t>(nao);
  const auto n = static_cast<size_t>(ncas);
  if (solution.gamma.size() != n * n || solution.gamma2.size() != n * n * n * n) {
    return err_config("nevpt2::lagrangian_gradient: RDMs of {} and {} elements for ncas {}",
                      solution.gamma.size(), solution.gamma2.size(), ncas);
  }
  const auto [alpha, beta] =
      FBF_TRY(active_strings(res, "nevpt2::lagrangian_gradient", molecule, solution.gamma, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  const size_t mo2 = nmo * nmo;
  for (const auto &[what, have, want] :
       {std::tuple{"the CI vector", solution.roots.vectors.num_elements(), dim},
        std::tuple{"corr's h_bar", corr.h_bar.num_elements(), mo2},
        std::tuple{"phi's h_bar", phi.h_bar.num_elements(), mo2},
        std::tuple{"corr's lambda", corr.lambda.num_elements(), static_cast<size_t>(naux) * mo2},
        std::tuple{"phi's lambda", phi.lambda.num_elements(), static_cast<size_t>(naux) * mo2},
        std::tuple{"corr's y", corr.y.num_elements(), dim},
        std::tuple{"phi's y", phi.y.num_elements(), dim},
        std::tuple{"corr's x", corr.x.num_elements(), mo2},
        std::tuple{"phi's x", phi.x.num_elements(), mo2},
        std::tuple{"corr's a", corr.a.num_elements(), mo2},
        std::tuple{"phi's a", phi.a.num_elements(), mo2}}) {
    if (have != want) {
      return err_config("nevpt2::lagrangian_gradient: {} holds {} doubles, and {} are wanted", what,
                        have, want);
    }
  }

  // corr + Phi, field by field.
  FBF_TRY(add_adjoint(res, corr.h_bar, corr.lambda, corr.y, corr.x, corr.a, phi));

  // The Z-vector at the root's own Fock, its right-hand sides g + A over the
  // packed pairs and 1/2 (y + y^c), the tex's 2 P (H - E) P row in
  // zvector's (H_0 - E) one.
  const auto naux32 = static_cast<int32_t>(naux);
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, naux32, c, ncore, ncas));
  const response::Fock fock =
      FBF_TRY(response::fock(res, molecule, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa,
                             solution.gamma, solution.gamma2));
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto packed = static_cast<int32_t>(fock.gradient.num_elements());
  Buffers::device<double> orbital_rhs(static_cast<size_t>(packed), res.device());
  FBF_TRY(response::pack_orbital_pairs(res, static_cast<int32_t>(nmo), ncore, ncas, corr.a.data(),
                                       orbital_rhs.data()));
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, packed, &kOne<double>, fock.gradient.data(),
                                                1, orbital_rhs.data(), 1)));
  Buffers::device<double> ci_rhs(dim, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ci_rhs.data(), corr.y.data(), dim * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, res.stream())));
  FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, static_cast<int32_t>(dim), &kHalf<double>,
                                                ci_rhs.data(), 1)));
  const std::array<double, 1> weights{1.0};
  response::ZVector multipliers = std::move(
      FBF_TRY(lagrangian_zvector(res, "nevpt2::lagrangian_gradient", h, b, naux32, solution, ncore,
                                 ncas, alpha, beta, rdm, ints, fock, weights, 1, orbital_rhs.data(),
                                 ci_rhs.data(), {}, zvector_tolerance))
          .front());

  // The gate: X^L symmetric in every block.
  const std::array<double, 6> asymmetry = FBF_TRY(
      lagrangian_asymmetry(res, molecule, basis, h, b, naux32, solution, ncore, ncas, rdm, weights,
                           solution.gamma, solution.gamma2, multipliers, corr.a.data(), 0.0));
  FBF_TRY(check_asymmetry("nevpt2::lagrangian_gradient", asymmetry, multipliers));

  const mcscf::MoAdjoint extra{
      .h_bar = corr.h_bar.data(), .lambda = corr.lambda.data(), .x = corr.x.data()};
  mcscf::GradientTerms terms = FBF_TRY(mcscf::df_gradient(
      res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, 0, &multipliers, &extra));
  return ScGradient{.terms = std::move(terms),
                    .energies = corr.energies,
                    .zc = std::move(zc),
                    .multipliers = std::move(multipliers),
                    .asymmetry = asymmetry};
}

// W's symmetry, |W_MN - W_NM|, is held to this: the adjoint takes W as
// given, and E[W]'s Lagrangian is the one of sym(W) only when they agree.
constexpr double kWeightSymmetryTolerance = 1e-12;
// tr W within this of 1 (or of 0) is a mix mcscf::df_gradient takes as one
// (its own kMixSumTolerance): the core at weight 1 (or off).
constexpr double kTraceTolerance = 1e-12;

// What both QD forms check and build before anything else: the solution's
// weights, by mcscf::sa_gradient_weights (its own, recovered from its RDMs
// only when it carries none; every one positive: a root of zero weight has
// no SA CI row, and its multiplier's right-hand side divides by 2 w_r), and
// the active strings; and the frozen core, 0 to ncore, and PC's
// conditioning floor (EnergyInputs::pc_floor, unread by SC), which every
// adjoint's forward takes.
struct QdSetup {
  std::vector<double> weights;
  ci::DeviceStringSet alpha;
  ci::DeviceStringSet beta;
  int32_t nfrozen = 0;
  double pc_floor = 0.0;
};

Result<QdSetup> qd_setup(DeviceResources &res, const std::string_view who, const Molecule &molecule,
                         const BasisSet &basis, const mcscf::Solution &solution,
                         const int32_t ncore, const int32_t ncas, const int32_t nfrozen,
                         const double pc_floor = 0.0) {
  const SphericalMatrixAO &c = solution.c;
  const int64_t nao = basis.nao_sphe();
  if (c.n() != nao || ncore < 0 || ncas < 1 || ncore + ncas > nao) {
    return err_config("{}: C over {} orbitals, ncore {} and ncas {}, but the basis has {} "
                      "spherical AOs",
                      who, c.n(), ncore, ncas, nao);
  }
  if (nfrozen < 0 || nfrozen > ncore) {
    return err_config("{}: {} frozen core orbitals of {}: 0 to the core", who, nfrozen, ncore);
  }
  const auto n = static_cast<size_t>(ncas);
  const size_t nroots = solution.roots.energies.size();
  if (solution.gamma.size() != n * n || solution.gamma2.size() != n * n * n * n ||
      solution.root_gammas.size() != nroots * n * n) {
    return err_config("{}: RDMs of {}, {} and {} elements for {} roots and ncas {}", who,
                      solution.gamma.size(), solution.gamma2.size(), solution.root_gammas.size(),
                      nroots, ncas);
  }
  std::vector<double> weights = FBF_TRY(mcscf::sa_gradient_weights(solution, n));
  for (size_t r = 0; r < nroots; ++r) {
    if (!(weights[r] >= response::kRotationWeightTolerance)) {
      return err_config("{}: root {} has weight {:.3e}, and a root of zero weight has no "
                        "state-averaged CI condition to differentiate",
                        who, r, weights[r]);
    }
  }
  auto [alpha, beta] = FBF_TRY(active_strings(res, who, molecule, solution.gamma, ncas));
  const int64_t dim = int64_t{alpha.size()} * beta.size();
  if (solution.roots.vectors.num_elements() != static_cast<size_t>(dim) * nroots) {
    return err_config("{}: {} CI doubles for {} roots of {} determinants", who,
                      solution.roots.vectors.num_elements(), nroots, dim);
  }
  return QdSetup{.weights = std::move(weights),
                 .alpha = std::move(alpha),
                 .beta = std::move(beta),
                 .nfrozen = nfrozen,
                 .pc_floor = pc_floor};
}

Status qd_inputs(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                 const BasisSet &aux, const SphericalMatrixAO &h, const Buffers::device<double> &b,
                 const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                 const std::function<Status(const MoBlocks &, std::span<const EnergyInputs>)> &use,
                 const int32_t nfrozen) {
  constexpr std::string_view kWho = "nevpt2::qd_inputs";
  const QdSetup setup =
      FBF_TRY(qd_setup(res, kWho, molecule, basis, solution, ncore, ncas, nfrozen));
  const SphericalMatrixAO &c = solution.c;
  const auto nmo = static_cast<int32_t>(c.n());
  const ScopedDeviceIndex on{res.dev_idx()};
  return with_qd_inputs(res, kWho, setup.alpha, setup.beta, static_cast<int32_t>(aux.nao_sphe()),
                        nmo, nmo, b.data(), c.data(), h.data(), ncore, ncas, setup.weights,
                        solution.roots.vectors.data(), setup.nfrozen, use);
}

// qd_sc_coupling's fold callback, spelled once in this unit.
using Folder = std::function<Result<QdFold>(std::span<const double>, std::span<const double>)>;

// One W's share of the stacked QD Lagrangian, from its adjoint to its
// right-hand sides, kept until the solve's columns come back for the gate.
struct QdColumn {
  std::vector<double> w;
  std::vector<double> rho;
  double core_shift = 0.0;
  QdScAdjoint corr;
  ZcMultipliers zc;
  mcscf::RootPairMultipliers pairs;
  mcscf::ZetaTerms zeta;
  std::vector<double> gamma_rho;
  std::vector<double> gamma2_rho;
};

// The SC-QD Lagrangians (@p pc null) or the PC-QD ones (@p pc set) at the
// symmetric weight matrices @p ws (k of them), every piece from W and
// nothing from H_eff: one result per W, in order. What does not depend on W
// is built once: the adjoints' forward arrays and RDMs (qd_adjoints), the
// CASSCF integrals, the state-averaged Fock, the CI-relaxed Hessian and its
// one Z-vector solve over the k columns (lagrangian_zvector). Per W: z^c and
// Phi^SA, tau split into zeta and t, the right-hand sides, then after the
// solve the gate on its own column. A refused PC class is the adjoint's
// error, and PC's elements (the same at every W) go to *@p pc. @p who names
// the caller, and @p setup is qd_setup's. A @p fold is added to every W's
// Lagrangian (its callers pass one W). eigenvalues is left empty.
Result<std::vector<QdScLagrangian>>
qd_lagrangians(DeviceResources &res, const std::string_view who, const QdSetup &setup,
               const Molecule &molecule, const BasisSet &basis, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b,
               const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
               const std::span<const std::vector<double>> ws, const int32_t batch,
               const double tolerance, std::vector<PcClassEnergies> *pc,
               const QdFold *fold = nullptr,
               const std::optional<std::array<int32_t, 2>> limit_pair = std::nullopt) {
  const SphericalMatrixAO &c = solution.c;
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const auto nmo32 = static_cast<int32_t>(c.n());
  const auto n = static_cast<size_t>(ncas);
  const size_t nroots = solution.roots.energies.size();
  if (ws.empty()) {
    return err_config("{}: no W", who);
  }
  std::vector<double> stacked;
  for (const std::vector<double> &w : ws) {
    if (w.size() != nroots * nroots) {
      return err_config("{}: W holds {} elements for {} roots: n^2", who, w.size(), nroots);
    }
    for (size_t i = 0; i < nroots; ++i) {
      for (size_t j = i; j < nroots; ++j) {
        const double wij = w[(i * nroots) + j];
        const double wji = w[(j * nroots) + i];
        if (!std::isfinite(wij) || !std::isfinite(wji) ||
            !(std::abs(wij - wji) <= kWeightSymmetryTolerance)) {
          return err_config("{}: W is not symmetric: W_{}{} = {} and W_{}{} = {} (to {:g})", who, i,
                            j, wij, j, i, wji, kWeightSymmetryTolerance);
        }
      }
    }
    stacked.insert(stacked.end(), w.begin(), w.end());
  }
  if (fold != nullptr) {
    const size_t nmo = static_cast<size_t>(c.n());
    if (fold->gamma.size() != n * n || fold->gamma2.size() != n * n * n * n ||
        (!fold->a.empty() && fold->a.size() != nmo * nmo)) {
      return err_config("{}: the fold's gamma, gamma2 and a hold {}, {} and {} elements: ncas^2, "
                        "ncas^4 and nmo^2 (or none) for ncas {} and nmo {}",
                        who, fold->gamma.size(), fold->gamma2.size(), fold->a.size(), n, nmo);
    }
  }
  const std::vector<double> &weights = setup.weights;
  const ci::DeviceStringSet &alpha = setup.alpha;
  const ci::DeviceStringSet &beta = setup.beta;
  const ScopedDeviceIndex on{res.dev_idx()};
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  const double *const vectors = solution.roots.vectors.data();

  // y_bar, h_bar, Lambda, X and A of E^corr[W], every W over one set of
  // forward arrays.
  std::vector<QdScAdjoint> adjoints = FBF_TRY(
      qd_adjoints(res, who, alpha, beta, naux, nmo32, nmo32, b.data(), c.data(), h.data(), ncore,
                  ncas, weights, vectors, stacked, batch, pc, setup.nfrozen, setup.pc_floor));
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, naux, c, ncore, ncas));
  const response::Fock averaged =
      FBF_TRY(response::fock(res, molecule, h, b, naux, c, ncore, ncas, ints.ppaa, ints.papa,
                             solution.gamma, solution.gamma2));
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto packed = static_cast<int32_t>(averaged.gradient.num_elements());
  const size_t k = ws.size();
  Buffers::device<double> orbital_rhs(static_cast<size_t>(packed) * k, res.device());
  Buffers::device<double> ci_rhs(dim * nroots * k, res.device());
  std::vector<double> theta_rhs;
  std::optional<response::Fock> fock_zero;

  std::vector<QdColumn> columns;
  for (size_t col = 0; col < k; ++col) {
    // rho = diag W, the root energies' weighting, and tr W, the core's.
    std::vector<double> rho(nroots);
    for (size_t r = 0; r < nroots; ++r) {
      rho[r] = ws[col][r * (nroots + 1)];
    }
    const double trace = std::accumulate(rho.begin(), rho.end(), 0.0);
    const double core_shift = std::abs(trace - 1.0) <= kTraceTolerance ? 0.0 : trace - 1.0;

    // Phi^SA onto the adjoint.
    QdScAdjoint corr = std::move(adjoints[col]);
    ZcMultipliers zc =
        FBF_TRY(zc_multipliers(res, nmo32, ncore, ncas, corr.a.data(), corr.epsilon.data()));
    const ZcAdjoint phi =
        FBF_TRY(zc_adjoint(res, alpha, beta, naux, nmo32, nmo32, b.data(), c.data(), h.data(),
                           ncore, ncas, weights, vectors, zc.z.data()));
    FBF_TRY(add_adjoint(res, corr.h_bar, corr.lambda, corr.y, corr.x, corr.a, phi));
    // A fold's orbital source, onto A: the right-hand side's and the gate's.
    if (fold != nullptr && !fold->a.empty()) {
      const Buffers::device<double> fold_a = FBF_TRY(to_device(res, fold->a));
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(fold->a.size()),
                                                    &kOne<double>, fold_a.data(), 1, corr.a.data(),
                                                    1)));
      // fold_a is freed on scope exit, while the axpy reads it.
      FBF_TRY(res.sync());
    }

    // tau, split into zeta and t; zeta's two consumers.
    mcscf::RootPairMultipliers pairs = FBF_TRY(
        mcscf::root_pair_multipliers(res, solution.roots, weights, corr.y.data(), limit_pair));
    mcscf::ZetaTerms zeta = FBF_TRY(mcscf::zeta_terms(res, molecule, h, b, naux, solution, ncore,
                                                      ncas, ints.ppaa, ints.papa, rdm, pairs.zeta));

    // The RDMs at rho, sum_r rho_r (gamma_r, Gamma_r), root by root: rdm12
    // takes weights >= 0 summing to 1 alone, and rho may be neither.
    std::vector<double> gamma_rho(n * n, 0.0);
    std::vector<double> gamma2_rho(n * n * n * n, 0.0);
    {
      Buffers::device<double> rdms_d((n * n) + (n * n * n * n), res.device());
      std::vector<double> one(nroots, 0.0);
      for (size_t r = 0; r < nroots; ++r) {
        if (rho[r] == 0.0) {
          continue;
        }
        std::ranges::fill(one, 0.0);
        one[r] = 1.0;
        FBF_TRY(rdm.rdm12(res, one, vectors, rdms_d.data(), rdms_d.data() + (n * n)));
        const std::vector<double> g1 = FBF_TRY(to_host(res, rdms_d.data(), n * n));
        const std::vector<double> g2 =
            FBF_TRY(to_host(res, rdms_d.data() + (n * n), n * n * n * n));
        for (size_t i = 0; i < g1.size(); ++i) {
          gamma_rho[i] += rho[r] * g1[i];
        }
        for (size_t i = 0; i < g2.size(); ++i) {
          gamma2_rho[i] += rho[r] * g2[i];
        }
      }
    }

    // A fold's RDMs, with rho's: g^rho is affine in them, so they add their
    // transition orbital gradient, the core off, and the gate sees them too.
    if (fold != nullptr) {
      for (size_t i = 0; i < gamma_rho.size(); ++i) {
        gamma_rho[i] += fold->gamma[i];
      }
      for (size_t i = 0; i < gamma2_rho.size(); ++i) {
        gamma2_rho[i] += fold->gamma2[i];
      }
    }

    // The right-hand sides, this W's column: g^rho + zeta . g + A over the
    // packed pairs, y_bar_r / (2 w_r) (zvector's CI rows are the SA
    // Hessian's over 2 w_r), and t. g^rho = sum_r rho_r g_r: the Fock's
    // gradient at rho's RDMs, which is affine in them, carries its zero-RDM
    // (core) part once, and tr W times is wanted, so core_shift more of it.
    const response::Fock fock_rho = FBF_TRY(response::fock(
        res, molecule, h, b, naux, c, ncore, ncas, ints.ppaa, ints.papa, gamma_rho, gamma2_rho));
    double *const orbital_col = orbital_rhs.data() + (col * static_cast<size_t>(packed));
    FBF_TRY(response::pack_orbital_pairs(res, nmo32, ncore, ncas, corr.a.data(), orbital_col));
    for (const double *const add :
         std::array<const double *, 2>{fock_rho.gradient.data(), zeta.orbital.data()}) {
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, packed, &kOne<double>, add, 1, orbital_col, 1)));
    }
    if (core_shift != 0.0) {
      if (!fock_zero) {
        fock_zero.emplace(FBF_TRY(response::fock(
            res, molecule, h, b, naux, c, ncore, ncas, ints.ppaa, ints.papa,
            std::vector<double>(n * n, 0.0), std::vector<double>(n * n * n * n, 0.0))));
      }
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(
          blas, packed, &core_shift, fock_zero->gradient.data(), 1, orbital_col, 1)));
    }
    double *const ci_col = ci_rhs.data() + (col * dim * nroots);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ci_col, corr.y.data(), dim * nroots * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    for (size_t r = 0; r < nroots; ++r) {
      const double scale = 1.0 / (2.0 * weights[r]);
      FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, static_cast<int32_t>(dim), &scale,
                                                    ci_col + (r * dim), 1)));
    }
    theta_rhs.insert(theta_rhs.end(), pairs.theta_rhs.begin(), pairs.theta_rhs.end());
    // fock_rho is freed at the end of the iteration, while the axpy reads it.
    FBF_TRY(res.sync());
    columns.push_back(QdColumn{.w = ws[col],
                               .rho = std::move(rho),
                               .core_shift = core_shift,
                               .corr = std::move(corr),
                               .zc = std::move(zc),
                               .pairs = std::move(pairs),
                               .zeta = std::move(zeta),
                               .gamma_rho = std::move(gamma_rho),
                               .gamma2_rho = std::move(gamma2_rho)});
  }

  std::vector<response::ZVector> multipliers = FBF_TRY(lagrangian_zvector(
      res, who, h, b, naux, solution, ncore, ncas, alpha, beta, rdm, ints, averaged, weights,
      static_cast<int32_t>(k), orbital_rhs.data(), ci_rhs.data(), theta_rhs, tolerance));

  std::vector<QdScLagrangian> out;
  for (size_t col = 0; col < k; ++col) {
    QdColumn &x = columns[col];
    // The gate, its CAS part at the rho-weighted RDMs plus zeta's, the core
    // at tr W.
    for (size_t i = 0; i < x.gamma_rho.size(); ++i) {
      x.gamma_rho[i] += x.zeta.gamma[i];
    }
    for (size_t i = 0; i < x.gamma2_rho.size(); ++i) {
      x.gamma2_rho[i] += x.zeta.gamma2[i];
    }
    // A fold's RDMs are no state's: their own active-active asymmetry,
    // 2 (F - F^T) at them with the core off, is the CSF term's (the fold's
    // caller adds it outside the Z-vector), and no multiplier removes it. So
    // the gate's source takes it out of that block alone; the packed pairs,
    // which the Z-vector solved for, keep everything.
    std::optional<Buffers::device<double>> gate_a;
    if (fold != nullptr) {
      const auto nmo = static_cast<size_t>(c.n());
      const std::vector<double> f_fold = FBF_TRY(lagrangian_fock(
          res, molecule, basis, h, b, naux, c, ncore, ncas, fold->gamma, fold->gamma2));
      const std::vector<double> f_zero = FBF_TRY(lagrangian_fock(
          res, molecule, basis, h, b, naux, c, ncore, ncas, std::vector<double>(n * n, 0.0),
          std::vector<double>(n * n * n * n, 0.0)));
      std::vector<double> a_host = FBF_TRY(to_host(res, x.corr.a.data(), nmo * nmo));
      const auto first = static_cast<size_t>(ncore);
      for (size_t q = first; q < first + n; ++q) {
        for (size_t p = first; p < first + n; ++p) {
          const size_t pq = p + (nmo * q);
          const size_t qp = q + (nmo * p);
          a_host[pq] -= 2.0 * ((f_fold[pq] - f_zero[pq]) - (f_fold[qp] - f_zero[qp]));
        }
      }
      gate_a.emplace(FBF_TRY(to_device(res, a_host)));
    }
    const std::array<double, 6> asymmetry = FBF_TRY(
        lagrangian_asymmetry(res, molecule, basis, h, b, naux, solution, ncore, ncas, rdm, weights,
                             std::move(x.gamma_rho), std::move(x.gamma2_rho), multipliers[col],
                             gate_a ? gate_a->data() : x.corr.a.data(), x.core_shift));
    FBF_TRY(check_asymmetry(who, asymmetry, multipliers[col]));
    // A fold's Hellmann-Feynman term rides zeta's transition pass, after the
    // gate (which had it through gamma_rho).
    if (fold != nullptr) {
      for (size_t i = 0; i < fold->gamma.size(); ++i) {
        x.zeta.gamma[i] += fold->gamma[i];
      }
      for (size_t i = 0; i < fold->gamma2.size(); ++i) {
        x.zeta.gamma2[i] += fold->gamma2[i];
      }
    }
    out.push_back(QdScLagrangian{.eigenvalues = {},
                                 .w = std::move(x.w),
                                 .rho = std::move(x.rho),
                                 .weights = weights,
                                 .corr = std::move(x.corr),
                                 .zc = std::move(x.zc),
                                 .pairs = std::move(x.pairs),
                                 .zeta = std::move(x.zeta),
                                 .multipliers = std::move(multipliers[col]),
                                 .asymmetry = asymmetry});
  }
  return out;
}

// qd_lagrangians at the one W @p w.
Result<QdScLagrangian>
qd_lagrangian(DeviceResources &res, const std::string_view who, const QdSetup &setup,
              const Molecule &molecule, const BasisSet &basis, const BasisSet &aux,
              const SphericalMatrixAO &h, const Buffers::device<double> &b,
              const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
              std::vector<double> w, const int32_t batch, const double tolerance,
              std::vector<PcClassEnergies> *pc, const QdFold *fold = nullptr,
              const std::optional<std::array<int32_t, 2>> limit_pair = std::nullopt) {
  const std::array<std::vector<double>, 1> ws{std::move(w)};
  std::vector<QdScLagrangian> one =
      FBF_TRY(qd_lagrangians(res, who, setup, molecule, basis, aux, h, b, solution, ncore, ncas, ws,
                             batch, tolerance, pc, fold, limit_pair));
  return std::move(one.front());
}

// The QD Lagrangian between states @p p and @p q (H_eff's eigenvalues
// ascending): H_eff = diag(E_r) + 1/2 (C + C^T) over qd_sc_energy's elements
// (@p pc null) or qd_pc_energy's (@p pc set; a refused class is its error,
// returned before any adjoint is built), its eigenvectors R (each signed so
// that its largest-magnitude component is positive, pipeline/nevpt2's
// convention), and qd_lagrangian at W = 1/2 (R_P R_Q^T + R_Q R_P^T), with
// H_eff's eigenvalues and R. At p = q that is state K's u_K u_K^T, bit for
// bit (1/2 (x + x) = x), refused within kRootGapFloor of any other state;
// at p != q it is the interstate coupling's W^PQ, trace 0, refused when E_P
// and E_Q are that close (h_PQ / (E_Q - E_P) is the consumer's). PC's energy
// and adjoint read @p pc_floor.
Result<QdScLagrangian>
qd_state_lagrangian(DeviceResources &res, const std::string_view who, const Molecule &molecule,
                    const BasisSet &basis, const BasisSet &aux, const SphericalMatrixAO &h,
                    const Buffers::device<double> &b, const mcscf::Solution &solution,
                    const int32_t ncore, const int32_t ncas, const int32_t p, const int32_t q,
                    const int32_t batch, const int32_t nfrozen, std::vector<PcClassEnergies> *pc,
                    const Folder &fold = {}, const double pc_floor = 0.0) {
  const QdSetup setup =
      FBF_TRY(qd_setup(res, who, molecule, basis, solution, ncore, ncas, nfrozen, pc_floor));
  const size_t nroots = solution.roots.energies.size();
  for (const int32_t state : {p, q}) {
    if (state < 0 || static_cast<size_t>(state) >= nroots) {
      return err_config("{}: state {} of {} roots", who, state, nroots);
    }
  }
  const SphericalMatrixAO &c = solution.c;
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const auto nmo32 = static_cast<int32_t>(c.n());
  const ScopedDeviceIndex on{res.dev_idx()};
  const double *const vectors = solution.roots.vectors.data();

  // H_eff = diag(E_r) + 1/2 (C + C^T), its eigenvectors, and W = u_K u_K^T.
  std::vector<double> totals;
  if (pc == nullptr) {
    const std::vector<ClassEnergies> elements =
        FBF_TRY(qd_sc_energy(res, setup.alpha, setup.beta, naux, nmo32, nmo32, b.data(), c.data(),
                             h.data(), ncore, ncas, setup.weights, vectors, batch, nfrozen));
    std::ranges::transform(elements, std::back_inserter(totals), &ClassEnergies::total);
  } else {
    // A refused class is the energy's error, returned before anything reads it.
    const std::vector<PcClassEnergies> elements = FBF_TRY(
        qd_pc_energy(res, setup.alpha, setup.beta, naux, nmo32, nmo32, b.data(), c.data(), h.data(),
                     ncore, ncas, setup.weights, vectors, batch, nfrozen, setup.pc_floor));
    std::ranges::transform(elements, std::back_inserter(totals), &PcClassEnergies::total);
  }
  std::vector<double> heff(nroots * nroots);
  for (size_t i = 0; i < nroots; ++i) {
    for (size_t j = 0; j < nroots; ++j) {
      heff[i + (nroots * j)] = (i == j ? solution.roots.energies[i] : 0.0) +
                               (0.5 * (totals[(i * nroots) + j] + totals[(j * nroots) + i]));
    }
  }
  std::vector<double> eigenvalues(nroots);
  {
    Buffers::device<double> heff_d = FBF_TRY(to_device(res, heff));
    Buffers::device<double> values_d(nroots, res.device());
    FBF_TRY(diagonalize<double>(res, static_cast<int32_t>(nroots), heff_d.data(), values_d.data()));
    heff = FBF_TRY(to_host(res, heff_d.data(), nroots * nroots));
    eigenvalues = FBF_TRY(to_host(res, values_d.data(), nroots));
  }
  const auto k = static_cast<size_t>(p);
  const auto m = static_cast<size_t>(q);
  if (k == m) {
    for (size_t l = 0; l < nroots; ++l) {
      if (l != k && std::abs(eigenvalues[k] - eigenvalues[l]) < mcscf::kRootGapFloor) {
        return err_numerical("{}: QD states {} and {} are {:.3e} Eh apart, under {:g}: the "
                             "state's eigenvector has no derivative at a degeneracy",
                             who, std::min(k, l), std::max(k, l),
                             std::abs(eigenvalues[k] - eigenvalues[l]), mcscf::kRootGapFloor);
      }
    }
  } else if (std::abs(eigenvalues[k] - eigenvalues[m]) < mcscf::kRootGapFloor) {
    return err_numerical("{}: QD states {} and {} are {:.3e} Eh apart, under {:g}: their "
                         "coupling h_PQ / (E_Q - E_P) is not defined at a degeneracy",
                         who, std::min(k, m), std::max(k, m),
                         std::abs(eigenvalues[k] - eigenvalues[m]), mcscf::kRootGapFloor);
  }
  for (size_t col = 0; col < nroots; ++col) {
    const std::span<double> v{heff.data() + (col * nroots), nroots};
    if (*std::ranges::max_element(v, {}, [](const double x) { return std::abs(x); }) < 0.0) {
      std::ranges::transform(v, v.begin(), std::negate<>{});
    }
  }
  const std::span<const double> u_p{heff.data() + (k * nroots), nroots};
  const std::span<const double> u_q{heff.data() + (m * nroots), nroots};
  std::vector<double> w(nroots * nroots);
  for (size_t i = 0; i < nroots; ++i) {
    for (size_t j = 0; j < nroots; ++j) {
      w[(i * nroots) + j] = 0.5 * ((u_p[i] * u_q[j]) + (u_q[i] * u_p[j]));
    }
  }
  std::optional<QdFold> folded;
  if (fold) {
    folded.emplace(FBF_TRY(fold(eigenvalues, heff)));
  }
  QdScLagrangian lagrangian = FBF_TRY(
      qd_lagrangian(res, who, setup, molecule, basis, aux, h, b, solution, ncore, ncas,
                    std::move(w), batch, kZvectorTolerance, pc, folded ? &*folded : nullptr));
  lagrangian.eigenvalues = std::move(eigenvalues);
  lagrangian.eigenvectors = std::move(heff);
  return lagrangian;
}

Result<QdScLagrangian>
qd_sc_lagrangian(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                 const BasisSet &aux, const SphericalMatrixAO &h, const Buffers::device<double> &b,
                 const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                 const int32_t state, const int32_t batch, const int32_t nfrozen) {
  return qd_state_lagrangian(res, "nevpt2::qd_sc_lagrangian", molecule, basis, aux, h, b, solution,
                             ncore, ncas, state, state, batch, nfrozen, nullptr);
}

Result<QdScLagrangian>
qd_sc_lagrangian(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                 const BasisSet &aux, const SphericalMatrixAO &h, const Buffers::device<double> &b,
                 const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                 const std::span<const double> w, const int32_t batch, const int32_t nfrozen) {
  constexpr std::string_view kWho = "nevpt2::qd_sc_lagrangian";
  const QdSetup setup =
      FBF_TRY(qd_setup(res, kWho, molecule, basis, solution, ncore, ncas, nfrozen));
  return qd_lagrangian(res, kWho, setup, molecule, basis, aux, h, b, solution, ncore, ncas,
                       std::vector<double>(w.begin(), w.end()), batch, kZvectorTolerance, nullptr);
}

// @p into's terms plus @p scale times @p from's, field by field.
void add_terms(mcscf::GradientTerms &into, const mcscf::GradientTerms &from, const double scale) {
  for (const auto field :
       {&mcscf::GradientTerms::kinetic, &mcscf::GradientTerms::potential,
        &mcscf::GradientTerms::overlap, &mcscf::GradientTerms::two_electron,
        &mcscf::GradientTerms::nuclear_repulsion, &mcscf::GradientTerms::total}) {
    for (size_t i = 0; i < (into.*field).size(); ++i) {
      (into.*field)[i] += scale * (from.*field)[i];
    }
  }
}

// The QD gradient's two derivative passes over @p lagrangian (SC's or PC's,
// state or W form, the same code), and the CASSCF roots' gap it reports
// (heff_gap is left infinite).
Result<QdScGradient> qd_passes(DeviceResources &res, const Molecule &molecule,
                               const BasisSet &basis,
                               const std::span<const PrimitivePairClass> classes,
                               const BasisSet &aux, const SphericalMatrixAO &h,
                               const Buffers::device<double> &b, const double *l,
                               const mcscf::Solution &solution, const int32_t ncore,
                               const int32_t ncas, QdScLagrangian lagrangian) {
  // E_cas at rho, (Z, z, Theta), and E^corr[W] + Phi^SA's arrays. df_gradient
  // takes a mix of weights >= 0 summing to 1, or of any sign summing to 0
  // (the core off). Any other rho is tr W p + d: p rho's positive part
  // summing to 1 (uniform when it has none), at tr W and with zero
  // multipliers (no response); d = rho - tr W p, summing to 0, with the
  // multipliers and the arrays, so that each is counted once.
  const std::vector<double> &rho = lagrangian.rho;
  const double trace = std::accumulate(rho.begin(), rho.end(), 0.0);
  const mcscf::MoAdjoint extra{.h_bar = lagrangian.corr.h_bar.data(),
                               .lambda = lagrangian.corr.lambda.data(),
                               .x = lagrangian.corr.x.data()};
  const bool one_mix = std::abs(trace) <= kTraceTolerance ||
                       (std::abs(trace - 1.0) <= kTraceTolerance &&
                        std::ranges::all_of(rho, [](const double v) { return v >= 0.0; }));
  mcscf::GradientTerms terms;
  if (one_mix) {
    terms = FBF_TRY(mcscf::df_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                       ncas, rho, &lagrangian.multipliers, &extra));
  } else {
    std::vector<double> p(rho.size());
    std::ranges::transform(rho, p.begin(), [](const double v) { return std::max(v, 0.0); });
    const double positive = std::accumulate(p.begin(), p.end(), 0.0);
    for (double &v : p) {
      v = positive > 0.0 ? v / positive : 1.0 / static_cast<double>(p.size());
    }
    std::vector<double> d(rho.size());
    for (size_t r = 0; r < rho.size(); ++r) {
      d[r] = rho[r] - (trace * p[r]);
    }
    const response::ZVector none{
        .orbital = FBF_TRY(
            zeroed(res, static_cast<int64_t>(lagrangian.multipliers.orbital.num_elements()))),
        .ci = FBF_TRY(zeroed(res, static_cast<int64_t>(lagrangian.multipliers.ci.num_elements()))),
        .theta = {},
        .iterations = 0,
        .residual = 0.0};
    const mcscf::GradientTerms core = FBF_TRY(mcscf::df_gradient(
        res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, p, &none, nullptr));
    terms = FBF_TRY(mcscf::df_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                       ncas, d, &lagrangian.multipliers, &extra));
    add_terms(terms, core, trace);
  }

  // zeta's sum_{r<s} zeta_sr <c_s|H|c_r>, Hellmann-Feynman at the folded RDMs.
  double max_zeta = 0.0;
  for (const double z : lagrangian.pairs.zeta) {
    max_zeta = std::max(max_zeta, std::abs(z));
  }
  const size_t count = terms.total.size();
  mcscf::GradientTerms transition{.kinetic = std::vector<double>(count, 0.0),
                                  .potential = std::vector<double>(count, 0.0),
                                  .overlap = std::vector<double>(count, 0.0),
                                  .two_electron = std::vector<double>(count, 0.0),
                                  .nuclear_repulsion = std::vector<double>(count, 0.0),
                                  .total = std::vector<double>(count, 0.0)};
  // A fold's RDMs ride zeta's (qd_lagrangian), so the pass runs for them too.
  const bool folded =
      std::ranges::any_of(lagrangian.zeta.gamma, [](const double v) { return v != 0.0; }) ||
      std::ranges::any_of(lagrangian.zeta.gamma2, [](const double v) { return v != 0.0; });
  if (max_zeta > 0.0 || folded) {
    transition = FBF_TRY(mcscf::df_transition_gradient(res, molecule, basis, classes, aux, h, b, l,
                                                       solution, ncore, ncas, lagrangian.zeta.gamma,
                                                       lagrangian.zeta.gamma2));
  }
  std::vector<double> total = terms.total;
  for (size_t i = 0; i < count; ++i) {
    total[i] += transition.total[i];
  }

  const std::vector<double> &energies = solution.roots.energies;
  double root_gap = std::numeric_limits<double>::infinity();
  for (size_t r = 0; r < energies.size(); ++r) {
    for (size_t s = r + 1; s < energies.size(); ++s) {
      root_gap = std::min(root_gap, std::abs(energies[r] - energies[s]));
    }
  }
  return QdScGradient{.total = std::move(total),
                      .terms = std::move(terms),
                      .transition = std::move(transition),
                      .lagrangian = std::move(lagrangian),
                      .max_zeta = max_zeta,
                      .root_gap = root_gap,
                      .heff_gap = std::numeric_limits<double>::infinity()};
}

// The smallest |lambda_K - lambda_L| of H_eff's @p eigenvalues, L != @p k.
double smallest_gap(const std::span<const double> eigenvalues, const size_t k) {
  double gap = std::numeric_limits<double>::infinity();
  for (size_t m = 0; m < eigenvalues.size(); ++m) {
    if (m != k) {
      gap = std::min(gap, std::abs(eigenvalues[k] - eigenvalues[m]));
    }
  }
  return gap;
}

Result<QdScGradient>
qd_sc_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
               const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
               const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
               const int32_t state, const int32_t batch, const int32_t nfrozen) {
  QdScLagrangian lagrangian = FBF_TRY(qd_sc_lagrangian(res, molecule, basis, aux, h, b, solution,
                                                       ncore, ncas, state, batch, nfrozen));
  QdScGradient g = FBF_TRY(qd_passes(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                     ncas, std::move(lagrangian)));
  g.heff_gap = smallest_gap(g.lagrangian.eigenvalues, static_cast<size_t>(state));
  return g;
}

Result<QdScGradient>
qd_sc_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
               const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
               const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
               const std::span<const double> w, const int32_t batch, const double zvector_tolerance,
               const int32_t nfrozen) {
  // qd_sc_lagrangian's W form, with the tolerance: its errors, named for it.
  constexpr std::string_view kWho = "nevpt2::qd_sc_lagrangian";
  const QdSetup setup =
      FBF_TRY(qd_setup(res, kWho, molecule, basis, solution, ncore, ncas, nfrozen));
  QdScLagrangian lagrangian = FBF_TRY(
      qd_lagrangian(res, kWho, setup, molecule, basis, aux, h, b, solution, ncore, ncas,
                    std::vector<double>(w.begin(), w.end()), batch, zvector_tolerance, nullptr));
  return qd_passes(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                   std::move(lagrangian));
}

Result<std::vector<QdScGradient>>
qd_sc_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
               const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
               const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
               const std::span<const std::vector<double>> ws, const int32_t batch,
               const double zvector_tolerance, const int32_t nfrozen) {
  constexpr std::string_view kWho = "nevpt2::qd_sc_gradient";
  const QdSetup setup =
      FBF_TRY(qd_setup(res, kWho, molecule, basis, solution, ncore, ncas, nfrozen));
  std::vector<QdScLagrangian> lagrangians =
      FBF_TRY(qd_lagrangians(res, kWho, setup, molecule, basis, aux, h, b, solution, ncore, ncas,
                             ws, batch, zvector_tolerance, nullptr));
  std::vector<QdScGradient> out;
  for (QdScLagrangian &lagrangian : lagrangians) {
    out.push_back(FBF_TRY(qd_passes(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                    ncas, std::move(lagrangian))));
  }
  return out;
}

Result<QdScGradient>
qd_sc_coupling(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
               const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
               const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
               const int32_t p, const int32_t q, const int32_t batch, const Folder &fold) {
  constexpr std::string_view kWho = "nevpt2::qd_sc_coupling";
  if (p == q) {
    return err_config("{}: states {} and {}: a coupling is between two states, and one state's "
                      "is its gradient (qd_sc_gradient)",
                      kWho, p, q);
  }
  QdScLagrangian lagrangian = FBF_TRY(qd_state_lagrangian(
      res, kWho, molecule, basis, aux, h, b, solution, ncore, ncas, p, q, batch, 0, nullptr, fold));
  QdScGradient g = FBF_TRY(qd_passes(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                     ncas, std::move(lagrangian)));
  g.heff_gap = std::abs(g.lagrangian.eigenvalues[static_cast<size_t>(q)] -
                        g.lagrangian.eigenvalues[static_cast<size_t>(p)]);
  return g;
}

// @p g's fields as QdPcGradient, PC's @p elements beside them, heff_gap
// @p gap.
QdPcGradient pc_result(QdScGradient g, std::vector<PcClassEnergies> elements, const double gap) {
  return QdPcGradient{.total = std::move(g.total),
                      .terms = std::move(g.terms),
                      .transition = std::move(g.transition),
                      .lagrangian = std::move(g.lagrangian),
                      .elements = std::move(elements),
                      .max_zeta = g.max_zeta,
                      .root_gap = g.root_gap,
                      .heff_gap = gap};
}

Result<QdPcGradient> qd_pc_gradient(DeviceResources &res, const Molecule &molecule,
                                    const BasisSet &basis,
                                    const std::span<const PrimitivePairClass> classes,
                                    const BasisSet &aux, const SphericalMatrixAO &h,
                                    const Buffers::device<double> &b, const double *l,
                                    const mcscf::Solution &solution, const int32_t ncore,
                                    const int32_t ncas, const int32_t state, const int32_t batch,
                                    const int32_t nfrozen, const double pc_floor) {
  // A refused class is qd_pc_energy's error, here and so the gradient's.
  std::vector<PcClassEnergies> elements;
  QdScLagrangian lagrangian = FBF_TRY(
      qd_state_lagrangian(res, "nevpt2::qd_pc_gradient", molecule, basis, aux, h, b, solution,
                          ncore, ncas, state, state, batch, nfrozen, &elements, {}, pc_floor));
  QdScGradient g = FBF_TRY(qd_passes(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                     ncas, std::move(lagrangian)));
  const double gap = smallest_gap(g.lagrangian.eigenvalues, static_cast<size_t>(state));
  return pc_result(std::move(g), std::move(elements), gap);
}

Result<QdPcGradient>
qd_pc_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
               const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
               const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
               const std::span<const double> w, const int32_t batch, const double zvector_tolerance,
               const int32_t nfrozen, const std::optional<std::array<int32_t, 2>> limit_pair,
               const double pc_floor) {
  constexpr std::string_view kWho = "nevpt2::qd_pc_gradient";
  const QdSetup setup =
      FBF_TRY(qd_setup(res, kWho, molecule, basis, solution, ncore, ncas, nfrozen, pc_floor));
  // A refused class is qd_pc_adjoint's error, the same as qd_pc_energy's.
  std::vector<PcClassEnergies> elements;
  QdScLagrangian lagrangian =
      FBF_TRY(qd_lagrangian(res, kWho, setup, molecule, basis, aux, h, b, solution, ncore, ncas,
                            std::vector<double>(w.begin(), w.end()), batch, zvector_tolerance,
                            &elements, nullptr, limit_pair));
  QdScGradient g = FBF_TRY(qd_passes(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                     ncas, std::move(lagrangian)));
  return pc_result(std::move(g), std::move(elements), std::numeric_limits<double>::infinity());
}

Result<std::vector<QdPcGradient>>
qd_pc_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
               const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
               const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
               const std::span<const std::vector<double>> ws, const int32_t batch,
               const double zvector_tolerance, const int32_t nfrozen, const double pc_floor) {
  constexpr std::string_view kWho = "nevpt2::qd_pc_gradient";
  const QdSetup setup =
      FBF_TRY(qd_setup(res, kWho, molecule, basis, solution, ncore, ncas, nfrozen, pc_floor));
  // A refused class is qd_adjoints' error, the same as qd_pc_energy's.
  std::vector<PcClassEnergies> elements;
  std::vector<QdScLagrangian> lagrangians =
      FBF_TRY(qd_lagrangians(res, kWho, setup, molecule, basis, aux, h, b, solution, ncore, ncas,
                             ws, batch, zvector_tolerance, &elements));
  std::vector<QdPcGradient> out;
  for (QdScLagrangian &lagrangian : lagrangians) {
    QdScGradient g = FBF_TRY(qd_passes(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                       ncas, std::move(lagrangian)));
    out.push_back(pc_result(std::move(g), elements, std::numeric_limits<double>::infinity()));
  }
  return out;
}

Result<QdPcGradient>
qd_pc_coupling(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
               const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
               const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
               const int32_t p, const int32_t q, const int32_t batch, const Folder &fold,
               const double pc_floor) {
  constexpr std::string_view kWho = "nevpt2::qd_pc_coupling";
  if (p == q) {
    return err_config("{}: states {} and {}: a coupling is between two states, and one state's "
                      "is its gradient (qd_pc_gradient)",
                      kWho, p, q);
  }
  // A refused class is qd_pc_energy's error, here and so the coupling's.
  std::vector<PcClassEnergies> elements;
  QdScLagrangian lagrangian =
      FBF_TRY(qd_state_lagrangian(res, kWho, molecule, basis, aux, h, b, solution, ncore, ncas, p,
                                  q, batch, 0, &elements, fold, pc_floor));
  QdScGradient g = FBF_TRY(qd_passes(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                     ncas, std::move(lagrangian)));
  const double gap = std::abs(g.lagrangian.eigenvalues[static_cast<size_t>(q)] -
                              g.lagrangian.eigenvalues[static_cast<size_t>(p)]);
  return pc_result(std::move(g), std::move(elements), gap);
}

// The root's string sets, after the checks sc_gradient and pc_gradient
// make of @p solution: one root (else Unsupported, naming @p method's
// gradient), and gamma of ncas^2 elements. @p who names the caller.
Result<std::pair<ci::DeviceStringSet, ci::DeviceStringSet>>
root_strings(DeviceResources &res, const std::string_view who, const std::string_view method,
             const Molecule &molecule, const mcscf::Solution &solution, const int32_t ncas) {
  const size_t nroots = solution.roots.energies.size();
  if (nroots != 1) {
    return err_unsupported("{}: the solution averages {} roots, and the "
                           "{}-NEVPT2 gradient is of a single-state CASSCF only",
                           who, nroots, method);
  }
  const auto n = static_cast<size_t>(std::max(ncas, 0));
  if (solution.gamma.size() != n * n) {
    return err_config("{}: gamma of {} elements for ncas {}", who, solution.gamma.size(), ncas);
  }
  double trace = 0.0;
  for (size_t t = 0; t < n; ++t) {
    trace += solution.gamma[t * (n + 1)];
  }
  const auto nelecas = static_cast<int32_t>(std::lround(trace));
  const int32_t n_beta = (nelecas - (molecule.multiplicity() - 1)) / 2;
  const ScopedDeviceIndex on{res.dev_idx()};
  ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nelecas - n_beta));
  ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, n_beta));
  return std::pair{std::move(alpha), std::move(beta)};
}

Result<ScGradient>
sc_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
            const int32_t batch, const int32_t nfrozen, const double zvector_tolerance) {
  const auto [alpha, beta] =
      FBF_TRY(root_strings(res, "nevpt2::sc_gradient", "SC", molecule, solution, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const auto nmo = static_cast<int32_t>(solution.c.n());
  const double *const c = solution.c.data();
  const double *const vector = solution.roots.vectors.data();
  ScAdjoint corr = FBF_TRY(sc_adjoint(res, alpha, beta, naux, nmo, nmo, b.data(), c, h.data(),
                                      ncore, ncas, vector, batch, nfrozen));
  ZcMultipliers zc =
      FBF_TRY(zc_multipliers(res, nmo, ncore, ncas, corr.a.data(), corr.epsilon.data()));
  const ZcAdjoint phi = FBF_TRY(zc_adjoint(res, alpha, beta, naux, nmo, nmo, b.data(), c, h.data(),
                                           ncore, ncas, std::array{1.0}, vector, zc.z.data()));
  return lagrangian_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                             std::move(corr), std::move(zc), phi, zvector_tolerance);
}

Result<PcGradient> pc_gradient(DeviceResources &res, const Molecule &molecule,
                               const BasisSet &basis,
                               const std::span<const PrimitivePairClass> classes,
                               const BasisSet &aux, const SphericalMatrixAO &h,
                               const Buffers::device<double> &b, const double *l,
                               const mcscf::Solution &solution, const int32_t ncore,
                               const int32_t ncas, const int32_t batch, const int32_t nfrozen,
                               const double pc_floor, const double zvector_tolerance) {
  const auto [alpha, beta] =
      FBF_TRY(root_strings(res, "nevpt2::pc_gradient", "PC", molecule, solution, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const auto nmo = static_cast<int32_t>(solution.c.n());
  const double *const c = solution.c.data();
  const double *const vector = solution.roots.vectors.data();
  // A refused class is the energy's error, here and so the gradient's.
  PcAdjoint corr = FBF_TRY(pc_adjoint(res, alpha, beta, naux, nmo, nmo, b.data(), c, h.data(),
                                      ncore, ncas, vector, batch, nfrozen, pc_floor));
  // PC is invariant to an inactive rotation only between degenerate orbital
  // energies, so its z^c is the same division as SC's, over its own A.
  ZcMultipliers zc =
      FBF_TRY(zc_multipliers(res, nmo, ncore, ncas, corr.a.data(), corr.epsilon.data()));
  const ZcAdjoint phi = FBF_TRY(zc_adjoint(res, alpha, beta, naux, nmo, nmo, b.data(), c, h.data(),
                                           ncore, ncas, std::array{1.0}, vector, zc.z.data()));
  ScGradient g = FBF_TRY(lagrangian_gradient(res, molecule, basis, classes, aux, h, b, l, solution,
                                             ncore, ncas,
                                             ScAdjoint{.energies = {},
                                                       .h_bar = std::move(corr.h_bar),
                                                       .lambda = std::move(corr.lambda),
                                                       .y = std::move(corr.y),
                                                       .x = std::move(corr.x),
                                                       .a = std::move(corr.a),
                                                       .epsilon = std::move(corr.epsilon)},
                                             std::move(zc), phi, zvector_tolerance));
  return PcGradient{.terms = std::move(g.terms),
                    .energies = corr.energies,
                    .zc = std::move(g.zc),
                    .multipliers = std::move(g.multipliers),
                    .asymmetry = g.asymmetry};
}

} // namespace free_breakfast::nevpt2
