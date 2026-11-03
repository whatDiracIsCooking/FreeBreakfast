/// @file
/// @brief lagrangian_gradient, sc_gradient and pc_gradient: the Z-vector of
/// the SC- or PC-NEVPT2 Lagrangian on a single-state DF-CASSCF over the CI-relaxed
/// Hessian, the gate that its orbital source term is symmetric in every
/// block, and the nuclear term as one mcscf::df_gradient pass over the
/// multipliers and the correlation's MO adjoint. And qd_sc_lagrangian: the
/// SC-QD-NEVPT2 Lagrangian's multipliers on a DF-SA-CASSCF, through the same
/// Z-vector and gate at the average's weights; qd_sc_gradient, its two
/// derivative passes (df_gradient at the mix rho, df_transition_gradient at
/// zeta's folded RDMs).

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
import free_breakfast.mcscf.relaxed_hessian;
import free_breakfast.mcscf.zvector;
import free_breakfast.mcscf.root_rotation; // kRootGapFloor
import free_breakfast.mcscf.gradient;
import :operands; // to_host, to_device

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// The Lagrangian's orbital source term X^L must be symmetric in every block
// at the solution (nuclear_gradient.tex, "A free, exact check of the whole
// solve"). Its antisymmetric part is the Z-vector's residual on the
// non-redundant pairs (to its 1e-10), z^c's division on the inactive ones
// (round-off), and on the active-active ones the CI equation's residual
// (the CI response's conjugate gradients, 1e-8 of their right-hand side,
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
// the five-point stencil's, exactly.
Result<std::array<double, 6>>
lagrangian_asymmetry(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                     const SphericalMatrixAO &h, const Buffers::device<double> &b,
                     const int32_t naux, const mcscf::Solution &solution, const int32_t ncore,
                     const int32_t ncas, ci::Rdm &rdm, const std::span<const double> weights,
                     std::vector<double> gamma, std::vector<double> gamma2, const mcscf::ZVector &z,
                     const double *a) {
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
  const std::vector<double> f_lagrangian =
      FBF_TRY(lagrangian_fock(res, molecule, basis, h, b, naux, c, ncore, ncas, gamma, gamma2));

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
// mcscf::zvector over the CI-relaxed Hessian at the solution, as
// mcscf::df_gradient builds it for its own (H_0's sigma, row-major with no
// reorder: h_eff is symmetric, and (tu|vw) column-major is (wv|ut)
// row-major; the orbital Hessian over @p averaged, the Fock at the
// solution's weighted RDMs; the response integrals), for the right-hand
// sides @p orbital_rhs (the packed pairs), @p ci_rhs (dim x nroots) and
// @p theta_rhs (zvector's).
Result<mcscf::ZVector>
lagrangian_zvector(DeviceResources &res, const SphericalMatrixAO &h,
                   const Buffers::device<double> &b, const int32_t naux,
                   const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                   const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta, ci::Rdm &rdm,
                   const CasscfIntegrals &ints, const response::Fock &averaged,
                   const std::span<const double> weights, const double *orbital_rhs,
                   const double *ci_rhs, const std::span<const double> theta_rhs) {
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
  mcscf::ResponseIntegrals response_ints{nmo,
                                         static_cast<size_t>(ncore),
                                         n,
                                         FBF_TRY(to_host(res, averaged.fock_inactive.data(), mo2)),
                                         FBF_TRY(to_host(res, ints.ppaa.data(), mo2 * n * n)),
                                         FBF_TRY(to_host(res, ints.papa.data(), mo2 * n * n))};
  mcscf::RelaxedHessian relaxed = FBF_TRY(mcscf::RelaxedHessian::create(
      res, hessian, sigma, rdm, alpha, beta, solution.roots, weights, std::move(response_ints)));
  return mcscf::zvector(res, relaxed, hessian, solution.roots, orbital_rhs, ci_rhs, 1e-10, 200,
                        theta_rhs);
}

// Refuse an asymmetric block of the gate, naming @p who.
Status check_asymmetry(const std::string_view who, const std::array<double, 6> &asymmetry,
                       const mcscf::ZVector &z) {
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

Result<ScGradient>
lagrangian_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                    const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                    const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                    ScAdjoint corr, ZcMultipliers zc, const ZcAdjoint &phi) {
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
  mcscf::ZVector multipliers =
      FBF_TRY(lagrangian_zvector(res, h, b, naux32, solution, ncore, ncas, alpha, beta, rdm, ints,
                                 fock, weights, orbital_rhs.data(), ci_rhs.data(), {}));

  // The gate: X^L symmetric in every block.
  const std::array<double, 6> asymmetry = FBF_TRY(
      lagrangian_asymmetry(res, molecule, basis, h, b, naux32, solution, ncore, ncas, rdm, weights,
                           solution.gamma, solution.gamma2, multipliers, corr.a.data()));
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

Result<QdScLagrangian>
qd_sc_lagrangian(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                 const BasisSet &aux, const SphericalMatrixAO &h, const Buffers::device<double> &b,
                 const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                 const int32_t state, const int32_t batch) {
  constexpr std::string_view kWho = "nevpt2::qd_sc_lagrangian";
  const SphericalMatrixAO &c = solution.c;
  const int64_t nao = basis.nao_sphe();
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  if (c.n() != nao || ncore < 0 || ncas < 1 || ncore + ncas > nao) {
    return err_config("{}: C over {} orbitals, ncore {} and ncas {}, but the basis has {} "
                      "spherical AOs",
                      kWho, c.n(), ncore, ncas, nao);
  }
  const auto nmo = static_cast<size_t>(nao);
  const auto nmo32 = static_cast<int32_t>(nao);
  const auto n = static_cast<size_t>(ncas);
  const size_t nroots = solution.roots.energies.size();
  if (solution.gamma.size() != n * n || solution.gamma2.size() != n * n * n * n ||
      solution.root_gammas.size() != nroots * n * n) {
    return err_config("{}: RDMs of {}, {} and {} elements for {} roots and ncas {}", kWho,
                      solution.gamma.size(), solution.gamma2.size(), solution.root_gammas.size(),
                      nroots, ncas);
  }
  if (state < 0 || static_cast<size_t>(state) >= nroots) {
    return err_config("{}: state {} of {} roots", kWho, state, nroots);
  }
  // The solution's weights, everywhere. A root of zero weight has no SA CI
  // row: its multiplier's right-hand side divides by 2 w_r.
  const std::vector<double> weights = FBF_TRY(mcscf::sa_gradient_weights(solution, n));
  for (size_t r = 0; r < nroots; ++r) {
    if (!(weights[r] >= mcscf::kRotationWeightTolerance)) {
      return err_config("{}: root {} has weight {:.3e}, and a root of zero weight has no "
                        "state-averaged CI condition to differentiate",
                        kWho, r, weights[r]);
    }
  }
  const auto [alpha, beta] = FBF_TRY(active_strings(res, kWho, molecule, solution.gamma, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  if (solution.roots.vectors.num_elements() != dim * nroots) {
    return err_config("{}: {} CI doubles for {} roots of {} determinants", kWho,
                      solution.roots.vectors.num_elements(), nroots, dim);
  }
  const double *const vectors = solution.roots.vectors.data();

  // H_eff = diag(E_r) + 1/2 (C + C^T), its eigenvectors, and W = u_K u_K^T.
  const std::vector<ClassEnergies> elements =
      FBF_TRY(qd_sc_energy(res, alpha, beta, naux, nmo32, nmo32, b.data(), c.data(), h.data(),
                           ncore, ncas, weights, vectors, batch));
  std::vector<double> heff(nroots * nroots);
  for (size_t i = 0; i < nroots; ++i) {
    for (size_t j = 0; j < nroots; ++j) {
      heff[i + (nroots * j)] =
          (i == j ? solution.roots.energies[i] : 0.0) +
          (0.5 * (elements[(i * nroots) + j].total + elements[(j * nroots) + i].total));
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
  const auto k = static_cast<size_t>(state);
  for (size_t l = 0; l < nroots; ++l) {
    if (l != k && std::abs(eigenvalues[k] - eigenvalues[l]) < mcscf::kRootGapFloor) {
      return err_numerical("{}: QD states {} and {} are {:.3e} Eh apart, under {:g}: the state's "
                           "eigenvector has no derivative at a degeneracy",
                           kWho, std::min(k, l), std::max(k, l),
                           std::abs(eigenvalues[k] - eigenvalues[l]), mcscf::kRootGapFloor);
    }
  }
  const std::span<const double> u{heff.data() + (k * nroots), nroots};
  std::vector<double> w(nroots * nroots);
  std::vector<double> rho(nroots);
  for (size_t i = 0; i < nroots; ++i) {
    for (size_t j = 0; j < nroots; ++j) {
      w[(i * nroots) + j] = u[i] * u[j];
    }
    rho[i] = u[i] * u[i];
  }

  // y_bar, h_bar, Lambda, X and A of E^corr[W] + Phi^SA.
  QdScAdjoint corr = FBF_TRY(qd_sc_adjoint(res, alpha, beta, naux, nmo32, nmo32, b.data(), c.data(),
                                           h.data(), ncore, ncas, weights, vectors, w, batch));
  ZcMultipliers zc =
      FBF_TRY(zc_multipliers(res, nmo32, ncore, ncas, corr.a.data(), corr.epsilon.data()));
  const ZcAdjoint phi = FBF_TRY(zc_adjoint(res, alpha, beta, naux, nmo32, nmo32, b.data(), c.data(),
                                           h.data(), ncore, ncas, weights, vectors, zc.z.data()));
  FBF_TRY(add_adjoint(res, corr.h_bar, corr.lambda, corr.y, corr.x, corr.a, phi));

  // tau, split into zeta and t; zeta's two consumers.
  mcscf::RootPairMultipliers pairs =
      FBF_TRY(mcscf::root_pair_multipliers(res, solution.roots, weights, corr.y.data()));
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, naux, c, ncore, ncas));
  mcscf::ZetaTerms zeta = FBF_TRY(mcscf::zeta_terms(res, molecule, h, b, naux, solution, ncore,
                                                    ncas, ints.ppaa, ints.papa, rdm, pairs.zeta));

  // The right-hand sides: g^rho + zeta . g + A over the packed pairs, y_bar_r
  // / (2 w_r) (zvector's CI rows are the SA Hessian's over 2 w_r), and t.
  Buffers::device<double> rdms_d((n * n) + (n * n * n * n), res.device());
  FBF_TRY(rdm.rdm12(res, rho, vectors, rdms_d.data(), rdms_d.data() + (n * n)));
  std::vector<double> gamma_rho = FBF_TRY(to_host(res, rdms_d.data(), n * n));
  std::vector<double> gamma2_rho = FBF_TRY(to_host(res, rdms_d.data() + (n * n), n * n * n * n));
  const response::Fock fock_rho = FBF_TRY(response::fock(
      res, molecule, h, b, naux, c, ncore, ncas, ints.ppaa, ints.papa, gamma_rho, gamma2_rho));
  const response::Fock averaged =
      FBF_TRY(response::fock(res, molecule, h, b, naux, c, ncore, ncas, ints.ppaa, ints.papa,
                             solution.gamma, solution.gamma2));
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto packed = static_cast<int32_t>(fock_rho.gradient.num_elements());
  Buffers::device<double> orbital_rhs(static_cast<size_t>(packed), res.device());
  FBF_TRY(response::pack_orbital_pairs(res, nmo32, ncore, ncas, corr.a.data(), orbital_rhs.data()));
  for (const double *const add :
       std::array<const double *, 2>{fock_rho.gradient.data(), zeta.orbital.data()}) {
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, packed, &kOne<double>, add, 1, orbital_rhs.data(), 1)));
  }
  Buffers::device<double> ci_rhs(dim * nroots, res.device());
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(ci_rhs.data(), corr.y.data(), dim * nroots * sizeof(double),
                                     wwr::wwrMemcpyDeviceToDevice, res.stream())));
  for (size_t r = 0; r < nroots; ++r) {
    const double scale = 1.0 / (2.0 * weights[r]);
    FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, static_cast<int32_t>(dim), &scale,
                                                  ci_rhs.data() + (r * dim), 1)));
  }
  mcscf::ZVector multipliers = FBF_TRY(
      lagrangian_zvector(res, h, b, naux, solution, ncore, ncas, alpha, beta, rdm, ints, averaged,
                         weights, orbital_rhs.data(), ci_rhs.data(), pairs.theta_rhs));

  // The gate, its CAS part at the rho-weighted RDMs plus zeta's.
  for (size_t i = 0; i < gamma_rho.size(); ++i) {
    gamma_rho[i] += zeta.gamma[i];
  }
  for (size_t i = 0; i < gamma2_rho.size(); ++i) {
    gamma2_rho[i] += zeta.gamma2[i];
  }
  const std::array<double, 6> asymmetry = FBF_TRY(lagrangian_asymmetry(
      res, molecule, basis, h, b, naux, solution, ncore, ncas, rdm, weights, std::move(gamma_rho),
      std::move(gamma2_rho), multipliers, corr.a.data()));
  FBF_TRY(check_asymmetry(kWho, asymmetry, multipliers));

  return QdScLagrangian{.eigenvalues = std::move(eigenvalues),
                        .w = std::move(w),
                        .rho = std::move(rho),
                        .weights = weights,
                        .corr = std::move(corr),
                        .zc = std::move(zc),
                        .pairs = std::move(pairs),
                        .zeta = std::move(zeta),
                        .multipliers = std::move(multipliers),
                        .asymmetry = asymmetry};
}

Result<QdScGradient> qd_sc_gradient(DeviceResources &res, const Molecule &molecule,
                                    const BasisSet &basis,
                                    const std::span<const PrimitivePairClass> classes,
                                    const BasisSet &aux, const SphericalMatrixAO &h,
                                    const Buffers::device<double> &b, const double *l,
                                    const mcscf::Solution &solution, const int32_t ncore,
                                    const int32_t ncas, const int32_t state, const int32_t batch) {
  QdScLagrangian lagrangian = FBF_TRY(
      qd_sc_lagrangian(res, molecule, basis, aux, h, b, solution, ncore, ncas, state, batch));

  // E_cas at rho, (Z, z, Theta), and E^corr[W] + Phi^SA's arrays.
  const mcscf::MoAdjoint extra{.h_bar = lagrangian.corr.h_bar.data(),
                               .lambda = lagrangian.corr.lambda.data(),
                               .x = lagrangian.corr.x.data()};
  mcscf::GradientTerms terms =
      FBF_TRY(mcscf::df_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                                 lagrangian.rho, &lagrangian.multipliers, &extra));

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
  if (max_zeta > 0.0) {
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
  const auto k = static_cast<size_t>(state);
  double heff_gap = std::numeric_limits<double>::infinity();
  for (size_t m = 0; m < lagrangian.eigenvalues.size(); ++m) {
    if (m != k) {
      heff_gap =
          std::min(heff_gap, std::abs(lagrangian.eigenvalues[k] - lagrangian.eigenvalues[m]));
    }
  }
  return QdScGradient{.total = std::move(total),
                      .terms = std::move(terms),
                      .transition = std::move(transition),
                      .lagrangian = std::move(lagrangian),
                      .max_zeta = max_zeta,
                      .root_gap = root_gap,
                      .heff_gap = heff_gap};
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

Result<ScGradient> sc_gradient(DeviceResources &res, const Molecule &molecule,
                               const BasisSet &basis,
                               const std::span<const PrimitivePairClass> classes,
                               const BasisSet &aux, const SphericalMatrixAO &h,
                               const Buffers::device<double> &b, const double *l,
                               const mcscf::Solution &solution, const int32_t ncore,
                               const int32_t ncas, const int32_t batch) {
  const auto [alpha, beta] =
      FBF_TRY(root_strings(res, "nevpt2::sc_gradient", "SC", molecule, solution, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const auto nmo = static_cast<int32_t>(solution.c.n());
  const double *const c = solution.c.data();
  const double *const vector = solution.roots.vectors.data();
  ScAdjoint corr = FBF_TRY(sc_adjoint(res, alpha, beta, naux, nmo, nmo, b.data(), c, h.data(),
                                      ncore, ncas, vector, batch));
  ZcMultipliers zc =
      FBF_TRY(zc_multipliers(res, nmo, ncore, ncas, corr.a.data(), corr.epsilon.data()));
  const ZcAdjoint phi = FBF_TRY(zc_adjoint(res, alpha, beta, naux, nmo, nmo, b.data(), c, h.data(),
                                           ncore, ncas, std::array{1.0}, vector, zc.z.data()));
  return lagrangian_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                             std::move(corr), std::move(zc), phi);
}

Result<PcGradient> pc_gradient(DeviceResources &res, const Molecule &molecule,
                               const BasisSet &basis,
                               const std::span<const PrimitivePairClass> classes,
                               const BasisSet &aux, const SphericalMatrixAO &h,
                               const Buffers::device<double> &b, const double *l,
                               const mcscf::Solution &solution, const int32_t ncore,
                               const int32_t ncas, const int32_t batch) {
  const auto [alpha, beta] =
      FBF_TRY(root_strings(res, "nevpt2::pc_gradient", "PC", molecule, solution, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const auto nmo = static_cast<int32_t>(solution.c.n());
  const double *const c = solution.c.data();
  const double *const vector = solution.roots.vectors.data();
  // A refused class is the energy's error, here and so the gradient's.
  PcAdjoint corr = FBF_TRY(pc_adjoint(res, alpha, beta, naux, nmo, nmo, b.data(), c, h.data(),
                                      ncore, ncas, vector, batch));
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
                                             std::move(zc), phi));
  return PcGradient{.terms = std::move(g.terms),
                    .energies = corr.energies,
                    .zc = std::move(g.zc),
                    .multipliers = std::move(g.multipliers),
                    .asymmetry = g.asymmetry};
}

} // namespace free_breakfast::nevpt2
