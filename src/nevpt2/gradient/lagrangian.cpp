/// @file
/// @brief lagrangian_gradient, sc_gradient and pc_gradient: the Z-vector of
/// the SC- or PC-NEVPT2 Lagrangian on a single-state DF-CASSCF over the CI-relaxed
/// Hessian, the gate that its orbital source term is symmetric in every
/// block, and the nuclear term as one mcscf::df_gradient pass over the
/// multipliers and the correlation's MO adjoint.

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
import free_breakfast.mcscf.gradient;
import :operands; // to_host

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
// F the generalized Fock at the root's RDMs plus z's transition RDMs (E_cas's
// and z . g_ci's terms), A^Z = 2 (F' - F'^T), F' the derivative of the
// root's F along C (1 + e kappa(Z)) (Z . g_orb's term: by the symmetry of the
// orbital Hessian at the solution, whose full gradient vanishes, the
// derivative of the gradient along Z), and @p a the correlation's and z^c's
// A. F is a polynomial of degree 4 in C, so F' is the five-point stencil's,
// exactly.
Result<std::array<double, 6>>
lagrangian_asymmetry(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                     const SphericalMatrixAO &h, const Buffers::device<double> &b,
                     const int32_t naux, const mcscf::Solution &solution, const int32_t ncore,
                     const int32_t ncas, ci::Rdm &rdm, const mcscf::ZVector &z, const double *a) {
  const SphericalMatrixAO &c = solution.c;
  const auto nmo = static_cast<size_t>(c.n());
  const auto n = static_cast<size_t>(ncas);
  const auto n32 = static_cast<int32_t>(nmo);
  const wwr::wwrblasHandle_t blas = res.blas().get();

  // gamma + gamma_T and Gamma + Gamma_T: half the difference of the RDMs of
  // c + z and c - z at weight 1, as mcscf::df_gradient takes them.
  std::vector<double> gamma = solution.gamma;
  std::vector<double> gamma2 = solution.gamma2;
  const auto dim = static_cast<size_t>(rdm.size());
  Buffers::device<double> shifted(dim, res.device());
  Buffers::device<double> gamma_d(n * n, res.device());
  Buffers::device<double> gamma2_d(n * n * n * n, res.device());
  for (const double sign : {1.0, -1.0}) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(shifted.data(), solution.roots.vectors.data(),
                                           dim * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                           res.stream())));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(dim), &sign,
                                                  z.ci.data(), 1, shifted.data(), 1)));
    const std::array<double, 1> one{1.0};
    FBF_TRY(rdm.rdm12(res, one, shifted.data(), gamma_d.data(), gamma2_d.data()));
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

  // F' along kappa(Z) by the stencil, at the root's RDMs.
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
  double trace = 0.0;
  for (size_t t = 0; t < n; ++t) {
    trace += solution.gamma[t * (n + 1)];
  }
  const auto nelecas = static_cast<int32_t>(std::lround(trace));
  const int32_t two_s = molecule.multiplicity() - 1;
  if (std::abs(trace - nelecas) > kNelecasTolerance || nelecas < two_s ||
      (nelecas - two_s) % 2 != 0 || nelecas - ((nelecas - two_s) / 2) > ncas) {
    return err_config("nevpt2::lagrangian_gradient: trace(gamma) {} is no electron count of "
                      "multiplicity {} in {} orbitals",
                      trace, molecule.multiplicity(), ncas);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const int32_t n_beta = (nelecas - two_s) / 2;
  const ci::DeviceStringSet alpha =
      FBF_TRY(ci::DeviceStringSet::create(res, ncas, nelecas - n_beta));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, n_beta));
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
  const wwr::wwrblasHandle_t blas = res.blas().get();
  for (const auto &[to, from] :
       {std::pair{&corr.h_bar, &phi.h_bar}, std::pair{&corr.lambda, &phi.lambda},
        std::pair{&corr.y, &phi.y}, std::pair{&corr.x, &phi.x}, std::pair{&corr.a, &phi.a}}) {
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(to->num_elements()),
                                                  &kOne<double>, from->data(), 1, to->data(), 1)));
  }

  // The CI-relaxed Hessian at the solution, as mcscf::df_gradient builds it
  // for its own Z-vector: H_0's sigma (row-major with no reorder: h_eff is
  // symmetric, and (tu|vw) column-major is (wv|ut) row-major), the orbital
  // Hessian over the Fock at the root's RDMs, and the response integrals.
  const auto naux32 = static_cast<int32_t>(naux);
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, naux32, c, ncore, ncas));
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  FBF_TRY(sigma.set_integrals(res, ints.hamiltonian.e_core,
                              FBF_TRY(to_host(res, ints.hamiltonian.h_eff.data(), n * n)),
                              FBF_TRY(to_host(res, ints.hamiltonian.eri.data(), n * n * n * n))));
  const response::Fock fock =
      FBF_TRY(response::fock(res, molecule, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa,
                             solution.gamma, solution.gamma2));
  response::Hessian hessian = FBF_TRY(
      response::Hessian::create(res, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa, fock));
  mcscf::ResponseIntegrals response_ints{nmo,
                                         static_cast<size_t>(ncore),
                                         n,
                                         FBF_TRY(to_host(res, fock.fock_inactive.data(), mo2)),
                                         FBF_TRY(to_host(res, ints.ppaa.data(), mo2 * n * n)),
                                         FBF_TRY(to_host(res, ints.papa.data(), mo2 * n * n))};
  const std::array<double, 1> weights{1.0};
  mcscf::RelaxedHessian relaxed = FBF_TRY(mcscf::RelaxedHessian::create(
      res, hessian, sigma, rdm, alpha, beta, solution.roots, weights, std::move(response_ints)));

  // The right-hand sides: g + A over the packed pairs, and 1/2 (y + y^c),
  // the tex's 2 P (H - E) P row in zvector's (H_0 - E) one.
  Buffers::device<double> orbital_rhs(static_cast<size_t>(hessian.size()), res.device());
  FBF_TRY(response::pack_orbital_pairs(res, static_cast<int32_t>(nmo), ncore, ncas, corr.a.data(),
                                       orbital_rhs.data()));
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, hessian.size(), &kOne<double>,
                                                fock.gradient.data(), 1, orbital_rhs.data(), 1)));
  Buffers::device<double> ci_rhs(dim, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ci_rhs.data(), corr.y.data(), dim * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, res.stream())));
  FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, static_cast<int32_t>(dim), &kHalf<double>,
                                                ci_rhs.data(), 1)));
  mcscf::ZVector multipliers = FBF_TRY(
      mcscf::zvector(res, relaxed, hessian, solution.roots, orbital_rhs.data(), ci_rhs.data()));

  // The gate: X^L symmetric in every block.
  const std::array<double, 6> asymmetry = FBF_TRY(lagrangian_asymmetry(
      res, molecule, basis, h, b, naux32, solution, ncore, ncas, rdm, multipliers, corr.a.data()));
  for (size_t k = 0; k < asymmetry.size(); ++k) {
    if (!(asymmetry[k] <= kAsymmetryTolerance)) {
      return err_numerical("nevpt2::lagrangian_gradient: the Lagrangian's orbital source term is "
                           "not symmetric on its {} block: max |X^L - X^L^T| {:.3e} (gate {:g}; "
                           "Z-vector {} iterations, residual {:.3e})",
                           kAsymmetryBlockNames[k], asymmetry[k], kAsymmetryTolerance,
                           multipliers.iterations, multipliers.residual);
    }
  }

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
                                           ncore, ncas, vector, zc.z.data()));
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
                                           ncore, ncas, vector, zc.z.data()));
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
