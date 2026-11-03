/// @file
/// @brief mcscf::df_nac: the CI-relaxed Hessian at the solution, the
/// transition RDMs of the pair, the transition orbital gradient
/// (transition_orbital_gradient) and its Z-vector; then h_IJ from one
/// df_transition_gradient over those multipliers, and d^CSF from
/// overlap_half_gradient at the antisymmetrized AO transition density (a
/// gemmwich). The combination overload is the same over the pairs' RDMs
/// summed, with D^X's antisymmetric part added to the right-hand side and its
/// own overlap_half_gradient.

module free_breakfast.mcscf.nac;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemcpyAsync, gemmwich's BLAS handle
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.active_space;
import free_breakfast.integrals.one_electron.gradient;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.response.fock;
import free_breakfast.response.hessian;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.relaxed_hessian;
import free_breakfast.mcscf.zvector;
import free_breakfast.mcscf.gradient;
import free_breakfast.mcscf.transition;
import free_breakfast.mcscf.root_rotation; // kRootGapFloor

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// The weighted gamma is the roots' plain average to this, absolute, at equal
// weights (df_gradient's own tolerance on the weighted sum), and then the
// weights are exactly 1 / nroots; otherwise they are recovered.
constexpr double kNacEqualWeights = 1e-10;
// Below this gap (hartree) the coupling is not divided by it: mult_ediff.
constexpr double kNacDegenerateGap = 1e-10;
// The Z-vector's tolerance, on ||H~ z_o + g||: zvector's default. The pair
// divides h_IJ by its gap after the solve, so its solve is held to this times
// the gap (README.md "The Z-vector's tolerance"), the gap taken no smaller
// than kNacSolveGapFloor so the tolerance stays reachable in doubles.
constexpr double kNacSolveTolerance = 1e-10;
constexpr double kNacSolveGapFloor = 1e-2;
// trace(gamma) is the active electron count to this, as df_gradient's.
constexpr double kNacElectronCount = 1e-6;

// @p count doubles from the device into a host vector, waited for.
Result<std::vector<double>> nac_download(DeviceResources &res, const double *const from,
                                         const size_t count) {
  std::vector<double> to(count);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to.data(), from, count * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return to;
}

// What both overloads check of the solution, the orbitals and the fitted
// integrals, in their order, under @p who; the active electron count.
Result<int32_t> nac_check(const DeviceResources &res, const Molecule &molecule,
                          const BasisSet &basis, const BasisSet &aux,
                          const Buffers::device<double> &b, const Solution &solution,
                          const int32_t ncore, const int32_t ncas, const std::string_view who) {
  const SphericalMatrixAO &c = solution.c;
  if (c.dev_idx() != res.dev_idx()) {
    return err_config("{}: C is on device {}, but the call passed device {}", who, c.dev_idx(),
                      res.dev_idx());
  }
  const int64_t nao = basis.nao_sphe();
  if (c.n() != nao) {
    return err_config("{}: C is {} square, but the basis has {} spherical AOs", who, c.n(), nao);
  }
  if (ncore < 0 || ncas < 1 || ncore + ncas > nao) {
    return err_config("{}: ncore {} and ncas {} over {} orbitals", who, ncore, ncas, nao);
  }
  const auto n = static_cast<size_t>(ncas);
  const size_t m = n * n;
  const size_t roots = solution.roots.energies.size();
  if (solution.gamma.size() != m || solution.gamma2.size() != m * m ||
      solution.root_gammas.size() != roots * m) {
    return err_config("{}: RDMs of {}, {} and {} root gamma elements for {} roots and ncas {}", who,
                      solution.gamma.size(), solution.gamma2.size(), solution.root_gammas.size(),
                      roots, ncas);
  }
  double trace = 0.0;
  for (size_t t = 0; t < n; ++t) {
    trace += solution.gamma[t * (n + 1)];
  }
  const auto nelecas = static_cast<int32_t>(std::lround(trace));
  const int32_t two_s = molecule.multiplicity() - 1;
  if (std::abs(trace - nelecas) > kNacElectronCount || nelecas < two_s ||
      (nelecas - two_s) % 2 != 0 || nelecas - ((nelecas - two_s) / 2) > ncas) {
    return err_config("{}: trace(gamma) {} is no electron count of multiplicity {} in {} orbitals",
                      who, trace, molecule.multiplicity(), ncas);
  }
  const int64_t naux = aux.nao_sphe();
  if (naux < 1 || b.num_elements() != static_cast<size_t>(naux * nao * nao)) {
    return err_config("{}: B holds {} doubles, but naux nao^2 is {} x {}^2", who, b.num_elements(),
                      naux, nao);
  }
  return nelecas;
}

// The CI objects and the CI-relaxed Hessian at the solution's orbitals,
// roots and weights, as df_gradient builds them for a root. Held on the heap:
// RelaxedHessian, the RDMs and the sigma point into their neighbours.
struct NacSetup {
  std::optional<ci::DeviceStringSet> alpha;
  std::optional<ci::DeviceStringSet> beta;
  std::optional<ci::Rdm> rdm;
  size_t dim = 0;
  std::optional<CasscfIntegrals> ints;
  std::optional<response::Fock> averaged;
  std::optional<ci::Sigma> sigma;
  std::optional<response::Hessian> hessian;
  std::optional<RelaxedHessian> relaxed;
};

// The CI strings and RDMs over the roots' determinants (@ref NacSetup's
// first four fields), the CI vectors checked against them under @p who.
Result<std::unique_ptr<NacSetup>> nac_ci(DeviceResources &res, const Molecule &molecule,
                                         const Solution &solution, const int32_t ncas,
                                         const int32_t nelecas, const std::string_view who) {
  const size_t roots = solution.roots.energies.size();
  auto s = std::make_unique<NacSetup>();
  const int32_t two_s = molecule.multiplicity() - 1;
  const int32_t n_beta = (nelecas - two_s) / 2;
  s->alpha.emplace(FBF_TRY(ci::DeviceStringSet::create(res, ncas, nelecas - n_beta)));
  s->beta.emplace(FBF_TRY(ci::DeviceStringSet::create(res, ncas, n_beta)));
  s->rdm.emplace(FBF_TRY(ci::Rdm::create(res, *s->alpha, *s->beta)));
  s->dim = static_cast<size_t>(s->rdm->size());
  if (solution.roots.vectors.num_elements() != s->dim * roots) {
    return err_config("{}: {} CI doubles for {} roots of {} determinants", who,
                      solution.roots.vectors.num_elements(), roots, s->dim);
  }
  return s;
}

Result<std::unique_ptr<NacSetup>> nac_setup(DeviceResources &res, const Molecule &molecule,
                                            const BasisSet &basis, const SphericalMatrixAO &h,
                                            const Buffers::device<double> &b, const int32_t q,
                                            const Solution &solution, const int32_t ncore,
                                            const int32_t ncas, const int32_t nelecas,
                                            const std::string_view who) {
  const SphericalMatrixAO &c = solution.c;
  const auto n = static_cast<size_t>(ncas);
  const size_t m = n * n;
  const size_t roots = solution.roots.energies.size();
  const auto mo = static_cast<size_t>(basis.nao_sphe());
  // Equal weights, when the weighted gamma is the roots' plain average, are
  // taken as exactly 1 / nroots, as they always were; unequal ones are
  // recovered from the RDMs as df_gradient recovers them.
  double spread = 0.0;
  for (size_t i = 0; i < m; ++i) {
    double mean = 0.0;
    for (size_t r = 0; r < roots; ++r) {
      mean += solution.root_gammas[(r * m) + i];
    }
    spread = std::max(spread, std::abs((mean / static_cast<double>(roots)) - solution.gamma[i]));
  }
  std::vector<double> weights(roots, 1.0 / static_cast<double>(roots));
  if (spread > kNacEqualWeights) {
    weights = FBF_TRY(sa_gradient_weights(solution, n));
  }

  std::unique_ptr<NacSetup> s = FBF_TRY(nac_ci(res, molecule, solution, ncas, nelecas, who));

  // The CI-relaxed Hessian at the solution's orbitals and roots, as
  // df_gradient builds it for a root: the sigma takes the host arrays
  // row-major with no reorder (h_eff symmetric, (tu|vw) column-major is
  // (wv|ut) row-major).
  s->ints.emplace(FBF_TRY(casscf_integrals(res, basis, molecule, h, b, q, c, ncore, ncas)));
  const CasscfIntegrals &ints = *s->ints;
  s->averaged.emplace(FBF_TRY(response::fock(res, molecule, h, b, q, c, ncore, ncas, ints.ppaa,
                                             ints.papa, solution.gamma, solution.gamma2)));
  s->sigma.emplace(FBF_TRY(ci::Sigma::create(res, *s->alpha, *s->beta)));
  FBF_TRY(s->sigma->set_integrals(res, ints.hamiltonian.e_core,
                                  FBF_TRY(nac_download(res, ints.hamiltonian.h_eff.data(), m)),
                                  FBF_TRY(nac_download(res, ints.hamiltonian.eri.data(), m * m))));
  s->hessian.emplace(FBF_TRY(
      response::Hessian::create(res, h, b, q, c, ncore, ncas, ints.ppaa, ints.papa, *s->averaged)));
  ResponseIntegrals response_ints{
      mo,
      static_cast<size_t>(ncore),
      n,
      FBF_TRY(nac_download(res, s->averaged->fock_inactive.data(), mo * mo)),
      FBF_TRY(nac_download(res, ints.ppaa.data(), mo * mo * m)),
      FBF_TRY(nac_download(res, ints.papa.data(), mo * mo * m))};
  s->relaxed.emplace(
      FBF_TRY(RelaxedHessian::create(res, *s->hessian, *s->sigma, *s->rdm, *s->alpha, *s->beta,
                                     solution.roots, weights, std::move(response_ints))));
  return s;
}

// 1/2 G[C_x a C_x^T], a the @p width square column-major on the host and C_x
// the @p width columns of C from @p first: a CSF or basis-motion term
// (README.md "The CSF term").
Result<std::vector<double>> nac_half_gradient(DeviceResources &res, const BasisSet &basis,
                                              const std::span<const PrimitivePairClass> classes,
                                              const SphericalMatrixAO &c, const size_t first,
                                              const size_t width, std::span<const double> a_mo) {
  const auto mo = static_cast<size_t>(basis.nao_sphe());
  Buffers::device<double> a_mo_device(a_mo.size(), res.device());
  Buffers::device<double> work(width * mo, res.device());
  SphericalMatrixAO a(res, basis);
  const wwr::wwrblasHandle_t blas = res.blas().get();
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(a_mo_device.data(), a_mo.data(), a_mo.size() * sizeof(double),
                                     wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(gpu_status(gemmwich<wwr::WWRBLAS_OP_T>(
      blas, static_cast<int32_t>(mo), static_cast<int32_t>(width), c.data() + (mo * first),
      a_mo_device.data(), work.data(), a.data())));
  std::vector<double> g = FBF_TRY(overlap_half_gradient(res, basis, classes, a));
  for (double &v : g) {
    v *= 0.5;
  }
  return g;
}

Result<Nac> df_nac(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                   const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                   const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                   const Solution &solution, const int32_t ncore, const int32_t ncas,
                   const int32_t bra, const int32_t ket, const NacOptions &options) {
  constexpr std::string_view kWho = "mcscf::df_nac";
  const SphericalMatrixAO &c = solution.c;
  const auto nroots = static_cast<int32_t>(solution.roots.energies.size());
  if (nroots < 2) {
    return err_config("mcscf::df_nac: a coupling needs two roots, and the solution has {}", nroots);
  }
  if (bra < 0 || bra >= nroots || ket < 0 || ket >= nroots || bra == ket) {
    return err_config("mcscf::df_nac: roots ({}, {}) of a solution with {} roots", bra, ket,
                      nroots);
  }
  const int32_t nelecas =
      FBF_TRY(nac_check(res, molecule, basis, aux, b, solution, ncore, ncas, kWho));
  const double gap = solution.roots.energies[static_cast<size_t>(ket)] -
                     solution.roots.energies[static_cast<size_t>(bra)];
  if (!options.mult_ediff && std::abs(gap) < kNacDegenerateGap) {
    return err_unsupported("mcscf::df_nac: roots {} and {} are {:.3e} hartree apart, too near a "
                           "degeneracy to divide by (mult_ediff does not)",
                           bra, ket, gap);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto q = static_cast<int32_t>(aux.nao_sphe());
  const auto n = static_cast<size_t>(ncas);
  const size_t m = n * n;
  const std::unique_ptr<NacSetup> s =
      FBF_TRY(nac_setup(res, molecule, basis, h, b, q, solution, ncore, ncas, nelecas, kWho));

  // The pair's symmetrized transition RDMs and its unsymmetrized gamma^IJ.
  Buffers::device<double> rdms((2 * m) + (m * m), res.device());
  double *const gamma_device = rdms.data();
  double *const gamma2_device = gamma_device + m;
  double *const gamma_ij_device = gamma2_device + (m * m);
  const double *const ci_bra = solution.roots.vectors.data() + (static_cast<size_t>(bra) * s->dim);
  const double *const ci_ket = solution.roots.vectors.data() + (static_cast<size_t>(ket) * s->dim);
  FBF_TRY(s->rdm->trans_rdm12(res, ci_bra, ci_ket, gamma_device, gamma2_device));
  FBF_TRY(s->rdm->trans_rdm1(res, ci_bra, ci_ket, gamma_ij_device));
  const std::vector<double> gamma = FBF_TRY(nac_download(res, gamma_device, m));
  const std::vector<double> gamma2 = FBF_TRY(nac_download(res, gamma2_device, m * m));
  const std::vector<double> gamma_ij = FBF_TRY(nac_download(res, gamma_ij_device, m));

  // The transition orbital gradient, the Z-vector's right-hand side
  // (transition_orbital_gradient). Its CI part and its part along the
  // rotations among the roots are zero (README.md "Unequal weights"), so the
  // Z-vector has neither; at unequal weights it still returns Theta, from
  // the rotations' coupling to the orbitals, which df_transition_gradient
  // folds into z_c.
  const Buffers::device<double> transition = FBF_TRY(transition_orbital_gradient(
      res, molecule, h, b, q, c, ncore, ncas, s->ints->ppaa, s->ints->papa, gamma, gamma2));
  const double tolerance =
      kNacSolveTolerance * std::min(1.0, std::max(std::abs(gap), kNacSolveGapFloor));
  const ZVector z = FBF_TRY(zvector(res, *s->relaxed, *s->hessian, solution.roots,
                                    transition.data(), nullptr, tolerance));

  // h_IJ: the Hellmann-Feynman terms at the transition RDMs and the
  // multipliers' on top, in one call.
  const GradientTerms terms = FBF_TRY(df_transition_gradient(
      res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, gamma, gamma2, &z));

  // d^CSF = 1/2 G[C_a (gamma^T - gamma) C_a^T]: the antisymmetrized gamma^IJ
  // column-major, element (t, u) = gamma_ut - gamma_tu of the row-major
  // gamma^IJ, carried to the AO basis (README.md "The CSF term").
  std::vector<double> antisymmetric(m);
  for (size_t u = 0; u < n; ++u) {
    for (size_t t = 0; t < n; ++t) {
      antisymmetric[t + (n * u)] = gamma_ij[(u * n) + t] - gamma_ij[(t * n) + u];
    }
  }
  std::vector<double> csf = FBF_TRY(
      nac_half_gradient(res, basis, classes, c, static_cast<size_t>(ncore), n, antisymmetric));

  std::vector<double> total(terms.total.size());
  for (size_t i = 0; i < total.size(); ++i) {
    const double csf_part = options.use_etfs ? 0.0 : csf[i];
    total[i] =
        options.mult_ediff ? terms.total[i] + (gap * csf_part) : (terms.total[i] / gap) + csf_part;
  }
  return Nac{.numerator = terms.total,
             .csf = std::move(csf),
             .total = std::move(total),
             .iterations = z.iterations,
             .residual = z.residual};
}

Result<NacCombinationTerms>
nac_combination_terms(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                      const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                      const Buffers::device<double> &b, const Solution &solution,
                      const int32_t ncore, const int32_t ncas, const std::span<const double> t,
                      const std::span<const double> d_x) {
  constexpr std::string_view kWho = "mcscf::df_nac (combination)";
  const SphericalMatrixAO &c = solution.c;
  const size_t roots = solution.roots.energies.size();
  if (roots < 1 || t.size() != roots * roots) {
    return err_config("mcscf::df_nac (combination): T has {} elements for {} roots", t.size(),
                      roots);
  }
  for (size_t i = 0; i < roots; ++i) {
    for (size_t j = 0; j < roots; ++j) {
      const double tij = t[(i * roots) + j];
      if (!std::isfinite(tij) || tij != -t[(j * roots) + i]) {
        return err_config("mcscf::df_nac (combination): T is not antisymmetric at ({}, {}): {} "
                          "against {}",
                          i, j, tij, t[(j * roots) + i]);
      }
    }
  }
  const int32_t nelecas =
      FBF_TRY(nac_check(res, molecule, basis, aux, b, solution, ncore, ncas, kWho));
  const auto mo = static_cast<size_t>(basis.nao_sphe());
  const auto n_core = static_cast<size_t>(ncore);
  const auto n = static_cast<size_t>(ncas);
  const size_t nocc = n_core + n;
  // The space of an orbital: 0 core, 1 active, 2 virtual.
  const auto space = [&](const size_t p) { return p < n_core ? 0 : (p < nocc ? 1 : 2); };
  if (!d_x.empty()) {
    if (d_x.size() != mo * mo) {
      return err_config("mcscf::df_nac (combination): D^X has {} elements for {} orbitals",
                        d_x.size(), mo);
    }
    for (size_t p = 0; p < mo; ++p) {
      for (size_t r = 0; r < mo; ++r) {
        const double v = d_x[(p * mo) + r];
        if (!std::isfinite(v) || (space(p) == space(r) && v != 0.0)) {
          return err_config("mcscf::df_nac (combination): D^X is {} at ({}, {}): it must be "
                            "finite, and zero within the core, the active and the virtual "
                            "blocks (their rotations are redundant)",
                            v, p, r);
        }
      }
    }
  }
  // The pairs M < N with T_MN != 0, each refused below the gap floor.
  double smallest = std::numeric_limits<double>::infinity();
  std::vector<std::pair<size_t, size_t>> pairs;
  for (size_t i = 0; i < roots; ++i) {
    for (size_t j = i + 1; j < roots; ++j) {
      if (t[(i * roots) + j] == 0.0) {
        continue;
      }
      const double gap = solution.roots.energies[j] - solution.roots.energies[i];
      if (std::abs(gap) < kRootGapFloor) {
        return err_numerical("mcscf::df_nac (combination): roots {} and {} are {:.3e} hartree "
                             "apart, under {:.0e}, and T couples them: the CI part divides by "
                             "the gap",
                             i, j, std::abs(gap), kRootGapFloor);
      }
      smallest = std::min(smallest, std::abs(gap));
      pairs.emplace_back(i, j);
    }
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const size_t m = n * n;
  const std::unique_ptr<NacSetup> s = FBF_TRY(nac_ci(res, molecule, solution, ncas, nelecas, kWho));

  // The combined RDMs: Sum_MN T_MN / (E_N - E_M) (gamma^MN, Gamma^MN)
  // symmetrized, 2 T_MN / (E_N - E_M) over M < N, for the CI part; and
  // gamma^T = Sum_MN T_MN gamma^MN = Sum_{M<N} T_MN (gamma^MN - gamma^MN^T),
  // gamma^MN unsymmetrized, for the CSF part.
  std::vector<double> gamma(m, 0.0);
  std::vector<double> gamma2(m * m, 0.0);
  std::vector<double> gamma_t(m, 0.0);
  Buffers::device<double> rdms((2 * m) + (m * m), res.device());
  double *const gamma_device = rdms.data();
  double *const gamma2_device = gamma_device + m;
  double *const gamma_mn_device = gamma2_device + (m * m);
  for (const auto &[i, j] : pairs) {
    const double *const ci_m = solution.roots.vectors.data() + (i * s->dim);
    const double *const ci_n = solution.roots.vectors.data() + (j * s->dim);
    FBF_TRY(s->rdm->trans_rdm12(res, ci_m, ci_n, gamma_device, gamma2_device));
    FBF_TRY(s->rdm->trans_rdm1(res, ci_m, ci_n, gamma_mn_device));
    const std::vector<double> pair_gamma = FBF_TRY(nac_download(res, gamma_device, m));
    const std::vector<double> pair_gamma2 = FBF_TRY(nac_download(res, gamma2_device, m * m));
    const std::vector<double> gamma_mn = FBF_TRY(nac_download(res, gamma_mn_device, m));
    const double tij = t[(i * roots) + j];
    const double scale = 2.0 * tij / (solution.roots.energies[j] - solution.roots.energies[i]);
    for (size_t k = 0; k < m; ++k) {
      gamma[k] += scale * pair_gamma[k];
    }
    for (size_t k = 0; k < m * m; ++k) {
      gamma2[k] += scale * pair_gamma2[k];
    }
    for (size_t a = 0; a < n; ++a) {
      for (size_t u = 0; u < n; ++u) {
        gamma_t[(a * n) + u] += tij * (gamma_mn[(a * n) + u] - gamma_mn[(u * n) + a]);
      }
    }
  }

  // The CSF part, 1/2 G[C_a (gamma^T^T - gamma^T) C_a^T], as the pair's.
  std::vector<double> antisymmetric(m);
  for (size_t u = 0; u < n; ++u) {
    for (size_t a = 0; a < n; ++a) {
      antisymmetric[a + (n * u)] = gamma_t[(u * n) + a] - gamma_t[(a * n) + u];
    }
  }
  std::vector<double> csf =
      FBF_TRY(nac_half_gradient(res, basis, classes, c, n_core, n, antisymmetric));

  // D^X's basis motion, 1/2 G[C (D^X^T - D^X) C^T] over every orbital.
  std::vector<double> basis_motion(csf.size(), 0.0);
  if (!d_x.empty()) {
    std::vector<double> antisymmetric_x(mo * mo);
    for (size_t r = 0; r < mo; ++r) {
      for (size_t p = 0; p < mo; ++p) {
        antisymmetric_x[p + (mo * r)] = d_x[(r * mo) + p] - d_x[(p * mo) + r];
      }
    }
    basis_motion = FBF_TRY(nac_half_gradient(res, basis, classes, c, 0, mo, antisymmetric_x));
  }
  return NacCombinationTerms{.gamma = std::move(gamma),
                             .gamma2 = std::move(gamma2),
                             .csf = std::move(csf),
                             .basis_motion = std::move(basis_motion),
                             .gap = smallest};
}

Result<NacCombination> df_nac(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                              const std::span<const PrimitivePairClass> classes,
                              const BasisSet &aux, const SphericalMatrixAO &h,
                              const Buffers::device<double> &b, const double *l,
                              const Solution &solution, const int32_t ncore, const int32_t ncas,
                              const std::span<const double> t, const std::span<const double> d_x) {
  constexpr std::string_view kWho = "mcscf::df_nac (combination)";
  const SphericalMatrixAO &c = solution.c;
  // Every check and refusal, the combined RDMs and the two terms outside the
  // Z-vector.
  NacCombinationTerms terms = FBF_TRY(
      nac_combination_terms(res, molecule, basis, classes, aux, b, solution, ncore, ncas, t, d_x));
  const int32_t nelecas =
      FBF_TRY(nac_check(res, molecule, basis, aux, b, solution, ncore, ncas, kWho));
  const auto mo = static_cast<size_t>(basis.nao_sphe());
  const auto n_core = static_cast<size_t>(ncore);
  const size_t nocc = n_core + static_cast<size_t>(ncas);
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto q = static_cast<int32_t>(aux.nao_sphe());
  const std::unique_ptr<NacSetup> s =
      FBF_TRY(nac_setup(res, molecule, basis, h, b, q, solution, ncore, ncas, nelecas, kWho));

  // The right-hand side: the transition orbital gradient at the combined
  // RDMs, plus D^X's own orbital gradient, D^X_pq - D^X_qp over the packed
  // pairs p > q (response::pack_orbital_pairs' order): q core and p active or
  // virtual, then q active and p virtual.
  const Buffers::device<double> transition =
      FBF_TRY(transition_orbital_gradient(res, molecule, h, b, q, c, ncore, ncas, s->ints->ppaa,
                                          s->ints->papa, terms.gamma, terms.gamma2));
  std::vector<double> rhs =
      FBF_TRY(nac_download(res, transition.data(), transition.num_elements()));
  if (!d_x.empty()) {
    size_t at = 0;
    for (size_t col = 0; col < nocc; ++col) {
      for (size_t row = col < n_core ? n_core : nocc; row < mo; ++row) {
        rhs[at++] += d_x[(row * mo) + col] - d_x[(col * mo) + row];
      }
    }
    if (at != rhs.size()) {
      return err_config("mcscf::df_nac (combination): {} packed pairs, but the transition "
                        "gradient has {}",
                        at, rhs.size());
    }
  }
  Buffers::device<double> rhs_device(rhs.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(rhs_device.data(), rhs.data(), rhs.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  const ZVector z = FBF_TRY(zvector(res, *s->relaxed, *s->hessian, solution.roots,
                                    rhs_device.data(), nullptr, kNacSolveTolerance));

  // The CI part and both responses: the Hellmann-Feynman terms at the
  // combined RDMs and the multipliers' on top, in one call.
  const GradientTerms response =
      FBF_TRY(df_transition_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                     ncas, terms.gamma, terms.gamma2, &z));

  std::vector<double> total(response.total.size());
  for (size_t i = 0; i < total.size(); ++i) {
    total[i] = response.total[i] + terms.csf[i] + terms.basis_motion[i];
  }
  return NacCombination{.response = response.total,
                        .csf = std::move(terms.csf),
                        .basis_motion = std::move(terms.basis_motion),
                        .total = std::move(total),
                        .rhs = std::move(rhs),
                        .gap = terms.gap,
                        .iterations = z.iterations,
                        .residual = z.residual};
}

} // namespace free_breakfast::mcscf
