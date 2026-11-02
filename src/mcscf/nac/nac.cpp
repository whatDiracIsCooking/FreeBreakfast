/// @file
/// @brief mcscf::df_nac: the CI-relaxed Hessian at the solution, the
/// transition RDMs of the pair, the transition orbital gradient (two Focks'
/// gradients' difference) and its Z-vector; then h_IJ from one
/// df_transition_gradient over those multipliers, and d^CSF from
/// overlap_half_gradient at the antisymmetrized AO transition density (a
/// gemmwich).

module free_breakfast.mcscf.nac;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // axpy, wwrMemcpyAsync
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

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// The weighted gamma is the roots' plain average to this, absolute, at equal
// weights (df_gradient's own tolerance on the weighted sum), and then the
// weights are exactly 1 / nroots; otherwise they are recovered.
constexpr double kNacEqualWeights = 1e-10;
// Below this gap (hartree) the coupling is not divided by it: mult_ediff.
constexpr double kNacDegenerateGap = 1e-10;
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

Result<Nac> df_nac(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                   const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                   const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                   const Solution &solution, const int32_t ncore, const int32_t ncas,
                   const int32_t bra, const int32_t ket, const NacOptions &options) {
  const SphericalMatrixAO &c = solution.c;
  const auto nroots = static_cast<int32_t>(solution.roots.energies.size());
  if (nroots < 2) {
    return err_config("mcscf::df_nac: a coupling needs two roots, and the solution has {}", nroots);
  }
  if (bra < 0 || bra >= nroots || ket < 0 || ket >= nroots || bra == ket) {
    return err_config("mcscf::df_nac: roots ({}, {}) of a solution with {} roots", bra, ket,
                      nroots);
  }
  if (c.dev_idx() != res.dev_idx()) {
    return err_config("mcscf::df_nac: C is on device {}, but the call passed device {}",
                      c.dev_idx(), res.dev_idx());
  }
  const int64_t nao = basis.nao_sphe();
  if (c.n() != nao) {
    return err_config("mcscf::df_nac: C is {} square, but the basis has {} spherical AOs", c.n(),
                      nao);
  }
  if (ncore < 0 || ncas < 1 || ncore + ncas > nao) {
    return err_config("mcscf::df_nac: ncore {} and ncas {} over {} orbitals", ncore, ncas, nao);
  }
  const auto n = static_cast<size_t>(ncas);
  const size_t m = n * n;
  const auto roots = static_cast<size_t>(nroots);
  if (solution.gamma.size() != m || solution.gamma2.size() != m * m ||
      solution.root_gammas.size() != roots * m) {
    return err_config("mcscf::df_nac: RDMs of {}, {} and {} root gamma elements for {} roots and "
                      "ncas {}",
                      solution.gamma.size(), solution.gamma2.size(), solution.root_gammas.size(),
                      nroots, ncas);
  }
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
  const double gap = solution.roots.energies[static_cast<size_t>(ket)] -
                     solution.roots.energies[static_cast<size_t>(bra)];
  if (!options.mult_ediff && std::abs(gap) < kNacDegenerateGap) {
    return err_unsupported("mcscf::df_nac: roots {} and {} are {:.3e} hartree apart, too near a "
                           "degeneracy to divide by (mult_ediff does not)",
                           bra, ket, gap);
  }
  double trace = 0.0;
  for (size_t t = 0; t < n; ++t) {
    trace += solution.gamma[t * (n + 1)];
  }
  const auto nelecas = static_cast<int32_t>(std::lround(trace));
  const int32_t two_s = molecule.multiplicity() - 1;
  if (std::abs(trace - nelecas) > kNacElectronCount || nelecas < two_s ||
      (nelecas - two_s) % 2 != 0 || nelecas - ((nelecas - two_s) / 2) > ncas) {
    return err_config("mcscf::df_nac: trace(gamma) {} is no electron count of multiplicity {} in "
                      "{} orbitals",
                      trace, molecule.multiplicity(), ncas);
  }
  const int64_t naux = aux.nao_sphe();
  if (naux < 1 || b.num_elements() != static_cast<size_t>(naux * nao * nao)) {
    return err_config("mcscf::df_nac: B holds {} doubles, but naux nao^2 is {} x {}^2",
                      b.num_elements(), naux, nao);
  }
  const auto q = static_cast<int32_t>(naux);
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto mo = static_cast<size_t>(nao);
  const auto n_core = static_cast<size_t>(ncore);

  // The CI strings, sigma (H_0) and RDMs over the roots' determinants.
  const int32_t n_beta = (nelecas - two_s) / 2;
  const ci::DeviceStringSet alpha =
      FBF_TRY(ci::DeviceStringSet::create(res, ncas, nelecas - n_beta));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, n_beta));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  if (solution.roots.vectors.num_elements() != dim * roots) {
    return err_config("mcscf::df_nac: {} CI doubles for {} roots of {} determinants",
                      solution.roots.vectors.num_elements(), nroots, dim);
  }

  // The CI-relaxed Hessian at the solution's orbitals and roots, as
  // df_gradient builds it for a root: the sigma takes the host arrays
  // row-major with no reorder (h_eff symmetric, (tu|vw) column-major is
  // (wv|ut) row-major).
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, q, c, ncore, ncas));
  const response::Fock averaged =
      FBF_TRY(response::fock(res, molecule, h, b, q, c, ncore, ncas, ints.ppaa, ints.papa,
                             solution.gamma, solution.gamma2));
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  FBF_TRY(sigma.set_integrals(res, ints.hamiltonian.e_core,
                              FBF_TRY(nac_download(res, ints.hamiltonian.h_eff.data(), m)),
                              FBF_TRY(nac_download(res, ints.hamiltonian.eri.data(), m * m))));
  response::Hessian hessian = FBF_TRY(
      response::Hessian::create(res, h, b, q, c, ncore, ncas, ints.ppaa, ints.papa, averaged));
  ResponseIntegrals response_ints{
      mo,
      n_core,
      n,
      FBF_TRY(nac_download(res, averaged.fock_inactive.data(), mo * mo)),
      FBF_TRY(nac_download(res, ints.ppaa.data(), mo * mo * m)),
      FBF_TRY(nac_download(res, ints.papa.data(), mo * mo * m))};
  RelaxedHessian relaxed = FBF_TRY(RelaxedHessian::create(
      res, hessian, sigma, rdm, alpha, beta, solution.roots, weights, std::move(response_ints)));

  // The pair's symmetrized transition RDMs and its unsymmetrized gamma^IJ.
  Buffers::device<double> rdms((2 * m) + (m * m), res.device());
  double *const gamma_device = rdms.data();
  double *const gamma2_device = gamma_device + m;
  double *const gamma_ij_device = gamma2_device + (m * m);
  const double *const ci_bra = solution.roots.vectors.data() + (static_cast<size_t>(bra) * dim);
  const double *const ci_ket = solution.roots.vectors.data() + (static_cast<size_t>(ket) * dim);
  FBF_TRY(rdm.trans_rdm12(res, ci_bra, ci_ket, gamma_device, gamma2_device));
  FBF_TRY(rdm.trans_rdm1(res, ci_bra, ci_ket, gamma_ij_device));
  const std::vector<double> gamma = FBF_TRY(nac_download(res, gamma_device, m));
  const std::vector<double> gamma2 = FBF_TRY(nac_download(res, gamma2_device, m * m));
  const std::vector<double> gamma_ij = FBF_TRY(nac_download(res, gamma_ij_device, m));

  // The transition orbital gradient, the Z-vector's right-hand side: fock's
  // gradient at the transition RDMs less its gradient at zero RDMs (the
  // core's own energy multiplies <I|J> = 0). Its CI part and its part along
  // the rotations among the roots are zero (README.md "Unequal weights"), so
  // the Z-vector has neither; at unequal weights it still returns Theta,
  // from the rotations' coupling to the orbitals, which
  // df_transition_gradient folds into z_c.
  response::Fock transition = FBF_TRY(
      response::fock(res, molecule, h, b, q, c, ncore, ncas, ints.ppaa, ints.papa, gamma, gamma2));
  const std::vector<double> zero_gamma(m, 0.0);
  const std::vector<double> zero_gamma2(m * m, 0.0);
  const response::Fock core = FBF_TRY(response::fock(
      res, molecule, h, b, q, c, ncore, ncas, ints.ppaa, ints.papa, zero_gamma, zero_gamma2));
  const wwr::wwrblasHandle_t blas = res.blas().get();
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(
      blas, static_cast<int32_t>(transition.gradient.num_elements()), &kNegOne<double>,
      core.gradient.data(), 1, transition.gradient.data(), 1)));
  const ZVector z =
      FBF_TRY(zvector(res, relaxed, hessian, solution.roots, transition.gradient.data()));

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
  Buffers::device<double> antisymmetric_device(m, res.device());
  Buffers::device<double> work(n * mo, res.device());
  SphericalMatrixAO a(res, basis);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(antisymmetric_device.data(), antisymmetric.data(),
                                         m * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                         res.stream())));
  FBF_TRY(gpu_status(
      gemmwich<wwr::WWRBLAS_OP_T>(blas, static_cast<int32_t>(nao), ncas, c.data() + (mo * n_core),
                                  antisymmetric_device.data(), work.data(), a.data())));
  std::vector<double> csf = FBF_TRY(overlap_half_gradient(res, basis, classes, a));
  for (double &v : csf) {
    v *= 0.5;
  }

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

} // namespace free_breakfast::mcscf
