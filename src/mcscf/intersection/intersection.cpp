/// @file
/// @brief mcscf::df_intersection: the CI objects and the CI-relaxed Hessian at
/// the solution once, with the equal-weight pairs' mixing left out; the
/// orbital gradients of the pair's 2 x 2 Hamiltonian's mean, half difference
/// and coupling, solved together in one zvectors, each column's true
/// residual checked; df_gradient at the two mixes and df_transition_gradient
/// at the transition RDMs over those columns; and the 2 x 2 diagonalized to
/// turn the three to its eigenvectors.

module free_breakfast.mcscf.intersection;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemcpyAsync
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.active_space;
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

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// trace(gamma) is the active electron count to this, as df_gradient's.
constexpr double kIntersectionElectronCount = 1e-6;
// The CI response's tolerance inside H~, RelaxedHessian::create's default.
constexpr double kIntersectionResponseTolerance = 1e-8;
// The Z-vector's columns: the 2 x 2 Hamiltonian's mean, half difference and
// coupling.
constexpr size_t kIntersectionColumns = 3;

// @p count doubles from the device into a host vector, waited for.
Result<std::vector<double>> intersection_download(DeviceResources &res, const double *const from,
                                                  const size_t count) {
  std::vector<double> to(count);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to.data(), from, count * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return to;
}

// Column @p col of @p z as one ZVector, the form df_gradient takes: its
// orbital and CI blocks copied out, its rotations, products and residual.
Result<ZVector> intersection_column(DeviceResources &res, const ZVectors &z, const size_t col) {
  const size_t ncol = z.iterations.size();
  const size_t orbital = z.orbital.num_elements() / ncol;
  const size_t ci = z.ci.num_elements() / ncol;
  const size_t theta = z.theta.size() / ncol;
  ZVector out{.orbital = Buffers::device<double>(orbital, res.device()),
              .ci = Buffers::device<double>(ci, res.device()),
              .theta = std::vector<double>(
                  z.theta.begin() + static_cast<std::ptrdiff_t>(col * theta),
                  z.theta.begin() + static_cast<std::ptrdiff_t>((col + 1) * theta)),
              .iterations = z.iterations[col],
              .residual = z.residuals[col]};
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.orbital.data(), z.orbital.data() + (col * orbital),
                                         orbital * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                         res.stream())));
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(out.ci.data(), z.ci.data() + (col * ci), ci * sizeof(double),
                                     wwr::wwrMemcpyDeviceToDevice, res.stream())));
  return out;
}

Result<Intersection>
df_intersection(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                const Solution &solution, const int32_t ncore, const int32_t ncas, const int32_t i,
                const int32_t j, const IntersectionOptions &options) {
  const SphericalMatrixAO &c = solution.c;
  const auto nroots = static_cast<int32_t>(solution.roots.energies.size());
  if (nroots < 2) {
    return err_config("mcscf::df_intersection: an intersection needs two roots, and the solution "
                      "has {}",
                      nroots);
  }
  if (i < 0 || i >= nroots || j < 0 || j >= nroots || i == j) {
    return err_config("mcscf::df_intersection: roots ({}, {}) of a solution with {} roots", i, j,
                      nroots);
  }
  if (!(options.tolerance > 0.0) || options.max_iterations < 1) {
    return err_config("mcscf::df_intersection: a Z-vector tolerance of {} and {} products at most",
                      options.tolerance, options.max_iterations);
  }
  if (c.dev_idx() != res.dev_idx()) {
    return err_config("mcscf::df_intersection: C is on device {}, but the call passed device {}",
                      c.dev_idx(), res.dev_idx());
  }
  const int64_t nao = basis.nao_sphe();
  if (c.n() != nao) {
    return err_config("mcscf::df_intersection: C is {} square, but the basis has {} spherical AOs",
                      c.n(), nao);
  }
  if (ncore < 0 || ncas < 1 || ncore + ncas > nao) {
    return err_config("mcscf::df_intersection: ncore {} and ncas {} over {} orbitals", ncore, ncas,
                      nao);
  }
  const auto n = static_cast<size_t>(ncas);
  const size_t m = n * n;
  const auto roots = static_cast<size_t>(nroots);
  if (solution.gamma.size() != m || solution.gamma2.size() != m * m ||
      solution.root_gammas.size() != roots * m) {
    return err_config("mcscf::df_intersection: RDMs of {}, {} and {} root gamma elements for {} "
                      "roots and ncas {}",
                      solution.gamma.size(), solution.gamma2.size(), solution.root_gammas.size(),
                      roots, ncas);
  }
  double trace = 0.0;
  for (size_t t = 0; t < n; ++t) {
    trace += solution.gamma[t * (n + 1)];
  }
  const auto nelecas = static_cast<int32_t>(std::lround(trace));
  const int32_t two_s = molecule.multiplicity() - 1;
  if (std::abs(trace - nelecas) > kIntersectionElectronCount || nelecas < two_s ||
      (nelecas - two_s) % 2 != 0 || nelecas - ((nelecas - two_s) / 2) > ncas) {
    return err_config("mcscf::df_intersection: trace(gamma) {} is no electron count of "
                      "multiplicity {} in {} orbitals",
                      trace, molecule.multiplicity(), ncas);
  }
  const int64_t naux = aux.nao_sphe();
  if (naux < 1 || b.num_elements() != static_cast<size_t>(naux * nao * nao)) {
    return err_config("mcscf::df_intersection: B holds {} doubles, but naux nao^2 is {} x {}^2",
                      b.num_elements(), naux, nao);
  }
  // The weights, as df_gradient takes them (the solution's own, recovered
  // only when it carries none); I and J must share theirs, or the rotation
  // between them divides by E_I - E_J.
  const std::vector<double> weights = FBF_TRY(sa_gradient_weights(solution, n));
  const double w_i = weights[static_cast<size_t>(i)];
  const double w_j = weights[static_cast<size_t>(j)];
  if (std::abs(w_i - w_j) > kRotationWeightTolerance * std::max(w_i, w_j)) {
    return err_unsupported("mcscf::df_intersection: roots {} and {} are weighted {:g} and {:g}; "
                           "at unequal weights the rotation between them divides by E_I - E_J, "
                           "which is singular at an intersection",
                           i, j, w_i, w_j);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto q = static_cast<int32_t>(naux);
  const auto mo = static_cast<size_t>(nao);

  // The CI strings, sigma (H_0) and RDMs over the roots' determinants.
  const int32_t n_beta = (nelecas - two_s) / 2;
  const ci::DeviceStringSet alpha =
      FBF_TRY(ci::DeviceStringSet::create(res, ncas, nelecas - n_beta));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, n_beta));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  if (solution.roots.vectors.num_elements() != dim * roots) {
    return err_config("mcscf::df_intersection: {} CI doubles for {} roots of {} determinants",
                      solution.roots.vectors.num_elements(), roots, dim);
  }

  // The CI-relaxed Hessian at the solution's orbitals and roots, as
  // df_gradient builds it: the sigma takes the host arrays row-major with no
  // reorder (h_eff symmetric, (tu|vw) column-major is (wv|ut) row-major).
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, q, c, ncore, ncas));
  const response::Fock averaged =
      FBF_TRY(response::fock(res, molecule, h, b, q, c, ncore, ncas, ints.ppaa, ints.papa,
                             solution.gamma, solution.gamma2));
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  FBF_TRY(
      sigma.set_integrals(res, ints.hamiltonian.e_core,
                          FBF_TRY(intersection_download(res, ints.hamiltonian.h_eff.data(), m)),
                          FBF_TRY(intersection_download(res, ints.hamiltonian.eri.data(), m * m))));
  response::Hessian hessian = FBF_TRY(
      response::Hessian::create(res, h, b, q, c, ncore, ncas, ints.ppaa, ints.papa, averaged));
  ResponseIntegrals response_ints{
      mo,
      static_cast<size_t>(ncore),
      n,
      FBF_TRY(intersection_download(res, averaged.fock_inactive.data(), mo * mo)),
      FBF_TRY(intersection_download(res, ints.ppaa.data(), mo * mo * m)),
      FBF_TRY(intersection_download(res, ints.papa.data(), mo * mo * m))};
  // The SA Lagrangian's response only: a pair of equal weights, I and J
  // among them, leaves its mixing out of the CI response (README.md "The
  // derivation").
  RelaxedHessian relaxed = FBF_TRY(RelaxedHessian::create(
      res, hessian, sigma, rdm, alpha, beta, solution.roots, weights, std::move(response_ints),
      kIntersectionResponseTolerance, rotating_pairs(weights)));

  // The pair's RDMs: each root's, and the symmetrized transition RDMs.
  Buffers::device<double> rdms(m + (m * m), res.device());
  double *const gamma_device = rdms.data();
  double *const gamma2_device = gamma_device + m;
  std::array<std::vector<double>, 2> gamma_root;
  std::array<std::vector<double>, 2> gamma2_root;
  for (const auto &[root, at] : {std::pair{i, size_t{0}}, std::pair{j, size_t{1}}}) {
    std::vector<double> one_hot(roots, 0.0);
    one_hot[static_cast<size_t>(root)] = 1.0;
    FBF_TRY(rdm.rdm12(res, one_hot, solution.roots.vectors.data(), gamma_device, gamma2_device));
    gamma_root[at] = FBF_TRY(intersection_download(res, gamma_device, m));
    gamma2_root[at] = FBF_TRY(intersection_download(res, gamma2_device, m * m));
  }
  const double *const ci_i = solution.roots.vectors.data() + (static_cast<size_t>(i) * dim);
  const double *const ci_j = solution.roots.vectors.data() + (static_cast<size_t>(j) * dim);
  FBF_TRY(rdm.trans_rdm12(res, ci_i, ci_j, gamma_device, gamma2_device));
  const std::vector<double> gamma_ij = FBF_TRY(intersection_download(res, gamma_device, m));
  const std::vector<double> gamma2_ij = FBF_TRY(intersection_download(res, gamma2_device, m * m));

  // The three right-hand sides, side by side, the orbital gradients of the
  // 2 x 2 Hamiltonian's three combinations at the solution's CI: the mean
  // (H_II + H_JJ) / 2, response::fock's at the mean RDMs; the half
  // difference (H_II - H_JJ) / 2 and the coupling H_IJ, each the transition
  // orbital gradient (no core part) at its RDMs.
  std::vector<double> mean(roots, 0.0);
  mean[static_cast<size_t>(i)] = 0.5;
  mean[static_cast<size_t>(j)] = 0.5;
  std::vector<double> half_difference(roots, 0.0);
  half_difference[static_cast<size_t>(i)] = 0.5;
  half_difference[static_cast<size_t>(j)] = -0.5;
  const auto combine = [](const std::vector<double> &a, const std::vector<double> &c) {
    std::vector<double> out(a.size());
    for (size_t k = 0; k < a.size(); ++k) {
      out[k] = 0.5 * (a[k] - c[k]);
    }
    return out;
  };
  const std::vector<double> gamma_x = combine(gamma_root[0], gamma_root[1]);
  const std::vector<double> gamma2_x = combine(gamma2_root[0], gamma2_root[1]);
  const size_t packed = static_cast<size_t>(hessian.size());
  Buffers::device<double> rhs(kIntersectionColumns * packed, res.device());
  const auto place = [&](const double *from, const size_t col) -> Status {
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(rhs.data() + (col * packed), from, packed * sizeof(double),
                                       wwr::wwrMemcpyDeviceToDevice, res.stream())));
    return res.sync();
  };
  {
    FBF_TRY(rdm.rdm12(res, mean, solution.roots.vectors.data(), gamma_device, gamma2_device));
    const response::Fock own =
        FBF_TRY(response::fock(res, molecule, h, b, q, c, ncore, ncas, ints.ppaa, ints.papa,
                               FBF_TRY(intersection_download(res, gamma_device, m)),
                               FBF_TRY(intersection_download(res, gamma2_device, m * m))));
    FBF_TRY(place(own.gradient.data(), 0));
  }
  for (const auto &[gamma, gamma2, col] :
       {std::tuple{&gamma_x, &gamma2_x, size_t{1}}, std::tuple{&gamma_ij, &gamma2_ij, size_t{2}}}) {
    const Buffers::device<double> own = FBF_TRY(transition_orbital_gradient(
        res, molecule, h, b, q, c, ncore, ncas, ints.ppaa, ints.papa, *gamma, *gamma2));
    FBF_TRY(place(own.data(), col));
  }

  // One block solve over the one Hessian, each column held to its tolerance:
  // the conjugate gradients stop on their recurrence, so the true residual,
  // from one more product, is checked here.
  const ZVectors z =
      FBF_TRY(zvectors(res, relaxed, hessian, solution.roots, kIntersectionColumns, rhs.data(),
                       nullptr, CpSolver::Schur, options.tolerance, options.max_iterations));
  for (size_t col = 0; col < kIntersectionColumns; ++col) {
    if (!(z.residuals[col] <= options.tolerance)) {
      return err_numerical("mcscf::df_intersection: Z-vector column {} of roots ({}, {}) ends at "
                           "||H~ z + g|| = {:.3e} after {} products, above its tolerance {:.3e}",
                           col, i, j, z.residuals[col], z.iterations[col], options.tolerance);
    }
  }
  const ZVector z_mean = FBF_TRY(intersection_column(res, z, 0));
  const ZVector z_x = FBF_TRY(intersection_column(res, z, 1));
  const ZVector z_h = FBF_TRY(intersection_column(res, z, 2));

  // Each set of multipliers to its own derivative call: the derivatives of
  // the 2 x 2 Hamiltonian's mean, half difference and coupling.
  const GradientTerms g_mean = FBF_TRY(df_gradient(res, molecule, basis, classes, aux, h, b, l,
                                                   solution, ncore, ncas, mean, &z_mean));
  const GradientTerms g_x = FBF_TRY(df_gradient(res, molecule, basis, classes, aux, h, b, l,
                                                solution, ncore, ncas, half_difference, &z_x));
  const GradientTerms g_h =
      FBF_TRY(df_transition_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                     ncas, gamma_ij, gamma2_ij, &z_h));

  // The 2 x 2 at the solution, from the sigma: H_KL = c_K . H_0 c_L, H_IJ
  // symmetrized (the core energy cancels from H_II - H_JJ). Its eigenvectors
  // are the solution's roots turned by phi, tan 2 phi = H_IJ / ((H_II -
  // H_JJ) / 2), |phi| <= pi / 4.
  Buffers::device<double> products(2 * dim, res.device());
  FBF_TRY(sigma.apply(res, 1, ci_i, products.data()));
  FBF_TRY(sigma.apply(res, 1, ci_j, products.data() + dim));
  const auto contract = [&](const double *bra, const double *ket) -> Result<double> {
    double value = 0.0;
    FBF_TRY(gpu_status(wwr::dot<double, int32_t>(res.blas().get(), static_cast<int32_t>(dim), bra,
                                                 1, ket, 1, &value)));
    FBF_TRY(res.sync());
    return value;
  };
  const double h_ii = FBF_TRY(contract(ci_i, products.data()));
  const double h_jj = FBF_TRY(contract(ci_j, products.data() + dim));
  const double h_ij_value = 0.5 * (FBF_TRY(contract(ci_i, products.data() + dim)) +
                                   FBF_TRY(contract(ci_j, products.data())));
  const double half_gap = 0.5 * (h_ii - h_jj);
  const double two_phi =
      half_gap == 0.0 ? (h_ij_value == 0.0 ? 0.0 : std::copysign(std::numbers::pi / 2, h_ij_value))
                      : std::atan(h_ij_value / half_gap);
  const double cos_2phi = std::cos(two_phi);
  const double sin_2phi = std::sin(two_phi);

  // The eigenvectors' derivatives: dH'_II = mean + cos 2phi x + sin 2phi h,
  // dH'_JJ = mean - (the same), dH'_IJ = -sin 2phi x + cos 2phi h.
  const size_t n3 = g_mean.total.size();
  Intersection out{.g_i = std::vector<double>(n3),
                   .g_j = std::vector<double>(n3),
                   .h_ij = std::vector<double>(n3),
                   .iterations = std::ranges::max(z.iterations),
                   .residual = std::ranges::max(z.residuals)};
  for (size_t k = 0; k < n3; ++k) {
    const double split = (cos_2phi * g_x.total[k]) + (sin_2phi * g_h.total[k]);
    out.g_i[k] = g_mean.total[k] + split;
    out.g_j[k] = g_mean.total[k] - split;
    out.h_ij[k] = (cos_2phi * g_h.total[k]) - (sin_2phi * g_x.total[k]);
  }
  return out;
}

} // namespace free_breakfast::mcscf
