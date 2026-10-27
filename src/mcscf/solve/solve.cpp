/// @file
/// @brief solve: the macro loop over casscf_integrals, ci::solve,
/// ci::Rdm::rdm12 and fock, and the augmented-Hessian orbital step over
/// Hessian under a trust radius.

module;

// The preconditioner's launcher, defined in ah_precondition.cu: declared in
// the GMF so it keeps external linkage (see the header).
#include "ah_precondition_bridge.h"

module free_breakfast.mcscf.solve;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr; // gemv, dgmm, nrm2, scal, gpu_status
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.expm;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.integrals.active_space;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.ci.solver;
import free_breakfast.response.fock;
import free_breakfast.response.hessian;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// A rise of the weighted energy past this rejects a step. Not 0: near
// convergence the energy's own noise (the screened J and K, the CI residual)
// is ~1e-12, and a rejection there would only shrink the radius for nothing.
constexpr double kEnergyRiseTolerance = 1e-11;
// The CI's residual tolerance follows ||g|| of the last iteration as
// min(kCiLoosest, kCiPerGradient ||g||^2), held at or above options.ci's: the
// steps converge quadratically, so the CI must keep ahead of the next ||g||,
// not the last.
constexpr double kCiLoosest = 1e-8;
constexpr double kCiPerGradient = 1e-4;
// The augmented Hessian's subspace solve: an absolute residual of
// kAhPerGradient ||g_model||, so the step is good to ~1% at any ||g||. Q
// holds at most kAhMaxSubspace columns, then restarts on the current step's
// direction; a micro iteration adds at most kAhMaxIterations.
constexpr double kAhPerGradient = 1e-2;
constexpr int32_t kAhMaxSubspace = 40;
constexpr int32_t kAhMaxIterations = 50;
// The preconditioner's denominator mu - diag(H) is held off zero here.
constexpr double kAhDenominatorFloor = 1e-8;
// A correction re-orthogonalized against Q to this norm or below has nothing
// new in it, and the solve stops on the solution it has: davidson_solve's
// default floor, which the subspace solve replaced.
constexpr double kAhLinearDependenceFloor = 1e-12;
// Micro iterations stop once the model gradient is this fraction of ||g||.
constexpr double kMicroGradientFraction = 0.1;
// The trust radius doubles when an accepted step's energy change is within
// this fraction of the model's prediction.
constexpr double kTrustAgreement = 0.25;
// A step along negative curvature, x.H~x < 0, has no minimum in the model, so
// the radius alone sets its length, and the model says nothing of where the
// surface turns up. It takes the whole radius only when the model has earned
// it: the last accepted step agreed with its prediction (kTrustAgreement, or
// there is none yet) and the gradient drives this one, |g.x| at least
// 1 / kCurvatureDominance of |x.H~x| / 2. Otherwise it is cut to
// kNegativeCurvatureFraction of the radius, the length two rejections would
// have halved it to, without the two macro iterations.
constexpr double kNegativeCurvatureFraction = 0.25;
constexpr double kCurvatureDominance = 10.0;
// The CI response's conjugate gradients stop at a residual this fraction of
// the right-hand side's, or after this many sigmas per root. Two roots closer
// than kResponseGapFloor (hartree) leave their mixing out.
constexpr double kResponseTolerance = 1e-8;
constexpr int32_t kResponseMaxIterations = 200;
constexpr double kResponseGapFloor = 1e-8;

// @p count doubles from the device into a host vector, waited for.
Result<std::vector<double>> casscf_download(DeviceResources &res, const double *const from,
                                            const size_t count) {
  std::vector<double> to(count);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to.data(), from, count * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return to;
}

// The host vector @p from onto the device, waited for (so it may go at once).
Status casscf_upload(DeviceResources &res, std::span<const double> from, double *const to) {
  if (from.empty()) {
    return {};
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to, from.data(), from.size_bytes(),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  return res.sync();
}

double casscf_dot(std::span<const double> a, std::span<const double> b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// The macro iteration's integrals the CI response contracts, on the host:
// F^I (nmo x nmo) and ppaa, papa (casscf_integrals' layout), all column-major
// as the device's.
struct ResponseIntegrals {
  size_t nmo;
  size_t ncore;
  size_t ncas;
  std::vector<double> fock_inactive;
  std::vector<double> ppaa;
  std::vector<double> papa;

  [[nodiscard]] double fi(const size_t p, const size_t q) const {
    return fock_inactive[p + (nmo * q)];
  }
  // (pq|uv) and (pu|qv), q over every orbital, u and v active.
  [[nodiscard]] double pq_uv(const size_t p, const size_t q, const size_t u, const size_t v) const {
    return ppaa[p + (nmo * (q + (nmo * (u + (ncas * v)))))];
  }
  [[nodiscard]] double pu_qv(const size_t p, const size_t q, const size_t u, const size_t v) const {
    return papa[p + (nmo * (q + (nmo * (u + (ncas * v)))))];
  }
};

// The first derivative of the active-space Hamiltonian along kappa (nmo x nmo,
// column-major), C <- C (1 + kappa): h_eff' into @p h1 (n x n) and (tu|vw)'
// into @p eri1 (n^4), both row-major as ci::Sigma::set_integrals takes them.
// With T = ncore + t,
//   (tu|vw)' = X_tuvw + X_utvw + X_vwtu + X_wvtu, X_tuvw = sum_p kappa_pT (pU|vw),
//   h_eff'_tu = sum_p kappa_pT F^I_pU + kappa_pU F^I_Tp
//             + sum_p,i kappa_pi (4 (pi|tu) - (pt|iu) - (pu|it)),
// the last F^I's change through the core density, i core: no J or K, since
// every integral it takes has two active indices.
void response_integrals(const ResponseIntegrals &ints, std::span<const double> kappa,
                        std::span<double> h1, std::span<double> eri1) {
  const size_t mo = ints.nmo;
  const size_t core = ints.ncore;
  const size_t n = ints.ncas;
  const auto at = [n](size_t t, size_t u, size_t v, size_t w) {
    return (((((t * n) + u) * n) + v) * n) + w;
  };
  std::vector<double> x(n * n * n * n, 0.0);
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      for (size_t v = 0; v < n; ++v) {
        for (size_t w = 0; w < n; ++w) {
          double sum = 0.0;
          for (size_t p = 0; p < mo; ++p) {
            sum += kappa[p + (mo * (core + t))] * ints.pq_uv(p, core + u, v, w);
          }
          x[at(t, u, v, w)] = sum;
        }
      }
    }
  }
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      for (size_t v = 0; v < n; ++v) {
        for (size_t w = 0; w < n; ++w) {
          eri1[at(t, u, v, w)] =
              x[at(t, u, v, w)] + x[at(u, t, v, w)] + x[at(v, w, t, u)] + x[at(w, v, t, u)];
        }
      }
      double sum = 0.0;
      for (size_t p = 0; p < mo; ++p) {
        sum += (kappa[p + (mo * (core + t))] * ints.fi(p, core + u)) +
               (kappa[p + (mo * (core + u))] * ints.fi(core + t, p));
        for (size_t i = 0; i < core; ++i) {
          sum += kappa[p + (mo * i)] *
                 ((4.0 * ints.pq_uv(p, i, t, u)) - ints.pu_qv(p, i, t, u) - ints.pu_qv(p, i, u, t));
        }
      }
      h1[(t * n) + u] = sum;
    }
  }
}

// The adjoint of response_integrals: the antisymmetric nmo x nmo
// (column-major) A = G - G^T, G_pq = d/dkappa_pq of
// sum_tu h_eff'_tu gamma_tu + sum_tuvw (tu|vw)'_tuvw gamma2_tuvw for the
// row-major @p gamma and @p gamma2, so that A's packed pairs are that sum's
// gradient in x (kappa_pq = x_pq = -kappa_qp).
void response_gradient(const ResponseIntegrals &ints, std::span<const double> gamma,
                       std::span<const double> gamma2, std::span<double> a) {
  const size_t mo = ints.nmo;
  const size_t core = ints.ncore;
  const size_t n = ints.ncas;
  const auto at = [n](size_t t, size_t u, size_t v, size_t w) {
    return (((((t * n) + u) * n) + v) * n) + w;
  };
  // d/dX of the two-electron part: X enters (tu|vw)' four times.
  std::vector<double> y(n * n * n * n);
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      for (size_t v = 0; v < n; ++v) {
        for (size_t w = 0; w < n; ++w) {
          y[at(t, u, v, w)] = gamma2[at(t, u, v, w)] + gamma2[at(u, t, v, w)] +
                              gamma2[at(v, w, t, u)] + gamma2[at(v, w, u, t)];
        }
      }
    }
  }
  std::vector<double> g(mo * mo, 0.0);
  for (size_t p = 0; p < mo; ++p) {
    for (size_t t = 0; t < n; ++t) {
      double sum = 0.0;
      for (size_t u = 0; u < n; ++u) {
        sum += (gamma[(t * n) + u] * ints.fi(p, core + u)) +
               (gamma[(u * n) + t] * ints.fi(core + u, p));
        for (size_t v = 0; v < n; ++v) {
          for (size_t w = 0; w < n; ++w) {
            sum += y[at(t, u, v, w)] * ints.pq_uv(p, core + u, v, w);
          }
        }
      }
      g[p + (mo * (core + t))] += sum;
    }
    for (size_t i = 0; i < core; ++i) {
      double sum = 0.0;
      for (size_t t = 0; t < n; ++t) {
        for (size_t u = 0; u < n; ++u) {
          sum += gamma[(t * n) + u] *
                 ((4.0 * ints.pq_uv(p, i, t, u)) - ints.pu_qv(p, i, t, u) - ints.pu_qv(p, i, u, t));
        }
      }
      g[p + (mo * i)] += sum;
    }
  }
  for (size_t q = 0; q < mo; ++q) {
    for (size_t p = 0; p < mo; ++p) {
      a[p + (mo * q)] = g[p + (mo * q)] - g[q + (mo * p)];
    }
  }
}

Result<Solution>
solve(DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
      const SphericalMatrixAO &h, const std::shared_ptr<const Buffers::device<double>> &u,
      const std::span<CoulombPairClass> j_classes, const std::span<OrderedPairClass> k_classes,
      const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas, const int32_t nelecas,
      const int32_t nroots, const std::span<const double> weights, const Options &options) {
  const auto nmo = static_cast<int32_t>(c.n());
  if (c.n() != basis.nao_sphe() || c.dev_idx() != res.dev_idx()) {
    return err_config("mcscf::solve: c must be {0} x {0}, basis's spherical order, on device {1} "
                      "(got {2} on device {3})",
                      basis.nao_sphe(), res.dev_idx(), c.n(), c.dev_idx());
  }
  if (ncore < 0 || ncas < 1 || ncore + ncas > nmo || nelecas < 0 || nelecas > 2 * ncas) {
    return err_config("mcscf::solve: needs ncore >= 0, ncas >= 1, ncore + ncas <= {} and nelecas "
                      "in [0, 2 ncas] (ncore {}, ncas {}, nelecas {})",
                      nmo, ncore, ncas, nelecas);
  }
  if (!(options.conv_tol > 0.0) || !(options.conv_tol_grad > 0.0) || options.max_cycle < 1 ||
      !(options.trust_radius > 0.0) || options.max_micro < 1) {
    return err_config("mcscf::solve: conv_tol, conv_tol_grad and trust_radius must be above 0 and "
                      "max_cycle and max_micro at least 1 ({}, {}, {}, {}, {})",
                      options.conv_tol, options.conv_tol_grad, options.trust_radius,
                      options.max_cycle, options.max_micro);
  }
  if (nroots < 1) {
    return err_config("mcscf::solve: nroots ({}) must be at least 1", nroots);
  }
  const auto roots_count = static_cast<size_t>(nroots);
  std::vector<double> w(weights.begin(), weights.end());
  if (w.empty()) {
    w.assign(roots_count, 1.0 / static_cast<double>(nroots));
  }
  const double weight_sum = std::accumulate(w.begin(), w.end(), 0.0);
  if (w.size() != roots_count || std::ranges::any_of(w, [](double x) { return !(x >= 0.0); }) ||
      std::abs(weight_sum - 1.0) > 1e-12) {
    return err_config("mcscf::solve: needs {} weights, each >= 0, summing to 1 (got {}, summing "
                      "to {})",
                      nroots, w.size(), weight_sum);
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const int32_t n_beta = nelecas / 2;
  const ci::StringSet alpha = FBF_TRY(ci::StringSet::create(ncas, nelecas - n_beta));
  const ci::StringSet beta = FBF_TRY(ci::StringSet::create(ncas, n_beta));
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  ci::Rdm ci_rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(sigma.size());
  if (roots_count > dim) {
    return err_config("mcscf::solve: nroots ({}) exceeds the {} determinants", nroots, dim);
  }

  const auto mo = static_cast<size_t>(nmo);
  const auto n = static_cast<size_t>(ncas);
  const int32_t n_pairs = (ncore * (nmo - ncore)) + (ncas * (nmo - ncore - ncas));
  const auto pairs = static_cast<size_t>(n_pairs);

  // C now, and C at the last accepted iteration, which a step starts from.
  SphericalMatrixAO c_now(res, basis);
  SphericalMatrixAO c_accepted(res, basis);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c_now.data(), c.data(), mo * mo * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  Buffers::device<double> diagonal(dim, res.device());
  Buffers::device<double> gamma_device(n * n, res.device());
  Buffers::device<double> gamma2_device(n * n * n * n, res.device());
  // The step: x packed, kappa(x), its skew_eig and exp(t kappa).
  Buffers::device<double> step(std::max<size_t>(pairs, 1), res.device());
  Buffers::device<double> kappa(mo * mo, res.device());
  Buffers::device<wwr::wwrDoubleComplex> kappa_u(mo * mo, res.device());
  Buffers::device<double> kappa_w(mo, res.device());
  Buffers::device<double> rotation(mo * mo, res.device());
  // The augmented Hessian's subspace, every buffer for the solve. Q, at most
  // q_max orthonormal columns, and beside it [g_model, H~ Q], the model
  // gradient first so one gemv takes a new column's overlaps with both; diag(H)
  // for the preconditioner; a residual and its correction; the step dx and
  // H~ dx, read back in one copy; and the projected problem, (q + 1)^2, its
  // eigenvalues, and the small vectors in and out of the gemvs.
  const int32_t q_max = std::max(1, std::min(n_pairs, kAhMaxSubspace));
  const auto q_width = static_cast<size_t>(q_max);
  const size_t ah_length = std::max<size_t>(pairs, 1);
  Buffers::device<double> ah_basis(ah_length * q_width, res.device());
  Buffers::device<double> ah_image(ah_length * (q_width + 1), res.device());
  Buffers::device<double> ah_diagonal(ah_length, res.device());
  Buffers::device<double> ah_residual(ah_length, res.device());
  Buffers::device<double> ah_correction(ah_length, res.device());
  Buffers::device<double> ah_step(2 * ah_length, res.device());
  Buffers::device<double> ah_projected((q_width + 1) * (q_width + 1), res.device());
  Buffers::device<double> ah_ritz(q_width + 1, res.device());
  Buffers::device<double> ah_coefficients(q_width + 1, res.device());
  Buffers::device<double> ah_overlaps(q_width + 2, res.device());
  // The CI response's: a second sigma, for H's derivative along kappa; the CI
  // blocks b = H' c, delta and c +- delta (dim x nroots each); conjugate
  // gradients' three vectors; the roots' overlaps; kappa, then the gradient's
  // antisymmetric matrix; the packed gradient; and the RDMs of c +- delta.
  ci::Sigma response_sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  const size_t ci_block = dim * roots_count;
  Buffers::device<double> ci_b(ci_block, res.device());
  Buffers::device<double> ci_delta(ci_block, res.device());
  Buffers::device<double> ci_pm(2 * ci_block, res.device());
  Buffers::device<double> cg_vectors(3 * dim, res.device());
  Buffers::device<double> ci_overlaps(roots_count, res.device());
  Buffers::device<double> response_matrix(mo * mo, res.device());
  Buffers::device<double> response_packed(ah_length, res.device());
  Buffers::device<double> response_gamma(2 * n * n, res.device());
  Buffers::device<double> response_gamma2(2 * n * n * n * n, res.device());

  // C_now = C_accepted exp(t kappa), from kappa_u and kappa_w.
  const auto rotate = [&](const double t) -> Status {
    FBF_TRY(skew_expm<double>(res, nmo, kappa_u.data(), kappa_w.data(), t, rotation.data()));
    return gpu_status(gemm(blas, nmo, &kOne<double>, c_accepted.data(), rotation.data(),
                           &kZero<double>, c_now.data()));
  };

  std::optional<ci::Roots> roots;
  std::vector<Iteration> log;
  bool have_accepted = false;
  double e_accepted = 0.0;
  double radius = options.trust_radius;
  double last_gradient = std::numeric_limits<double>::infinity();
  // The step from the accepted orbitals, t x: E changes by
  // t g.x + t^2 x.Hx / 2 in the model; t is 1, halved on each rejection.
  bool have_step = false;
  // Whether the last accepted step's energy change agreed with its model:
  // true before the first, false after a rejection.
  bool model_agreed = true;
  double step_gx = 0.0;
  double step_xhx = 0.0;
  double step_norm = 0.0;
  double step_t = 1.0;
  // The step x and the model's H~ x.
  std::vector<double> x_total(pairs, 0.0);
  std::vector<double> hx_total(pairs, 0.0);

  for (int32_t cycle = 1; cycle <= options.max_cycle; ++cycle) {
    // 1. The active-space Hamiltonian and ppaa, papa over C.
    CasscfIntegrals ints =
        FBF_TRY(casscf_integrals(res, basis, molecule, h, u, j_classes, k_classes, c_now, ncore,
                                 ncas, options.tau_j, options.tau_k));

    // 2. The CI, from the last roots. The sigma takes host arrays, row-major,
    //    and the Hamiltonian's are column-major, which needs no reorder: h_eff
    //    is symmetric, and (t, u, v, w) at t + n(u + n(v + n w)) is row-major
    //    (w, v, u, t), with (wv|ut) = (tu|vw) for real ERIs. That symmetry is
    //    exact only to roundoff and screening: (tu|vw) and (vw|tu) are
    //    separate screened J builds, and C^T J C is not bitwise symmetric.
    //    set_integrals does not confine it to one triangle (Σ_u (vu|uw),
    //    (ii|jj) and (ij|ji) run over every index), so the CI sees the other
    //    side of that asymmetry, which is far below its tolerance.
    {
      const std::vector<double> h_eff =
          FBF_TRY(casscf_download(res, ints.hamiltonian.h_eff.data(), n * n));
      const std::vector<double> eri =
          FBF_TRY(casscf_download(res, ints.hamiltonian.eri.data(), n * n * n * n));
      FBF_TRY(sigma.set_integrals(res, ints.hamiltonian.e_core, h_eff, eri));
    }
    FBF_TRY(sigma.diagonal(res, diagonal.data()));
    ci::SolverOptions ci = options.ci;
    ci.davidson.residual_tolerance =
        std::max(options.ci.davidson.residual_tolerance,
                 std::min(kCiLoosest, kCiPerGradient * last_gradient * last_gradient));
    // Warm from the last iteration's roots, rejected or not: the nearest H's
    // (see the README). The first is cold, from ci::solve's own guesses.
    const std::span<const double> start =
        roots ? std::span<const double>{roots->vectors.data(), ci_block}
              : std::span<const double>{};
    ci::Roots solved =
        FBF_TRY(ci::solve(res, sigma, alpha, beta, diagonal.data(), nroots, ci, start));
    roots = std::move(solved);

    // 3. The weighted RDMs, the energy and the gradient.
    FBF_TRY(ci_rdm.rdm12(res, w, roots->vectors.data(), gamma_device.data(), gamma2_device.data()));
    std::vector<double> gamma = FBF_TRY(casscf_download(res, gamma_device.data(), n * n));
    std::vector<double> gamma2 = FBF_TRY(casscf_download(res, gamma2_device.data(), n * n * n * n));
    response::Fock fock =
        FBF_TRY(response::fock(res, basis, molecule, h, u, j_classes, k_classes, c_now, ncore, ncas,
                               ints.ppaa, gamma, gamma2, options.tau_j, options.tau_k));
    const std::vector<double> g = FBF_TRY(casscf_download(res, fock.gradient.data(), pairs));
    const double g_norm = std::sqrt(casscf_dot(g, g));
    last_gradient = g_norm;
    const double energy = fock.energy;
    const double change = have_accepted ? energy - e_accepted : 0.0;
    Iteration record{
        energy, change, g_norm, radius, roots->davidson.iterations, roots->fp32_iterations,
        0,      0,      true};

    // 4. Convergence, then the rejection of a rise.
    const bool converged = n_pairs == 0 || (have_accepted && std::abs(change) < options.conv_tol &&
                                            g_norm < options.conv_tol_grad);
    if (!converged && have_accepted && change > kEnergyRiseTolerance) {
      step_t *= 0.5;
      model_agreed = false;
      radius = step_t * step_norm;
      record.accepted = false;
      record.trust_radius = radius;
      log.push_back(record);
      FBF_LOG_DEBUG("mcscf::solve: macro {} E {:.12f} dE {:.3e} |g| {:.3e} rejected, step {:.3e}",
                    cycle, energy, change, g_norm, step_t * step_norm);
      FBF_TRY(rotate(step_t));
      continue;
    }
    if (have_step) {
      const double predicted = (step_t * step_gx) + (0.5 * step_t * step_t * step_xhx);
      model_agreed = predicted < 0.0 && std::abs((change / predicted) - 1.0) < kTrustAgreement;
      if (model_agreed) {
        radius = std::min(options.trust_radius, 2.0 * radius);
        record.trust_radius = radius;
      }
    }
    have_accepted = true;
    e_accepted = energy;
    if (converged) {
      log.push_back(record);
      FBF_LOG_INFO("mcscf::solve: converged in {} macro iterations, E {:.12f}, |g| {:.3e}", cycle,
                   energy, g_norm);
      std::vector<double> root_gammas(roots_count * n * n);
      Buffers::device<double> root_gamma(n * n, res.device());
      for (size_t r = 0; r < roots_count; ++r) {
        FBF_TRY(sigma.rdm1(res, roots->vectors.data() + (r * dim), root_gamma.data()));
        const std::vector<double> one = FBF_TRY(casscf_download(res, root_gamma.data(), n * n));
        std::ranges::copy(one, root_gammas.begin() + static_cast<std::ptrdiff_t>(r * n * n));
      }
      return Solution{
          std::move(c_now),  std::move(*roots),      energy,        g_norm, std::move(gamma),
          std::move(gamma2), std::move(root_gammas), std::move(log)};
    }
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(c_accepted.data(), c_now.data(), mo * mo * sizeof(double),
                                       wwr::wwrMemcpyDeviceToDevice, stream)));

    // 5. The step: augmented-Hessian solves of the quadratic model
    //    E + g.x + x.H~x / 2, H~ = H + R the orbital Hessian with the CI
    //    relaxed (below), each from the model gradient g + H~ x the last left,
    //    until it is small, the radius is reached or max_micro is. H~ is fixed
    //    for the macro iteration, so the solves share one subspace Q (see the
    //    README).
    response::Hessian hessian = FBF_TRY(
        response::Hessian::create(res, basis, h, u, j_classes, k_classes, c_now, ncore, ncas,
                                  ints.ppaa, ints.papa, fock, options.tau_j, options.tau_k));
    // diag(H) for the preconditioner, copied once for the macro iteration, and
    // read back for its scale.
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ah_diagonal.data(), hessian.diagonal().data(),
                                           pairs * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                           stream)));
    const std::vector<double> h_diagonal =
        FBF_TRY(casscf_download(res, hessian.diagonal().data(), pairs));
    double h_scale = 1.0;
    for (const double d : h_diagonal) {
      h_scale = std::max(h_scale, std::abs(d));
    }

    // The CI response R x = H_oc delta (README.md, "The CI response"),
    // delta_r = -(H_0 - E_r)^-1 (1 - c_r c_r^T) H'(x) c_r, so that H~ = H + R
    // is the Hessian of the energy with the CI relaxed. The integrals it
    // contracts come to the host once per macro iteration.
    const ResponseIntegrals response_ints{
        mo,
        static_cast<size_t>(ncore),
        n,
        FBF_TRY(casscf_download(res, fock.fock_inactive.data(), mo * mo)),
        FBF_TRY(casscf_download(res, ints.ppaa.data(), mo * mo * n * n)),
        FBF_TRY(casscf_download(res, ints.papa.data(), mo * mo * n * n))};
    const double *const ci_c = roots->vectors.data();
    const auto ci_root = [&](auto *base, const size_t r) { return base + (r * dim); };
    const auto dim32 = static_cast<int32_t>(dim);
    // v <- v - C (C^T v): off every root.
    const auto ci_project = [&](double *v) -> Status {
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, dim32, nroots,
                                                    &kOne<double>, ci_c, dim32, v, 1,
                                                    &kZero<double>, ci_overlaps.data(), 1)));
      return gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, dim32, nroots,
                                                   &kNegOne<double>, ci_c, dim32,
                                                   ci_overlaps.data(), 1, &kOne<double>, v, 1));
    };
    // y += R x.
    const auto respond = [&](const double *x, double *y) -> Status {
      // H'(x) from kappa(x), and b_r = H' c_r for every root.
      FBF_TRY(response::unpack_orbital_pairs(res, nmo, ncore, ncas, x, response_matrix.data()));
      const std::vector<double> kappa_host =
          FBF_TRY(casscf_download(res, response_matrix.data(), mo * mo));
      std::vector<double> h1(n * n);
      std::vector<double> eri1(n * n * n * n);
      response_integrals(response_ints, kappa_host, h1, eri1);
      FBF_TRY(response_sigma.set_integrals(res, 0.0, h1, eri1));
      FBF_TRY(response_sigma.apply(res, nroots, ci_c, ci_b.data()));

      // delta_r: conjugate gradients on (H_0 - E_r) off every root, and the
      // other roots' part exactly, -(c_s . b_r) / (E_s - E_r) c_s.
      double *const cg_r = cg_vectors.data();
      double *const cg_p = cg_r + dim;
      double *const cg_ap = cg_p + dim;
      for (size_t r = 0; r < roots_count; ++r) {
        const double e_r = roots->energies[r];
        double *const delta = ci_root(ci_delta.data(), r);
        FBF_TRY(gpu_status(wwr::wwrMemsetAsync(delta, 0, dim * sizeof(double), stream)));
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(cg_r, ci_root(ci_b.data(), r), dim * sizeof(double),
                                               wwr::wwrMemcpyDeviceToDevice, stream)));
        FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, dim32, &kNegOne<double>, cg_r, 1)));
        FBF_TRY(ci_project(cg_r));
        double rhs_norm = 0.0;
        FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, dim32, cg_r, 1, &rhs_norm)));
        double rr = rhs_norm * rhs_norm;
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(cg_p, cg_r, dim * sizeof(double),
                                               wwr::wwrMemcpyDeviceToDevice, stream)));
        for (int32_t it = 0; it < kResponseMaxIterations && rr > 0.0; ++it) {
          FBF_TRY(sigma.apply(res, 1, cg_p, cg_ap));
          const double neg_e = -e_r;
          FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, dim32, &neg_e, cg_p, 1, cg_ap, 1)));
          FBF_TRY(ci_project(cg_ap));
          double pap = 0.0;
          FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, dim32, cg_p, 1, cg_ap, 1, &pap)));
          // H_0 - E_r is positive off the roots in their spin; a direction it
          // is not positive along (roundoff's other spins) ends the solve.
          if (!(pap > 0.0)) {
            break;
          }
          const double alpha_step = rr / pap;
          const double neg_alpha = -alpha_step;
          FBF_TRY(
              gpu_status(wwr::axpy<double, int32_t>(blas, dim32, &alpha_step, cg_p, 1, delta, 1)));
          FBF_TRY(
              gpu_status(wwr::axpy<double, int32_t>(blas, dim32, &neg_alpha, cg_ap, 1, cg_r, 1)));
          double r_norm = 0.0;
          FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, dim32, cg_r, 1, &r_norm)));
          const double rr_new = r_norm * r_norm;
          if (r_norm <= kResponseTolerance * rhs_norm) {
            break;
          }
          const double beta_step = rr_new / rr;
          FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, dim32, &beta_step, cg_p, 1)));
          FBF_TRY(
              gpu_status(wwr::axpy<double, int32_t>(blas, dim32, &kOne<double>, cg_r, 1, cg_p, 1)));
          rr = rr_new;
        }
        for (size_t s = 0; s < roots_count; ++s) {
          const double gap = roots->energies[s] - e_r;
          if (s == r || std::abs(gap) < kResponseGapFloor) {
            continue;
          }
          double overlap = 0.0;
          FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, dim32, ci_root(ci_c, s), 1,
                                                       ci_root(ci_b.data(), r), 1, &overlap)));
          const double coefficient = -overlap / gap;
          FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, dim32, &coefficient, ci_root(ci_c, s),
                                                        1, delta, 1)));
        }
      }

      // R x = H_oc delta, the gradient in x of h_eff'.gamma_T +
      // 1/2 (tu|vw)'.Gamma_T, gamma_T = sum_r w_r (<c_r|E|delta_r> +
      // <delta_r|E|c_r>): half the difference of the RDMs of c + delta and
      // c - delta.
      double *const plus = ci_pm.data();
      double *const minus = plus + ci_block;
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim32, nroots, &kOne<double>, ci_c, dim32,
          &kOne<double>, ci_delta.data(), dim32, plus, dim32)));
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim32, nroots, &kOne<double>, ci_c, dim32,
          &kNegOne<double>, ci_delta.data(), dim32, minus, dim32)));
      FBF_TRY(ci_rdm.rdm12(res, w, plus, response_gamma.data(), response_gamma2.data()));
      FBF_TRY(ci_rdm.rdm12(res, w, minus, response_gamma.data() + (n * n),
                           response_gamma2.data() + (n * n * n * n)));
      const std::vector<double> g1 =
          FBF_TRY(casscf_download(res, response_gamma.data(), 2 * n * n));
      const std::vector<double> g2 =
          FBF_TRY(casscf_download(res, response_gamma2.data(), 2 * n * n * n * n));
      std::vector<double> gamma_t(n * n);
      std::vector<double> gamma2_t(n * n * n * n);
      for (size_t i = 0; i < gamma_t.size(); ++i) {
        gamma_t[i] = 0.5 * (g1[i] - g1[i + gamma_t.size()]);
      }
      for (size_t i = 0; i < gamma2_t.size(); ++i) {
        gamma2_t[i] = 0.25 * (g2[i] - g2[i + gamma2_t.size()]);
      }
      std::vector<double> a(mo * mo);
      response_gradient(response_ints, gamma_t, gamma2_t, a);
      FBF_TRY(casscf_upload(res, a, response_matrix.data()));
      FBF_TRY(response::pack_orbital_pairs(res, nmo, ncore, ncas, response_matrix.data(),
                                           response_packed.data()));
      return gpu_status(wwr::axpy<double, int32_t>(blas, n_pairs, &kOne<double>,
                                                   response_packed.data(), 1, y, 1));
    };

    // Q's column j at ah_basis + j n, H~ q_j at ah_image + (j + 1) n, and the
    // model gradient at ah_image. The small problem's pieces on the host:
    // A = Q^T H~ Q (q_max x q_max, both triangles) and b = Q^T g_model.
    const auto column = [&](double *base, const int32_t j) {
      return base + (static_cast<size_t>(j) * pairs);
    };
    // H~ q_j into its column of ah_image: H, then R.
    const auto apply = [&](const int32_t j) -> Status {
      const double *const x = column(ah_basis.data(), j);
      double *const y = column(ah_image.data(), j + 1);
      FBF_TRY(hessian.apply(res, 1, x, y));
      return respond(x, y);
    };
    // [Q, H~ Q] times the host coefficients c (q of them) into ah_step's two
    // columns, read back as (Q c, H~ Q c).
    const auto expand = [&](const int32_t q,
                            std::span<const double> c) -> Result<std::vector<double>> {
      FBF_TRY(casscf_upload(res, c, ah_coefficients.data()));
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, n_pairs, q, &kOne<double>, ah_basis.data(), n_pairs,
          ah_coefficients.data(), 1, &kZero<double>, ah_step.data(), 1)));
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, n_pairs, q, &kOne<double>, column(ah_image.data(), 1), n_pairs,
          ah_coefficients.data(), 1, &kZero<double>, ah_step.data() + pairs, 1)));
      return casscf_download(res, ah_step.data(), 2 * pairs);
    };
    std::vector<double> a_host(q_width * q_width);
    std::vector<double> b_host(q_width);
    std::vector<double> projected;
    int32_t q = 0;

    std::vector<double> g_model = g;
    std::ranges::fill(x_total, 0.0);
    std::ranges::fill(hx_total, 0.0);
    for (int32_t micro = 1; micro <= options.max_micro; ++micro) {
      // g_model into ah_image's first column, and b = Q^T g_model.
      FBF_TRY(casscf_upload(res, g_model, ah_image.data()));
      if (q > 0) {
        FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
            blas, wwr::WWRBLAS_OP_T, n_pairs, q, &kOne<double>, ah_basis.data(), n_pairs,
            ah_image.data(), 1, &kZero<double>, ah_overlaps.data(), 1)));
        const std::vector<double> b =
            FBF_TRY(casscf_download(res, ah_overlaps.data(), static_cast<size_t>(q)));
        std::ranges::copy(b, b_host.begin());
      }
      const double g_model_norm = std::sqrt(casscf_dot(g_model, g_model));
      const double tolerance = std::clamp(kAhPerGradient * g_model_norm / h_scale, 1e-14, 1e-4);

      // The lowest root mu, (u_0, u), of [[0, b^T], [b, A]], and Q grown by
      // its preconditioned residual until that is below the tolerance, the
      // correction has nothing new, or kAhMaxIterations columns are added.
      double mu = 0.0;
      std::vector<double> root;
      for (int32_t added = 0;; ++added) {
        const int32_t k = q + 1;
        const auto kz = static_cast<size_t>(k);
        projected.assign(kz * kz, 0.0);
        for (size_t j = 0; j + 1 < kz; ++j) {
          projected[j + 1] = b_host[j];
          for (size_t i = j; i + 1 < kz; ++i) {
            projected[(i + 1) + ((j + 1) * kz)] = a_host[i + (j * q_width)];
          }
        }
        FBF_TRY(casscf_upload(res, projected, ah_projected.data()));
        FBF_TRY(diagonalize<double>(res, k, ah_projected.data(), ah_ritz.data()));
        const std::vector<double> ritz = FBF_TRY(casscf_download(res, ah_ritz.data(), kz));
        root = FBF_TRY(casscf_download(res, ah_projected.data(), kz));
        mu = ritz.front();
        const double spectrum = std::max(std::abs(ritz.front()), std::abs(ritz.back()));

        // r = u_0 g_model + H~ Q u - mu Q u: its first row, b.u - mu
        // u_0, is 0 by the small problem's own equation.
        FBF_TRY(casscf_upload(res, root, ah_coefficients.data()));
        FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, n_pairs, k, &kOne<double>, ah_image.data(), n_pairs,
            ah_coefficients.data(), 1, &kZero<double>, ah_residual.data(), 1)));
        if (q > 0) {
          const double neg_mu = -mu;
          FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
              blas, wwr::WWRBLAS_OP_N, n_pairs, q, &neg_mu, ah_basis.data(), n_pairs,
              ah_coefficients.data() + 1, 1, &kOne<double>, ah_residual.data(), 1)));
        }
        double residual_norm = 0.0;
        FBF_TRY(gpu_status(
            wwr::nrm2<double, int32_t>(blas, n_pairs, ah_residual.data(), 1, &residual_norm)));
        if (residual_norm <= tolerance * std::max(std::abs(mu), spectrum) ||
            added == kAhMaxIterations) {
          break;
        }
        FBF_TRY(gpu_status(device::ah_precondition(stream, n_pairs, mu, kAhDenominatorFloor,
                                                   ah_diagonal.data(), ah_residual.data(),
                                                   ah_correction.data())));

        // Full: restart Q on the current step's direction Q u / ||u||, whose
        // H~ image is a gemv of the old one's, so no product is lost;
        // the root is then (u_0, ||u||).
        if (q == q_max) {
          double u_norm = 0.0;
          for (size_t i = 1; i < kz; ++i) {
            u_norm += root[i] * root[i];
          }
          u_norm = std::sqrt(u_norm);
          if (u_norm == 0.0) {
            return err_numerical("mcscf::solve: the augmented Hessian's lowest root has no "
                                 "component in its subspace to restart on (macro iteration {})",
                                 cycle);
          }
          std::vector<double> c(root.begin() + 1, root.end());
          for (double &ci : c) {
            ci /= u_norm;
          }
          FBF_TRY(expand(q, c));
          double ac = 0.0;
          double bc = 0.0;
          for (size_t i = 0; i < c.size(); ++i) {
            bc += b_host[i] * c[i];
            for (size_t j = 0; j < c.size(); ++j) {
              ac += c[i] * a_host[i + (j * q_width)] * c[j];
            }
          }
          FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ah_basis.data(), ah_step.data(),
                                                 pairs * sizeof(double),
                                                 wwr::wwrMemcpyDeviceToDevice, stream)));
          FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(column(ah_image.data(), 1), ah_step.data() + pairs,
                                                 pairs * sizeof(double),
                                                 wwr::wwrMemcpyDeviceToDevice, stream)));
          a_host[0] = ac;
          b_host[0] = bc;
          root = {root[0], u_norm};
          q = 1;
        }

        // The correction off Q, twice, then normalized as Q's next column.
        for (int32_t pass = 0; pass < 2; ++pass) {
          FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
              blas, wwr::WWRBLAS_OP_T, n_pairs, q, &kOne<double>, ah_basis.data(), n_pairs,
              ah_correction.data(), 1, &kZero<double>, ah_overlaps.data(), 1)));
          FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
              blas, wwr::WWRBLAS_OP_N, n_pairs, q, &kNegOne<double>, ah_basis.data(), n_pairs,
              ah_overlaps.data(), 1, &kOne<double>, ah_correction.data(), 1)));
        }
        double correction_norm = 0.0;
        FBF_TRY(gpu_status(
            wwr::nrm2<double, int32_t>(blas, n_pairs, ah_correction.data(), 1, &correction_norm)));
        if (correction_norm <= kAhLinearDependenceFloor) {
          break;
        }
        const double inverse = 1.0 / correction_norm;
        FBF_TRY(gpu_status(
            wwr::scal<double, int32_t>(blas, n_pairs, &inverse, ah_correction.data(), 1)));
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(column(ah_basis.data(), q), ah_correction.data(),
                                               pairs * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                               stream)));
        FBF_TRY(apply(q));
        ++record.hessian_products;
        // The new column's overlaps with [g_model, H~ Q]: b's new entry
        // and A's new row, mirrored into its column.
        FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
            blas, wwr::WWRBLAS_OP_T, n_pairs, q + 2, &kOne<double>, ah_image.data(), n_pairs,
            column(ah_basis.data(), q), 1, &kZero<double>, ah_overlaps.data(), 1)));
        const std::vector<double> w =
            FBF_TRY(casscf_download(res, ah_overlaps.data(), static_cast<size_t>(q) + 2));
        const auto qz = static_cast<size_t>(q);
        b_host[qz] = w[0];
        for (size_t j = 0; j <= qz; ++j) {
          a_host[qz + (j * q_width)] = w[j + 1];
          a_host[j + (qz * q_width)] = w[j + 1];
        }
        ++q;
      }

      // The step dx = Q u / u_0, and H~ dx from H~ Q: no product
      // past the solve's.
      if (root[0] == 0.0) {
        return err_numerical("mcscf::solve: the augmented Hessian's lowest root has no "
                             "component along the gradient (macro iteration {})",
                             cycle);
      }
      std::vector<double> dx(pairs, 0.0);
      std::vector<double> hdx(pairs, 0.0);
      if (q > 0) {
        std::vector<double> c(root.begin() + 1, root.end());
        for (double &ci : c) {
          ci /= root[0];
        }
        const std::vector<double> both = FBF_TRY(expand(q, c));
        std::copy_n(both.begin(), pairs, dx.begin());
        std::copy_n(both.begin() + static_cast<std::ptrdiff_t>(pairs), pairs, hdx.begin());
      }
      const double dx_norm = std::sqrt(casscf_dot(dx, dx));
      const double remaining = radius - std::sqrt(casscf_dot(x_total, x_total));
      const double s = dx_norm > remaining ? remaining / dx_norm : 1.0;
      for (size_t i = 0; i < pairs; ++i) {
        x_total[i] += s * dx[i];
        hx_total[i] += s * hdx[i];
        g_model[i] += s * hdx[i];
      }
      record.micro_iterations = micro;
      if (s < 1.0 || std::sqrt(casscf_dot(g_model, g_model)) <= kMicroGradientFraction * g_norm) {
        break;
      }
    }
    // A step along negative curvature the model has not earned is cut short
    // (kNegativeCurvatureFraction, above).
    const double curvature = casscf_dot(x_total, hx_total);
    const double length = std::sqrt(casscf_dot(x_total, x_total));
    const double short_length = kNegativeCurvatureFraction * radius;
    if (curvature < 0.0 && length > short_length &&
        (!model_agreed ||
         -0.5 * curvature > kCurvatureDominance * std::abs(casscf_dot(g, x_total)))) {
      const double scale = short_length / length;
      for (size_t i = 0; i < pairs; ++i) {
        x_total[i] *= scale;
        hx_total[i] *= scale;
      }
      FBF_LOG_DEBUG("mcscf::solve: macro {} negative curvature {:.3e}, step cut to {:.3e}", cycle,
                    curvature, short_length);
    }
    have_step = true;
    step_t = 1.0;
    step_gx = casscf_dot(g, x_total);
    step_xhx = casscf_dot(x_total, hx_total);
    step_norm = std::sqrt(casscf_dot(x_total, x_total));
    log.push_back(record);
    FBF_LOG_DEBUG("mcscf::solve: macro {} E {:.12f} dE {:.3e} |g| {:.3e} ci {} + {} micro {} Hx {} "
                  "step {:.3e}",
                  cycle, energy, change, g_norm, record.ci_fp32_iterations, record.ci_iterations,
                  record.micro_iterations, record.hessian_products, step_norm);

    // 6. C <- C exp(kappa(x)).
    FBF_TRY(casscf_upload(res, x_total, step.data()));
    FBF_TRY(response::unpack_orbital_pairs(res, nmo, ncore, ncas, step.data(), kappa.data()));
    FBF_TRY(skew_eig<double>(res, nmo, kappa.data(), kappa_u.data(), kappa_w.data()));
    FBF_TRY(rotate(1.0));
  }
  return err_max_iterations_exceeded(
      "mcscf::solve: not converged in {} macro iterations (last |g| {:.3e})", options.max_cycle,
      last_gradient);
}

} // namespace free_breakfast::mcscf
