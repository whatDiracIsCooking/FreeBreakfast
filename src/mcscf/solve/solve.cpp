/// @file
/// @brief casscf_solve: the macro loop over casscf_integrals, ci_solve,
/// CiRdm::rdm12 and casscf_fock, and the augmented-Hessian orbital step over
/// OrbitalHessian under a trust radius.

module;

// The preconditioner's launcher, defined in ah_precondition.cu: declared in
// the GMF so it keeps external linkage (see the header).
#include "ah_precondition_bridge.h"

module free_breakfast.mcscf.solve;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr; // gemm, dgmm, wwrMemcpy2DAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.davidson;
import free_breakfast.recipes.expm;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.integrals.active_space;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.ci.solver;
import free_breakfast.mcscf.fock;
import free_breakfast.mcscf.hessian;

#include "diagnostics.h"

namespace free_breakfast {

// A rise of the weighted energy past this rejects a step. Not 0: near
// convergence the energy's own noise (the screened J and K, the CI residual)
// is ~1e-12, and a rejection there would only shrink the radius for nothing.
constexpr double kEnergyRiseTolerance = 1e-11;
// The CI's residual tolerance follows ||g|| of the last iteration as
// min(kCiLoosest, kCiPerGradient ||g||), held at or above options.ci's.
constexpr double kCiLoosest = 1e-8;
constexpr double kCiPerGradient = 1e-4;
// The augmented Hessian's Davidson: an absolute residual of
// kAhPerGradient ||g_model||, so the step is good to ~1% at any ||g||.
constexpr double kAhPerGradient = 1e-2;
constexpr int32_t kAhMaxSubspace = 40;
constexpr int32_t kAhMaxIterations = 50;
// The preconditioner's denominator theta - diag(H) is held off zero here.
constexpr double kAhDenominatorFloor = 1e-8;
// Micro iterations stop once the model gradient is this fraction of ||g||.
constexpr double kMicroGradientFraction = 0.1;
// The trust radius doubles when an accepted step's energy change is within
// this fraction of the model's prediction.
constexpr double kTrustAgreement = 0.25;
// The CI response's BFGS pairs: at most this many steps are kept, the oldest
// dropped, and a step is taken only when y.s and s.Bs are at least
// kResponseSafeguard times their factors' norms: the curvature condition, held
// off roundoff.
constexpr size_t kResponseMemory = 8;
constexpr double kResponseSafeguard = 1e-8;

// [[0, g^T], [g, H + D]] over the (n + 1)-vectors (v_0, v), g the model
// gradient on the device and D = R diag(scale) R^T the CI response's
// rank-@p rank correction (R n x rank, column-major): the augmented Hessian
// whose lowest eigenvector gives the step x = v / v_0, (H + D - lambda) x = -g.
// xs, ys (n x columns) and rv (rank x columns) are the caller's, allocated
// once for the solve: held across OrbitalHessian::apply, which takes the
// workspace, so not the workspace's, and a Davidson applies this every
// iteration, so not allocated per call.
struct AugmentedHessian {
  OrbitalHessian &hessian;
  const double *gradient;
  const double *response;
  const double *response_scale;
  int32_t rank;
  int32_t n;
  int32_t columns;
  double *xs;
  double *ys;
  double *rv;
  int32_t products = 0;

  Status apply(DeviceResources &res, const int32_t k, const double *const x, double *const y) {
    if (k > columns) {
      return err_config("casscf_solve: the augmented Hessian's buffers hold {} columns, given {}",
                        columns, k);
    }
    const wwr::wwrblasHandle_t blas = res.blas().get();
    const wwr::wwrStream_t stream = res.stream();
    const int32_t m = n + 1;
    const size_t width = static_cast<size_t>(n) * sizeof(double);
    const size_t pitch = static_cast<size_t>(m) * sizeof(double);
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(xs, width, x + 1, pitch, width, static_cast<size_t>(k),
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(hessian.apply(res, k, xs, ys));
    products += k;
    // D v = R diag(scale) (R^T v).
    if (rank > 0) {
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N,
                                                    rank, k, n, &kOne<double>, response, n, xs, n,
                                                    &kZero<double>, rv, rank)));
      FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, rank, k, rv, rank,
                                                    response_scale, 1, rv, rank)));
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n,
                                                    k, rank, &kOne<double>, response, n, rv, rank,
                                                    &kOne<double>, ys, n)));
    }
    // H v + g v_0, and g^T v into row 0.
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, k,
                                                  1, &kOne<double>, gradient, n, x, m,
                                                  &kOne<double>, ys, n)));
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, 1, k,
                                                  n, &kOne<double>, gradient, n, xs, n,
                                                  &kZero<double>, y, m)));
    return gpu_status(wwr::wwrMemcpy2DAsync(y + 1, pitch, ys, width, width, static_cast<size_t>(k),
                                            wwr::wwrMemcpyDeviceToDevice, stream));
  }
};

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

Result<CasscfSolution>
casscf_solve(DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
             const SphericalMatrixAO &h, const std::shared_ptr<const Buffers::device<double>> &u,
             const std::span<CoulombPairClass> j_classes,
             const std::span<OrderedPairClass> k_classes, const SphericalMatrixAO &c,
             const int32_t ncore, const int32_t ncas, const int32_t nelecas, const int32_t nroots,
             const std::span<const double> weights, const CasscfOptions &options) {
  const auto nmo = static_cast<int32_t>(c.n());
  if (c.n() != basis.nao_sphe() || c.dev_idx() != res.dev_idx()) {
    return err_config("casscf_solve: c must be {0} x {0}, basis's spherical order, on device {1} "
                      "(got {2} on device {3})",
                      basis.nao_sphe(), res.dev_idx(), c.n(), c.dev_idx());
  }
  if (ncore < 0 || ncas < 1 || ncore + ncas > nmo || nelecas < 0 || nelecas > 2 * ncas) {
    return err_config("casscf_solve: needs ncore >= 0, ncas >= 1, ncore + ncas <= {} and nelecas "
                      "in [0, 2 ncas] (ncore {}, ncas {}, nelecas {})",
                      nmo, ncore, ncas, nelecas);
  }
  if (!(options.conv_tol > 0.0) || !(options.conv_tol_grad > 0.0) || options.max_cycle < 1 ||
      !(options.trust_radius > 0.0) || options.max_micro < 1) {
    return err_config("casscf_solve: conv_tol, conv_tol_grad and trust_radius must be above 0 and "
                      "max_cycle and max_micro at least 1 ({}, {}, {}, {}, {})",
                      options.conv_tol, options.conv_tol_grad, options.trust_radius,
                      options.max_cycle, options.max_micro);
  }
  if (nroots < 1) {
    return err_config("casscf_solve: nroots ({}) must be at least 1", nroots);
  }
  const auto roots_count = static_cast<size_t>(nroots);
  std::vector<double> w(weights.begin(), weights.end());
  if (w.empty()) {
    w.assign(roots_count, 1.0 / static_cast<double>(nroots));
  }
  const double weight_sum = std::accumulate(w.begin(), w.end(), 0.0);
  if (w.size() != roots_count || std::ranges::any_of(w, [](double x) { return !(x >= 0.0); }) ||
      std::abs(weight_sum - 1.0) > 1e-12) {
    return err_config("casscf_solve: needs {} weights, each >= 0, summing to 1 (got {}, summing "
                      "to {})",
                      nroots, w.size(), weight_sum);
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const int32_t n_beta = nelecas / 2;
  const StringSet alpha = FBF_TRY(StringSet::create(ncas, nelecas - n_beta));
  const StringSet beta = FBF_TRY(StringSet::create(ncas, n_beta));
  CiSigma sigma = FBF_TRY(CiSigma::create(res, alpha, beta));
  CiRdm ci_rdm = FBF_TRY(CiRdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(sigma.size());
  if (roots_count > dim) {
    return err_config("casscf_solve: nroots ({}) exceeds the {} determinants", nroots, dim);
  }

  const auto mo = static_cast<size_t>(nmo);
  const auto n = static_cast<size_t>(ncas);
  const int32_t n_pairs = (ncore * (nmo - ncore)) + (ncas * (nmo - ncore - ncas));
  const auto pairs = static_cast<size_t>(n_pairs);
  const int32_t m = n_pairs + 1;

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
  // The augmented Hessian, every buffer for the solve: its Davidson runs up
  // to ah_columns columns, so no apply takes more. The model gradient; D's
  // terms and scales, at most 2 kResponseMemory; (0, diag(H + D)) for the
  // preconditioner; the guess e_0, set once; the operator's xs, ys and rv;
  // and the Ritz vector and value side by side, (v_0, v, lambda), read back
  // in one copy: v first, so it keeps the pool's alignment, which the
  // Davidson's BLAS on it may choose its kernels (and roundings) by.
  const int32_t ah_columns = std::max(2, std::min(m, kAhMaxSubspace));
  const auto ah_width = static_cast<size_t>(ah_columns);
  const size_t response_max = 2 * kResponseMemory;
  Buffers::device<double> model_gradient(std::max<size_t>(pairs, 1), res.device());
  Buffers::device<double> response_device(std::max<size_t>(pairs * response_max, 1), res.device());
  Buffers::device<double> scale_device(response_max, res.device());
  Buffers::device<double> ah_diagonal(static_cast<size_t>(m), res.device());
  Buffers::device<double> ah_guess(static_cast<size_t>(m), res.device());
  Buffers::device<double> ah_xs(std::max<size_t>(pairs * ah_width, 1), res.device());
  Buffers::device<double> ah_ys(std::max<size_t>(pairs * ah_width, 1), res.device());
  Buffers::device<double> ah_rv(response_max * ah_width, res.device());
  Buffers::device<double> ah_pair(static_cast<size_t>(m) + 1, res.device());
  {
    std::vector<double> unit(static_cast<size_t>(m), 0.0);
    unit[0] = 1.0;
    FBF_TRY(casscf_upload(res, unit, ah_guess.data()));
  }

  // C_now = C_accepted exp(t kappa), from kappa_u and kappa_w.
  const auto rotate = [&](const double t) -> Status {
    FBF_TRY(skew_expm<double>(res, nmo, kappa_u.data(), kappa_w.data(), t, rotation.data()));
    return gpu_status(wwr::gemm<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo, nmo, nmo, &kOne<double>, c_accepted.data(),
        nmo, rotation.data(), nmo, &kZero<double>, c_now.data(), nmo));
  };

  std::optional<CiRoots> roots;
  std::vector<CasscfIteration> log;
  bool have_accepted = false;
  double e_accepted = 0.0;
  double radius = options.trust_radius;
  double last_gradient = std::numeric_limits<double>::infinity();
  // The step from the accepted orbitals, t x: E changes by
  // t g.x + t^2 x.Hx / 2 in the model; t is 1, halved on each rejection.
  bool have_step = false;
  double step_gx = 0.0;
  double step_xhx = 0.0;
  double step_norm = 0.0;
  double step_t = 1.0;
  // The step x and the model's (H + D) x, and g at the accepted orbitals.
  std::vector<double> x_total(pairs, 0.0);
  std::vector<double> hx_total(pairs, 0.0);
  std::vector<double> g_accepted;
  // The CI response: H, at fixed CI, misses the CI's relaxation, which is
  // what leaves a two-step loop linear. D is the BFGS correction from each
  // accepted step s and the gradient's actual change y (CI relaxed at both
  // ends), as terms (vector, scale) of D = sum scale v v^T, so B = H + D
  // learns the relaxation along the steps taken.
  std::deque<std::pair<std::vector<double>, double>> response;

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
    CiSolverOptions ci = options.ci;
    ci.davidson.residual_tolerance = std::max(options.ci.davidson.residual_tolerance,
                                              std::min(kCiLoosest, kCiPerGradient * last_gradient));
    // Cold each iteration: no warm start yet (#282, see the README).
    roots.emplace(FBF_TRY(ci_solve(res, sigma, alpha, beta, diagonal.data(), nroots, ci)));

    // 3. The weighted RDMs, the energy and the gradient.
    FBF_TRY(ci_rdm.rdm12(res, w, roots->vectors.data(), gamma_device.data(), gamma2_device.data()));
    std::vector<double> gamma = FBF_TRY(casscf_download(res, gamma_device.data(), n * n));
    std::vector<double> gamma2 = FBF_TRY(casscf_download(res, gamma2_device.data(), n * n * n * n));
    CasscfFock fock =
        FBF_TRY(casscf_fock(res, basis, molecule, h, u, j_classes, k_classes, c_now, ncore, ncas,
                            ints.ppaa, gamma, gamma2, options.tau_j, options.tau_k));
    const std::vector<double> g = FBF_TRY(casscf_download(res, fock.gradient.data(), pairs));
    const double g_norm = std::sqrt(casscf_dot(g, g));
    last_gradient = g_norm;
    const double energy = fock.energy;
    const double change = have_accepted ? energy - e_accepted : 0.0;
    CasscfIteration record{energy, change, g_norm, radius, roots->davidson.iterations, 0, 0, true};

    // 4. Convergence, then the rejection of a rise.
    const bool converged = n_pairs == 0 || (have_accepted && std::abs(change) < options.conv_tol &&
                                            g_norm < options.conv_tol_grad);
    if (!converged && have_accepted && change > kEnergyRiseTolerance) {
      step_t *= 0.5;
      radius = step_t * step_norm;
      record.accepted = false;
      record.trust_radius = radius;
      log.push_back(record);
      FBF_LOG_DEBUG("casscf_solve: macro {} E {:.12f} dE {:.3e} |g| {:.3e} rejected, step {:.3e}",
                    cycle, energy, change, g_norm, step_t * step_norm);
      FBF_TRY(rotate(step_t));
      continue;
    }
    if (have_step) {
      const double predicted = (step_t * step_gx) + (0.5 * step_t * step_t * step_xhx);
      if (predicted < 0.0 && std::abs((change / predicted) - 1.0) < kTrustAgreement) {
        radius = std::min(options.trust_radius, 2.0 * radius);
        record.trust_radius = radius;
      }
      // BFGS over the step s = t x, with B s = t (H + D) x and y the
      // gradient's actual change: B + y y^T / (y.s) - (Bs)(Bs)^T / (s.Bs),
      // which keeps B positive definite when y.s > 0.
      std::vector<double> y(pairs);
      std::vector<double> s(pairs);
      std::vector<double> bs(pairs);
      for (size_t i = 0; i < pairs; ++i) {
        s[i] = step_t * x_total[i];
        bs[i] = step_t * hx_total[i];
        y[i] = g[i] - g_accepted[i];
      }
      const double ys = casscf_dot(y, s);
      const double sbs = casscf_dot(s, bs);
      const double floor = kResponseSafeguard * std::sqrt(casscf_dot(s, s));
      if (ys > floor * std::sqrt(casscf_dot(y, y)) && sbs > floor * std::sqrt(casscf_dot(bs, bs))) {
        response.emplace_back(std::move(y), 1.0 / ys);
        response.emplace_back(std::move(bs), -1.0 / sbs);
        while (response.size() > 2 * kResponseMemory) {
          response.pop_front();
        }
      }
    }
    g_accepted = g;
    have_accepted = true;
    e_accepted = energy;
    if (converged) {
      log.push_back(record);
      FBF_LOG_INFO("casscf_solve: converged in {} macro iterations, E {:.12f}, |g| {:.3e}", cycle,
                   energy, g_norm);
      std::vector<double> root_gammas(roots_count * n * n);
      Buffers::device<double> root_gamma(n * n, res.device());
      for (size_t r = 0; r < roots_count; ++r) {
        FBF_TRY(sigma.rdm1(res, roots->vectors.data() + (r * dim), root_gamma.data()));
        const std::vector<double> one = FBF_TRY(casscf_download(res, root_gamma.data(), n * n));
        std::ranges::copy(one, root_gammas.begin() + static_cast<std::ptrdiff_t>(r * n * n));
      }
      return CasscfSolution{
          std::move(c_now),  std::move(*roots),      energy,        g_norm, std::move(gamma),
          std::move(gamma2), std::move(root_gammas), std::move(log)};
    }
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(c_accepted.data(), c_now.data(), mo * mo * sizeof(double),
                                       wwr::wwrMemcpyDeviceToDevice, stream)));

    // 5. The step: augmented-Hessian solves of the quadratic model
    //    E + g.x + x.Hx / 2, each from the model gradient g + H x the last
    //    left, until it is small, the radius is reached or max_micro is.
    OrbitalHessian hessian =
        FBF_TRY(OrbitalHessian::create(res, basis, h, u, j_classes, k_classes, c_now, ncore, ncas,
                                       ints.ppaa, ints.papa, fock, options.tau_j, options.tau_k));
    // (0, diag(H + D)) for the preconditioner, built here and uploaded once
    // for the macro iteration, and D's terms beside it.
    std::vector<double> ah_diagonal_host(static_cast<size_t>(m), 0.0);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ah_diagonal_host.data() + 1, hessian.diagonal().data(),
                                           pairs * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                           stream)));
    FBF_TRY(res.sync());
    const std::span<double> h_diagonal{ah_diagonal_host.data() + 1, pairs};
    const auto rank = static_cast<int32_t>(response.size());
    std::vector<double> response_host(pairs * response.size());
    std::vector<double> scale_host(response.size());
    for (size_t j = 0; j < response.size(); ++j) {
      const auto &[r, scale] = response[j];
      std::ranges::copy(r, response_host.begin() + static_cast<std::ptrdiff_t>(j * pairs));
      scale_host[j] = scale;
      for (size_t i = 0; i < pairs; ++i) {
        h_diagonal[i] += scale * r[i] * r[i];
      }
    }
    FBF_TRY(casscf_upload(res, response_host, response_device.data()));
    FBF_TRY(casscf_upload(res, scale_host, scale_device.data()));
    FBF_TRY(casscf_upload(res, ah_diagonal_host, ah_diagonal.data()));
    double h_scale = 1.0;
    for (const double d : h_diagonal) {
      h_scale = std::max(h_scale, std::abs(d));
    }
    // r / (theta - (0, diag(H + D))), the denominator off zero, on the device:
    // theta goes by value, so a Davidson iteration's correction neither
    // allocates nor syncs.
    const auto precondition = [&](DeviceResources &r, std::span<const double> theta,
                                  const double *residual, double *correction) -> Status {
      for (size_t col = 0; col < theta.size(); ++col) {
        const size_t at = col * static_cast<size_t>(m);
        FBF_TRY(gpu_status(device::ah_precondition(r.stream(), m, theta[col], kAhDenominatorFloor,
                                                   ah_diagonal.data(), residual + at,
                                                   correction + at)));
      }
      return {};
    };

    std::vector<double> g_model = g;
    std::ranges::fill(x_total, 0.0);
    std::ranges::fill(hx_total, 0.0);
    for (int32_t micro = 1; micro <= options.max_micro; ++micro) {
      // Not waited for: g_model is next written after the readback below,
      // whose sync on this stream follows the copy.
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(model_gradient.data(), g_model.data(),
                                             pairs * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                             stream)));
      AugmentedHessian op{hessian,
                          model_gradient.data(),
                          response_device.data(),
                          scale_device.data(),
                          rank,
                          n_pairs,
                          ah_columns,
                          ah_xs.data(),
                          ah_ys.data(),
                          ah_rv.data()};
      const double g_model_norm = std::sqrt(casscf_dot(g_model, g_model));
      DavidsonOptions ah;
      ah.residual_tolerance = std::clamp(kAhPerGradient * g_model_norm / h_scale, 1e-14, 1e-4);
      ah.max_iterations = kAhMaxIterations;
      const auto info =
          davidson_solve<double>(res, op, precondition, m, 1, ah_columns, ah_guess.data(), 1,
                                 ah_pair.data() + m, ah_pair.data(), ah);
      record.hessian_products += op.products;
      // Out of iterations or of new directions, the last Ritz pair still
      // gives a step of the model, and the energy test guards it.
      if (!info && info.error().kind != ErrorKind::MaxIterationsExceeded &&
          info.error().kind != ErrorKind::Divergence) {
        return std::unexpected{info.error()};
      }
      const std::vector<double> pair =
          FBF_TRY(casscf_download(res, ah_pair.data(), static_cast<size_t>(m) + 1));
      const std::span<const double> v{pair.data(), static_cast<size_t>(m)};
      const double lambda = pair[static_cast<size_t>(m)];
      if (v[0] == 0.0) {
        return err_numerical("casscf_solve: the augmented Hessian's lowest root has no "
                             "component along the gradient (macro iteration {})",
                             cycle);
      }
      std::vector<double> dx(pairs);
      for (size_t i = 0; i < pairs; ++i) {
        dx[i] = v[i + 1] / v[0];
      }
      const double dx_norm = std::sqrt(casscf_dot(dx, dx));
      const double remaining = radius - std::sqrt(casscf_dot(x_total, x_total));
      const double s = dx_norm > remaining ? remaining / dx_norm : 1.0;
      // H dx = lambda dx - g_model, the root's own equation.
      for (size_t i = 0; i < pairs; ++i) {
        const double hdx = s * ((lambda * dx[i]) - g_model[i]);
        x_total[i] += s * dx[i];
        hx_total[i] += hdx;
        g_model[i] += hdx;
      }
      record.micro_iterations = micro;
      if (s < 1.0 || std::sqrt(casscf_dot(g_model, g_model)) <= kMicroGradientFraction * g_norm) {
        break;
      }
    }
    have_step = true;
    step_t = 1.0;
    step_gx = casscf_dot(g, x_total);
    step_xhx = casscf_dot(x_total, hx_total);
    step_norm = std::sqrt(casscf_dot(x_total, x_total));
    log.push_back(record);
    FBF_LOG_DEBUG("casscf_solve: macro {} E {:.12f} dE {:.3e} |g| {:.3e} ci {} micro {} Hx {} "
                  "step {:.3e}",
                  cycle, energy, change, g_norm, record.ci_iterations, record.micro_iterations,
                  record.hessian_products, step_norm);

    // 6. C <- C exp(kappa(x)).
    FBF_TRY(casscf_upload(res, x_total, step.data()));
    FBF_TRY(unpack_orbital_pairs(res, nmo, ncore, ncas, step.data(), kappa.data()));
    FBF_TRY(skew_eig<double>(res, nmo, kappa.data(), kappa_u.data(), kappa_w.data()));
    FBF_TRY(rotate(1.0));
  }
  return err_max_iterations_exceeded(
      "casscf_solve: not converged in {} macro iterations (last |g| {:.3e})", options.max_cycle,
      last_gradient);
}

} // namespace free_breakfast
