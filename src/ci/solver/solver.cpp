/// @file
/// @brief CiSpinSquare's uploads and launches, S² on a CI vector over the
/// replacement lists; and ci_solve's device preconditioner, guesses and
/// P-space.

module;

// The launchers, defined in ci_solver.cu: declared in the GMF so they keep
// external linkage (see the header).
#include "ci_solver_bridge.h"

module free_breakfast.ci.solver;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;              // CiLayout
import free_breakfast.recipes.diagonalize;   // diagonalize
import free_breakfast.recipes.orthogonalize; // orthonormalize

#include "diagnostics.h"

namespace free_breakfast {

static_assert(LinearOperator<CiSpinSquare, double>);

CiSpinSquare::CiSpinSquare(DeviceResources &res, const int32_t n_orbitals, const int32_t n_alpha,
                           const int32_t n_beta, const int32_t alpha_per, const double constant,
                           const bool one_set)
    : dev_idx_{res.dev_idx()}, n_orbitals_{n_orbitals}, n_alpha_{n_alpha}, n_beta_{n_beta},
      alpha_per_{alpha_per}, constant_{constant}, one_set_{one_set},
      alpha_links_(static_cast<size_t>(n_alpha) * static_cast<size_t>(alpha_per) * 3, res.device()),
      beta_slots_(static_cast<size_t>(n_beta) * static_cast<size_t>(n_orbitals) *
                      static_cast<size_t>(n_orbitals),
                  res.device()) {}

Result<CiSpinSquare> CiSpinSquare::create(DeviceResources &res, const StringSet &alpha,
                                          const StringSet &beta) {
  if (alpha.n_orbitals() != beta.n_orbitals()) {
    return err_config("CiSpinSquare: alpha strings span {} orbitals, beta {}", alpha.n_orbitals(),
                      beta.n_orbitals());
  }
  if (static_cast<int64_t>(alpha.size()) * beta.size() > std::numeric_limits<int32_t>::max()) {
    return err_config("CiSpinSquare: {} x {} determinants is more than an int32_t counts",
                      alpha.size(), beta.size());
  }
  const auto n = static_cast<size_t>(alpha.n_orbitals());
  // S²'s constant, S_z (S_z + 1) + N_β.
  const double two_sz = alpha.n_electrons() - beta.n_electrons();
  const double constant = 0.25 * two_sz * (two_sz + 2.0) + beta.n_electrons();

  // Each α link a†_p a_q names the slot q · n + p of its partner a†_q a_p.
  std::vector<int32_t> links;
  links.reserve(static_cast<size_t>(alpha.size()) *
                static_cast<size_t>(alpha.replacements_per_string()) * 3);
  for (int32_t address = 0; address < alpha.size(); ++address) {
    for (const Replacement &r : alpha.replacements(address)) {
      links.push_back(r.target);
      links.push_back(
          static_cast<int32_t>(static_cast<size_t>(r.q) * n + static_cast<size_t>(r.p)));
      links.push_back(r.sign);
    }
  }
  // Each (p, q) appears at most once in a list, so a slot holds one link.
  std::vector<int32_t> slots(static_cast<size_t>(beta.size()) * n * n, 0);
  for (int32_t address = 0; address < beta.size(); ++address) {
    int32_t *const table = slots.data() + static_cast<size_t>(address) * n * n;
    for (const Replacement &r : beta.replacements(address)) {
      table[static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)] = r.sign * (r.target + 1);
    }
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  CiSpinSquare spin(res, alpha.n_orbitals(), alpha.size(), beta.size(),
                    alpha.replacements_per_string(), constant,
                    alpha.n_electrons() == beta.n_electrons());
  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(spin.alpha_links_.data(), links.data(),
                                         sizeof(int32_t) * links.size(), wwr::wwrMemcpyHostToDevice,
                                         stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(spin.beta_slots_.data(), slots.data(),
                                         sizeof(int32_t) * slots.size(), wwr::wwrMemcpyHostToDevice,
                                         stream)));
  // Pageable sources, which die at return.
  FBF_TRY(res.sync());
  return spin;
}

Status CiSpinSquare::check(DeviceResources &res, const CiLayout layout) const {
  if (res.dev_idx() != dev_idx_) {
    return err_config("CiSpinSquare: tables are on device {}, but the call passed device {}",
                      dev_idx_, res.dev_idx());
  }
  if (layout == CiLayout::symmetric && !one_set_) {
    return err_config("CiSpinSquare: the symmetric layout needs one set of strings for both spins");
  }
  return {};
}

Status CiSpinSquare::launch(DeviceResources &res, const int32_t k, const double shift,
                            const double scale, const bool accumulate, const double *const x,
                            double *const y, const CiLayout layout) {
  FBF_TRY(check(res, layout));
  const int32_t length = size(layout);
  if (k < 0 || static_cast<int64_t>(k) * length > std::numeric_limits<int32_t>::max()) {
    return err_config("CiSpinSquare: a block of {} vectors of {} is outside an int32_t's rows", k,
                      length);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::ci_spin_square(res.stream(), layout == CiLayout::symmetric, k * length,
                                           length, n_beta_, n_orbitals_, alpha_links_.data(),
                                           alpha_per_, beta_slots_.data(), constant_, shift, scale,
                                           accumulate, x, y));
}

Status CiSpinSquare::apply(DeviceResources &res, const int32_t k, const double *const x,
                           double *const y, const CiLayout layout) {
  return launch(res, k, 0.0, 1.0, false, x, y, layout);
}

Status CiSpinSquare::penalize(DeviceResources &res, const int32_t k, const double lambda,
                              const double target, const double *const x, double *const y,
                              const CiLayout layout) {
  return launch(res, k, target, lambda, true, x, y, layout);
}

Status CiSpinSquare::penalize_diagonal(DeviceResources &res, const double lambda,
                                       const double target, const double *const diagonal,
                                       double *const shifted, const CiLayout layout,
                                       const double *const exchange) {
  FBF_TRY(check(res, layout));
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(
      device::ci_spin_diagonal(res.stream(), layout == CiLayout::symmetric, size(layout), n_beta_,
                               n_orbitals_, alpha_links_.data(), alpha_per_, beta_slots_.data(),
                               constant_, target, lambda, exchange, diagonal, shifted));
}

Status ci_precondition(DeviceResources &res, const int32_t n, const std::span<const double> theta,
                       const double *const shifted, const double *const ritz_vectors,
                       const double *const residual, double *const correction, const bool olsen,
                       const int32_t p, const int32_t *const picks,
                       const double *const pspace_vectors, const double *const pspace_values) {
  const wwr::wwrStream_t stream = res.stream();
  const auto nz = static_cast<size_t>(n);
  const wwr::wwrblasHandle_t blas = res.blas().get();
  if (p == 0) {
    if (!olsen) {
      for (size_t col = 0; col < theta.size(); ++col) {
        FBF_TRY(gpu_status(device::ci_precondition(
            stream, n, theta[col], shifted, residual + (col * nz), correction + (col * nz))));
      }
      return {};
    }
    // x^T M⁻¹ x and x^T M⁻¹ r per column, written and read on the device: ε
    // never comes back to the host.
    const auto [dots] = res.device_regions<double>(2 * theta.size());
    const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t,
                                            CreatePolicy<wwr::wwrblasStatus_t>>
        device_results{blas, wwr::WWRBLAS_POINTER_MODE_DEVICE};
    for (size_t col = 0; col < theta.size(); ++col) {
      const double *const x = ritz_vectors + (col * nz);
      const double *const r = residual + (col * nz);
      double *const c = correction + (col * nz);
      double *const pair = dots + (2 * col);
      // M⁻¹ x into the correction column; M is diagonal, so x^T M⁻¹ r is
      // (M⁻¹ x)^T r and M⁻¹ r is never formed on its own.
      FBF_TRY(gpu_status(device::ci_precondition(stream, n, theta[col], shifted, x, c)));
      FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, n, x, 1, c, 1, pair)));
      FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, n, c, 1, r, 1, pair + 1)));
      FBF_TRY(gpu_status(device::ci_olsen(stream, n, theta[col], shifted, x, r, pair, c)));
    }
    return {};
  }

  // M = θ − (H_PP ⊕ diag(Q)): the diagonal over every element, then the P
  // ones replaced by U (θ − Λ)⁻¹ U^T of their gathered part, its
  // denominators held off zero as the diagonal's are. in may be out: P is
  // gathered before the diagonal writes. The correction is always Olsen's
  // here: over an exact P block the plain M⁻¹ r is all but x, which the
  // Davidson projects away (see the README). M⁻¹ x is a column of its own,
  // since M is no longer diagonal and r − ε x goes through M⁻¹ whole.
  const auto pz = static_cast<size_t>(p);
  const auto [dots, gathered, rotated, inverse_x] =
      res.device_regions<double>(2 * theta.size(), pz, pz, nz);
  const auto apply_inverse = [&](const double t, const double *const in,
                                 double *const out) -> Status {
    FBF_TRY(gpu_status(device::ci_gather(stream, n, 1, p, picks, in, gathered)));
    FBF_TRY(gpu_status(device::ci_precondition(stream, n, t, shifted, in, out)));
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, p, p, &kOne<double>,
                                                  pspace_vectors, p, gathered, 1, &kZero<double>,
                                                  rotated, 1)));
    FBF_TRY(gpu_status(device::ci_precondition(stream, p, t, pspace_values, rotated, rotated)));
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, p, p, &kOne<double>,
                                                  pspace_vectors, p, rotated, 1, &kZero<double>,
                                                  gathered, 1)));
    return gpu_status(device::ci_scatter(stream, n, 1, p, picks, gathered, out));
  };
  for (size_t col = 0; col < theta.size(); ++col) {
    const double *const x = ritz_vectors + (col * nz);
    const double *const r = residual + (col * nz);
    double *const c = correction + (col * nz);
    double *const pair = dots + (2 * col);
    FBF_TRY(apply_inverse(theta[col], x, inverse_x));
    {
      const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t,
                                              CreatePolicy<wwr::wwrblasStatus_t>>
          device_results{blas, wwr::WWRBLAS_POINTER_MODE_DEVICE};
      FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, n, x, 1, inverse_x, 1, pair)));
      FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, n, inverse_x, 1, r, 1, pair + 1)));
    }
    // r − ε x, then M⁻¹ of it in place.
    FBF_TRY(gpu_status(device::ci_olsen(stream, n, theta[col], nullptr, x, r, pair, c)));
    FBF_TRY(apply_inverse(theta[col], c, c));
  }
  return {};
}

Status ci_lowest(DeviceResources &res, const int32_t n, const double *const values,
                 const int32_t count, int32_t *const picks) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto [scratch, survivors] = res.device_regions<uint64_t, int32_t>(
      static_cast<size_t>(device::kCiLowestScratch), static_cast<size_t>(count));
  return gpu_status(device::ci_lowest(res.stream(), n, values, count, scratch, survivors, picks));
}

Status ci_guesses(DeviceResources &res, const int32_t n, const int32_t count,
                  const int32_t *const picks, const int32_t p, const double *const columns,
                  double *const guess, const bool canonical) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const wwr::wwrblasHandle_t blas = res.blas().get();
  if (columns == nullptr) {
    FBF_TRY(gpu_status(device::ci_guess(stream, n, count, picks, guess)));
  } else {
    FBF_TRY(gpu_status(device::ci_guess(stream, n, count, nullptr, guess)));
    FBF_TRY(gpu_status(device::ci_scatter(stream, n, count, p, picks, columns, guess)));
  }

  if (canonical) {
    // Orthonormalized as a block. Unit vectors on distinct determinants, or
    // orthonormal columns on P, plus a 1e-3 spread are independent by
    // construction, so only a bug drops one: the floor is zero, leaving
    // orthonormalize's own rounding cut.
    const int32_t kept = FBF_TRY(orthonormalize<double>(res, n, count, guess, 0.0));
    if (kept != count) {
      return err_numerical("ci_guesses: {} of {} guess columns were linearly dependent",
                           count - kept, count);
    }
    return {};
  }

  // Modified Gram-Schmidt, twice, of each column against the ones before;
  // the dots and norms come back as host scalars.
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{blas, wwr::WWRBLAS_POINTER_MODE_HOST};
  const auto nz = static_cast<size_t>(n);
  for (int32_t g = 0; g < count; ++g) {
    double *const column = guess + (static_cast<size_t>(g) * nz);
    for (int32_t pass = 0; pass < 2; ++pass) {
      for (int32_t h = 0; h < g; ++h) {
        const double *const earlier = guess + (static_cast<size_t>(h) * nz);
        double overlap = 0.0;
        FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, n, earlier, 1, column, 1, &overlap)));
        const double neg_overlap = -overlap;
        FBF_TRY(
            gpu_status(wwr::axpy<double, int32_t>(blas, n, &neg_overlap, earlier, 1, column, 1)));
      }
    }
    double norm = 0.0;
    FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, n, column, 1, &norm)));
    const double inv_norm = 1.0 / norm;
    FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, n, &inv_norm, column, 1)));
  }
  return {};
}

Status ci_pspace_hamiltonian(DeviceResources &res, const StringSet &alpha, const StringSet &beta,
                             const double *const contraction, const double *const diagonal,
                             const double penalty, const int32_t p, const int32_t *const picks,
                             double *const hamiltonian, const CiLayout layout) {
  const bool symmetric = layout == CiLayout::symmetric;
  if (alpha.n_orbitals() != beta.n_orbitals() || contraction == nullptr || p < 0 ||
      static_cast<int64_t>(alpha.size()) * beta.size() > std::numeric_limits<int32_t>::max() ||
      (symmetric && alpha.n_electrons() != beta.n_electrons())) {
    return err_config("ci_pspace_hamiltonian: needs one orbital count, a contraction, p >= 0, at "
                      "most an int32_t of determinants and, in the symmetric layout, one set for "
                      "both spins (orbitals {} and {}, electrons {} and {}, p {})",
                      alpha.n_orbitals(), beta.n_orbitals(), alpha.n_electrons(),
                      beta.n_electrons(), p);
  }
  if (p == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // λ (S² − S(S+1)): diag(S²) is S_z (S_z + 1) + N_β less the doubly occupied
  // orbitals, S = |N_α − N_β| / 2.
  const int32_t two_sz = alpha.n_electrons() - beta.n_electrons();
  const int32_t two_s = std::abs(two_sz);
  const double spin_shift =
      (0.25 * two_sz * (two_sz + 2)) + beta.n_electrons() - (0.25 * two_s * (two_s + 2));
  const std::span<const uint64_t> alpha_strings = alpha.strings();
  const std::span<const uint64_t> beta_strings = beta.strings();
  const auto [strings_a, strings_b] =
      res.device_regions<uint64_t>(alpha_strings.size(), symmetric ? 0 : beta_strings.size());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(strings_a, alpha_strings.data(),
                                         alpha_strings.size() * sizeof(uint64_t),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  if (!symmetric) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(strings_b, beta_strings.data(),
                                           beta_strings.size() * sizeof(uint64_t),
                                           wwr::wwrMemcpyHostToDevice, stream)));
  }
  FBF_TRY(gpu_status(device::ci_pspace(stream, symmetric, p, picks, beta.size(), alpha.n_orbitals(),
                                       strings_a, symmetric ? strings_a : strings_b, contraction,
                                       diagonal, penalty, spin_shift, hamiltonian)));
  // The strings' pageable sources are the caller's, and may go at return.
  return res.sync();
}

Status ci_pspace(DeviceResources &res, const StringSet &alpha, const StringSet &beta,
                 const CiLayout layout, const int32_t n, const int32_t p,
                 const double *const values, const double *const contraction,
                 const double *const diagonal, const double penalty, int32_t *const picks,
                 double *const vectors, double *const eigenvalues, const int32_t n_roots,
                 double *const root_values, double *const root_vectors) {
  FBF_TRY(ci_lowest(res, n, values, p, picks));
  FBF_TRY(ci_pspace_hamiltonian(res, alpha, beta, contraction, diagonal, penalty, p, picks, vectors,
                                layout));
  FBF_TRY(diagonalize<double>(res, p, vectors, eigenvalues));
  if (p < n) {
    return {};
  }
  // P is a permutation of every element, so the scatter writes all of them.
  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(root_values, eigenvalues,
                                         static_cast<size_t>(n_roots) * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  return gpu_status(device::ci_scatter(stream, n, n_roots, p, picks, vectors, root_vectors));
}

} // namespace free_breakfast
