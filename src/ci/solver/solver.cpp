/// @file
/// @brief S² on a CI vector, over the replacement lists: on the host, and
/// CiSpinSquare's uploads and launches; and ci_solve's device preconditioner.

module;

// The launchers, defined in ci_solver.cu: declared in the GMF so they keep
// external linkage (see the header).
#include "ci_solver_bridge.h"

module free_breakfast.ci.solver;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.ci.strings;

#include "diagnostics.h"

namespace free_breakfast {

Status host_spin_square(const StringSet &alpha, const StringSet &beta,
                        const std::span<const double> c, const std::span<double> s2c) {
  const size_t dim = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  if (alpha.n_orbitals() != beta.n_orbitals() || c.size() != dim || s2c.size() != dim) {
    return err_config("ci spin_square: orbitals {} and {}, {} determinants, given c of {} and "
                      "s2c of {}",
                      alpha.n_orbitals(), beta.n_orbitals(), dim, c.size(), s2c.size());
  }
  const auto n = static_cast<size_t>(alpha.n_orbitals());
  const auto nb = static_cast<size_t>(beta.size());

  // S² = S_z (S_z + 1) + N_β − Σ_pq E^β_pq E^α_qp. The constant part first.
  const double two_sz = alpha.n_electrons() - beta.n_electrons();
  const double constant = 0.25 * two_sz * (two_sz + 2.0) + beta.n_electrons();
  for (size_t i = 0; i < dim; ++i) {
    s2c[i] = constant * c[i];
  }

  // Then, per determinant K, scatter −<J|E^β_pq E^α_qp|K> c_K. An α entry of
  // K_α's list is E^α_ab (a created, b annihilated); its partner in K_β's
  // list is E^β_ba, found through a table of that list by (p, q), since each
  // (p, q) appears at most once. Both operators are even in their spin, so
  // each carries only its own list's sign.
  std::vector<int32_t> slot(n * n);
  for (int32_t kb = 0; kb < beta.size(); ++kb) {
    const std::span<const Replacement> beta_list = beta.replacements(kb);
    std::ranges::fill(slot, -1);
    for (size_t j = 0; j < beta_list.size(); ++j) {
      slot[static_cast<size_t>(beta_list[j].p) * n + static_cast<size_t>(beta_list[j].q)] =
          static_cast<int32_t>(j);
    }
    for (int32_t ka = 0; ka < alpha.size(); ++ka) {
      const double ck = c[static_cast<size_t>(ka) * nb + static_cast<size_t>(kb)];
      if (ck == 0.0) {
        continue;
      }
      for (const Replacement &ra : alpha.replacements(ka)) {
        const int32_t j = slot[static_cast<size_t>(ra.q) * n + static_cast<size_t>(ra.p)];
        if (j < 0) {
          continue;
        }
        const Replacement &rb = beta_list[static_cast<size_t>(j)];
        s2c[static_cast<size_t>(ra.target) * nb + static_cast<size_t>(rb.target)] -=
            ra.sign * rb.sign * ck;
      }
    }
  }
  return {};
}

static_assert(LinearOperator<CiSpinSquare, double>);

CiSpinSquare::CiSpinSquare(DeviceResources &res, const int32_t n_orbitals, const int32_t n_alpha,
                           const int32_t n_beta, const int32_t alpha_per, const double constant)
    : dev_idx_{res.dev_idx()}, n_orbitals_{n_orbitals}, n_alpha_{n_alpha}, n_beta_{n_beta},
      alpha_per_{alpha_per}, constant_{constant},
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
  // host_spin_square's constant, S_z (S_z + 1) + N_β.
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
                    alpha.replacements_per_string(), constant);
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

Status CiSpinSquare::launch(DeviceResources &res, const int32_t k, const double shift,
                            const double scale, const bool accumulate, const double *const x,
                            double *const y) {
  if (res.dev_idx() != dev_idx_) {
    return err_config("CiSpinSquare: tables are on device {}, but the call passed device {}",
                      dev_idx_, res.dev_idx());
  }
  if (k < 0 || static_cast<int64_t>(k) * size() > std::numeric_limits<int32_t>::max()) {
    return err_config("CiSpinSquare: a block of {} vectors of {} is outside an int32_t's rows", k,
                      size());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::ci_spin_square(res.stream(), k * size(), size(), n_beta_, n_orbitals_,
                                           alpha_links_.data(), alpha_per_, beta_slots_.data(),
                                           constant_, shift, scale, accumulate, x, y));
}

Status CiSpinSquare::apply(DeviceResources &res, const int32_t k, const double *const x,
                           double *const y) {
  return launch(res, k, 0.0, 1.0, false, x, y);
}

Status CiSpinSquare::penalize(DeviceResources &res, const int32_t k, const double lambda,
                              const double target, const double *const x, double *const y) {
  return launch(res, k, target, lambda, true, x, y);
}

Status CiSpinSquare::penalize_diagonal(DeviceResources &res, const double lambda,
                                       const double target, const double *const diagonal,
                                       double *const shifted) {
  if (res.dev_idx() != dev_idx_) {
    return err_config("CiSpinSquare: tables are on device {}, but the call passed device {}",
                      dev_idx_, res.dev_idx());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::ci_spin_diagonal(res.stream(), size(), n_beta_, n_orbitals_,
                                             alpha_links_.data(), alpha_per_, beta_slots_.data(),
                                             constant_, target, lambda, diagonal, shifted));
}

Status ci_precondition(DeviceResources &res, const int32_t n, const double theta,
                       const double *const shifted, const double *const residual,
                       double *const correction) {
  return gpu_status(device::ci_precondition(res.stream(), n, theta, shifted, residual, correction));
}

Status ci_guesses(DeviceResources &res, const int32_t n, const int32_t count,
                  const double *const diagonal, double *const guess) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const wwr::wwrblasHandle_t blas = res.blas().get();
  Buffers::device<double> partial_values(device::kCiLowestBlocks, res.device());
  Buffers::device<int32_t> partial_index(device::kCiLowestBlocks, res.device());
  Buffers::device<int32_t> picks(static_cast<size_t>(count), res.device());
  FBF_TRY(gpu_status(device::ci_lowest(stream, n, diagonal, count, partial_values.data(),
                                       partial_index.data(), picks.data())));
  FBF_TRY(gpu_status(device::ci_guess(stream, n, count, picks.data(), guess)));

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

} // namespace free_breakfast
