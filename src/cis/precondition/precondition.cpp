/// @file
/// @brief The diagonal's and the preconditioner's launches, and the guess's
/// host select.

module;

// The launchers, defined in cis_precondition.cu: declared in the GMF so they
// keep external linkage (see the header).
#include "cis_precondition_bridge.h"

module free_breakfast.cis.precondition;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.recipes.davidson; // DavidsonPreconditioner

#include "diagnostics.h"

namespace free_breakfast::cis {

static_assert(DavidsonPreconditioner<Preconditioner, double>);

Status diagonal(DeviceResources &res, const int32_t n_occ, const int32_t n_virt,
                const double *const orbital_energies, double *const delta) {
  if (n_occ < 1 || n_virt < 1 ||
      static_cast<int64_t>(n_occ) * n_virt > std::numeric_limits<int32_t>::max()) {
    return err_config("cis::diagonal: needs n_occ >= 1, n_virt >= 1 and n_occ * n_virt to fit an "
                      "int32_t (n_occ = {}, n_virt = {})",
                      n_occ, n_virt);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::diagonal(res.stream(), n_occ, n_virt, orbital_energies, delta));
}

Status guess(DeviceResources &res, const int32_t n_ov, const double *const delta,
             const int32_t guess_count, double *const vectors) {
  if (guess_count < 1 || guess_count > n_ov) {
    return err_config("cis::guess: needs 1 <= guess_count <= n_ov (guess_count = {}, n_ov = {})",
                      guess_count, n_ov);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const auto n = static_cast<size_t>(n_ov);
  const auto count = static_cast<size_t>(guess_count);

  // The select is a host partial sort: it runs once per solve, over a vector
  // a few MB at most, and a host comparator on (Δε, index) is what makes the
  // tie order exact and the result deterministic.
  std::vector<double> host(n);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), delta, n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  std::vector<int32_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + guess_count, order.end(),
                    [&host](const int32_t l, const int32_t r) {
                      return std::tie(host[static_cast<size_t>(l)], l) <
                             std::tie(host[static_cast<size_t>(r)], r);
                    });

  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(vectors, 0, n * count * sizeof(double), stream)));
  // kOne is static storage, so the pageable source outlives the copy and
  // nothing here needs a sync.
  for (size_t g = 0; g < count; ++g) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(vectors + (g * n) + static_cast<size_t>(order[g]),
                                           &kOne<double>, sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
  }
  return {};
}

Preconditioner::Preconditioner(DeviceResources &res, const int32_t n_ov, const double *const delta,
                               const int32_t max_columns)
    : dev_idx_{res.dev_idx()}, n_ov_{n_ov}, delta_{delta},
      theta_(static_cast<size_t>(std::max(max_columns, 0)), res.device()) {}

Status Preconditioner::operator()(DeviceResources &res, const std::span<const double> theta,
                                  const double *const /*ritz_vectors*/,
                                  const double *const residual, double *const correction) const {
  if (res.dev_idx() != dev_idx_) {
    return err_config("cis::Preconditioner: built on device {}, called on device {}", dev_idx_,
                      res.dev_idx());
  }
  if (theta.size() > theta_.num_elements()) {
    return err_config("cis::Preconditioner: {} columns, more than the {} it was built for",
                      theta.size(), theta_.num_elements());
  }
  if (theta.empty()) {
    return {};
  }
  const ScopedDeviceIndex on{dev_idx_};
  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(theta_.data(), theta.data(), theta.size_bytes(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(device::precondition(stream, n_ov_, static_cast<int32_t>(theta.size()),
                                          theta_.data(), delta_, residual, correction)));
  // theta is the caller's pageable memory, which a pageable upload does not
  // promise to have read by return (docs/style.md "Host-device copies").
  return res.sync();
}

} // namespace free_breakfast::cis
