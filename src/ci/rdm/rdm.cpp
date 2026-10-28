/// @file
/// @brief Rdm's launches: the state-averaged 1- and 2-RDM on the device.

module;

// The launcher, defined in ci_rdm.cu: declared in the GMF so it keeps external
// linkage (see the header).
#include "ci_rdm_bridge.h"

module free_breakfast.ci.rdm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma; // Layout

#include "diagnostics.h"

namespace free_breakfast::ci {

// What rdm12 refuses in the weights: none, or any that do not make an
// average.
Status check_rdm12_weights(const std::span<const double> weights) {
  if (weights.empty()) {
    return err_config("ci::Rdm::rdm12: no weights");
  }
  double sum = 0.0;
  for (const double w : weights) {
    if (!(w >= 0.0)) {
      return err_config("ci::Rdm::rdm12: weight {} is negative", w);
    }
    sum += w;
  }
  if (!(std::abs(sum - 1.0) <= 1e-12)) {
    return err_config("ci::Rdm::rdm12: the {} weights sum to {:.17g}, not 1", weights.size(), sum);
  }
  return {};
}

// The bytes of D one chunk of determinants may take: big enough that the
// products over a chunk are wide ones, small enough to stay a modest scratch.
constexpr size_t kRdmChunkBytes = size_t{64} << 20;

Result<Rdm> Rdm::create(DeviceResources &res, const DeviceStringSet &alpha,
                        const DeviceStringSet &beta) {
  if (alpha.dev_idx() != res.dev_idx() || beta.dev_idx() != res.dev_idx()) {
    return err_config("ci::Rdm: strings are on devices {} and {}, but the call passed device {}",
                      alpha.dev_idx(), beta.dev_idx(), res.dev_idx());
  }
  if (alpha.n_orbitals() != beta.n_orbitals()) {
    return err_config("ci::Rdm: alpha strings span {} orbitals, beta {}", alpha.n_orbitals(),
                      beta.n_orbitals());
  }
  if (alpha.n_orbitals() == 0) {
    return err_config("ci::Rdm: the strings span no orbitals");
  }
  if (static_cast<int64_t>(alpha.size()) * beta.size() > std::numeric_limits<int32_t>::max()) {
    return err_config("ci::Rdm: {} x {} determinants is more than an int32_t counts", alpha.size(),
                      beta.size());
  }
  return Rdm(res, alpha, beta);
}

Rdm::Rdm(DeviceResources &res, const DeviceStringSet &alpha, const DeviceStringSet &beta)
    : dev_idx_{res.dev_idx()}, n_orbitals_{alpha.n_orbitals()}, n_alpha_{alpha.size()},
      n_beta_{beta.size()}, alpha_per_{alpha.replacements_per_string()},
      beta_per_{beta.replacements_per_string()},
      one_set_{alpha.n_electrons() == beta.n_electrons()},
      alpha_links_{alpha.device_replacements()}, beta_links_{beta.device_replacements()} {}

Status Rdm::rdm12(DeviceResources &res, const std::span<const double> weights,
                  const double *const vectors, double *const gamma, double *const gamma2,
                  const Layout layout) {
  if (res.dev_idx() != dev_idx_) {
    return err_config("ci::Rdm: tables are on device {}, but the call passed device {}", dev_idx_,
                      res.dev_idx());
  }
  FBF_TRY(check_rdm12_weights(weights));
  const bool symmetric = layout == Layout::symmetric;
  if (symmetric && !one_set_) {
    return err_config("ci::Rdm: the symmetric layout needs one set of strings for both spins");
  }
  const auto n = static_cast<size_t>(n_orbitals_);
  // A vector's rows: every determinant, or the triangle's.
  const int32_t rows = size(layout);
  const auto dim = static_cast<size_t>(rows);
  const size_t nn = n * n;
  const auto nn32 = static_cast<int32_t>(nn);
  const auto chunk =
      static_cast<int32_t>(std::clamp<size_t>(kRdmChunkBytes / (sizeof(double) * nn), 1, dim));

  // The workspace: M, then one chunk of D.
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const auto [m, d] = res.device_regions<double>(nn * nn, static_cast<size_t>(chunk) * nn);
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(m, 0, sizeof(double) * nn * nn, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(gamma, 0, sizeof(double) * nn, stream)));

  // Per root and chunk of determinants K: D_{tu,K} = ⟨c| E_tu |K⟩ as a
  // count × n² column-major block, then M += w Dᵀ D, its lower triangle, and
  // γ += w Dᵀ c. In the symmetric layout K is a triangle element, its row of D
  // scaled by √2 off the diagonal, and c the packed vector.
  for (size_t r = 0; r < weights.size(); ++r) {
    if (weights[r] == 0.0) {
      continue;
    }
    const double *const c = vectors + r * dim;
    for (int32_t begin = 0; begin < rows; begin += chunk) {
      const int32_t count = std::min(chunk, rows - begin);
      FBF_TRY(gpu_status(
          wwr::wwrMemsetAsync(d, 0, sizeof(double) * static_cast<size_t>(count) * nn, stream)));
      FBF_TRY(
          gpu_status(device::rdm_gather_d(stream, symmetric, begin, count, n_orbitals_, n_beta_,
                                          alpha_links_, alpha_per_, beta_links_, beta_per_, c, d)));
      FBF_TRY(gpu_status(wwr::syrk<double, int32_t>(res.blas().get(), wwr::WWRBLAS_FILL_MODE_LOWER,
                                                    wwr::WWRBLAS_OP_T, nn32, count, &weights[r], d,
                                                    count, &kOne<double>, m, nn32)));
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
          res.blas().get(), wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nn32, 1, count, &weights[r], d,
          count, c + begin, static_cast<int32_t>(dim), &kOne<double>, gamma, nn32)));
    }
  }
  return gpu_status(device::rdm_assemble(stream, n_orbitals_, m, gamma, gamma2));
}

} // namespace free_breakfast::ci
