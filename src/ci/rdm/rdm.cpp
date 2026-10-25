/// @file
/// @brief The spin-summed 1-RDM, and the state-averaged 1- and 2-RDM on the
/// host and the device.

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

#include "diagnostics.h"

namespace free_breakfast {

Status rdm1(const StringSet &alpha, const StringSet &beta, const std::span<const double> c,
            const std::span<double> gamma) {
  const size_t dim = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  const auto n = static_cast<size_t>(alpha.n_orbitals());
  if (alpha.n_orbitals() != beta.n_orbitals() || c.size() != dim || gamma.size() != n * n) {
    return err_config("ci rdm1: orbitals {} and {}, {} determinants, given c of {} and gamma of {}",
                      alpha.n_orbitals(), beta.n_orbitals(), dim, c.size(), gamma.size());
  }
  const auto nb = static_cast<size_t>(beta.size());
  std::ranges::fill(gamma, 0.0);
  // a†_p a_q |K> = sign |J> adds c_J sign c_K to γ_pq, for each spin's list.
  for (int32_t ka = 0; ka < alpha.size(); ++ka) {
    for (int32_t kb = 0; kb < beta.size(); ++kb) {
      const double ck = c[static_cast<size_t>(ka) * nb + static_cast<size_t>(kb)];
      for (const Replacement &r : alpha.replacements(ka)) {
        const size_t j = static_cast<size_t>(r.target) * nb + static_cast<size_t>(kb);
        gamma[static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)] += r.sign * c[j] * ck;
      }
      for (const Replacement &r : beta.replacements(kb)) {
        const size_t j = static_cast<size_t>(ka) * nb + static_cast<size_t>(r.target);
        gamma[static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)] += r.sign * c[j] * ck;
      }
    }
  }
  return {};
}

// What both rdm12s refuse in the weights: none, or any that do not make an
// average.
Status check_rdm12_weights(const std::span<const double> weights) {
  if (weights.empty()) {
    return err_config("ci rdm12: no weights");
  }
  double sum = 0.0;
  for (const double w : weights) {
    if (!(w >= 0.0)) {
      return err_config("ci rdm12: weight {} is negative", w);
    }
    sum += w;
  }
  if (!(std::abs(sum - 1.0) <= 1e-12)) {
    return err_config("ci rdm12: the {} weights sum to {:.17g}, not 1", weights.size(), sum);
  }
  return {};
}

Status host_rdm12(const StringSet &alpha, const StringSet &beta,
                  const std::span<const double> vectors, const std::span<const double> weights,
                  const std::span<double> gamma, const std::span<double> gamma2) {
  const size_t dim = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  const auto n = static_cast<size_t>(alpha.n_orbitals());
  if (alpha.n_orbitals() != beta.n_orbitals() || vectors.size() != weights.size() * dim ||
      gamma.size() != n * n || gamma2.size() != n * n * n * n) {
    return err_config("ci rdm12: orbitals {} and {}, {} determinants, {} weights, given vectors "
                      "of {}, gamma of {} and gamma2 of {}",
                      alpha.n_orbitals(), beta.n_orbitals(), dim, weights.size(), vectors.size(),
                      gamma.size(), gamma2.size());
  }
  FBF_TRY(check_rdm12_weights(weights));
  const size_t nn = n * n;
  const auto nb = static_cast<size_t>(beta.size());

  std::vector<double> one(nn, 0.0);
  std::vector<double> two(nn * nn, 0.0);
  std::vector<double> root_gamma(nn);
  std::vector<double> d(nn);
  for (size_t r = 0; r < weights.size(); ++r) {
    const double weight = weights[r];
    const std::span<const double> c = vectors.subspan(r * dim, dim);
    FBF_TRY(rdm1(alpha, beta, c, root_gamma));
    for (size_t tu = 0; tu < nn; ++tu) {
      one[tu] += weight * root_gamma[tu];
    }
    // ⟨c| E_tu E_vw |c⟩ = Σ_K ⟨c| E_tu |K⟩ ⟨K| E_vw |c⟩, and for c real
    // ⟨K| E_vw |c⟩ = ⟨c| E_wv |K⟩. a†_p a_q |K> = sign |J> in K's lists is
    // ⟨J| E_pq |K⟩ = sign, so d_pq = ⟨c| E_pq |K⟩ gathers c_J sign.
    for (int32_t ka = 0; ka < alpha.size(); ++ka) {
      for (int32_t kb = 0; kb < beta.size(); ++kb) {
        std::ranges::fill(d, 0.0);
        for (const Replacement &x : alpha.replacements(ka)) {
          const size_t j = static_cast<size_t>(x.target) * nb + static_cast<size_t>(kb);
          d[static_cast<size_t>(x.p) * n + static_cast<size_t>(x.q)] += x.sign * c[j];
        }
        for (const Replacement &x : beta.replacements(kb)) {
          const size_t j = static_cast<size_t>(ka) * nb + static_cast<size_t>(x.target);
          d[static_cast<size_t>(x.p) * n + static_cast<size_t>(x.q)] += x.sign * c[j];
        }
        for (size_t tu = 0; tu < nn; ++tu) {
          for (size_t v = 0; v < n; ++v) {
            for (size_t w = 0; w < n; ++w) {
              two[tu * nn + v * n + w] += weight * d[tu] * d[w * n + v];
            }
          }
        }
      }
    }
  }
  // − δ_uv E_tw, weighted: − δ_uv γ_tw.
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      for (size_t w = 0; w < n; ++w) {
        two[((t * n + u) * n + u) * n + w] -= one[t * n + w];
      }
    }
  }
  std::ranges::copy(one, gamma.begin());
  std::ranges::copy(two, gamma2.begin());
  return {};
}

// The bytes of D one chunk of determinants may take: big enough that the
// products over a chunk are wide ones, small enough to stay a modest scratch.
constexpr size_t kRdmChunkBytes = size_t{64} << 20;

// A set's replacement lists as CiRdm's gather takes them: target, unpacked
// pair p · n + q, sign, three int32_t per replacement, by string then list
// order.
std::vector<int32_t> rdm_link_table(const StringSet &set) {
  const auto n = static_cast<size_t>(set.n_orbitals());
  std::vector<int32_t> links;
  links.reserve(static_cast<size_t>(set.size()) *
                static_cast<size_t>(set.replacements_per_string()) * 3);
  for (int32_t address = 0; address < set.size(); ++address) {
    for (const Replacement &r : set.replacements(address)) {
      links.push_back(r.target);
      links.push_back(
          static_cast<int32_t>(static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)));
      links.push_back(r.sign);
    }
  }
  return links;
}

CiRdm::CiRdm(DeviceResources &res, const int32_t n_orbitals, const int32_t n_alpha,
             const int32_t n_beta, const int32_t alpha_per, const int32_t beta_per)
    : dev_idx_{res.dev_idx()}, n_orbitals_{n_orbitals}, n_alpha_{n_alpha}, n_beta_{n_beta},
      alpha_per_{alpha_per}, beta_per_{beta_per},
      alpha_links_(static_cast<size_t>(n_alpha) * static_cast<size_t>(alpha_per) * 3, res.device()),
      beta_links_(static_cast<size_t>(n_beta) * static_cast<size_t>(beta_per) * 3, res.device()) {}

Result<CiRdm> CiRdm::create(DeviceResources &res, const StringSet &alpha, const StringSet &beta) {
  if (alpha.n_orbitals() != beta.n_orbitals()) {
    return err_config("CiRdm: alpha strings span {} orbitals, beta {}", alpha.n_orbitals(),
                      beta.n_orbitals());
  }
  if (alpha.n_orbitals() == 0) {
    return err_config("CiRdm: the strings span no orbitals");
  }
  if (static_cast<int64_t>(alpha.size()) * beta.size() > std::numeric_limits<int32_t>::max()) {
    return err_config("CiRdm: {} x {} determinants is more than an int32_t counts", alpha.size(),
                      beta.size());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  CiRdm rdm(res, alpha.n_orbitals(), alpha.size(), beta.size(), alpha.replacements_per_string(),
            beta.replacements_per_string());
  const std::vector<int32_t> alpha_links = rdm_link_table(alpha);
  const std::vector<int32_t> beta_links = rdm_link_table(beta);
  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(rdm.alpha_links_.data(), alpha_links.data(),
                                         sizeof(int32_t) * alpha_links.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(rdm.beta_links_.data(), beta_links.data(),
                                         sizeof(int32_t) * beta_links.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  // Pageable sources, which die at return.
  FBF_TRY(res.sync());
  return rdm;
}

Status CiRdm::rdm12(DeviceResources &res, const std::span<const double> weights,
                    const double *const vectors, double *const gamma, double *const gamma2) {
  if (res.dev_idx() != dev_idx_) {
    return err_config("CiRdm: tables are on device {}, but the call passed device {}", dev_idx_,
                      res.dev_idx());
  }
  FBF_TRY(check_rdm12_weights(weights));
  const auto n = static_cast<size_t>(n_orbitals_);
  const auto dim = static_cast<size_t>(size());
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
  // γ += w Dᵀ c.
  for (size_t r = 0; r < weights.size(); ++r) {
    if (weights[r] == 0.0) {
      continue;
    }
    const double *const c = vectors + r * dim;
    for (int32_t begin = 0; begin < size(); begin += chunk) {
      const int32_t count = std::min(chunk, size() - begin);
      FBF_TRY(gpu_status(
          wwr::wwrMemsetAsync(d, 0, sizeof(double) * static_cast<size_t>(count) * nn, stream)));
      FBF_TRY(gpu_status(device::ci_rdm_gather_d(stream, begin, count, n_beta_, alpha_links_.data(),
                                                 alpha_per_, beta_links_.data(), beta_per_, c, d)));
      FBF_TRY(gpu_status(wwr::syrk<double, int32_t>(res.blas().get(), wwr::WWRBLAS_FILL_MODE_LOWER,
                                                    wwr::WWRBLAS_OP_T, nn32, count, &weights[r], d,
                                                    count, &kOne<double>, m, nn32)));
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
          res.blas().get(), wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nn32, 1, count, &weights[r], d,
          count, c + begin, static_cast<int32_t>(dim), &kOne<double>, gamma, nn32)));
    }
  }
  return gpu_status(device::ci_rdm_assemble(stream, n_orbitals_, m, gamma, gamma2));
}

} // namespace free_breakfast
