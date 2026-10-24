/// @file
/// @brief The host sigma and diagonal, and CiSigma's uploads and launches.

module;

// The launchers, defined in ci_sigma.cu: declared in the GMF so they keep
// external linkage (see the header).
#include "ci_sigma_bridge.h"

module free_breakfast.ci.sigma;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;  // kOne, kZero
import free_breakfast.core.primitives; // math::align_up
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.recipes.linear_operator;
import free_breakfast.ci.strings;

#include "diagnostics.h"

namespace free_breakfast {

static_assert(LinearOperator<CiSigma, double>);

// What both entry points refuse: the shapes they would otherwise index past.
Status check_sigma_shapes(const StringSet &alpha, const StringSet &beta, size_t h_eff, size_t eri,
                          size_t vector) {
  if (alpha.n_orbitals() != beta.n_orbitals()) {
    return err_config("ci sigma: alpha strings span {} orbitals, beta {}", alpha.n_orbitals(),
                      beta.n_orbitals());
  }
  const auto n = static_cast<size_t>(alpha.n_orbitals());
  if (h_eff != n * n || eri != n * n * n * n) {
    return err_config("ci sigma: {} orbitals want h_eff of {} and eri of {}, given {} and {}", n,
                      n * n, n * n * n * n, h_eff, eri);
  }
  const size_t dim = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  if (vector != dim) {
    return err_config("ci sigma: {} determinants, given a vector of {}", dim, vector);
  }
  return {};
}

Status host_sigma(const StringSet &alpha, const StringSet &beta, const double e_core,
                  const std::span<const double> h_eff, const std::span<const double> eri,
                  const std::span<const double> c, const std::span<double> sigma) {
  if (Status shapes = check_sigma_shapes(alpha, beta, h_eff.size(), eri.size(), c.size());
      !shapes) {
    return shapes;
  }
  if (sigma.size() != c.size()) {
    return err_config("ci sigma: c has {} elements, sigma {}", c.size(), sigma.size());
  }
  const auto n = static_cast<size_t>(alpha.n_orbitals());
  const auto nb = static_cast<size_t>(beta.size());
  const auto at = [&](size_t t, size_t u, size_t v, size_t w) {
    return eri[((t * n + u) * n + v) * n + w];
  };

  // H = E_core + Σ_tw k_tw E_tw + ½ Σ_tuvw (tu|vw) E_tu E_vw, where the
  // −δ_uv E_tw of the two-electron term has gone into
  // k_tw = h_tw − ½ Σ_u (tu|uw).
  std::vector<double> k(n * n);
  for (size_t t = 0; t < n; ++t) {
    for (size_t w = 0; w < n; ++w) {
      double sum = 0.0;
      for (size_t u = 0; u < n; ++u) {
        sum += at(t, u, u, w);
      }
      k[t * n + w] = h_eff[t * n + w] - 0.5 * sum;
    }
  }

  for (size_t i = 0; i < c.size(); ++i) {
    sigma[i] = e_core * c[i];
  }

  // For each determinant K = (K_α, K_β), whose replacement lists reach every
  // determinant one E_tu away:
  //   D_K[vw] = (E_vw c)_K = Σ_I <I|E_wv|K> c_I, gathered over K's lists,
  //   σ_K += Σ_vw k_vw D_K[vw],
  //   G_K[tu] = ½ Σ_vw (tu|vw) D_K[vw],
  //   σ_I += <I|E_tu|K> G_K[tu], scattered over K's lists.
  // E_tu = E^α_tu + E^β_tu, and K's α list moves K_α with K_β held (its β
  // list the reverse), so the gather and the scatter each walk both lists and
  // the product E_tu E_vw carries the α–α, β–β and α–β terms alike.
  std::vector<double> d(n * n);
  std::vector<double> g(n * n);
  for (int32_t ka = 0; ka < alpha.size(); ++ka) {
    for (int32_t kb = 0; kb < beta.size(); ++kb) {
      const size_t det = static_cast<size_t>(ka) * nb + static_cast<size_t>(kb);
      std::ranges::fill(d, 0.0);
      for (const Replacement &r : alpha.replacements(ka)) {
        const size_t from = static_cast<size_t>(r.target) * nb + static_cast<size_t>(kb);
        d[static_cast<size_t>(r.q) * n + static_cast<size_t>(r.p)] += r.sign * c[from];
      }
      for (const Replacement &r : beta.replacements(kb)) {
        const size_t from = static_cast<size_t>(ka) * nb + static_cast<size_t>(r.target);
        d[static_cast<size_t>(r.q) * n + static_cast<size_t>(r.p)] += r.sign * c[from];
      }

      double one_electron = 0.0;
      for (size_t vw = 0; vw < n * n; ++vw) {
        one_electron += k[vw] * d[vw];
      }
      sigma[det] += one_electron;

      for (size_t tu = 0; tu < n * n; ++tu) {
        double sum = 0.0;
        for (size_t vw = 0; vw < n * n; ++vw) {
          sum += eri[tu * n * n + vw] * d[vw];
        }
        g[tu] = 0.5 * sum;
      }

      for (const Replacement &r : alpha.replacements(ka)) {
        const size_t to = static_cast<size_t>(r.target) * nb + static_cast<size_t>(kb);
        sigma[to] += r.sign * g[static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)];
      }
      for (const Replacement &r : beta.replacements(kb)) {
        const size_t to = static_cast<size_t>(ka) * nb + static_cast<size_t>(r.target);
        sigma[to] += r.sign * g[static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)];
      }
    }
  }
  return {};
}

Status host_diagonal(const StringSet &alpha, const StringSet &beta, const double e_core,
                     const std::span<const double> h_eff, const std::span<const double> eri,
                     const std::span<double> diagonal) {
  if (Status shapes = check_sigma_shapes(alpha, beta, h_eff.size(), eri.size(), diagonal.size());
      !shapes) {
    return shapes;
  }
  const auto n = static_cast<size_t>(alpha.n_orbitals());
  const auto nb = static_cast<size_t>(beta.size());
  const auto at = [&](size_t t, size_t u, size_t v, size_t w) {
    return eri[((t * n + u) * n + v) * n + w];
  };

  // Spin orbital s is orbital s % n, alpha for s < n, so ascending s is α
  // ascending then β ascending.
  std::vector<size_t> occupied;
  for (int32_t ia = 0; ia < alpha.size(); ++ia) {
    for (int32_t ib = 0; ib < beta.size(); ++ib) {
      occupied.clear();
      const uint64_t sa = alpha.strings()[static_cast<size_t>(ia)];
      const uint64_t sb = beta.strings()[static_cast<size_t>(ib)];
      for (size_t o = 0; o < n; ++o) {
        if (((sa >> o) & 1U) != 0) {
          occupied.push_back(o);
        }
      }
      for (size_t o = 0; o < n; ++o) {
        if (((sb >> o) & 1U) != 0) {
          occupied.push_back(n + o);
        }
      }

      double sum = e_core;
      for (const size_t i : occupied) {
        sum += h_eff[(i % n) * n + (i % n)];
      }
      for (size_t x = 0; x < occupied.size(); ++x) {
        for (size_t y = x + 1; y < occupied.size(); ++y) {
          const size_t i = occupied[x] % n;
          const size_t j = occupied[y] % n;
          sum += at(i, i, j, j);
          if ((occupied[x] < n) == (occupied[y] < n)) {
            sum -= at(i, j, j, i);
          }
        }
      }
      diagonal[static_cast<size_t>(ia) * nb + static_cast<size_t>(ib)] = sum;
    }
  }
  return {};
}

// The bytes of D one chunk of apply's rows may take: big enough that the gemm
// over a chunk is a wide one, small enough that G dominates the scratch.
constexpr size_t kCiSigmaChunkBytes = size_t{64} << 20;

// The packed index of orbital pair (t, u), symmetric in the two.
size_t ci_pair(const size_t t, const size_t u) {
  const size_t hi = std::max(t, u);
  return (hi * (hi + 1) / 2) + std::min(t, u);
}

// A set's replacement lists as the device takes them: target, packed pair,
// sign, three int32_t per replacement, by string then list order.
std::vector<int32_t> ci_link_table(const StringSet &set) {
  std::vector<int32_t> links;
  links.reserve(static_cast<size_t>(set.size()) *
                static_cast<size_t>(set.replacements_per_string()) * 3);
  for (int32_t address = 0; address < set.size(); ++address) {
    for (const Replacement &r : set.replacements(address)) {
      links.push_back(r.target);
      links.push_back(
          static_cast<int32_t>(ci_pair(static_cast<size_t>(r.p), static_cast<size_t>(r.q))));
      links.push_back(r.sign);
    }
  }
  return links;
}

template <typename T>
Status ci_upload(const wwr::wwrStream_t stream, Buffers::device<T> &to,
                 const std::vector<T> &from) {
  return gpu_status(wwr::wwrMemcpyAsync(to.data(), from.data(), sizeof(T) * from.size(),
                                        wwr::wwrMemcpyHostToDevice, stream));
}

CiSigma::CiSigma(DeviceResources &res, const int32_t n_orbitals, const int32_t n_alpha,
                 const int32_t n_beta, const int32_t alpha_per, const int32_t beta_per)
    : dev_idx_{res.dev_idx()}, n_orbitals_{n_orbitals}, n_alpha_{n_alpha}, n_beta_{n_beta},
      alpha_per_{alpha_per}, beta_per_{beta_per},
      alpha_links_(static_cast<size_t>(n_alpha) * static_cast<size_t>(alpha_per) * 3, res.device()),
      beta_links_(static_cast<size_t>(n_beta) * static_cast<size_t>(beta_per) * 3, res.device()),
      alpha_strings_(static_cast<size_t>(n_alpha), res.device()),
      beta_strings_(static_cast<size_t>(n_beta), res.device()),
      contraction_(ci_pair(static_cast<size_t>(n_orbitals), 0) *
                       (ci_pair(static_cast<size_t>(n_orbitals), 0) + 1),
                   res.device()),
      diagonal_terms_(static_cast<size_t>(n_orbitals) * (1 + (2 * static_cast<size_t>(n_orbitals))),
                      res.device()) {}

Result<CiSigma> CiSigma::create(DeviceResources &res, const StringSet &alpha,
                                const StringSet &beta) {
  if (alpha.n_orbitals() != beta.n_orbitals()) {
    return err_config("CiSigma: alpha strings span {} orbitals, beta {}", alpha.n_orbitals(),
                      beta.n_orbitals());
  }
  if (alpha.n_orbitals() == 0) {
    return err_config("CiSigma: the strings span no orbitals");
  }
  if (static_cast<int64_t>(alpha.size()) * beta.size() > std::numeric_limits<int32_t>::max()) {
    return err_config("CiSigma: {} x {} determinants is more than an int32_t counts", alpha.size(),
                      beta.size());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  CiSigma sigma(res, alpha.n_orbitals(), alpha.size(), beta.size(), alpha.replacements_per_string(),
                beta.replacements_per_string());
  const std::vector<int32_t> alpha_links = ci_link_table(alpha);
  const std::vector<int32_t> beta_links = ci_link_table(beta);
  const std::vector<uint64_t> alpha_strings{alpha.strings().begin(), alpha.strings().end()};
  const std::vector<uint64_t> beta_strings{beta.strings().begin(), beta.strings().end()};
  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(ci_upload(stream, sigma.alpha_links_, alpha_links));
  FBF_TRY(ci_upload(stream, sigma.beta_links_, beta_links));
  FBF_TRY(ci_upload(stream, sigma.alpha_strings_, alpha_strings));
  FBF_TRY(ci_upload(stream, sigma.beta_strings_, beta_strings));
  // Pageable sources, which die at return.
  FBF_TRY(res.sync());
  return sigma;
}

Status CiSigma::set_integrals(DeviceResources &res, const double e_core,
                              const std::span<const double> h_eff,
                              const std::span<const double> eri) {
  if (res.dev_idx() != dev_idx_) {
    return err_config("CiSigma: tables are on device {}, but the call passed device {}", dev_idx_,
                      res.dev_idx());
  }
  const auto n = static_cast<size_t>(n_orbitals_);
  if (h_eff.size() != n * n || eri.size() != n * n * n * n) {
    return err_config("CiSigma: {} orbitals want h_eff of {} and eri of {}, given {} and {}", n,
                      n * n, n * n * n * n, h_eff.size(), eri.size());
  }
  const auto at = [&](size_t t, size_t u, size_t v, size_t w) {
    return eri[((t * n + u) * n + v) * n + w];
  };

  // Column pair(t, u) holds ½ (tu|vw) down the rows pair(v, w), and the last
  // column k_vw = h_vw − ½ Σ_u (vu|uw): host_sigma's split of H, packed.
  const size_t n_pairs = ci_pair(n, 0);
  std::vector<double> contraction(n_pairs * (n_pairs + 1));
  for (size_t v = 0; v < n; ++v) {
    for (size_t w = 0; w <= v; ++w) {
      const size_t row = ci_pair(v, w);
      for (size_t t = 0; t < n; ++t) {
        for (size_t u = 0; u <= t; ++u) {
          contraction[(ci_pair(t, u) * n_pairs) + row] = 0.5 * at(t, u, v, w);
        }
      }
      double sum = 0.0;
      for (size_t u = 0; u < n; ++u) {
        sum += at(v, u, u, w);
      }
      contraction[(n_pairs * n_pairs) + row] = h_eff[v * n + w] - 0.5 * sum;
    }
  }
  std::vector<double> terms(n * (1 + 2 * n));
  for (size_t i = 0; i < n; ++i) {
    terms[i] = h_eff[i * n + i];
    for (size_t j = 0; j < n; ++j) {
      terms[n + (i * n) + j] = at(i, i, j, j);
      terms[n + (n * n) + (i * n) + j] = at(i, j, j, i);
    }
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  // A failed upload leaves the tables half-written, so they count as unset
  // until both are in.
  has_integrals_ = false;
  FBF_TRY(ci_upload(res.stream(), contraction_, contraction));
  FBF_TRY(ci_upload(res.stream(), diagonal_terms_, terms));
  FBF_TRY(res.sync());
  e_core_ = e_core;
  has_integrals_ = true;
  return {};
}

Status CiSigma::apply(DeviceResources &res, const int32_t k, const double *const x,
                      double *const y) {
  if (!has_integrals_) {
    return err_config("CiSigma: apply before set_integrals");
  }
  if (res.dev_idx() != dev_idx_) {
    return err_config("CiSigma: tables are on device {}, but the call passed device {}", dev_idx_,
                      res.dev_idx());
  }
  if (k < 0 || static_cast<int64_t>(k) * size() > std::numeric_limits<int32_t>::max()) {
    return err_config("CiSigma: a block of {} vectors of {} is outside an int32_t's rows", k,
                      size());
  }
  if (k == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const int32_t rows = k * size();
  const auto n_pairs = static_cast<int32_t>(ci_pair(static_cast<size_t>(n_orbitals_), 0));
  const auto pairs = static_cast<size_t>(n_pairs);
  const int32_t chunk = static_cast<int32_t>(std::clamp<size_t>(
      kCiSigmaChunkBytes / (sizeof(double) * pairs), 1, static_cast<size_t>(rows)));

  // G for every row, then D for one chunk.
  const size_t g_bytes = math::align_up(sizeof(double) * static_cast<size_t>(rows) * (pairs + 1),
                                        DeviceResources::kWorkspaceAlignment);
  const size_t d_bytes = sizeof(double) * static_cast<size_t>(chunk) * pairs;
  std::byte *const base = res.device_workspace(g_bytes + d_bytes).data();
  auto *const g = reinterpret_cast<double *>(base);
  auto *const d = reinterpret_cast<double *>(base + g_bytes);

  for (int32_t begin = 0; begin < rows; begin += chunk) {
    const int32_t count = std::min(chunk, rows - begin);
    FBF_TRY(gpu_status(
        wwr::wwrMemsetAsync(d, 0, sizeof(double) * static_cast<size_t>(count) * pairs, stream)));
    FBF_TRY(
        gpu_status(device::ci_gather_d(stream, begin, count, size(), n_beta_, alpha_links_.data(),
                                       alpha_per_, beta_links_.data(), beta_per_, x, d)));
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
        res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, count, n_pairs + 1, n_pairs,
        &kOne<double>, d, count, contraction_.data(), n_pairs, &kZero<double>, g + begin, rows)));
  }
  return gpu_status(device::ci_sigma_from_g(stream, rows, size(), n_beta_, n_pairs,
                                            alpha_links_.data(), alpha_per_, beta_links_.data(),
                                            beta_per_, e_core_, x, g, y));
}

Status CiSigma::diagonal(DeviceResources &res, double *const diagonal) {
  if (!has_integrals_) {
    return err_config("CiSigma: diagonal before set_integrals");
  }
  if (res.dev_idx() != dev_idx_) {
    return err_config("CiSigma: tables are on device {}, but the call passed device {}", dev_idx_,
                      res.dev_idx());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::ci_diagonal(res.stream(), n_orbitals_, n_alpha_, n_beta_,
                                        alpha_strings_.data(), beta_strings_.data(), e_core_,
                                        diagonal_terms_.data(), diagonal));
}

} // namespace free_breakfast
