/// @file
/// @brief The host sigma and diagonal.

module free_breakfast.ci.sigma;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.ci.strings;

namespace free_breakfast {

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

} // namespace free_breakfast
