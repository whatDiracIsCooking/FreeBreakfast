/// @file
/// @brief The host references' loops.

module free_breakfast.test.ci_oracles;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.ci.strings;

#include "diagnostics.h"

namespace free_breakfast::test {

// What host_sigma and host_diagonal refuse: the shapes they would otherwise index past.
Status check_host_sigma_shapes(const ci::StringSet &alpha, const ci::StringSet &beta, size_t h_eff,
                               size_t eri, size_t vector) {
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

Status host_sigma(const ci::StringSet &alpha, const ci::StringSet &beta, const double e_core,
                  const std::span<const double> h_eff, const std::span<const double> eri,
                  const std::span<const double> c, const std::span<double> sigma) {
  if (Status shapes = check_host_sigma_shapes(alpha, beta, h_eff.size(), eri.size(), c.size());
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
      for (const ci::Replacement &r : alpha.replacements(ka)) {
        const size_t from = static_cast<size_t>(r.target) * nb + static_cast<size_t>(kb);
        d[static_cast<size_t>(r.q) * n + static_cast<size_t>(r.p)] += r.sign * c[from];
      }
      for (const ci::Replacement &r : beta.replacements(kb)) {
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

      for (const ci::Replacement &r : alpha.replacements(ka)) {
        const size_t to = static_cast<size_t>(r.target) * nb + static_cast<size_t>(kb);
        sigma[to] += r.sign * g[static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)];
      }
      for (const ci::Replacement &r : beta.replacements(kb)) {
        const size_t to = static_cast<size_t>(ka) * nb + static_cast<size_t>(r.target);
        sigma[to] += r.sign * g[static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)];
      }
    }
  }
  return {};
}

Status host_diagonal(const ci::StringSet &alpha, const ci::StringSet &beta, const double e_core,
                     const std::span<const double> h_eff, const std::span<const double> eri,
                     const std::span<double> diagonal) {
  if (Status shapes =
          check_host_sigma_shapes(alpha, beta, h_eff.size(), eri.size(), diagonal.size());
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

Status host_rdm1(const ci::StringSet &alpha, const ci::StringSet &beta,
                 const std::span<const double> c, const std::span<double> gamma) {
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
      for (const ci::Replacement &r : alpha.replacements(ka)) {
        const size_t j = static_cast<size_t>(r.target) * nb + static_cast<size_t>(kb);
        gamma[static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)] += r.sign * c[j] * ck;
      }
      for (const ci::Replacement &r : beta.replacements(kb)) {
        const size_t j = static_cast<size_t>(ka) * nb + static_cast<size_t>(r.target);
        gamma[static_cast<size_t>(r.p) * n + static_cast<size_t>(r.q)] += r.sign * c[j] * ck;
      }
    }
  }
  return {};
}

// What host_rdm12 refuses in the weights, as ci::Rdm::rdm12 does: none, or any
// that do not make an average.
Status check_host_rdm12_weights(const std::span<const double> weights) {
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

Status host_rdm12(const ci::StringSet &alpha, const ci::StringSet &beta,
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
  FBF_TRY(check_host_rdm12_weights(weights));
  const size_t nn = n * n;
  const auto nb = static_cast<size_t>(beta.size());

  std::vector<double> one(nn, 0.0);
  std::vector<double> two(nn * nn, 0.0);
  std::vector<double> root_gamma(nn);
  std::vector<double> d(nn);
  for (size_t r = 0; r < weights.size(); ++r) {
    const double weight = weights[r];
    const std::span<const double> c = vectors.subspan(r * dim, dim);
    FBF_TRY(host_rdm1(alpha, beta, c, root_gamma));
    for (size_t tu = 0; tu < nn; ++tu) {
      one[tu] += weight * root_gamma[tu];
    }
    // ⟨c| E_tu E_vw |c⟩ = Σ_K ⟨c| E_tu |K⟩ ⟨K| E_vw |c⟩, and for c real
    // ⟨K| E_vw |c⟩ = ⟨c| E_wv |K⟩. a†_p a_q |K> = sign |J> in K's lists is
    // ⟨J| E_pq |K⟩ = sign, so d_pq = ⟨c| E_pq |K⟩ gathers c_J sign.
    for (int32_t ka = 0; ka < alpha.size(); ++ka) {
      for (int32_t kb = 0; kb < beta.size(); ++kb) {
        std::ranges::fill(d, 0.0);
        for (const ci::Replacement &x : alpha.replacements(ka)) {
          const size_t j = static_cast<size_t>(x.target) * nb + static_cast<size_t>(kb);
          d[static_cast<size_t>(x.p) * n + static_cast<size_t>(x.q)] += x.sign * c[j];
        }
        for (const ci::Replacement &x : beta.replacements(kb)) {
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

Status host_spin_square(const ci::StringSet &alpha, const ci::StringSet &beta,
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
    const std::span<const ci::Replacement> beta_list = beta.replacements(kb);
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
      for (const ci::Replacement &ra : alpha.replacements(ka)) {
        const int32_t j = slot[static_cast<size_t>(ra.q) * n + static_cast<size_t>(ra.p)];
        if (j < 0) {
          continue;
        }
        const ci::Replacement &rb = beta_list[static_cast<size_t>(j)];
        s2c[static_cast<size_t>(ra.target) * nb + static_cast<size_t>(rb.target)] -=
            ra.sign * rb.sign * ck;
      }
    }
  }
  return {};
}

} // namespace free_breakfast::test
