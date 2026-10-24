// Host cases for ci.sigma: σ = Hc and diag(H) against a dense H built by the
// Slater–Condon rules directly on determinants, as spin-orbital bit patterns,
// which shares nothing with the module's replacement lists; and that dense H's
// lowest eigenvalues against PySCF's fci.direct_spin1 (data/ci/direct_spin1.json,
// written by tools/emit_fci_reference.py).

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.json;
import free_breakfast.ci;

namespace {

namespace fb = free_breakfast;

// The active-space Hamiltonian as host_sigma takes it.
struct Integrals {
  std::int32_t n;
  double e_core;
  std::vector<double> h1;  // n x n
  std::vector<double> eri; // n^4, (tu|vw)
};

// Random real integrals with the symmetry of real orbitals: h symmetric, eri
// 8-fold. Not a physical Hamiltonian, so nothing about it is special.
Integrals random_integrals(std::int32_t n, std::uint64_t seed) {
  std::mt19937_64 engine{seed};
  std::uniform_real_distribution<double> uniform{-1.0, 1.0};
  const auto m = static_cast<std::size_t>(n);
  Integrals out{n, uniform(engine), std::vector<double>(m * m), std::vector<double>(m * m * m * m)};
  for (std::size_t t = 0; t < m; ++t) {
    for (std::size_t u = 0; u <= t; ++u) {
      out.h1[t * m + u] = out.h1[u * m + t] = uniform(engine);
    }
  }
  const auto at = [&](std::size_t t, std::size_t u, std::size_t v, std::size_t w) -> double & {
    return out.eri[((t * m + u) * m + v) * m + w];
  };
  for (std::size_t t = 0; t < m; ++t) {
    for (std::size_t u = 0; u <= t; ++u) {
      for (std::size_t v = 0; v < m; ++v) {
        for (std::size_t w = 0; w <= v; ++w) {
          if (t * (t + 1) / 2 + u < v * (v + 1) / 2 + w) {
            continue;
          }
          const double x = uniform(engine);
          at(t, u, v, w) = at(u, t, v, w) = at(t, u, w, v) = at(u, t, w, v) = x;
          at(v, w, t, u) = at(w, v, t, u) = at(v, w, u, t) = at(w, v, u, t) = x;
        }
      }
    }
  }
  return out;
}

// A random vector of unit norm.
std::vector<double> random_vector(std::size_t size, std::uint64_t seed) {
  std::mt19937_64 engine{seed};
  std::uniform_real_distribution<double> uniform{-1.0, 1.0};
  std::vector<double> c(size);
  double norm = 0.0;
  for (double &x : c) {
    x = uniform(engine);
    norm += x * x;
  }
  for (double &x : c) {
    x /= std::sqrt(norm);
  }
  return c;
}

// The operator on spin orbital k (orbital k % n, alpha for k < n) applied to
// |s>, s a 2n-bit pattern whose creators are ordered by ascending k: the sign
// it picks up passing the occupied spin orbitals below k, or 0 when it
// annihilates the state.
std::int32_t apply(std::uint64_t &s, std::int32_t k, bool create) {
  const bool occupied = ((s >> k) & 1U) != 0;
  if (occupied == create) {
    return 0;
  }
  const std::uint64_t below = (std::uint64_t{1} << k) - 1;
  s ^= std::uint64_t{1} << k;
  return std::popcount(s & below) % 2 == 0 ? 1 : -1;
}

// H over the determinants (I_a, I_b), alpha-major, by the Slater–Condon rules
// on spin-orbital patterns s = I_a | I_b << n.
std::vector<double> dense_hamiltonian(const fb::StringSet &alpha, const fb::StringSet &beta,
                                      const Integrals &ints) {
  const std::int32_t n = ints.n;
  const auto m = static_cast<std::size_t>(n);
  // Spin-orbital integrals: zero unless each charge distribution is one spin.
  const auto h = [&](std::int32_t a, std::int32_t b) {
    return a / n == b / n
               ? ints.h1[static_cast<std::size_t>(a % n) * m + static_cast<std::size_t>(b % n)]
               : 0.0;
  };
  const auto g = [&](std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t d) {
    if (a / n != b / n || c / n != d / n) {
      return 0.0;
    }
    const auto i = [&](std::int32_t x) { return static_cast<std::size_t>(x % n); };
    return ints.eri[((i(a) * m + i(b)) * m + i(c)) * m + i(d)];
  };

  std::vector<std::uint64_t> dets;
  for (const std::uint64_t sa : alpha.strings()) {
    for (const std::uint64_t sb : beta.strings()) {
      dets.push_back(sa | (sb << n));
    }
  }
  const auto bits = [](std::uint64_t s) {
    std::vector<std::int32_t> out;
    for (; s != 0; s &= s - 1) {
      out.push_back(std::countr_zero(s));
    }
    return out;
  };

  const std::size_t dim = dets.size();
  std::vector<double> hm(dim * dim, 0.0);
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      const std::uint64_t bra = dets[row];
      const std::uint64_t ket = dets[col];
      const std::vector<std::int32_t> particles = bits(bra & ~ket);
      const std::vector<std::int32_t> holes = bits(ket & ~bra);
      double value = 0.0;
      if (particles.empty()) {
        // Summed in the order host_diagonal documents, so the two agree
        // exactly: E_core, the h_ii, then each pair i < j.
        const std::vector<std::int32_t> occ = bits(ket);
        value = ints.e_core;
        for (const std::int32_t i : occ) {
          value += h(i, i);
        }
        for (std::size_t x = 0; x < occ.size(); ++x) {
          for (std::size_t y = x + 1; y < occ.size(); ++y) {
            value += g(occ[x], occ[x], occ[y], occ[y]);
            if (occ[x] / n == occ[y] / n) {
              value -= g(occ[x], occ[y], occ[y], occ[x]);
            }
          }
        }
      } else if (particles.size() == 1) {
        // <bra| = sign <ket| a_q† a_p ... : bra = sign a_p† a_q |ket>.
        const std::int32_t p = particles[0];
        const std::int32_t q = holes[0];
        std::uint64_t s = ket;
        const std::int32_t sign = apply(s, q, false) * apply(s, p, true);
        double sum = h(p, q);
        for (const std::int32_t j : bits(ket)) {
          if (j != q) {
            sum += g(p, q, j, j) - g(p, j, j, q);
          }
        }
        value = sign * sum;
      } else if (particles.size() == 2) {
        // bra = sign a_a† a_b† a_j a_i |ket>, and <bra|H|ket> = sign <ab||ij>.
        const std::int32_t a = particles[0];
        const std::int32_t b = particles[1];
        const std::int32_t i = holes[0];
        const std::int32_t j = holes[1];
        std::uint64_t s = ket;
        const std::int32_t sign =
            apply(s, i, false) * apply(s, j, false) * apply(s, b, true) * apply(s, a, true);
        value = sign * (g(a, i, b, j) - g(a, j, b, i));
      }
      hm[row * dim + col] = value;
    }
  }
  return hm;
}

// The eigenvalues of a dense symmetric matrix, ascending: Householder to
// tridiagonal, then bisection on the Sturm count. Backward stable, so each is
// within a few ulps of ‖A‖.
std::vector<double> eigenvalues(std::vector<double> a, std::size_t n) {
  const auto at = [&](std::size_t i, std::size_t j) -> double & { return a[i * n + j]; };
  std::vector<double> v(n);
  std::vector<double> p(n);
  for (std::size_t k = 0; k + 2 < n; ++k) {
    double norm = 0.0;
    for (std::size_t i = k + 1; i < n; ++i) {
      norm += at(i, k) * at(i, k);
    }
    norm = std::sqrt(norm);
    if (norm == 0.0) {
      continue;
    }
    const double alpha = at(k + 1, k) > 0.0 ? -norm : norm;
    double vnorm = 0.0;
    for (std::size_t i = k + 1; i < n; ++i) {
      v[i] = at(i, k) - (i == k + 1 ? alpha : 0.0);
      vnorm += v[i] * v[i];
    }
    vnorm = std::sqrt(vnorm);
    for (std::size_t i = k + 1; i < n; ++i) {
      v[i] /= vnorm;
    }
    // B <- (I - 2vv^T) B (I - 2vv^T) = B - 2 v w^T - 2 w v^T, w = p - (v.p) v.
    double vp = 0.0;
    for (std::size_t i = k + 1; i < n; ++i) {
      p[i] = 0.0;
      for (std::size_t j = k + 1; j < n; ++j) {
        p[i] += at(i, j) * v[j];
      }
      vp += v[i] * p[i];
    }
    for (std::size_t i = k + 1; i < n; ++i) {
      p[i] -= vp * v[i];
    }
    for (std::size_t i = k + 1; i < n; ++i) {
      for (std::size_t j = k + 1; j < n; ++j) {
        at(i, j) -= 2.0 * (v[i] * p[j] + p[i] * v[j]);
      }
    }
    at(k + 1, k) = at(k, k + 1) = alpha;
    for (std::size_t i = k + 2; i < n; ++i) {
      at(i, k) = at(k, i) = 0.0;
    }
  }
  std::vector<double> d(n);
  std::vector<double> e2(n, 0.0); // e2[i] = (A_{i,i-1})^2
  double lo = std::numeric_limits<double>::max();
  double hi = std::numeric_limits<double>::lowest();
  for (std::size_t i = 0; i < n; ++i) {
    d[i] = at(i, i);
    const double below = i > 0 ? std::abs(at(i, i - 1)) : 0.0;
    const double above = i + 1 < n ? std::abs(at(i + 1, i)) : 0.0;
    e2[i] = below * below;
    lo = std::min(lo, d[i] - below - above);
    hi = std::max(hi, d[i] + below + above);
  }
  // How many eigenvalues lie below x.
  const auto count = [&](double x) {
    std::size_t below = 0;
    double q = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
      q = d[i] - x - (i > 0 ? e2[i] / q : 0.0);
      if (q == 0.0) {
        q = -std::numeric_limits<double>::epsilon() * (std::abs(x) + 1.0);
      }
      below += q < 0.0 ? 1 : 0;
    }
    return below;
  };
  std::vector<double> out(n);
  for (std::size_t k = 0; k < n; ++k) {
    double a = lo;
    double b = hi;
    for (std::int32_t it = 0; it < 200 && b - a > 1e-14 * std::max(1.0, std::abs(a)); ++it) {
      const double mid = 0.5 * (a + b);
      (count(mid) > k ? b : a) = mid;
    }
    out[k] = 0.5 * (a + b);
  }
  return out;
}

// Check host_sigma on random vectors and host_diagonal against the dense H,
// and return that H.
std::vector<double> check_against_dense(const fb::StringSet &alpha, const fb::StringSet &beta,
                                        const Integrals &ints) {
  const std::vector<double> hm = dense_hamiltonian(alpha, beta, ints);
  const std::size_t dim =
      static_cast<std::size_t>(alpha.size()) * static_cast<std::size_t>(beta.size());

  std::vector<double> diagonal(dim);
  const auto diag_status = fb::host_diagonal(alpha, beta, ints.e_core, ints.h1, ints.eri, diagonal);
  EXPECT_TRUE(diag_status.has_value()) << diag_status.error().message;
  for (std::size_t i = 0; i < dim; ++i) {
    EXPECT_EQ(diagonal[i], hm[i * dim + i]) << "determinant " << i;
  }

  for (std::uint64_t seed = 1; seed <= 3; ++seed) {
    const std::vector<double> c = random_vector(dim, seed);
    std::vector<double> sigma(dim);
    const auto status = fb::host_sigma(alpha, beta, ints.e_core, ints.h1, ints.eri, c, sigma);
    EXPECT_TRUE(status.has_value()) << status.error().message;
    double worst = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      double expected = 0.0;
      for (std::size_t j = 0; j < dim; ++j) {
        expected += hm[i * dim + j] * c[j];
      }
      worst = std::max(worst, std::abs(sigma[i] - expected));
    }
    EXPECT_LE(worst, 1e-12) << "seed " << seed;
  }
  return hm;
}

TEST(CiSigma, MatchesSlaterCondonOnRandomIntegrals) {
  // (n, N_alpha, N_beta): closed and open shells, an empty spin, a full one,
  // and a single orbital.
  const std::array<std::array<std::int32_t, 3>, 8> shapes{{
      {1, 1, 0},
      {2, 1, 1},
      {4, 2, 2},
      {5, 3, 2},
      {5, 1, 4},
      {6, 3, 3},
      {6, 0, 3},
      {6, 6, 5},
  }};
  for (const auto &[n, n_alpha, n_beta] : shapes) {
    SCOPED_TRACE(std::format("n = {}, N_alpha = {}, N_beta = {}", n, n_alpha, n_beta));
    const auto alpha = fb::StringSet::create(n, n_alpha);
    const auto beta = fb::StringSet::create(n, n_beta);
    ASSERT_TRUE(alpha.has_value() && beta.has_value());
    const Integrals ints = random_integrals(n, 100 + static_cast<std::uint64_t>(n));
    check_against_dense(*alpha, *beta, ints);
  }
}

TEST(CiSigma, DenseSpectrumMatchesDirectSpin1) {
  const auto doc =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "ci" / "direct_spin1.json");
  ASSERT_TRUE(doc.has_value()) << doc.error().message;
  ASSERT_FALSE((*doc)["cases"].empty());
  for (const auto &entry : (*doc)["cases"]) {
    SCOPED_TRACE(entry["name"].get<std::string>());
    const Integrals ints{entry["n_orbitals"].get<std::int32_t>(), entry["e_core"].get<double>(),
                         entry["h1"]["values"].get<std::vector<double>>(),
                         entry["eri"]["values"].get<std::vector<double>>()};
    const auto alpha = fb::StringSet::create(ints.n, entry["n_alpha"].get<std::int32_t>());
    const auto beta = fb::StringSet::create(ints.n, entry["n_beta"].get<std::int32_t>());
    ASSERT_TRUE(alpha.has_value() && beta.has_value());

    const std::vector<double> hm = check_against_dense(*alpha, *beta, ints);
    const std::size_t dim =
        static_cast<std::size_t>(alpha->size()) * static_cast<std::size_t>(beta->size());
    const std::vector<double> energies = eigenvalues(hm, dim);
    const auto &roots = entry["roots"];
    ASSERT_LE(roots.size(), dim);
    for (std::size_t k = 0; k < roots.size(); ++k) {
      EXPECT_NEAR(energies[k], roots[k]["energy"].get<double>(), 1e-10) << "root " << k;
    }
  }
}

TEST(CiSigma, RejectsMismatchedShapes) {
  const auto four = fb::StringSet::create(4, 2);
  const auto five = fb::StringSet::create(5, 2);
  ASSERT_TRUE(four.has_value() && five.has_value());
  const Integrals ints = random_integrals(4, 7);
  const std::size_t dim =
      static_cast<std::size_t>(four->size()) * static_cast<std::size_t>(four->size());
  std::vector<double> c(dim, 1.0);
  std::vector<double> sigma(dim, 0.0);

  const auto kind = [](const fb::Status &status) {
    return status.has_value() ? std::optional<fb::ErrorKind>{} : status.error().kind;
  };
  const auto invalid = std::optional{fb::ErrorKind::InvalidConfig};
  EXPECT_EQ(kind(fb::host_sigma(*four, *five, 0.0, ints.h1, ints.eri, c, sigma)), invalid);
  EXPECT_EQ(
      kind(fb::host_sigma(*four, *four, 0.0, std::span{ints.h1}.first(15), ints.eri, c, sigma)),
      invalid);
  EXPECT_EQ(
      kind(fb::host_sigma(*four, *four, 0.0, ints.h1, std::span{ints.eri}.first(255), c, sigma)),
      invalid);
  EXPECT_EQ(kind(fb::host_sigma(*four, *four, 0.0, ints.h1, ints.eri, std::span{c}.first(35),
                                std::span{sigma}.first(35))),
            invalid);
  EXPECT_EQ(
      kind(fb::host_sigma(*four, *four, 0.0, ints.h1, ints.eri, c, std::span{sigma}.first(35))),
      invalid);
  EXPECT_EQ(
      kind(fb::host_diagonal(*four, *four, 0.0, ints.h1, ints.eri, std::span{sigma}.first(35))),
      invalid);
  // Nothing was written by the refusals.
  EXPECT_TRUE(std::ranges::all_of(sigma, [](double x) { return x == 0.0; }));
}

} // namespace
