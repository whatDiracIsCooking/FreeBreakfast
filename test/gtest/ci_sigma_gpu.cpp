// GPU cases for ci.sigma's CiSigma: σ on random blocks against host_sigma,
// diag(H) against host_diagonal, and the 1-RDM against ci.rdm's rdm1, over the
// golden active spaces of data/ci/direct_spin1.json, random integrals at
// CAS(8,8) and CAS(10,10) and in sectors with N_alpha != N_beta, and
// CAS(12,12) in the slow tier (its 1-RDM in the fast one); and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.recipes;
import free_breakfast.ci;

namespace {

namespace fb = free_breakfast;

static_assert(fb::LinearOperator<fb::CiSigma, double>);

// The active-space Hamiltonian as host_sigma and set_integrals take it.
struct Integrals {
  std::int32_t n;
  double e_core;
  std::vector<double> h1;  // n x n
  std::vector<double> eri; // n^4, (tu|vw)
};

// Random real integrals with the symmetry of real orbitals: h symmetric, eri
// 8-fold. The host sigma is the oracle, so nothing about them need be
// physical.
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

std::vector<double> random_block(std::size_t size, std::uint64_t seed) {
  std::mt19937_64 engine{seed};
  std::uniform_real_distribution<double> uniform{-1.0, 1.0};
  std::vector<double> c(size);
  for (double &x : c) {
    x = uniform(engine);
  }
  return c;
}

fb::Buffers::device<double> to_device(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return d;
}

std::vector<double> to_host(fb::DeviceResources &res, const fb::Buffers::device<double> &d) {
  std::vector<double> host(d.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// The 1-RDM of a random unit vector, to 1e-12 (N_α + N_β), its trace, of
// rdm1's on the host.
void check_rdm1(fb::DeviceResources &res, fb::CiSigma &sigma, const fb::StringSet &alpha,
                const fb::StringSet &beta, std::uint64_t seed) {
  const auto dim = static_cast<std::size_t>(sigma.size());
  const auto n = static_cast<std::size_t>(alpha.n_orbitals());
  std::vector<double> c = random_block(dim, seed);
  const double norm = std::sqrt(std::inner_product(c.begin(), c.end(), c.begin(), 0.0));
  for (double &x : c) {
    x /= norm;
  }
  const auto x = to_device(res, c);
  fb::Buffers::device<double> gamma(n * n, res.device());
  const auto device = sigma.rdm1(res, x.data(), gamma.data());
  ASSERT_TRUE(device.has_value()) << device.error().message;
  const std::vector<double> got = to_host(res, gamma);
  std::vector<double> expected(n * n);
  const auto host = fb::rdm1(alpha, beta, c, expected);
  ASSERT_TRUE(host.has_value()) << host.error().message;
  const double electrons = alpha.n_electrons() + beta.n_electrons();
  double worst = 0.0;
  for (std::size_t i = 0; i < n * n; ++i) {
    worst = std::max(worst, std::abs(got[i] - expected[i]));
  }
  EXPECT_LE(worst, 1e-12 * std::max(1.0, electrons));
}

// σ for a random block of k vectors, each column to 1e-12 ‖σ‖ of host_sigma's,
// diag(H) to 1e-12 of host_diagonal's, and the 1-RDM by check_rdm1.
void check_against_host(fb::DeviceResources &res, std::int32_t n_alpha, std::int32_t n_beta,
                        const Integrals &ints, std::int32_t k, std::uint64_t seed) {
  const auto alpha = fb::StringSet::create(ints.n, n_alpha);
  const auto beta = fb::StringSet::create(ints.n, n_beta);
  ASSERT_TRUE(alpha.has_value() && beta.has_value());
  auto sigma = fb::CiSigma::create(res, *alpha, *beta);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  const auto set = sigma->set_integrals(res, ints.e_core, ints.h1, ints.eri);
  ASSERT_TRUE(set.has_value()) << set.error().message;
  const auto dim = static_cast<std::size_t>(sigma->size());
  ASSERT_EQ(dim, static_cast<std::size_t>(alpha->size()) * static_cast<std::size_t>(beta->size()));

  const std::vector<double> c = random_block(dim * static_cast<std::size_t>(k), seed);
  const auto x = to_device(res, c);
  fb::Buffers::device<double> y(c.size(), res.device());
  const auto applied = sigma->apply(res, k, x.data(), y.data());
  ASSERT_TRUE(applied.has_value()) << applied.error().message;
  const std::vector<double> got = to_host(res, y);

  std::vector<double> expected(dim);
  for (std::int32_t j = 0; j < k; ++j) {
    const std::span<const double> column{c.data() + j * dim, dim};
    const auto host =
        fb::host_sigma(*alpha, *beta, ints.e_core, ints.h1, ints.eri, column, expected);
    ASSERT_TRUE(host.has_value()) << host.error().message;
    double norm = 0.0;
    double error = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      norm += expected[i] * expected[i];
      const double delta = got[j * dim + i] - expected[i];
      error += delta * delta;
    }
    EXPECT_LE(std::sqrt(error), 1e-12 * std::sqrt(norm)) << "column " << j;
  }

  fb::Buffers::device<double> diagonal(dim, res.device());
  const auto diag = sigma->diagonal(res, diagonal.data());
  ASSERT_TRUE(diag.has_value()) << diag.error().message;
  std::vector<double> host_diag(dim);
  const auto host = fb::host_diagonal(*alpha, *beta, ints.e_core, ints.h1, ints.eri, host_diag);
  ASSERT_TRUE(host.has_value()) << host.error().message;
  const std::vector<double> device_diag = to_host(res, diagonal);
  double worst = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    worst = std::max(worst, std::abs(device_diag[i] - host_diag[i]));
  }
  EXPECT_LE(worst, 1e-12);

  check_rdm1(res, *sigma, *alpha, *beta, seed + 1);
}

TEST(CiSigmaGpu, MatchesHostOnDirectSpin1Cases) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto doc =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "ci" / "direct_spin1.json");
  ASSERT_TRUE(doc.has_value()) << doc.error().message;
  ASSERT_FALSE((*doc)["cases"].empty());
  for (const auto &entry : (*doc)["cases"]) {
    SCOPED_TRACE(entry["name"].get<std::string>());
    const Integrals ints{entry["n_orbitals"].get<std::int32_t>(), entry["e_core"].get<double>(),
                         entry["h1"]["values"].get<std::vector<double>>(),
                         entry["eri"]["values"].get<std::vector<double>>()};
    check_against_host(res, entry["n_alpha"].get<std::int32_t>(),
                       entry["n_beta"].get<std::int32_t>(), ints, 3, 7);
  }
}

TEST(CiSigmaGpu, MatchesHostOnRandomIntegrals) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // (n, N_alpha, N_beta, k): CAS(8,8), CAS(10,10), and 11 orbitals over
  // (5, 4), whose 152,460 determinants each span two of apply's 127,100-row
  // chunks, in a block of two that reuses G; open shells, an empty spin, a
  // full one and a single orbital.
  const std::array<std::array<std::int32_t, 4>, 10> shapes{{
      {8, 4, 4, 2},
      {10, 5, 5, 3},
      {11, 5, 4, 2},
      {8, 5, 3, 2},
      {9, 3, 6, 1},
      {1, 1, 0, 1},
      {5, 1, 4, 2},
      {6, 0, 3, 2},
      {6, 6, 5, 2},
      {7, 4, 2, 4},
  }};
  for (const auto &[n, n_alpha, n_beta, k] : shapes) {
    SCOPED_TRACE(std::format("n = {}, N_alpha = {}, N_beta = {}, k = {}", n, n_alpha, n_beta, k));
    check_against_host(res, n_alpha, n_beta,
                       random_integrals(n, 300 + static_cast<std::uint64_t>(n)), k, 11);
  }
}

TEST(CiSigmaGpu, MatchesHostAtCas1212) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  fb::DeviceResources res{0};
  check_against_host(res, 6, 6, random_integrals(12, 412), 1, 13);
}

// 853,776 determinants over 78 pairs: rdm1's gather spans eight chunks, so the
// gemv's sum across them is exercised. Needs no integrals.
TEST(CiSigmaGpu, Rdm1MatchesHostAtCas1212) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto set = fb::StringSet::create(12, 6);
  ASSERT_TRUE(set.has_value());
  auto sigma = fb::CiSigma::create(res, *set, *set);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  check_rdm1(res, *sigma, *set, *set, 17);
}

TEST(CiSigmaGpu, Refuses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto four = fb::StringSet::create(4, 2);
  const auto five = fb::StringSet::create(5, 2);
  const auto none = fb::StringSet::create(0, 0);
  ASSERT_TRUE(four.has_value() && five.has_value() && none.has_value());
  EXPECT_FALSE(fb::CiSigma::create(res, *four, *five).has_value());
  EXPECT_FALSE(fb::CiSigma::create(res, *none, *none).has_value());

  auto sigma = fb::CiSigma::create(res, *four, *four);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  fb::Buffers::device<double> x(static_cast<std::size_t>(sigma->size()), res.device());
  fb::Buffers::device<double> y(static_cast<std::size_t>(sigma->size()), res.device());
  EXPECT_FALSE(sigma->apply(res, 1, x.data(), y.data()).has_value());
  EXPECT_FALSE(sigma->diagonal(res, y.data()).has_value());

  const Integrals ints = random_integrals(4, 5);
  EXPECT_FALSE(
      sigma->set_integrals(res, ints.e_core, ints.h1, std::span{ints.eri}.first(10)).has_value());
  EXPECT_FALSE(
      sigma->set_integrals(res, ints.e_core, std::span{ints.h1}.first(3), ints.eri).has_value());
  ASSERT_TRUE(sigma->set_integrals(res, ints.e_core, ints.h1, ints.eri).has_value());
  EXPECT_FALSE(sigma->apply(res, -1, x.data(), y.data()).has_value());
  EXPECT_TRUE(sigma->apply(res, 0, x.data(), y.data()).has_value());
}

} // namespace
