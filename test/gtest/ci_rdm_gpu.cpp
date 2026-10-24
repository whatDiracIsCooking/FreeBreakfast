// GPU cases for ci.rdm's CiRdm: the device γ and Γ against host_rdm12, and
// both against PySCF's make_rdm12 (and make_rdm12s summed over the spins) over
// the CI vectors stored in data/ci/direct_spin1.json, written by
// tools/emit_fci_reference.py; γ against rdm1 and CiSigma::rdm1; the RDM
// identities; and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.ci;

namespace {

namespace fb = free_breakfast;

// The issue's: the device against the host reference, and either against
// PySCF's, whose vectors are the ones stored.
constexpr double kHostTol = 1e-12;
constexpr double kPyscfTol = 1e-8;
// The identities are sums of up to n² elements of Γ.
constexpr double kIdentityTol = 1e-10;

double max_abs_diff(std::span<const double> a, std::span<const double> b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

double max_abs(std::span<const double> a) {
  double worst = 0.0;
  for (const double x : a) {
    worst = std::max(worst, std::abs(x));
  }
  return worst;
}

// Γ_tuvw = Γ_vwtu; Σ_v Γ_tuvv = (N − 1) γ_tu; Σ_tu Γ_ttuu = N (N − 1).
void expect_identities(std::span<const double> gamma, std::span<const double> gamma2, std::size_t n,
                       std::int32_t electrons) {
  const auto at = [&](std::size_t t, std::size_t u, std::size_t v, std::size_t w) {
    return gamma2[((t * n + u) * n + v) * n + w];
  };
  double swap = 0.0;
  double partial = 0.0;
  double full = 0.0;
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      double sum = 0.0;
      for (std::size_t v = 0; v < n; ++v) {
        sum += at(t, u, v, v);
        for (std::size_t w = 0; w < n; ++w) {
          swap = std::max(swap, std::abs(at(t, u, v, w) - at(v, w, t, u)));
        }
      }
      partial = std::max(partial, std::abs(sum - (electrons - 1) * gamma[t * n + u]));
      full += at(t, t, u, u);
    }
  }
  EXPECT_LE(swap, kIdentityTol) << "Gamma_tuvw != Gamma_vwtu";
  EXPECT_LE(partial, kIdentityTol) << "sum_v Gamma_tuvv != (N - 1) gamma_tu";
  EXPECT_NEAR(full, static_cast<double>(electrons) * (electrons - 1), kIdentityTol);
}

// γ and Γ of @p vectors at @p weights, on the device and the host, checked
// against each other, γ against Σ_r w_r rdm1(c_r), and the identities.
struct Rdm12 {
  std::vector<double> gamma;
  std::vector<double> gamma2;
};

// @p host on the device.
fb::Buffers::device<double> on_device(fb::DeviceResources &res, std::span<const double> host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // The source may be a temporary.
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

// @p device on the host.
std::vector<double> on_host(fb::DeviceResources &res, const fb::Buffers::device<double> &device) {
  std::vector<double> host(device.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// CiRdm::rdm12 of the host @p vectors, through the device and back.
Rdm12 device_rdm12(fb::DeviceResources &res, const fb::StringSet &alpha, const fb::StringSet &beta,
                   std::span<const double> vectors, std::span<const double> weights) {
  const auto n = static_cast<std::size_t>(alpha.n_orbitals());
  auto rdm = fb::CiRdm::create(res, alpha, beta);
  EXPECT_TRUE(rdm.has_value()) << rdm.error().message;
  if (!rdm) {
    return {};
  }
  const auto c = on_device(res, vectors);
  fb::Buffers::device<double> gamma(n * n, res.device());
  fb::Buffers::device<double> gamma2(n * n * n * n, res.device());
  const auto got = rdm->rdm12(res, weights, c.data(), gamma.data(), gamma2.data());
  EXPECT_TRUE(got.has_value()) << got.error().message;
  return Rdm12{on_host(res, gamma), on_host(res, gamma2)};
}

std::pair<Rdm12, Rdm12> both_rdm12(fb::DeviceResources &res, const fb::StringSet &alpha,
                                   const fb::StringSet &beta, std::span<const double> vectors,
                                   std::span<const double> weights) {
  const auto n = static_cast<std::size_t>(alpha.n_orbitals());
  const std::size_t dim =
      static_cast<std::size_t>(alpha.size()) * static_cast<std::size_t>(beta.size());
  Rdm12 device = device_rdm12(res, alpha, beta, vectors, weights);
  Rdm12 host{std::vector<double>(n * n), std::vector<double>(n * n * n * n)};
  const auto want = fb::host_rdm12(alpha, beta, vectors, weights, host.gamma, host.gamma2);
  EXPECT_TRUE(want.has_value()) << want.error().message;

  const double scale = std::max(1.0, max_abs(host.gamma2));
  EXPECT_LE(max_abs_diff(device.gamma2, host.gamma2), kHostTol * scale) << "device Gamma vs host";
  EXPECT_LE(max_abs_diff(device.gamma, host.gamma), kHostTol) << "device gamma vs host";

  std::vector<double> averaged(n * n, 0.0);
  std::vector<double> one(n * n);
  for (std::size_t r = 0; r < weights.size(); ++r) {
    EXPECT_TRUE(fb::rdm1(alpha, beta, vectors.subspan(r * dim, dim), one).has_value());
    for (std::size_t i = 0; i < one.size(); ++i) {
      averaged[i] += weights[r] * one[i];
    }
  }
  EXPECT_LE(max_abs_diff(device.gamma, averaged), kHostTol) << "device gamma vs rdm1";
  EXPECT_LE(max_abs_diff(host.gamma, averaged), kHostTol) << "host gamma vs rdm1";

  expect_identities(device.gamma, device.gamma2, n, alpha.n_electrons() + beta.n_electrons());
  return {std::move(device), std::move(host)};
}

// rdm12's γ of one state against CiSigma::rdm1, the device 1-RDM a
// calculation uses, over the same vector.
void expect_matches_ci_sigma_rdm1(fb::DeviceResources &res, const fb::StringSet &alpha,
                                  const fb::StringSet &beta, std::span<const double> c,
                                  std::span<const double> gamma) {
  auto sigma = fb::CiSigma::create(res, alpha, beta);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  fb::Buffers::device<double> c_dev(c.size(), res.device());
  fb::Buffers::device<double> gamma_dev(gamma.size(), res.device());
  ASSERT_EQ(wwr::wwrMemcpyAsync(c_dev.data(), c.data(), c.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(sigma->rdm1(res, c_dev.data(), gamma_dev.data()).has_value());
  std::vector<double> got(gamma.size());
  ASSERT_EQ(wwr::wwrMemcpyAsync(got.data(), gamma_dev.data(), got.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());
  EXPECT_LE(max_abs_diff(gamma, got), kHostTol) << "rdm12 gamma vs CiSigma::rdm1";
}

// Random unit vectors, @p roots of them, over the sector (n, N_α, N_β).
std::vector<double> random_unit_vectors(std::size_t dim, std::size_t roots) {
  std::mt19937_64 engine{dim * 31 + roots};
  std::uniform_real_distribution<double> uniform{-1.0, 1.0};
  std::vector<double> vectors(dim * roots);
  for (std::size_t r = 0; r < roots; ++r) {
    const std::span<double> c{vectors.data() + r * dim, dim};
    double norm = 0.0;
    for (double &x : c) {
      x = uniform(engine);
      norm += x * x;
    }
    for (double &x : c) {
      x /= std::sqrt(norm);
    }
  }
  return vectors;
}

TEST(CiRdm12Gpu, MatchesDirectSpin1) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  auto doc = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "ci" / "direct_spin1.json");
  ASSERT_TRUE(doc.has_value()) << doc.error().message;
  const fb::json &cases = (*doc)["cases"];
  ASSERT_FALSE(cases.empty());
  std::int32_t averaged = 0;
  for (const auto &entry : cases) {
    SCOPED_TRACE(entry["name"].get<std::string>());
    const auto n = entry["n_orbitals"].get<std::int32_t>();
    const auto alpha = fb::StringSet::create(n, entry["n_alpha"].get<std::int32_t>());
    const auto beta = fb::StringSet::create(n, entry["n_beta"].get<std::int32_t>());
    ASSERT_TRUE(alpha.has_value() && beta.has_value());
    ASSERT_FALSE(entry["rdm12"].empty());
    for (const auto &average : entry["rdm12"]) {
      const auto weights = average["weights"].get<std::vector<double>>();
      SCOPED_TRACE(std::format("{} roots", weights.size()));
      averaged += weights.size() > 1 ? 1 : 0;
      const auto vectors = average["vectors"]["values"].get<std::vector<double>>();
      const auto [device, host] = both_rdm12(res, *alpha, *beta, vectors, weights);

      const auto rdm1 = average["rdm1"]["values"].get<std::vector<double>>();
      const auto rdm2 = average["rdm2"]["values"].get<std::vector<double>>();
      const auto rdm2s = average["rdm2_from_rdm12s"]["values"].get<std::vector<double>>();
      if (weights.size() == 1) {
        expect_matches_ci_sigma_rdm1(res, *alpha, *beta, vectors, device.gamma);
      }
      EXPECT_LE(max_abs_diff(device.gamma, rdm1), kPyscfTol);
      EXPECT_LE(max_abs_diff(host.gamma, rdm1), kPyscfTol);
      EXPECT_LE(max_abs_diff(device.gamma2, rdm2), kPyscfTol) << "device vs make_rdm12";
      EXPECT_LE(max_abs_diff(host.gamma2, rdm2), kPyscfTol) << "host vs make_rdm12";
      EXPECT_LE(max_abs_diff(device.gamma2, rdm2s), kPyscfTol) << "device vs make_rdm12s";
      EXPECT_LE(max_abs_diff(host.gamma2, rdm2s), kPyscfTol) << "host vs make_rdm12s";
    }
  }
  EXPECT_GT(averaged, 0) << "no weighted case of more than one root";
}

// Random unit vectors at three weights, over sectors that cover open and
// closed shells, an empty spin, a full one, a single orbital and N_α < N_β.
TEST(CiRdm12Gpu, MatchesHostOnRandomSectors) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  struct Sector {
    std::int32_t n, n_alpha, n_beta;
  };
  const std::array<double, 3> weights{0.5, 0.3, 0.2};
  for (const Sector s : {Sector{6, 3, 3}, Sector{7, 4, 2}, Sector{8, 3, 5}, Sector{5, 0, 3},
                         Sector{5, 5, 2}, Sector{1, 1, 0}, Sector{2, 1, 1}}) {
    SCOPED_TRACE(std::format("n {} alpha {} beta {}", s.n, s.n_alpha, s.n_beta));
    const auto alpha = fb::StringSet::create(s.n, s.n_alpha);
    const auto beta = fb::StringSet::create(s.n, s.n_beta);
    ASSERT_TRUE(alpha.has_value() && beta.has_value());
    const std::size_t dim =
        static_cast<std::size_t>(alpha->size()) * static_cast<std::size_t>(beta->size());
    const std::vector<double> vectors = random_unit_vectors(dim, weights.size());
    both_rdm12(res, *alpha, *beta, vectors, weights);
  }
}

// A sector big enough that D is gathered in two chunks of determinants: 11
// orbitals hold 69 327 determinants a chunk, and (4, 4) has 108 900.
TEST(CiRdm12Gpu, MatchesHostAcrossChunks) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  fb::DeviceResources res{0};
  const auto set = fb::StringSet::create(11, 4);
  ASSERT_TRUE(set.has_value());
  const std::size_t dim =
      static_cast<std::size_t>(set->size()) * static_cast<std::size_t>(set->size());
  ASSERT_EQ(dim, 108900U);
  const std::array<double, 1> weights{1.0};
  both_rdm12(res, *set, *set, random_unit_vectors(dim, 1), weights);
}

TEST(CiRdm12Gpu, Refuses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto four = fb::StringSet::create(4, 2);
  const auto five = fb::StringSet::create(5, 2);
  const auto none = fb::StringSet::create(0, 0);
  ASSERT_TRUE(four.has_value() && five.has_value() && none.has_value());
  const std::size_t dim =
      static_cast<std::size_t>(four->size()) * static_cast<std::size_t>(four->size());
  const std::vector<double> two_vectors = random_unit_vectors(dim, 2);
  constexpr double kSentinel = 42.0;
  const std::size_t g1 = 16;
  const std::size_t g2 = 256;

  // CiRdm refuses sets it cannot gather over.
  EXPECT_FALSE(fb::CiRdm::create(res, *four, *five).has_value());
  EXPECT_FALSE(fb::CiRdm::create(res, *none, *none).has_value());

  auto rdm = fb::CiRdm::create(res, *four, *four);
  ASSERT_TRUE(rdm.has_value()) << rdm.error().message;
  ASSERT_EQ(static_cast<std::size_t>(rdm->size()), dim);
  const auto c = on_device(res, two_vectors);

  // Both refuse weights that do not make an average, and leave γ and Γ as
  // they were; host_rdm12 also refuses the shapes it is given.
  const auto refused = [&](std::span<const double> vectors, std::span<const double> weights,
                           std::size_t gamma_size, std::size_t gamma2_size, bool on_device_too) {
    std::vector<double> gamma(gamma_size, kSentinel);
    std::vector<double> gamma2(gamma2_size, kSentinel);
    const auto got = fb::host_rdm12(*four, *four, vectors, weights, gamma, gamma2);
    ASSERT_FALSE(got.has_value());
    EXPECT_EQ(got.error().kind, fb::ErrorKind::InvalidConfig) << got.error().message;
    EXPECT_TRUE(std::ranges::all_of(gamma, [](double x) { return x == kSentinel; }));
    EXPECT_TRUE(std::ranges::all_of(gamma2, [](double x) { return x == kSentinel; }));
    if (!on_device_too) {
      return;
    }
    auto gamma_dev = on_device(res, std::vector<double>(g1, kSentinel));
    auto gamma2_dev = on_device(res, std::vector<double>(g2, kSentinel));
    const auto device = rdm->rdm12(res, weights, c.data(), gamma_dev.data(), gamma2_dev.data());
    ASSERT_FALSE(device.has_value());
    EXPECT_EQ(device.error().kind, fb::ErrorKind::InvalidConfig) << device.error().message;
    EXPECT_TRUE(
        std::ranges::all_of(on_host(res, gamma_dev), [](double x) { return x == kSentinel; }));
    EXPECT_TRUE(
        std::ranges::all_of(on_host(res, gamma2_dev), [](double x) { return x == kSentinel; }));
  };
  const std::span<const double> one_vector = std::span{two_vectors}.first(dim);
  {
    SCOPED_TRACE("two weights for one vector");
    refused(one_vector, std::array{0.5, 0.5}, g1, g2, false);
  }
  {
    SCOPED_TRACE("one weight for two vectors");
    refused(two_vectors, std::array{1.0}, g1, g2, false);
  }
  {
    SCOPED_TRACE("no weights");
    refused(std::span<const double>{}, std::span<const double>{}, g1, g2, true);
  }
  {
    SCOPED_TRACE("a negative weight");
    refused(two_vectors, std::array{1.5, -0.5}, g1, g2, true);
  }
  {
    SCOPED_TRACE("a NaN weight");
    refused(two_vectors, std::array{1.0, std::numeric_limits<double>::quiet_NaN()}, g1, g2, true);
  }
  {
    SCOPED_TRACE("weights summing past 1 + 1e-12");
    refused(two_vectors, std::array{0.7, 0.3 + 1e-11}, g1, g2, true);
  }
  {
    SCOPED_TRACE("a short vector");
    refused(one_vector.first(dim - 1), std::array{1.0}, g1, g2, false);
  }
  {
    SCOPED_TRACE("a short gamma");
    refused(one_vector, std::array{1.0}, g1 - 1, g2, false);
  }
  {
    SCOPED_TRACE("a short Gamma");
    refused(one_vector, std::array{1.0}, g1, g2 - 1, false);
  }
  {
    SCOPED_TRACE("sets over different orbital counts");
    std::vector<double> gamma(g1, kSentinel);
    std::vector<double> gamma2(g2, kSentinel);
    EXPECT_FALSE(fb::host_rdm12(*four, *five,
                                std::vector<double>(static_cast<std::size_t>(four->size()) *
                                                    static_cast<std::size_t>(five->size())),
                                std::array{1.0}, gamma, gamma2)
                     .has_value());
  }

  // Within 1e-12 of 1 is accepted.
  std::vector<double> gamma(g1);
  std::vector<double> gamma2(g2);
  const std::array weights{0.7, 0.3 + 1e-13};
  fb::Buffers::device<double> gamma_dev(g1, res.device());
  fb::Buffers::device<double> gamma2_dev(g2, res.device());
  EXPECT_TRUE(rdm->rdm12(res, weights, c.data(), gamma_dev.data(), gamma2_dev.data()).has_value());
  EXPECT_TRUE(fb::host_rdm12(*four, *four, two_vectors, weights, gamma, gamma2).has_value());
  EXPECT_TRUE(res.sync().has_value());
}

} // namespace
