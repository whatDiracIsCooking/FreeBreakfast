// GPU cases for ci.rdm's transition RDMs, ci::Rdm::trans_rdm1 and
// ci::Rdm::trans_rdm12: against PySCF's trans_rdm1 (transposed: its [p, q] is
// <bra| q+ p |ket>, ci::Rdm's <bra| E_pq |ket>) and the symmetrized
// trans_rdm12 over the pairs of CI vectors stored in data/ci/trans_rdm.json,
// written by tools/emit/trans_rdm_reference.py, in both layouts; bra = ket
// against rdm12 at weight 1; the symmetrized RDMs of random vectors against
// the polarization (R(a + b) − R(a − b)) / 4 of host_rdm12 and of rdm12,
// over sectors that cover open shells, an empty and a full spin and
// N_α < N_β, and across chunks of D in both layouts; and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.ci;
import free_breakfast.test.ci_oracles;

namespace {

namespace fb = free_breakfast;

// The issue's, against PySCF and between the device's own routes, relative
// to the largest element where Γ's run past 1.
constexpr double kTol = 1e-12;
// The polarization subtracts two RDMs of order 1 for one of order 1/4.
constexpr double kPolarizationTol = 1e-11;

double max_abs_diff(std::span<const double> a, std::span<const double> b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

double scale_of(std::span<const double> a) {
  double worst = 1.0;
  for (const double x : a) {
    worst = std::max(worst, std::abs(x));
  }
  return worst;
}

fb::Buffers::device<double> on_device(fb::DeviceResources &res, std::span<const double> host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

std::vector<double> on_host(fb::DeviceResources &res, const fb::Buffers::device<double> &device) {
  std::vector<double> host(device.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The symmetric layout of one full-layout vector over m strings for both
// spins: the triangle a >= b, off-diagonal elements times sqrt(2).
std::vector<double> pack(std::span<const double> full, std::size_t m) {
  std::vector<double> out(m * (m + 1) / 2);
  for (std::size_t a = 0; a < m; ++a) {
    for (std::size_t b = 0; b <= a; ++b) {
      const double x = full[(a * m) + b];
      out[(a * (a + 1) / 2) + b] = a == b ? x : std::numbers::sqrt2 * x;
    }
  }
  return out;
}

bool swap_symmetric(std::span<const double> c, std::size_t m) {
  for (std::size_t a = 0; a < m; ++a) {
    for (std::size_t b = 0; b < a; ++b) {
      if (std::abs(c[(a * m) + b] - c[(b * m) + a]) > 1e-12) {
        return false;
      }
    }
  }
  return true;
}

std::vector<double> random_unit_vector(std::size_t dim, std::uint64_t seed) {
  std::mt19937_64 engine{seed};
  std::uniform_real_distribution<double> uniform{-1.0, 1.0};
  std::vector<double> c(dim);
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

struct Transition {
  std::vector<double> gamma_ij; // trans_rdm1
  std::vector<double> gamma;    // trans_rdm12's
  std::vector<double> gamma2;
};

// trans_rdm1 and trans_rdm12 of the host (@p bra, @p ket) in @p layout,
// through the device and back.
Transition device_transition(fb::DeviceResources &res, fb::ci::Rdm &rdm, std::size_t n,
                             std::span<const double> bra, std::span<const double> ket,
                             fb::ci::Layout layout) {
  const auto bra_dev = on_device(res, bra);
  const auto ket_dev = on_device(res, ket);
  fb::Buffers::device<double> gamma_ij(n * n, res.device());
  fb::Buffers::device<double> gamma(n * n, res.device());
  fb::Buffers::device<double> gamma2(n * n * n * n, res.device());
  const auto one = rdm.trans_rdm1(res, bra_dev.data(), ket_dev.data(), gamma_ij.data(), layout);
  EXPECT_TRUE(one.has_value()) << one.error().message;
  const auto two =
      rdm.trans_rdm12(res, bra_dev.data(), ket_dev.data(), gamma.data(), gamma2.data(), layout);
  EXPECT_TRUE(two.has_value()) << two.error().message;
  return {on_host(res, gamma_ij), on_host(res, gamma), on_host(res, gamma2)};
}

// rdm12 of the one host vector @p c at weight 1 in @p layout.
std::pair<std::vector<double>, std::vector<double>> device_rdm12(fb::DeviceResources &res,
                                                                 fb::ci::Rdm &rdm, std::size_t n,
                                                                 std::span<const double> c,
                                                                 fb::ci::Layout layout) {
  const auto c_dev = on_device(res, c);
  fb::Buffers::device<double> gamma(n * n, res.device());
  fb::Buffers::device<double> gamma2(n * n * n * n, res.device());
  const std::array weight{1.0};
  const auto got = rdm.rdm12(res, weight, c_dev.data(), gamma.data(), gamma2.data(), layout);
  EXPECT_TRUE(got.has_value()) << got.error().message;
  return {on_host(res, gamma), on_host(res, gamma2)};
}

// Γ_tuvw = Γ_vwtu and γ_tu = γ_ut, as for an RDM: the symmetrized pair's.
void expect_symmetric(std::span<const double> gamma, std::span<const double> gamma2,
                      std::size_t n) {
  double one = 0.0;
  double two = 0.0;
  for (std::size_t tu = 0; tu < n * n; ++tu) {
    const std::size_t t = tu / n;
    const std::size_t u = tu % n;
    one = std::max(one, std::abs(gamma[tu] - gamma[(u * n) + t]));
    for (std::size_t vw = 0; vw < n * n; ++vw) {
      two = std::max(two, std::abs(gamma2[(tu * n * n) + vw] - gamma2[(vw * n * n) + tu]));
    }
  }
  EXPECT_LE(one, kTol) << "symmetrized gamma not symmetric";
  EXPECT_LE(two, kTol * scale_of(gamma2)) << "Gamma_tuvw != Gamma_vwtu";
}

// The device's trans_rdm1 and trans_rdm12 of (bra, ket) against PySCF's, in
// @p layout (bra and ket already in it), @p pyscf_trans_rdm1 as PySCF returns
// it, the transpose of γ^IJ; and with bra and ket swapped,
// γ^JI = γ^IJᵀ and the symmetrized pair unchanged.
void check_pair(fb::DeviceResources &res, fb::ci::Rdm &rdm, std::size_t n,
                std::span<const double> bra, std::span<const double> ket,
                std::span<const double> pyscf_trans_rdm1, std::span<const double> rdm1,
                std::span<const double> rdm2, fb::ci::Layout layout) {
  const Transition got = device_transition(res, rdm, n, bra, ket, layout);
  const double scale = scale_of(rdm2);
  std::vector<double> want_ij(n * n);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      want_ij[(t * n) + u] = pyscf_trans_rdm1[(u * n) + t];
    }
  }
  EXPECT_LE(max_abs_diff(got.gamma_ij, want_ij), kTol) << "trans_rdm1 vs PySCF's transpose";
  EXPECT_LE(max_abs_diff(got.gamma, rdm1), kTol) << "trans_rdm12 gamma vs PySCF";
  EXPECT_LE(max_abs_diff(got.gamma2, rdm2), kTol * scale) << "trans_rdm12 Gamma vs PySCF";
  expect_symmetric(got.gamma, got.gamma2, n);

  // The symmetrized γ is trans_rdm1's symmetric part.
  std::vector<double> half(n * n);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      half[(t * n) + u] = 0.5 * (got.gamma_ij[(t * n) + u] + got.gamma_ij[(u * n) + t]);
    }
  }
  EXPECT_LE(max_abs_diff(got.gamma, half), kTol) << "gamma vs trans_rdm1's symmetric part";

  const Transition swapped = device_transition(res, rdm, n, ket, bra, layout);
  std::vector<double> transposed(n * n);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      transposed[(t * n) + u] = got.gamma_ij[(u * n) + t];
    }
  }
  EXPECT_LE(max_abs_diff(swapped.gamma_ij, transposed), kTol) << "gamma^JI != gamma^IJ^T";
  EXPECT_LE(max_abs_diff(swapped.gamma, got.gamma), kTol) << "trans_rdm12 gamma not bra <-> ket";
  EXPECT_LE(max_abs_diff(swapped.gamma2, got.gamma2), kTol * scale)
      << "trans_rdm12 Gamma not bra <-> ket";
}

// bra = ket = @p c: trans_rdm1 and trans_rdm12 against rdm12 at weight 1, to
// rounding.
void check_diagonal(fb::DeviceResources &res, fb::ci::Rdm &rdm, std::size_t n,
                    std::span<const double> c, fb::ci::Layout layout) {
  const Transition got = device_transition(res, rdm, n, c, c, layout);
  const auto [gamma, gamma2] = device_rdm12(res, rdm, n, c, layout);
  EXPECT_LE(max_abs_diff(got.gamma_ij, gamma), kTol) << "trans_rdm1(c, c) vs rdm12";
  EXPECT_LE(max_abs_diff(got.gamma, gamma), kTol) << "trans_rdm12(c, c) gamma vs rdm12";
  EXPECT_LE(max_abs_diff(got.gamma2, gamma2), kTol * scale_of(gamma2))
      << "trans_rdm12(c, c) Gamma vs rdm12";
}

TEST(CiTransRdmGpu, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  auto doc = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "ci" / "trans_rdm.json");
  ASSERT_TRUE(doc.has_value()) << doc.error().message;
  std::int32_t pairs = 0;
  std::int32_t packed = 0;
  std::int32_t triplets = 0;
  for (const auto &entry : (*doc)["cases"]) {
    SCOPED_TRACE(entry["name"].get<std::string>());
    const auto n = entry["n_orbitals"].get<std::int32_t>();
    const auto n_alpha = entry["n_alpha"].get<std::int32_t>();
    const auto n_beta = entry["n_beta"].get<std::int32_t>();
    const auto alpha = fb::ci::DeviceStringSet::create(res, n, n_alpha);
    const auto beta = fb::ci::DeviceStringSet::create(res, n, n_beta);
    ASSERT_TRUE(alpha.has_value() && beta.has_value());
    auto rdm = fb::ci::Rdm::create(res, *alpha, *beta);
    ASSERT_TRUE(rdm.has_value()) << rdm.error().message;
    const auto nn = static_cast<std::size_t>(n);
    const auto m = static_cast<std::size_t>(alpha->size());
    ASSERT_FALSE(entry["pairs"].empty());
    for (const auto &pair : entry["pairs"]) {
      const auto roots = pair["roots"].get<std::vector<std::int32_t>>();
      SCOPED_TRACE(std::format("bra root {}, ket root {}", roots[0], roots[1]));
      const auto s2 = pair["s2"].get<std::vector<double>>();
      const auto bra = pair["bra"]["values"].get<std::vector<double>>();
      const auto ket = pair["ket"]["values"].get<std::vector<double>>();
      const auto trans_rdm1 = pair["trans_rdm1"]["values"].get<std::vector<double>>();
      const auto rdm1 = pair["rdm1"]["values"].get<std::vector<double>>();
      const auto rdm2 = pair["rdm2"]["values"].get<std::vector<double>>();
      ASSERT_EQ(bra.size(), static_cast<std::size_t>(rdm->size()));
      triplets += std::abs(s2[0] - 2.0) < 1e-6 && std::abs(s2[1] - 2.0) < 1e-6 ? 1 : 0;

      // Orthogonal roots: tr γ^IJ = N ⟨I|J⟩ = 0.
      double trace = 0.0;
      for (std::size_t t = 0; t < nn; ++t) {
        trace += trans_rdm1[(t * nn) + t];
      }
      EXPECT_NEAR(trace, 0.0, 1e-10);

      {
        SCOPED_TRACE("full layout");
        check_pair(res, *rdm, nn, bra, ket, trans_rdm1, rdm1, rdm2, fb::ci::Layout::full);
        check_diagonal(res, *rdm, nn, bra, fb::ci::Layout::full);
      }
      // Two swap-symmetric vectors (even S over one set) pack.
      if (n_alpha == n_beta && swap_symmetric(bra, m) && swap_symmetric(ket, m)) {
        SCOPED_TRACE("symmetric layout");
        check_pair(res, *rdm, nn, pack(bra, m), pack(ket, m), trans_rdm1, rdm1, rdm2,
                   fb::ci::Layout::symmetric);
        check_diagonal(res, *rdm, nn, pack(bra, m), fb::ci::Layout::symmetric);
        ++packed;
      }
      ++pairs;
    }
  }
  EXPECT_GE(pairs, 8);
  EXPECT_GE(packed, 4);
  EXPECT_GE(triplets, 3);
}

// rdm12 at weight 1 of a ± b is R(a) + R(b) ± 2 T(a, b), so the symmetrized
// pair is T = (R(a + b) − R(a − b)) / 4: @p got against that of @p plus and
// @p minus, each a (γ, Γ).
void expect_polarization(const Transition &got, std::span<const double> plus,
                         std::span<const double> plus2, std::span<const double> minus,
                         std::span<const double> minus2) {
  std::vector<double> want(plus.size());
  std::vector<double> want2(plus2.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    want[i] = 0.25 * (plus[i] - minus[i]);
  }
  for (std::size_t i = 0; i < want2.size(); ++i) {
    want2[i] = 0.25 * (plus2[i] - minus2[i]);
  }
  EXPECT_LE(max_abs_diff(got.gamma, want), kPolarizationTol) << "gamma vs polarization";
  EXPECT_LE(max_abs_diff(got.gamma2, want2), kPolarizationTol * scale_of(plus2))
      << "Gamma vs polarization";
}

// The symmetrized pair of random unit vectors (a, b) against the
// polarization, from host_rdm12 (an independent route) and from rdm12; and
// bra = ket against rdm12. Over open shells, an empty and a full spin, one
// orbital and N_α < N_β, in the full layout.
TEST(CiTransRdmGpu, MatchesHostPolarization) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  struct Sector {
    std::int32_t n, n_alpha, n_beta;
  };
  const std::array weight{1.0};
  for (const Sector s : {Sector{6, 3, 3}, Sector{7, 4, 2}, Sector{8, 3, 5}, Sector{5, 0, 3},
                         Sector{5, 5, 2}, Sector{1, 1, 0}, Sector{2, 1, 1}}) {
    SCOPED_TRACE(std::format("n {} alpha {} beta {}", s.n, s.n_alpha, s.n_beta));
    const auto alpha = fb::ci::DeviceStringSet::create(res, s.n, s.n_alpha);
    const auto beta = fb::ci::DeviceStringSet::create(res, s.n, s.n_beta);
    ASSERT_TRUE(alpha.has_value() && beta.has_value());
    auto rdm = fb::ci::Rdm::create(res, *alpha, *beta);
    ASSERT_TRUE(rdm.has_value()) << rdm.error().message;
    const auto n = static_cast<std::size_t>(s.n);
    const auto dim = static_cast<std::size_t>(rdm->size());
    const std::vector<double> a = random_unit_vector(dim, dim * 7 + 1);
    const std::vector<double> b = random_unit_vector(dim, dim * 7 + 2);
    std::vector<double> sum(dim);
    std::vector<double> difference(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      sum[i] = a[i] + b[i];
      difference[i] = a[i] - b[i];
    }

    const Transition got = device_transition(res, *rdm, n, a, b, fb::ci::Layout::full);
    std::vector<double> plus(n * n);
    std::vector<double> plus2(n * n * n * n);
    std::vector<double> minus(n * n);
    std::vector<double> minus2(n * n * n * n);
    ASSERT_TRUE(fb::test::host_rdm12(*alpha, *beta, sum, weight, plus, plus2).has_value());
    ASSERT_TRUE(fb::test::host_rdm12(*alpha, *beta, difference, weight, minus, minus2).has_value());
    {
      SCOPED_TRACE("host_rdm12");
      expect_polarization(got, plus, plus2, minus, minus2);
    }
    {
      SCOPED_TRACE("rdm12");
      const auto [p1, p2] = device_rdm12(res, *rdm, n, sum, fb::ci::Layout::full);
      const auto [m1, m2] = device_rdm12(res, *rdm, n, difference, fb::ci::Layout::full);
      expect_polarization(got, p1, p2, m1, m2);
    }
    expect_symmetric(got.gamma, got.gamma2, n);
    check_diagonal(res, *rdm, n, a, fb::ci::Layout::full);
  }
}

// D over several chunks: 11 orbitals hold 34 663 rows a chunk in
// trans_rdm12 (two chunks of D share rdm12's 64 MiB) and 69 327 in rdm12, and
// (4, 4) has 108 900 determinants, (5, 5) 106 953 triangle rows. Against the
// polarization of rdm12 (which MatchesHostAcrossChunks in ci_rdm_gpu.cpp
// holds to host_rdm12), and bra = ket against rdm12, in each layout.
TEST(CiTransRdmGpu, MatchesPolarizationAcrossChunks) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  for (const auto [electrons, layout] :
       {std::pair{4, fb::ci::Layout::full}, std::pair{5, fb::ci::Layout::symmetric}}) {
    SCOPED_TRACE(std::format("N {} per spin", electrons));
    const auto set = fb::ci::DeviceStringSet::create(res, 11, electrons);
    ASSERT_TRUE(set.has_value());
    auto rdm = fb::ci::Rdm::create(res, *set, *set);
    ASSERT_TRUE(rdm.has_value()) << rdm.error().message;
    const std::size_t n = 11;
    const auto rows = static_cast<std::size_t>(rdm->size(layout));
    ASSERT_GT(rows, 3U * 34663U);
    const std::vector<double> a = random_unit_vector(rows, rows * 7 + 1);
    const std::vector<double> b = random_unit_vector(rows, rows * 7 + 2);
    std::vector<double> sum(rows);
    std::vector<double> difference(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      sum[i] = a[i] + b[i];
      difference[i] = a[i] - b[i];
    }
    const Transition got = device_transition(res, *rdm, n, a, b, layout);
    const auto [p1, p2] = device_rdm12(res, *rdm, n, sum, layout);
    const auto [m1, m2] = device_rdm12(res, *rdm, n, difference, layout);
    expect_polarization(got, p1, p2, m1, m2);
    expect_symmetric(got.gamma, got.gamma2, n);
    check_diagonal(res, *rdm, n, a, layout);
  }
}

TEST(CiTransRdmGpu, Refuses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // The symmetric layout wants one set for both spins: (4, 1) and (4, 3)
  // have the same number of strings, and are still two sets.
  const auto one = fb::ci::DeviceStringSet::create(res, 4, 1);
  const auto three = fb::ci::DeviceStringSet::create(res, 4, 3);
  ASSERT_TRUE(one.has_value() && three.has_value());
  auto mixed = fb::ci::Rdm::create(res, *one, *three);
  ASSERT_TRUE(mixed.has_value()) << mixed.error().message;
  constexpr double kSentinel = 42.0;
  const auto c = on_device(res, random_unit_vector(16, 3));
  auto gamma = on_device(res, std::vector<double>(16, kSentinel));
  auto gamma2 = on_device(res, std::vector<double>(256, kSentinel));
  const auto untouched = [&] {
    EXPECT_TRUE(std::ranges::all_of(on_host(res, gamma), [](double x) { return x == kSentinel; }));
    EXPECT_TRUE(std::ranges::all_of(on_host(res, gamma2), [](double x) { return x == kSentinel; }));
  };

  const auto refused_one =
      mixed->trans_rdm1(res, c.data(), c.data(), gamma.data(), fb::ci::Layout::symmetric);
  ASSERT_FALSE(refused_one.has_value());
  EXPECT_EQ(refused_one.error().kind, fb::ErrorKind::InvalidConfig) << refused_one.error().message;
  untouched();
  const auto refused_two = mixed->trans_rdm12(res, c.data(), c.data(), gamma.data(), gamma2.data(),
                                              fb::ci::Layout::symmetric);
  ASSERT_FALSE(refused_two.has_value());
  EXPECT_EQ(refused_two.error().kind, fb::ErrorKind::InvalidConfig) << refused_two.error().message;
  untouched();

  // The full layout over the same two sets is accepted.
  EXPECT_TRUE(mixed->trans_rdm1(res, c.data(), c.data(), gamma.data()).has_value());
  EXPECT_TRUE(mixed->trans_rdm12(res, c.data(), c.data(), gamma.data(), gamma2.data()).has_value());
  EXPECT_TRUE(res.sync().has_value());
}

} // namespace
