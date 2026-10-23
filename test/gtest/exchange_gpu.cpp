// GPU cases for integrals.eri.integral_direct.exchange: the Cartesian exchange
// matrix K of H2O in STO-3G (s, p; O's p shell precedes the H s shells, so
// (p, s) and (s, p) are both reached) and cc-pVDZ (s, p, d, so every k class
// through (dd|dd)) against libcint's K of the same density, recorded by PySCF
// (test/gtest/data/exchange/, copied from FelizJueves's fixtures by
// tools/extract_integral_golden.py), unscreened (tau = 0) and screened at the
// default (1e-13) and at 1e-14, all to one bar; the H10 chain's block factors,
// as the build makes them, a pinned share of them under max|D|; the symmetric
// sweep against the full one, and on a non-symmetric density, that it mirrors
// rather than computes the a > b blocks; that a tau no bound reaches screens
// everything; plus, as coulomb_gpu.cpp, the zeroing and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;

namespace {

namespace fb = free_breakfast;

// FelizJueves holds its K to libcint at 1e-12 of scale (test/oracle/
// tolerance.py); the same bar here, against max|K| as coulomb_gpu.cpp. The
// recorded density is a Hilbert matrix, every element positive, so max|K| is
// not a scale cancellation has shrunk.
constexpr double kRelTol = 1e-12;

struct Golden {
  fb::BasisSet basis;
  std::vector<double> density; // nao x nao, row-major; each is symmetric
  std::vector<double> vk;
};

Golden read_golden(const std::string &stem) {
  const auto doc =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "exchange" / (stem + ".json"));
  EXPECT_TRUE(doc) << stem;
  if (!doc) {
    return {};
  }
  std::vector<fb::Shell> shells;
  for (const auto &sh : (*doc)["shells"]) {
    const auto c = sh["centre"].get<std::vector<double>>();
    shells.emplace_back(sh["l"].get<std::int32_t>(), sh["exponents"].get<std::vector<double>>(),
                        sh["coefficients"].get<std::vector<double>>(),
                        fb::Vec<double, 3>{c[0], c[1], c[2]});
  }
  const auto &arrays = (*doc)["arrays"];
  Golden g{fb::BasisSet(std::move(shells)), arrays["density"]["values"].get<std::vector<double>>(),
           arrays["vk"]["values"].get<std::vector<double>>()};
  EXPECT_EQ(g.basis.nao_cart(), (*doc)["nao"].get<std::uint32_t>()) << stem;
  return g;
}

std::vector<double> download(fb::DeviceResources &res, const fb::CartesianMatrixAO &m) {
  std::vector<double> host(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), m.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// A matrix onto the device, element m * n + q of host to the device's; for a
// symmetric one, row- and column-major are one layout.
void upload(fb::DeviceResources &res, const std::vector<double> &host, fb::CartesianMatrixAO &m) {
  ASSERT_EQ(host.size(), m.num_elements());
  ASSERT_EQ(wwr::wwrMemcpyAsync(m.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
  ASSERT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

// The ordered classes of basis, by way of mmd's.
fb::Result<std::vector<fb::OrderedPairClass>> ordered(fb::DeviceResources &res,
                                                      const fb::BasisSet &basis) {
  const auto mmd = fb::primitive_pair_classes(res, basis);
  if (!mmd) {
    return std::unexpected(mmd.error());
  }
  return fb::ordered_pair_classes(res, basis, *mmd);
}

// K of the density d (g's own unless given) over g's basis at tau, by the
// symmetric sweep or the full one, downloaded.
std::vector<double> golden_k(const Golden &g, const double tau, const bool symmetric = true,
                             const std::vector<double> *d = nullptr) {
  fb::DeviceResources res{0};
  const auto classes = ordered(res, g.basis);
  EXPECT_TRUE(classes);
  if (!classes) {
    return {};
  }
  fb::CartesianMatrixAO density(res, g.basis);
  upload(res, d != nullptr ? *d : g.density, density);
  fb::CartesianMatrixAO k(res, g.basis);
  const fb::Status status = fb::exchange(res, *classes, density, k, tau, symmetric);
  EXPECT_TRUE(status) << (status ? "" : status.error().message);
  return download(res, k);
}

double max_abs(const std::vector<double> &m) {
  double scale = 0.0;
  for (const double v : m) {
    scale = std::max(scale, std::abs(v));
  }
  return scale;
}

class ExchangeGolden : public ::testing::TestWithParam<const char *> {};

// At tau = 0 nothing is screened. Screened, every skipped term is under tau, but
// K_ab sums many of them, and their number grows with the record pairs covering
// the element: with the dressed bound, K at tau = 1e-10 misses libcint by up to
// 6.9e-10 of max|K| (H2O/cc-pVDZ; FelizJueves's own screened host K misses it
// by the same 3.2e-9). So K's default is 1e-13, the largest round tau at which
// all three molecules meet 1e-12 max|K| (the chain, the tightest, at 7.5e-13 on
// the RX 9060 XT), and every run here is held to that one bar: tau = 0, the
// default, and 1e-14. Each run's worst error, and how far the screen moved K,
// is recorded.
TEST_P(ExchangeGolden, MatchesLibcint) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.vk.empty());
  const double scale = max_abs(g.vk);
  const std::size_t n = g.basis.nao_cart();
  const std::vector<double> unscreened = golden_k(g, 0.0);
  ASSERT_EQ(unscreened.size(), g.vk.size());
  for (const double tau : {0.0, 1e-13, 1e-14}) {
    const std::vector<double> got = tau == 0.0 ? unscreened : golden_k(g, tau);
    ASSERT_EQ(got.size(), g.vk.size()) << "tau " << tau;
    double worst = 0.0;
    double moved = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t e = i * n + j;
        const double err = std::abs(got[e] - g.vk[e]);
        worst = std::max(worst, err);
        moved = std::max(moved, std::abs(got[e] - unscreened[e]));
        EXPECT_LE(err, kRelTol * scale) << "tau " << tau << ", K(" << i << ", " << j << ")";
        // An a < b block is mirrored, but K_ab and K_ba still collect their
        // adds in the atomics' order, and an a == b block is computed whole:
        // symmetric to rounding, not bitwise.
        EXPECT_LE(std::abs(got[e] - got[j * n + i]), kRelTol * scale) << "tau " << tau;
      }
    }
    std::cout << "[ libcint  ] " << GetParam() << " tau " << tau << ": worst " << worst << " ("
              << worst / scale << " of max|K|), the screen moved K by " << moved << "\n";
    RecordProperty(std::format("tau_{}_worst_abs_error", tau), std::format("{:.3e}", worst));
    RecordProperty(std::format("tau_{}_worst_rel_error", tau),
                   std::format("{:.3e}", worst / scale));
    RecordProperty(std::format("tau_{}_screen_max_abs_change", tau), std::format("{:.3e}", moved));
  }
}

// The symmetric sweep against the full one, unscreened and at the default, to
// the same bar; and the share of the full sweep's (bra record, ket record)
// pairs the symmetric one launches, read off the classes' groups as the build
// cuts them, recorded.
TEST_P(ExchangeGolden, SymmetricSweepMatchesTheFullOne) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.vk.empty());
  const double scale = max_abs(g.vk);
  for (const double tau : {0.0, 1e-13}) {
    const std::vector<double> halved = golden_k(g, tau, true);
    const std::vector<double> full = golden_k(g, tau, false);
    ASSERT_EQ(halved.size(), full.size());
    double worst = 0.0;
    for (std::size_t e = 0; e < full.size(); ++e) {
      worst = std::max(worst, std::abs(halved[e] - full[e]));
    }
    EXPECT_LE(worst, kRelTol * scale) << "tau " << tau;
    RecordProperty(std::format("tau_{}_worst_abs_difference", tau), std::format("{:.3e}", worst));
  }

  fb::DeviceResources res{0};
  const auto classes = ordered(res, g.basis);
  ASSERT_TRUE(classes);
  std::int64_t all = 0;
  std::int64_t cut = 0;
  for (const fb::OrderedPairClass &bra : *classes) {
    for (const fb::OrderedPairClass &ket : *classes) {
      for (const fb::OrderedPairGroup &b : ket.groups) {
        std::int32_t n_bra = 0;
        for (const fb::OrderedPairGroup &a : bra.groups) {
          if (a.shell <= b.shell) {
            n_bra += a.count;
          }
        }
        all += static_cast<std::int64_t>(bra.n_rec) * b.count;
        cut += static_cast<std::int64_t>(n_bra) * b.count;
      }
    }
  }
  EXPECT_LT(cut, all);
  RecordProperty("launched_share_of_full", std::format("{:.3f}", static_cast<double>(cut) / all));
}

INSTANTIATE_TEST_SUITE_P(H2O, ExchangeGolden, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));
// Five H2 4 bohr apart in cc-pVDZ, with a converged RHF density.
INSTANTIATE_TEST_SUITE_P(Chain, ExchangeGolden, ::testing::Values("h10_chain_ccpvdz"));

// The H10 chain's RHF density decays, and with the dressed record bound a
// group's reach stops well short of the 23-bohr chain, so the density factor
// fires: a share of the block factors exchange makes (every block, the full
// sweep: 30 shells squared times the four (lc, ld) of s and p) falls below
// max|D|, and none is above it. The counts are recorded and pinned (measured
// on the RX 9060 XT): 48 of the 3,600 at the default, 1e-13; 144 at 1e-10, the
// count FelizJueves's own density_decay gives over its host record bounds; and
// 242 at 1e-6. The share shrinks with tau, since a smaller tau lengthens every
// reach.
// A count is a sum of comparisons against the reach's cutoff, so a backend
// whose exp or sqrt rounds differently could move it by a block at a boundary;
// the nonzero share is the property, the pin a regression check.
TEST(Exchange, TheChainsBlockFactorsFallBelowTheDensityMaximum) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden("h10_chain_ccpvdz");
  ASSERT_FALSE(g.vk.empty());
  fb::DeviceResources res{0};
  const auto classes = ordered(res, g.basis);
  ASSERT_TRUE(classes);
  fb::CartesianMatrixAO density(res, g.basis);
  upload(res, g.density, density);
  const double dmax = max_abs(g.density);
  for (const auto &[tau, pinned] : {std::pair{1e-13, std::int64_t{48}},
                                    {1e-10, std::int64_t{144}},
                                    {1e-6, std::int64_t{242}}}) {
    const auto factors = fb::exchange_block_factors(res, *classes, density, tau, false);
    ASSERT_TRUE(factors) << factors.error().message;
    ASSERT_EQ(factors->size(), std::size_t{3600});
    std::int64_t below = 0;
    double least = dmax;
    for (const double f : *factors) {
      ASSERT_LE(f, dmax) << "tau " << tau;
      ASSERT_GT(f, 0.0) << "tau " << tau;
      below += f < dmax ? 1 : 0;
      least = std::min(least, f);
    }
    std::cout << "[ factors  ] tau " << tau << ": " << below << " of " << factors->size()
              << " below max|D| " << dmax << ", least " << least << "\n";
    EXPECT_GT(below, 0) << "tau " << tau;
    EXPECT_EQ(below, pinned) << "tau " << tau;
    RecordProperty(std::format("tau_{}_below_max_d", tau), std::format("{}", below));
  }
}

// A non-symmetric density tells the two sweeps apart: the full one computes
// every block directly, so its K is not symmetric, while the symmetric one
// computes the a <= b blocks only and mirrors the a < b. So the symmetric K
// matches the full one wherever shell(i) <= shell(j), and elsewhere its
// transpose, not itself: the a > b blocks are never computed.
TEST(Exchange, TheSymmetricSweepMirrorsTheUpperBlocks) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden("h2o_ccpvdz");
  ASSERT_FALSE(g.vk.empty());
  const std::size_t n = g.basis.nao_cart();
  std::vector<double> skewed = g.density;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      skewed[i * n + j] += 0.1 * (static_cast<double>(i) - static_cast<double>(j)) / n;
    }
  }
  const std::vector<double> halved = golden_k(g, 0.0, true, &skewed);
  const std::vector<double> full = golden_k(g, 0.0, false, &skewed);
  ASSERT_EQ(halved.size(), n * n);
  ASSERT_EQ(full.size(), n * n);
  const double scale = max_abs(full);
  const auto &aos = g.basis.aos_cart();
  double asymmetry = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      asymmetry = std::max(asymmetry, std::abs(full[i * n + j] - full[j * n + i]));
      const bool upper = aos[i].shell_index <= aos[j].shell_index;
      const double want = upper ? full[i * n + j] : full[j * n + i];
      EXPECT_LE(std::abs(halved[i * n + j] - want), kRelTol * scale)
          << "K(" << i << ", " << j << ")";
    }
  }
  // Else the case could not tell a mirrored block from a computed one.
  EXPECT_GT(asymmetry, 1e-6 * scale);
}

TEST(Exchange, ZeroesTheMatrixFirst) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::CartesianMatrixAO density(res, basis);
  fb::CartesianMatrixAO k(res, basis);
  const double stale = 42.0;
  ASSERT_EQ(wwr::wwrMemcpyAsync(k.data(), &stale, sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  // No classes: the only work left is the zeroing.
  ASSERT_TRUE(fb::exchange(res, {}, density, k));
  EXPECT_EQ(download(res, k)[0], 0.0);
}

// A tau above every Ip * Iq * max|D| breaks every lane at its first ket, so K
// is exactly zero: the screen reaches the kernel and fires.
TEST(Exchange, ATauAboveEveryBoundScreensEverything) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden("h2o_sto3g");
  ASSERT_FALSE(g.vk.empty());
  for (const double v : golden_k(g, 1e300)) {
    ASSERT_EQ(v, 0.0);
  }
}

TEST(Exchange, RefusesANegativeOrNaNTau) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const auto classes = ordered(res, basis);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, basis);
  fb::CartesianMatrixAO k(res, basis);
  for (const double tau : {-1e-10, std::numeric_limits<double>::quiet_NaN()}) {
    const fb::Status status = fb::exchange(res, *classes, density, k, tau);
    ASSERT_FALSE(status) << tau;
    EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig) << tau;
  }
}

TEST(Exchange, RefusesADensityOfAnotherOrder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet one({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::BasisSet two({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0}),
                          fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 1.0})});
  const auto classes = ordered(res, two);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, one);
  fb::CartesianMatrixAO k(res, two);
  const fb::Status status = fb::exchange(res, *classes, density, k);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

// Classes of a larger basis than K's: a shell past K's last AO is refused
// before anything is launched, not written out of bounds.
TEST(Exchange, RefusesClassesOfALargerBasis) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet one({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::BasisSet two({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0}),
                          fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 1.0})});
  const auto classes = ordered(res, two);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, one);
  fb::CartesianMatrixAO k(res, one);
  const fb::Status status = fb::exchange(res, *classes, density, k);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(Exchange, RefusesAClassAboveD) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(3, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const auto classes = ordered(res, basis);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, basis);
  fb::CartesianMatrixAO k(res, basis);
  const fb::Status status = fb::exchange(res, *classes, density, k);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
