// GPU cases for integrals.eri.integral_direct.exchange: the Cartesian exchange
// matrix K of H2O in STO-3G (s, p; O's p shell precedes the H s shells, so
// (p, s) and (s, p) are both reached) and cc-pVDZ (s, p, d, so every k class
// through (dd|dd)) against libcint's K of the same density, recorded by PySCF
// (test/gtest/data/exchange/, copied from FelizJueves's fixtures by
// tools/extract_integral_golden.py), unscreened (tau = 0) and screened at the
// default (1e-13) and at 1e-14, all to one bar; the H10 chain's block factors,
// as the build makes them, a pinned share of them under max|D|; a
// non-symmetric density's K as its symmetric part's sweep plus its
// antisymmetric part's, against libcint, and an antisymmetric density's K
// antisymmetric; that a tau no bound reaches screens everything; plus, as
// coulomb_gpu.cpp, the zeroing and the refusals, a subset or a reordering of
// the classes among them.

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
// sweep of the given symmetry, downloaded.
std::vector<double> golden_k(const Golden &g, const double tau,
                             const fb::DensitySymmetry symmetry = fb::DensitySymmetry::Symmetric,
                             const std::vector<double> *d = nullptr) {
  fb::DeviceResources res{0};
  auto classes = ordered(res, g.basis);
  EXPECT_TRUE(classes);
  if (!classes) {
    return {};
  }
  fb::CartesianMatrixAO density(res, g.basis);
  upload(res, d != nullptr ? *d : g.density, density);
  fb::CartesianMatrixAO k(res, g.basis);
  const fb::Status status = fb::exchange(res, *classes, density, k, tau, symmetry);
  EXPECT_TRUE(status) << (status ? "" : status.error().message);
  return download(res, k);
}

// A matrix's symmetric part (sign 1) or antisymmetric part (sign -1).
std::vector<double> part(const std::vector<double> &m, const std::size_t n, const double sign) {
  std::vector<double> out(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      out[i * n + j] = 0.5 * (m[i * n + j] + sign * m[j * n + i]);
    }
  }
  return out;
}

// A non-symmetric density: g's recorded one plus a smooth antisymmetric
// matrix, so its symmetric part is the recorded density, whose K libcint gave,
// and its antisymmetric part is not zero.
std::vector<double> skewed(const Golden &g) {
  const std::size_t n = g.basis.nao_cart();
  std::vector<double> d = g.density;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      d[i * n + j] += 0.1 * (static_cast<double>(i) - static_cast<double>(j)) / n;
    }
  }
  return d;
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

// A non-symmetric density D as two sweeps, its symmetric part S's
// (Symmetric) plus its antisymmetric part A's (Antisymmetric), unscreened and
// at the default, to the same bar. S is the recorded density, so its K is held
// to libcint's vk; libcint gave no K of a non-symmetric density, so the sum is
// held to D's K made of computed blocks only: by K[D]^T = K[D^T], its a <= b
// blocks are the Symmetric sweep of D's own and its a > b blocks the
// transposes of the a < b blocks of D^T's, none of them mirrored. A wrong sign
// in the antisymmetric mirror is then an error of 2 |K[A]| in every a > b
// block. And K[A] is antisymmetric to rounding.
TEST_P(ExchangeGolden, ANonSymmetricDensityIsTwoSweeps) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.vk.empty());
  const std::size_t n = g.basis.nao_cart();
  const std::vector<double> d = skewed(g);
  const std::vector<double> sym = part(d, n, 1.0);
  const std::vector<double> anti = part(d, n, -1.0);
  std::vector<double> d_t(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      d_t[j * n + i] = d[i * n + j];
    }
  }
  const auto &aos = g.basis.aos_cart();
  const double scale = max_abs(g.vk);
  for (const double tau : {0.0, 1e-13}) {
    const std::vector<double> k_sym = golden_k(g, tau, fb::DensitySymmetry::Symmetric, &sym);
    const std::vector<double> k_anti = golden_k(g, tau, fb::DensitySymmetry::Antisymmetric, &anti);
    const std::vector<double> upper = golden_k(g, tau, fb::DensitySymmetry::Symmetric, &d);
    const std::vector<double> lower = golden_k(g, tau, fb::DensitySymmetry::Symmetric, &d_t);
    ASSERT_EQ(k_sym.size(), n * n);
    ASSERT_EQ(k_anti.size(), n * n);
    ASSERT_EQ(upper.size(), n * n);
    ASSERT_EQ(lower.size(), n * n);
    double worst = 0.0;
    double worst_anti = 0.0;
    double size_anti = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t e = i * n + j;
        const std::size_t t = j * n + i;
        EXPECT_LE(std::abs(k_sym[e] - g.vk[e]), kRelTol * scale)
            << "tau " << tau << ", K[S](" << i << ", " << j << ")";
        const double want = aos[i].shell_index <= aos[j].shell_index ? upper[e] : lower[t];
        const double err = std::abs(k_sym[e] + k_anti[e] - want);
        worst = std::max(worst, err);
        EXPECT_LE(err, kRelTol * scale) << "tau " << tau << ", K[D](" << i << ", " << j << ")";
        // An a < b block is mirrored, but an a == b block is computed whole:
        // antisymmetric to rounding, not bitwise.
        worst_anti = std::max(worst_anti, std::abs(k_anti[e] + k_anti[t]));
        size_anti = std::max(size_anti, std::abs(k_anti[e]));
      }
    }
    EXPECT_LE(worst_anti, kRelTol * scale) << "tau " << tau;
    // Else the case could not tell the antisymmetric mirror from a symmetric
    // one, or from none.
    EXPECT_GT(size_anti, 1e-6 * scale) << "tau " << tau;
    std::cout << "[ two-sweep] " << GetParam() << " tau " << tau << ": worst " << worst << " ("
              << worst / scale << " of max|vk|), K[A] up to " << size_anti << ", antisymmetric to "
              << worst_anti << "\n";
    RecordProperty(std::format("tau_{}_two_sweep_worst_abs_error", tau),
                   std::format("{:.3e}", worst));
    RecordProperty(std::format("tau_{}_antisymmetry_error", tau),
                   std::format("{:.3e}", worst_anti));
  }
}

INSTANTIATE_TEST_SUITE_P(H2O, ExchangeGolden, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));
// Five H2 4 bohr apart in cc-pVDZ, with a converged RHF density.
INSTANTIATE_TEST_SUITE_P(Chain, ExchangeGolden, ::testing::Values("h10_chain_ccpvdz"));

// The H10 chain's RHF density decays, and with the dressed record bound a
// group's reach stops well short of the 23-bohr chain, so the density factor
// fires: a share of the block factors exchange makes (its a <= b blocks: the
// 30 shells' 465 pairs a <= b, times the four (lc, ld) of s and p) falls below
// max|D|, and none is above it. The counts are recorded and pinned (measured
// on the RX 9060 XT): 24 of the 1,860 at the default, 1e-13; 72 at 1e-10, half
// the 144 FelizJueves's own density_decay gives over every block of its host
// record bounds (a factor is the same for a block and its transpose, and no
// a == b block is under max|D|); and 121 at 1e-6. The share shrinks with tau,
// since a smaller tau lengthens every reach.
// A count is a sum of comparisons against the reach's cutoff, so a backend
// whose exp or sqrt rounds differently could move it by a block at a boundary;
// the nonzero share is the property, the pin a regression check.
TEST(Exchange, TheChainsBlockFactorsFallBelowTheDensityMaximum) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden("h10_chain_ccpvdz");
  ASSERT_FALSE(g.vk.empty());
  fb::DeviceResources res{0};
  auto classes = ordered(res, g.basis);
  ASSERT_TRUE(classes);
  fb::CartesianMatrixAO density(res, g.basis);
  upload(res, g.density, density);
  const double dmax = max_abs(g.density);
  for (const auto &[tau, pinned] :
       {std::pair{1e-13, std::int64_t{24}}, {1e-10, std::int64_t{72}}, {1e-6, std::int64_t{121}}}) {
    const auto factors = fb::exchange_block_factors(res, *classes, density, tau);
    ASSERT_TRUE(factors) << factors.error().message;
    ASSERT_EQ(factors->size(), std::size_t{1860});
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
  auto classes = ordered(res, basis);
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
  auto classes = ordered(res, two);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, one);
  fb::CartesianMatrixAO k(res, two);
  const fb::Status status = fb::exchange(res, *classes, density, k);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

// Classes of a larger basis than K's: a basis of another AO count is refused
// before anything is launched, not written out of bounds.
TEST(Exchange, RefusesClassesOfALargerBasis) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet one({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::BasisSet two({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0}),
                          fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 1.0})});
  auto classes = ordered(res, two);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, one);
  fb::CartesianMatrixAO k(res, one);
  const fb::Status status = fb::exchange(res, *classes, density, k);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

// A bra's block tables are made against every class of its set, so the
// classes go in whole and in order: H2O/STO-3G's four ((s, s), (s, p),
// (p, s), (p, p)) less one, or with two swapped, are refused by both entry
// points before anything is launched.
TEST(Exchange, RefusesASubsetOrAReorderingOfTheClasses) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden("h2o_sto3g");
  ASSERT_FALSE(g.vk.empty());
  fb::DeviceResources res{0};
  auto classes = ordered(res, g.basis);
  ASSERT_TRUE(classes);
  ASSERT_EQ(classes->size(), std::size_t{4});
  const fb::CartesianMatrixAO density(res, g.basis);
  fb::CartesianMatrixAO k(res, g.basis);
  const std::span<fb::OrderedPairClass> all{*classes};
  for (const std::span<fb::OrderedPairClass> subset : {all.first(3), all.last(3)}) {
    const fb::Status status = fb::exchange(res, subset, density, k);
    ASSERT_FALSE(status);
    EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
    const auto factors = fb::exchange_block_factors(res, subset, density);
    ASSERT_FALSE(factors);
    EXPECT_EQ(factors.error().kind, fb::ErrorKind::InvalidConfig);
  }
  std::swap((*classes)[1], (*classes)[2]);
  const fb::Status status = fb::exchange(res, *classes, density, k);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
  // Back in order, the same classes are accepted.
  std::swap((*classes)[1], (*classes)[2]);
  EXPECT_TRUE(fb::exchange(res, *classes, density, k));
}

TEST(Exchange, RefusesAClassAboveD) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(3, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  auto classes = ordered(res, basis);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, basis);
  fb::CartesianMatrixAO k(res, basis);
  const fb::Status status = fb::exchange(res, *classes, density, k);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
