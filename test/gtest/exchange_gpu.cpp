// GPU cases for integrals.exchange: the Cartesian exchange matrix K of H2O in
// STO-3G (s, p; O's p shell precedes the H s shells, so (p, s) and (s, p) are
// both reached) and cc-pVDZ (s, p, d, so every k class through (dd|dd))
// against libcint's K of the same density, recorded by PySCF
// (test/gtest/data/exchange/, copied from FelizJueves's fixtures by
// tools/extract_integral_golden.py), both unscreened (tau = 0) and at the
// default screen; that a tau no bound reaches screens everything; plus, as
// coulomb_gpu.cpp, the zeroing and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.linalg.matrix_ao;
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

// A symmetric matrix onto the device: row- and column-major are one layout.
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

// K of g's density at tau, downloaded.
std::vector<double> golden_k(const Golden &g, const double tau) {
  fb::DeviceResources res{0};
  const auto classes = ordered(res, g.basis);
  EXPECT_TRUE(classes);
  if (!classes) {
    return {};
  }
  fb::CartesianMatrixAO density(res, g.basis);
  upload(res, g.density, density);
  fb::CartesianMatrixAO k(res, g.basis);
  const fb::Status status = fb::exchange(res, *classes, density, k, tau);
  EXPECT_TRUE(status) << (status ? "" : status.error().message);
  return download(res, k);
}

class ExchangeGolden : public ::testing::TestWithParam<const char *> {};

// At tau = 0 nothing is screened; at the default each skipped primitive
// quartet's bound is under 1e-10, and the screened K must still meet the same
// bar against libcint. How far the screen moved K is recorded, not bounded.
TEST_P(ExchangeGolden, MatchesLibcint) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.vk.empty());
  double scale = 0.0;
  for (const double v : g.vk) {
    scale = std::max(scale, std::abs(v));
  }
  const std::size_t n = g.basis.nao_cart();
  const std::vector<double> unscreened = golden_k(g, 0.0);
  const std::vector<double> screened = golden_k(g, 1e-10);
  for (const auto &[name, got] : {std::pair{"unscreened", &unscreened}, {"screened", &screened}}) {
    ASSERT_EQ(got->size(), g.vk.size()) << name;
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const double err = std::abs((*got)[i * n + j] - g.vk[i * n + j]);
        worst = std::max(worst, err);
        EXPECT_LE(err, kRelTol * scale) << name << " K(" << i << ", " << j << ")";
        // K_ab and K_ba are computed separately (nothing is mirrored), so
        // symmetric to rounding, not bitwise.
        EXPECT_LE(std::abs((*got)[i * n + j] - (*got)[j * n + i]), kRelTol * scale) << name;
      }
    }
    RecordProperty(std::format("{}_worst_abs_error", name), std::format("{:.3e}", worst));
    RecordProperty(std::format("{}_worst_rel_error", name), std::format("{:.3e}", worst / scale));
  }
  double moved = 0.0;
  for (std::size_t i = 0; i < n * n; ++i) {
    moved = std::max(moved, std::abs(screened[i] - unscreened[i]));
  }
  RecordProperty("screen_max_abs_change", std::format("{:.3e}", moved));
}

INSTANTIATE_TEST_SUITE_P(H2O, ExchangeGolden, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));

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
