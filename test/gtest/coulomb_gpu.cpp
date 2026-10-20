// GPU cases for integrals.coulomb: the Cartesian Coulomb matrix J of H2O in
// STO-3G (s, p; O's p shell precedes the H s shells, so the la <= lb swap is
// exercised, in the density's read as in J's scatter) and cc-pVDZ (s, p, d, so
// every e_hermite class through (4, 4)) against libcint's J of the same
// density, recorded by PySCF (test/gtest/data/coulomb/, copied from
// FelizJueves's fixtures by tools/extract_integral_golden.py), plus, as
// overlap_gpu.cpp, the zeroing and the refusals.

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

// FelizJueves holds its J to libcint at 1e-12 of scale (test/oracle/
// tolerance.py); the same bar here, against max|J| as overlap_gpu.cpp. The
// recorded density is a Hilbert matrix, every element positive, so max|J| is
// not a scale cancellation has shrunk.
constexpr double kRelTol = 1e-12;

struct Golden {
  fb::BasisSet basis;
  std::vector<double> density; // nao x nao, row-major; each is symmetric
  std::vector<double> vj;
};

Golden read_golden(const std::string &stem) {
  const auto doc =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "coulomb" / (stem + ".json"));
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
           arrays["vj"]["values"].get<std::vector<double>>()};
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

class CoulombGolden : public ::testing::TestWithParam<const char *> {};

TEST_P(CoulombGolden, MatchesLibcint) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.vj.empty());
  fb::DeviceResources res{0};
  const auto classes = fb::primitive_pair_classes(res, g.basis);
  ASSERT_TRUE(classes);
  fb::CartesianMatrixAO density(res, g.basis);
  upload(res, g.density, density);
  fb::CartesianMatrixAO j(res, g.basis);
  const fb::Status status = fb::coulomb(res, *classes, density, j);
  ASSERT_TRUE(status) << (status ? "" : status.error().message);
  const std::vector<double> got = download(res, j);
  ASSERT_EQ(got.size(), g.vj.size());

  double scale = 0.0;
  for (const double v : g.vj) {
    scale = std::max(scale, std::abs(v));
  }
  const std::size_t n = g.basis.nao_cart();
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      const double err = std::abs(got[i * n + k] - g.vj[i * n + k]);
      worst = std::max(worst, err);
      EXPECT_LE(err, kRelTol * scale) << "J(" << i << ", " << k << ")";
      // The atomics add each mirror pair in their own order, so symmetric to
      // rounding, not bitwise.
      EXPECT_LE(std::abs(got[i * n + k] - got[k * n + i]), kRelTol * scale);
    }
  }
  RecordProperty("worst_abs_error", std::format("{:.3e}", worst));
}

INSTANTIATE_TEST_SUITE_P(H2O, CoulombGolden, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));

TEST(Coulomb, ZeroesTheMatrixFirst) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::CartesianMatrixAO density(res, basis);
  fb::CartesianMatrixAO j(res, basis);
  const double stale = 42.0;
  ASSERT_EQ(wwr::wwrMemcpyAsync(j.data(), &stale, sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  // No classes: the only work left is the zeroing.
  ASSERT_TRUE(fb::coulomb(res, {}, density, j));
  EXPECT_EQ(download(res, j)[0], 0.0);
}

TEST(Coulomb, RefusesADensityOfAnotherOrder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet one({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::BasisSet two({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0}),
                          fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 1.0})});
  const auto classes = fb::primitive_pair_classes(res, two);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, one);
  fb::CartesianMatrixAO j(res, two);
  const fb::Status status = fb::coulomb(res, *classes, density, j);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(Coulomb, RefusesAClassAboveD) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(3, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const auto classes = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(classes);
  const fb::CartesianMatrixAO density(res, basis);
  fb::CartesianMatrixAO j(res, basis);
  const fb::Status status = fb::coulomb(res, *classes, density, j);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
