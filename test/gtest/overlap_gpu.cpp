// GPU cases for integrals.one_electron.overlap: the Cartesian S of H2O in
// STO-3G (s, p; O's p shell precedes the H s shells, so the la <= lb swap is
// exercised) and cc-pVDZ (s, p, d) against libcint's int1e_ovlp, recorded by
// PySCF (test/gtest/data/one_electron/, copied from FelizJueves's fixtures by
// tools/extract_one_electron_golden.py), plus the zeroing and the class-range
// refusal.

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

// FelizJueves holds its one-electron kernels to libcint at 1e-12 * max|S|
// (test/oracle/tolerance.py); the same bar here.
constexpr double kRelTol = 1e-12;

struct Golden {
  fb::BasisSet basis;
  std::vector<double> ovlp; // nao x nao, row-major; S is symmetric
};

Golden read_golden(const std::string &stem) {
  const auto doc =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "one_electron" / (stem + ".json"));
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
  Golden g{fb::BasisSet(std::move(shells)),
           (*doc)["arrays"]["int1e_ovlp"]["values"].get<std::vector<double>>()};
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

// S on the device for basis: records, upload, launch.
std::vector<double> device_overlap(fb::DeviceResources &res, const fb::BasisSet &basis) {
  const auto classes = fb::primitive_pair_classes(basis);
  EXPECT_TRUE(classes);
  const auto device = fb::upload_primitive_pairs(res, *classes);
  EXPECT_TRUE(device);
  fb::CartesianMatrixAO s(res, basis);
  const fb::Status status = fb::overlap(res, *device, s);
  EXPECT_TRUE(status) << (status ? "" : status.error().message);
  return download(res, s);
}

class OverlapGolden : public ::testing::TestWithParam<const char *> {};

TEST_P(OverlapGolden, MatchesLibcint) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.ovlp.empty());
  fb::DeviceResources res{0};
  const std::vector<double> got = device_overlap(res, g.basis);
  ASSERT_EQ(got.size(), g.ovlp.size());

  double scale = 0.0;
  for (const double v : g.ovlp) {
    scale = std::max(scale, std::abs(v));
  }
  const std::size_t n = g.basis.nao_cart();
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double err = std::abs(got[i * n + j] - g.ovlp[i * n + j]);
      worst = std::max(worst, err);
      EXPECT_LE(err, kRelTol * scale) << "S(" << i << ", " << j << ")";
      // The atomics add each mirror pair in their own order, so symmetric to
      // rounding, not bitwise.
      EXPECT_LE(std::abs(got[i * n + j] - got[j * n + i]), kRelTol * scale);
    }
  }
  RecordProperty("worst_abs_error", std::format("{:.3e}", worst));
}

INSTANTIATE_TEST_SUITE_P(H2O, OverlapGolden, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));

TEST(Overlap, ZeroesTheMatrixFirst) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  fb::CartesianMatrixAO s(res, basis);
  const double stale = 42.0;
  ASSERT_EQ(wwr::wwrMemcpyAsync(s.data(), &stale, sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  // No classes: the only work left is the zeroing.
  ASSERT_TRUE(fb::overlap(res, {}, s));
  EXPECT_EQ(download(res, s)[0], 0.0);
}

TEST(Overlap, RefusesAClassAboveD) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis({fb::Shell(3, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const auto classes = fb::primitive_pair_classes(basis);
  ASSERT_TRUE(classes);
  const auto device = fb::upload_primitive_pairs(res, *classes);
  ASSERT_TRUE(device);
  fb::CartesianMatrixAO s(res, basis);
  const fb::Status status = fb::overlap(res, *device, s);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
