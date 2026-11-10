// GPU cases for the top of the density-fitting aux ladder: the metric (P|Q)
// through (i|i) and the three-centre (mn|P) through (ff|i), over the
// hand-built basis of data/df_high_aux/ -- one shell per l on two skew
// centres, orbitals to f and aux to i -- against PySCF's int2c2e and
// int3c2e.
//
// WHY A HAND-BUILT BASIS. df_metric_gpu.cpp and df_three_center_gpu.cpp run
// the two fitting sets this tree carries, whose aux shells stop at g: between
// them they reach (g|g) and (ff|g) and nothing above. The emitted ladder goes
// to (i|i) and (ff|i) (kMetricMaxMomentum and kThreeCenterAuxMaxMomentum),
// and no carried set has an i fitting shell, so the classes above g have no
// other way to be run. This fixture is the smallest basis that reaches every
// one of them, which df_high_aux_host.cpp's coverage case checks rather than
// asserts: that check reads only the manifest, so it needs no device.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting;

#include "test_workspaces.h"

#include "df_high_aux_fixture.h"

namespace {

namespace fb = free_breakfast;
using fb::test::high_aux::Fixture;
using fb::test::high_aux::fixture;

// The same bar the carried sets hold to, 1e-12 of max (df_metric_gpu.cpp,
// df_three_center_gpu.cpp), and these classes clear it by two orders: at
// L = 12 the measured worst is 1.2e-14 of max|M| for the metric and 1.6e-14
// of max|(mn|P)| for the three-centre, on the RTX 3080. The tower does not
// visibly lose accuracy up the aux ladder, so there is no reason to grant it
// a looser tolerance than the g classes get. Each case prints what it
// achieved, so a regression that still passes shows up as a number.
constexpr double kRelTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "df_high_aux";

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The golden (mn|P), gathered from the manifest's aux pieces into the call's
// layout: P + naux (n + nao m), P fastest. Each piece is (nao, nao, slab) in
// Fortran order, df_reference.py's split (df_three_center_gpu.cpp gathers the
// same shape).
std::vector<double> golden_eri3(const fb::json &doc, const std::size_t nao,
                                const std::size_t naux) {
  std::vector<double> golden(naux * nao * nao, std::numeric_limits<double>::quiet_NaN());
  std::size_t covered = 0;
  for (const fb::json &piece : doc["int3c2e"]) {
    const auto range = piece["aux"].get<std::vector<std::size_t>>();
    const auto slab = fb::read_npy(kDataDir / piece["file"].get<std::string>());
    EXPECT_TRUE(slab) << slab.error().message;
    EXPECT_EQ(slab->shape, (std::vector<std::size_t>{nao, nao, range[1] - range[0]}));
    EXPECT_EQ(range[0], covered);
    for (std::size_t p = range[0]; p < range[1]; ++p) {
      for (std::size_t n = 0; n < nao; ++n) {
        for (std::size_t m = 0; m < nao; ++m) {
          golden[p + (naux * (n + (nao * m)))] =
              slab->data[m + (nao * (n + (nao * (p - range[0]))))];
        }
      }
    }
    covered = range[1];
  }
  EXPECT_EQ(covered, naux);
  return golden;
}

double max_abs(const std::vector<double> &a) {
  double scale = 0.0;
  for (const double v : a) {
    scale = std::max(scale, std::abs(v));
  }
  return scale;
}

// (P|Q) over aux s..i: all 28 classes, (i|i) at L = 12 the top one.
TEST(DfHighAux, MetricThroughIiMatchesInt2c2e) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(kDataDir / "df_high_aux.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const Fixture f = fixture(*doc);
  fb::DeviceResources res{0, fb::test::kWorkspaces};

  const auto golden = fb::read_npy(kDataDir / (*doc)["int2c2e"].get<std::string>());
  ASSERT_TRUE(golden) << golden.error().message;
  ASSERT_EQ(golden->shape, (std::vector<std::size_t>{f.naux, f.naux}));
  const double scale = max_abs(golden->data);

  const auto u = fb::upload_spherical_transform_matrix(res, f.aux);
  ASSERT_TRUE(u) << u.error().message;
  std::array<std::vector<double>, 2> runs;
  for (std::vector<double> &run : runs) {
    fb::MatrixAOPair m(res, f.aux, *u);
    const fb::Status status = fb::metric(res, f.aux, m);
    ASSERT_TRUE(status) << status.error().message;
    run = download(res, m.spherical().data(), f.naux * f.naux);
  }

  double worst = 0.0;
  std::size_t at = 0;
  double asymmetry = 0.0;
  for (std::size_t q = 0; q < f.naux; ++q) {
    for (std::size_t p = 0; p < f.naux; ++p) {
      const std::size_t i = p + (f.naux * q);
      if (const double d = std::abs(runs[0][i] - golden->data[i]); d > worst) {
        worst = d;
        at = i;
      }
      asymmetry = std::max(asymmetry, std::abs(runs[0][i] - runs[0][q + (f.naux * p)]));
    }
  }
  EXPECT_LE(worst, kRelTol * scale) << "worst |M - int2c2e| " << worst << " at (" << at % f.naux
                                    << ", " << at / f.naux << "), max|M| " << scale;
  std::cout << std::format("[ metric   ] high aux: naux {}, max|M| {:.4f}, worst "
                           "|M - int2c2e| / max|M| {:.3e}\n",
                           f.naux, scale, worst / scale);
  EXPECT_EQ(asymmetry, 0.0) << "the spherical metric is not exactly symmetric";
  EXPECT_EQ(std::memcmp(runs[0].data(), runs[1].data(), sizeof(double) * runs[0].size()), 0)
      << "two runs differ";
}

// (mn|P) over orbitals s..f and aux s..i: all 70 classes, (ff|i) at L = 12.
TEST(DfHighAux, ThreeCenterThroughFfIMatchesInt3c2e) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(kDataDir / "df_high_aux.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const Fixture f = fixture(*doc);
  fb::DeviceResources res{0, fb::test::kWorkspaces};

  const std::vector<double> golden = golden_eri3(*doc, f.nao, f.naux);
  const double scale = max_abs(golden);
  ASSERT_GT(scale, 0.0);

  const auto classes = fb::primitive_pair_classes(res, f.basis);
  ASSERT_TRUE(classes) << classes.error().message;
  fb::Buffers::device<double> b(f.naux * f.nao * f.nao, res.device());
  std::array<std::vector<double>, 2> runs;
  for (std::vector<double> &run : runs) {
    const fb::Status status = fb::three_center(res, f.basis, *classes, f.aux, b.data());
    ASSERT_TRUE(status) << status.error().message;
    run = download(res, b.data(), b.num_elements());
  }

  double worst = 0.0;
  std::array<std::size_t, 3> at{};
  double asymmetry = 0.0;
  for (std::size_t m = 0; m < f.nao; ++m) {
    for (std::size_t n = 0; n < f.nao; ++n) {
      for (std::size_t p = 0; p < f.naux; ++p) {
        const std::size_t i = p + (f.naux * (n + (f.nao * m)));
        if (const double d = std::abs(runs[0][i] - golden[i]); d > worst) {
          worst = d;
          at = {m, n, p};
        }
        asymmetry =
            std::max(asymmetry, std::abs(runs[0][i] - runs[0][p + (f.naux * (m + (f.nao * n)))]));
      }
    }
  }
  EXPECT_LE(worst, kRelTol * scale)
      << "worst |(mn|P) - int3c2e| " << worst << " at (" << at[0] << ", " << at[1] << " | " << at[2]
      << "), max|(mn|P)| " << scale;
  std::cout << std::format("[ eri3     ] high aux: nao {}, naux {}, max|(mn|P)| {:.4f}, worst "
                           "|(mn|P) - int3c2e| / max|(mn|P)| {:.3e}\n",
                           f.nao, f.naux, scale, worst / scale);
  EXPECT_EQ(asymmetry, 0.0) << "(mn|P) and (nm|P) differ";
  EXPECT_EQ(std::memcmp(runs[0].data(), runs[1].data(), sizeof(double) * runs[0].size()), 0)
      << "two runs differ";
}

} // namespace
