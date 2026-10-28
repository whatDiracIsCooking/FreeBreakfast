// GPU cases for integrals.eri.density_fitting.fit: from PySCF's int2c2e (P|Q)
// and int3c2e (mn|P) (data/density_fitting/), uploaded, B = L^-1 (P|mn) gives
// B^T B = (mn|P) M^-1 (P|ls), formed on the host with a Cholesky solve of its
// own; and a metric that is not positive definite is Numerical, naming the
// pivot.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.integrals.eri.density_fitting.fit;

namespace {

namespace fb = free_breakfast;

// The issue's bar: B^T B against the host's (mn|P) M^-1 (P|ls) to 1e-12 of
// max|(mn|P) M^-1 (P|ls)|.
constexpr double kRelTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "density_fitting";

// One golden case: the metric, naux x naux column-major, and (P|mn) as the
// naux x nao^2 matrix the fit takes, element (P, n + nao m) at
// P + naux (n + nao m).
struct Golden {
  std::int32_t nao = 0;
  std::int32_t naux = 0;
  std::vector<double> metric;
  std::vector<double> b;
};

Golden read_golden(const std::string &name) {
  const auto doc = fb::read_json(kDataDir / "density_fitting.json");
  EXPECT_TRUE(doc) << (doc ? "" : doc.error().message);
  if (!doc) {
    return {};
  }
  for (const auto &c : (*doc)["integrals"]) {
    if (c["name"].get<std::string>() != name) {
      continue;
    }
    Golden g;
    g.nao = c["nao"].get<std::int32_t>();
    g.naux = c["naux"].get<std::int32_t>();
    const auto nao = static_cast<std::size_t>(g.nao);
    const auto naux = static_cast<std::size_t>(g.naux);

    const auto metric = fb::read_npy(kDataDir / c["int2c2e"].get<std::string>());
    EXPECT_TRUE(metric) << name;
    if (!metric) {
      return {};
    }
    EXPECT_EQ(metric->shape, (std::vector<std::size_t>{naux, naux})) << name;
    g.metric = metric->data;

    // Each piece is (nao, nao, end - begin), Fortran order: (mn|P) at
    // m + nao (n + nao (P - begin)). Transposed here into [P][n][m].
    g.b.assign(naux * nao * nao, std::numeric_limits<double>::quiet_NaN());
    std::size_t covered = 0;
    for (const auto &piece : c["int3c2e"]) {
      const auto range = piece["aux"].get<std::vector<std::size_t>>();
      const auto slab = fb::read_npy(kDataDir / piece["file"].get<std::string>());
      EXPECT_TRUE(slab) << name;
      if (!slab) {
        return {};
      }
      EXPECT_EQ(slab->shape, (std::vector<std::size_t>{nao, nao, range[1] - range[0]})) << name;
      for (std::size_t p = range[0]; p < range[1]; ++p) {
        for (std::size_t n = 0; n < nao; ++n) {
          for (std::size_t m = 0; m < nao; ++m) {
            g.b[p + naux * (n + nao * m)] = slab->data[m + nao * (n + nao * (p - range[0]))];
          }
        }
      }
      EXPECT_EQ(range[0], covered) << name;
      covered = range[1];
    }
    EXPECT_EQ(covered, naux) << name;
    return g;
  }
  ADD_FAILURE() << "no integrals case " << name;
  return {};
}

void upload(fb::DeviceResources &res, const std::vector<double> &host,
            fb::Buffers::device<double> &d) {
  ASSERT_EQ(host.size(), d.num_elements());
  ASSERT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
  ASSERT_TRUE(res.sync().has_value());
}

std::vector<double> download(fb::DeviceResources &res, const fb::Buffers::device<double> &d) {
  std::vector<double> host(d.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// X = M^-1 A for the naux x cols A, column-major, by a host Cholesky
// factorization of M and two triangular solves per column: the oracle's own,
// sharing nothing with the device's cholesky and trsm.
std::vector<double> host_solve(const std::vector<double> &metric, const std::vector<double> &a,
                               const std::size_t naux) {
  std::vector<double> l(naux * naux, 0.0);
  for (std::size_t j = 0; j < naux; ++j) {
    double d = metric[j + naux * j];
    for (std::size_t k = 0; k < j; ++k) {
      d -= l[j + naux * k] * l[j + naux * k];
    }
    EXPECT_GT(d, 0.0) << "host Cholesky, column " << j;
    l[j + naux * j] = std::sqrt(d);
    for (std::size_t i = j + 1; i < naux; ++i) {
      double s = metric[i + naux * j];
      for (std::size_t k = 0; k < j; ++k) {
        s -= l[i + naux * k] * l[j + naux * k];
      }
      l[i + naux * j] = s / l[j + naux * j];
    }
  }
  std::vector<double> x = a;
  const std::size_t cols = a.size() / naux;
  for (std::size_t c = 0; c < cols; ++c) {
    double *const y = x.data() + naux * c;
    for (std::size_t i = 0; i < naux; ++i) { // L y = a
      for (std::size_t k = 0; k < i; ++k) {
        y[i] -= l[i + naux * k] * y[k];
      }
      y[i] /= l[i + naux * i];
    }
    for (std::size_t i = naux; i-- > 0;) { // L^T x = y
      for (std::size_t k = i + 1; k < naux; ++k) {
        y[i] -= l[k + naux * i] * y[k];
      }
      y[i] /= l[i + naux * i];
    }
  }
  return x;
}

// B^T B against (mn|P) M^-1 (P|ls), over every pair of columns (mn), (ls)
// with (mn) <= (ls): both are symmetric in the two pairs.
void expect_fit_matches_host(const std::string &name) {
  const Golden g = read_golden(name);
  ASSERT_GT(g.naux, 0) << name;
  const auto naux = static_cast<std::size_t>(g.naux);
  const std::size_t pairs = g.b.size() / naux;

  fb::DeviceResources res{0};
  fb::Buffers::device<double> metric(g.metric.size(), res.device());
  fb::Buffers::device<double> b(g.b.size(), res.device());
  upload(res, g.metric, metric);
  upload(res, g.b, b);
  const fb::Status fitted = fb::density_fit(res, g.nao, g.naux, metric.data(), b.data());
  ASSERT_TRUE(fitted) << name << ": " << fitted.error().message;
  const std::vector<double> fit = download(res, b);

  const std::vector<double> x = host_solve(g.metric, g.b, naux);
  double scale = 0.0;
  double worst = 0.0;
  for (std::size_t c2 = 0; c2 < pairs; ++c2) {
    const double *const a2 = g.b.data() + naux * c2;
    const double *const b2 = fit.data() + naux * c2;
    for (std::size_t c1 = 0; c1 <= c2; ++c1) {
      const double *const x1 = x.data() + naux * c1;
      const double *const b1 = fit.data() + naux * c1;
      double want = 0.0;
      double got = 0.0;
      for (std::size_t q = 0; q < naux; ++q) {
        want += x1[q] * a2[q];
        got += b1[q] * b2[q];
      }
      scale = std::max(scale, std::abs(want));
      worst = std::max(worst, std::abs(got - want));
    }
  }
  ASSERT_GT(scale, 0.0) << name;
  EXPECT_LE(worst, kRelTol * scale) << name << ": max|B^T B - (mn|P) M^-1 (P|ls)| = " << worst
                                    << ", " << worst / scale << " of max " << scale;
}

TEST(DfFit, BtBIsTheFittedEriInCcPvdz) {
  FBF_REQUIRE_GPU();
  expect_fit_matches_host("h2o_ccpvdz");
}

// cc-pVTZ-JKFIT reaches g fitting functions over O's f orbitals.
TEST(DfFit, BtBIsTheFittedEriInCcPvtz) {
  FBF_REQUIRE_GPU();
  expect_fit_matches_host("h2o_ccpvtz");
}

// The golden metric with one diagonal element made negative: the leading
// minors before it are untouched and positive, so cholesky stops exactly there.
TEST(DfFit, NonPositiveDefiniteMetricIsNumericalNamingThePivot) {
  FBF_REQUIRE_GPU();
  Golden g = read_golden("h2o_ccpvdz");
  ASSERT_GT(g.naux, 40);
  const auto naux = static_cast<std::size_t>(g.naux);
  constexpr std::size_t kBroken = 40; // 0-based; the 1-based pivot 41
  g.metric[kBroken + naux * kBroken] = -g.metric[kBroken + naux * kBroken];

  fb::DeviceResources res{0};
  fb::Buffers::device<double> metric(g.metric.size(), res.device());
  fb::Buffers::device<double> b(g.b.size(), res.device());
  upload(res, g.metric, metric);
  upload(res, g.b, b);
  const fb::Status fitted = fb::density_fit(res, g.nao, g.naux, metric.data(), b.data());
  ASSERT_FALSE(fitted);
  EXPECT_EQ(fitted.error().kind, fb::ErrorKind::Numerical);
  EXPECT_NE(fitted.error().message.find("pivot 41 of 116"), std::string::npos)
      << fitted.error().message;
}

TEST(DfFit, EmptyFitSucceedsAndNegativeCountsAreInvalid) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  EXPECT_TRUE(fb::density_fit(res, 0, 5, nullptr, nullptr));
  EXPECT_TRUE(fb::density_fit(res, 5, 0, nullptr, nullptr));
  const fb::Status negative = fb::density_fit(res, -1, 5, nullptr, nullptr);
  ASSERT_FALSE(negative);
  EXPECT_EQ(negative.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
