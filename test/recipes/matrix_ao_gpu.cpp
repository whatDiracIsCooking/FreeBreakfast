// GPU cases for recipes.matrix_ao: the layout orders, the uploaded U, cartesian_to_spherical
// and spherical_to_cartesian against the host U^T M U and U M U^T built from
// spherical_transform_matrix, the trace pairing the two share, the identity copy
// below d, and the pair's checks on U and on the device.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

const fb::Vec<double, 3> kOrigin{0.0, 0.0, 0.0};

// s + p + d + s: nao_cart = 1 + 3 + 6 + 1 = 11, nao_sphe = 1 + 3 + 5 + 1 = 10.
fb::BasisSet spd_basis() {
  return fb::BasisSet({fb::Shell(0, {{3.42, 0.15}}, kOrigin), fb::Shell(1, {{0.62, 1.0}}, kOrigin),
                       fb::Shell(2, {{0.8, 1.0}}, kOrigin), fb::Shell(0, {{0.5, 1.0}}, kOrigin)});
}

// s + p only: U is the identity.
fb::BasisSet sp_basis() {
  return fb::BasisSet({fb::Shell(0, {{1.0, 1.0}}, kOrigin), fb::Shell(1, {{1.0, 1.0}}, kOrigin)});
}

// Column-major n x n, element (i, j) at i + j*n.
std::vector<double> random_matrix(std::size_t n, std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> m(n * n);
  for (auto &v : m) {
    v = dist(gen);
  }
  return m;
}

template <fb::BasisLayout L>
void upload(fb::DeviceResources &res, fb::MatrixAO<L> &m, const std::vector<double> &host) {
  ASSERT_EQ(host.size(), m.num_elements());
  ASSERT_EQ(wwr::wwrMemcpyAsync(m.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

template <fb::BasisLayout L>
std::vector<double> download(fb::DeviceResources &res, const fb::MatrixAO<L> &m) {
  std::vector<double> host(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), m.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// The host references. u is spherical_transform_matrix's nc x ns; it, m and the
// result are all column-major.
std::vector<double> host_cartesian_to_spherical(const std::vector<double> &u,
                                                const std::vector<double> &m, std::size_t nc,
                                                std::size_t ns) {
  std::vector<double> out(ns * ns, 0.0);
  for (std::size_t a = 0; a < ns; ++a) {
    for (std::size_t b = 0; b < ns; ++b) {
      double sum = 0.0;
      for (std::size_t i = 0; i < nc; ++i) {
        for (std::size_t j = 0; j < nc; ++j) {
          sum += u[i + (a * nc)] * m[i + (j * nc)] * u[j + (b * nc)];
        }
      }
      out[a + (b * ns)] = sum;
    }
  }
  return out;
}

std::vector<double> host_spherical_to_cartesian(const std::vector<double> &u,
                                                const std::vector<double> &m, std::size_t nc,
                                                std::size_t ns) {
  std::vector<double> out(nc * nc, 0.0);
  for (std::size_t i = 0; i < nc; ++i) {
    for (std::size_t j = 0; j < nc; ++j) {
      double sum = 0.0;
      for (std::size_t a = 0; a < ns; ++a) {
        for (std::size_t b = 0; b < ns; ++b) {
          sum += u[i + (a * nc)] * m[a + (b * ns)] * u[j + (b * nc)];
        }
      }
      out[i + (j * nc)] = sum;
    }
  }
  return out;
}

// tr(A B) for column-major n x n A and B.
double trace_of_product(const std::vector<double> &a, const std::vector<double> &b, std::size_t n) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      sum += a[i + (k * n)] * b[k + (i * n)];
    }
  }
  return sum;
}

void expect_near_all(const std::vector<double> &got, const std::vector<double> &want) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], 1e-12) << "element " << i;
  }
}

// upload_spherical_transform_matrix for one basis, for pairs over it to share.
struct Fixture {
  fb::DeviceResources res{0};
  fb::BasisSet basis;
  std::shared_ptr<const fb::Buffers::device<double>> u;

  explicit Fixture(fb::BasisSet b) : basis(std::move(b)) {
    auto made = fb::upload_spherical_transform_matrix(res, basis);
    EXPECT_TRUE(made);
    if (made) {
      u = std::move(*made);
    }
  }

  fb::MatrixAOPair pair() { return fb::MatrixAOPair(res, basis, u); }
};

TEST(MatrixAO, OrdersFollowTheLayout) {
  FBF_REQUIRE_GPU();

  Fixture fx(spd_basis());
  const fb::MatrixAOPair pair = fx.pair();
  EXPECT_EQ(pair.cartesian().n(), 11U);
  EXPECT_EQ(pair.spherical().n(), 10U);
  EXPECT_EQ(pair.cartesian().num_elements(), 121U);
  EXPECT_EQ(pair.dev_idx(), fx.res.dev_idx());

  // Zero-initialised.
  for (const double v : download(fx.res, pair.spherical())) {
    EXPECT_EQ(v, 0.0);
  }
}

TEST(MatrixAO, TransformIsUploadedAsIs) {
  FBF_REQUIRE_GPU();

  Fixture fx(spd_basis());
  ASSERT_NE(fx.u, nullptr);
  const std::vector<double> u = fb::spherical_transform_matrix(fx.basis);
  ASSERT_EQ(fx.u->num_elements(), u.size());
  std::vector<double> device_u(u.size());
  ASSERT_EQ(wwr::wwrMemcpyAsync(device_u.data(), fx.u->data(), u.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, fx.res.stream()),
            wwr::wwrSuccess);
  ASSERT_EQ(wwr::wwrStreamSynchronize(fx.res.stream()), wwr::wwrSuccess);
  EXPECT_EQ(device_u, u);
}

TEST(MatrixAO, NoTransformBelowD) {
  FBF_REQUIRE_GPU();

  EXPECT_EQ(Fixture(sp_basis()).u, nullptr);
  EXPECT_EQ(Fixture(fb::BasisSet{}).u, nullptr);
}

TEST(MatrixAO, CartesianToSphericalMatchesHostSandwich) {
  FBF_REQUIRE_GPU();

  Fixture fx(spd_basis());
  fb::MatrixAOPair pair = fx.pair();
  const std::vector<double> m = random_matrix(11, 7);
  upload(fx.res, pair.cartesian(), m);
  ASSERT_TRUE(pair.cartesian_to_spherical(fx.res));
  expect_near_all(download(fx.res, pair.spherical()),
                  host_cartesian_to_spherical(fb::spherical_transform_matrix(fx.basis), m, 11, 10));
}

TEST(MatrixAO, SphericalToCartesianMatchesHostSandwich) {
  FBF_REQUIRE_GPU();

  Fixture fx(spd_basis());
  fb::MatrixAOPair pair = fx.pair();
  const std::vector<double> m = random_matrix(10, 11);
  upload(fx.res, pair.spherical(), m);
  ASSERT_TRUE(pair.spherical_to_cartesian(fx.res));
  expect_near_all(download(fx.res, pair.cartesian()),
                  host_spherical_to_cartesian(fb::spherical_transform_matrix(fx.basis), m, 11, 10));
}

// tr(U D U^T F) = tr(D U^T F U): a contraction of a lifted matrix with a
// Cartesian one equals the spherical contraction with its reduction. The two
// pairs share one U.
TEST(MatrixAO, BothDirectionsPreserveTheTracePairing) {
  FBF_REQUIRE_GPU();

  Fixture fx(spd_basis());
  fb::MatrixAOPair density = fx.pair();
  fb::MatrixAOPair fock = fx.pair();
  EXPECT_EQ(fx.u.use_count(), 3);

  const std::vector<double> d = random_matrix(10, 3);
  const std::vector<double> f = random_matrix(11, 5);
  upload(fx.res, density.spherical(), d);
  upload(fx.res, fock.cartesian(), f);
  ASSERT_TRUE(density.spherical_to_cartesian(fx.res));
  ASSERT_TRUE(fock.cartesian_to_spherical(fx.res));

  const double cartesian = trace_of_product(download(fx.res, density.cartesian()), f, 11);
  const double spherical = trace_of_product(d, download(fx.res, fock.spherical()), 10);
  EXPECT_NEAR(cartesian, spherical, 1e-12);
}

TEST(MatrixAO, BelowDBothProductsCopy) {
  FBF_REQUIRE_GPU();

  Fixture fx(sp_basis());
  fb::MatrixAOPair pair = fx.pair();
  ASSERT_EQ(fx.u, nullptr);

  const std::vector<double> m = random_matrix(4, 9);
  upload(fx.res, pair.cartesian(), m);
  ASSERT_TRUE(pair.cartesian_to_spherical(fx.res));
  EXPECT_EQ(download(fx.res, pair.spherical()), m);

  const std::vector<double> m2 = random_matrix(4, 10);
  upload(fx.res, pair.spherical(), m2);
  ASSERT_TRUE(pair.spherical_to_cartesian(fx.res));
  EXPECT_EQ(download(fx.res, pair.cartesian()), m2);
}

TEST(MatrixAO, EmptyBasisSucceedsWithoutWork) {
  FBF_REQUIRE_GPU();

  Fixture fx{fb::BasisSet{}};
  fb::MatrixAOPair pair = fx.pair();
  EXPECT_EQ(pair.cartesian().data(), nullptr);
  EXPECT_TRUE(pair.cartesian_to_spherical(fx.res));
  EXPECT_TRUE(pair.spherical_to_cartesian(fx.res));
}

// The threadsafe style re-executes the binary for the child, so the child
// initialises its own GPU runtime instead of inheriting a forked one.
TEST(MatrixAODeathTest, PairAbortsOnAUThatDoesNotFitTheBasis) {
  FBF_REQUIRE_GPU();
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  // A d shell needs a U.
  EXPECT_DEATH(
      {
        fb::DeviceResources res{0};
        const fb::MatrixAOPair pair(res, spd_basis(), nullptr);
      },
      "");
  // An s/p basis must not get one.
  EXPECT_DEATH(
      {
        Fixture fx(spd_basis());
        const fb::MatrixAOPair pair(fx.res, sp_basis(), fx.u);
      },
      "");
}

} // namespace
