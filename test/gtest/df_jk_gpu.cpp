// GPU cases for integrals.eri.density_fitting.jk: from PySCF's int2c2e (P|Q)
// and int3c2e (mn|P) (data/density_fitting/), fitted by density_fit, J of the
// golden density and K of two host factors of it against PySCF's DF vj and vk;
// K exactly symmetric; the two factors' K agreeing; the two-factor K of a
// random non-symmetric A B^T against the host contraction over B, against
// integral-direct exchange within the DF error, and at A = B = C against the
// one-factor K; and the empty, negative and too-large counts.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.molecule;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.integrals.eri.density_fitting.jk;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.primitive_pair_classes;
import free_breakfast.pipeline.density_fitting;

namespace {

namespace fb = free_breakfast;

// The issue's bar: J and K against PySCF's DF vj and vk to 1e-12 of max|vj|
// and max|vk|, and the two factors' K to each other to 1e-12 of max|K|.
constexpr double kRelTol = 1e-12;
// 2 K[C C^T] by the two-factor gemm against the one-factor syrk, of max|K|
// (2.3e-16 measured on HIP).
constexpr double kRoundingTol = 1e-13;
// The two-factor DF K against integral-direct exchange, of max|K|: the fitting
// error of cc-pVDZ-JKFIT on a random density, which weights every AO product
// alike, tight ones included. 8.0e-3 measured for the test's A B^T; a random
// A A^T, which needs neither the new overload's asymmetry nor the antisymmetric
// sweep, misses by 1.5e-2 the same way. A K transposed, or its factors
// swapped, misses by about 0.75: its antisymmetric part is 0.26 of max|K|
// 0.69.
constexpr double kFittingTol = 3e-2;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "density_fitting";

// The h2o_ccpvdz case: the metric, naux x naux column-major; (P|mn) as the
// naux x nao^2 matrix the fit takes, element (P, n + nao m) at
// P + naux (n + nao m); and jk's D, vj and vk, nao x nao column-major.
struct Golden {
  std::int32_t nao = 0;
  std::int32_t naux = 0;
  std::int32_t occupied = 0;
  std::vector<double> metric;
  std::vector<double> b;
  std::vector<double> density;
  std::vector<double> vj;
  std::vector<double> vk;
};

std::vector<double> read_square(const std::string &file, const std::size_t n) {
  const auto a = fb::read_npy(kDataDir / file);
  EXPECT_TRUE(a) << file;
  if (!a) {
    return {};
  }
  EXPECT_EQ(a->shape, (std::vector<std::size_t>{n, n})) << file;
  return a->data;
}

Golden read_golden() {
  const std::string name = "h2o_ccpvdz";
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
    g.metric = read_square(c["int2c2e"].get<std::string>(), naux);

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

    const auto &jk = c["jk"];
    g.occupied = jk["occupied"].get<std::int32_t>();
    g.density = read_square(jk["density"].get<std::string>(), nao);
    g.vj = read_square(jk["vj"].get<std::string>(), nao);
    g.vk = read_square(jk["vk"].get<std::string>(), nao);
    return g;
  }
  ADD_FAILURE() << "no integrals case " << name;
  return {};
}

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *d) {
  ASSERT_EQ(wwr::wwrMemcpyAsync(d, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
  ASSERT_TRUE(res.sync().has_value());
}

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// C (n x rank, column-major) with D = 2 C C^T, from the eigendecomposition
// D = V diag(w) V^T by cyclic Jacobi: C = V_+ (w_+ / 2)^(1/2) over the rank
// largest eigenvalues, which must be positive.
std::vector<double> eigen_factor(std::vector<double> a, const std::size_t n,
                                 const std::size_t rank) {
  std::vector<double> v(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    v[i + n * i] = 1.0;
  }
  for (int32_t sweep = 0; sweep < 64; ++sweep) {
    double off = 0.0;
    double diag = 0.0;
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t p = 0; p < n; ++p) {
        (p == q ? diag : off) += a[p + n * q] * a[p + n * q];
      }
    }
    if (off <= 1e-32 * diag) {
      break;
    }
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        const double apq = a[p + n * q];
        if (apq == 0.0) {
          continue;
        }
        const double theta = (a[q + n * q] - a[p + n * p]) / (2.0 * apq);
        const double t = std::copysign(1.0, theta) / (std::abs(theta) + std::hypot(theta, 1.0));
        const double c = 1.0 / std::hypot(t, 1.0);
        const double s = t * c;
        // A <- R^T A R with R the (p, q) rotation, columns then rows.
        for (std::size_t r = 0; r < n; ++r) {
          const double arp = a[r + n * p];
          const double arq = a[r + n * q];
          a[r + n * p] = c * arp - s * arq;
          a[r + n * q] = s * arp + c * arq;
        }
        for (std::size_t r = 0; r < n; ++r) {
          const double apr = a[p + n * r];
          const double aqr = a[q + n * r];
          a[p + n * r] = c * apr - s * aqr;
          a[q + n * r] = s * apr + c * aqr;
        }
        for (std::size_t r = 0; r < n; ++r) {
          const double vrp = v[r + n * p];
          const double vrq = v[r + n * q];
          v[r + n * p] = c * vrp - s * vrq;
          v[r + n * q] = s * vrp + c * vrq;
        }
      }
    }
  }
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::sort(order, std::greater{}, [&](const std::size_t i) { return a[i + n * i]; });
  std::vector<double> c(n * rank);
  for (std::size_t i = 0; i < rank; ++i) {
    const double w = a[order[i] + n * order[i]];
    EXPECT_GT(w, 0.0) << "eigenvalue " << i;
    const double scale = std::sqrt(w / 2.0);
    for (std::size_t r = 0; r < n; ++r) {
      c[r + n * i] = v[r + n * order[i]] * scale;
    }
  }
  return c;
}

// C (n x rank, column-major) with D = 2 C C^T by a pivoted Cholesky of D / 2,
// stopped after rank columns: another factor of the same D, unrelated to the
// eigenvectors but by an orthogonal rotation.
std::vector<double> cholesky_factor(const std::vector<double> &d, const std::size_t n,
                                    const std::size_t rank) {
  std::vector<double> residual(d.size());
  std::ranges::transform(d, residual.begin(), [](const double x) { return x / 2.0; });
  std::vector<double> c(n * rank);
  for (std::size_t i = 0; i < rank; ++i) {
    std::size_t pivot = 0;
    for (std::size_t r = 1; r < n; ++r) {
      if (residual[r + n * r] > residual[pivot + n * pivot]) {
        pivot = r;
      }
    }
    const double dp = residual[pivot + n * pivot];
    EXPECT_GT(dp, 0.0) << "pivot " << i;
    for (std::size_t r = 0; r < n; ++r) {
      c[r + n * i] = residual[r + n * pivot] / std::sqrt(dp);
    }
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t p = 0; p < n; ++p) {
        residual[p + n * q] -= c[p + n * i] * c[q + n * i];
      }
    }
  }
  return c;
}

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

// 2 C C^T, n x n column-major.
std::vector<double> density_of(const std::vector<double> &c, const std::size_t n) {
  const std::size_t rank = c.size() / n;
  std::vector<double> d(n * n, 0.0);
  for (std::size_t q = 0; q < n; ++q) {
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t i = 0; i < rank; ++i) {
        d[p + n * q] += 2.0 * c[p + n * i] * c[q + n * i];
      }
    }
  }
  return d;
}

void expect_bitwise_symmetric(const std::vector<double> &k, const std::size_t n,
                              const std::string &what) {
  std::size_t mismatched = 0;
  for (std::size_t q = 0; q < n; ++q) {
    for (std::size_t p = q + 1; p < n; ++p) {
      if (std::bit_cast<std::uint64_t>(k[p + n * q]) !=
          std::bit_cast<std::uint64_t>(k[q + n * p])) {
        ++mismatched;
      }
    }
  }
  EXPECT_EQ(mismatched, 0U) << what << ": K(m, n) and K(n, m) differ in bits";
}

// An n x rank factor, column-major, uniform on [-1/2, 1/2) from a fixed seed.
std::vector<double> random_factor(const std::size_t n, const std::size_t rank,
                                  const std::uint64_t seed) {
  std::mt19937_64 engine{seed};
  std::uniform_real_distribution<double> uniform{-0.5, 0.5};
  std::vector<double> a(n * rank);
  std::ranges::generate(a, [&] { return uniform(engine); });
  return a;
}

// A B^T, n x n column-major, of the n x rank factors a and b.
std::vector<double> product_of(const std::vector<double> &a, const std::vector<double> &b,
                               const std::size_t n) {
  const std::size_t rank = a.size() / n;
  std::vector<double> d(n * n, 0.0);
  for (std::size_t q = 0; q < n; ++q) {
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t i = 0; i < rank; ++i) {
        d[p + n * q] += a[p + n * i] * b[q + n * i];
      }
    }
  }
  return d;
}

// The transpose of the n x n column-major a.
std::vector<double> transpose_of(const std::vector<double> &a, const std::size_t n) {
  std::vector<double> t(a.size());
  for (std::size_t q = 0; q < n; ++q) {
    for (std::size_t p = 0; p < n; ++p) {
      t[q + n * p] = a[p + n * q];
    }
  }
  return t;
}

// K_mn = sum_Q sum_ls B_Q,ml B_Q,ns D_ls on the host, from D itself and not a
// factor of it: per Q, T = B_Q D, then K += T B_Q^T. B is [Q][n][m], B_Q,ml at
// Q + naux (l + nao m), symmetric in m and l.
std::vector<double> fitted_exchange(const std::vector<double> &b, const std::size_t nao,
                                    const std::size_t naux, const std::vector<double> &d) {
  std::vector<double> k(nao * nao, 0.0);
  std::vector<double> t(nao * nao);
  for (std::size_t q = 0; q < naux; ++q) {
    const auto bq = [&](const std::size_t m, const std::size_t l) {
      return b[q + naux * (l + nao * m)];
    };
    for (std::size_t s = 0; s < nao; ++s) {
      for (std::size_t m = 0; m < nao; ++m) {
        double sum = 0.0;
        for (std::size_t l = 0; l < nao; ++l) {
          sum += bq(m, l) * d[l + nao * s];
        }
        t[m + nao * s] = sum;
      }
    }
    for (std::size_t n = 0; n < nao; ++n) {
      for (std::size_t m = 0; m < nao; ++m) {
        double sum = 0.0;
        for (std::size_t s = 0; s < nao; ++s) {
          sum += t[m + nao * s] * bq(n, s);
        }
        k[m + nao * n] += sum;
      }
    }
  }
  return k;
}

// The fitted B on the device, from the golden tensors through density_fit.
class DfJk : public ::testing::Test {
protected:
  void SetUp() override {
    FBF_REQUIRE_GPU();
    g_ = read_golden();
    ASSERT_GT(g_.naux, 0);
    ASSERT_GT(g_.occupied, 0);
    res_.emplace(0);
    metric_.emplace(g_.metric.size(), res_->device());
    b_.emplace(g_.b.size(), res_->device());
    upload(*res_, g_.metric, metric_->data());
    upload(*res_, g_.b, b_->data());
    const fb::Status fitted = fb::density_fit(*res_, g_.nao, g_.naux, metric_->data(), b_->data());
    ASSERT_TRUE(fitted) << fitted.error().message;
  }

  // K of the factor c (nao x occupied), read back.
  std::vector<double> exchange_of(const std::vector<double> &c) {
    fb::Buffers::device<double> d_c(c.size(), res_->device());
    fb::Buffers::device<double> d_k(g_.vk.size(), res_->device());
    upload(*res_, c, d_c.data());
    const fb::Status built =
        fb::df_exchange(*res_, g_.nao, g_.naux, g_.occupied, b_->data(), d_c.data(), d_k.data());
    EXPECT_TRUE(built) << built.error().message;
    return download(*res_, d_k.data(), d_k.num_elements());
  }

  // K of the two-factor density left right^T (each nao x rank), read back.
  std::vector<double> exchange_of(const std::vector<double> &left,
                                  const std::vector<double> &right) {
    const auto rank = static_cast<std::int32_t>(left.size() / static_cast<std::size_t>(g_.nao));
    fb::Buffers::device<double> d_left(left.size(), res_->device());
    fb::Buffers::device<double> d_right(right.size(), res_->device());
    fb::Buffers::device<double> d_k(g_.vk.size(), res_->device());
    upload(*res_, left, d_left.data());
    upload(*res_, right, d_right.data());
    const fb::Status built = fb::df_exchange(*res_, g_.nao, g_.naux, rank, b_->data(),
                                             d_left.data(), d_right.data(), d_k.data());
    EXPECT_TRUE(built) << built.error().message;
    return download(*res_, d_k.data(), d_k.num_elements());
  }

  // The fitted B, read back.
  std::vector<double> fitted_b() { return download(*res_, b_->data(), b_->num_elements()); }

  Golden g_;
  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Buffers::device<double>> metric_;
  std::optional<fb::Buffers::device<double>> b_;
};

TEST_F(DfJk, CoulombMatchesPyscfDfVj) {
  fb::Buffers::device<double> d(g_.density.size(), res_->device());
  fb::Buffers::device<double> j(g_.vj.size(), res_->device());
  upload(*res_, g_.density, d.data());
  const fb::Status built = fb::df_coulomb(*res_, g_.nao, g_.naux, b_->data(), d.data(), j.data());
  ASSERT_TRUE(built) << built.error().message;
  const std::vector<double> got = download(*res_, j.data(), j.num_elements());
  const double scale = max_abs(g_.vj);
  const double worst = max_diff(got, g_.vj);
  ASSERT_GT(scale, 0.0);
  EXPECT_LE(worst, kRelTol * scale)
      << "max|J - vj| = " << worst << ", " << worst / scale << " of max " << scale;
}

// The eigen-factor reproduces D, so K of it is K of the golden density.
TEST_F(DfJk, ExchangeOfTheEigenFactorMatchesPyscfDfVkAndIsSymmetric) {
  const auto nao = static_cast<std::size_t>(g_.nao);
  const auto occupied = static_cast<std::size_t>(g_.occupied);
  const std::vector<double> c = eigen_factor(g_.density, nao, occupied);
  const double d_scale = max_abs(g_.density);
  EXPECT_LE(max_diff(density_of(c, nao), g_.density), 1e-13 * d_scale);

  const std::vector<double> got = exchange_of(c);
  const double scale = max_abs(g_.vk);
  const double worst = max_diff(got, g_.vk);
  ASSERT_GT(scale, 0.0);
  EXPECT_LE(worst, kRelTol * scale)
      << "max|K - vk| = " << worst << ", " << worst / scale << " of max " << scale;
  expect_bitwise_symmetric(got, nao, "eigen factor");
}

// A pivoted-Cholesky factor of the same D: another C, the same K.
TEST_F(DfJk, ExchangeOfTwoFactorsOfOneDensityAgrees) {
  const auto nao = static_cast<std::size_t>(g_.nao);
  const auto occupied = static_cast<std::size_t>(g_.occupied);
  const std::vector<double> eigen = eigen_factor(g_.density, nao, occupied);
  const std::vector<double> cholesky = cholesky_factor(g_.density, nao, occupied);
  const double d_scale = max_abs(g_.density);
  EXPECT_LE(max_diff(density_of(cholesky, nao), g_.density), 1e-13 * d_scale);
  // Not the same factor: the comparison below is not trivially exact.
  EXPECT_GT(max_diff(eigen, cholesky), 1e-3);

  const std::vector<double> k_eigen = exchange_of(eigen);
  const std::vector<double> k_cholesky = exchange_of(cholesky);
  const double scale = max_abs(k_eigen);
  const double worst = max_diff(k_cholesky, k_eigen);
  ASSERT_GT(scale, 0.0);
  EXPECT_LE(worst, kRelTol * scale)
      << "max|K(cholesky) - K(eigen)| = " << worst << ", " << worst / scale << " of max " << scale;
  expect_bitwise_symmetric(k_cholesky, nao, "cholesky factor");
  EXPECT_LE(max_diff(k_cholesky, g_.vk), kRelTol * max_abs(g_.vk));
}

// A random A B^T, neither symmetric nor definite: K by the two-factor
// overload against the host contraction of D itself over the same B, and
// K[A B^T]^T = K[B A^T].
TEST_F(DfJk, ExchangeOfTwoFactorsMatchesTheContractionOverB) {
  const auto nao = static_cast<std::size_t>(g_.nao);
  constexpr std::size_t kRank = 3;
  const std::vector<double> a = random_factor(nao, kRank, 380);
  const std::vector<double> b = random_factor(nao, kRank, 381);
  const std::vector<double> d = product_of(a, b, nao);
  // Not symmetric, so not a density the one-factor form can take.
  EXPECT_GT(max_diff(d, transpose_of(d, nao)), 0.1 * max_abs(d));

  const std::vector<double> expected = fitted_exchange(fitted_b(), nao, g_.naux, d);
  const std::vector<double> got = exchange_of(a, b);
  const double scale = max_abs(expected);
  const double worst = max_diff(got, expected);
  ASSERT_GT(scale, 0.0);
  EXPECT_LE(worst, kRelTol * scale)
      << "max|K - reference| = " << worst << ", " << worst / scale << " of max " << scale;
  EXPECT_GT(max_diff(got, transpose_of(got, nao)), 0.1 * scale) << "K of a non-symmetric D";

  const std::vector<double> swapped = exchange_of(b, a);
  EXPECT_LE(max_diff(swapped, transpose_of(got, nao)), kRelTol * scale)
      << "K[B A^T] is not K[A B^T]^T";
}

// K[C C^T] by the overload is half the one-factor K[2 C C^T]: a gemm against a
// syrk over the same X, so equal to rounding.
TEST_F(DfJk, ExchangeOfOneFactorTwiceIsTheOneFactorExchange) {
  const auto nao = static_cast<std::size_t>(g_.nao);
  const auto occupied = static_cast<std::size_t>(g_.occupied);
  const std::vector<double> c = eigen_factor(g_.density, nao, occupied);
  const std::vector<double> one = exchange_of(c);
  std::vector<double> twice = exchange_of(c, c);
  std::ranges::transform(twice, twice.begin(), [](const double x) { return 2.0 * x; });
  const double scale = max_abs(one);
  const double worst = max_diff(twice, one);
  ASSERT_GT(scale, 0.0);
  EXPECT_LE(worst, kRoundingTol * scale)
      << "max|2 K[C C^T] - K(C)| = " << worst << ", " << worst / scale << " of max " << scale;
  std::cout << std::format("[ df jk    ] max|2 K[C C^T] - df_exchange(C)| / max {:.3e}\n",
                           worst / scale);
}

// The same A B^T over this tree's own basis and B (the DensityFitting step's),
// against integral-direct exchange: S = (D + D^T) / 2 and the antisymmetric
// rest each by its own sweep, unscreened, carried to the spherical layout.
// Their difference is the fitting error, not rounding.
TEST(DfJkDirect, ExchangeOfTwoFactorsMatchesIntegralDirectWithinTheFittingError) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(kDataDir / "density_fitting.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const fb::json *c = nullptr;
  for (const fb::json &entry : (*doc)["integrals"]) {
    if (entry["name"].get<std::string>() == "h2o_ccpvdz") {
      c = &entry;
    }
  }
  ASSERT_NE(c, nullptr);

  fb::DeviceResources res{0};
  std::vector<fb::Atom> atoms;
  for (const auto &atom : (*c)["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  const fb::MoleculeOutput molecule{fb::Molecule(res, std::move(atoms))};
  auto basis = fb::BasisSetStep{.basis = (*c)["basis"].get<std::string>()}.run(
      fb::View<fb::MoleculeOutput>{molecule});
  ASSERT_TRUE(basis) << basis.error().message;
  auto classes = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  ASSERT_TRUE(classes) << classes.error().message;
  auto fit = fb::DensityFittingStep{.aux_basis = (*c)["aux_basis"].get<std::string>()}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
          molecule, *basis, *classes},
      res);
  ASSERT_TRUE(fit) << fit.error().message;
  auto k_classes = fb::ordered_pair_classes(res, basis->basis, classes->classes);
  ASSERT_TRUE(k_classes) << k_classes.error().message;
  auto u = fb::upload_spherical_transform_matrix(res, basis->basis);
  ASSERT_TRUE(u) << u.error().message;

  const std::int32_t nao = static_cast<std::int32_t>(basis->basis.nao_sphe());
  const auto n = static_cast<std::size_t>(nao);
  constexpr std::size_t kRank = 3;
  const std::vector<double> a = random_factor(n, kRank, 380);
  const std::vector<double> b = random_factor(n, kRank, 381);
  const std::vector<double> d = product_of(a, b, n);

  fb::Buffers::device<double> d_a(a.size(), res.device());
  fb::Buffers::device<double> d_b(b.size(), res.device());
  fb::Buffers::device<double> d_k(n * n, res.device());
  upload(res, a, d_a.data());
  upload(res, b, d_b.data());
  const fb::Status built = fb::df_exchange(res, nao, fit->naux, static_cast<std::int32_t>(kRank),
                                           fit->b.data(), d_a.data(), d_b.data(), d_k.data());
  ASSERT_TRUE(built) << built.error().message;
  const std::vector<double> got = download(res, d_k.data(), d_k.num_elements());

  // K of sign (D + sign D^T) / 2, by one sweep of that symmetry.
  const std::vector<double> dt = transpose_of(d, n);
  const auto direct = [&](const double sign, const fb::DensitySymmetry symmetry) {
    std::vector<double> part(d.size());
    for (std::size_t i = 0; i < d.size(); ++i) {
      part[i] = (d[i] + sign * dt[i]) / 2.0;
    }
    fb::MatrixAOPair density(res, basis->basis, *u);
    fb::MatrixAOPair k(res, basis->basis, *u);
    upload(res, part, density.spherical().data());
    EXPECT_TRUE(density.spherical_to_cartesian(res));
    const fb::Status swept =
        fb::exchange(res, *k_classes, density.cartesian(), k.cartesian(), 0.0, symmetry);
    EXPECT_TRUE(swept) << swept.error().message;
    EXPECT_TRUE(k.cartesian_to_spherical(res));
    return download(res, k.spherical().data(), k.spherical().num_elements());
  };
  const std::vector<double> symmetric = direct(1.0, fb::DensitySymmetry::Symmetric);
  const std::vector<double> antisymmetric = direct(-1.0, fb::DensitySymmetry::Antisymmetric);
  std::vector<double> expected(d.size());
  std::ranges::transform(symmetric, antisymmetric, expected.begin(), std::plus{});

  const double scale = max_abs(expected);
  const double worst = max_diff(got, expected);
  ASSERT_GT(scale, 0.0);
  EXPECT_LE(worst, kFittingTol * scale)
      << "max|K_df - K_direct| = " << worst << ", " << worst / scale << " of max " << scale;
  std::cout << std::format("[ df jk    ] two-factor K against integral-direct: max {:.4f}, worst "
                           "|K_df - K_direct| / max {:.3e}\n",
                           scale, worst / scale);
}

// naux = 0 or nocc (rank) = 0 is J or K = 0, written over whatever was there;
// nao = 0 writes nothing; a negative count, or naux nocc (rank) past int32_t,
// is InvalidConfig. Both K forms alike.
TEST(DfJkEmpty, EmptyCountsAndNegativeCounts) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kNao = 3;
  const std::vector<double> ones(kNao * kNao, 1.0);
  fb::Buffers::device<double> out(ones.size(), res.device());

  upload(res, ones, out.data());
  ASSERT_TRUE(fb::df_coulomb(res, kNao, 0, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), std::vector<double>(ones.size(), 0.0));
  upload(res, ones, out.data());
  ASSERT_TRUE(fb::df_exchange(res, kNao, 0, 2, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), std::vector<double>(ones.size(), 0.0));
  upload(res, ones, out.data());
  ASSERT_TRUE(fb::df_exchange(res, kNao, 4, 0, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), std::vector<double>(ones.size(), 0.0));

  upload(res, ones, out.data());
  ASSERT_TRUE(fb::df_exchange(res, kNao, 0, 2, nullptr, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), std::vector<double>(ones.size(), 0.0));
  upload(res, ones, out.data());
  ASSERT_TRUE(fb::df_exchange(res, kNao, 4, 0, nullptr, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), std::vector<double>(ones.size(), 0.0));

  EXPECT_TRUE(fb::df_coulomb(res, 0, 4, nullptr, nullptr, nullptr));
  EXPECT_TRUE(fb::df_exchange(res, 0, 4, 2, nullptr, nullptr, nullptr));
  EXPECT_TRUE(fb::df_exchange(res, 0, 4, 2, nullptr, nullptr, nullptr, nullptr));
  constexpr std::int32_t kHuge = 1 << 16; // kHuge^2 = 2^32
  for (const fb::Status refused :
       {fb::df_coulomb(res, -1, 4, nullptr, nullptr, nullptr),
        fb::df_coulomb(res, 3, -1, nullptr, nullptr, nullptr),
        fb::df_exchange(res, 3, 4, -1, nullptr, nullptr, nullptr),
        fb::df_exchange(res, 3, kHuge, kHuge, nullptr, nullptr, nullptr),
        fb::df_exchange(res, -1, 4, 2, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange(res, 3, -1, 2, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange(res, 3, 4, -1, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange(res, 3, kHuge, kHuge, nullptr, nullptr, nullptr, nullptr)}) {
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
  }
}

} // namespace
