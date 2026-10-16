// GPU cases for linalg.diis: the extrapolation against a host solve of the
// same bordered system, through the ring's wrap-around; the start-up,
// singular and reset paths; and DIIS on a linear fixed point, where it is
// GMRES and must reach the solution in at most len + 1 steps.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.linalg;

namespace {

namespace fb = free_breakfast;

using HostVector = std::vector<double>;

double dot(const HostVector &a, const HostVector &b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    s += a[i] * b[i];
  }
  return s;
}

// The reference: solve the bordered system for the pairs in age order by
// dense Gaussian elimination, and return sum_i c_i F_i.
HostVector host_extrapolate(const std::deque<HostVector> &focks,
                            const std::deque<HostVector> &residuals) {
  const std::size_t m = focks.size();
  const std::size_t dim = m + 1;
  std::vector<double> b(dim * dim, -1.0);
  std::vector<double> rhs(dim, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      b[(i * dim) + j] = dot(residuals[i], residuals[j]);
    }
  }
  b[(m * dim) + m] = 0.0;
  rhs[m] = -1.0;
  for (std::size_t col = 0; col < dim; ++col) {
    std::size_t pivot = col;
    for (std::size_t r = col + 1; r < dim; ++r) {
      if (std::abs(b[(r * dim) + col]) > std::abs(b[(pivot * dim) + col])) {
        pivot = r;
      }
    }
    for (std::size_t j = 0; j < dim; ++j) {
      std::swap(b[(col * dim) + j], b[(pivot * dim) + j]);
    }
    std::swap(rhs[col], rhs[pivot]);
    for (std::size_t r = col + 1; r < dim; ++r) {
      const double factor = b[(r * dim) + col] / b[(col * dim) + col];
      for (std::size_t j = col; j < dim; ++j) {
        b[(r * dim) + j] -= factor * b[(col * dim) + j];
      }
      rhs[r] -= factor * rhs[col];
    }
  }
  std::vector<double> c(dim, 0.0);
  for (std::size_t k = dim; k-- > 0;) {
    double s = rhs[k];
    for (std::size_t j = k + 1; j < dim; ++j) {
      s -= b[(k * dim) + j] * c[j];
    }
    c[k] = s / b[(k * dim) + k];
  }
  HostVector out(focks.front().size(), 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = 0; k < out.size(); ++k) {
      out[k] += c[i] * focks[i][k];
    }
  }
  return out;
}

// Device staging for one Diis: the current pair and the output.
struct Staging {
  fb::DeviceResources &res;
  std::size_t len;
  fb::Buffers::device<double> fock{len, res.device()};
  fb::Buffers::device<double> residual{len, res.device()};
  fb::Buffers::device<double> out{len, res.device()};

  void upload(fb::Buffers::device<double> &d, const HostVector &host) {
    ASSERT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), len * sizeof(double),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              wwr::wwrSuccess);
    ASSERT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  }

  HostVector download(const fb::Buffers::device<double> &d) {
    HostVector host(len);
    EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), len * sizeof(double),
                                  wwr::wwrMemcpyDeviceToHost, res.stream()),
              wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
    return host;
  }

  fb::Status push(fb::Diis &diis, const HostVector &f, const HostVector &e) {
    upload(fock, f);
    upload(residual, e);
    return diis.push_and_extrapolate(res, fock.data(), residual.data(), out.data());
  }
};

HostVector random_vector(std::mt19937 &gen, std::size_t len) {
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  HostVector v(len);
  for (auto &x : v) {
    x = dist(gen);
  }
  return v;
}

TEST(Diis, CapacityIsClampedToOneThroughTheLimit) {
  FBF_REQUIRE_GPU();

  fb::DeviceResources res{0};
  EXPECT_EQ(fb::Diis(res, 4, 0).capacity(), 1U);
  EXPECT_EQ(fb::Diis(res, 4, 6).capacity(), 6U);
  EXPECT_EQ(fb::Diis(res, 4, 1000).capacity(), fb::kDiisMaxHistoryLimit);
}

TEST(Diis, FirstPushLeavesTheOutputUntouched) {
  FBF_REQUIRE_GPU();

  constexpr std::size_t kLen = 9;
  fb::DeviceResources res{0};
  fb::Diis diis{res, kLen, 4};
  Staging io{res, kLen};
  const HostVector sentinel(kLen, 42.0);
  io.upload(io.out, sentinel);

  std::mt19937 gen{7};
  ASSERT_TRUE(io.push(diis, random_vector(gen, kLen), random_vector(gen, kLen)));
  EXPECT_EQ(diis.size(), 1U);
  EXPECT_FALSE(diis.last_extrapolation_singular());
  EXPECT_EQ(io.download(io.out), sentinel);
}

// Eleven pushes into a ring of four: the history fills at the fourth and then
// wraps nearly twice, so the Gram and coefficients are checked in rotated
// physical order against a reference that keeps age order.
TEST(Diis, ExtrapolationMatchesAHostSolveThroughTheRingWrap) {
  FBF_REQUIRE_GPU();

  constexpr std::size_t kLen = 37;
  constexpr std::size_t kCap = 4;
  fb::DeviceResources res{0};
  fb::Diis diis{res, kLen, kCap};
  Staging io{res, kLen};

  std::mt19937 gen{2024};
  std::deque<HostVector> focks;
  std::deque<HostVector> residuals;
  for (std::size_t step = 0; step < 11; ++step) {
    const auto f = random_vector(gen, kLen);
    const auto e = random_vector(gen, kLen);
    focks.push_back(f);
    residuals.push_back(e);
    if (focks.size() > kCap) {
      focks.pop_front();
      residuals.pop_front();
    }
    ASSERT_TRUE(io.push(diis, f, e));
    EXPECT_EQ(diis.size(), focks.size());
    if (focks.size() < 2) {
      continue;
    }
    EXPECT_FALSE(diis.last_extrapolation_singular()) << "step " << step;
    const auto expected = host_extrapolate(focks, residuals);
    const auto actual = io.download(io.out);
    for (std::size_t k = 0; k < kLen; ++k) {
      EXPECT_NEAR(actual[k], expected[k], 1e-11) << "step " << step << ", element " << k;
    }
  }
}

// Two equal residuals make the Gram rank one, so the bordered system is
// singular and the output is the newest Fock.
TEST(Diis, SingularSubspaceFallsBackToTheNewestFock) {
  FBF_REQUIRE_GPU();

  constexpr std::size_t kLen = 16;
  fb::DeviceResources res{0};
  fb::Diis diis{res, kLen, 4};
  Staging io{res, kLen};

  std::mt19937 gen{99};
  const auto e = random_vector(gen, kLen);
  ASSERT_TRUE(io.push(diis, random_vector(gen, kLen), e));
  const auto newest = random_vector(gen, kLen);
  ASSERT_TRUE(io.push(diis, newest, e));
  EXPECT_TRUE(diis.last_extrapolation_singular());
  EXPECT_EQ(io.download(io.out), newest);
}

TEST(Diis, ResetDropsTheSubspace) {
  FBF_REQUIRE_GPU();

  constexpr std::size_t kLen = 8;
  fb::DeviceResources res{0};
  fb::Diis diis{res, kLen, 3};
  Staging io{res, kLen};

  std::mt19937 gen{5};
  for (std::size_t i = 0; i < 5; ++i) {
    ASSERT_TRUE(io.push(diis, random_vector(gen, kLen), random_vector(gen, kLen)));
  }
  EXPECT_EQ(diis.size(), 3U);

  diis.reset();
  EXPECT_EQ(diis.size(), 0U);
  const HostVector sentinel(kLen, -3.0);
  io.upload(io.out, sentinel);
  std::deque<HostVector> focks{random_vector(gen, kLen)};
  std::deque<HostVector> residuals{random_vector(gen, kLen)};
  ASSERT_TRUE(io.push(diis, focks[0], residuals[0]));
  EXPECT_EQ(io.download(io.out), sentinel);

  // The next push extrapolates over the two pairs since the reset and no
  // others: the Gram entries from before it are never read.
  focks.push_back(random_vector(gen, kLen));
  residuals.push_back(random_vector(gen, kLen));
  ASSERT_TRUE(io.push(diis, focks[1], residuals[1]));
  const auto expected = host_extrapolate(focks, residuals);
  const auto actual = io.download(io.out);
  for (std::size_t k = 0; k < kLen; ++k) {
    EXPECT_NEAR(actual[k], expected[k], 1e-12) << "element " << k;
  }
}

// x <- A x + b with F = A x + b and e = F - x. On a linear map DIIS is GMRES,
// so with the whole history kept it reaches the fixed point in at most len + 1
// steps, where plain iteration with spectral radius 0.9 is still far off.
TEST(Diis, LinearFixedPointConvergesWithinLenPlusOneSteps) {
  FBF_REQUIRE_GPU();

  constexpr std::size_t kLen = 6;
  // A = Q diag(lambda) Q^T, Q a Householder reflection, so the spectrum is
  // exactly lambda.
  const std::array<double, kLen> lambda{0.9, -0.8, 0.5, 0.3, -0.2, 0.7};
  std::mt19937 gen{11};
  auto v = random_vector(gen, kLen);
  const double vv = dot(v, v);
  std::vector<double> a(kLen * kLen, 0.0);
  for (std::size_t i = 0; i < kLen; ++i) {
    for (std::size_t j = 0; j < kLen; ++j) {
      for (std::size_t k = 0; k < kLen; ++k) {
        const double qik = (i == k ? 1.0 : 0.0) - (2.0 * v[i] * v[k] / vv);
        const double qjk = (j == k ? 1.0 : 0.0) - (2.0 * v[j] * v[k] / vv);
        a[(i * kLen) + j] += qik * lambda[k] * qjk;
      }
    }
  }
  const auto b = random_vector(gen, kLen);
  const auto apply = [&](const HostVector &x) {
    HostVector y = b;
    for (std::size_t i = 0; i < kLen; ++i) {
      for (std::size_t j = 0; j < kLen; ++j) {
        y[i] += a[(i * kLen) + j] * x[j];
      }
    }
    return y;
  };

  fb::DeviceResources res{0};
  fb::Diis diis{res, kLen, kLen + 2};
  Staging io{res, kLen};

  HostVector x(kLen, 0.0);
  double residual_norm = 0.0;
  for (std::size_t step = 0; step <= kLen + 1; ++step) {
    const auto f = apply(x);
    HostVector e(kLen);
    for (std::size_t i = 0; i < kLen; ++i) {
      e[i] = f[i] - x[i];
    }
    residual_norm = std::sqrt(dot(e, e));
    if (residual_norm < 1e-10) {
      break;
    }
    ASSERT_TRUE(io.push(diis, f, e));
    x = diis.size() < 2 ? f : io.download(io.out);
  }
  EXPECT_LT(residual_norm, 1e-10);
}

// out may alias the current Fock: it is copied into the history first.
TEST(Diis, OutputMayAliasTheCurrentFock) {
  FBF_REQUIRE_GPU();

  constexpr std::size_t kLen = 12;
  fb::DeviceResources res{0};
  fb::Diis diis{res, kLen, 3};
  Staging io{res, kLen};

  std::mt19937 gen{3};
  std::deque<HostVector> focks;
  std::deque<HostVector> residuals;
  for (std::size_t i = 0; i < 3; ++i) {
    focks.push_back(random_vector(gen, kLen));
    residuals.push_back(random_vector(gen, kLen));
    io.upload(io.fock, focks.back());
    io.upload(io.residual, residuals.back());
    ASSERT_TRUE(diis.push_and_extrapolate(res, io.fock.data(), io.residual.data(), io.fock.data()));
  }
  const auto expected = host_extrapolate(focks, residuals);
  const auto actual = io.download(io.fock);
  for (std::size_t k = 0; k < kLen; ++k) {
    EXPECT_NEAR(actual[k], expected[k], 1e-12) << "element " << k;
  }
}

} // namespace
