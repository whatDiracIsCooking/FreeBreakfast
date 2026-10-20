// GPU cases for linalg.orthogonalize: X = S^{-1/2} against operands whose
// answer is known -- a diagonal S (X is the elementwise inverse root), a dense
// well-conditioned S (X^T S X == I), and a singular S (the null mode is
// dropped, and X stays finite).

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.linalg;

namespace {

namespace fb = free_breakfast;

// Column-major n x n on the host, element (i, j) at i + j*n.
template <typename T>
using HostMatrix = std::vector<T>;

template <typename T>
HostMatrix<T> matmul(const HostMatrix<T> &a, const HostMatrix<T> &b, std::int32_t n,
                     bool transpose_a = false) {
  HostMatrix<T> c(a.size(), T{0});
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t k = 0; k < n; ++k) {
      const T bkj = b[k + j * n];
      for (std::int32_t i = 0; i < n; ++i) {
        const T aik = transpose_a ? a[k + i * n] : a[i + k * n];
        c[i + j * n] += aik * bkj;
      }
    }
  }
  return c;
}

// A A^T + n I: symmetric, positive definite, and well-conditioned.
template <typename T>
HostMatrix<T> well_conditioned_spd(std::int32_t n) {
  std::mt19937 gen{12345};
  std::uniform_real_distribution<T> dist{T{-1}, T{1}};
  HostMatrix<T> a(static_cast<std::size_t>(n) * n);
  for (auto &v : a) {
    v = dist(gen);
  }
  HostMatrix<T> s(a.size(), T{0});
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = 0; i < n; ++i) {
      for (std::int32_t k = 0; k < n; ++k) {
        s[i + j * n] += a[i + k * n] * a[j + k * n];
      }
    }
    s[j + j * n] += static_cast<T>(n);
  }
  return s;
}

// orthogonalize's operand and result for one n, plus host staging. Every
// intermediate is res's.
template <typename T>
struct Orthogonalizer {
  fb::DeviceResources &res;
  std::int32_t n;
  std::size_t nn = static_cast<std::size_t>(n) * n;
  fb::Buffers::device<T> s{nn, res.device()};
  fb::Buffers::device<T> x{nn, res.device()};

  void upload(const HostMatrix<T> &host) {
    ASSERT_EQ(wwr::wwrMemcpyAsync(s.data(), host.data(), nn * sizeof(T), wwr::wwrMemcpyHostToDevice,
                                  res.stream()),
              wwr::wwrSuccess);
    ASSERT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  }

  HostMatrix<T> download(const fb::Buffers::device<T> &d) {
    HostMatrix<T> host(nn);
    EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), nn * sizeof(T), wwr::wwrMemcpyDeviceToHost,
                                  res.stream()),
              wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
    return host;
  }

  fb::Status operator()(T floor = fb::kOrthogonalizeEigenvalueFloor<T>) {
    return fb::orthogonalize<T>(res, n, s.data(), x.data(), floor);
  }
};

TEST(Orthogonalize, EmptyOperandSucceedsWithoutTouchingAnything) {
  FBF_REQUIRE_GPU();

  fb::DeviceResources res{0};
  EXPECT_TRUE(fb::orthogonalize<double>(res, 0, nullptr, nullptr));
  EXPECT_EQ(res.device_workspace(0).num_elements(), 0U);
}

// Every intermediate comes from res's workspace, which grows with n: one DeviceResources serves
// a small operand and then a larger one, in either precision.
TEST(Orthogonalize, GrowsTheResourcesWorkspaceWithTheOperand) {
  FBF_REQUIRE_GPU();

  fb::DeviceResources res{0};

  Orthogonalizer<float> small{res, 4};
  small.upload(well_conditioned_spd<float>(4));
  ASSERT_TRUE(small());
  const std::size_t small_bytes = res.device_workspace(0).num_elements();
  EXPECT_GT(small_bytes, 0U);

  Orthogonalizer<double> large{res, 64};
  large.upload(well_conditioned_spd<double>(64));
  ASSERT_TRUE(large());
  EXPECT_GT(res.device_workspace(0).num_elements(), small_bytes);
}

// A diagonal S's eigenvectors are the unit vectors, so X = diag(d^{-1/2}).
TEST(Orthogonalize, DiagonalOperandGivesTheInverseRootOfEachEntry) {
  FBF_REQUIRE_GPU();

  constexpr std::int32_t kN = 5;
  const std::array<double, kN> d{0.25, 1.0, 4.0, 9.0, 100.0};
  HostMatrix<double> s(kN * kN, 0.0);
  for (std::int32_t i = 0; i < kN; ++i) {
    s[i + i * kN] = d[static_cast<std::size_t>(i)];
  }

  fb::DeviceResources res{0};
  Orthogonalizer<double> run{res, kN};
  run.upload(s);
  ASSERT_TRUE(run());

  const auto x = run.download(run.x);
  for (std::int32_t j = 0; j < kN; ++j) {
    for (std::int32_t i = 0; i < kN; ++i) {
      const double expected = i == j ? 1.0 / std::sqrt(d[static_cast<std::size_t>(i)]) : 0.0;
      EXPECT_NEAR(x[i + j * kN], expected, 1e-14) << "at (" << i << ", " << j << ")";
    }
  }
}

template <typename T>
void check_dense_orthogonalizer(T tolerance) {
  constexpr std::int32_t kN = 37; // not a multiple of any block size
  const auto s = well_conditioned_spd<T>(kN);

  fb::DeviceResources res{0};
  Orthogonalizer<T> run{res, kN};
  run.upload(s);
  ASSERT_TRUE(run());

  const auto x = run.download(run.x);
  // S itself is left untouched, though syevd overwrites its input.
  EXPECT_EQ(run.download(run.s), s);

  // X is symmetric, and X^T S X == I.
  const auto xtsx = matmul(x, matmul(s, x, kN), kN, /*transpose_a=*/true);
  for (std::int32_t j = 0; j < kN; ++j) {
    for (std::int32_t i = 0; i < kN; ++i) {
      EXPECT_NEAR(x[i + j * kN], x[j + i * kN], tolerance)
          << "X not symmetric at " << i << ", " << j;
      EXPECT_NEAR(xtsx[i + j * kN], i == j ? T{1} : T{0}, tolerance)
          << "X^T S X at (" << i << ", " << j << ")";
    }
  }
}

TEST(Orthogonalize, DenseOperandDouble) {
  FBF_REQUIRE_GPU();
  check_dense_orthogonalizer<double>(1e-12);
}

TEST(Orthogonalize, DenseOperandFloat) {
  FBF_REQUIRE_GPU();
  check_dense_orthogonalizer<float>(1e-4F);
}

// S = v v^T + w w^T in 3 dimensions has one zero eigenvalue. Its mode is
// dropped, so X is finite and X^T S X is a rank-2 projector, not I.
TEST(Orthogonalize, SingularOperandDropsTheNullModeAndStaysFinite) {
  FBF_REQUIRE_GPU();

  constexpr std::int32_t kN = 3;
  const std::array<double, kN> v{1.0, 2.0, 0.5};
  const std::array<double, kN> w{-1.0, 0.5, 3.0};
  HostMatrix<double> s(kN * kN);
  for (std::int32_t j = 0; j < kN; ++j) {
    for (std::int32_t i = 0; i < kN; ++i) {
      const auto ui = static_cast<std::size_t>(i);
      const auto uj = static_cast<std::size_t>(j);
      s[i + j * kN] = v[ui] * v[uj] + w[ui] * w[uj];
    }
  }

  fb::DeviceResources res{0};
  Orthogonalizer<double> run{res, kN};
  run.upload(s);
  // A floor well above the rounding-level eigenvalue syevd returns for the
  // null mode.
  ASSERT_TRUE(run(1e-8));

  const auto x = run.download(run.x);
  for (const double e : x) {
    EXPECT_TRUE(std::isfinite(e));
  }

  // P = X^T S X: idempotent with trace 2.
  const auto p = matmul(x, matmul(s, x, kN), kN, /*transpose_a=*/true);
  const auto pp = matmul(p, p, kN);
  double trace = 0.0;
  for (std::int32_t i = 0; i < kN; ++i) {
    trace += p[i + i * kN];
  }
  EXPECT_NEAR(trace, 2.0, 1e-10);
  for (std::size_t k = 0; k < p.size(); ++k) {
    EXPECT_NEAR(pp[k], p[k], 1e-10);
  }
}

} // namespace
