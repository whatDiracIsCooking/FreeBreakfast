// GPU cases for linalg.diagonalize: A = U diag(w) U^T against operands whose
// answer is known -- a diagonal A (w is its sorted diagonal), and a dense
// symmetric A (U orthonormal, U diag(w) U^T == A, w ascending).

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

// A symmetric matrix with entries in [-1, 1].
template <typename T>
HostMatrix<T> random_symmetric(std::int32_t n) {
  std::mt19937 gen{2468};
  std::uniform_real_distribution<T> dist{T{-1}, T{1}};
  HostMatrix<T> a(static_cast<std::size_t>(n) * n);
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = j; i < n; ++i) {
      a[i + j * n] = a[j + i * n] = dist(gen);
    }
  }
  return a;
}

// diagonalize's operand and eigenvalues for one n, plus host staging.
template <typename T>
struct Diagonalizer {
  fb::DeviceResources &res;
  std::int32_t n;
  std::size_t nn = static_cast<std::size_t>(n) * n;
  fb::Buffers::device<T> a{nn, res.device()};
  fb::Buffers::device<T> w{static_cast<std::size_t>(n), res.device()};

  void upload(const HostMatrix<T> &host) {
    ASSERT_EQ(wwr::wwrMemcpyAsync(a.data(), host.data(), nn * sizeof(T), wwr::wwrMemcpyHostToDevice,
                                  res.stream()),
              wwr::wwrSuccess);
    ASSERT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  }

  std::vector<T> download(const fb::Buffers::device<T> &d) {
    std::vector<T> host(d.num_elements());
    EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), host.size() * sizeof(T),
                                  wwr::wwrMemcpyDeviceToHost, res.stream()),
              wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
    return host;
  }

  fb::Status operator()() { return fb::diagonalize<T>(res, n, a.data(), w.data()); }
};

TEST(Diagonalize, EmptyOperandSucceedsWithoutTouchingAnything) {
  FBF_REQUIRE_GPU();

  fb::DeviceResources res{0};
  EXPECT_TRUE(fb::diagonalize<double>(res, 0, nullptr, nullptr));
  EXPECT_EQ(res.device_workspace(0).num_elements(), 0U);
}

// syevd's work is res's workspace: the call grows it from empty.
TEST(Diagonalize, TakesSyevdsWorkFromTheResourcesWorkspace) {
  FBF_REQUIRE_GPU();

  fb::DeviceResources res{0};
  Diagonalizer<float> run{res, 16};
  run.upload(random_symmetric<float>(16));
  ASSERT_TRUE(run());
  EXPECT_GT(res.device_workspace(0).num_elements(), 0U);
}

// A diagonal A's eigenvalues are its diagonal, sorted, and its eigenvectors
// the unit vectors up to sign.
TEST(Diagonalize, DiagonalOperandGivesItsSortedDiagonal) {
  FBF_REQUIRE_GPU();

  constexpr std::int32_t kN = 5;
  const std::array<double, kN> d{4.0, -1.0, 9.0, 0.25, 2.0};
  HostMatrix<double> a(kN * kN, 0.0);
  for (std::int32_t i = 0; i < kN; ++i) {
    a[i + i * kN] = d[static_cast<std::size_t>(i)];
  }

  fb::DeviceResources res{0};
  Diagonalizer<double> run{res, kN};
  run.upload(a);
  ASSERT_TRUE(run());

  auto sorted = d;
  std::ranges::sort(sorted);
  const auto w = run.download(run.w);
  const auto u = run.download(run.a);
  for (std::int32_t j = 0; j < kN; ++j) {
    const double wj = w[static_cast<std::size_t>(j)];
    EXPECT_NEAR(wj, sorted[static_cast<std::size_t>(j)], 1e-14) << "w[" << j << "]";
    // Column j is the unit vector of the diagonal entry equal to w[j].
    for (std::int32_t i = 0; i < kN; ++i) {
      const double expected =
          d[static_cast<std::size_t>(i)] == sorted[static_cast<std::size_t>(j)] ? 1.0 : 0.0;
      EXPECT_NEAR(std::abs(u[i + j * kN]), expected, 1e-14) << "at (" << i << ", " << j << ")";
    }
  }
}

template <typename T>
void check_dense_decomposition(T tolerance) {
  constexpr std::int32_t kN = 37; // not a multiple of any block size
  const auto a = random_symmetric<T>(kN);

  fb::DeviceResources res{0};
  Diagonalizer<T> run{res, kN};
  run.upload(a);
  ASSERT_TRUE(run());

  const auto w = run.download(run.w);
  const auto u = run.download(run.a);
  EXPECT_TRUE(std::ranges::is_sorted(w));

  // U^T U == I and U diag(w) U^T == A.
  for (std::int32_t j = 0; j < kN; ++j) {
    for (std::int32_t i = 0; i < kN; ++i) {
      T utu = T{0};
      T udut = T{0};
      for (std::int32_t k = 0; k < kN; ++k) {
        utu += u[k + i * kN] * u[k + j * kN];
        udut += u[i + k * kN] * w[static_cast<std::size_t>(k)] * u[j + k * kN];
      }
      EXPECT_NEAR(utu, i == j ? T{1} : T{0}, tolerance) << "U^T U at (" << i << ", " << j << ")";
      EXPECT_NEAR(udut, a[i + j * kN], tolerance) << "U diag(w) U^T at (" << i << ", " << j << ")";
    }
  }
}

TEST(Diagonalize, DenseOperandDouble) {
  FBF_REQUIRE_GPU();
  check_dense_decomposition<double>(1e-12);
}

TEST(Diagonalize, DenseOperandFloat) {
  FBF_REQUIRE_GPU();
  check_dense_decomposition<float>(1e-4F);
}

} // namespace
