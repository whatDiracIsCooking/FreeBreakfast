// GPU cases for linalg/blas's :gemm: the three gemm overloads against a host
// reference, with non-symmetric operands (so a swapped or wrongly transposed
// operand cannot pass) and a non-zero beta (so C's prior contents must be
// accumulated, not overwritten), through both a BlasHandle and a raw
// wwrblasHandle_t.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.linalg.blas;

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;

constexpr std::int32_t kN = 37; // odd and not a tile multiple

// Column-major n x n on the host, element (i, j) at i + j*n.
template <typename T>
std::vector<T> random_matrix(std::int32_t n, std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<T> dist{T{-1}, T{1}};
  std::vector<T> m(static_cast<std::size_t>(n) * n);
  for (auto &v : m) {
    v = dist(gen);
  }
  return m;
}

// Element (i, j) of op(m): m itself, or its transpose. Real T only, so _C is _T.
template <typename T>
T op_at(const std::vector<T> &m, bool transpose, std::int32_t i, std::int32_t j, std::int32_t n) {
  return transpose ? m[j + i * n] : m[i + j * n];
}

// alpha * op(A) * op(B) + beta * C.
template <typename T>
std::vector<T> reference(bool trans_a, bool trans_b, T alpha, const std::vector<T> &a,
                         const std::vector<T> &b, T beta, const std::vector<T> &c, std::int32_t n) {
  std::vector<T> out(c.size());
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = 0; i < n; ++i) {
      T sum{0};
      for (std::int32_t k = 0; k < n; ++k) {
        sum += op_at(a, trans_a, i, k, n) * op_at(b, trans_b, k, j, n);
      }
      out[i + j * n] = alpha * sum + beta * c[i + j * n];
    }
  }
  return out;
}

// A, B and C on the device, with host staging for upload and download.
template <typename T>
struct Operands {
  std::shared_ptr<fb::Device> dev;
  std::size_t nn = static_cast<std::size_t>(kN) * kN;
  fb::Buffers::pinned<T> stage{nn};
  fb::Buffers::device<T> a{nn, dev};
  fb::Buffers::device<T> b{nn, dev};
  fb::Buffers::device<T> c{nn, dev};

  void upload(fb::Buffers::device<T> &d, const std::vector<T> &host) {
    std::ranges::copy(host, stage.data());
    ASSERT_EQ(ext::copy(d, stage, dev->stream()), wwr::wwrSuccess);
    ASSERT_EQ(ext::sync(dev->stream()), wwr::wwrSuccess);
  }

  std::vector<T> download_c() {
    EXPECT_EQ(ext::copy(stage, c, dev->stream()), wwr::wwrSuccess);
    EXPECT_EQ(ext::sync(dev->stream()), wwr::wwrSuccess);
    return {stage.data(), stage.data() + nn};
  }
};

template <typename T>
void expect_matches(const std::vector<T> &got, const std::vector<T> &expected) {
  const T tolerance = std::is_same_v<T, float> ? T{1e-4} : T{1e-12};
  for (std::int32_t j = 0; j < kN; ++j) {
    for (std::int32_t i = 0; i < kN; ++i) {
      ASSERT_NEAR(got[i + j * kN], expected[i + j * kN], tolerance)
          << "at (" << i << ", " << j << ")";
    }
  }
}

template <typename T>
class Gemm : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Gemm, RealTypes);

// The untransposed overload, through both arms of blas_handle: the wrapper
// itself, and the raw handle it holds.
TYPED_TEST(Gemm, UntransposedMatchesHostReference) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Operands<T> ops{.dev = dev};

  const auto a = random_matrix<T>(kN, 1);
  const auto b = random_matrix<T>(kN, 2);
  const auto c = random_matrix<T>(kN, 3);
  const T alpha{1.5};
  const T beta{-0.5};
  const auto expected = reference(false, false, alpha, a, b, beta, c, kN);

  ops.upload(ops.a, a);
  ops.upload(ops.b, b);

  ops.upload(ops.c, c);
  ASSERT_EQ(fb::gemm(blas, kN, &alpha, ops.a.data(), ops.b.data(), &beta, ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.download_c(), expected);

  ops.upload(ops.c, c);
  ASSERT_EQ(fb::gemm(blas.get(), kN, &alpha, ops.a.data(), ops.b.data(), &beta, ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.download_c(), expected);
}

// The op overload, over every N/T combination on each side. The (N, N) case
// pins that the untransposed overload is exactly this one with OP_N twice.
TYPED_TEST(Gemm, EveryTransposeMatchesHostReference) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Operands<T> ops{.dev = dev};

  const auto a = random_matrix<T>(kN, 4);
  const auto b = random_matrix<T>(kN, 5);
  const auto c = random_matrix<T>(kN, 6);
  const T alpha{0.75};
  const T beta{2.0};

  ops.upload(ops.a, a);
  ops.upload(ops.b, b);

  for (const bool trans_a : {false, true}) {
    for (const bool trans_b : {false, true}) {
      SCOPED_TRACE(std::format("op_a = {}, op_b = {}", trans_a ? 'T' : 'N', trans_b ? 'T' : 'N'));
      const auto op_a = trans_a ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;
      const auto op_b = trans_b ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;

      ops.upload(ops.c, c);
      ASSERT_EQ(
          fb::gemm(blas, op_a, op_b, kN, &alpha, ops.a.data(), ops.b.data(), &beta, ops.c.data()),
          wwr::WWRBLAS_STATUS_SUCCESS);
      expect_matches(ops.download_c(), reference(trans_a, trans_b, alpha, a, b, beta, c, kN));
    }
  }
}

// The op_both overload: the same op on A and B, against the matching (op, op)
// reference -- so OP_T is pinned to A^T B^T, not (A B)^T.
TYPED_TEST(Gemm, SharedOpMatchesHostReference) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Operands<T> ops{.dev = dev};

  const auto a = random_matrix<T>(kN, 7);
  const auto b = random_matrix<T>(kN, 8);
  const auto c = random_matrix<T>(kN, 9);
  const T alpha{-1.25};
  const T beta{0.5};

  ops.upload(ops.a, a);
  ops.upload(ops.b, b);

  for (const bool trans : {false, true}) {
    SCOPED_TRACE(std::format("op_both = {}", trans ? 'T' : 'N'));
    const auto op = trans ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;

    ops.upload(ops.c, c);
    ASSERT_EQ(fb::gemm(blas, op, kN, &alpha, ops.a.data(), ops.b.data(), &beta, ops.c.data()),
              wwr::WWRBLAS_STATUS_SUCCESS);
    expect_matches(ops.download_c(), reference(trans, trans, alpha, a, b, beta, c, kN));
  }
}

} // namespace
