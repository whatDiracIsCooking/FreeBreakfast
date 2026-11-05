// GPU cases for recipes/blas's :gemm: gemm, square and rectangular, with its
// ops defaulted and given, against a host reference, with non-symmetric
// operands (so a swapped or wrongly transposed operand cannot pass) and a
// non-zero beta (so C's prior contents must be accumulated, not overwritten),
// through both a BlasHandle and a raw wwrblasHandle_t.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.recipes.blas;

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

// The defaulted call, A B, through both arms of blas_handle: the wrapper
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

// An op as a type, so a test can pass it to a lambda and get it back as a
// template argument.
template <wwr::wwrblasOperation_t Op>
using OpConstant = std::integral_constant<wwr::wwrblasOperation_t, Op>;

// Explicit ops, over every N/T combination on each side. The (N, N) case pins
// that the defaulted call is exactly this one with OP_N twice.
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

  const auto check = [&]<wwr::wwrblasOperation_t OpA, wwr::wwrblasOperation_t OpB>(
                         OpConstant<OpA>, OpConstant<OpB>) {
    constexpr bool trans_a = OpA == wwr::WWRBLAS_OP_T;
    constexpr bool trans_b = OpB == wwr::WWRBLAS_OP_T;
    SCOPED_TRACE(std::format("OpA = {}, OpB = {}", trans_a ? 'T' : 'N', trans_b ? 'T' : 'N'));

    ops.upload(ops.c, c);
    // Parenthesized: the comma in the template arguments would split the macro's.
    ASSERT_EQ(
        (fb::gemm<OpA, OpB>(blas, kN, &alpha, ops.a.data(), ops.b.data(), &beta, ops.c.data())),
        wwr::WWRBLAS_STATUS_SUCCESS);
    expect_matches(ops.download_c(), reference(trans_a, trans_b, alpha, a, b, beta, c, kN));
  };
  using N = OpConstant<wwr::WWRBLAS_OP_N>;
  using Tr = OpConstant<wwr::WWRBLAS_OP_T>;
  check(N{}, N{});
  check(N{}, Tr{});
  check(Tr{}, N{});
  check(Tr{}, Tr{});
}

// One op given: it is OpA, and OpB stays N -- so gemm<OP_T> is A^T B, not
// A^T B^T. OP_C on a real T is the same as OP_T.
TYPED_TEST(Gemm, OneOpLeavesBUntransposed) {
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
  const auto expected = reference(true, false, alpha, a, b, beta, c, kN);

  ops.upload(ops.a, a);
  ops.upload(ops.b, b);

  ops.upload(ops.c, c);
  ASSERT_EQ(fb::gemm<wwr::WWRBLAS_OP_T>(blas, kN, &alpha, ops.a.data(), ops.b.data(), &beta,
                                        ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.download_c(), expected);

  ops.upload(ops.c, c);
  ASSERT_EQ(fb::gemm<wwr::WWRBLAS_OP_C>(blas, kN, &alpha, ops.a.data(), ops.b.data(), &beta,
                                        ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.download_c(), expected);
}

// The rectangular overload's shape: three distinct extents, so a leading
// dimension taken from the wrong one reads out of place (or out of bounds).
constexpr std::int32_t kRectM = 23;
constexpr std::int32_t kRectN = 31;
constexpr std::int32_t kRectK = 37;

// Column-major rows x cols on the host.
template <typename T>
std::vector<T> random_rect(std::int32_t rows, std::int32_t cols, std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<T> dist{T{-1}, T{1}};
  std::vector<T> m(static_cast<std::size_t>(rows) * cols);
  for (auto &v : m) {
    v = dist(gen);
  }
  return m;
}

// alpha * op(A) * op(B) + beta * C for m x k op(A) and k x n op(B), each stored
// packed: A is m x k or k x m, B is k x n or n x k.
template <typename T>
std::vector<T> rect_reference(bool trans_a, bool trans_b, T alpha, const std::vector<T> &a,
                              const std::vector<T> &b, T beta, const std::vector<T> &c) {
  const std::int32_t lda = trans_a ? kRectK : kRectM;
  const std::int32_t ldb = trans_b ? kRectN : kRectK;
  std::vector<T> out(c.size());
  for (std::int32_t j = 0; j < kRectN; ++j) {
    for (std::int32_t i = 0; i < kRectM; ++i) {
      T sum{0};
      for (std::int32_t p = 0; p < kRectK; ++p) {
        const T a_ip = trans_a ? a[p + i * lda] : a[i + p * lda];
        const T b_pj = trans_b ? b[j + p * ldb] : b[p + j * ldb];
        sum += a_ip * b_pj;
      }
      out[i + j * kRectM] = alpha * sum + beta * c[i + j * kRectM];
    }
  }
  return out;
}

// The rectangular overload over every N/T combination, each operand packed in
// the shape its op implies, with a non-zero beta.
TYPED_TEST(Gemm, RectangularMatchesHostReference) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  constexpr std::size_t kMk = static_cast<std::size_t>(kRectM) * kRectK;
  constexpr std::size_t kKn = static_cast<std::size_t>(kRectK) * kRectN;
  constexpr std::size_t kMn = static_cast<std::size_t>(kRectM) * kRectN;
  fb::Buffers::pinned<T> stage{std::max({kMk, kKn, kMn})};
  fb::Buffers::device<T> da{kMk, dev};
  fb::Buffers::device<T> db{kKn, dev};
  fb::Buffers::device<T> dc{kMn, dev};
  const auto upload = [&](fb::Buffers::device<T> &d, const std::vector<T> &host) {
    std::ranges::copy(host, stage.data());
    ASSERT_EQ(wwr::wwrMemcpyAsync(d.data(), stage.data(), host.size() * sizeof(T),
                                  wwr::wwrMemcpyHostToDevice, dev->stream()),
              wwr::wwrSuccess);
    ASSERT_EQ(ext::sync(dev->stream()), wwr::wwrSuccess);
  };
  const auto download_c = [&] {
    EXPECT_EQ(wwr::wwrMemcpyAsync(stage.data(), dc.data(), kMn * sizeof(T),
                                  wwr::wwrMemcpyDeviceToHost, dev->stream()),
              wwr::wwrSuccess);
    EXPECT_EQ(ext::sync(dev->stream()), wwr::wwrSuccess);
    return std::vector<T>(stage.data(), stage.data() + kMn);
  };

  // The element count is the same either way round, so one A and one B serve
  // every op: only how they are read changes.
  const auto a = random_rect<T>(kRectM, kRectK, 10);
  const auto b = random_rect<T>(kRectK, kRectN, 11);
  const auto c = random_rect<T>(kRectM, kRectN, 12);
  const T alpha{0.5};
  const T beta{-1.5};
  upload(da, a);
  upload(db, b);

  const auto check = [&]<wwr::wwrblasOperation_t OpA, wwr::wwrblasOperation_t OpB>(
                         OpConstant<OpA>, OpConstant<OpB>) {
    constexpr bool trans_a = OpA == wwr::WWRBLAS_OP_T;
    constexpr bool trans_b = OpB == wwr::WWRBLAS_OP_T;
    SCOPED_TRACE(std::format("OpA = {}, OpB = {}", trans_a ? 'T' : 'N', trans_b ? 'T' : 'N'));

    upload(dc, c);
    ASSERT_EQ((fb::gemm<OpA, OpB>(blas, kRectM, kRectN, kRectK, &alpha, da.data(), db.data(), &beta,
                                  dc.data())),
              wwr::WWRBLAS_STATUS_SUCCESS);
    const auto got = download_c();
    const auto expected = rect_reference(trans_a, trans_b, alpha, a, b, beta, c);
    const T tolerance = std::is_same_v<T, float> ? T{1e-4} : T{1e-12};
    for (std::size_t i = 0; i < kMn; ++i) {
      ASSERT_NEAR(got[i], expected[i], tolerance) << "at " << i;
    }
  };
  using N = OpConstant<wwr::WWRBLAS_OP_N>;
  using Tr = OpConstant<wwr::WWRBLAS_OP_T>;
  check(N{}, N{});
  check(N{}, Tr{});
  check(Tr{}, N{});
  check(Tr{}, Tr{});
}

// A value of the op type that is none of N, T, C on either backend:
// CUBLAS_OP_CONJG on CUDA (unsupported, and not named by wwr), nothing on HIP,
// whose ops are 111-113. Within both enums' value ranges, so the cast is a
// constant expression.
constexpr auto kNotAnOp = static_cast<wwr::wwrblasOperation_t>(3);

// Whether gemm<OpA, OpB> can be called at all: a requires-expression must sit
// in a template to report an ill-formed call as false rather than as an error.
template <wwr::wwrblasOperation_t OpA, wwr::wwrblasOperation_t OpB>
constexpr bool kGemmTakes = requires(const fb::BlasHandle &h, const double *p, double *c) {
  fb::gemm<OpA, OpB>(h, kN, p, p, p, p, c);
};

// Every BlasOp on each side, and nothing outside it.
static_assert(kGemmTakes<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N>);
static_assert(kGemmTakes<wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_C>);
static_assert(kGemmTakes<wwr::WWRBLAS_OP_C, wwr::WWRBLAS_OP_T>);
static_assert(!kGemmTakes<kNotAnOp, wwr::WWRBLAS_OP_N>);
static_assert(!kGemmTakes<wwr::WWRBLAS_OP_N, kNotAnOp>);

// The same constraint on the rectangular overload.
template <wwr::wwrblasOperation_t OpA, wwr::wwrblasOperation_t OpB>
constexpr bool kRectGemmTakes = requires(const fb::BlasHandle &h, const double *p, double *c) {
  fb::gemm<OpA, OpB>(h, kRectM, kRectN, kRectK, p, p, p, p, c);
};

static_assert(kRectGemmTakes<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N>);
static_assert(kRectGemmTakes<wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_C>);
static_assert(!kRectGemmTakes<kNotAnOp, wwr::WWRBLAS_OP_N>);
static_assert(!kRectGemmTakes<wwr::WWRBLAS_OP_N, kNotAnOp>);

} // namespace
