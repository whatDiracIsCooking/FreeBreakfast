// GPU cases for linalg/blas's :gmmwich: op_x(X)^T diag(d) op_x(X) against a
// host reference that forms the dense diagonal, square and rectangular, for
// both op_x, with a non-symmetric X and a d of distinct mixed-sign entries (so
// a diagonal on the wrong side of X, or a transpose in the wrong place, cannot
// pass), plus the argument check and the pointer-mode restore.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.linalg.blas;

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;

constexpr std::int32_t kN = 37; // odd and not a tile multiple
constexpr std::int32_t kM = 23; // columns of op_x(X), order of C; != kN

// A rows x cols column-major matrix on the host, element (i, j) at i + j*rows.
template <typename T>
struct Rect {
  std::int32_t rows;
  std::int32_t cols;
  std::vector<T> v;

  T at(std::int32_t i, std::int32_t j) const { return v[i + j * rows]; }
};

template <typename T>
Rect<T> random_rect(std::int32_t rows, std::int32_t cols, std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<T> dist{T{-1}, T{1}};
  Rect<T> r{rows, cols, std::vector<T>(static_cast<std::size_t>(rows) * cols)};
  for (auto &e : r.v) {
    e = dist(gen);
  }
  return r;
}

template <typename T>
std::vector<T> random_vector(std::int32_t n, std::uint32_t seed) {
  return random_rect<T>(n, 1, seed).v;
}

// The n x n dense diagonal the device side never forms.
template <typename T>
Rect<T> diagonal(const std::vector<T> &d) {
  const auto n = static_cast<std::int32_t>(d.size());
  Rect<T> r{n, n, std::vector<T>(static_cast<std::size_t>(n) * n, T{0})};
  for (std::int32_t i = 0; i < n; ++i) {
    r.v[i + i * n] = d[i];
  }
  return r;
}

template <typename T>
Rect<T> transposed(const Rect<T> &a) {
  Rect<T> t{a.cols, a.rows, std::vector<T>(a.v.size())};
  for (std::int32_t j = 0; j < a.cols; ++j) {
    for (std::int32_t i = 0; i < a.rows; ++i) {
      t.v[j + i * a.cols] = a.at(i, j);
    }
  }
  return t;
}

template <typename T>
Rect<T> product(const Rect<T> &a, const Rect<T> &b) {
  Rect<T> c{a.rows, b.cols, std::vector<T>(static_cast<std::size_t>(a.rows) * b.cols, T{0})};
  for (std::int32_t j = 0; j < b.cols; ++j) {
    for (std::int32_t k = 0; k < a.cols; ++k) {
      const T bkj = b.at(k, j);
      for (std::int32_t i = 0; i < a.rows; ++i) {
        c.v[i + j * a.rows] += a.at(i, k) * bkj;
      }
    }
  }
  return c;
}

template <typename T>
void upload(const std::shared_ptr<fb::Device> &dev, fb::Buffers::device<T> &d,
            const std::vector<T> &host) {
  fb::Buffers::pinned<T> stage(host.size());
  std::ranges::copy(host, stage.data());
  ASSERT_EQ(ext::copy(d, stage, dev->stream()), wwr::wwrSuccess);
  ASSERT_EQ(ext::sync(dev->stream()), wwr::wwrSuccess);
}

template <typename T>
std::vector<T> download(const std::shared_ptr<fb::Device> &dev, const fb::Buffers::device<T> &d,
                        std::size_t count) {
  fb::Buffers::pinned<T> stage(count);
  EXPECT_EQ(ext::copy(stage, d, dev->stream()), wwr::wwrSuccess);
  EXPECT_EQ(ext::sync(dev->stream()), wwr::wwrSuccess);
  return {stage.data(), stage.data() + count};
}

// One n-term contraction (the dgmm is exact per element): scale by n.
template <typename T>
void expect_matches(const std::vector<T> &got, const Rect<T> &expected) {
  ASSERT_EQ(got.size(), expected.v.size());
  const T tolerance = (std::is_same_v<T, float> ? T{1e-5} : T{1e-13}) * kN;
  for (std::int32_t j = 0; j < expected.cols; ++j) {
    for (std::int32_t i = 0; i < expected.rows; ++i) {
      ASSERT_NEAR(got[i + j * expected.rows], expected.at(i, j), tolerance)
          << "at (" << i << ", " << j << ")";
    }
  }
}

// X, d, work and C on the device for the square forms.
template <typename T>
struct Square {
  std::shared_ptr<fb::Device> dev;
  Rect<T> x = random_rect<T>(kN, kN, 1);
  std::vector<T> d = random_vector<T>(kN, 2);
  fb::Buffers::device<T> dx{x.v.size(), dev};
  fb::Buffers::device<T> dd{d.size(), dev};
  fb::Buffers::device<T> work{x.v.size(), dev};
  fb::Buffers::device<T> c{x.v.size(), dev};

  void upload_operands() {
    upload(dev, dx, x.v);
    upload(dev, dd, d);
  }
  std::vector<T> get(const fb::Buffers::device<T> &b) { return download(dev, b, x.v.size()); }
};

template <typename T>
class Gmmwich : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Gmmwich, RealTypes);

// The default overload is X^T diag(d) X, through both arms of blas_handle; work
// is left holding diag(d) X, as documented.
TYPED_TEST(Gmmwich, DefaultIsXTransposeDiagX) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Square<T> ops{.dev = dev};
  ops.upload_operands();

  const auto dx = product(diagonal(ops.d), ops.x);
  const auto expected = product(transposed(ops.x), dx);

  ASSERT_EQ(fb::gmmwich(blas, kN, ops.dx.data(), ops.dd.data(), ops.work.data(), ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.get(ops.c), expected);
  expect_matches(ops.get(ops.work), dx);

  ASSERT_EQ(
      fb::gmmwich(blas.get(), kN, ops.dx.data(), ops.dd.data(), ops.work.data(), ops.c.data()),
      wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.get(ops.c), expected);
}

// op = OP_T is X diag(d) X^T, with work left holding X diag(d) -- the form
// orthogonalize uses -- and op = OP_N matches the default overload.
TYPED_TEST(Gmmwich, OpSelectsTheSide) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Square<T> ops{.dev = dev};
  ops.upload_operands();
  const auto diag = diagonal(ops.d);

  ASSERT_EQ(fb::gmmwich(blas, wwr::WWRBLAS_OP_N, kN, ops.dx.data(), ops.dd.data(), ops.work.data(),
                        ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.get(ops.c), product(transposed(ops.x), product(diag, ops.x)));

  const auto xd = product(ops.x, diag);
  ASSERT_EQ(fb::gmmwich(blas, wwr::WWRBLAS_OP_T, kN, ops.dx.data(), ops.dd.data(), ops.work.data(),
                        ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.get(ops.c), product(xd, transposed(ops.x)));
  expect_matches(ops.get(ops.work), xd);
}

// OP_C is rejected before anything is enqueued: C keeps its prior contents.
TYPED_TEST(Gmmwich, RejectsAnOpOtherThanNOrT) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Square<T> ops{.dev = dev};

  const auto sentinel = random_rect<T>(kN, kN, 3).v;
  upload(dev, ops.c, sentinel);

  EXPECT_EQ(fb::gmmwich(blas, wwr::WWRBLAS_OP_C, kN, ops.dx.data(), ops.dd.data(), ops.work.data(),
                        ops.c.data()),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  EXPECT_EQ(ops.get(ops.c), sentinel);
}

// A handle in device pointer mode gets the right answer (the gemm's scalars are
// read from host memory regardless) and is handed back in device pointer mode.
TYPED_TEST(Gmmwich, RestoresDevicePointerMode) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Square<T> ops{.dev = dev};
  ops.upload_operands();

  ASSERT_EQ(wwr::wwrblasSetPointerMode(blas.get(), wwr::WWRBLAS_POINTER_MODE_DEVICE),
            wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(fb::gmmwich(blas, kN, ops.dx.data(), ops.dd.data(), ops.work.data(), ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);

  wwr::wwrblasPointerMode_t mode = wwr::WWRBLAS_POINTER_MODE_HOST;
  ASSERT_EQ(wwr::wwrblasGetPointerMode(blas.get(), &mode), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(mode, wwr::WWRBLAS_POINTER_MODE_DEVICE);

  expect_matches(ops.get(ops.c), product(transposed(ops.x), product(diagonal(ops.d), ops.x)));
}

// The general overload with m != n, both op_x, so a swapped dimension or leading
// dimension fails rather than reading in bounds by luck. X is stored n x m for
// op_x = N and m x n for op_x = T, and work comes back in X's stored shape.
TYPED_TEST(Gmmwich, GeneralRectangularMatchesHostReference) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);

  const auto d = random_vector<T>(kN, 4);
  const auto diag = diagonal(d);
  fb::Buffers::device<T> dd(d.size(), dev);
  fb::Buffers::device<T> dx(static_cast<std::size_t>(kM) * kN, dev);
  fb::Buffers::device<T> work(static_cast<std::size_t>(kM) * kN, dev);
  fb::Buffers::device<T> c(static_cast<std::size_t>(kM) * kM, dev);
  upload(dev, dd, d);

  for (const bool trans_x : {false, true}) {
    SCOPED_TRACE(std::format("op_x = {}", trans_x ? 'T' : 'N'));
    const auto op_x = trans_x ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;

    // x is X as stored; opx is op_x(X), always n x m.
    const auto x = trans_x ? random_rect<T>(kM, kN, 5) : random_rect<T>(kN, kM, 5);
    const auto opx = trans_x ? transposed(x) : x;
    const auto expected_c = product(transposed(opx), product(diag, opx));
    const auto expected_work = trans_x ? product(x, diag) : product(diag, x);
    upload(dev, dx, x.v);

    ASSERT_EQ(fb::gmmwich(blas, op_x, kM, kN, dx.data(), dd.data(), work.data(), c.data()),
              wwr::WWRBLAS_STATUS_SUCCESS);
    expect_matches(download(dev, c, expected_c.v.size()), expected_c);
    expect_matches(download(dev, work, expected_work.v.size()), expected_work);
  }
}

} // namespace
