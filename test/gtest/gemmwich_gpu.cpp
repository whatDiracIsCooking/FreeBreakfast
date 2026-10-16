// GPU cases for linalg/blas's :gemmwich: the general op_x(X)^T op_f(F) op_x(X)
// with a rectangular X, and the square X^T F X / X F X^T, against a host
// reference, with non-symmetric X and F (so the two orders, or a transpose on
// the wrong side, cannot pass), plus the argument check and the pointer-mode
// restore.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.linalg.blas;

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;

constexpr std::int32_t kN = 37; // odd and not a tile multiple
constexpr std::size_t kNN = static_cast<std::size_t>(kN) * kN;

// Column-major n x n on the host, element (i, j) at i + j*n.
template <typename T>
std::vector<T> random_matrix(std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<T> dist{T{-1}, T{1}};
  std::vector<T> m(kNN);
  for (auto &v : m) {
    v = dist(gen);
  }
  return m;
}

template <typename T>
std::vector<T> transpose(const std::vector<T> &m) {
  std::vector<T> t(m.size());
  for (std::int32_t j = 0; j < kN; ++j) {
    for (std::int32_t i = 0; i < kN; ++i) {
      t[j + i * kN] = m[i + j * kN];
    }
  }
  return t;
}

template <typename T>
std::vector<T> matmul(const std::vector<T> &a, const std::vector<T> &b) {
  std::vector<T> c(a.size(), T{0});
  for (std::int32_t j = 0; j < kN; ++j) {
    for (std::int32_t k = 0; k < kN; ++k) {
      const T bkj = b[k + j * kN];
      for (std::int32_t i = 0; i < kN; ++i) {
        c[i + j * kN] += a[i + k * kN] * bkj;
      }
    }
  }
  return c;
}

// X, F, work and C on the device, with host staging for upload and download.
template <typename T>
struct Operands {
  std::shared_ptr<fb::Device> dev;
  fb::Buffers::pinned<T> stage{kNN};
  fb::Buffers::device<T> x{kNN, dev};
  fb::Buffers::device<T> f{kNN, dev};
  fb::Buffers::device<T> work{kNN, dev};
  fb::Buffers::device<T> c{kNN, dev};

  void upload(fb::Buffers::device<T> &d, const std::vector<T> &host) {
    std::ranges::copy(host, stage.data());
    ASSERT_EQ(ext::copy(d, stage, dev->stream()), wwr::wwrSuccess);
    ASSERT_EQ(ext::sync(dev->stream()), wwr::wwrSuccess);
  }

  std::vector<T> download(const fb::Buffers::device<T> &d) {
    EXPECT_EQ(ext::copy(stage, d, dev->stream()), wwr::wwrSuccess);
    EXPECT_EQ(ext::sync(dev->stream()), wwr::wwrSuccess);
    return {stage.data(), stage.data() + kNN};
  }
};

// Two chained products of n-term sums: the error grows with n, so scale by it.
template <typename T>
void expect_matches(const std::vector<T> &got, const std::vector<T> &expected) {
  const T tolerance = (std::is_same_v<T, float> ? T{1e-5} : T{1e-13}) * kN;
  for (std::int32_t j = 0; j < kN; ++j) {
    for (std::int32_t i = 0; i < kN; ++i) {
      ASSERT_NEAR(got[i + j * kN], expected[i + j * kN], tolerance)
          << "at (" << i << ", " << j << ")";
    }
  }
}

template <typename T>
class Gemmwich : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Gemmwich, RealTypes);

// The default overload is X^T F X, through both arms of blas_handle; work is
// left holding F X, as documented.
TYPED_TEST(Gemmwich, DefaultIsXTransposeFX) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Operands<T> ops{.dev = dev};

  const auto x = random_matrix<T>(1);
  const auto f = random_matrix<T>(2);
  const auto fx = matmul(f, x);
  const auto expected = matmul(transpose(x), fx);

  ops.upload(ops.x, x);
  ops.upload(ops.f, f);

  ASSERT_EQ(fb::gemmwich(blas, kN, ops.x.data(), ops.f.data(), ops.work.data(), ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.download(ops.c), expected);
  expect_matches(ops.download(ops.work), fx);

  ASSERT_EQ(fb::gemmwich(blas.get(), kN, ops.x.data(), ops.f.data(), ops.work.data(), ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.download(ops.c), expected);
}

// op = OP_T is X F X^T, and op = OP_N matches the default overload.
TYPED_TEST(Gemmwich, OpSelectsTheSide) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Operands<T> ops{.dev = dev};

  const auto x = random_matrix<T>(3);
  const auto f = random_matrix<T>(4);

  ops.upload(ops.x, x);
  ops.upload(ops.f, f);

  ASSERT_EQ(fb::gemmwich(blas, wwr::WWRBLAS_OP_N, kN, ops.x.data(), ops.f.data(), ops.work.data(),
                         ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.download(ops.c), matmul(transpose(x), matmul(f, x)));

  ASSERT_EQ(fb::gemmwich(blas, wwr::WWRBLAS_OP_T, kN, ops.x.data(), ops.f.data(), ops.work.data(),
                         ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.download(ops.c), matmul(x, matmul(f, transpose(x))));
}

// OP_C is rejected before anything is enqueued: C keeps its prior contents.
TYPED_TEST(Gemmwich, RejectsAnOpOtherThanNOrT) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Operands<T> ops{.dev = dev};

  const auto sentinel = random_matrix<T>(5);
  ops.upload(ops.c, sentinel);

  EXPECT_EQ(fb::gemmwich(blas, wwr::WWRBLAS_OP_C, kN, ops.x.data(), ops.f.data(), ops.work.data(),
                         ops.c.data()),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  EXPECT_EQ(ops.download(ops.c), sentinel);
}

// A handle in device pointer mode gets the right answer (the scalars are read
// from host memory regardless) and is handed back in device pointer mode.
TYPED_TEST(Gemmwich, RestoresDevicePointerMode) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Operands<T> ops{.dev = dev};

  const auto x = random_matrix<T>(6);
  const auto f = random_matrix<T>(7);
  ops.upload(ops.x, x);
  ops.upload(ops.f, f);

  ASSERT_EQ(wwr::wwrblasSetPointerMode(blas.get(), wwr::WWRBLAS_POINTER_MODE_DEVICE),
            wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(fb::gemmwich(blas, kN, ops.x.data(), ops.f.data(), ops.work.data(), ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);

  wwr::wwrblasPointerMode_t mode = wwr::WWRBLAS_POINTER_MODE_HOST;
  ASSERT_EQ(wwr::wwrblasGetPointerMode(blas.get(), &mode), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(mode, wwr::WWRBLAS_POINTER_MODE_DEVICE);

  expect_matches(ops.download(ops.c), matmul(transpose(x), matmul(f, x)));
}

// ---------------------------------------------------------------------------
// The general overload, with a rectangular X
// ---------------------------------------------------------------------------

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

// Every op_x / op_f combination with m != n, so a swapped dimension or leading
// dimension fails rather than reading in bounds by luck. X is stored n x m for
// op_x = N and m x n for op_x = T; work comes back as op_f(F) op_x(X).
TYPED_TEST(Gemmwich, GeneralRectangularMatchesHostReference) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);

  const auto f = random_rect<T>(kN, kN, 8);
  fb::Buffers::device<T> df(f.v.size(), dev);
  fb::Buffers::device<T> dx(static_cast<std::size_t>(kM) * kN, dev);
  fb::Buffers::device<T> work(static_cast<std::size_t>(kN) * kM, dev);
  fb::Buffers::device<T> c(static_cast<std::size_t>(kM) * kM, dev);
  upload(dev, df, f.v);

  for (const bool trans_x : {false, true}) {
    for (const bool trans_f : {false, true}) {
      SCOPED_TRACE(std::format("op_x = {}, op_f = {}", trans_x ? 'T' : 'N', trans_f ? 'T' : 'N'));
      const auto op_x = trans_x ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;
      const auto op_f = trans_f ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;

      // x is X as stored; opx is op_x(X), always n x m.
      const auto x = trans_x ? random_rect<T>(kM, kN, 9) : random_rect<T>(kN, kM, 9);
      const auto opx = trans_x ? transposed(x) : x;
      const auto opf = trans_f ? transposed(f) : f;
      const auto expected_work = product(opf, opx);
      const auto expected_c = product(transposed(opx), expected_work);
      upload(dev, dx, x.v);

      ASSERT_EQ(fb::gemmwich(blas, op_x, op_f, kM, kN, dx.data(), df.data(), work.data(), c.data()),
                wwr::WWRBLAS_STATUS_SUCCESS);
      expect_matches(download(dev, c, expected_c.v.size()), expected_c);
      expect_matches(download(dev, work, expected_work.v.size()), expected_work);
    }
  }
}

// op_f is checked too, not just op_x.
TYPED_TEST(Gemmwich, GeneralRejectsAnOpFOtherThanNOrT) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  fb::Buffers::device<T> buf(kNN, dev);

  EXPECT_EQ(fb::gemmwich(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_C, kN, kN, buf.data(), buf.data(),
                         buf.data(), buf.data()),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
}

} // namespace
