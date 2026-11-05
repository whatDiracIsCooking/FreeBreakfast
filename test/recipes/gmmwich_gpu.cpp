// GPU cases for recipes/blas's :gmmwich: OpX(X)^T diag(d) OpX(X) against a
// host reference that forms the dense diagonal, square and rectangular, for
// both OpX, with a non-symmetric X and a d of distinct mixed-sign entries (so
// a diagonal on the wrong side of X, or a transpose in the wrong place, cannot
// pass), plus which ops it takes and the pointer-mode restore; and on a complex
// T, OpX(X)^H diag(d) OpX(X) against a std::complex reference.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.recipes.blas;

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;

constexpr std::int32_t kN = 37; // odd and not a tile multiple
constexpr std::int32_t kM = 23; // columns of OpX(X), order of C; != kN

// An op as a type, so a test can pass it to a lambda and get it back as a
// template argument.
template <wwr::wwrblasOperation_t Op>
using OpConstant = std::integral_constant<wwr::wwrblasOperation_t, Op>;

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

// OpX = OP_T is X diag(d) X^T, with work left holding X diag(d) -- the form
// orthogonalize uses -- and OpX = OP_N matches the defaulted call.
TYPED_TEST(Gmmwich, OpSelectsTheSide) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);
  Square<T> ops{.dev = dev};
  ops.upload_operands();
  const auto diag = diagonal(ops.d);

  ASSERT_EQ(fb::gmmwich<wwr::WWRBLAS_OP_N>(blas, kN, ops.dx.data(), ops.dd.data(), ops.work.data(),
                                           ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.get(ops.c), product(transposed(ops.x), product(diag, ops.x)));

  const auto xd = product(ops.x, diag);
  ASSERT_EQ(fb::gmmwich<wwr::WWRBLAS_OP_T>(blas, kN, ops.dx.data(), ops.dd.data(), ops.work.data(),
                                           ops.c.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches(ops.get(ops.c), product(xd, transposed(ops.x)));
  expect_matches(ops.get(ops.work), xd);
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

// The general overload with m != n, both OpX, so a swapped dimension or leading
// dimension fails rather than reading in bounds by luck. X is stored n x m for
// OpX = N and m x n for OpX = T, and work comes back in X's stored shape.
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

  const auto check = [&]<wwr::wwrblasOperation_t OpX>(OpConstant<OpX>) {
    constexpr bool trans_x = OpX == wwr::WWRBLAS_OP_T;
    SCOPED_TRACE(std::format("OpX = {}", trans_x ? 'T' : 'N'));

    // x is X as stored; opx is OpX(X), always n x m.
    const auto x = trans_x ? random_rect<T>(kM, kN, 5) : random_rect<T>(kN, kM, 5);
    const auto opx = trans_x ? transposed(x) : x;
    const auto expected_c = product(transposed(opx), product(diag, opx));
    const auto expected_work = trans_x ? product(x, diag) : product(diag, x);
    upload(dev, dx, x.v);

    ASSERT_EQ(fb::gmmwich<OpX>(blas, kM, kN, dx.data(), dd.data(), work.data(), c.data()),
              wwr::WWRBLAS_STATUS_SUCCESS);
    expect_matches(download(dev, c, expected_c.v.size()), expected_c);
    expect_matches(download(dev, work, expected_work.v.size()), expected_work);
  };
  check(OpConstant<wwr::WWRBLAS_OP_N>{});
  check(OpConstant<wwr::WWRBLAS_OP_T>{});
}

// Whether each overload's gmmwich<OpX> can be called at all: a
// requires-expression must sit in a template to report an ill-formed call as
// false rather than as an error.
template <wwr::wwrblasOperation_t OpX>
constexpr bool kSquareTakes = requires(const fb::BlasHandle &h, const double *p, double *w) {
  fb::gmmwich<OpX>(h, kN, p, p, w, w);
};
template <wwr::wwrblasOperation_t OpX>
constexpr bool kGeneralTakes = requires(const fb::BlasHandle &h, const double *p, double *w) {
  fb::gmmwich<OpX>(h, kM, kN, p, p, w, w);
};

// N and T, and not OP_C: on a real T it is T under another name.
static_assert(kSquareTakes<wwr::WWRBLAS_OP_N> && kGeneralTakes<wwr::WWRBLAS_OP_N>);
static_assert(kSquareTakes<wwr::WWRBLAS_OP_T> && kGeneralTakes<wwr::WWRBLAS_OP_T>);
static_assert(!kSquareTakes<wwr::WWRBLAS_OP_C>);
static_assert(!kGeneralTakes<wwr::WWRBLAS_OP_C>);

// ---------------------------------------------------------------------------
// Complex T: the sandwich is OpX(X)^H diag(d) OpX(X)
// ---------------------------------------------------------------------------

// The host reference is std::complex, which has the arithmetic the vendor
// complex types lack; the two share a layout but are converted by value.
template <typename Z>
using HostComplex = std::complex<wwr::ComplexToRealType<Z>>;

template <typename Z>
std::vector<Z> to_device_values(const std::vector<HostComplex<Z>> &a) {
  std::vector<Z> out;
  out.reserve(a.size());
  for (const auto &z : a) {
    out.push_back(Z{z.real(), z.imag()});
  }
  return out;
}

template <typename R>
Rect<std::complex<R>> random_complex_rect(std::int32_t rows, std::int32_t cols,
                                          std::uint32_t seed) {
  const auto re = random_rect<R>(rows, cols, seed);
  const auto im = random_rect<R>(rows, cols, seed + 1000);
  Rect<std::complex<R>> z{rows, cols, std::vector<std::complex<R>>(re.v.size())};
  for (std::size_t i = 0; i < z.v.size(); ++i) {
    z.v[i] = {re.v[i], im.v[i]};
  }
  return z;
}

template <typename R>
Rect<std::complex<R>> adjoint(const Rect<std::complex<R>> &a) {
  auto t = transposed(a);
  for (auto &z : t.v) {
    z = std::conj(z);
  }
  return t;
}

template <typename Z>
void expect_matches_complex(const std::vector<Z> &got, const Rect<HostComplex<Z>> &expected) {
  using R = wwr::ComplexToRealType<Z>;
  ASSERT_EQ(got.size(), expected.v.size());
  // A complex product is four real ones: twice the real tolerance.
  const R tolerance = (std::is_same_v<R, float> ? R{2e-5} : R{2e-13}) * kN;
  for (std::int32_t j = 0; j < expected.cols; ++j) {
    for (std::int32_t i = 0; i < expected.rows; ++i) {
      const Z g = got[i + j * expected.rows];
      ASSERT_LE(std::abs(HostComplex<Z>{g.x, g.y} - expected.at(i, j)), tolerance)
          << "at (" << i << ", " << j << ")";
    }
  }
}

template <typename Z>
class GmmwichComplex : public ::testing::Test {};

using ComplexTypes = ::testing::Types<wwr::wwrFloatComplex, wwr::wwrDoubleComplex>;
TYPED_TEST_SUITE(GmmwichComplex, ComplexTypes);

// Both OpX in {N, C} with m != n, then the square default, with complex X and a
// complex d (the dgmm takes d as T, so nothing about it need be real), so a
// transpose where a conjugate transpose belongs fails.
TYPED_TEST(GmmwichComplex, AdjointSandwichMatchesHostReference) {
  FBF_REQUIRE_GPU();
  using Z = TypeParam;
  using R = wwr::ComplexToRealType<Z>;

  auto dev = std::make_shared<fb::Device>(0);
  fb::BlasHandle blas(dev);

  const auto d = random_complex_rect<R>(kN, 1, 6).v;
  const auto diag = diagonal(d);
  fb::Buffers::device<Z> dd(d.size(), dev);
  fb::Buffers::device<Z> dx(static_cast<std::size_t>(kM) * kN, dev);
  fb::Buffers::device<Z> work(static_cast<std::size_t>(kM) * kN, dev);
  fb::Buffers::device<Z> c(static_cast<std::size_t>(kM) * kM, dev);
  upload(dev, dd, to_device_values<Z>(d));

  const auto check = [&]<wwr::wwrblasOperation_t OpX>(OpConstant<OpX>) {
    constexpr bool adj_x = OpX == wwr::WWRBLAS_OP_C;
    SCOPED_TRACE(std::format("OpX = {}", adj_x ? 'C' : 'N'));

    const auto x = adj_x ? random_complex_rect<R>(kM, kN, 7) : random_complex_rect<R>(kN, kM, 7);
    const auto opx = adj_x ? adjoint(x) : x;
    const auto expected_c = product(adjoint(opx), product(diag, opx));
    const auto expected_work = adj_x ? product(x, diag) : product(diag, x);
    upload(dev, dx, to_device_values<Z>(x.v));

    ASSERT_EQ(fb::gmmwich<OpX>(blas, kM, kN, dx.data(), dd.data(), work.data(), c.data()),
              wwr::WWRBLAS_STATUS_SUCCESS);
    expect_matches_complex(download(dev, c, expected_c.v.size()), expected_c);
    expect_matches_complex(download(dev, work, expected_work.v.size()), expected_work);
  };
  check(OpConstant<wwr::WWRBLAS_OP_N>{});
  check(OpConstant<wwr::WWRBLAS_OP_C>{});

  // The square overload, on buffers of its own: copy wants matching sizes.
  const auto x = random_complex_rect<R>(kN, kN, 8);
  fb::Buffers::device<Z> sx(x.v.size(), dev);
  fb::Buffers::device<Z> swork(x.v.size(), dev);
  fb::Buffers::device<Z> sc(x.v.size(), dev);
  upload(dev, sx, to_device_values<Z>(x.v));
  ASSERT_EQ(fb::gmmwich(blas, kN, sx.data(), dd.data(), swork.data(), sc.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_matches_complex(download(dev, sc, x.v.size()), product(adjoint(x), product(diag, x)));
}

template <wwr::wwrblasOperation_t OpX>
constexpr bool kComplexTakes =
    requires(const fb::BlasHandle &h, const wwr::wwrDoubleComplex *p, wwr::wwrDoubleComplex *w) {
      fb::gmmwich<OpX>(h, kM, kN, p, p, w, w);
    };

// On a complex T the adjoint is OP_C, and OP_T is left out: (X^T)^H = conj(X)
// is no BLAS op.
static_assert(kComplexTakes<wwr::WWRBLAS_OP_N> && kComplexTakes<wwr::WWRBLAS_OP_C>);
static_assert(!kComplexTakes<wwr::WWRBLAS_OP_T>);

} // namespace
