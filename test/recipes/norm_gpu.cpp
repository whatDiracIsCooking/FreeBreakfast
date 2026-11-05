// GPU cases for recipes.norm: each Norm in both precisions and both pointer
// modes against a host reference, on a contiguous and a strided vector, plus
// an empty one and the handle's pointer mode left as it was. ell_inf is exact
// (it only selects and drops a sign); ell_1 and ell_2 are compared to a
// tolerance, since BLAS sums in its own order.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;
using fb::Norm;

constexpr std::int32_t kN = 37; // odd, so no block size divides it
constexpr std::int32_t kIncx = 3;

template <typename T>
void upload(fb::DeviceResources &res, fb::Buffers::device<T> &d, const std::vector<T> &host) {
  ASSERT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(T),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

template <typename T>
T download_one(fb::DeviceResources &res, const fb::Buffers::device<T> &d) {
  T host{};
  EXPECT_EQ(
      wwr::wwrMemcpyAsync(&host, d.data(), sizeof(T), wwr::wwrMemcpyDeviceToHost, res.stream()),
      wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// Alternating signs and a magnitude that wanders, so neither the first, the
// last nor the largest signed element is the ell_inf answer. Element 20 of the
// strided view is the largest magnitude, and negative; the element just past
// it, off the stride, is larger still and must not be seen.
template <typename T>
std::vector<T> vector_of(const std::int32_t incx) {
  std::vector<T> x(static_cast<std::size_t>(kN) * incx);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const T magnitude = T{1} + static_cast<T>((i * 7) % 11);
    x[i] = (i % 2 == 0) ? magnitude : -magnitude;
  }
  x[static_cast<std::size_t>(20) * incx] = T{-50};
  if (incx > 1) {
    x[(static_cast<std::size_t>(20) * incx) + 1] = T{1000};
  }
  return x;
}

template <Norm which, typename T>
T reference(const std::vector<T> &x, const std::int32_t incx) {
  T acc{0};
  for (std::int32_t i = 0; i < kN; ++i) {
    const T v = x[static_cast<std::size_t>(i) * incx];
    if constexpr (which == Norm::l1) {
      acc += std::abs(v);
    } else if constexpr (which == Norm::l2) {
      acc += v * v;
    } else {
      acc = std::max(acc, std::abs(v));
    }
  }
  return which == Norm::l2 ? std::sqrt(acc) : acc;
}

template <Norm which, typename T>
void expect_norm(const T got, const T want) {
  if constexpr (which == Norm::inf) {
    EXPECT_EQ(got, want);
  } else {
    EXPECT_NEAR(got, want, want * T{16} * std::numeric_limits<T>::epsilon());
  }
}

template <Norm which, typename T>
void check_host_mode(const std::int32_t incx) {
  fb::DeviceResources res{0};
  const auto x = vector_of<T>(incx);
  fb::Buffers::device<T> dx(x.size(), res.device());
  upload(res, dx, x);

  T got{-1};
  ASSERT_TRUE(fb::norm<which>(res, kN, dx.data(), incx, &got));
  expect_norm<which>(got, reference<which>(x, incx));
}

template <Norm which, typename T>
void check_device_mode(const std::int32_t incx) {
  fb::DeviceResources res{0};
  const auto x = vector_of<T>(incx);
  fb::Buffers::device<T> dx(x.size(), res.device());
  fb::Buffers::device<T> dresult(1, res.device());
  upload(res, dx, x);

  ASSERT_EQ(wwr::wwrblasSetPointerMode(res.blas().get(), wwr::WWRBLAS_POINTER_MODE_DEVICE),
            wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_TRUE(fb::norm<which>(res, kN, dx.data(), incx, dresult.data()));
  wwr::wwrblasPointerMode_t mode{};
  ASSERT_EQ(wwr::wwrblasGetPointerMode(res.blas().get(), &mode), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(mode, wwr::WWRBLAS_POINTER_MODE_DEVICE);
  ASSERT_EQ(wwr::wwrblasSetPointerMode(res.blas().get(), wwr::WWRBLAS_POINTER_MODE_HOST),
            wwr::WWRBLAS_STATUS_SUCCESS);
  expect_norm<which>(download_one(res, dresult), reference<which>(x, incx));
}

template <typename T>
class NormGpu : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NormGpu, RealTypes);

TYPED_TEST(NormGpu, HostModeContiguous) {
  FBF_REQUIRE_GPU();
  check_host_mode<Norm::l1, TypeParam>(1);
  check_host_mode<Norm::l2, TypeParam>(1);
  check_host_mode<Norm::inf, TypeParam>(1);
}

TYPED_TEST(NormGpu, HostModeStrided) {
  FBF_REQUIRE_GPU();
  check_host_mode<Norm::l1, TypeParam>(kIncx);
  check_host_mode<Norm::l2, TypeParam>(kIncx);
  check_host_mode<Norm::inf, TypeParam>(kIncx);
}

TYPED_TEST(NormGpu, DeviceModeContiguous) {
  FBF_REQUIRE_GPU();
  check_device_mode<Norm::l1, TypeParam>(1);
  check_device_mode<Norm::l2, TypeParam>(1);
  check_device_mode<Norm::inf, TypeParam>(1);
}

TYPED_TEST(NormGpu, DeviceModeStrided) {
  FBF_REQUIRE_GPU();
  check_device_mode<Norm::l1, TypeParam>(kIncx);
  check_device_mode<Norm::l2, TypeParam>(kIncx);
  check_device_mode<Norm::inf, TypeParam>(kIncx);
}

TYPED_TEST(NormGpu, EmptyIsZeroInBothModes) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  fb::DeviceResources res{0};
  T got{-7};
  ASSERT_TRUE(fb::norm<Norm::inf>(res, 0, static_cast<const T *>(nullptr), 1, &got));
  EXPECT_EQ(got, T{0});

  fb::Buffers::device<T> dresult(1, res.device());
  upload(res, dresult, std::vector<T>{T{-7}});
  ASSERT_EQ(wwr::wwrblasSetPointerMode(res.blas().get(), wwr::WWRBLAS_POINTER_MODE_DEVICE),
            wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_TRUE(fb::norm<Norm::l2>(res, 0, static_cast<const T *>(nullptr), 1, dresult.data()));
  ASSERT_EQ(wwr::wwrblasSetPointerMode(res.blas().get(), wwr::WWRBLAS_POINTER_MODE_HOST),
            wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(download_one(res, dresult), T{0});
}

} // namespace
