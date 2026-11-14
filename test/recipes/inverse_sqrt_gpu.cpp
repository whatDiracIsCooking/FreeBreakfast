// GPU cases for recipes/orthogonalize's inverse_sqrt kernel, against the host,
// bit for bit, in float and double.
//
// The bar is bitwise, by analysis: the kernel's 1 / sqrt(x) is a correctly
// rounded sqrt and a correctly rounded division in IEEE-754 on both vendors,
// as is the host's T{1} / std::sqrt(x), so the two are the same bits. On HIP
// that holds only with contraction off in the kernel: contracted, the AMDGPU
// back end folds 1 / sqrt into a refined rsq, an ulp off on most of these
// elements (inverse_sqrt.cu has the pragma and why). The elements cover each
// side of the floor (below it, at it, one ulp above it, a negative value, both
// zeros) and ordinary values over the type's range, repeated past three of the
// launch's blocks, so a block boundary falls inside them; elements past count
// must be untouched. rust/recipes/sys/tests/inverse_sqrt.rs is the same case
// for the Rust fork.
//
// The launcher is reached through its bridge header, which brings in wwr's
// textual runtime surface (wwr/runtime_api.h). clang rejects that next to the
// same surface imported from wwr.runtime_api in a plain TU, so this TU imports
// nothing that carries it, and manages its device memory and stream through
// the textual wwr* calls, as test/gtest/sort_bounds_gpu.cpp does.

#include <gtest/gtest.h>

// After every standard header: under HIP a libc++ header must not follow the
// vendor runtime header this brings in (see primitives.h).
#include "gpu_requirement.h"
#include "inverse_sqrt_bridge.h"

import std;

namespace {

namespace fb = free_breakfast;

// Elements the kernel maps; then kTail more it must not touch.
constexpr std::int32_t kCount = 777;
constexpr std::int32_t kTail = 5;

// kCount + kTail elements, the pool repeated, for the orthogonalizer's
// default floor.
template <typename T>
std::vector<T> elements(const T floor) {
  const std::array<T, 17> pool{
      T{-1},
      T{-0.0},
      T{0},
      std::numeric_limits<T>::min(),
      floor / T{2},
      floor,
      std::nextafter(floor, std::numeric_limits<T>::infinity()),
      T{1e-8},
      T{0.25},
      T{0.5},
      T{1},
      T{2},
      T{3},
      T{7.123456789},
      T{1e10},
      T{1e30},
      std::numeric_limits<T>::max(),
  };
  std::vector<T> host(kCount + kTail);
  for (std::size_t i = 0; i < host.size(); ++i) {
    host[i] = pool[i % pool.size()];
  }
  return host;
}

// Every element as the host maps it, bit for bit; the tail as it was.
template <typename T>
void maps_each_element_as_the_host_does() {
  // orthogonalize's kOrthogonalizeEigenvalueFloor, which this TU cannot import.
  const T floor = T{1e-10};
  const std::vector<T> host = elements(floor);
  const std::size_t bytes = sizeof(T) * host.size();

  wwr::wwrStream_t stream = nullptr;
  ASSERT_EQ(wwr::wwrStreamCreate(&stream), wwr::wwrSuccess);
  void *device = nullptr;
  ASSERT_EQ(wwr::wwrMalloc(&device, bytes), wwr::wwrSuccess);
  ASSERT_EQ(wwr::wwrMemcpyAsync(device, host.data(), bytes, wwr::wwrMemcpyHostToDevice, stream),
            wwr::wwrSuccess);
  ASSERT_EQ(fb::device::inverse_sqrt<T>(stream, kCount, static_cast<T *>(device), floor),
            wwr::wwrSuccess);
  std::vector<T> got(host.size());
  ASSERT_EQ(wwr::wwrMemcpyAsync(got.data(), device, bytes, wwr::wwrMemcpyDeviceToHost, stream),
            wwr::wwrSuccess);
  ASSERT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrFree(device), wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamDestroy(stream), wwr::wwrSuccess);

  using Bits = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
  std::string misses;
  for (std::size_t i = 0; i < host.size(); ++i) {
    const T x = host[i];
    const T expected = i >= static_cast<std::size_t>(kCount) ? x
                       : x <= floor                          ? T{0}
                                                             : T{1} / std::sqrt(x);
    if (std::bit_cast<Bits>(got[i]) != std::bit_cast<Bits>(expected)) {
      misses += std::format("[{}] {:a}: {:a}, expected {:a}\n", i, x, got[i], expected);
    }
  }
  EXPECT_TRUE(misses.empty()) << "differ:\n" << misses;
}

TEST(InverseSqrt, MapsEachDoubleAsTheHostDoes) {
  FBF_REQUIRE_GPU();
  maps_each_element_as_the_host_does<double>();
}

TEST(InverseSqrt, MapsEachFloatAsTheHostDoes) {
  FBF_REQUIRE_GPU();
  maps_each_element_as_the_host_does<float>();
}

} // namespace
