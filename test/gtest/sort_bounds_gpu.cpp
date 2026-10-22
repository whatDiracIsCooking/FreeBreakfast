// GPU cases for coulomb/sort_bounds: one ket class sorted descending by
// Iq * Dmax, every gathered array checked bit for bit against its record order
// permuted by a host stable sort of the same keys. The class has 3600 records,
// past the merge sort's 2816-key tile, so its merge pass runs too. Its zeta,
// P, bounds, Dmax and D_q are made up, with ties and zero Dmax among them: the
// sort only reads la, lb and n_rec of the class, and moves the rest.
//
// The launcher is reached through its bridge header, which brings in wwr's
// textual runtime surface (wwr/runtime_api.h). clang rejects that next to the
// same surface imported from wwr.runtime_api in a plain TU (an ODR mismatch on
// wwrMallocAsync), so this TU imports nothing that carries it, and manages its
// device memory and stream through the textual wwr* calls, as
// exchange_helpers_gpu.cpp does.

#include <gtest/gtest.h>

// After every standard header: under HIP a libc++ header must not follow the
// vendor runtime header this brings in (see primitives.h).
#include "gpu_requirement.h"
#include "sort_bounds_bridge.h"

import std;

namespace {

namespace fb = free_breakfast;

constexpr std::size_t kRecords = 3600;
// Any number of Hermite functions will do: the sort only moves columns.
constexpr std::int32_t kHermite = 10;

// A device array of n T, freed on destruction.
template <typename T>
class DeviceArray {
public:
  explicit DeviceArray(const std::size_t n) {
    void *p = nullptr;
    EXPECT_EQ(wwr::wwrMalloc(&p, sizeof(T) * n), wwr::wwrSuccess);
    data_ = static_cast<T *>(p);
  }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  DeviceArray(DeviceArray &&other) noexcept : data_(std::exchange(other.data_, nullptr)) {}
  DeviceArray &operator=(DeviceArray &&) = delete;
  ~DeviceArray() {
    if (data_ != nullptr) {
      EXPECT_EQ(wwr::wwrFree(data_), wwr::wwrSuccess);
    }
  }
  [[nodiscard]] T *data() const { return data_; }

private:
  T *data_ = nullptr;
};

// The stream every launch goes on, made and destroyed by each case.
class Stream {
public:
  Stream() { EXPECT_EQ(wwr::wwrStreamCreate(&stream_), wwr::wwrSuccess); }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  ~Stream() { EXPECT_EQ(wwr::wwrStreamDestroy(stream_), wwr::wwrSuccess); }
  [[nodiscard]] wwr::wwrStream_t get() const { return stream_; }

private:
  wwr::wwrStream_t stream_ = nullptr;
};

template <typename T>
DeviceArray<T> upload(const Stream &stream, const std::vector<T> &host) {
  DeviceArray<T> dev(host.size());
  EXPECT_EQ(wwr::wwrMemcpyAsync(dev.data(), host.data(), sizeof(T) * host.size(),
                                wwr::wwrMemcpyHostToDevice, stream.get()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream.get()), wwr::wwrSuccess);
  return dev;
}

std::vector<double> download(const Stream &stream, const double *device, std::size_t n) {
  std::vector<double> host(n);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(double) * n, wwr::wwrMemcpyDeviceToHost,
                                stream.get()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream.get()), wwr::wwrSuccess);
  return host;
}

std::uint64_t bits(const double x) {
  return std::bit_cast<std::uint64_t>(x);
}

TEST(SortBounds, SortsAKetClassDescendingAndGathersEveryArray) {
  FBF_REQUIRE_GPU();
  const Stream stream;
  constexpr std::size_t n = kRecords;

  // Bounds from a short cycle and Dmax from a longer one, so products repeat;
  // every seventh Dmax zero, so its key is -0.0 and must sort last. zeta and
  // P are distinct per record, so a misplaced gather shows.
  std::vector<double> zeta(n);
  std::vector<double> p_x(n);
  std::vector<double> p_y(n);
  std::vector<double> p_z(n);
  std::vector<double> bound(n);
  std::vector<double> dmax(n);
  std::vector<double> dq(n * kHermite);
  for (std::size_t k = 0; k < n; ++k) {
    const auto x = static_cast<double>(k);
    zeta[k] = 0.1 + x;
    p_x[k] = 0.25 * x;
    p_y[k] = -0.5 * x;
    p_z[k] = 1.0 / (1.0 + x);
    bound[k] = 1.0 + static_cast<double>(k % 13);
    dmax[k] = k % 7 == 0 ? 0.0 : std::ldexp(1.0, -static_cast<std::int32_t>(k % 29));
    // q-major, as dq writes D_q: row q, column k.
    for (std::size_t q = 0; q < kHermite; ++q) {
      dq[q * n + k] = x + 0.01 * static_cast<double>(q);
    }
  }
  const auto zeta_d = upload(stream, zeta);
  const auto p_x_d = upload(stream, p_x);
  const auto p_y_d = upload(stream, p_y);
  const auto p_z_d = upload(stream, p_z);
  const auto bound_d = upload(stream, bound);
  const auto dmax_d = upload(stream, dmax);
  const auto dq_d = upload(stream, dq);

  const fb::PrimitivePairClassView cls{
      .la = 0,
      .lb = 0,
      .n_rec = static_cast<std::int32_t>(n),
      .zeta = zeta_d.data(),
      .p_x = p_x_d.data(),
      .p_y = p_y_d.data(),
      .p_z = p_z_d.data(),
  };
  const DeviceArray<double> key(n);
  const DeviceArray<double> key_alt(n);
  const DeviceArray<std::uint32_t> perm(n);
  const DeviceArray<std::uint32_t> perm_alt(n);
  const DeviceArray<double> zeta_s(n);
  const DeviceArray<double> p_x_s(n);
  const DeviceArray<double> p_y_s(n);
  const DeviceArray<double> p_z_s(n);
  const DeviceArray<double> bound_s(n);
  const DeviceArray<double> dmax_s(n);
  const DeviceArray<double> dq_s(n * kHermite);
  ASSERT_EQ(fb::device::sort_bounds_kets(stream.get(), cls, kHermite, bound_d.data(), dmax_d.data(),
                                         dq_d.data(), key.data(), key_alt.data(), perm.data(),
                                         perm_alt.data(), zeta_s.data(), p_x_s.data(), p_y_s.data(),
                                         p_z_s.data(), bound_s.data(), dmax_s.data(), dq_s.data()),
            wwr::wwrSuccess);

  // The order the device must give: key = -(Iq * Dmax) ascending in total
  // order, ties in record order.
  std::vector<double> keys(n);
  std::vector<std::size_t> order(n);
  for (std::size_t k = 0; k < n; ++k) {
    keys[k] = -(bound[k] * dmax[k]);
    order[k] = k;
  }
  std::ranges::stable_sort(order, [&](const std::size_t a, const std::size_t b) {
    return std::strong_order(keys[a], keys[b]) < 0;
  });

  const auto zeta_h = download(stream, zeta_s.data(), n);
  const auto p_x_h = download(stream, p_x_s.data(), n);
  const auto p_y_h = download(stream, p_y_s.data(), n);
  const auto p_z_h = download(stream, p_z_s.data(), n);
  const auto bound_h = download(stream, bound_s.data(), n);
  const auto dmax_h = download(stream, dmax_s.data(), n);
  const auto dq_h = download(stream, dq_s.data(), n * kHermite);

  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t from = order[k];
    ASSERT_EQ(bits(bound_h[k]), bits(bound[from])) << "row " << k;
    ASSERT_EQ(bits(dmax_h[k]), bits(dmax[from])) << "row " << k;
    ASSERT_EQ(bits(zeta_h[k]), bits(zeta[from])) << "row " << k;
    ASSERT_EQ(bits(p_x_h[k]), bits(p_x[from])) << "row " << k;
    ASSERT_EQ(bits(p_y_h[k]), bits(p_y[from])) << "row " << k;
    ASSERT_EQ(bits(p_z_h[k]), bits(p_z[from])) << "row " << k;
    for (std::size_t q = 0; q < kHermite; ++q) {
      ASSERT_EQ(bits(dq_h[q * n + k]), bits(dq[q * n + from])) << "row " << k << ", q " << q;
    }
    if (k > 0) {
      ASSERT_GE(bound_h[k - 1] * dmax_h[k - 1], bound_h[k] * dmax_h[k]) << "row " << k;
    }
  }
  EXPECT_EQ(dmax_h[n - 1], 0.0);
}

} // namespace
