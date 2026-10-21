// GPU cases for integrals.coulomb.sort_bounds: one ket class sorted descending
// by Iq * Dmax, every gathered array checked bit for bit against its record
// order permuted by a host stable sort of the same keys. The class is one s
// shell of 60 primitives against itself, 3600 records: past the merge sort's
// 2816-key tile, so its merge pass runs too. Bounds, Dmax and D_q are made up,
// with ties and zero Dmax among them, so the order is the keys' and not zeta's.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals;
import free_breakfast.integrals.coulomb.sort_bounds;

namespace {

namespace fb = free_breakfast;

constexpr std::int32_t kPrimitives = 60;
// Any row width will do: the sort only moves rows.
constexpr std::int32_t kHermite = 10;

std::vector<double> download(fb::DeviceResources &res, const double *device, std::size_t n) {
  std::vector<double> host(n);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(double) * n, wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

std::vector<double> pinned(const fb::MirroredBuffer<double> &field, std::size_t n) {
  return {field.pinned().data(), field.pinned().data() + n};
}

TEST(SortBounds, SortsAKetClassDescendingAndGathersEveryArray) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  std::vector<double> exponents;
  std::vector<double> coefficients;
  for (std::int32_t i = 0; i < kPrimitives; ++i) {
    exponents.push_back(0.1 * std::pow(1.2, i));
    coefficients.push_back(1.0);
  }
  const fb::BasisSet basis(
      {fb::Shell(0, exponents, coefficients, fb::Vec<double, 3>{0.1, -0.2, 0.3})});
  const auto classes = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(classes);
  ASSERT_EQ(classes->size(), 1U);
  const fb::PrimitivePairClass &cls = (*classes)[0];
  const auto n = static_cast<std::size_t>(cls.n_rec);
  ASSERT_GT(n, 2816U);

  // Bounds from a short cycle and Dmax from a longer one, so products repeat;
  // every seventh Dmax zero, so its key is -0.0 and must sort last.
  std::vector<double> bound(n);
  std::vector<double> dmax(n);
  std::vector<double> dq(n * kHermite);
  for (std::size_t k = 0; k < n; ++k) {
    bound[k] = 1.0 + static_cast<double>(k % 13);
    dmax[k] = k % 7 == 0 ? 0.0 : std::ldexp(1.0, -static_cast<std::int32_t>(k % 29));
    for (std::size_t q = 0; q < kHermite; ++q) {
      dq[k * kHermite + q] = static_cast<double>(k) + 0.01 * static_cast<double>(q);
    }
  }
  fb::Buffers::device<double> bound_d(n, res.device());
  fb::Buffers::device<double> dmax_d(n, res.device());
  fb::Buffers::device<double> dq_d(n * kHermite, res.device());
  for (const auto &[dst, src] :
       {std::pair{bound_d.data(), &bound}, {dmax_d.data(), &dmax}, {dq_d.data(), &dq}}) {
    ASSERT_EQ(wwr::wwrMemcpyAsync(dst, src->data(), sizeof(double) * src->size(),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              wwr::wwrSuccess);
  }
  // The records' copy to the device is queued on the class's stream.
  ASSERT_EQ(wwr::wwrStreamSynchronize(cls.device_stream()->stream()), wwr::wwrSuccess);

  fb::SortedKets kets(res.device(), cls, kHermite);
  ASSERT_EQ(kets.sort(res.stream(), bound_d.data(), dmax_d.data(), dq_d.data()), wwr::wwrSuccess);

  // The order the device must give: key = -(Iq * Dmax) ascending in total
  // order, ties in record order.
  std::vector<double> key(n);
  std::vector<std::size_t> perm(n);
  for (std::size_t k = 0; k < n; ++k) {
    key[k] = -(bound[k] * dmax[k]);
    perm[k] = k;
  }
  std::ranges::stable_sort(perm, [&](const std::size_t a, const std::size_t b) {
    return std::strong_order(key[a], key[b]) < 0;
  });

  const fb::PrimitivePairClassView view = kets.view();
  EXPECT_EQ(view.la, cls.la);
  EXPECT_EQ(view.lb, cls.lb);
  EXPECT_EQ(view.n_rec, cls.n_rec);
  EXPECT_EQ(view.s, nullptr);
  const auto zeta = pinned(cls.zeta, n);
  const auto p_x = pinned(cls.p_x, n);
  const auto p_y = pinned(cls.p_y, n);
  const auto p_z = pinned(cls.p_z, n);
  const auto zeta_s = download(res, view.zeta, n);
  const auto p_x_s = download(res, view.p_x, n);
  const auto p_y_s = download(res, view.p_y, n);
  const auto p_z_s = download(res, view.p_z, n);
  const auto bound_s = download(res, kets.bound(), n);
  const auto dmax_s = download(res, kets.dmax(), n);
  const auto dq_s = download(res, kets.dq(), n * kHermite);

  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t from = perm[k];
    ASSERT_EQ(std::bit_cast<std::uint64_t>(bound_s[k]), std::bit_cast<std::uint64_t>(bound[from]))
        << "row " << k;
    ASSERT_EQ(std::bit_cast<std::uint64_t>(dmax_s[k]), std::bit_cast<std::uint64_t>(dmax[from]))
        << "row " << k;
    ASSERT_EQ(zeta_s[k], zeta[from]) << "row " << k;
    ASSERT_EQ(p_x_s[k], p_x[from]) << "row " << k;
    ASSERT_EQ(p_y_s[k], p_y[from]) << "row " << k;
    ASSERT_EQ(p_z_s[k], p_z[from]) << "row " << k;
    for (std::size_t q = 0; q < kHermite; ++q) {
      ASSERT_EQ(dq_s[k * kHermite + q], dq[from * kHermite + q]) << "row " << k << ", q " << q;
    }
    if (k > 0) {
      ASSERT_GE(bound_s[k - 1] * dmax_s[k - 1], bound_s[k] * dmax_s[k]) << "row " << k;
    }
  }
  EXPECT_EQ(dmax_s[n - 1], 0.0);
}

} // namespace
