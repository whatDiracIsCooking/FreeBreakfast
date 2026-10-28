// response::pack_orbital_pairs and unpack_orbital_pairs on the device: the
// packed order element by element against a host enumeration of the pairs,
// unpack as its antisymmetric inverse, and the refusals. Runs the device, so
// needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.response;

namespace {

namespace fb = free_breakfast;

// Copies in res.stream()'s order, waited for.
std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *data) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(data, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

// The packed order, as src/response/README.md documents it: (p, q) per packed
// element.
std::vector<std::pair<std::size_t, std::size_t>> packed_pairs(std::size_t nmo, std::size_t ncore,
                                                              std::size_t ncas) {
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  for (std::size_t q = 0; q < ncore; ++q) {
    for (std::size_t p = ncore; p < nmo; ++p) {
      pairs.emplace_back(p, q);
    }
  }
  for (std::size_t q = ncore; q < ncore + ncas; ++q) {
    for (std::size_t p = ncore + ncas; p < nmo; ++p) {
      pairs.emplace_back(p, q);
    }
  }
  return pairs;
}

// The packed order, element by element, and unpack as its antisymmetric
// inverse, over every combination of empty and non-empty spaces.
TEST(ResponseOrbitalPairs, PacksAndUnpacksTheNonRedundantPairs) {
  FBF_REQUIRE_GPU();
  fb::NodeResources node;
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  constexpr std::size_t kNmo = 7;
  const std::vector<std::pair<std::int32_t, std::int32_t>> spaces{{2, 3}, {0, 3}, {3, 0}, {0, 7},
                                                                  {7, 0}, {2, 5}, {1, 1}};
  for (const auto &[ncore, ncas] : spaces) {
    const auto order =
        packed_pairs(kNmo, static_cast<std::size_t>(ncore), static_cast<std::size_t>(ncas));
    std::vector<double> full(kNmo * kNmo);
    for (std::size_t i = 0; i < full.size(); ++i) {
      full[i] = static_cast<double>(i + 1);
    }
    fb::Buffers::device<double> full_device(full.size(), res.device());
    fb::Buffers::device<double> packed(std::max<std::size_t>(order.size(), 1), res.device());
    upload(res, full, full_device.data());
    ASSERT_TRUE(
        fb::response::pack_orbital_pairs(res, kNmo, ncore, ncas, full_device.data(), packed.data())
            .has_value());
    const std::vector<double> got = download(res, packed.data(), order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      const auto [p, q] = order[i];
      EXPECT_EQ(got[i], full[p + (kNmo * q)]) << ncore << ", " << ncas << ": element " << i;
    }

    std::vector<double> x(order.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = static_cast<double>(i) + 0.5;
    }
    if (!x.empty()) {
      upload(res, x, packed.data());
    }
    ASSERT_TRUE(fb::response::unpack_orbital_pairs(res, kNmo, ncore, ncas, packed.data(),
                                                   full_device.data())
                    .has_value());
    const std::vector<double> kappa = download(res, full_device.data(), full.size());
    std::vector<double> want(kNmo * kNmo, 0.0);
    for (std::size_t i = 0; i < order.size(); ++i) {
      const auto [p, q] = order[i];
      want[p + (kNmo * q)] = x[i];
      want[q + (kNmo * p)] = -x[i];
    }
    for (std::size_t i = 0; i < want.size(); ++i) {
      EXPECT_EQ(kappa[i], want[i]) << ncore << ", " << ncas << ": element " << i;
    }
  }
  const fb::Status refused = fb::response::pack_orbital_pairs(res, kNmo, 4, 4, nullptr, nullptr);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
  const fb::Status negative =
      fb::response::unpack_orbital_pairs(res, kNmo, -1, 2, nullptr, nullptr);
  ASSERT_FALSE(negative.has_value());
  EXPECT_EQ(negative.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
