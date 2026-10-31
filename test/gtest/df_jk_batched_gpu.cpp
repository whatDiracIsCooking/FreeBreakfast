// GPU cases for the block forms of integrals.eri.density_fitting.jk: J and the
// two-factor K of several densities in one call, over data/density_fitting/'s
// h2o_ccpvdz B fitted by density_fit, against the one-density calls density
// by density; a symmetric indefinite block shaped as CPHF's D_U = 2 L R^T;
// and the empty, negative and too-large counts.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.integrals.eri.density_fitting.jk;

namespace {

namespace fb = free_breakfast;

// The issue's bar: each density's J and K from the block against the
// one-density call, of the largest |J| or |K| of the block. The same products
// in other gemm shapes, so rounding.
constexpr double kBatchedTol = 1e-13;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "density_fitting";

// The h2o_ccpvdz case's metric, naux x naux, and (P|mn) as the naux x nao^2
// matrix the fit takes, element (P, n + nao m) at P + naux (n + nao m).
struct Golden {
  std::int32_t nao = 0;
  std::int32_t naux = 0;
  std::int32_t occupied = 0;
  std::vector<double> metric;
  std::vector<double> b;
};

Golden read_golden() {
  const auto doc = fb::read_json(kDataDir / "density_fitting.json");
  EXPECT_TRUE(doc) << (doc ? "" : doc.error().message);
  if (!doc) {
    return {};
  }
  for (const auto &c : (*doc)["integrals"]) {
    if (c["name"].get<std::string>() != "h2o_ccpvdz") {
      continue;
    }
    Golden g;
    g.nao = c["nao"].get<std::int32_t>();
    g.naux = c["naux"].get<std::int32_t>();
    g.occupied = c["jk"]["occupied"].get<std::int32_t>();
    const auto nao = static_cast<std::size_t>(g.nao);
    const auto naux = static_cast<std::size_t>(g.naux);
    const auto metric = fb::read_npy(kDataDir / c["int2c2e"].get<std::string>());
    EXPECT_TRUE(metric);
    if (!metric) {
      return {};
    }
    g.metric = metric->data;
    // Each piece is (nao, nao, end - begin), Fortran order: (mn|P) at
    // m + nao (n + nao (P - begin)). Transposed here into [P][n][m].
    g.b.assign(naux * nao * nao, std::numeric_limits<double>::quiet_NaN());
    for (const auto &piece : c["int3c2e"]) {
      const auto range = piece["aux"].get<std::vector<std::size_t>>();
      const auto slab = fb::read_npy(kDataDir / piece["file"].get<std::string>());
      EXPECT_TRUE(slab);
      if (!slab) {
        return {};
      }
      for (std::size_t p = range[0]; p < range[1]; ++p) {
        for (std::size_t n = 0; n < nao; ++n) {
          for (std::size_t m = 0; m < nao; ++m) {
            g.b[p + naux * (n + nao * m)] = slab->data[m + nao * (n + nao * (p - range[0]))];
          }
        }
      }
    }
    return g;
  }
  ADD_FAILURE() << "no integrals case h2o_ccpvdz";
  return {};
}

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *d) {
  ASSERT_EQ(wwr::wwrMemcpyAsync(d, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());
}

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// count elements uniform on [-1/2, 1/2) from a fixed seed.
std::vector<double> random_values(const std::size_t count, const std::uint64_t seed) {
  std::mt19937_64 engine{seed};
  std::uniform_real_distribution<double> uniform{-0.5, 0.5};
  std::vector<double> a(count);
  std::ranges::generate(a, [&] { return uniform(engine); });
  return a;
}

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

// The fitted B on the device.
class DfJkBatched : public ::testing::Test {
protected:
  void SetUp() override {
    FBF_REQUIRE_GPU();
    g_ = read_golden();
    ASSERT_GT(g_.naux, 0);
    res_.emplace(0);
    fb::Buffers::device<double> metric(g_.metric.size(), res_->device());
    b_.emplace(g_.b.size(), res_->device());
    upload(*res_, g_.metric, metric.data());
    upload(*res_, g_.b, b_->data());
    const fb::Status fitted = fb::density_fit(*res_, g_.nao, g_.naux, metric.data(), b_->data());
    ASSERT_TRUE(fitted) << fitted.error().message;
  }

  std::size_t nn() const {
    return static_cast<std::size_t>(g_.nao) * static_cast<std::size_t>(g_.nao);
  }

  // J of count packed densities, by one block call and by count one-density
  // calls, read back: {block, one at a time}.
  std::pair<std::vector<double>, std::vector<double>>
  coulomb_both_ways(const std::vector<double> &densities) {
    const std::size_t count = densities.size() / nn();
    fb::Buffers::device<double> d(densities.size(), res_->device());
    fb::Buffers::device<double> j(densities.size(), res_->device());
    upload(*res_, densities, d.data());
    const fb::Status block = fb::df_coulomb(*res_, g_.nao, g_.naux, static_cast<int32_t>(count),
                                            b_->data(), d.data(), j.data());
    EXPECT_TRUE(block) << block.error().message;
    std::vector<double> batched = download(*res_, j.data(), j.num_elements());
    for (std::size_t x = 0; x < count; ++x) {
      const fb::Status one = fb::df_coulomb(*res_, g_.nao, g_.naux, b_->data(),
                                            d.data() + (x * nn()), j.data() + (x * nn()));
      EXPECT_TRUE(one) << one.error().message;
    }
    return {std::move(batched), download(*res_, j.data(), j.num_elements())};
  }

  // K of count two-factor densities L_x R_x^T (each nao x rank, packed side by
  // side), by one block call and by count one-density calls, read back.
  std::pair<std::vector<double>, std::vector<double>>
  exchange_both_ways(const std::vector<double> &left, const std::vector<double> &right,
                     const std::int32_t rank) {
    const std::size_t factor = static_cast<std::size_t>(g_.nao) * static_cast<std::size_t>(rank);
    const std::size_t count = left.size() / factor;
    fb::Buffers::device<double> d_left(left.size(), res_->device());
    fb::Buffers::device<double> d_right(right.size(), res_->device());
    fb::Buffers::device<double> k(count * nn(), res_->device());
    upload(*res_, left, d_left.data());
    upload(*res_, right, d_right.data());
    const fb::Status block =
        fb::df_exchange(*res_, g_.nao, g_.naux, rank, static_cast<int32_t>(count), b_->data(),
                        d_left.data(), d_right.data(), k.data());
    EXPECT_TRUE(block) << block.error().message;
    std::vector<double> batched = download(*res_, k.data(), k.num_elements());
    for (std::size_t x = 0; x < count; ++x) {
      const fb::Status one =
          fb::df_exchange(*res_, g_.nao, g_.naux, rank, b_->data(), d_left.data() + (x * factor),
                          d_right.data() + (x * factor), k.data() + (x * nn()));
      EXPECT_TRUE(one) << one.error().message;
    }
    return {std::move(batched), download(*res_, k.data(), k.num_elements())};
  }

  void expect_close(const std::pair<std::vector<double>, std::vector<double>> &both,
                    const std::string &what) {
    const double scale = max_abs(both.second);
    const double worst = max_diff(both.first, both.second);
    ASSERT_GT(scale, 0.0) << what;
    EXPECT_LE(worst, kBatchedTol * scale) << what << ": max|block - one at a time| = " << worst
                                          << ", " << worst / scale << " of max " << scale;
    std::cout << std::format("[ df jk    ] {}: max|block - one at a time| / max {:.3e}\n", what,
                             worst / scale);
  }

  Golden g_;
  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Buffers::device<double>> b_;
};

// Five random densities, neither symmetric nor definite: J reads D as a
// vector, so any D.
TEST_F(DfJkBatched, CoulombOfABlockMatchesOneDensityAtATime) {
  expect_close(coulomb_both_ways(random_values(5 * nn(), 595)), "J of 5 random D");
}

// Four random A_x B_x^T of rank 3, each its own pair of factors.
TEST_F(DfJkBatched, ExchangeOfABlockMatchesOneDensityAtATime) {
  constexpr std::int32_t kRank = 3;
  const std::size_t factors = 4 * static_cast<std::size_t>(g_.nao) * kRank;
  expect_close(exchange_both_ways(random_values(factors, 596), random_values(factors, 597), kRank),
               "K of 4 random A B^T");
}

// CPHF's block: D_x = 2 L_x R_x^T with L_x = [T_x C] and R_x = [C T_x], T_x
// random and C shared, so each D_x symmetric and indefinite. Its J of D_x and
// K of L_x R_x^T, rank 2 n_occ, as response::cphf builds them.
TEST_F(DfJkBatched, ResponseDensitiesOfABlockMatchOneDensityAtATime) {
  constexpr std::size_t kCount = 9;
  const auto nao = static_cast<std::size_t>(g_.nao);
  const auto occ = static_cast<std::size_t>(g_.occupied);
  const std::vector<double> c = random_values(nao * occ, 598);
  const std::vector<double> t = random_values(kCount * nao * occ, 599);
  std::vector<double> left(kCount * 2 * nao * occ);
  std::vector<double> right(left.size());
  std::vector<double> densities(kCount * nn(), 0.0);
  for (std::size_t x = 0; x < kCount; ++x) {
    const auto tx = t.begin() + static_cast<std::ptrdiff_t>(x * nao * occ);
    const auto lx = left.begin() + static_cast<std::ptrdiff_t>(x * 2 * nao * occ);
    const auto rx = right.begin() + static_cast<std::ptrdiff_t>(x * 2 * nao * occ);
    std::copy_n(tx, nao * occ, lx);
    std::ranges::copy(c, lx + static_cast<std::ptrdiff_t>(nao * occ));
    std::ranges::copy(c, rx);
    std::copy_n(tx, nao * occ, rx + static_cast<std::ptrdiff_t>(nao * occ));
    for (std::size_t q = 0; q < nao; ++q) {
      for (std::size_t p = 0; p < nao; ++p) {
        double sum = 0.0;
        for (std::size_t i = 0; i < 2 * occ; ++i) {
          sum += lx[static_cast<std::ptrdiff_t>(p + nao * i)] *
                 rx[static_cast<std::ptrdiff_t>(q + nao * i)];
        }
        densities[p + nao * q + nn() * x] = 2.0 * sum;
      }
    }
  }
  expect_close(coulomb_both_ways(densities), "J of 9 D_U");
  expect_close(exchange_both_ways(left, right, 2 * g_.occupied), "K of 9 L R^T");
}

// naux = 0 or rank = 0 is J or K = 0 over every density, written over
// whatever was there; nao = 0 or count = 0 writes nothing; a negative count,
// or naux count rank past int32_t, is InvalidConfig.
TEST(DfJkBatchedEmpty, EmptyCountsAndNegativeCounts) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kNao = 3;
  constexpr std::int32_t kCount = 2;
  const std::vector<double> ones(kCount * kNao * kNao, 1.0);
  const std::vector<double> zeros(ones.size(), 0.0);
  fb::Buffers::device<double> out(ones.size(), res.device());

  upload(res, ones, out.data());
  ASSERT_TRUE(fb::df_coulomb(res, kNao, 0, kCount, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), zeros);
  upload(res, ones, out.data());
  ASSERT_TRUE(fb::df_exchange(res, kNao, 0, 2, kCount, nullptr, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), zeros);
  upload(res, ones, out.data());
  ASSERT_TRUE(fb::df_exchange(res, kNao, 4, 0, kCount, nullptr, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), zeros);

  upload(res, ones, out.data());
  ASSERT_TRUE(fb::df_coulomb(res, kNao, 4, 0, nullptr, nullptr, out.data()));
  ASSERT_TRUE(fb::df_exchange(res, kNao, 4, 2, 0, nullptr, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), ones);
  EXPECT_TRUE(fb::df_coulomb(res, 0, 4, kCount, nullptr, nullptr, nullptr));
  EXPECT_TRUE(fb::df_exchange(res, 0, 4, 2, kCount, nullptr, nullptr, nullptr, nullptr));

  constexpr std::int32_t kHuge = 1 << 16; // kHuge^2 = 2^32
  for (const fb::Status refused :
       {fb::df_coulomb(res, -1, 4, kCount, nullptr, nullptr, nullptr),
        fb::df_coulomb(res, 3, -1, kCount, nullptr, nullptr, nullptr),
        fb::df_coulomb(res, 3, 4, -1, nullptr, nullptr, nullptr),
        fb::df_coulomb(res, kHuge, 4, kCount, nullptr, nullptr, nullptr),
        fb::df_exchange(res, -1, 4, 2, kCount, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange(res, 3, -1, 2, kCount, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange(res, 3, 4, -1, kCount, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange(res, 3, 4, 2, -1, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange(res, 3, kHuge, kHuge / 2, kCount, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange(res, kHuge, 4, 2, kCount, nullptr, nullptr, nullptr, nullptr)}) {
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
  }
}

} // namespace
