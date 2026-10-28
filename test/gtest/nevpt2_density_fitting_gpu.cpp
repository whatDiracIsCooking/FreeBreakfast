// GPU cases for nevpt2.density_fitting. From the golden B blocks of every
// case in data/nevpt2/ (the demo's DF goldens): h2e and every external slab
// against the golden four-index block at 1e-10 absolute, the demo's
// --check-blocks, whole and walked in several batch sizes, the last batch
// partial. And mo_blocks over H2O/cc-pVDZ's fitted B and DF-RHF orbitals
// (data/density_fitting/): h2e and the slabs that lie inside (pq|uv) and
// (pu|qv) against casscf_integrals' density-fitted ppaa and papa over the
// same B, and B_vc against B_cv. Then the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.active_space;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.nevpt2;

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;
using nevpt2::ExtBlock;

// The issue's bar, the demo's --check-blocks: the same B contracted by a gemm
// here and by NumPy there, so ~1e-15 is expected.
constexpr double kGoldenTol = 1e-10;
// Against casscf_integrals over the same B and C, of the block's max: the same
// products, summed in another order (df_mo_gpu.cpp's kSameFitTol).
constexpr double kSameFitTol = 1e-12;

const std::filesystem::path kNevptDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2";
const std::filesystem::path kDfDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "density_fitting";

constexpr std::array<ExtBlock, 10> kAllBlocks{
    ExtBlock::Sr,    ExtBlock::Si,  ExtBlock::Sijrs, ExtBlock::Sijr, ExtBlock::Srsi,
    ExtBlock::SrsiT, ExtBlock::Srs, ExtBlock::Sij,   ExtBlock::Sir1, ExtBlock::Sir2};

// The golden array each block is: SrsiT is Srsi's, batched along axis 1.
std::string golden_name(const ExtBlock block) {
  switch (block) {
  case ExtBlock::Sr:
    return "h2e_v_Sr";
  case ExtBlock::Si:
    return "h2e_v_Si";
  case ExtBlock::Sijrs:
    return "cvcv";
  case ExtBlock::Sijr:
    return "h2e_v_Sijr";
  case ExtBlock::Srsi:
  case ExtBlock::SrsiT:
    return "h2e_v_Srsi";
  case ExtBlock::Srs:
    return "h2e_v_Srs";
  case ExtBlock::Sij:
    return "h2e_v_Sij";
  case ExtBlock::Sir1:
    return "h2e_v1_Sir";
  case ExtBlock::Sir2:
    return "h2e_v2_Sir";
  }
  return {};
}

fb::NpyArray read_array(const std::filesystem::path &path) {
  auto a = fb::read_npy(path);
  EXPECT_TRUE(a) << path << ": " << (a ? "" : a.error().message);
  return a ? std::move(*a) : fb::NpyArray{};
}

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *d) {
  ASSERT_EQ(wwr::wwrMemcpyAsync(d, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
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

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

std::size_t volume(const std::array<std::int32_t, 4> &shape) {
  return std::ranges::fold_left(
      shape, std::size_t{1},
      [](const std::size_t n, const std::int32_t d) { return n * static_cast<std::size_t>(d); });
}

// The column-major 4-index @p whole of dimensions @p shape, restricted to
// [b0, b1) along @p axis.
std::vector<double> slice(const std::vector<double> &whole,
                          const std::array<std::int32_t, 4> &shape, const std::int32_t axis,
                          const std::int32_t b0, const std::int32_t b1) {
  std::array<std::size_t, 4> n{};
  for (std::size_t k = 0; k < 4; ++k) {
    n[k] = static_cast<std::size_t>(shape[k]);
  }
  std::array<std::size_t, 4> lo{};
  std::array<std::size_t, 4> hi = n;
  lo[static_cast<std::size_t>(axis)] = static_cast<std::size_t>(b0);
  hi[static_cast<std::size_t>(axis)] = static_cast<std::size_t>(b1);
  std::vector<double> out;
  for (std::size_t l = lo[3]; l < hi[3]; ++l) {
    for (std::size_t k = lo[2]; k < hi[2]; ++k) {
      for (std::size_t j = lo[1]; j < hi[1]; ++j) {
        for (std::size_t i = lo[0]; i < hi[0]; ++i) {
          out.push_back(whole[i + n[0] * (j + n[1] * (k + n[2] * l))]);
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// From the golden B blocks
// ---------------------------------------------------------------------------

class Nevpt2DensityFittingGolden : public ::testing::TestWithParam<const char *> {
protected:
  void SetUp() override {
    FBF_REQUIRE_GPU();
    dir_ = kNevptDir / GetParam();
    const fb::NpyArray aa = read_array(dir_ / "B_aa.npy");
    const fb::NpyArray ca = read_array(dir_ / "B_ca.npy");
    const fb::NpyArray va = read_array(dir_ / "B_va.npy");
    const fb::NpyArray cv = read_array(dir_ / "B_cv.npy");
    ASSERT_EQ(aa.shape.size(), 3U);
    ASSERT_EQ(cv.shape.size(), 3U);
    const std::size_t naux = aa.shape[0];
    const std::size_t na = aa.shape[1];
    const std::size_t nc = cv.shape[1];
    const std::size_t nv = cv.shape[2];
    ASSERT_EQ(ca.shape, (std::vector{naux, nc, na}));
    ASSERT_EQ(va.shape, (std::vector{naux, nv, na}));

    // B_vc, B_cv with its orbital pair swapped.
    std::vector<double> vc(naux * nv * nc);
    for (std::size_t i = 0; i < nc; ++i) {
      for (std::size_t r = 0; r < nv; ++r) {
        for (std::size_t q = 0; q < naux; ++q) {
          vc[q + naux * (r + nv * i)] = cv.data[q + naux * (i + nc * r)];
        }
      }
    }

    res_.emplace(0);
    const auto &dev = res_->device();
    blocks_.emplace(nevpt2::MoBlocks{.naux = static_cast<std::int32_t>(naux),
                                     .ncore = static_cast<std::int32_t>(nc),
                                     .ncas = static_cast<std::int32_t>(na),
                                     .nvirt = static_cast<std::int32_t>(nv),
                                     .aa = fb::Buffers::device<double>(aa.data.size(), dev),
                                     .ca = fb::Buffers::device<double>(ca.data.size(), dev),
                                     .va = fb::Buffers::device<double>(va.data.size(), dev),
                                     .cv = fb::Buffers::device<double>(cv.data.size(), dev),
                                     .vc = fb::Buffers::device<double>(vc.size(), dev)});
    upload(*res_, aa.data, blocks_->aa.data());
    upload(*res_, ca.data, blocks_->ca.data());
    upload(*res_, va.data, blocks_->va.data());
    upload(*res_, cv.data, blocks_->cv.data());
    upload(*res_, vc, blocks_->vc.data());
  }

  // The golden block, its shape checked against block_shape.
  std::vector<double> golden(const std::string &name, const std::array<std::int32_t, 4> &shape) {
    const fb::NpyArray g = read_array(dir_ / (name + ".npy"));
    std::vector<std::size_t> want;
    for (const std::int32_t d : shape) {
      want.push_back(static_cast<std::size_t>(d));
    }
    EXPECT_EQ(g.shape, want) << name;
    return g.data;
  }

  // slab over [b0, b1), read back.
  std::vector<double> slab(const ExtBlock block, const std::int32_t b0, const std::int32_t b1) {
    std::array<std::int32_t, 4> shape = nevpt2::block_shape(*blocks_, block);
    shape[static_cast<std::size_t>(nevpt2::batch_axis(block))] = b1 - b0;
    const std::size_t count = volume(shape);
    fb::Buffers::device<double> out(count, res_->device());
    const fb::Status built = nevpt2::slab(*res_, *blocks_, block, b0, b1, out.data());
    EXPECT_TRUE(built) << built.error().message;
    return download(*res_, out.data(), count);
  }

  std::filesystem::path dir_;
  std::optional<fb::DeviceResources> res_;
  std::optional<nevpt2::MoBlocks> blocks_;
};

TEST_P(Nevpt2DensityFittingGolden, H2eMatchesGolden) {
  const std::int32_t a = blocks_->ncas;
  const std::array<std::int32_t, 4> shape{a, a, a, a};
  const std::size_t count = volume(shape);
  fb::Buffers::device<double> out(count, res_->device());
  const fb::Status built = nevpt2::h2e(*res_, *blocks_, out.data());
  ASSERT_TRUE(built) << built.error().message;
  EXPECT_LE(max_diff(download(*res_, out.data(), count), golden("h2e", shape)), kGoldenTol);
}

TEST_P(Nevpt2DensityFittingGolden, WholeSlabsMatchGoldenBlocks) {
  for (const ExtBlock block : kAllBlocks) {
    const std::array<std::int32_t, 4> shape = nevpt2::block_shape(*blocks_, block);
    const std::int32_t extent = shape[static_cast<std::size_t>(nevpt2::batch_axis(block))];
    const std::vector<double> want = golden(golden_name(block), shape);
    const double worst = max_diff(slab(block, 0, extent), want);
    EXPECT_LE(worst, kGoldenTol) << golden_name(block) << " (block "
                                 << static_cast<std::int32_t>(block) << ")";
  }
}

// Each batch size walks the whole extent; where it does not divide it, the
// last batch is partial.
TEST_P(Nevpt2DensityFittingGolden, BatchedSlabsMatchGoldenSlices) {
  for (const ExtBlock block : kAllBlocks) {
    const std::array<std::int32_t, 4> shape = nevpt2::block_shape(*blocks_, block);
    const std::int32_t axis = nevpt2::batch_axis(block);
    const std::int32_t extent = shape[static_cast<std::size_t>(axis)];
    const std::vector<double> want = golden(golden_name(block), shape);
    for (const std::int32_t batch : {1, 2, 3, 4}) {
      for (std::int32_t b0 = 0; b0 < extent; b0 += batch) {
        const std::int32_t b1 = std::min(b0 + batch, extent);
        const double worst = max_diff(slab(block, b0, b1), slice(want, shape, axis, b0, b1));
        EXPECT_LE(worst, kGoldenTol)
            << golden_name(block) << " (block " << static_cast<std::int32_t>(block) << ") batch "
            << batch << " [" << b0 << ", " << b1 << ")";
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2DensityFittingGolden,
                         ::testing::Values("n2_ccpvdz_cas44_df", "n2_ccpvdz_cas44_casscf_df",
                                           "n2_ccpvdz_cas88_df", "n2_ccpvdz_cas1010_df",
                                           "n2_ccpvtz_cas1010_df"));

// ---------------------------------------------------------------------------
// mo_blocks over a fitted B, against casscf_integrals
// ---------------------------------------------------------------------------

// The h2o_ccpvdz case of data/density_fitting/, as df_mo_gpu.cpp reads it:
// the metric, (mn|P) transposed into [P][n][m], and the DF-RHF orbitals.
struct DfCase {
  fb::json entry;
  std::int32_t nao = 0;
  std::int32_t naux = 0;
  std::int32_t nocc = 0;
  std::vector<double> metric;
  std::vector<double> b;
  std::vector<double> c;
};

DfCase read_df_case() {
  auto doc = fb::read_json(kDfDir / "density_fitting.json");
  EXPECT_TRUE(doc) << (doc ? "" : doc.error().message);
  if (!doc) {
    return {};
  }
  for (const auto &c : (*doc)["integrals"]) {
    if (c["name"].get<std::string>() != "h2o_ccpvdz") {
      continue;
    }
    DfCase g;
    g.entry = c;
    g.nao = c["nao"].get<std::int32_t>();
    g.naux = c["naux"].get<std::int32_t>();
    const auto nao = static_cast<std::size_t>(g.nao);
    const auto naux = static_cast<std::size_t>(g.naux);
    g.metric = read_array(kDfDir / c["int2c2e"].get<std::string>()).data;
    g.b.assign(naux * nao * nao, std::numeric_limits<double>::quiet_NaN());
    for (const auto &piece : c["int3c2e"]) {
      // (nao, nao, end - begin), Fortran order: (mn|P) at m + nao (n + nao (P - begin)).
      const auto range = piece["aux"].get<std::vector<std::size_t>>();
      const auto slab = read_array(kDfDir / piece["file"].get<std::string>()).data;
      EXPECT_EQ(slab.size(), nao * nao * (range[1] - range[0]));
      if (slab.size() != nao * nao * (range[1] - range[0])) {
        return {};
      }
      for (std::size_t p = range[0]; p < range[1]; ++p) {
        for (std::size_t n = 0; n < nao; ++n) {
          for (std::size_t m = 0; m < nao; ++m) {
            g.b[p + naux * (n + nao * m)] = slab[m + nao * (n + nao * (p - range[0]))];
          }
        }
      }
    }
    g.nocc = c["mo"]["occupied"].get<std::int32_t>();
    g.c = read_array(kDfDir / c["mo"]["coefficients"].get<std::string>()).data;
    return g;
  }
  ADD_FAILURE() << "no integrals case h2o_ccpvdz";
  return {};
}

TEST(Nevpt2DensityFitting, MoBlocksMatchCasscfIntegrals) {
  FBF_REQUIRE_GPU();
  const DfCase g = read_df_case();
  ASSERT_GT(g.naux, 0);
  fb::DeviceResources res(0);

  std::vector<fb::Atom> atoms;
  for (const auto &atom : g.entry["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  const fb::Molecule molecule(res, std::move(atoms));
  const auto stamp = fb::resolve_basis(g.entry["basis"].get<std::string>());
  ASSERT_TRUE(stamp) << stamp.error().message;
  const auto basis = fb::place(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(basis) << basis.error().message;
  ASSERT_EQ(basis->nao_sphe(), static_cast<std::uint32_t>(g.nao));

  fb::Buffers::device<double> metric(g.metric.size(), res.device());
  fb::Buffers::device<double> b(g.b.size(), res.device());
  upload(res, g.metric, metric.data());
  upload(res, g.b, b.data());
  const fb::Status fitted = fb::density_fit(res, g.nao, g.naux, metric.data(), b.data());
  ASSERT_TRUE(fitted) << fitted.error().message;
  const fb::SphericalMatrixAO h(res, *basis);
  fb::SphericalMatrixAO orbitals(res, *basis);
  upload(res, g.c, orbitals.data());

  // Two occupied and two virtual orbitals active, the rest core and virtual.
  constexpr std::int32_t kNcas = 4;
  const std::int32_t ncore = g.nocc - 2;
  auto ints = fb::casscf_integrals(res, *basis, molecule, h, b, g.naux, orbitals, ncore, kNcas);
  ASSERT_TRUE(ints) << ints.error().message;
  auto blocks =
      nevpt2::mo_blocks(res, g.naux, g.nao, g.nao, b.data(), orbitals.data(), ncore, kNcas);
  ASSERT_TRUE(blocks) << blocks.error().message;
  ASSERT_EQ(blocks->nvirt, g.nao - ncore - kNcas);

  const auto n = static_cast<std::size_t>(g.nao);
  const auto na = static_cast<std::size_t>(kNcas);
  const auto c0 = std::size_t{0};
  const auto a0 = static_cast<std::size_t>(ncore);
  const auto v0 = a0 + na;
  const std::vector<double> eri = download(res, ints->hamiltonian.eri.data(), na * na * na * na);
  const std::vector<double> ppaa = download(res, ints->ppaa.data(), n * n * na * na);
  const std::vector<double> papa = download(res, ints->papa.data(), n * n * na * na);
  // (pq|uv) and (pu|qv), p and q absolute, u and v active from 0.
  const auto pq_uv = [&](std::size_t p, std::size_t q, std::size_t u, std::size_t v) {
    return ppaa[p + n * (q + n * (u + na * v))];
  };
  const auto pu_qv = [&](std::size_t p, std::size_t u, std::size_t q, std::size_t v) {
    return papa[p + n * (q + n * (u + na * v))];
  };

  // h2e[t, u, v, w] = (tv|uw).
  {
    const std::size_t count = na * na * na * na;
    fb::Buffers::device<double> out(count, res.device());
    const fb::Status built = nevpt2::h2e(res, *blocks, out.data());
    ASSERT_TRUE(built) << built.error().message;
    const std::vector<double> got = download(res, out.data(), count);
    std::vector<double> want;
    for (std::size_t w = 0; w < na; ++w) {
      for (std::size_t v = 0; v < na; ++v) {
        for (std::size_t u = 0; u < na; ++u) {
          for (std::size_t t = 0; t < na; ++t) {
            want.push_back(eri[t + na * (v + na * (u + na * w))]);
          }
        }
      }
    }
    EXPECT_LE(max_diff(got, want), kSameFitTol * max_abs(want)) << "h2e";
  }

  // Each slab inside ppaa or papa, element [i0, i1, i2, i3] of the physicist
  // block, its indices relative to their own spaces.
  using Element = std::function<double(std::size_t, std::size_t, std::size_t, std::size_t)>;
  const std::vector<std::pair<ExtBlock, Element>> checks{
      // Sr [r,a,b,c] = (rb|ac)
      {ExtBlock::Sr, [&](auto r, auto a, auto bb, auto c) { return pq_uv(v0 + r, a0 + bb, a, c); }},
      // Si [t,u,i,v] = (it|uv)
      {ExtBlock::Si, [&](auto t, auto u, auto i, auto v) { return pq_uv(c0 + i, a0 + t, u, v); }},
      // Sir1 [r,t,i,u] = (ri|tu)
      {ExtBlock::Sir1, [&](auto r, auto t, auto i, auto u) { return pq_uv(v0 + r, c0 + i, t, u); }},
      // Srs [r,s,t,u] = (rt|su)
      {ExtBlock::Srs, [&](auto r, auto s, auto t, auto u) { return pu_qv(v0 + r, t, v0 + s, u); }},
      // Sij [t,u,i,j] = (it|ju)
      {ExtBlock::Sij, [&](auto t, auto u, auto i, auto j) { return pu_qv(c0 + i, t, c0 + j, u); }},
      // Sir2 [r,u,t,i] = (rt|iu)
      {ExtBlock::Sir2, [&](auto r, auto u, auto t, auto i) { return pu_qv(v0 + r, t, c0 + i, u); }},
  };
  for (const auto &[block, element] : checks) {
    const std::array<std::int32_t, 4> shape = nevpt2::block_shape(*blocks, block);
    const std::int32_t extent = shape[static_cast<std::size_t>(nevpt2::batch_axis(block))];
    const std::size_t count = volume(shape);
    fb::Buffers::device<double> out(count, res.device());
    const fb::Status built = nevpt2::slab(res, *blocks, block, 0, extent, out.data());
    ASSERT_TRUE(built) << built.error().message;
    const std::vector<double> got = download(res, out.data(), count);
    std::vector<double> want;
    for (std::size_t l = 0; l < static_cast<std::size_t>(shape[3]); ++l) {
      for (std::size_t k = 0; k < static_cast<std::size_t>(shape[2]); ++k) {
        for (std::size_t j = 0; j < static_cast<std::size_t>(shape[1]); ++j) {
          for (std::size_t i = 0; i < static_cast<std::size_t>(shape[0]); ++i) {
            want.push_back(element(i, j, k, l));
          }
        }
      }
    }
    ASSERT_GT(max_abs(want), 0.0) << static_cast<std::int32_t>(block);
    EXPECT_LE(max_diff(got, want), kSameFitTol * max_abs(want))
        << "block " << static_cast<std::int32_t>(block);
  }

  // B_vc is B_cv with its orbital pair swapped: a permute, so exact.
  const auto naux = static_cast<std::size_t>(g.naux);
  const auto nc = static_cast<std::size_t>(blocks->ncore);
  const auto nv = static_cast<std::size_t>(blocks->nvirt);
  const std::vector<double> cv = download(res, blocks->cv.data(), naux * nc * nv);
  const std::vector<double> vc = download(res, blocks->vc.data(), naux * nc * nv);
  std::vector<double> want(vc.size());
  for (std::size_t i = 0; i < nc; ++i) {
    for (std::size_t r = 0; r < nv; ++r) {
      for (std::size_t q = 0; q < naux; ++q) {
        want[q + naux * (r + nv * i)] = cv[q + naux * (i + nc * r)];
      }
    }
  }
  EXPECT_EQ(vc, want);
}

// ---------------------------------------------------------------------------
// The batch axes, and the refusals
// ---------------------------------------------------------------------------

TEST(Nevpt2DensityFittingRefusals, BatchOutsideTheExtentAndTooManyOrbitalsAreInvalid) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const auto &dev = res.device();
  constexpr std::int32_t kNaux = 3;
  constexpr std::int32_t kNcore = 2;
  constexpr std::int32_t kNcas = 2;
  constexpr std::int32_t kNvirt = 3;
  const nevpt2::MoBlocks blocks{.naux = kNaux,
                                .ncore = kNcore,
                                .ncas = kNcas,
                                .nvirt = kNvirt,
                                .aa = fb::Buffers::device<double>(kNaux * kNcas * kNcas, dev),
                                .ca = fb::Buffers::device<double>(kNaux * kNcore * kNcas, dev),
                                .va = fb::Buffers::device<double>(kNaux * kNvirt * kNcas, dev),
                                .cv = fb::Buffers::device<double>(kNaux * kNcore * kNvirt, dev),
                                .vc = fb::Buffers::device<double>(kNaux * kNvirt * kNcore, dev)};
  for (const auto &[b0, b1] : {std::pair{-1, 1}, std::pair{2, 1}, std::pair{0, kNvirt + 1}}) {
    const fb::Status built = nevpt2::slab(res, blocks, ExtBlock::Sr, b0, b1, nullptr);
    ASSERT_FALSE(built) << b0 << ", " << b1;
    EXPECT_EQ(built.error().kind, fb::ErrorKind::InvalidConfig) << built.error().message;
  }
  // Si batches over the core: kNvirt is past it.
  const fb::Status core = nevpt2::slab(res, blocks, ExtBlock::Si, 0, kNvirt, nullptr);
  ASSERT_FALSE(core);
  EXPECT_EQ(core.error().kind, fb::ErrorKind::InvalidConfig);
  // An empty slab writes nothing, so may take no output.
  EXPECT_TRUE(nevpt2::slab(res, blocks, ExtBlock::Sr, 1, 1, nullptr));

  for (const auto &[ncore, ncas] : {std::pair{-1, 2}, std::pair{2, -1}, std::pair{4, 4}}) {
    const auto made = nevpt2::mo_blocks(res, kNaux, 7, 7, nullptr, nullptr, ncore, ncas);
    ASSERT_FALSE(made) << ncore << ", " << ncas;
    EXPECT_EQ(made.error().kind, fb::ErrorKind::InvalidConfig) << made.error().message;
  }
}

TEST(Nevpt2DensityFitting, BatchAxesAreTheDemos) {
  EXPECT_EQ(nevpt2::batch_axis(ExtBlock::Si), 2);
  EXPECT_EQ(nevpt2::batch_axis(ExtBlock::Sij), 2);
  EXPECT_EQ(nevpt2::batch_axis(ExtBlock::SrsiT), 1);
  for (const ExtBlock block : {ExtBlock::Sr, ExtBlock::Sijrs, ExtBlock::Sijr, ExtBlock::Srsi,
                               ExtBlock::Srs, ExtBlock::Sir1, ExtBlock::Sir2}) {
    EXPECT_EQ(nevpt2::batch_axis(block), 0) << static_cast<std::int32_t>(block);
  }
}

} // namespace
