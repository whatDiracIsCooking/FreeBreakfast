// The footprints (#1044) of the NEVPT2 library routines (#1052): nevpt2::rdms
// and rdms_adjoint, mo_blocks, h2e, slab and class_energies. Each is held to
// the pool peak of one call by test/support/pool_footprint.h's two bars,
// sound (the measured peak is no more than the footprint's) and tight
// (within G x A). The adaptive routines, rdms' tile and the classes' slab
// batch, are measured at their footprint, the least they run with (one
// determinant, a batch of one); then under a pool cap with the budget
// unconstrained, where they must take more and stay under the cap; and their
// energies at the least are held to the unconstrained run's.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.npy;
import free_breakfast.ci;
import free_breakfast.nevpt2;

#include "pool_footprint.h"

namespace {

namespace fb = free_breakfast;
namespace fbt = free_breakfast::test;
namespace nevpt2 = free_breakfast::nevpt2;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2";

// The SC and PC energies at the least scratch against the unconstrained
// run's: the tile and the batch only reorder sums, each a handful of
// rounding errors of a 1e-1 Eh quantity.
constexpr double kReorderTol = 1e-10;

// @p call held to @p fp on @p res, as test/gtest/footprint_gpu.cpp holds the
// RHF chain's routines: once to grow the workspace to the most the call asks
// of it (exactly the footprint's on a fresh DeviceResources), then again,
// measured, with @p live allocations live at its peak.
template <typename Call>
void expect_routine_footprint(fb::DeviceResources &res, const fb::MemoryFootprint &fp,
                              const std::size_t live, Call &&call) {
  const std::size_t device_before = res.device_workspace(0).num_elements();
  call();
  EXPECT_EQ(res.device_workspace(0).num_elements(), std::max(device_before, fp.device.workspace));
  const fbt::PoolPeak peak = fbt::measure_pool_peak(*res.device(), call);
  fbt::expect_footprint(peak, {.persistent = fp.device.persistent, .scratch = fp.device.scratch},
                        live);
}

fb::Buffers::device<double> on_device(fb::DeviceResources &res, std::span<const double> host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync());
  return d;
}

std::vector<double> random_vector(const std::size_t count, const std::uint64_t seed) {
  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<double> v(count);
  for (double &x : v) {
    x = uniform(engine);
  }
  return v;
}

fb::NpyArray read_array(const std::filesystem::path &path) {
  auto a = fb::read_npy(path);
  EXPECT_TRUE(a) << path << ": " << (a ? "" : a.error().message);
  return a ? std::move(*a) : fb::NpyArray{};
}

// A column-major npy array in row-major order, as the RDMs are read.
std::vector<double> row_major(const fb::NpyArray &a) {
  std::vector<std::size_t> stride(a.shape.size(), 1);
  for (std::size_t d = 1; d < a.shape.size(); ++d) {
    stride[d] = stride[d - 1] * a.shape[d - 1];
  }
  std::vector<double> out(a.data.size());
  for (std::size_t at = 0; at < out.size(); ++at) {
    std::size_t rest = at;
    std::size_t cm = 0;
    for (std::size_t d = a.shape.size(); d-- > 0;) {
      cm += (rest % a.shape[d]) * stride[d];
      rest /= a.shape[d];
    }
    out[at] = a.data[cm];
  }
  return out;
}

// ---------------------------------------------------------------------------
// rdms and rdms_adjoint: one scratch block, at one determinant its footprint
// ---------------------------------------------------------------------------

// A random CI vector over n orbitals and n / 2 electrons a spin, and random
// (tu|vw): what the build reads, which the footprint does not depend on.
struct RdmInputs {
  fb::ci::DeviceStringSet strings;
  std::int64_t ndet;
  fb::Buffers::device<double> bra;
  fb::Buffers::device<double> ket;
  fb::Buffers::device<double> eri;
};

RdmInputs rdm_inputs(fb::DeviceResources &res, const std::int32_t n) {
  auto strings = fb::ci::DeviceStringSet::create(res, n, n / 2);
  EXPECT_TRUE(strings);
  const auto ndet = static_cast<std::int64_t>(strings->size()) * strings->size();
  const auto n4 = static_cast<std::size_t>(n) * n * n * n;
  return {.strings = std::move(*strings),
          .ndet = ndet,
          .bra = on_device(res, random_vector(static_cast<std::size_t>(ndet), 1)),
          .ket = on_device(res, random_vector(static_cast<std::size_t>(ndet), 2)),
          .eri = on_device(res, random_vector(n4, 3))};
}

TEST(Nevpt2Footprint, RdmsAtOneDeterminant) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kN = 6;
  fb::DeviceResources inputs{0};
  RdmInputs in = rdm_inputs(inputs, kN);
  const auto n6 = static_cast<std::size_t>(kN) * kN * kN * kN * kN * kN;
  const auto n4 = static_cast<std::size_t>(kN) * kN * kN * kN;
  fb::Buffers::device<double> dm1(kN * kN, inputs.device());
  fb::Buffers::device<double> dm2(n4, inputs.device());
  fb::Buffers::device<double> dm3(n6, inputs.device());
  fb::Buffers::device<double> f3ca(n6, inputs.device());
  fb::Buffers::device<double> f3ac(n6, inputs.device());
  for (const bool transition : {false, true}) {
    for (const bool f3 : {true, false}) {
      SCOPED_TRACE(std::format("transition {} f3 {}", transition, f3));
      const fb::MemoryFootprint fp = nevpt2::rdms_footprint(kN, 1, transition, f3);
      EXPECT_EQ(fp.device.persistent + fp.device.workspace, 0U);
      fb::DeviceResources res{0};
      // The one scratch block, budgeted at its footprint: a tile of one.
      expect_routine_footprint(res, fp, 1, [&] {
        const auto tiles = nevpt2::rdms(
            res, in.strings, in.strings, in.bra.data(), transition ? in.ket.data() : in.bra.data(),
            in.eri.data(), dm1.data(), dm2.data(), dm3.data(), f3 ? f3ca.data() : nullptr,
            f3 ? f3ac.data() : nullptr, fp.device.scratch);
        ASSERT_TRUE(tiles) << tiles.error().message;
        EXPECT_EQ(*tiles, in.ndet);
      });
    }
  }
}

TEST(Nevpt2Footprint, RdmsAdjointAtOneDeterminant) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kN = 6;
  fb::DeviceResources inputs{0};
  RdmInputs in = rdm_inputs(inputs, kN);
  const auto n6 = static_cast<std::size_t>(kN) * kN * kN * kN * kN * kN;
  const auto n4 = static_cast<std::size_t>(kN) * kN * kN * kN;
  const fb::Buffers::device<double> dm3_bar = on_device(inputs, random_vector(n6, 4));
  const fb::Buffers::device<double> f3ca_bar = on_device(inputs, random_vector(n6, 5));
  const fb::Buffers::device<double> f3ac_bar = on_device(inputs, random_vector(n6, 6));
  fb::Buffers::device<double> bra_bar(static_cast<std::size_t>(in.ndet), inputs.device());
  fb::Buffers::device<double> ket_bar(static_cast<std::size_t>(in.ndet), inputs.device());
  fb::Buffers::device<double> eri_bar(n4, inputs.device());
  for (const bool transition : {false, true}) {
    for (const bool f3 : {true, false}) {
      SCOPED_TRACE(std::format("transition {} f3 {}", transition, f3));
      const fb::MemoryFootprint fp = nevpt2::rdms_adjoint_footprint(kN, 1, transition, f3);
      fb::DeviceResources res{0};
      expect_routine_footprint(res, fp, 1, [&] {
        const double *const f3ca = f3 ? f3ca_bar.data() : nullptr;
        const double *const f3ac = f3 ? f3ac_bar.data() : nullptr;
        const auto tiles =
            transition
                ? nevpt2::rdms_adjoint(res, in.strings, in.strings, in.bra.data(), in.ket.data(),
                                       in.eri.data(), dm3_bar.data(), f3ca, f3ac, bra_bar.data(),
                                       ket_bar.data(), eri_bar.data(), fp.device.scratch)
                : nevpt2::rdms_adjoint(res, in.strings, in.strings, in.bra.data(), in.eri.data(),
                                       dm3_bar.data(), f3ca, f3ac, bra_bar.data(), eri_bar.data(),
                                       fp.device.scratch);
        ASSERT_TRUE(tiles) << tiles.error().message;
        EXPECT_EQ(*tiles, in.ndet);
      });
    }
  }
}

// Under a 64 MiB pool cap with no max_bytes, the build sizes its tile from
// the budget: wider than one determinant, narrower than all 4900 of
// CAS(8,8) (165 MB of scratch whole), and the pool's peak under the cap.
TEST(Nevpt2Footprint, RdmsTileFromTheBudgetUnderACap) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kN = 8;
  constexpr std::uint64_t kCap = std::uint64_t{64} << 20;
  fb::DeviceResources res{0, fb::DeviceResources::kDefaultPoolReleaseThreshold, 0, kCap};
  RdmInputs in = rdm_inputs(res, kN);
  const auto n6 = static_cast<std::size_t>(kN) * kN * kN * kN * kN * kN;
  fb::Buffers::device<double> dm3(n6, res.device());
  fb::Buffers::device<double> f3ca(n6, res.device());
  fb::Buffers::device<double> f3ac(n6, res.device());
  ASSERT_GT(nevpt2::rdms_footprint(kN, in.ndet).device.scratch, kCap);
  res.device()->reset_pool_high_water();
  const auto tiles = nevpt2::rdms(res, in.strings, in.strings, in.bra.data(), in.eri.data(),
                                  dm3.data(), f3ca.data(), f3ac.data());
  ASSERT_TRUE(tiles) << tiles.error().message;
  ASSERT_TRUE(res.sync());
  EXPECT_GT(*tiles, 1);
  EXPECT_LT(*tiles, in.ndet);
  EXPECT_LE(res.device()->pool_usage().used_high, kCap);
}

// ---------------------------------------------------------------------------
// mo_blocks, h2e and slab
// ---------------------------------------------------------------------------

// Random B and C: 3 core, 4 active and 9 virtual orbitals over 16 AOs and 40
// fitting functions.
struct BlockInputs {
  static constexpr std::int32_t kNaux = 40;
  static constexpr std::int32_t kNao = 16;
  static constexpr std::int32_t kNcore = 3;
  static constexpr std::int32_t kNcas = 4;
  static constexpr std::int32_t kNvirt = kNao - kNcore - kNcas;
  fb::Buffers::device<double> b;
  fb::Buffers::device<double> c;
};

BlockInputs block_inputs(fb::DeviceResources &res) {
  using I = BlockInputs;
  return {
      .b = on_device(res, random_vector(static_cast<std::size_t>(I::kNaux) * I::kNao * I::kNao, 7)),
      .c = on_device(res, random_vector(static_cast<std::size_t>(I::kNao) * I::kNao, 8))};
}

TEST(Nevpt2Footprint, MoBlocks) {
  FBF_REQUIRE_GPU();
  using I = BlockInputs;
  fb::DeviceResources inputs{0};
  const BlockInputs in = block_inputs(inputs);
  const fb::MemoryFootprint fp =
      nevpt2::mo_blocks_footprint(I::kNaux, I::kNao, I::kNao, I::kNcore, I::kNcas);
  fb::DeviceResources res{0};
  // The five blocks; each b_mo's X is the workspace.
  expect_routine_footprint(res, fp, 5, [&] {
    const auto blocks = nevpt2::mo_blocks(res, I::kNaux, I::kNao, I::kNao, in.b.data(), in.c.data(),
                                          I::kNcore, I::kNcas);
    ASSERT_TRUE(blocks) << blocks.error().message;
  });
}

TEST(Nevpt2Footprint, H2eAndSlabs) {
  FBF_REQUIRE_GPU();
  using I = BlockInputs;
  fb::DeviceResources inputs{0};
  const BlockInputs in = block_inputs(inputs);
  const auto blocks = nevpt2::mo_blocks(inputs, I::kNaux, I::kNao, I::kNao, in.b.data(),
                                        in.c.data(), I::kNcore, I::kNcas);
  ASSERT_TRUE(blocks) << blocks.error().message;
  ASSERT_TRUE(inputs.sync());
  {
    fb::Buffers::device<double> out(
        static_cast<std::size_t>(I::kNcas) * I::kNcas * I::kNcas * I::kNcas, inputs.device());
    fb::DeviceResources res{0};
    // The gemm's product.
    expect_routine_footprint(res, nevpt2::h2e_footprint(I::kNcas), 1,
                             [&] { ASSERT_TRUE(nevpt2::h2e(res, *blocks, out.data())); });
  }
  for (const nevpt2::ExtBlock block :
       {nevpt2::ExtBlock::Sr, nevpt2::ExtBlock::Si, nevpt2::ExtBlock::Sijrs, nevpt2::ExtBlock::Sijr,
        nevpt2::ExtBlock::Srsi, nevpt2::ExtBlock::SrsiT, nevpt2::ExtBlock::Srs,
        nevpt2::ExtBlock::Sij, nevpt2::ExtBlock::Sir1, nevpt2::ExtBlock::Sir2}) {
    SCOPED_TRACE(std::format("block {}", static_cast<std::int32_t>(block)));
    EXPECT_EQ(nevpt2::block_shape(*blocks, block),
              nevpt2::block_shape(I::kNcore, I::kNcas, I::kNvirt, block));
    std::array<std::int32_t, 4> shape = nevpt2::block_shape(*blocks, block);
    constexpr std::int32_t kWidth = 2;
    shape[static_cast<std::size_t>(nevpt2::batch_axis(block))] = kWidth;
    fb::Buffers::device<double> out(
        static_cast<std::size_t>(shape[0]) * shape[1] * shape[2] * shape[3], inputs.device());
    const fb::MemoryFootprint fp =
        nevpt2::slab_footprint(I::kNcore, I::kNcas, I::kNvirt, block, kWidth);
    fb::DeviceResources res{0};
    // The gemm's product, but for Sijrs, which writes out directly.
    expect_routine_footprint(res, fp, block == nevpt2::ExtBlock::Sijrs ? 0 : 1, [&] {
      ASSERT_TRUE(nevpt2::slab(res, *blocks, block, 1, 1 + kWidth, out.data()));
    });
  }
}

// ---------------------------------------------------------------------------
// class_energies, over a case of test/gtest/data/nevpt2/
// ---------------------------------------------------------------------------

// One case's MO blocks, inputs and RDMs on the device, as
// nevpt2_energy_gpu.cpp loads them, and its active electrons, tr dm1.
struct EnergyCase {
  std::optional<nevpt2::MoBlocks> blocks;
  std::vector<fb::Buffers::device<double>> arrays;
  nevpt2::EnergyInputs in;
  std::int32_t nelecas = 0;
};

EnergyCase energy_case(fb::DeviceResources &res, const std::string &name) {
  const std::filesystem::path dir = kDataDir / name;
  const fb::NpyArray aa = read_array(dir / "B_aa.npy");
  const fb::NpyArray cv = read_array(dir / "B_cv.npy");
  EnergyCase g;
  if (aa.shape.size() != 3 || cv.shape.size() != 3) {
    ADD_FAILURE() << dir << ": B blocks missing";
    return g;
  }
  const std::size_t naux = aa.shape[0];
  const std::size_t na = aa.shape[1];
  const std::size_t nc = cv.shape[1];
  const std::size_t nv = cv.shape[2];
  std::vector<double> vc(naux * nv * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    for (std::size_t r = 0; r < nv; ++r) {
      for (std::size_t q = 0; q < naux; ++q) {
        vc[q + naux * (r + nv * i)] = cv.data[q + naux * (i + nc * r)];
      }
    }
  }
  g.blocks.emplace(nevpt2::MoBlocks{.naux = static_cast<std::int32_t>(naux),
                                    .ncore = static_cast<std::int32_t>(nc),
                                    .ncas = static_cast<std::int32_t>(na),
                                    .nvirt = static_cast<std::int32_t>(nv),
                                    .aa = on_device(res, aa.data),
                                    .ca = on_device(res, read_array(dir / "B_ca.npy").data),
                                    .va = on_device(res, read_array(dir / "B_va.npy").data),
                                    .cv = on_device(res, cv.data),
                                    .vc = on_device(res, vc)});
  const auto put = [&](const char *array, const bool rdm) {
    const fb::NpyArray a = read_array(dir / (std::string{array} + ".npy"));
    g.arrays.push_back(on_device(res, rdm ? row_major(a) : a.data));
    return static_cast<const double *>(g.arrays.back().data());
  };
  g.in.h1e = put("h1e", false);
  g.in.h2e = put("h2e", false);
  g.in.e_core = put("e_core", false);
  g.in.e_virt = put("e_virt", false);
  g.in.h1e_v_sr = put("h1e_v_Sr", false);
  g.in.h1e_v_si = put("h1e_v_Si", false);
  g.in.h1e_v_sir = put("h1e_v_Sir", false);
  g.in.dm1 = put("dm1", true);
  g.in.dm2 = put("dm2", true);
  g.in.dm3 = put("dm3", true);
  g.in.f3ca = put("f3ca", true);
  g.in.f3ac = put("f3ac", true);
  const fb::NpyArray dm1 = read_array(dir / "dm1.npy");
  double trace = 0.0;
  for (std::size_t p = 0; p < na; ++p) {
    trace += dm1.data[p + (na * p)];
  }
  g.nelecas = static_cast<std::int32_t>(std::lround(trace));
  return g;
}

class Nevpt2ClassEnergiesFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

// At a batch of one, the footprint of kBatchFromBudget, and at the whole
// extent, batch 0.
TEST_P(Nevpt2ClassEnergiesFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  const auto [name, batch] = GetParam();
  fb::DeviceResources inputs{0};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const fb::MemoryFootprint fp =
      nevpt2::class_energies_footprint(b.ncore, b.ncas, b.nvirt, g.nelecas, batch);
  EXPECT_EQ(fp.device.persistent, 0U);
  fb::DeviceResources res{0};
  // At Si's walk, the peak class here: delta, its six intermediates, the
  // slab, its product and the tables.
  expect_routine_footprint(res, fp, 10, [&] {
    const auto e = nevpt2::class_energies(res, b, g.in, batch);
    ASSERT_TRUE(e) << e.error().message;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2ClassEnergiesFootprint,
                         ::testing::Combine(::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                                              std::string{"n2_ccpvdz_cas88_df"}),
                                            ::testing::Values(1, 0)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

// The SC and PC energies with every adaptive part at its least, rdms' tile
// at one determinant and the slab batch at one, against kBatchFromBudget
// with rdms' default on an uncapped pool: the same to kReorderTol. Then
// under a cap the batch from the budget takes more than its least, and the
// pool stays under the cap.
TEST(Nevpt2Footprint, EnergiesAtTheLeastMatchUnconstrained) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  EnergyCase g = energy_case(res, "n2_ccpvdz_cas44_df");
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const std::int32_t n = b.ncas;
  const auto n4 = static_cast<std::size_t>(n) * n * n * n;
  const auto n6 = n4 * static_cast<std::size_t>(n) * n;

  // rdms over the case's CI vector, (tu|vw) = h2e[t, v, u, w].
  const fb::NpyArray h2e = read_array(kDataDir / "n2_ccpvdz_cas44_df" / "h2e.npy");
  std::vector<double> eri(n4);
  const auto un = static_cast<std::size_t>(n);
  for (std::size_t t = 0; t < un; ++t) {
    for (std::size_t u = 0; u < un; ++u) {
      for (std::size_t v = 0; v < un; ++v) {
        for (std::size_t w = 0; w < un; ++w) {
          eri[t + un * (u + un * (v + un * w))] = h2e.data[t + un * (v + un * (u + un * w))];
        }
      }
    }
  }
  auto strings = fb::ci::DeviceStringSet::create(res, n, g.nelecas / 2);
  ASSERT_TRUE(strings) << strings.error().message;
  const fb::Buffers::device<double> ci =
      on_device(res, row_major(read_array(kDataDir / "n2_ccpvdz_cas44_df" / "ci.npy")));
  const fb::Buffers::device<double> eri_d = on_device(res, eri);
  struct Run {
    fb::Buffers::device<double> dm3, f3ca, f3ac;
    nevpt2::ClassEnergies sc;
    nevpt2::PcClassEnergies pc;
  };
  const auto run = [&](const bool least) {
    Run r{.dm3 = fb::Buffers::device<double>(n6, res.device()),
          .f3ca = fb::Buffers::device<double>(n6, res.device()),
          .f3ac = fb::Buffers::device<double>(n6, res.device())};
    const std::size_t tile = nevpt2::rdms_footprint(n).device.scratch;
    const auto tiles = least ? nevpt2::rdms(res, *strings, *strings, ci.data(), eri_d.data(),
                                            r.dm3.data(), r.f3ca.data(), r.f3ac.data(), tile)
                             : nevpt2::rdms(res, *strings, *strings, ci.data(), eri_d.data(),
                                            r.dm3.data(), r.f3ca.data(), r.f3ac.data());
    EXPECT_TRUE(tiles);
    nevpt2::EnergyInputs in = g.in;
    in.dm3 = r.dm3.data();
    in.f3ca = r.f3ca.data();
    in.f3ac = r.f3ac.data();
    const std::int32_t batch = least ? 1 : nevpt2::kBatchFromBudget;
    auto sc = nevpt2::class_energies(res, b, in, batch);
    EXPECT_TRUE(sc) << sc.error().message;
    auto pc = nevpt2::pc_class_energies(res, b, in, batch);
    EXPECT_TRUE(pc) << pc.error().message;
    if (sc && pc) {
      r.sc = *sc;
      r.pc = *pc;
    }
    return r;
  };
  const Run least = run(true);
  const Run whole = run(false);
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(least.sc.energies[k], whole.sc.energies[k], kReorderTol) << "SC class " << k;
    EXPECT_NEAR(least.pc.energies[k], whole.pc.energies[k], kReorderTol) << "PC class " << k;
  }
  EXPECT_NEAR(least.sc.total, whole.sc.total, kReorderTol);
  EXPECT_NEAR(least.pc.total, whole.pc.total, kReorderTol);
}

// Under a pool cap the batch from the budget takes more than a batch of one,
// less than the whole extent, and stays under the cap. At CAS(4,4) the walks
// are most of the call (n^6 is 32 KB): the cap, halfway between the least
// and the whole past the headroom and the workspace, leaves the peak class's
// walk part of its extent.
TEST(Nevpt2Footprint, ClassEnergiesBatchFromTheBudgetUnderACap) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources inputs{0};
  const EnergyCase g = energy_case(inputs, "n2_ccpvdz_cas44_df");
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const fb::MemoryFootprint least =
      nevpt2::class_energies_footprint(b.ncore, b.ncas, b.nvirt, g.nelecas, 1);
  const fb::MemoryFootprint whole =
      nevpt2::class_energies_footprint(b.ncore, b.ncas, b.nvirt, g.nelecas, 0);
  ASSERT_LT(least.device.scratch, whole.device.scratch);
  // A cap halfway between the least and the whole, above the 32 MiB of
  // headroom the walk leaves and the workspace the first call grows.
  const std::uint64_t cap = (std::uint64_t{32} << 20) + least.device.workspace +
                            ((least.device.scratch + whole.device.scratch) / 2);
  fb::DeviceResources res{0, fb::DeviceResources::kDefaultPoolReleaseThreshold, 0, cap};
  ASSERT_TRUE(nevpt2::class_energies(res, b, g.in, nevpt2::kBatchFromBudget));
  const fbt::PoolPeak peak = fbt::measure_pool_peak(*res.device(), [&] {
    const auto e = nevpt2::class_energies(res, b, g.in, nevpt2::kBatchFromBudget);
    ASSERT_TRUE(e) << e.error().message;
  });
  EXPECT_GT(peak.used, least.device.scratch);
  EXPECT_LT(peak.used, whole.device.scratch);
  EXPECT_LE(res.device()->pool_usage().used_high, cap);
}

} // namespace
