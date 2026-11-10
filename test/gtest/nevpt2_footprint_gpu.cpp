// The footprints (#1044) of the NEVPT2 library routines (#1052): nevpt2::rdms
// and rdms_adjoint, mo_blocks, h2e, slab and class_energies, their reverse
// passes, the SC and PC forward tangents (#1182, #1183), the SC and PC
// Hessian-vector products (#1180, #1184) and the PC metric lifts. Each is
// held to
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
#include "test_workspaces.h"

import free_breakfast.recipes.diagonalize; // syevd_lwork

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
  const std::size_t device_before = res.workspace_high_water().device.workspace;
  call();
  EXPECT_EQ(res.workspace_high_water().device.workspace,
            std::max(device_before, fp.device.workspace));
  const fbt::PoolPeak peak = fbt::measure_pool_peak(*res.device(), call);
  std::cout << std::format("[ footprint ] measured {} (G {}), footprint {} + workspace {}\n",
                           peak.used, peak.granularity, fp.device.persistent + fp.device.scratch,
                           fp.device.workspace);
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
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
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
      fb::DeviceResources res{0, fb::test::kWorkspaces};
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
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
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
      fb::DeviceResources res{0, fb::test::kWorkspaces};
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

// Under a 96 MiB pool cap with no max_bytes, the build sizes its tile from
// the budget less the 64 MiB headroom: wider than one determinant, narrower
// than all 4900 of CAS(8,8) (165 MB of scratch whole), and the pool's peak
// under the cap.
TEST(Nevpt2Footprint, RdmsTileFromTheBudgetUnderACap) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kN = 8;
  constexpr std::uint64_t kCap = std::uint64_t{96} << 20;
  fb::DeviceResources res{0, {}, fb::DeviceResources::kDefaultPoolReleaseThreshold, 0, kCap};
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
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const BlockInputs in = block_inputs(inputs);
  const fb::MemoryFootprint fp =
      nevpt2::mo_blocks_footprint(I::kNaux, I::kNao, I::kNao, I::kNcore, I::kNcas);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
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
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const BlockInputs in = block_inputs(inputs);
  const auto blocks = nevpt2::mo_blocks(inputs, I::kNaux, I::kNao, I::kNao, in.b.data(),
                                        in.c.data(), I::kNcore, I::kNcas);
  ASSERT_TRUE(blocks) << blocks.error().message;
  ASSERT_TRUE(inputs.sync());
  {
    fb::Buffers::device<double> out(
        static_cast<std::size_t>(I::kNcas) * I::kNcas * I::kNcas * I::kNcas, inputs.device());
    fb::DeviceResources res{0, fb::test::kWorkspaces};
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
    fb::DeviceResources res{0, fb::test::kWorkspaces};
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
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const fb::MemoryFootprint fp =
      nevpt2::class_energies_footprint(b.ncore, b.ncas, b.nvirt, g.nelecas, batch);
  EXPECT_EQ(fp.device.persistent, 0U);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
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

// syevd's work at every order through @p max, diagonalize's own query: what
// a PC footprint's workspace is sized from.
std::vector<std::size_t> lworks(fb::DeviceResources &res, const std::int32_t max) {
  std::vector<std::size_t> out;
  for (std::int32_t order = 0; order <= max; ++order) {
    out.push_back(fb::syevd_lwork<double>(res, order).value());
  }
  return out;
}

class Nevpt2PcClassEnergiesFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

// pc_class_energies at a batch of one and at the whole extent, its modes
// dropped as the run drops them (PcClassEnergies::dropped, the data's).
TEST_P(Nevpt2PcClassEnergiesFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  const auto [name, batch] = GetParam();
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const auto pc = nevpt2::pc_class_energies(inputs, b, g.in, batch);
  ASSERT_TRUE(pc) << pc.error().message;
  const std::vector<std::size_t> lwork = lworks(inputs, b.ncas * b.ncas * b.ncas);
  const fb::MemoryFootprint fp =
      nevpt2::pc_class_energies_footprint(b.ncore, b.ncas, b.nvirt, batch, lwork, pc->dropped);
  EXPECT_EQ(fp.device.persistent, 0U);
  std::cout << std::format("dropped {}\n", pc->dropped);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  expect_routine_footprint(res, fp, 10, [&] {
    if (std::getenv("TRACE_PC") != nullptr) {
      setenv("FBF_TRACE", "1", 1);
    }
    const auto e = nevpt2::pc_class_energies(res, b, g.in, batch);
    unsetenv("FBF_TRACE");
    ASSERT_TRUE(e) << e.error().message;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2PcClassEnergiesFootprint,
                         ::testing::Combine(::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                                              std::string{"n2_ccpvdz_cas88_df"}),
                                            ::testing::Values(1, 0)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

// Every bar class_energies_adjoint takes, zeroed, over @p b's counts: the
// gradient's, Λ over the blocks' nmo among them.
struct Bars {
  std::vector<fb::Buffers::device<double>> held;
  nevpt2::EnergyBars bars;
};

Bars every_bar(fb::DeviceResources &res, const nevpt2::MoBlocks &b) {
  const auto n = static_cast<std::size_t>(b.ncas);
  const auto nc = static_cast<std::size_t>(b.ncore);
  const auto nv = static_cast<std::size_t>(b.nvirt);
  const std::size_t nmo = nc + n + nv;
  Bars out;
  out.held.reserve(13);
  const auto zero = [&](const std::size_t count) {
    out.held.emplace_back(count, res.device());
    return out.held.back().data();
  };
  out.bars = {.h1e = zero(n * n),
              .h2e = zero(n * n * n * n),
              .dm1 = zero(n * n),
              .dm2 = zero(n * n * n * n),
              .dm3 = zero(n * n * n * n * n * n),
              .f3ca = zero(n * n * n * n * n * n),
              .f3ac = zero(n * n * n * n * n * n),
              .h1e_v_sr = zero(nv * n),
              .h1e_v_si = zero(n * nc),
              .h1e_v_sir = zero(nv * nc),
              .e_core = zero(nc),
              .e_virt = zero(nv),
              .lambda = zero(static_cast<std::size_t>(b.naux) * nmo * nmo)};
  return out;
}

class Nevpt2ClassEnergiesAdjointFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

// class_energies_adjoint at a batch of one and at the whole extent.
TEST_P(Nevpt2ClassEnergiesAdjointFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  const auto [name, batch] = GetParam();
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const Bars bars = every_bar(inputs, b);
  const fb::MemoryFootprint fp =
      nevpt2::class_energies_adjoint_footprint(b.naux, b.ncore, b.ncas, b.nvirt, g.nelecas, batch);
  EXPECT_EQ(fp.device.persistent, 0U);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  expect_routine_footprint(res, fp, 10, [&] {
    if (std::getenv("TRACE_ADJ") != nullptr) {
      setenv("FBF_TRACE", "1", 1);
    }
    const auto e = nevpt2::class_energies_adjoint(res, b, g.in, batch, bars.bars);
    unsetenv("FBF_TRACE");
    ASSERT_TRUE(e) << e.error().message;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2ClassEnergiesAdjointFootprint,
                         ::testing::Combine(::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                                              std::string{"n2_ccpvdz_cas88_df"}),
                                            ::testing::Values(1, 0)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

class Nevpt2PcClassEnergiesAdjointFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

// pc_class_energies_adjoint at a batch of one and at the whole extent, its
// modes dropped as the run drops them; and the footprint at the rank that
// holds the most, no less.
TEST_P(Nevpt2PcClassEnergiesAdjointFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  const auto [name, batch] = GetParam();
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const Bars bars = every_bar(inputs, b);
  const auto pc = nevpt2::pc_class_energies(inputs, b, g.in, batch);
  ASSERT_TRUE(pc) << pc.error().message;
  const std::vector<std::size_t> lwork = lworks(inputs, b.ncas * b.ncas * b.ncas);
  const fb::MemoryFootprint fp = nevpt2::pc_class_energies_adjoint_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, batch, lwork, pc->dropped);
  const fb::MemoryFootprint most = nevpt2::pc_effective_hamiltonian_adjoint_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, 1, batch, lwork);
  EXPECT_GE(most.device.scratch, fp.device.scratch);
  EXPECT_GE(most.device.workspace, fp.device.workspace);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  expect_routine_footprint(res, fp, 10, [&] {
    if (std::getenv("TRACE_ADJ") != nullptr) {
      setenv("FBF_TRACE", "1", 1);
    }
    const auto e = nevpt2::pc_class_energies_adjoint(res, b, g.in, batch, bars.bars);
    unsetenv("FBF_TRACE");
    ASSERT_TRUE(e) << e.error().message;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2PcClassEnergiesAdjointFootprint,
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
  fb::DeviceResources res{0, fb::test::kWorkspaces};
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
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, "n2_ccpvdz_cas44_df");
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const fb::MemoryFootprint least =
      nevpt2::class_energies_footprint(b.ncore, b.ncas, b.nvirt, g.nelecas, 1);
  const fb::MemoryFootprint whole =
      nevpt2::class_energies_footprint(b.ncore, b.ncas, b.nvirt, g.nelecas, 0);
  ASSERT_LT(least.device.scratch, whole.device.scratch);
  // A cap halfway between the least and the whole, above the 64 MiB of
  // headroom the walk leaves and the workspace the first call grows.
  const std::uint64_t cap = fb::kPoolHeadroomBytes + least.device.workspace +
                            ((least.device.scratch + whole.device.scratch) / 2);
  fb::DeviceResources res{0, least, fb::DeviceResources::kDefaultPoolReleaseThreshold, 0, cap};
  ASSERT_TRUE(nevpt2::class_energies(res, b, g.in, nevpt2::kBatchFromBudget));
  const fbt::PoolPeak peak = fbt::measure_pool_peak(*res.device(), [&] {
    const auto e = nevpt2::class_energies(res, b, g.in, nevpt2::kBatchFromBudget);
    ASSERT_TRUE(e) << e.error().message;
  });
  EXPECT_GT(peak.used, least.device.scratch);
  EXPECT_LT(peak.used, whole.device.scratch);
  EXPECT_LE(res.device()->pool_usage().used_high, cap);
}

// ---------------------------------------------------------------------------
// The SC forward tangents (#1182): class_energies_tangent and
// sc_effective_hamiltonian_tangent
// ---------------------------------------------------------------------------

// Random tangents over @p b's shapes, on the device: what a direction reads,
// which the footprints do not depend on, only which fields are given.
struct Directions {
  std::vector<fb::Buffers::device<double>> held;
  std::vector<nevpt2::MoBlocks> blocks;
  std::vector<nevpt2::EnergyDirection> along;
};

// @p count random doubles from @p seed, on the device, kept in @p d.
const double *random_on_device(fb::DeviceResources &res, Directions &d, const std::size_t count,
                               const std::uint64_t seed) {
  d.held.push_back(on_device(res, random_vector(count, seed)));
  return d.held.back().data();
}

// Random B tangents shaped as @p b, kept in @p d (reserved by the caller).
const nevpt2::MoBlocks *random_blocks(fb::DeviceResources &res, Directions &d,
                                      const nevpt2::MoBlocks &b, const std::uint64_t seed) {
  d.blocks.push_back(
      nevpt2::MoBlocks{.naux = b.naux,
                       .ncore = b.ncore,
                       .ncas = b.ncas,
                       .nvirt = b.nvirt,
                       .aa = on_device(res, random_vector(b.aa.num_elements(), seed)),
                       .ca = on_device(res, random_vector(b.ca.num_elements(), seed + 1)),
                       .va = on_device(res, random_vector(b.va.num_elements(), seed + 2)),
                       .cv = on_device(res, random_vector(b.cv.num_elements(), seed + 3)),
                       .vc = on_device(res, random_vector(b.vc.num_elements(), seed + 4))});
  return &d.blocks.back();
}

// The integral and orbital-energy tangents of @p e over @p b's shapes.
void random_shared(fb::DeviceResources &res, Directions &d, nevpt2::EnergyInputs &e,
                   const nevpt2::MoBlocks &b, const std::uint64_t seed) {
  const auto n = static_cast<std::size_t>(b.ncas);
  const auto nc = static_cast<std::size_t>(b.ncore);
  const auto nv = static_cast<std::size_t>(b.nvirt);
  e.h1e = random_on_device(res, d, n * n, seed);
  e.h2e = random_on_device(res, d, n * n * n * n, seed + 1);
  e.e_core = random_on_device(res, d, nc, seed + 2);
  e.e_virt = random_on_device(res, d, nv, seed + 3);
  e.h1e_v_sr = random_on_device(res, d, nv * n, seed + 4);
  e.h1e_v_si = random_on_device(res, d, n * nc, seed + 5);
  e.h1e_v_sir = random_on_device(res, d, nv * nc, seed + 6);
}

// The RDM tangents of @p e, f3ca's and f3ac's when @p f3.
void random_rdms(fb::DeviceResources &res, Directions &d, nevpt2::EnergyInputs &e,
                 const std::int32_t ncas, const bool f3, const std::uint64_t seed) {
  const auto n = static_cast<std::size_t>(ncas);
  const std::size_t n6 = n * n * n * n * n * n;
  e.dm1 = random_on_device(res, d, n * n, seed);
  e.dm2 = random_on_device(res, d, n * n * n * n, seed + 1);
  e.dm3 = random_on_device(res, d, n6, seed + 2);
  if (f3) {
    e.f3ca = random_on_device(res, d, n6, seed + 3);
    e.f3ac = random_on_device(res, d, n6, seed + 4);
  }
}

// Two directions over @p b: every field and B, and the RDMs alone, so the
// footprint sees a direction that moves no slab beside one that does.
Directions two_directions(fb::DeviceResources &res, const nevpt2::MoBlocks &b) {
  Directions d;
  d.held.reserve(64);
  d.blocks.reserve(1);
  nevpt2::EnergyDirection all;
  random_shared(res, d, all.in, b, 100);
  random_rdms(res, d, all.in, b.ncas, true, 200);
  all.blocks = random_blocks(res, d, b, 300);
  nevpt2::EnergyDirection rdms_only;
  random_rdms(res, d, rdms_only.in, b.ncas, true, 400);
  d.along = {all, rdms_only};
  return d;
}

// The allocations live at a tangent's peak, at most: delta, B + s Ḃ's five
// blocks, plus and dd, a class's blocks and each direction's tangent of
// each, Eq. 40's scalar, the slab buffers and bars, the four tables and the
// slab tangents' buffers, and a builder's shifted inputs and its own.
constexpr std::size_t kTangentLive = 40;

class Nevpt2ClassEnergiesTangentFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

// class_energies_tangent along two directions at a batch of one and at the
// whole extent.
TEST_P(Nevpt2ClassEnergiesTangentFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  const auto [name, batch] = GetParam();
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const Directions d = two_directions(inputs, b);
  const fb::MemoryFootprint fp = nevpt2::class_energies_tangent_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, g.nelecas, batch, d.along);
  EXPECT_EQ(fp.device.persistent, 0U);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  expect_routine_footprint(res, fp, kTangentLive, [&] {
    const auto t = nevpt2::class_energies_tangent(res, b, g.in, batch, d.along);
    ASSERT_TRUE(t) << t.error().message;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2ClassEnergiesTangentFootprint,
                         ::testing::Combine(::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                                              std::string{"n2_ccpvdz_cas88_df"}),
                                            ::testing::Values(1, 0)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

// The one-function classes' η̇ (Srs at two active electrons, Sij at two
// holes): CAS(4,4)'s inputs with dm1 rescaled to the trace that makes the
// class one-function. Only the trace is read for that, so the values are
// as good as any for the footprint.
TEST(Nevpt2Footprint, ClassEnergiesTangentOneFunctionClasses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, "n2_ccpvdz_cas44_df");
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const Directions d = two_directions(inputs, b);
  const std::vector<double> dm1 =
      row_major(read_array(kDataDir / "n2_ccpvdz_cas44_df" / "dm1.npy"));
  for (const std::int32_t nelecas : {2, (2 * b.ncas) - 2}) {
    SCOPED_TRACE(std::format("nelecas {}", nelecas));
    std::vector<double> scaled = dm1;
    for (double &x : scaled) {
      x *= static_cast<double>(nelecas) / static_cast<double>(g.nelecas);
    }
    const fb::Buffers::device<double> dm1_d = on_device(inputs, scaled);
    nevpt2::EnergyInputs in = g.in;
    in.dm1 = dm1_d.data();
    const fb::MemoryFootprint fp = nevpt2::class_energies_tangent_footprint(
        b.naux, b.ncore, b.ncas, b.nvirt, nelecas, 0, d.along);
    fb::DeviceResources res{0, fb::test::kWorkspaces};
    expect_routine_footprint(res, fp, kTangentLive, [&] {
      const auto t = nevpt2::class_energies_tangent(res, b, in, 0, d.along);
      ASSERT_TRUE(t) << t.error().message;
    });
  }
}

class Nevpt2ScEffectiveHamiltonianTangentFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

// sc_effective_hamiltonian_tangent over two states at a batch of one and at
// the whole extent: the case's inputs on the diagonal, random transition
// RDMs off it (no f3ca or f3ac, which only the diagonal reads), and per
// direction its shared tangents once, the diagonal's RDM tangents with f3
// and the transitions' without.
TEST_P(Nevpt2ScEffectiveHamiltonianTangentFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kStates = 2;
  const auto [name, batch] = GetParam();
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  Directions d;
  d.held.reserve(128);
  d.blocks.reserve(1);
  std::vector<nevpt2::EnergyInputs> pairs(kStates * kStates, g.in);
  for (std::int32_t i = 0; i < kStates; ++i) {
    for (std::int32_t j = 0; j < kStates; ++j) {
      if (i != j) {
        nevpt2::EnergyInputs &p = pairs[static_cast<std::size_t>((i * kStates) + j)];
        random_rdms(inputs, d, p, b.ncas, false, 500 + (10 * ((i * kStates) + j)));
        p.f3ca = nullptr;
        p.f3ac = nullptr;
      }
    }
  }
  // Direction 0 moves every field and B, direction 1 the RDMs alone.
  nevpt2::EnergyDirection shared;
  random_shared(inputs, d, shared.in, b, 600);
  shared.blocks = random_blocks(inputs, d, b, 700);
  std::vector<nevpt2::EnergyDirection> along;
  for (std::int32_t u = 0; u < 2; ++u) {
    for (std::int32_t p = 0; p < kStates * kStates; ++p) {
      nevpt2::EnergyDirection e = u == 0 ? shared : nevpt2::EnergyDirection{};
      random_rdms(inputs, d, e.in, b.ncas, p % (kStates + 1) == 0, 800 + (10 * ((u * 4) + p)));
      along.push_back(e);
    }
  }
  const fb::MemoryFootprint fp = nevpt2::sc_effective_hamiltonian_tangent_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, g.nelecas, kStates, batch, along);
  EXPECT_EQ(fp.device.persistent, 0U);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  expect_routine_footprint(res, fp, kTangentLive, [&] {
    const auto t = nevpt2::sc_effective_hamiltonian_tangent(res, b, pairs, batch, along);
    ASSERT_TRUE(t) << t.error().message;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2ScEffectiveHamiltonianTangentFootprint,
                         ::testing::Combine(::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                                              std::string{"n2_ccpvdz_cas88_df"}),
                                            ::testing::Values(1, 0)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

// ---------------------------------------------------------------------------
// The PC forward tangents (#1183): pc_class_energies_tangent and
// pc_effective_hamiltonian_tangent, along the SC tangents' directions
// ---------------------------------------------------------------------------

// The allocations live at a PC tangent's peak, at most, at two states and
// two directions: the identities and B + s Ḃ's five blocks; per ket U, T,
// T₁, V, V_nᵀ K U and the two leaks, and per transition T, T₁ and its two
// leaks; every entry's four walk bars; per ket and direction Ṫ, ṫ1, K̇' and
// Ṡ', and per transition and direction Ṫ and ṫ1; V_kᵀ U and the Θ's; the
// slab buffers and bars, their tangents' buffers, plus and dd; a slab's
// eight rows; and a polarization's three matrices, shifted inputs and
// builder.
constexpr std::size_t kPcTangentLive = 120;

class Nevpt2PcClassEnergiesTangentFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

// pc_class_energies_tangent along two directions at a batch of one and at
// the whole extent, its modes dropped as the run drops them; and the
// footprint at the rank that holds the most, no less.
TEST_P(Nevpt2PcClassEnergiesTangentFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  const auto [name, batch] = GetParam();
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const Directions d = two_directions(inputs, b);
  const auto pc = nevpt2::pc_class_energies(inputs, b, g.in, batch);
  ASSERT_TRUE(pc) << pc.error().message;
  const std::vector<std::size_t> lwork = lworks(inputs, b.ncas * b.ncas * b.ncas);
  const fb::MemoryFootprint fp = nevpt2::pc_class_energies_tangent_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, batch, lwork, d.along, pc->dropped);
  const fb::MemoryFootprint most = nevpt2::pc_effective_hamiltonian_tangent_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, 1, batch, lwork, d.along);
  EXPECT_EQ(fp.device.persistent, 0U);
  EXPECT_GE(most.device.scratch, fp.device.scratch);
  EXPECT_GE(most.device.workspace, fp.device.workspace);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  expect_routine_footprint(res, fp, kPcTangentLive, [&] {
    const auto t = nevpt2::pc_class_energies_tangent(res, b, g.in, batch, d.along);
    ASSERT_TRUE(t) << t.error().message;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2PcClassEnergiesTangentFootprint,
                         ::testing::Combine(::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                                              std::string{"n2_ccpvdz_cas88_df"}),
                                            ::testing::Values(1, 0)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

// Two states over a case, as the PC-QD footprints walk them: the case's
// inputs on the diagonal, random transition RDMs off it (no f3ca or f3ac,
// which only the diagonal reads), every pair's metric cut at @p floor; and
// two directions, [u n² + p]: direction 0 moves every field and B, its
// shared tangents once, direction 1 the RDMs alone, the diagonal's RDM
// tangents with f3 and the transitions' without.
constexpr std::int32_t kQdStates = 2;

struct TwoStates {
  Directions d;
  std::vector<nevpt2::EnergyInputs> pairs;
  std::vector<nevpt2::EnergyDirection> along;
};

TwoStates two_states(fb::DeviceResources &res, const EnergyCase &g, const double floor = 0.0) {
  const nevpt2::MoBlocks &b = *g.blocks;
  TwoStates out;
  out.d.held.reserve(128);
  out.d.blocks.reserve(1);
  out.pairs.assign(kQdStates * kQdStates, g.in);
  for (std::int32_t i = 0; i < kQdStates; ++i) {
    for (std::int32_t j = 0; j < kQdStates; ++j) {
      nevpt2::EnergyInputs &p = out.pairs[static_cast<std::size_t>((i * kQdStates) + j)];
      p.pc_floor = floor;
      if (i != j) {
        random_rdms(res, out.d, p, b.ncas, false, 500 + (10 * ((i * kQdStates) + j)));
        p.f3ca = nullptr;
        p.f3ac = nullptr;
      }
    }
  }
  nevpt2::EnergyDirection shared;
  random_shared(res, out.d, shared.in, b, 600);
  shared.blocks = random_blocks(res, out.d, b, 700);
  for (std::int32_t u = 0; u < 2; ++u) {
    for (std::int32_t p = 0; p < kQdStates * kQdStates; ++p) {
      nevpt2::EnergyDirection e = u == 0 ? shared : nevpt2::EnergyDirection{};
      random_rdms(res, out.d, e.in, b.ncas, p % (kQdStates + 1) == 0, 800 + (10 * ((u * 4) + p)));
      out.along.push_back(e);
    }
  }
  return out;
}

// Each ket's dropped counts over @p pairs, as pc_effective_hamiltonian drops
// them (element [J n + J]'s).
std::vector<std::array<std::int64_t, 8>> ket_dropped(fb::DeviceResources &res,
                                                     const nevpt2::MoBlocks &b,
                                                     std::span<const nevpt2::EnergyInputs> pairs,
                                                     const std::int32_t batch) {
  const auto h = nevpt2::pc_effective_hamiltonian(res, b, pairs, batch);
  EXPECT_TRUE(h) << h.error().message;
  std::vector<std::array<std::int64_t, 8>> out;
  for (std::int32_t j = 0; j < kQdStates && h; ++j) {
    out.push_back((*h)[static_cast<std::size_t>((j * kQdStates) + j)].dropped);
  }
  return out;
}

class Nevpt2PcEffectiveHamiltonianTangentFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

// pc_effective_hamiltonian_tangent over two_states at a batch of one and at
// the whole extent, each ket's modes dropped as the energy drops them.
TEST_P(Nevpt2PcEffectiveHamiltonianTangentFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  const auto [name, batch] = GetParam();
  // Under slow_test.h: CAS(8,8), 1.7 s on the RTX 3080 at either batch.
  if (name == "n2_ccpvdz_cas88_df") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const TwoStates q = two_states(inputs, g);
  const std::vector<nevpt2::EnergyInputs> &pairs = q.pairs;
  const std::vector<nevpt2::EnergyDirection> &along = q.along;
  const std::vector<std::array<std::int64_t, 8>> dropped = ket_dropped(inputs, b, pairs, batch);
  ASSERT_EQ(dropped.size(), static_cast<std::size_t>(kQdStates));
  const std::vector<std::size_t> lwork = lworks(inputs, b.ncas * b.ncas * b.ncas);
  const fb::MemoryFootprint fp = nevpt2::pc_effective_hamiltonian_tangent_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, kQdStates, batch, lwork, along, dropped);
  const fb::MemoryFootprint most = nevpt2::pc_effective_hamiltonian_tangent_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, kQdStates, batch, lwork, along);
  EXPECT_EQ(fp.device.persistent, 0U);
  EXPECT_GE(most.device.scratch, fp.device.scratch);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  expect_routine_footprint(res, fp, kPcTangentLive, [&] {
    const auto t = nevpt2::pc_effective_hamiltonian_tangent(res, b, pairs, batch, along);
    ASSERT_TRUE(t) << t.error().message;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2PcEffectiveHamiltonianTangentFootprint,
                         ::testing::Combine(::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                                              std::string{"n2_ccpvdz_cas88_df"}),
                                            ::testing::Values(1, 0)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

// ---------------------------------------------------------------------------
// The SC Hessian-vector products (#1180): class_energies_hvp and
// sc_effective_hamiltonian_hvp
// ---------------------------------------------------------------------------

// The allocations live at a product's peak, at most: the state's identity,
// one rung's shifted inputs (twelve fields and B's five blocks) beside a
// tangent's peak (kTangentLive), or line_derivative's eleven scratch bars
// and the shifted inputs beside a fixed walk's (block bars, a class's
// blocks and slabs).
constexpr std::size_t kHvpLive = kTangentLive + 18;

class Nevpt2ClassEnergiesHvpFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t, bool>> {};

// class_energies_hvp along two directions (every field and B; the RDMs
// alone), with or without partners (the two swapped, so the RDMs' direction,
// which populates no empty perturber, partners the other: no refusal), at a
// batch of one and whole. N₂'s symmetry empties perturbers in every case,
// so the second tables run, as the footprint counts them.
TEST_P(Nevpt2ClassEnergiesHvpFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("n2_ccpvdz_cas44_df_batch0");
  const auto [name, batch, partnered] = GetParam();
  // CAS(4,4) is the family's representative; its partnered batch of one
  // (2.2 s on the 3080) and CAS(8,8) (7-14 s) are past slow_test.h's rule.
  if (name != "n2_ccpvdz_cas44_df" || (batch == 1 && partnered)) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const Directions d = two_directions(inputs, b);
  const std::vector<nevpt2::EnergyDirection> partners =
      partnered ? std::vector{d.along[1], d.along[0]} : std::vector<nevpt2::EnergyDirection>{};
  const std::array<Bars, 2> bars{every_bar(inputs, b), every_bar(inputs, b)};
  const std::array<nevpt2::EnergyBars, 2> each{bars[0].bars, bars[1].bars};
  const fb::MemoryFootprint fp = nevpt2::class_energies_hvp_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, g.nelecas, batch, d.along, partners);
  EXPECT_EQ(fp.device.persistent, 0U);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  expect_routine_footprint(res, fp, kHvpLive, [&] {
    const auto p = nevpt2::class_energies_hvp(res, b, g.in, batch, d.along, each, partners);
    ASSERT_TRUE(p) << p.error().message;
    std::size_t populated = 0;
    for (const auto &u : p->populated) {
      populated += static_cast<std::size_t>(std::ranges::fold_left(u, 0, std::plus{}));
    }
    EXPECT_GT(populated, 0U) << "no empty perturber populated";
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2ClassEnergiesHvpFootprint,
                         ::testing::Combine(::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                                              std::string{"n2_ccpvdz_cas88_df"}),
                                            ::testing::Values(1, 0), ::testing::Bool()),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param)) +
                                  (std::get<2>(info.param) ? "_partnered" : "");
                         });

class Nevpt2ScEffectiveHamiltonianHvpFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

// sc_effective_hamiltonian_hvp over two states, the tangent test's pairs and
// directions, at a batch of one and whole: every bar wanted, a transition
// that allocates nothing (its footprint empty), and H̄ of trace 1.2, so
// Sijrs's bars go through its scaled buffers. Both kets are the case's
// state, so every empty tuple is empty for the other bra too, and the pair
// second tables run, as the footprint counts them at the most.
TEST_P(Nevpt2ScEffectiveHamiltonianHvpFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("n2_ccpvdz_cas44_df_batch0");
  constexpr std::int32_t kStates = 2;
  const auto [name, batch] = GetParam();
  // CAS(4,4) whole is the family's representative (0.9 s on the 3080); its
  // batch of one (3.5 s) and CAS(8,8) (15-17 s) are past slow_test.h's rule.
  if (name != "n2_ccpvdz_cas44_df" || batch == 1) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  Directions d;
  d.blocks.reserve(1);
  std::vector<nevpt2::EnergyInputs> pairs(kStates * kStates, g.in);
  for (std::int32_t i = 0; i < kStates; ++i) {
    for (std::int32_t j = 0; j < kStates; ++j) {
      if (i != j) {
        nevpt2::EnergyInputs &p = pairs[static_cast<std::size_t>((i * kStates) + j)];
        random_rdms(inputs, d, p, b.ncas, false, 500 + (10 * ((i * kStates) + j)));
        p.f3ca = nullptr;
        p.f3ac = nullptr;
      }
    }
  }
  nevpt2::EnergyDirection shared;
  random_shared(inputs, d, shared.in, b, 600);
  shared.blocks = random_blocks(inputs, d, b, 700);
  std::vector<nevpt2::EnergyDirection> along;
  for (std::int32_t u = 0; u < 2; ++u) {
    for (std::int32_t p = 0; p < kStates * kStates; ++p) {
      nevpt2::EnergyDirection e = u == 0 ? shared : nevpt2::EnergyDirection{};
      random_rdms(inputs, d, e.in, b.ncas, p % (kStates + 1) == 0, 800 + (10 * ((u * 4) + p)));
      along.push_back(e);
    }
  }
  // Per direction, the first state's every bar and the second's own RDM
  // bars beside the first's shared ones.
  std::vector<Bars> held;
  std::vector<nevpt2::EnergyBars> bars;
  for (std::int32_t u = 0; u < 2; ++u) {
    held.push_back(every_bar(inputs, b));
    held.push_back(every_bar(inputs, b));
    nevpt2::EnergyBars first = held[held.size() - 2].bars;
    nevpt2::EnergyBars second = first;
    const nevpt2::EnergyBars &own = held.back().bars;
    second.dm1 = own.dm1;
    second.dm2 = own.dm2;
    second.dm3 = own.dm3;
    second.f3ca = own.f3ca;
    second.f3ac = own.f3ac;
    bars.push_back(first);
    bars.push_back(second);
  }
  const std::vector<double> hbar{0.7, 0.2, 0.2, 0.5};
  const fb::MemoryFootprint fp = nevpt2::sc_effective_hamiltonian_hvp_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, g.nelecas, kStates, batch, hbar, along, {},
      fb::MemoryFootprint{});
  EXPECT_EQ(fp.device.persistent, 0U);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  expect_routine_footprint(res, fp, kHvpLive, [&] {
    const auto p = nevpt2::sc_effective_hamiltonian_hvp(
        res, b, pairs, hbar, batch, along, bars,
        [](std::int64_t, std::int64_t, std::int64_t, const nevpt2::EnergyBars &) -> fb::Status {
          return {};
        });
    ASSERT_TRUE(p) << p.error().message;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2ScEffectiveHamiltonianHvpFootprint,
                         ::testing::Combine(::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                                              std::string{"n2_ccpvdz_cas88_df"}),
                                            ::testing::Values(1, 0)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

// ---------------------------------------------------------------------------
// The PC Hessian-vector products (#1184): pc_class_energies_hvp and
// pc_effective_hamiltonian_hvp; and pc_metric_lifts
// ---------------------------------------------------------------------------

// The PC Hessian's conditioning floor (nevpt2_energy_gpu.cpp's kPcHvpFloor):
// the cut raised past the modes the product refuses to differentiate twice.
constexpr double kPcHvpFloor = 1e-3;

// The allocations live at a PC Hessian-vector product's peak, at most, at
// two states and two directions: the tangent's (kPcTangentLive) or a fixed
// walk's, beside every class's record (per ket U, S's four modes, M, M_Δ
// and two tangents of each, U̇ and the span bar's derivative per direction),
// the bars' scratch (eleven a state, three a transition), the transitions'
// bars and every pair's shifted inputs.
constexpr std::size_t kPcHvpLive = 400;

class Nevpt2PcClassEnergiesHvpFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

// pc_class_energies_hvp along two directions, every bar wanted, at a batch
// of one and at the whole extent, at the Hessian's floor and its modes
// dropped as the run drops them; and the footprint at the rank that holds
// the most, no less.
TEST_P(Nevpt2PcClassEnergiesHvpFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("n2_ccpvdz_cas44_df_batch0");
  const auto [name, batch] = GetParam();
  // Under slow_test.h: CAS(8,8), past 1.5 s on the RTX 3080 at either batch.
  if (name == "n2_ccpvdz_cas88_df") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  nevpt2::EnergyInputs in = g.in;
  in.pc_floor = kPcHvpFloor;
  const Directions d = two_directions(inputs, b);
  std::vector<Bars> held;
  std::vector<nevpt2::EnergyBars> bars;
  for (std::size_t u = 0; u < d.along.size(); ++u) {
    held.push_back(every_bar(inputs, b));
    bars.push_back(held.back().bars);
  }
  const auto pc = nevpt2::pc_class_energies(inputs, b, in, batch);
  ASSERT_TRUE(pc) << pc.error().message;
  const std::vector<std::size_t> lwork = lworks(inputs, b.ncas * b.ncas * b.ncas);
  const fb::MemoryFootprint fp = nevpt2::pc_class_energies_hvp_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, batch, lwork, d.along, pc->dropped);
  const fb::MemoryFootprint most = nevpt2::pc_effective_hamiltonian_hvp_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, 1, batch, lwork, d.along);
  EXPECT_EQ(fp.device.persistent, 0U);
  EXPECT_GE(most.device.scratch, fp.device.scratch);
  EXPECT_GE(most.device.workspace, fp.device.workspace);
  std::cout << std::format("dropped {}\n", pc->dropped);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  expect_routine_footprint(res, fp, kPcHvpLive, [&] {
    const auto t = nevpt2::pc_class_energies_hvp(res, b, in, batch, d.along, bars);
    ASSERT_TRUE(t) << t.error().message;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2PcClassEnergiesHvpFootprint,
                         ::testing::Combine(::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                                              std::string{"n2_ccpvdz_cas88_df"}),
                                            ::testing::Values(1, 0)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

// Per direction and state, the bars pc_effective_hamiltonian_hvp adds onto,
// [u n + J], every field wanted: each state's own RDMs' bars, and the shared
// fields' one set per direction.
struct StateBars {
  std::vector<fb::Buffers::device<double>> held;
  std::vector<nevpt2::EnergyBars> bars;
};

StateBars state_bars(fb::DeviceResources &res, const nevpt2::MoBlocks &b, const std::size_t dirs) {
  StateBars out;
  out.held.reserve(dirs * (8 + (5 * kQdStates)));
  const auto zero = [&](const std::size_t count) {
    out.held.emplace_back(count, res.device());
    return out.held.back().data();
  };
  const auto n = static_cast<std::size_t>(b.ncas);
  const auto nc = static_cast<std::size_t>(b.ncore);
  const auto nv = static_cast<std::size_t>(b.nvirt);
  const std::size_t nmo = nc + n + nv;
  const std::size_t n6 = n * n * n * n * n * n;
  for (std::size_t u = 0; u < dirs; ++u) {
    const nevpt2::EnergyBars shared{.h1e = zero(n * n),
                                    .h2e = zero(n * n * n * n),
                                    .h1e_v_sr = zero(nv * n),
                                    .h1e_v_si = zero(n * nc),
                                    .h1e_v_sir = zero(nv * nc),
                                    .e_core = zero(nc),
                                    .e_virt = zero(nv),
                                    .lambda = zero(static_cast<std::size_t>(b.naux) * nmo * nmo)};
    for (std::int32_t j = 0; j < kQdStates; ++j) {
      nevpt2::EnergyBars e = shared;
      e.dm1 = zero(n * n);
      e.dm2 = zero(n * n * n * n);
      e.dm3 = zero(n6);
      e.f3ca = zero(n6);
      e.f3ac = zero(n6);
      out.bars.push_back(e);
    }
  }
  return out;
}

class Nevpt2PcEffectiveHamiltonianHvpFootprint
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

// pc_effective_hamiltonian_hvp over two_states at the Hessian's floor, at a
// batch of one and at the whole extent, each ket's modes dropped as the
// energy drops them, every bar wanted and each transition's handed to a
// callback that holds nothing; H̄'s trace is not one, as a coupling's
// weights.
TEST_P(Nevpt2PcEffectiveHamiltonianHvpFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("n2_ccpvdz_cas44_df_batch0");
  const auto [name, batch] = GetParam();
  // Under slow_test.h: CAS(8,8), past 1.5 s on the RTX 3080 at either batch.
  if (name == "n2_ccpvdz_cas88_df") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const TwoStates q = two_states(inputs, g, kPcHvpFloor);
  const std::size_t dirs = q.along.size() / (kQdStates * kQdStates);
  const StateBars sb = state_bars(inputs, b, dirs);
  const std::vector<double> hbar{0.6, 0.3, 0.2, 0.7};
  const std::vector<std::array<std::int64_t, 8>> dropped = ket_dropped(inputs, b, q.pairs, batch);
  ASSERT_EQ(dropped.size(), static_cast<std::size_t>(kQdStates));
  const std::vector<std::size_t> lwork = lworks(inputs, b.ncas * b.ncas * b.ncas);
  const fb::MemoryFootprint fp = nevpt2::pc_effective_hamiltonian_hvp_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, kQdStates, batch, lwork, q.along, dropped);
  const fb::MemoryFootprint most = nevpt2::pc_effective_hamiltonian_hvp_footprint(
      b.naux, b.ncore, b.ncas, b.nvirt, kQdStates, batch, lwork, q.along);
  EXPECT_EQ(fp.device.persistent, 0U);
  EXPECT_GE(most.device.scratch, fp.device.scratch);
  std::cout << std::format("dropped {}\n", dropped);
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  expect_routine_footprint(res, fp, kPcHvpLive, [&] {
    const auto t = nevpt2::pc_effective_hamiltonian_hvp(
        res, b, q.pairs, hbar, batch, q.along, sb.bars,
        [](std::int64_t, std::int64_t, std::int64_t, const nevpt2::EnergyBars &) -> fb::Status {
          return {};
        });
    ASSERT_TRUE(t) << t.error().message;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2PcEffectiveHamiltonianHvpFootprint,
                         ::testing::Combine(::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                                              std::string{"n2_ccpvdz_cas88_df"}),
                                            ::testing::Values(1, 0)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

class Nevpt2PcMetricLiftsFootprint : public ::testing::TestWithParam<std::string> {};

// Both pc_metric_lifts at the default cut, the ket's modes dropped as the
// energy drops them: one state along random first and second RDM
// derivatives, and two_states' ket 0 with random first derivatives per pair.
TEST_P(Nevpt2PcMetricLiftsFootprint, SoundAndTight) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("0");
  const std::string name = GetParam();
  fb::DeviceResources inputs{0, fb::test::kWorkspaces};
  const EnergyCase g = energy_case(inputs, name);
  ASSERT_TRUE(g.blocks.has_value());
  const nevpt2::MoBlocks &b = *g.blocks;
  const std::vector<std::size_t> lwork = lworks(inputs, b.ncas * b.ncas * b.ncas);
  TwoStates q = two_states(inputs, g);
  std::vector<nevpt2::EnergyInputs> first(kQdStates * kQdStates);
  for (std::size_t p = 0; p < first.size(); ++p) {
    random_rdms(inputs, q.d, first[p], b.ncas, false, 900 + (10 * p));
  }
  nevpt2::EnergyInputs second;
  random_rdms(inputs, q.d, second, b.ncas, false, 990);
  const std::vector<std::array<std::int64_t, 8>> dropped = ket_dropped(inputs, b, q.pairs, 0);
  ASSERT_EQ(dropped.size(), static_cast<std::size_t>(kQdStates));
  std::cout << std::format("dropped {}\n", dropped.front());
  {
    SCOPED_TRACE("one state");
    const fb::MemoryFootprint fp = nevpt2::pc_metric_lifts_footprint(
        b.ncore, b.ncas, b.nvirt, 1, lwork, std::span{dropped}.first(1));
    EXPECT_GE(nevpt2::pc_metric_lifts_footprint(b.ncore, b.ncas, b.nvirt, 1, lwork).device.scratch,
              fp.device.scratch);
    fb::DeviceResources res{0, fb::test::kWorkspaces};
    expect_routine_footprint(res, fp, kPcTangentLive, [&] {
      const auto l = nevpt2::pc_metric_lifts(res, b, g.in, first.front(), second);
      ASSERT_TRUE(l) << l.error().message;
    });
  }
  {
    SCOPED_TRACE("two states");
    const fb::MemoryFootprint fp = nevpt2::pc_metric_lifts_footprint(
        b.ncore, b.ncas, b.nvirt, kQdStates, lwork, std::span{dropped}.first(1));
    fb::DeviceResources res{0, fb::test::kWorkspaces};
    expect_routine_footprint(res, fp, kPcTangentLive, [&] {
      const auto l = nevpt2::pc_metric_lifts(res, b, q.pairs, first, second, 0,
                                             std::numeric_limits<double>::infinity());
      ASSERT_TRUE(l) << l.error().message;
    });
  }
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2PcMetricLiftsFootprint,
                         ::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                           std::string{"n2_ccpvdz_cas88_df"}));

} // namespace
