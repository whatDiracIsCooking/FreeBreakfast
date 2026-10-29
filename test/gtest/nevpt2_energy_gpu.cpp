// GPU cases for nevpt2.energy's class_energies and pc_class_energies: from
// PySCF's dm1/dm2/dm3/f3ca/f3ac and the slabs nevpt2.density_fitting builds
// over the golden B blocks (data/nevpt2/), every SC class norm and energy and
// the total against PySCF's DF-NEVPT2, and every PC class energy and the total
// against block2's, at 1e-7 Ha (the demo's tolerance), on every case that
// carries dm3 and f3, walked whole and in batches of three, the last one
// partial. Then the chain: nevpt2::rdms' dm3/f3ca/f3ac from the golden CI
// vector in place of PySCF's, on every case, CAS(10,10)'s included.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.npy;
import free_breakfast.ci;
import free_breakfast.nevpt2;

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

// The demo's tolerance against the golden class energies: the einsums'
// atomics reassociate every sum, and PySCF's are NumPy's. Against block2's PC
// energies the largest miss is Sr's, which block2 builds from the dense 4-RDM
// rather than the f3 digests: 6.8e-10 at cc-pVTZ in the demo.
constexpr double kEnergyTol = 1e-7;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2";

constexpr std::array<const char *, 8> kClassNames{"Sr",   "Si",  "Sijrs", "Sijr",
                                                  "Srsi", "Srs", "Sij",   "Sir"};

fb::NpyArray read_array(const std::filesystem::path &path) {
  auto a = fb::read_npy(path);
  EXPECT_TRUE(a) << path << ": " << (a ? "" : a.error().message);
  return a ? std::move(*a) : fb::NpyArray{};
}

// @p a, column-major as io.npy returns it, row-major (C order): the layout
// class_energies takes the RDMs in, PySCF's own.
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

fb::Buffers::device<double> upload(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

// One case's integrals on the device: the MO blocks the slabs are built from
// and every column-major input class_energies reads, as data/nevpt2/ stores
// them. The RDMs are the caller's.
struct Integrals {
  std::optional<nevpt2::MoBlocks> blocks;
  std::vector<fb::Buffers::device<double>> arrays;
  nevpt2::EnergyInputs in;
};

Integrals load_integrals(fb::DeviceResources &res, const std::filesystem::path &dir) {
  const fb::NpyArray aa = read_array(dir / "B_aa.npy");
  const fb::NpyArray ca = read_array(dir / "B_ca.npy");
  const fb::NpyArray va = read_array(dir / "B_va.npy");
  const fb::NpyArray cv = read_array(dir / "B_cv.npy");
  Integrals g;
  if (aa.shape.size() != 3 || cv.shape.size() != 3) {
    ADD_FAILURE() << dir << ": B blocks missing";
    return g;
  }
  const std::size_t naux = aa.shape[0];
  const std::size_t na = aa.shape[1];
  const std::size_t nc = cv.shape[1];
  const std::size_t nv = cv.shape[2];
  // B_vc, B_cv with its orbital pair swapped.
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
                                    .aa = upload(res, aa.data),
                                    .ca = upload(res, ca.data),
                                    .va = upload(res, va.data),
                                    .cv = upload(res, cv.data),
                                    .vc = upload(res, vc)});
  const auto put = [&](const char *name, const std::vector<std::size_t> &shape) {
    const fb::NpyArray a = read_array(dir / (std::string{name} + ".npy"));
    EXPECT_EQ(a.shape, shape) << name;
    g.arrays.push_back(upload(res, a.data));
    return static_cast<const double *>(g.arrays.back().data());
  };
  g.in.h1e = put("h1e", {na, na});
  g.in.h2e = put("h2e", {na, na, na, na});
  g.in.e_core = put("e_core", {nc});
  g.in.e_virt = put("e_virt", {nv});
  g.in.h1e_v_sr = put("h1e_v_Sr", {nv, na});
  g.in.h1e_v_si = put("h1e_v_Si", {na, nc});
  g.in.h1e_v_sir = put("h1e_v_Sir", {nv, nc});
  return g;
}

// class_energies against the golden class_norms, class_energies and
// e_nevpt2_total.
void expect_golden(const std::filesystem::path &dir, const nevpt2::ClassEnergies &got,
                   const std::string &what) {
  const fb::NpyArray norms = read_array(dir / "class_norms.npy");
  const fb::NpyArray energies = read_array(dir / "class_energies.npy");
  const fb::NpyArray total = read_array(dir / "e_nevpt2_total.npy");
  ASSERT_EQ(norms.data.size(), 8U);
  ASSERT_EQ(energies.data.size(), 8U);
  ASSERT_EQ(total.data.size(), 1U);
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(got.norms[k], norms.data[k], kEnergyTol) << what << " " << kClassNames[k];
    EXPECT_NEAR(got.energies[k], energies.data[k], kEnergyTol) << what << " " << kClassNames[k];
  }
  EXPECT_NEAR(got.total, total.data[0], kEnergyTol) << what;
}

// The PC modes each class's metric cuts as null, per case: the demo's
// measured table (docs/pc-nevpt2.md, "The singular-metric convention"), the
// same on both its cards and in every DF twin. Sr, Si and Sir drop; the
// n- and n²-dimensional classes drop nothing, and Sijrs has no metric.
std::array<std::int64_t, 8> pc_dropped(const std::string &name) {
  if (name.starts_with("n2_ccpvdz_cas44")) {
    return {44, 44, 0, 0, 0, 0, 0, 2};
  }
  if (name == "n2_ccpvdz_cas88_df") {
    return {16, 16, 0, 0, 0, 0, 0, 2};
  }
  return {10, 10, 0, 0, 0, 0, 0, 1}; // both CAS(10,10)s
}

// pc_class_energies against the golden's block2 pc_class_energies and
// e_pc_total, and the dropped counts against pc_dropped.
void expect_pc_golden(const std::filesystem::path &dir, const std::string &name,
                      const nevpt2::PcClassEnergies &got, const std::string &what) {
  const fb::NpyArray energies = read_array(dir / "pc_class_energies.npy");
  const fb::NpyArray total = read_array(dir / "e_pc_total.npy");
  ASSERT_EQ(energies.data.size(), 8U);
  ASSERT_EQ(total.data.size(), 1U);
  const std::array<std::int64_t, 8> dropped = pc_dropped(name);
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(got.energies[k], energies.data[k], kEnergyTol) << what << " PC " << kClassNames[k];
    EXPECT_EQ(got.dropped[k], dropped[k]) << what << " PC " << kClassNames[k];
  }
  EXPECT_NEAR(got.total, total.data[0], kEnergyTol) << what << " PC";
}

// ---------------------------------------------------------------------------
// From PySCF's RDMs
// ---------------------------------------------------------------------------

// The cases data/nevpt2/README.md says carry dm3, f3ca and f3ac.
// std::string, not const char *: ctest names each case by printing its
// parameter, and a pointer prints as its address.
const std::array<std::string, 3> kRdmCases{"n2_ccpvdz_cas44_df", "n2_ccpvdz_cas44_casscf_df",
                                           "n2_ccpvdz_cas88_df"};

class Nevpt2EnergyPyscfRdms
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {};

TEST_P(Nevpt2EnergyPyscfRdms, ClassEnergiesMatchGolden) {
  FBF_REQUIRE_GPU();
  const auto [name, batch] = GetParam();
  const std::filesystem::path dir = kDataDir / name;
  fb::DeviceResources res(0);
  Integrals g = load_integrals(res, dir);
  ASSERT_TRUE(g.blocks.has_value());
  const auto rdm = [&](const char *array) {
    g.arrays.push_back(upload(res, row_major(read_array(dir / (std::string{array} + ".npy")))));
    return static_cast<const double *>(g.arrays.back().data());
  };
  g.in.dm1 = rdm("dm1");
  g.in.dm2 = rdm("dm2");
  g.in.dm3 = rdm("dm3");
  g.in.f3ca = rdm("f3ca");
  g.in.f3ac = rdm("f3ac");

  const std::string what = name + " batch " + std::to_string(batch);
  const auto got = nevpt2::class_energies(res, *g.blocks, g.in, batch);
  ASSERT_TRUE(got) << got.error().message;
  expect_golden(dir, *got, what);
  const auto pc = nevpt2::pc_class_energies(res, *g.blocks, g.in, batch);
  ASSERT_TRUE(pc) << pc.error().message;
  expect_pc_golden(dir, name, *pc, what);
}

// Batch 0 is each class's whole extent in one slab. 3 divides none of the
// virtual counts (19, 17), so every virtual walk ends in a partial slab.
INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2EnergyPyscfRdms,
                         ::testing::Combine(::testing::ValuesIn(kRdmCases),
                                            ::testing::Values(0, 3)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

// ---------------------------------------------------------------------------
// The chain: nevpt2::rdms -> nevpt2::slab -> class_energies
// ---------------------------------------------------------------------------

// Every converted case, CAS(10,10)'s included: they carry no dm3 or f3, so
// this is where they are covered. n_alpha = n_beta = per_spin throughout.
struct ChainCase {
  const char *name;
  std::int32_t n;
  std::int32_t per_spin;
};
// What ctest names a case by: gtest would print the struct's bytes, a
// pointer among them.
void PrintTo(const ChainCase &c, std::ostream *os) {
  *os << c.name;
}

constexpr std::array kChainCases{
    ChainCase{"n2_ccpvdz_cas44_df", 4, 2}, ChainCase{"n2_ccpvdz_cas44_casscf_df", 4, 2},
    ChainCase{"n2_ccpvdz_cas88_df", 8, 4}, ChainCase{"n2_ccpvdz_cas1010_df", 10, 5},
    ChainCase{"n2_ccpvtz_cas1010_df", 10, 5}};

// rdms' budget. Its default is the card's free memory less 512 MiB, which at
// CAS(10,10) one case takes nearly all of, and a case running beside it on
// the same card under parallel ctest then gets none. 256 MiB is about 800
// determinants a tile there, so the CAS(10,10) builds are tiled.
constexpr std::size_t kRdmBudget = std::size_t{256} << 20;

class Nevpt2EnergyDeviceRdms
    : public ::testing::TestWithParam<std::tuple<ChainCase, std::int32_t>> {};

TEST_P(Nevpt2EnergyDeviceRdms, ClassEnergiesMatchGolden) {
  FBF_REQUIRE_GPU();
  const auto [c, batch] = GetParam();
  const std::filesystem::path dir = kDataDir / c.name;
  fb::DeviceResources res(0);
  Integrals g = load_integrals(res, dir);
  ASSERT_TRUE(g.blocks.has_value());
  ASSERT_EQ(g.blocks->ncas, c.n);
  const auto n = static_cast<std::size_t>(c.n);

  // rdms reads (tu|vw) at t + n (u + n (v + n w)); h2e[p, q, r, s] = (pr|qs)
  // at p + n (q + n (r + n s)), so (tu|vw) = h2e[t, v, u, w].
  const fb::NpyArray h2e = read_array(dir / "h2e.npy");
  std::vector<double> eri(n * n * n * n);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      for (std::size_t v = 0; v < n; ++v) {
        for (std::size_t w = 0; w < n; ++w) {
          eri[t + n * (u + n * (v + n * w))] = h2e.data[t + n * (v + n * (u + n * w))];
        }
      }
    }
  }
  auto strings = fb::ci::DeviceStringSet::create(res, c.n, c.per_spin);
  ASSERT_TRUE(strings) << strings.error().message;
  const fb::Buffers::device<double> ci = upload(res, row_major(read_array(dir / "ci.npy")));
  const fb::Buffers::device<double> eri_d = upload(res, eri);
  const std::size_t n6 = n * n * n * n * n * n;
  fb::Buffers::device<double> dm3(n6, res.device());
  fb::Buffers::device<double> f3ca(n6, res.device());
  fb::Buffers::device<double> f3ac(n6, res.device());
  const auto tiles = fb::nevpt2::rdms(res, *strings, *strings, ci.data(), eri_d.data(), dm3.data(),
                                      f3ca.data(), f3ac.data(), kRdmBudget);
  ASSERT_TRUE(tiles) << tiles.error().message;

  g.arrays.push_back(upload(res, row_major(read_array(dir / "dm1.npy"))));
  g.in.dm1 = g.arrays.back().data();
  g.arrays.push_back(upload(res, row_major(read_array(dir / "dm2.npy"))));
  g.in.dm2 = g.arrays.back().data();
  g.in.dm3 = dm3.data();
  g.in.f3ca = f3ca.data();
  g.in.f3ac = f3ac.data();

  const std::string what = std::string{c.name} + " batch " + std::to_string(batch);
  const auto got = nevpt2::class_energies(res, *g.blocks, g.in, batch);
  ASSERT_TRUE(got) << got.error().message;
  expect_golden(dir, *got, what);
  const auto pc = nevpt2::pc_class_energies(res, *g.blocks, g.in, batch);
  ASSERT_TRUE(pc) << pc.error().message;
  expect_pc_golden(dir, c.name, *pc, what);
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2EnergyDeviceRdms,
                         ::testing::Combine(::testing::ValuesIn(kChainCases),
                                            ::testing::Values(0, 3)),
                         [](const auto &info) {
                           return std::string{std::get<0>(info.param).name} + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

// ---------------------------------------------------------------------------
// PC refusals
// ---------------------------------------------------------------------------

// Zero RDMs make every metric zero: PC has no internal function to span, so
// it refuses with a Numerical error naming the first class, Sr, rather than
// return a number.
TEST(Nevpt2PcEnergy, ZeroMetricIsRefused) {
  FBF_REQUIRE_GPU();
  const std::filesystem::path dir = kDataDir / "n2_ccpvdz_cas44_df";
  fb::DeviceResources res(0);
  Integrals g = load_integrals(res, dir);
  ASSERT_TRUE(g.blocks.has_value());
  const auto n = static_cast<std::size_t>(g.blocks->ncas);
  g.arrays.push_back(upload(res, std::vector<double>(n * n * n * n * n * n, 0.0)));
  const double *const zeros = g.arrays.back().data();
  g.in.dm1 = zeros;
  g.in.dm2 = zeros;
  g.in.dm3 = zeros;
  g.in.f3ca = zeros;
  g.in.f3ac = zeros;

  const auto pc = nevpt2::pc_class_energies(res, *g.blocks, g.in, 0);
  ASSERT_FALSE(pc);
  EXPECT_EQ(pc.error().kind, fb::ErrorKind::Numerical);
  EXPECT_NE(pc.error().message.find("Sr"), std::string::npos) << pc.error().message;
}

} // namespace
