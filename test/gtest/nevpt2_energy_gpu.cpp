// GPU cases for nevpt2.energy's class_energies and pc_class_energies: from
// PySCF's dm1/dm2/dm3/f3ca/f3ac and the slabs nevpt2.density_fitting builds
// over the golden B blocks (data/nevpt2/), every SC class norm and energy and
// the total against PySCF's DF-NEVPT2, and every PC class energy and the total
// against block2's, at 1e-7 Ha (the demo's tolerance), on every case that
// carries dm3 and f3, walked whole and in batches of three, the last one
// partial. Then the chain: nevpt2::rdms' dm3/f3ca/f3ac from the golden CI
// vector in place of PySCF's, on every case, CAS(10,10)'s included. And PC's
// invariance to a rotation of the active orbitals, which the pipeline step
// relies on: it does not rotate to each root's natural orbitals. Last, SC's
// reverse pass (class_energies_adjoint): Park Eq. 40 and Euler's identity
// from its bars, its bars against a Richardson ladder of class_energies
// itself, and the same bars whatever the batch.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

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

// @p a, column-major as io.npy returns it, with each of @p axes (each n long)
// carried by the active rotation @p u, n x n column-major: the orbitals
// φ'_p = Σ_q φ_q u(q, p), so x'[.., p, ..] = Σ_q u(q, p) x[.., q, ..] on every
// axis, the integrals' and the RDMs' alike. Nothing when @p u is empty.
void rotate(fb::NpyArray &a, std::initializer_list<std::size_t> axes,
            const std::vector<double> &u) {
  if (u.empty()) {
    return;
  }
  std::size_t stride = 1;
  for (std::size_t d = 0; d < a.shape.size(); ++d) {
    if (std::ranges::find(axes, d) != axes.end()) {
      const std::size_t n = a.shape[d];
      EXPECT_EQ(u.size(), n * n) << "axis " << d;
      std::vector<double> out(a.data.size(), 0.0);
      for (std::size_t at = 0; at < a.data.size(); ++at) {
        const std::size_t q = (at / stride) % n;
        const std::size_t base = at - (q * stride);
        for (std::size_t p = 0; p < n; ++p) {
          out[base + (p * stride)] += u[q + (n * p)] * a.data[at];
        }
      }
      a.data = std::move(out);
    }
    stride *= a.shape[d];
  }
}

// A random n x n orthogonal matrix, column-major, from a fixed seed: Gram-Schmidt
// over normal deviates, twice for each column.
std::vector<double> random_rotation(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 engine(seed);
  std::normal_distribution<double> normal;
  std::vector<double> u(n * n);
  for (double &x : u) {
    x = normal(engine);
  }
  for (std::size_t j = 0; j < n; ++j) {
    double *const col = u.data() + (n * j);
    for (std::int32_t pass = 0; pass < 2; ++pass) {
      for (std::size_t k = 0; k < j; ++k) {
        const double *const prev = u.data() + (n * k);
        const double dot = std::inner_product(col, col + n, prev, 0.0);
        std::transform(col, col + n, prev, col, [dot](double c, double v) { return c - dot * v; });
      }
    }
    const double norm = std::sqrt(std::inner_product(col, col + n, col, 0.0));
    std::transform(col, col + n, col, [norm](double c) { return c / norm; });
  }
  return u;
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
// them, the active orbitals rotated by @p u when it is given. The RDMs are
// the caller's.
struct Integrals {
  std::optional<nevpt2::MoBlocks> blocks;
  std::vector<fb::Buffers::device<double>> arrays;
  nevpt2::EnergyInputs in;
};

Integrals load_integrals(fb::DeviceResources &res, const std::filesystem::path &dir,
                         const std::vector<double> &u = {}) {
  fb::NpyArray aa = read_array(dir / "B_aa.npy");
  fb::NpyArray ca = read_array(dir / "B_ca.npy");
  fb::NpyArray va = read_array(dir / "B_va.npy");
  rotate(aa, {1, 2}, u);
  rotate(ca, {2}, u);
  rotate(va, {2}, u);
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
  // Each array, its active axes rotated by u.
  const auto put = [&](const char *name, const std::vector<std::size_t> &shape,
                       std::initializer_list<std::size_t> active) {
    fb::NpyArray a = read_array(dir / (std::string{name} + ".npy"));
    EXPECT_EQ(a.shape, shape) << name;
    rotate(a, active, u);
    g.arrays.push_back(upload(res, a.data));
    return static_cast<const double *>(g.arrays.back().data());
  };
  g.in.h1e = put("h1e", {na, na}, {0, 1});
  g.in.h2e = put("h2e", {na, na, na, na}, {0, 1, 2, 3});
  g.in.e_core = put("e_core", {nc}, {});
  g.in.e_virt = put("e_virt", {nv}, {});
  g.in.h1e_v_sr = put("h1e_v_Sr", {nv, na}, {1});
  g.in.h1e_v_si = put("h1e_v_Si", {na, nc}, {0});
  g.in.h1e_v_sir = put("h1e_v_Sir", {nv, nc}, {});
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
  // Under slow_test.h: the cc-pVTZ CAS(10,10), and its cc-pVDZ batch 3, each
  // ~2.3 s of tiled rdms. cc-pVDZ batch 0 stands for the tiling, CAS(8,8)
  // batch 3 for the batching.
  if (std::string_view{c.name} == "n2_ccpvtz_cas1010_df" ||
      (std::string_view{c.name} == "n2_ccpvdz_cas1010_df" && batch != 0)) {
    FBF_SKIP_IF_FAST_TESTS();
  }
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
// Invariance to the active orbitals
// ---------------------------------------------------------------------------

// The pipeline step does not rotate the active orbitals to each root's
// natural orbitals, as PySCF's NEVPT.kernel does (cas_natorb), and leans on
// both variants being invariant to it. For PC the argument is that a
// rotation U carries each class's S and K by the same congruence, which
// leaves the pencil (K, S)'s eigenvalues, and the cut, relative to S's
// largest eigenvalue, unchanged. Measured here: every input with an active
// index rotated by a random U, then every SC and PC class energy, SC's norms
// and PC's dropped counts against the unrotated run's. SC is the control: its
// invariance is PySCF's (tools/emit/nevpt2_reference.py checks it to 1e-10),
// so an SC miss would be the rotation's fault, not PC's.
//
// Both cases cut modes (Sr, Si and Sir; pc_dropped), so the cut is tested too.
constexpr double kInvarianceTol = 1e-10;

class Nevpt2PcInvariance : public ::testing::TestWithParam<std::string> {};

TEST_P(Nevpt2PcInvariance, ActiveRotationLeavesEnergiesUnchanged) {
  FBF_REQUIRE_GPU();
  const std::string name = GetParam();
  const std::filesystem::path dir = kDataDir / name;
  fb::DeviceResources res(0);
  const auto run = [&](const std::vector<double> &u)
      -> std::optional<std::pair<nevpt2::ClassEnergies, nevpt2::PcClassEnergies>> {
    Integrals g = load_integrals(res, dir, u);
    if (!g.blocks.has_value()) {
      return std::nullopt;
    }
    const auto rdm = [&](const char *array) {
      fb::NpyArray a = read_array(dir / (std::string{array} + ".npy"));
      std::vector<std::size_t> axes(a.shape.size());
      std::iota(axes.begin(), axes.end(), std::size_t{0});
      for (const std::size_t axis : axes) {
        rotate(a, {axis}, u);
      }
      g.arrays.push_back(upload(res, row_major(a)));
      return static_cast<const double *>(g.arrays.back().data());
    };
    g.in.dm1 = rdm("dm1");
    g.in.dm2 = rdm("dm2");
    g.in.dm3 = rdm("dm3");
    g.in.f3ca = rdm("f3ca");
    g.in.f3ac = rdm("f3ac");
    auto sc = nevpt2::class_energies(res, *g.blocks, g.in, 0);
    EXPECT_TRUE(sc) << sc.error().message;
    auto pc = nevpt2::pc_class_energies(res, *g.blocks, g.in, 0);
    EXPECT_TRUE(pc) << pc.error().message;
    if (!sc || !pc) {
      return std::nullopt;
    }
    return std::pair{*sc, *pc};
  };

  const auto plain = run({});
  ASSERT_TRUE(plain.has_value());
  const std::size_t n = read_array(dir / "dm1.npy").shape[0];
  const auto rotated = run(random_rotation(n, 455));
  ASSERT_TRUE(rotated.has_value());
  const auto &[sc, pc] = *plain;
  const auto &[sc_u, pc_u] = *rotated;
  double worst = 0.0;
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(sc_u.norms[k], sc.norms[k], kInvarianceTol) << name << " SC " << kClassNames[k];
    EXPECT_NEAR(sc_u.energies[k], sc.energies[k], kInvarianceTol)
        << name << " SC " << kClassNames[k];
    EXPECT_NEAR(pc_u.energies[k], pc.energies[k], kInvarianceTol)
        << name << " PC " << kClassNames[k];
    EXPECT_EQ(pc_u.dropped[k], pc.dropped[k]) << name << " PC " << kClassNames[k];
    worst = std::max(worst, std::abs(pc_u.energies[k] - pc.energies[k]));
  }
  EXPECT_NEAR(pc_u.total, pc.total, kInvarianceTol) << name << " PC";
  EXPECT_EQ(pc.dropped, pc_dropped(name)) << name;
  std::cout << std::format("[ nevpt2   ] {}: PC under a random active rotation, worst class "
                           "{:.2e}, total {:+.2e}; SC total {:+.2e}\n",
                           name, worst, pc_u.total - pc.total, sc_u.total - sc.total);
}

// CAS(4,4) and CAS(8,8); each cuts modes in Sr, Si and Sir.
INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2PcInvariance,
                         ::testing::Values(std::string{"n2_ccpvdz_cas44_df"},
                                           std::string{"n2_ccpvdz_cas88_df"}),
                         [](const auto &info) { return info.param; });

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

// ---------------------------------------------------------------------------
// The SC reverse pass
// ---------------------------------------------------------------------------

// Everything class_energies reads, on the host: the B blocks and the
// integrals column-major as data/nevpt2/ stores them, the RDMs row-major.
// The same type holds a direction: the fields the reverse pass has bars on.
struct HostCase {
  std::size_t naux = 0, nc = 0, na = 0, nv = 0;
  std::vector<double> aa, ca, va, cv, h1e, h2e, e_core, e_virt, h1e_v_sr, h1e_v_si, h1e_v_sir;
  std::vector<double> dm1, dm2, dm3, f3ca, f3ac;
};

HostCase load_host_case(const std::filesystem::path &dir) {
  HostCase h;
  const fb::NpyArray aa = read_array(dir / "B_aa.npy");
  const fb::NpyArray cv = read_array(dir / "B_cv.npy");
  h.naux = aa.shape.at(0);
  h.na = aa.shape.at(1);
  h.nc = cv.shape.at(1);
  h.nv = cv.shape.at(2);
  h.aa = aa.data;
  h.cv = cv.data;
  h.ca = read_array(dir / "B_ca.npy").data;
  h.va = read_array(dir / "B_va.npy").data;
  h.h1e = read_array(dir / "h1e.npy").data;
  h.h2e = read_array(dir / "h2e.npy").data;
  h.e_core = read_array(dir / "e_core.npy").data;
  h.e_virt = read_array(dir / "e_virt.npy").data;
  h.h1e_v_sr = read_array(dir / "h1e_v_Sr.npy").data;
  h.h1e_v_si = read_array(dir / "h1e_v_Si.npy").data;
  h.h1e_v_sir = read_array(dir / "h1e_v_Sir.npy").data;
  h.dm1 = row_major(read_array(dir / "dm1.npy"));
  h.dm2 = row_major(read_array(dir / "dm2.npy"));
  h.dm3 = row_major(read_array(dir / "dm3.npy"));
  h.f3ca = row_major(read_array(dir / "f3ca.npy"));
  h.f3ac = row_major(read_array(dir / "f3ac.npy"));
  return h;
}

// @p x + @p t @p dx over the fields the reverse pass has bars on; the rest is
// @p x's.
HostCase displaced(const HostCase &x, const HostCase &dx, const double t) {
  HostCase out = x;
  for (auto [y, d] :
       {std::pair{&out.aa, &dx.aa}, std::pair{&out.ca, &dx.ca}, std::pair{&out.va, &dx.va},
        std::pair{&out.cv, &dx.cv}, std::pair{&out.e_core, &dx.e_core},
        std::pair{&out.e_virt, &dx.e_virt}, std::pair{&out.h1e_v_sr, &dx.h1e_v_sr},
        std::pair{&out.h1e_v_si, &dx.h1e_v_si}, std::pair{&out.h1e_v_sir, &dx.h1e_v_sir}}) {
    for (std::size_t k = 0; k < y->size(); ++k) {
      (*y)[k] += t * (*d)[k];
    }
  }
  return out;
}

// A seeded random direction over the fields of @p group ("B", "h1e_v", "e"
// or "all") among those the reverse pass has bars on, each entry a normal
// deviate scaled by its array's largest magnitude; zero elsewhere.
HostCase random_direction(const HostCase &x, const std::string &group, const std::uint64_t seed) {
  std::mt19937_64 engine(seed);
  std::normal_distribution<double> normal;
  HostCase d = x;
  for (auto [name, y, from] :
       {std::tuple{"B", &d.aa, &x.aa}, std::tuple{"B", &d.ca, &x.ca}, std::tuple{"B", &d.va, &x.va},
        std::tuple{"B", &d.cv, &x.cv}, std::tuple{"e", &d.e_core, &x.e_core},
        std::tuple{"e", &d.e_virt, &x.e_virt}, std::tuple{"h1e_v", &d.h1e_v_sr, &x.h1e_v_sr},
        std::tuple{"h1e_v", &d.h1e_v_si, &x.h1e_v_si},
        std::tuple{"h1e_v", &d.h1e_v_sir, &x.h1e_v_sir}}) {
    double scale = 0.0;
    for (const double v : *from) {
      scale = std::max(scale, std::abs(v));
    }
    const bool on = group == "all" || group == name;
    for (double &v : *y) {
      v = on ? scale * normal(engine) : 0.0;
    }
  }
  return d;
}

// @p h on the device: its MO blocks (B_vc from B_cv) and its EnergyInputs.
struct DeviceCase {
  std::optional<nevpt2::MoBlocks> blocks;
  std::vector<fb::Buffers::device<double>> arrays;
  nevpt2::EnergyInputs in;
};

DeviceCase upload_case(fb::DeviceResources &res, const HostCase &h) {
  std::vector<double> vc(h.naux * h.nv * h.nc);
  for (std::size_t i = 0; i < h.nc; ++i) {
    for (std::size_t r = 0; r < h.nv; ++r) {
      for (std::size_t q = 0; q < h.naux; ++q) {
        vc[q + h.naux * (r + h.nv * i)] = h.cv[q + h.naux * (i + h.nc * r)];
      }
    }
  }
  DeviceCase g;
  g.blocks.emplace(nevpt2::MoBlocks{.naux = static_cast<std::int32_t>(h.naux),
                                    .ncore = static_cast<std::int32_t>(h.nc),
                                    .ncas = static_cast<std::int32_t>(h.na),
                                    .nvirt = static_cast<std::int32_t>(h.nv),
                                    .aa = upload(res, h.aa),
                                    .ca = upload(res, h.ca),
                                    .va = upload(res, h.va),
                                    .cv = upload(res, h.cv),
                                    .vc = upload(res, vc)});
  const auto put = [&](const std::vector<double> &a) {
    g.arrays.push_back(upload(res, a));
    return static_cast<const double *>(g.arrays.back().data());
  };
  g.in = {.h1e = put(h.h1e),
          .h2e = put(h.h2e),
          .e_core = put(h.e_core),
          .e_virt = put(h.e_virt),
          .h1e_v_sr = put(h.h1e_v_sr),
          .h1e_v_si = put(h.h1e_v_si),
          .h1e_v_sir = put(h.h1e_v_sir),
          .dm1 = put(h.dm1),
          .dm2 = put(h.dm2),
          .dm3 = put(h.dm3),
          .f3ca = put(h.f3ca),
          .f3ac = put(h.f3ac)};
  return g;
}

std::vector<double> download_host(fb::DeviceResources &res, const fb::Buffers::device<double> &d) {
  std::vector<double> host(d.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The reverse pass's bars, on the host.
struct HostBars {
  std::vector<double> h1e_v_sr, h1e_v_si, h1e_v_sir, e_core, e_virt, lambda;
};

// class_energies_adjoint over @p h, every bar wanted, each zeroed first.
std::optional<std::pair<nevpt2::ClassEnergiesAdjoint, HostBars>>
run_adjoint(fb::DeviceResources &res, const HostCase &h, const std::int32_t batch) {
  DeviceCase g = upload_case(res, h);
  const std::size_t nmo = h.nc + h.na + h.nv;
  const auto zeros = [&](const std::size_t count) {
    return upload(res, std::vector<double>(count, 0.0));
  };
  fb::Buffers::device<double> sr = zeros(h.nv * h.na);
  fb::Buffers::device<double> si = zeros(h.na * h.nc);
  fb::Buffers::device<double> sir = zeros(h.nv * h.nc);
  fb::Buffers::device<double> ec = zeros(h.nc);
  fb::Buffers::device<double> ev = zeros(h.nv);
  fb::Buffers::device<double> lambda = zeros(h.naux * nmo * nmo);
  const nevpt2::EnergyBars bars{.h1e_v_sr = sr.data(),
                                .h1e_v_si = si.data(),
                                .h1e_v_sir = sir.data(),
                                .e_core = ec.data(),
                                .e_virt = ev.data(),
                                .lambda = lambda.data()};
  auto got = nevpt2::class_energies_adjoint(res, *g.blocks, g.in, batch, bars);
  EXPECT_TRUE(got) << got.error().message;
  if (!got) {
    return std::nullopt;
  }
  return std::pair{*got, HostBars{.h1e_v_sr = download_host(res, sr),
                                  .h1e_v_si = download_host(res, si),
                                  .h1e_v_sir = download_host(res, sir),
                                  .e_core = download_host(res, ec),
                                  .e_virt = download_host(res, ev),
                                  .lambda = download_host(res, lambda)}};
}

double dot(const std::vector<double> &a, const std::vector<double> &b) {
  EXPECT_EQ(a.size(), b.size());
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// <Λ, dB> over the corners the blocks occupy in Λ, (Q, p, q) at
// Q + naux (p + nmo q): aa at (c + t, c + u), ca at (i, c + t), va at
// (c + a + r, c + t), cv at (i, c + a + r). B_vc is read from cv's corner.
double lambda_dot(const std::vector<double> &lambda, const HostCase &h, const HostCase &db) {
  const std::size_t nmo = h.nc + h.na + h.nv;
  const auto corner = [&](const std::vector<double> &b, const std::size_t np, const std::size_t nq,
                          const std::size_t p0, const std::size_t q0) {
    double sum = 0.0;
    for (std::size_t q = 0; q < nq; ++q) {
      for (std::size_t p = 0; p < np; ++p) {
        for (std::size_t x = 0; x < h.naux; ++x) {
          sum += lambda[x + h.naux * ((p0 + p) + nmo * (q0 + q))] * b[x + h.naux * (p + np * q)];
        }
      }
    }
    return sum;
  };
  return corner(db.aa, h.na, h.na, h.nc, h.nc) + corner(db.ca, h.nc, h.na, 0, h.nc) +
         corner(db.va, h.nv, h.na, h.nc + h.na, h.nc) + corner(db.cv, h.nc, h.nv, 0, h.nc + h.na);
}

// Σ <bar, dx> over every field the reverse pass has a bar on.
double bars_dot(const HostBars &bars, const HostCase &h, const HostCase &dx) {
  return lambda_dot(bars.lambda, h, dx) + dot(bars.h1e_v_sr, dx.h1e_v_sr) +
         dot(bars.h1e_v_si, dx.h1e_v_si) + dot(bars.h1e_v_sir, dx.h1e_v_sir) +
         dot(bars.e_core, dx.e_core) + dot(bars.e_virt, dx.e_virt);
}

// CAS(8,8): at CAS(4,4) Si vanishes by symmetry (its slab (σπ|ππ) and h1e_v_Si
// are zero), and so would its bars.
const std::string kAdjointCase = "n2_ccpvdz_cas88_df";

// Park Eq. 40, P N + Q H = E per class, assembled from the bars: an exact
// identity, so a mis-transcribed index in any term fails it. Euler's
// identity checks the slab and h1e_v bars the same way, step-free: at fixed
// orbital energies E is homogeneous of degree 4 in (B, h1e_v²), so
// <Λ, B> + 2 Σ <h1e_v-bar, h1e_v> = 4 E. And the energies are
// class_energies' golden ones.
TEST(Nevpt2EnergyAdjoint, Eq40AndEulerFromTheBars) {
  FBF_REQUIRE_GPU();
  const std::filesystem::path dir = kDataDir / kAdjointCase;
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(dir);
  const auto got = run_adjoint(res, h, 0);
  ASSERT_TRUE(got.has_value());
  const auto &[adj, bars] = *got;
  expect_golden(dir, adj.energies, "class_energies_adjoint");
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(adj.eq40[k], adj.energies.energies[k],
                1e-12 * std::max(1.0, std::abs(adj.energies.energies[k])))
        << kClassNames[k];
  }
  const double euler = lambda_dot(bars.lambda, h, h) + 2.0 * dot(bars.h1e_v_sr, h.h1e_v_sr) +
                       2.0 * dot(bars.h1e_v_si, h.h1e_v_si) +
                       2.0 * dot(bars.h1e_v_sir, h.h1e_v_sir);
  EXPECT_NEAR(euler, 4.0 * adj.energies.total, 1e-12 * std::abs(adj.energies.total));
  std::cout << std::format(
      "[ nevpt2   ] reverse: worst Eq. 40 miss {:.2e}, Euler {:+.2e}\n",
      [&] {
        double worst = 0.0;
        for (std::size_t k = 0; k < 8; ++k) {
          worst = std::max(worst, std::abs(adj.eq40[k] - adj.energies.energies[k]));
        }
        return worst;
      }(),
      euler - (4.0 * adj.energies.total));
}

// The bars against class_energies itself, differenced along a seeded random
// direction over (B, h1e_v_*, e_core, e_virt), all of them and each group
// alone, with h1e, h2e and the RDMs fixed. Central differences D(t) at
// t = t0 / 2^k; their successive changes must fall by a factor in (3.2, 4.8)
// over at least three consecutive halvings (the O(t²) regime), and the
// Richardson extrapolant (4 D(t/2) - D(t)) / 3 at that run's finest pair must
// equal Σ <bar, dx>.
class Nevpt2EnergyAdjointLadder : public ::testing::TestWithParam<std::string> {};

TEST_P(Nevpt2EnergyAdjointLadder, BarsMatchRichardsonLadder) {
  FBF_REQUIRE_GPU();
  const std::string group = GetParam();
  const std::filesystem::path dir = kDataDir / kAdjointCase;
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(dir);
  const HostCase dx = random_direction(h, group, 560);
  const auto got = run_adjoint(res, h, 0);
  ASSERT_TRUE(got.has_value());
  const double predicted = bars_dot(got->second, h, dx);

  const auto energy = [&](const double t) {
    DeviceCase g = upload_case(res, displaced(h, dx, t));
    const auto e = nevpt2::class_energies(res, *g.blocks, g.in, 0);
    EXPECT_TRUE(e) << e.error().message;
    return e ? e->total : std::nan("");
  };
  constexpr double kT0 = 1e-2;
  constexpr std::int32_t kRungs = 10;
  std::vector<double> d;
  for (std::int32_t k = 0; k < kRungs; ++k) {
    const double t = std::ldexp(kT0, -k);
    d.push_back((energy(t) - energy(-t)) / (2.0 * t));
  }
  // ratio[k] = (D_k - D_{k+1}) / (D_{k+1} - D_{k+2}); the longest run of
  // ratios in (3.2, 4.8).
  std::int32_t best_end = -1;
  std::int32_t best_len = 0;
  std::int32_t len = 0;
  std::string ladder;
  for (std::int32_t k = 0; k + 2 < kRungs; ++k) {
    const double ratio = (d[k] - d[k + 1]) / (d[k + 1] - d[k + 2]);
    ladder += std::format(" {:.3f}", ratio);
    len = (ratio > 3.2 && ratio < 4.8) ? len + 1 : 0;
    if (len > best_len) {
      best_len = len;
      best_end = k;
    }
  }
  ASSERT_GE(best_len, 3) << "ratios:" << ladder;
  // The run's finest pair is (D_{end+1}, D_{end+2}).
  const auto fine = static_cast<std::size_t>(best_end + 2);
  const double richardson = ((4.0 * d[fine]) - d[fine - 1]) / 3.0;
  EXPECT_NEAR(richardson, predicted, 1e-7 * std::abs(predicted)) << "ratios:" << ladder;
  std::cout << std::format("[ nevpt2   ] reverse ladder {}: Σ<bar, dx> {:.12e}, Richardson "
                           "{:.12e}, relative {:.2e}, ratios{}\n",
                           group, predicted, richardson,
                           std::abs(richardson - predicted) / std::abs(predicted), ladder);
}

INSTANTIATE_TEST_SUITE_P(Groups, Nevpt2EnergyAdjointLadder,
                         ::testing::Values(std::string{"all"}, std::string{"B"},
                                           std::string{"h1e_v"}, std::string{"e"}),
                         [](const auto &info) { return info.param; });

// The walk's batch only splits it: walked in threes (every virtual walk ends
// in a partial slab), every bar is the whole walk's to round-off.
TEST(Nevpt2EnergyAdjoint, BatchedBarsMatchWhole) {
  FBF_REQUIRE_GPU();
  const std::filesystem::path dir = kDataDir / kAdjointCase;
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(dir);
  const auto whole = run_adjoint(res, h, 0);
  const auto batched = run_adjoint(res, h, 3);
  ASSERT_TRUE(whole.has_value());
  ASSERT_TRUE(batched.has_value());
  const HostBars &a = whole->second;
  const HostBars &b = batched->second;
  for (const auto &[name, x, y] :
       {std::tuple{"lambda", &a.lambda, &b.lambda},
        std::tuple{"h1e_v_sr", &a.h1e_v_sr, &b.h1e_v_sr},
        std::tuple{"h1e_v_si", &a.h1e_v_si, &b.h1e_v_si},
        std::tuple{"h1e_v_sir", &a.h1e_v_sir, &b.h1e_v_sir},
        std::tuple{"e_core", &a.e_core, &b.e_core}, std::tuple{"e_virt", &a.e_virt, &b.e_virt}}) {
    ASSERT_EQ(x->size(), y->size()) << name;
    double scale = 0.0;
    double worst = 0.0;
    for (std::size_t k = 0; k < x->size(); ++k) {
      scale = std::max(scale, std::abs((*x)[k]));
      worst = std::max(worst, std::abs((*x)[k] - (*y)[k]));
    }
    EXPECT_GT(scale, 0.0) << name;
    EXPECT_LE(worst, 1e-12 * scale) << name;
  }
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(batched->first.energies.energies[k], whole->first.energies.energies[k], 1e-12)
        << kClassNames[k];
    EXPECT_NEAR(batched->first.eq40[k], whole->first.eq40[k], 1e-12) << kClassNames[k];
  }
}

} // namespace
