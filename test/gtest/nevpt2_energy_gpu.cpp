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
// from its bars, its bars on every input against a Richardson ladder of
// class_energies itself, and the same bars whatever the batch. And PC's
// (pc_class_energies_adjoint): the same, homogeneity in (S, S₁, K) in place
// of Eq. 40 and the ladder kept inside the energy's domain, its invariance to
// inactive rotations at degenerate orbital energies, and its refusals.

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

// @p x + @p t @p dx over every field.
HostCase displaced(const HostCase &x, const HostCase &dx, const double t) {
  HostCase out = x;
  for (auto [y, d] :
       {std::pair{&out.aa, &dx.aa}, std::pair{&out.ca, &dx.ca}, std::pair{&out.va, &dx.va},
        std::pair{&out.cv, &dx.cv}, std::pair{&out.e_core, &dx.e_core},
        std::pair{&out.e_virt, &dx.e_virt}, std::pair{&out.h1e_v_sr, &dx.h1e_v_sr},
        std::pair{&out.h1e_v_si, &dx.h1e_v_si}, std::pair{&out.h1e_v_sir, &dx.h1e_v_sir},
        std::pair{&out.h1e, &dx.h1e}, std::pair{&out.h2e, &dx.h2e}, std::pair{&out.dm1, &dx.dm1},
        std::pair{&out.dm2, &dx.dm2}, std::pair{&out.dm3, &dx.dm3}, std::pair{&out.f3ca, &dx.f3ca},
        std::pair{&out.f3ac, &dx.f3ac}}) {
    for (std::size_t k = 0; k < y->size(); ++k) {
      (*y)[k] += t * (*d)[k];
    }
  }
  return out;
}

// @p a, an n^rank row-major tensor, averaged with its index-reversed self,
// a[p, q, .., u] = a[u, .., q, p]: the one permutation symmetry PySCF's
// dm1, dm2 and dm3 (make_dm123's, not reordered) carry, which is hermiticity.
void reversal_symmetrize(std::vector<double> &a, const std::size_t n) {
  std::vector<double> out(a.size());
  for (std::size_t at = 0; at < a.size(); ++at) {
    std::size_t rest = at;
    std::size_t reversed = 0;
    for (std::size_t m = a.size(); m > 1; m /= n) {
      reversed = (reversed * n) + (rest % n);
      rest /= n;
    }
    out[at] = 0.5 * (a[at] + a[reversed]);
  }
  a = std::move(out);
}

// The RDMs' share of a direction, against the other fields' 1: along raw
// RDM directions at full scale, Sr's energy is singular a few ladder rungs
// from the base point. Measured at CAS(8,8), seed 560, the five RDMs
// together: Sr's ratios run 5.69 0.03 -0.75 -30.6 5.50 4.29 4.07 4.02 while
// every other class's settle to 4.00 from the third, a t⁴ coefficient 7e5
// times the t² one, so a singularity at |t| ≈ 1e-3 against the ladder's
// t0 = 1e-2. Its Richardson extrapolant then misses Σ <bar, dx> by 3.9e-7;
// the same direction a tenth as long agrees to 7e-10. The direction is
// shortened, not the tolerance.
constexpr double kRdmScale = 0.1;

// A seeded random direction over the fields of @p group ("B", "h1e_v", "e",
// "h" for h1e and h2e, "rdm" for dm1, dm2, dm3, f3ca and f3ac, or "all"),
// each entry a normal deviate scaled by its array's largest magnitude (the
// RDMs' by kRdmScale of it); zero elsewhere. The RDMs move as raw tensors,
// not along a CI displacement, so no spin or norm constraint applies; dm1,
// dm2 and dm3 are reversal-symmetrized, the one permutation symmetry the
// RDMs the builders read carry. f3ca and f3ac carry none, and h1e and h2e
// are moved as raw arrays.
HostCase random_direction(const HostCase &x, const std::string &group, const std::uint64_t seed) {
  std::mt19937_64 engine(seed);
  std::normal_distribution<double> normal;
  HostCase d = x;
  for (auto [name, y, from] :
       {std::tuple{"B", &d.aa, &x.aa}, std::tuple{"B", &d.ca, &x.ca}, std::tuple{"B", &d.va, &x.va},
        std::tuple{"B", &d.cv, &x.cv}, std::tuple{"e", &d.e_core, &x.e_core},
        std::tuple{"e", &d.e_virt, &x.e_virt}, std::tuple{"h1e_v", &d.h1e_v_sr, &x.h1e_v_sr},
        std::tuple{"h1e_v", &d.h1e_v_si, &x.h1e_v_si},
        std::tuple{"h1e_v", &d.h1e_v_sir, &x.h1e_v_sir}, std::tuple{"h", &d.h1e, &x.h1e},
        std::tuple{"h", &d.h2e, &x.h2e}, std::tuple{"rdm", &d.dm1, &x.dm1},
        std::tuple{"rdm", &d.dm2, &x.dm2}, std::tuple{"rdm", &d.dm3, &x.dm3},
        std::tuple{"rdm", &d.f3ca, &x.f3ca}, std::tuple{"rdm", &d.f3ac, &x.f3ac}}) {
    double scale = 0.0;
    for (const double v : *from) {
      scale = std::max(scale, std::abs(v));
    }
    if (std::string_view{name} == "rdm") {
      scale *= kRdmScale;
    }
    const bool on = group == "all" || group == name;
    for (double &v : *y) {
      v = on ? scale * normal(engine) : 0.0;
    }
  }
  for (std::vector<double> *rdm : {&d.dm1, &d.dm2, &d.dm3}) {
    reversal_symmetrize(*rdm, x.na);
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
  std::vector<double> h1e, h2e, dm1, dm2, dm3, f3ca, f3ac;
};

// Every bar a reverse pass takes, zeroed on the device, for a case shaped as
// @p h, and the EnergyBars over them.
struct DeviceBars {
  std::vector<fb::Buffers::device<double>> arrays;
  nevpt2::EnergyBars bars;
};

DeviceBars zero_bars(fb::DeviceResources &res, const HostCase &h) {
  const std::size_t nmo = h.nc + h.na + h.nv;
  DeviceBars b;
  const auto zeros = [&](const std::size_t count) {
    b.arrays.push_back(upload(res, std::vector<double>(count, 0.0)));
    return b.arrays.back().data();
  };
  b.arrays.reserve(13);
  b.bars = {.h1e = zeros(h.h1e.size()),
            .h2e = zeros(h.h2e.size()),
            .dm1 = zeros(h.dm1.size()),
            .dm2 = zeros(h.dm2.size()),
            .dm3 = zeros(h.dm3.size()),
            .f3ca = zeros(h.f3ca.size()),
            .f3ac = zeros(h.f3ac.size()),
            .h1e_v_sr = zeros(h.nv * h.na),
            .h1e_v_si = zeros(h.na * h.nc),
            .h1e_v_sir = zeros(h.nv * h.nc),
            .e_core = zeros(h.nc),
            .e_virt = zeros(h.nv),
            .lambda = zeros(h.naux * nmo * nmo)};
  return b;
}

// @p b's bars, on the host.
HostBars download_bars(fb::DeviceResources &res, const DeviceBars &b) {
  const auto at = [&](const std::size_t k) { return download_host(res, b.arrays[k]); };
  return HostBars{.h1e_v_sr = at(7),
                  .h1e_v_si = at(8),
                  .h1e_v_sir = at(9),
                  .e_core = at(10),
                  .e_virt = at(11),
                  .lambda = at(12),
                  .h1e = at(0),
                  .h2e = at(1),
                  .dm1 = at(2),
                  .dm2 = at(3),
                  .dm3 = at(4),
                  .f3ca = at(5),
                  .f3ac = at(6)};
}

// class_energies_adjoint over @p h, every bar wanted, each zeroed first.
std::optional<std::pair<nevpt2::ClassEnergiesAdjoint, HostBars>>
run_adjoint(fb::DeviceResources &res, const HostCase &h, const std::int32_t batch) {
  DeviceCase g = upload_case(res, h);
  const DeviceBars b = zero_bars(res, h);
  auto got = nevpt2::class_energies_adjoint(res, *g.blocks, g.in, batch, b.bars);
  EXPECT_TRUE(got) << got.error().message;
  if (!got) {
    return std::nullopt;
  }
  return std::pair{*got, download_bars(res, b)};
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

// Σ <bar, dx> over every field: B through Λ, and every EnergyInputs array.
double bars_dot(const HostBars &bars, const HostCase &h, const HostCase &dx) {
  return lambda_dot(bars.lambda, h, dx) + dot(bars.h1e_v_sr, dx.h1e_v_sr) +
         dot(bars.h1e_v_si, dx.h1e_v_si) + dot(bars.h1e_v_sir, dx.h1e_v_sir) +
         dot(bars.e_core, dx.e_core) + dot(bars.e_virt, dx.e_virt) + dot(bars.h1e, dx.h1e) +
         dot(bars.h2e, dx.h2e) + dot(bars.dm1, dx.dm1) + dot(bars.dm2, dx.dm2) +
         dot(bars.dm3, dx.dm3) + dot(bars.f3ca, dx.f3ca) + dot(bars.f3ac, dx.f3ac);
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

// A step ladder: central differences D(t) of @p energy at t = t0 / 2^k,
// k < 10; the longest run of consecutive ratios (D_k - D_{k+1}) /
// (D_{k+1} - D_{k+2}) in (3.2, 4.8), the O(t²) regime; and the Richardson
// extrapolant (4 D(t/2) - D(t)) / 3 at that run's finest pair.
struct Ladder {
  std::vector<double> d;
  double richardson = std::nan("");
  std::int32_t run = 0;
  std::string ratios;
};

Ladder richardson_ladder(const std::function<double(double)> &energy, const double t0) {
  constexpr std::int32_t kRungs = 10;
  Ladder l;
  std::vector<double> &d = l.d;
  for (std::int32_t k = 0; k < kRungs; ++k) {
    const double t = std::ldexp(t0, -k);
    d.push_back((energy(t) - energy(-t)) / (2.0 * t));
  }
  std::int32_t best_end = -1;
  std::int32_t len = 0;
  for (std::int32_t k = 0; k + 2 < kRungs; ++k) {
    const double ratio = (d[k] - d[k + 1]) / (d[k + 1] - d[k + 2]);
    l.ratios += std::format(" {:.3f}", ratio);
    len = (ratio > 3.2 && ratio < 4.8) ? len + 1 : 0;
    if (len > l.run) {
      l.run = len;
      best_end = k;
    }
  }
  if (l.run > 0) {
    // The run's finest pair is (D_{end+1}, D_{end+2}).
    const auto fine = static_cast<std::size_t>(best_end + 2);
    l.richardson = ((4.0 * d[fine]) - d[fine - 1]) / 3.0;
  }
  return l;
}

// The bars against class_energies itself, differenced along a seeded random
// direction over every input (B, h1e_v_*, the orbital energies, h1e and h2e,
// the RDMs), all of them and each group alone. Central differences D(t) at
// t = t0 / 2^k; their successive changes must fall by a factor in (3.2, 4.8)
// over at least three consecutive halvings (the O(t²) regime), and the
// Richardson extrapolant (4 D(t/2) - D(t)) / 3 at that run's finest pair must
// equal Σ <bar, dx>.
class Nevpt2EnergyAdjointLadder : public ::testing::TestWithParam<std::string> {};

TEST_P(Nevpt2EnergyAdjointLadder, BarsMatchRichardsonLadder) {
  FBF_REQUIRE_GPU();
  const std::string group = GetParam();
  // Under slow_test.h: "all", every input dense at once, is ~115 s on the
  // gfx1200. The cost follows the data: each class_energies is ~0.2 s with
  // only the orbital energies moved, ~0.7 s with B, ~5.7 s with everything
  // (presumably einsum's atomic per term, cheap while N₂'s symmetry-sparse
  // arrays make most terms exact zeros). Each group alone stands for it in
  // the fast tier.
  if (group == "all") {
    FBF_SKIP_IF_FAST_TESTS();
  }
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
  const Ladder l = richardson_ladder(energy, 1e-2);
  ASSERT_GE(l.run, 3) << "ratios:" << l.ratios;
  EXPECT_NEAR(l.richardson, predicted, 1e-7 * std::abs(predicted)) << "ratios:" << l.ratios;
  std::cout << std::format("[ nevpt2   ] reverse ladder {}: Σ<bar, dx> {:.12e}, Richardson "
                           "{:.12e}, relative {:.2e}, ratios{}\n",
                           group, predicted, l.richardson,
                           std::abs(l.richardson - predicted) / std::abs(predicted), l.ratios);
}

INSTANTIATE_TEST_SUITE_P(Groups, Nevpt2EnergyAdjointLadder,
                         ::testing::Values(std::string{"all"}, std::string{"B"},
                                           std::string{"h1e_v"}, std::string{"e"}, std::string{"h"},
                                           std::string{"rdm"}),
                         [](const auto &info) { return info.param; });

// Every bar of @p b against @p a's to 1e-12 of its largest element, and
// every one formed (not all zero): the walk's (Λ, h1e_v's, the orbital
// energies'), and with @p sweep the sweep's (h1e, h2e, the RDMs').
void expect_same_bars(const HostBars &a, const HostBars &b, const bool sweep = true) {
  for (const auto &[name, x, y, swept] :
       {std::tuple{"lambda", &a.lambda, &b.lambda, false},
        std::tuple{"h1e_v_sr", &a.h1e_v_sr, &b.h1e_v_sr, false},
        std::tuple{"h1e_v_si", &a.h1e_v_si, &b.h1e_v_si, false},
        std::tuple{"h1e_v_sir", &a.h1e_v_sir, &b.h1e_v_sir, false},
        std::tuple{"e_core", &a.e_core, &b.e_core, false},
        std::tuple{"e_virt", &a.e_virt, &b.e_virt, false}, std::tuple{"h1e", &a.h1e, &b.h1e, true},
        std::tuple{"h2e", &a.h2e, &b.h2e, true}, std::tuple{"dm1", &a.dm1, &b.dm1, true},
        std::tuple{"dm2", &a.dm2, &b.dm2, true}, std::tuple{"dm3", &a.dm3, &b.dm3, true},
        std::tuple{"f3ca", &a.f3ca, &b.f3ca, true}, std::tuple{"f3ac", &a.f3ac, &b.f3ac, true}}) {
    if (swept && !sweep) {
      continue;
    }
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
}

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
  expect_same_bars(whole->second, batched->second);
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(batched->first.energies.energies[k], whole->first.energies.energies[k], 1e-12)
        << kClassNames[k];
    EXPECT_NEAR(batched->first.eq40[k], whole->first.eq40[k], 1e-12) << kClassNames[k];
  }
}

// ---------------------------------------------------------------------------
// The PC reverse pass
// ---------------------------------------------------------------------------

// pc_class_energies_adjoint over @p h, every bar wanted, each zeroed first.
std::optional<std::pair<nevpt2::PcClassEnergiesAdjoint, HostBars>>
run_pc_adjoint(fb::DeviceResources &res, const HostCase &h, const std::int32_t batch) {
  DeviceCase g = upload_case(res, h);
  const DeviceBars b = zero_bars(res, h);
  auto got = nevpt2::pc_class_energies_adjoint(res, *g.blocks, g.in, batch, b.bars);
  EXPECT_TRUE(got) << got.error().message;
  if (!got) {
    return std::nullopt;
  }
  return std::pair{*got, download_bars(res, b)};
}

// pc_class_energies over @p h.
fb::Result<nevpt2::PcClassEnergies> pc_energies(fb::DeviceResources &res, const HostCase &h) {
  DeviceCase g = upload_case(res, h);
  return nevpt2::pc_class_energies(res, *g.blocks, g.in, 0);
}

// The step-free identities, as SC's Eq40AndEulerFromTheBars. A PC class
// energy -c bᵀ (K + Δ S)⁺ b is homogeneous of degree one in (S, S₁, K) (the
// cut is relative to S's largest eigenvalue), so Σ_G <M_G, G> over the blocks
// each class builds them from is its energy; a mis-transcribed index in any
// bar on S, K or S₁, or in its way onto the blocks, fails it. Euler's identity
// in (B, h1e_v²) holds for PC as for SC, since y is linear in the slab and in
// h1e_v. And the energies are the golden's block2 ones.
TEST(Nevpt2PcEnergyAdjoint, HomogeneityAndEulerFromTheBars) {
  FBF_REQUIRE_GPU();
  const std::filesystem::path dir = kDataDir / kAdjointCase;
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(dir);
  const auto got = run_pc_adjoint(res, h, 0);
  ASSERT_TRUE(got.has_value());
  const auto &[adj, bars] = *got;
  expect_pc_golden(dir, kAdjointCase, adj.energies, "pc_class_energies_adjoint");
  double worst = 0.0;
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(adj.homogeneity[k], adj.energies.energies[k],
                1e-12 * std::max(1.0, std::abs(adj.energies.energies[k])))
        << kClassNames[k];
    worst = std::max(worst, std::abs(adj.homogeneity[k] - adj.energies.energies[k]));
  }
  const double euler = lambda_dot(bars.lambda, h, h) + 2.0 * dot(bars.h1e_v_sr, h.h1e_v_sr) +
                       2.0 * dot(bars.h1e_v_si, h.h1e_v_si) +
                       2.0 * dot(bars.h1e_v_sir, h.h1e_v_sir);
  EXPECT_NEAR(euler, 4.0 * adj.energies.total, 1e-12 * std::abs(adj.energies.total));
  std::cout << std::format(
      "[ nevpt2   ] PC reverse: worst homogeneity miss {:.2e}, Euler {:+.2e}\n", worst,
      euler - (4.0 * adj.energies.total));
}

// A random antisymmetric n x n generator, column-major, from a fixed seed.
std::vector<double> random_kappa(const std::size_t n, const std::uint64_t seed) {
  std::mt19937_64 engine(seed);
  std::normal_distribution<double> normal;
  std::vector<double> kappa(n * n, 0.0);
  for (std::size_t q = 0; q < n; ++q) {
    for (std::size_t p = q + 1; p < n; ++p) {
      const double x = normal(engine);
      kappa[p + (n * q)] = x;
      kappa[q + (n * p)] = -x;
    }
  }
  return kappa;
}

// exp(t kappa), column-major, by its Taylor series: |t kappa| is at most a
// few hundredths along the ladder, so thirty terms are past round-off.
std::vector<double> exp_kappa(const std::vector<double> &kappa, const std::size_t n,
                              const double t) {
  std::vector<double> out(n * n, 0.0);
  std::vector<double> term(n * n, 0.0);
  for (std::size_t p = 0; p < n; ++p) {
    out[p + (n * p)] = 1.0;
    term[p + (n * p)] = 1.0;
  }
  for (std::int32_t j = 1; j <= 30; ++j) {
    std::vector<double> next(n * n, 0.0);
    for (std::size_t c = 0; c < n; ++c) {
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t r = 0; r < n; ++r) {
          next[r + (n * c)] += term[r + (n * k)] * kappa[k + (n * c)] * t / j;
        }
      }
    }
    term = std::move(next);
    for (std::size_t k = 0; k < n * n; ++k) {
      out[k] += term[k];
    }
  }
  return out;
}

// @p x, a packed tensor of @p shape (column-major, as rotate reads it), with
// @p u on each axis of @p axes in turn.
std::vector<double> rotated(const std::vector<double> &x, const std::vector<std::size_t> &shape,
                            const std::vector<std::size_t> &axes, const std::vector<double> &u) {
  fb::NpyArray a{.shape = shape, .data = x};
  for (const std::size_t axis : axes) {
    rotate(a, {axis}, u);
  }
  return a.data;
}

// The first-order action of the generator @p kappa on @p axes of @p x: the sum
// over the axes of @p kappa on that axis alone, d/dt of rotated(x, exp(t
// kappa)) at t = 0.
std::vector<double> rotation_tangent(const std::vector<double> &x,
                                     const std::vector<std::size_t> &shape,
                                     const std::vector<std::size_t> &axes,
                                     const std::vector<double> &kappa) {
  std::vector<double> out(x.size(), 0.0);
  for (const std::size_t axis : axes) {
    const std::vector<double> one = rotated(x, shape, {axis}, kappa);
    for (std::size_t k = 0; k < x.size(); ++k) {
      out[k] += one[k];
    }
  }
  return out;
}

// The bars against pc_class_energies itself, along the same seeded groups as
// SC's ladder and with its ratio test and tolerance, kept inside the
// energy's domain three ways.
//
// - **The RDMs move along a rotation.** A raw dm1-dm3 direction gives S's
//   dropped modes first-order eigenvalues, so at any step the rank changes
//   (or the gap check refuses), and the energy has no derivative along it.
//   PC's "rdm" is dm1, dm2 and dm3 along an active rotation exp(t kappa) of
//   all their axes, the RDMs of the rotated state in the fixed orbitals,
//   which carries every metric by a congruence and keeps its spectrum; f3ca
//   and f3ac, which enter K alone, along SC's raw direction. dm1-dm3's share
//   of Σ <bar, dx> is the rotation's tangent.
// - **h and the RDMs are moved at CAS(4,4).** K's eigenvalues over S's kept
//   range move as 1 / σ under a K that no longer annihilates the state's
//   near-null functions, and CAS(8,8)'s Sr and Si keep modes at σ / σ_max =
//   1.0e-10 and 3.1e-10. Measured along "h" there: at t = 1e-2 Sr's smallest
//   denominator is -2.3e6, and -1.6e4 at t = 7.8e-5, so the energy has a
//   pole within about 1e-8 of the base point and no ladder exists. CAS(4,4)
//   keeps every mode above 1.6e-5 (Sir; Sr 9.1e-4). Its Si is zero by N₂'s
//   symmetry, so Si's bars are the slab groups' at CAS(8,8), and the
//   homogeneity sums'.
// - **The ladder starts at t0 = 1e-3**, not SC's 1e-2: at CAS(4,4) a step of
//   1e-2 along "h", "rdm" or "all" makes a denominator negative (Sr, Si or
//   Sir), and so does 5e-3 along "h" and "rdm". The tolerance is SC's.
//
// "h1e_v" has no ratios to test: at fixed S, K and Δ the eigenbasis rows y
// are affine in h1e_v and each class energy is quadratic in y, so every
// central difference is the derivative. Each must equal Σ <bar, dx>, to SC's
// tolerance.
class Nevpt2PcEnergyAdjointLadder
    : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(Nevpt2PcEnergyAdjointLadder, BarsMatchRichardsonLadder) {
  FBF_REQUIRE_GPU();
  const auto [name, group] = GetParam();
  const std::filesystem::path dir = kDataDir / name;
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(dir);
  HostCase straight = random_direction(h, group, 561);
  const bool rotate_rdms = group == "rdm" || group == "all";
  const std::vector<double> kappa = random_kappa(h.na, 562);
  const auto shape = [&](const std::size_t rank) { return std::vector<std::size_t>(rank, h.na); };
  const auto all_axes = [](const std::size_t rank) {
    std::vector<std::size_t> axes(rank);
    std::iota(axes.begin(), axes.end(), std::size_t{0});
    return axes;
  };
  constexpr std::array kRotated{std::pair{&HostCase::dm1, std::size_t{2}},
                                std::pair{&HostCase::dm2, std::size_t{4}},
                                std::pair{&HostCase::dm3, std::size_t{6}}};
  HostCase dx = straight;
  for (const auto [field, rank] : kRotated) {
    std::ranges::fill(straight.*field, 0.0);
    dx.*field = rotate_rdms ? rotation_tangent(h.*field, shape(rank), all_axes(rank), kappa)
                            : std::vector<double>((h.*field).size(), 0.0);
  }
  const auto got = run_pc_adjoint(res, h, 0);
  ASSERT_TRUE(got.has_value());
  const double predicted = bars_dot(got->second, h, dx);

  const auto energy = [&](const double t) {
    HostCase x = displaced(h, straight, t);
    if (rotate_rdms) {
      const std::vector<double> u = exp_kappa(kappa, h.na, t);
      for (const auto [field, rank] : kRotated) {
        x.*field = rotated(h.*field, shape(rank), all_axes(rank), u);
      }
    }
    const auto e = pc_energies(res, x);
    EXPECT_TRUE(e) << "t " << t << ": " << e.error().message;
    return e ? e->total : std::nan("");
  };
  const Ladder l = richardson_ladder(energy, 1e-3);
  if (group == "h1e_v") {
    for (const double d : l.d) {
      EXPECT_NEAR(d, predicted, 1e-7 * std::abs(predicted));
    }
  } else {
    ASSERT_GE(l.run, 3) << "ratios:" << l.ratios;
    EXPECT_NEAR(l.richardson, predicted, 1e-7 * std::abs(predicted)) << "ratios:" << l.ratios;
  }
  const double got_d = group == "h1e_v" ? l.d.front() : l.richardson;
  std::cout << std::format("[ nevpt2   ] PC reverse ladder {} {}: Σ<bar, dx> {:.12e}, {} "
                           "{:.12e}, relative {:.2e}, ratios{}\n",
                           name, group, predicted, group == "h1e_v" ? "D(t0)" : "Richardson", got_d,
                           std::abs(got_d - predicted) / std::abs(predicted), l.ratios);
}

// The slab groups at CAS(8,8), where Si is not zero; h, the RDMs and all of
// them together at CAS(4,4). Each is a few seconds at most, so "all" is in
// the fast tier here.
INSTANTIATE_TEST_SUITE_P(
    Groups, Nevpt2PcEnergyAdjointLadder,
    ::testing::Values(std::tuple{kAdjointCase, std::string{"B"}},
                      std::tuple{kAdjointCase, std::string{"h1e_v"}},
                      std::tuple{kAdjointCase, std::string{"e"}},
                      std::tuple{std::string{"n2_ccpvdz_cas44_df"}, std::string{"h"}},
                      std::tuple{std::string{"n2_ccpvdz_cas44_df"}, std::string{"rdm"}},
                      std::tuple{std::string{"n2_ccpvdz_cas44_df"}, std::string{"all"}}),
    [](const auto &info) { return std::get<0>(info.param) + "_" + std::get<1>(info.param); });

// PC's invariance to the inactive orbitals, step-free from the bars. With the
// core orbital energies all equal, and the virtual ones, the canonical PC
// energy is the PC energy of any core and any virtual basis: every class sums
// a quadratic form in its tuples' rows y over every ordered core and virtual
// index, at one Δ. So its derivative along a rotation within the core, or
// within the virtuals, is zero: Σ <bar, dx> over the inputs that rotation
// moves (B_ca, B_cv, h1e_v_Si and h1e_v_Sir's core axes; B_va, B_cv, h1e_v_Sr
// and h1e_v_Sir's virtual ones), against the scale of its terms. The orbital
// energies do not move at first order, and nothing active does. SC is the
// control: its contraction to one perturber per external tuple is not
// invariant (Park), so its derivative along the same rotation is not zero.
TEST(Nevpt2PcEnergyAdjoint, InactiveRotationsAreRoundOff) {
  FBF_REQUIRE_GPU();
  const std::filesystem::path dir = kDataDir / kAdjointCase;
  fb::DeviceResources res(0);
  HostCase h = load_host_case(dir);
  for (std::vector<double> *e : {&h.e_core, &h.e_virt}) {
    const double mean = std::accumulate(e->begin(), e->end(), 0.0) / static_cast<double>(e->size());
    std::ranges::fill(*e, mean);
  }
  const auto pc = run_pc_adjoint(res, h, 0);
  const auto sc = run_adjoint(res, h, 0);
  ASSERT_TRUE(pc.has_value());
  ASSERT_TRUE(sc.has_value());

  // The direction: every field zero, then the rotation's tangent on the axes
  // of the space it rotates.
  const auto direction = [&](const bool core) {
    HostCase dx = random_direction(h, "none", 0);
    const std::size_t m = core ? h.nc : h.nv;
    const std::vector<double> kappa = random_kappa(m, core ? 563 : 564);
    const auto put = [&](std::vector<double> HostCase::*field, std::vector<std::size_t> shape,
                         const std::size_t axis) {
      const std::vector<double> t = rotation_tangent(h.*field, shape, {axis}, kappa);
      for (std::size_t k = 0; k < t.size(); ++k) {
        (dx.*field)[k] += t[k];
      }
    };
    if (core) {
      put(&HostCase::ca, {h.naux, h.nc, h.na}, 1);
      put(&HostCase::cv, {h.naux, h.nc, h.nv}, 1);
      put(&HostCase::h1e_v_si, {h.na, h.nc}, 1);
      put(&HostCase::h1e_v_sir, {h.nv, h.nc}, 1);
    } else {
      put(&HostCase::va, {h.naux, h.nv, h.na}, 1);
      put(&HostCase::cv, {h.naux, h.nc, h.nv}, 2);
      put(&HostCase::h1e_v_sr, {h.nv, h.na}, 0);
      put(&HostCase::h1e_v_sir, {h.nv, h.nc}, 0);
    }
    return dx;
  };
  // Σ <bar, dx>, and the sum of its terms' magnitudes: B's through Λ and each
  // h1e_v's apart.
  const auto derivative = [&](const HostBars &bars, const HostCase &dx) {
    const std::array terms{lambda_dot(bars.lambda, h, dx), dot(bars.h1e_v_sr, dx.h1e_v_sr),
                           dot(bars.h1e_v_si, dx.h1e_v_si), dot(bars.h1e_v_sir, dx.h1e_v_sir)};
    double sum = 0.0;
    double scale = 0.0;
    for (const double t : terms) {
      sum += t;
      scale += std::abs(t);
    }
    return std::pair{sum, scale};
  };
  for (const bool core : {true, false}) {
    const HostCase dx = direction(core);
    const auto [g_pc, scale_pc] = derivative(pc->second, dx);
    const auto [g_sc, scale_sc] = derivative(sc->second, dx);
    const char *const what = core ? "core-core" : "virtual-virtual";
    EXPECT_GT(scale_pc, 0.0) << what;
    EXPECT_LE(std::abs(g_pc), 1e-12 * scale_pc) << what;
    EXPECT_GT(std::abs(g_sc), 1e-6 * scale_sc) << what << ": SC, the control";
    std::cout << std::format("[ nevpt2   ] {} rotation at degenerate orbital energies: PC {:+.2e} "
                             "of {:.2e}, SC {:+.2e} of {:.2e}\n",
                             what, g_pc, scale_pc, g_sc, scale_sc);
  }
}

// The walk's batch only splits it, as for SC: the bars the walk forms (Λ,
// h1e_v's, the orbital energies') match the whole walk's to 1e-12. The
// sweep's bars (h1e, h2e, the RDMs') are not compared element by element:
// they go through the solve's U, whose run-to-run spread (einsum's atomics
// reassociate S and K, and CAS(8,8)'s Sr and Si keep modes at 1e-10 of their
// largest) already puts two whole walks 1.3e-8 of the largest element apart
// on the gfx1200. Their batch independence is the homogeneity sums', at
// 1e-12.
TEST(Nevpt2PcEnergyAdjoint, BatchedBarsMatchWhole) {
  FBF_REQUIRE_GPU();
  const std::filesystem::path dir = kDataDir / kAdjointCase;
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(dir);
  const auto whole = run_pc_adjoint(res, h, 0);
  const auto batched = run_pc_adjoint(res, h, 3);
  ASSERT_TRUE(whole.has_value());
  ASSERT_TRUE(batched.has_value());
  expect_same_bars(whole->second, batched->second, false);
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(batched->first.energies.energies[k], whole->first.energies.energies[k], 1e-12)
        << kClassNames[k];
    EXPECT_NEAR(batched->first.homogeneity[k], whole->first.homogeneity[k], 1e-12)
        << kClassNames[k];
  }
}

// A class the energy refuses is refused by the adjoint, with the energy's
// error: never a derivative with that class summed in as zero. Two states:
// zero RDMs, whose Sr metric has no positive eigenvalue; and the golden's
// with every virtual orbital energy at -1000 Eh, so that Sr's first
// denominator λ + ε_r is negative.
TEST(Nevpt2PcEnergyAdjoint, RefusesWhatTheEnergyRefuses) {
  FBF_REQUIRE_GPU();
  const std::filesystem::path dir = kDataDir / kAdjointCase;
  fb::DeviceResources res(0);
  const HostCase golden = load_host_case(dir);
  HostCase zero_rdms = golden;
  for (std::vector<double> *rdm :
       {&zero_rdms.dm1, &zero_rdms.dm2, &zero_rdms.dm3, &zero_rdms.f3ca, &zero_rdms.f3ac}) {
    std::ranges::fill(*rdm, 0.0);
  }
  HostCase low_virtuals = golden;
  std::ranges::fill(low_virtuals.e_virt, -1000.0);
  for (const auto &[what, h, needle] :
       {std::tuple{"zero RDMs", &zero_rdms, "no positive eigenvalue"},
        std::tuple{"virtuals at -1000 Eh", &low_virtuals, "not positive"}}) {
    const auto energy = pc_energies(res, *h);
    ASSERT_FALSE(energy) << what;
    DeviceCase g = upload_case(res, *h);
    const DeviceBars b = zero_bars(res, *h);
    const auto adj = nevpt2::pc_class_energies_adjoint(res, *g.blocks, g.in, 0, b.bars);
    ASSERT_FALSE(adj) << what;
    EXPECT_EQ(adj.error().kind, fb::ErrorKind::Numerical) << what;
    EXPECT_EQ(adj.error().kind, energy.error().kind) << what;
    EXPECT_EQ(adj.error().message, energy.error().message) << what;
    EXPECT_NE(adj.error().message.find("Sr"), std::string::npos)
        << what << ": " << adj.error().message;
    EXPECT_NE(adj.error().message.find(needle), std::string::npos)
        << what << ": " << adj.error().message;
  }
}

} // namespace
