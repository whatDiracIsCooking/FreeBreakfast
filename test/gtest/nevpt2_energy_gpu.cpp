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
// class_energies itself, and the same bars whatever the batch; and its
// forward tangent (class_energies_tangent), dual to those bars along a block
// of random directions on every case, each with empty perturbers; and its
// Hessian-vector product (class_energies_hvp) against a ladder of those
// bars, with CAS(8,8)'s empty perturbers' K6 against a mixed second
// difference of the energy, and their ρ_τ refusal. And PC's
// (pc_class_energies_adjoint): the same, homogeneity in (S, S₁, K) in place
// of Eq. 40 and the ladder kept inside the energy's domain, its invariance to
// inactive rotations at degenerate orbital energies, and its refusals. And
// SC-QD's (sc_effective_hamiltonian_adjoint), over data/qd_nevpt2/: Eq. 40
// and Euler from its bars under a random H̄ on all six cases, one state as
// class_energies_adjoint, the batch, and a Richardson ladder of
// Σ H̄_IJ H_IJ along every input, every pair's transition RDMs included.
// And PC-QD's (pc_effective_hamiltonian_adjoint), the same way: homogeneity
// and Euler on all six cases, one state as pc_class_energies_adjoint, the
// batch, its ladders (every group on three cases, the slabs and orbital
// energies on the other three), and its refusals. And PC's forward tangent
// (pc_class_energies_tangent), dual to pc_class_energies_adjoint's bars on a
// golden whose cuts drop exact null modes alone and on PSB3 twisted to 90°,
// whose Sr and Si cuts drop real ones, and its refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.npy;
import free_breakfast.ci;
import free_breakfast.nevpt2;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

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

// @p dir's dm1, dm2 and dm3, and with @p f3 its f3ca and f3ac, into @p h.
void load_rdms(HostCase &h, const std::filesystem::path &dir, const bool f3) {
  h.dm1 = row_major(read_array(dir / "dm1.npy"));
  h.dm2 = row_major(read_array(dir / "dm2.npy"));
  h.dm3 = row_major(read_array(dir / "dm3.npy"));
  if (f3) {
    h.f3ca = row_major(read_array(dir / "f3ca.npy"));
    h.f3ac = row_major(read_array(dir / "f3ac.npy"));
  }
}

// @p dir's B blocks, integrals and orbital energies, and with @p rdms its
// RDMs.
HostCase load_host_case(const std::filesystem::path &dir, const bool rdms = true) {
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
  if (rdms) {
    load_rdms(h, dir, true);
  }
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
  // the fast tier. And B, 16.1 s on the gfx1200 against 1.9 s on the RTX 3080
  // (#793).
  if (group == "all" || group == "B") {
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

// Every bar of @p b against @p a's to @p relative of its largest element,
// and every one formed (not all zero): the walk's (Λ, h1e_v's, the orbital
// energies'), and with @p sweep the sweep's (h1e, h2e, the RDMs').
void expect_same_bars(const HostBars &a, const HostBars &b, const bool sweep,
                      const double relative) {
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
    EXPECT_LE(worst, relative * scale) << name;
  }
}

// The walk's batch only splits it: walked in threes (every virtual walk ends
// in a partial slab), every bar is the whole walk's to round-off.
//
// Round-off here is recipes::einsum's floor, not the batch's: every term is
// added with an atomicAdd (src/recipes/einsum/einsum.cu), so two whole walks
// differ as much as a batched and a whole one do. dm2's bar, the worst, as a
// fraction of its largest element (#640):
//   batched vs whole, gfx1200, 50 runs: median 2.9e-13, max 1.06e-12
//   whole vs whole,   gfx1200, 30 runs: median 3.0e-13, max 6.6e-13
//   batched vs whole, RTX 3080, 20 runs: median 3.1e-13, max 1.02e-12
//   whole vs whole,   RTX 3080, 20 runs: median 4.3e-13, max 8.9e-13
// (dm3, h2e, dm1 and f3ca up to 7.6e-13-8.5e-13). The bar was 1e-12, on that
// floor, and failed one run in 20-50 on each card; 5e-12 after it was still
// a measured floor, so it is the standing rule, 1e-10 of the largest element
// (#775: https://github.com/whatDiracIsCooking/FreeBreakfast/issues/775#issuecomment-6608877300).
constexpr double kBatchedBarsTolerance = 1e-10;

TEST(Nevpt2EnergyAdjoint, BatchedBarsMatchWhole) {
  FBF_REQUIRE_GPU();
  const std::filesystem::path dir = kDataDir / kAdjointCase;
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(dir);
  const auto whole = run_adjoint(res, h, 0);
  const auto batched = run_adjoint(res, h, 3);
  ASSERT_TRUE(whole.has_value());
  ASSERT_TRUE(batched.has_value());
  expect_same_bars(whole->second, batched->second, true, kBatchedBarsTolerance);
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(batched->first.energies.energies[k], whole->first.energies.energies[k], 1e-12)
        << kClassNames[k];
    EXPECT_NEAR(batched->first.eq40[k], whole->first.eq40[k], 1e-12) << kClassNames[k];
  }
}

// ---------------------------------------------------------------------------
// The SC forward tangent
// ---------------------------------------------------------------------------

// @p dx on the device as a direction: an array that is all zero is null (the
// tangent's zero), and so are the blocks when B does not move. @p g is @p dx
// uploaded, and must outlive the direction.
nevpt2::EnergyDirection direction_of(const DeviceCase &g, const HostCase &dx) {
  const auto moves = [](const std::vector<double> &a) {
    return std::ranges::any_of(a, [](const double v) { return v != 0.0; });
  };
  const auto given = [&](const std::vector<double> &a, const double *const p) {
    return moves(a) ? p : nullptr;
  };
  const bool moves_b = moves(dx.aa) || moves(dx.ca) || moves(dx.va) || moves(dx.cv);
  return {.in = {.h1e = given(dx.h1e, g.in.h1e),
                 .h2e = given(dx.h2e, g.in.h2e),
                 .e_core = given(dx.e_core, g.in.e_core),
                 .e_virt = given(dx.e_virt, g.in.e_virt),
                 .h1e_v_sr = given(dx.h1e_v_sr, g.in.h1e_v_sr),
                 .h1e_v_si = given(dx.h1e_v_si, g.in.h1e_v_si),
                 .h1e_v_sir = given(dx.h1e_v_sir, g.in.h1e_v_sir),
                 .dm1 = given(dx.dm1, g.in.dm1),
                 .dm2 = given(dx.dm2, g.in.dm2),
                 .dm3 = given(dx.dm3, g.in.dm3),
                 .f3ca = given(dx.f3ca, g.in.f3ca),
                 .f3ac = given(dx.f3ac, g.in.f3ac)},
          .blocks = moves_b ? &*g.blocks : nullptr};
}

// Σ <bar, dx>'s summands, one per input: B through Λ, then every
// EnergyInputs array.
std::vector<double> bars_summands(const HostBars &bars, const HostCase &h, const HostCase &dx) {
  return {lambda_dot(bars.lambda, h, dx),  dot(bars.h1e_v_sr, dx.h1e_v_sr),
          dot(bars.h1e_v_si, dx.h1e_v_si), dot(bars.h1e_v_sir, dx.h1e_v_sir),
          dot(bars.e_core, dx.e_core),     dot(bars.e_virt, dx.e_virt),
          dot(bars.h1e, dx.h1e),           dot(bars.h2e, dx.h2e),
          dot(bars.dm1, dx.dm1),           dot(bars.dm2, dx.dm2),
          dot(bars.dm3, dx.dm3),           dot(bars.f3ca, dx.f3ca),
          dot(bars.f3ac, dx.f3ac)};
}

// The forward tangent (class_energies_tangent) against the reverse pass's
// bars, which Nevpt2EnergyAdjointLadder validates: on one block call along
// seeded random directions (each input group alone, then all of them), every
// direction's Ė must be Σ <bar, dx> to 1e-10 of that sum's largest summand.
// A ladder of class_energies would add nothing the bars' ladder does not
// hold. Every case has norm-guarded, empty perturbers, zero by N₂'s symmetry
// (at CAS(4,4) all of Si's, whose (σπ|ππ) and h1e_v_Si vanish, and over half
// of Sr's, Sijr's, Srsi's and Srs's; at CAS(8,8) a fifth of Sijr's and a
// quarter of Srsi's): their bars are zero, so a nonzero tangent there fails
// duality. Walked whole, and in threes (a partial last slab).
class Nevpt2EnergyTangent : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t>> {
};

TEST_P(Nevpt2EnergyTangent, DualToTheBars) {
  FBF_REQUIRE_GPU();
  const auto [name, batch] = GetParam();
  // CAS(8,8) is ~7 s a case on the gfx1200 (CAS(4,4) 0.2-0.8 s); batched,
  // the CAS(4,4) cases stand for it in the fast tier.
  if (name == "n2_ccpvdz_cas88_df" && batch != 0) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const std::filesystem::path dir = kDataDir / name;
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(dir);
  const auto adj = run_adjoint(res, h, 0);
  ASSERT_TRUE(adj.has_value());

  const std::array<std::string, 6> groups{"B", "h1e_v", "e", "h", "rdm", "all"};
  std::vector<HostCase> dx;
  for (std::size_t u = 0; u < groups.size(); ++u) {
    dx.push_back(random_direction(h, groups[u], 805 + u));
  }
  DeviceCase g = upload_case(res, h);
  std::vector<DeviceCase> gd;
  gd.reserve(dx.size());
  for (const HostCase &d : dx) {
    gd.push_back(upload_case(res, d));
  }
  std::vector<nevpt2::EnergyDirection> directions;
  for (std::size_t u = 0; u < dx.size(); ++u) {
    directions.push_back(direction_of(gd[u], dx[u]));
  }
  const auto got = nevpt2::class_energies_tangent(res, *g.blocks, g.in, batch, directions);
  ASSERT_TRUE(got) << got.error().message;
  ASSERT_EQ(got->total_dot.size(), groups.size());

  double worst = 0.0;
  for (std::size_t u = 0; u < groups.size(); ++u) {
    const std::vector<double> summands = bars_summands(adj->second, h, dx[u]);
    const double predicted = std::accumulate(summands.begin(), summands.end(), 0.0);
    double scale = 0.0;
    for (const double x : summands) {
      scale = std::max(scale, std::abs(x));
    }
    const double miss = std::abs(got->total_dot[u] - predicted);
    EXPECT_LE(miss, 1e-10 * scale)
        << groups[u] << ": Ė " << got->total_dot[u] << ", Σ<bar, dx> " << predicted;
    worst = std::max(worst, miss / scale);
    std::cout << std::format("[ nevpt2   ] tangent {} batch {} {}: Ė {:+.12e}, Σ<bar, dx> "
                             "{:+.12e}, miss / largest summand {:.2e}\n",
                             name, batch, groups[u], got->total_dot[u], predicted, miss / scale);
  }
  std::size_t empty = 0;
  std::string per_class;
  for (std::size_t k = 0; k < 8; ++k) {
    const auto guarded =
        std::ranges::count_if(got->norm[k], [](const double x) { return std::abs(x) <= 1e-14; });
    empty += static_cast<std::size_t>(guarded);
    per_class += std::format(" {} {}/{}", kClassNames[k], guarded, got->norm[k].size());
  }
  EXPECT_GT(empty, 0U) << "no norm-guarded perturber:" << per_class;
  std::cout << std::format("[ nevpt2   ] tangent {} batch {}: worst {:.2e}; guarded{}\n", name,
                           batch, worst, per_class);
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2EnergyTangent,
                         ::testing::Combine(::testing::ValuesIn(kRdmCases),
                                            ::testing::Values(0, 3)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + "_batch" +
                                  std::to_string(std::get<1>(info.param));
                         });

// ---------------------------------------------------------------------------
// The SC Hessian-vector product
// ---------------------------------------------------------------------------

// CAS(8,8): its empty perturbers, zero by N₂'s symmetry, are in two H
// classes (24 of Sijr's 102, all 120 of Srsi's guarded), which K6 and the
// ρ_τ refusal are about. CAS(4,4) has more (all of Si's, over half of Sr's,
// Sijr's, Srsi's and Srs's) but also real perturbers with norms down to
// 6e-8, whose Rayleigh quotients put a pole of the energy within ~1e-5 of
// the base point along a random direction of every input: no ladder
// reaches 1e-7 there.
const std::string kHvpCase = "n2_ccpvdz_cas88_df";

// Every bar field of @p b, named, in HostBars' order of the fields.
std::vector<std::pair<std::string, const std::vector<double> *>> bar_fields(const HostBars &b) {
  return {{"lambda", &b.lambda},     {"h1e_v_sr", &b.h1e_v_sr},
          {"h1e_v_si", &b.h1e_v_si}, {"h1e_v_sir", &b.h1e_v_sir},
          {"e_core", &b.e_core},     {"e_virt", &b.e_virt},
          {"h1e", &b.h1e},           {"h2e", &b.h2e},
          {"dm1", &b.dm1},           {"dm2", &b.dm2},
          {"dm3", &b.dm3},           {"f3ca", &b.f3ca},
          {"f3ac", &b.f3ac}};
}

// class_energies_hvp over @p h along each of @p dx (with @p partner, one per
// direction, when not empty), every bar wanted.
fb::Result<std::pair<nevpt2::ClassEnergiesHvp, std::vector<HostBars>>>
run_hvp(fb::DeviceResources &res, const HostCase &h, const std::vector<HostCase> &dx,
        const std::vector<HostCase> &partner) {
  DeviceCase g = upload_case(res, h);
  std::vector<DeviceCase> gd;
  std::vector<DeviceCase> gp;
  gd.reserve(dx.size());
  gp.reserve(partner.size());
  std::vector<nevpt2::EnergyDirection> directions;
  std::vector<nevpt2::EnergyDirection> partners;
  std::vector<DeviceBars> b;
  std::vector<nevpt2::EnergyBars> bars;
  for (const HostCase &d : dx) {
    gd.push_back(upload_case(res, d));
    directions.push_back(direction_of(gd.back(), d));
    b.push_back(zero_bars(res, h));
    bars.push_back(b.back().bars);
  }
  for (const HostCase &d : partner) {
    gp.push_back(upload_case(res, d));
    partners.push_back(direction_of(gp.back(), d));
  }
  auto got = nevpt2::class_energies_hvp(res, *g.blocks, g.in, 0, directions, bars, partners);
  if (!got) {
    return std::unexpected(got.error());
  }
  std::vector<HostBars> host;
  for (const DeviceBars &x : b) {
    host.push_back(download_bars(res, x));
  }
  return std::pair{std::move(*got), std::move(host)};
}

// The Hessian-vector product against a ladder of class_energies_adjoint's
// bars, field by field, along seeded random directions: every input at
// once, and B alone and the RDMs alone. Central differences D(t) of the bars
// at t = 1e-3 / 2^k, k < 8, their Richardson extrapolants R_k =
// (4 D_{k+1} - D_k) / 3, and the R_k that agrees best with R_{k+1} (the
// largest field-wise change, relative to the field's largest element,
// smallest): every field of the product must be that R_k's to 1e-7 of the
// field's largest element. Every direction but the RDMs' breaks N₂'s
// symmetry and populates the empty perturbers, and the ladder sees their
// K6: off the base point their norms clear the guard, so the bars there
// are the ray's limit, K6's. Without K6, Λ's misses 5e-3 (B) and 2e-3
// (every input).
class Nevpt2EnergyHvp : public ::testing::TestWithParam<std::string> {};

TEST_P(Nevpt2EnergyHvp, BarsMatchRichardsonLadder) {
  FBF_REQUIRE_GPU();
  const std::string group = GetParam();
  // ~9 s a group on the RTX 3080 (one product, sixteen reverse passes):
  // every input at once stands for the groups in the fast tier.
  if (group != "all") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const std::filesystem::path dir = kDataDir / kHvpCase;
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(dir);
  const HostCase dx = random_direction(h, group, 811);
  const auto got = run_hvp(res, h, {dx}, {});
  ASSERT_TRUE(got) << got.error().message;
  const HostBars &hvp = got->second.front();

  constexpr std::int32_t kRungs = 8;
  constexpr double kT0 = 1e-3;
  std::vector<std::vector<std::vector<double>>> d; // [k][field]
  for (std::int32_t k = 0; k < kRungs; ++k) {
    const double t = std::ldexp(kT0, -k);
    const auto plus = run_adjoint(res, displaced(h, dx, t), 0);
    const auto minus = run_adjoint(res, displaced(h, dx, -t), 0);
    ASSERT_TRUE(plus.has_value() && minus.has_value());
    const auto fp = bar_fields(plus->second);
    const auto fm = bar_fields(minus->second);
    std::vector<std::vector<double>> rung;
    for (std::size_t f = 0; f < fp.size(); ++f) {
      std::vector<double> x(fp[f].second->size());
      for (std::size_t e = 0; e < x.size(); ++e) {
        x[e] = ((*fp[f].second)[e] - (*fm[f].second)[e]) / (2.0 * t);
      }
      rung.push_back(std::move(x));
    }
    d.push_back(std::move(rung));
  }
  const auto fields = bar_fields(hvp);
  std::vector<double> scale(fields.size(), 0.0);
  for (std::size_t f = 0; f < fields.size(); ++f) {
    for (const double x : *fields[f].second) {
      scale[f] = std::max(scale[f], std::abs(x));
    }
  }
  // R_k per field, and the k whose R_k moves least to R_{k+1}.
  const auto richardson = [&](const std::size_t k, const std::size_t f, const std::size_t e) {
    return ((4.0 * d[k + 1][f][e]) - d[k][f][e]) / 3.0;
  };
  std::size_t best = 0;
  double best_change = std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k + 2 < static_cast<std::size_t>(kRungs); ++k) {
    double change = 0.0;
    for (std::size_t f = 0; f < fields.size(); ++f) {
      for (std::size_t e = 0; e < fields[f].second->size(); ++e) {
        if (scale[f] > 0.0) {
          change =
              std::max(change, std::abs(richardson(k + 1, f, e) - richardson(k, f, e)) / scale[f]);
        }
      }
    }
    if (change < best_change) {
      best_change = change;
      best = k;
    }
  }
  double worst = 0.0;
  std::string report;
  for (std::size_t f = 0; f < fields.size(); ++f) {
    double miss = 0.0;
    for (std::size_t e = 0; e < fields[f].second->size(); ++e) {
      miss = std::max(miss, std::abs((*fields[f].second)[e] - richardson(best, f, e)));
    }
    EXPECT_GT(scale[f], 0.0) << group << " " << fields[f].first;
    EXPECT_LE(miss, 1e-7 * scale[f]) << group << " " << fields[f].first;
    const double relative = scale[f] > 0.0 ? miss / scale[f] : 0.0;
    worst = std::max(worst, relative);
    report += std::format(" {} {:.1e}", fields[f].first, relative);
  }
  std::int64_t populated = 0;
  for (const std::int64_t n : got->first.populated.front()) {
    populated += n;
  }
  std::cout << std::format("[ nevpt2   ] hvp ladder {}: rung {} (R change {:.1e}), worst {:.2e}, "
                           "{} empty perturbers populated;{}\n",
                           group, best, best_change, worst, populated, report);
}

INSTANTIATE_TEST_SUITE_P(Groups, Nevpt2EnergyHvp,
                         ::testing::Values(std::string{"all"}, std::string{"B"},
                                           std::string{"rdm"}),
                         [](const auto &info) { return info.param; });

// A direction that keeps N₂'s symmetry, so populates no empty perturber: B
// and h1e_v's rows scaled entry by entry by seeded normal factors (B's one
// factor per orbital pair, the same for every auxiliary index, so every slab
// entry is scaled too and a symmetry zero stays zero), and h1e, h2e, the
// RDMs and the orbital energies random (the perturbers' integrals do not
// read them).
HostCase symmetric_direction(const HostCase &x, const std::uint64_t seed) {
  std::mt19937_64 engine(seed);
  std::normal_distribution<double> normal;
  HostCase d = random_direction(x, "h", seed + 1);
  for (const std::string group : {"rdm", "e"}) {
    const HostCase r = random_direction(x, group, seed + 2);
    d = displaced(d, r, 1.0);
  }
  for (auto [y, from, pairs] :
       {std::tuple{&d.aa, &x.aa, x.na * x.na}, std::tuple{&d.ca, &x.ca, x.nc * x.na},
        std::tuple{&d.va, &x.va, x.nv * x.na}, std::tuple{&d.cv, &x.cv, x.nc * x.nv}}) {
    for (std::size_t pq = 0; pq < pairs; ++pq) {
      const double f = normal(engine);
      for (std::size_t q = 0; q < x.naux; ++q) {
        (*y)[q + (x.naux * pq)] = f * (*from)[q + (x.naux * pq)];
      }
    }
  }
  for (auto [y, from] : {std::pair{&d.h1e_v_sr, &x.h1e_v_sr}, std::pair{&d.h1e_v_si, &x.h1e_v_si},
                         std::pair{&d.h1e_v_sir, &x.h1e_v_sir}}) {
    for (std::size_t e = 0; e < y->size(); ++e) {
      (*y)[e] = normal(engine) * (*from)[e];
    }
  }
  return d;
}

// The empty perturbers' two branches, on a pair of directions. K6: u breaks
// the symmetry (every input random) and v = u / 2 + w with w keeping it, so
// every empty perturber v populates it populates along u's own perturber,
// and H is proportional to N on the pair; ⟨v, HVP u⟩ must then be a ladder
// of the energy's mixed second difference along (u, v),
// (E(t, t) - E(t, -t) - E(-t, t) + E(-t, -t)) / 4t², to 1e-7; without K6
// it misses by 9e-4. The bars' ladder sees K6 along one direction; this
// sees it across a pair, at the pair's η. And
// the refusal: two independent random B directions populate the same empty
// perturbers with different functions, so H is not proportional to N there,
// and the call is refused as Numerical, naming the class and the perturber.
TEST(Nevpt2EnergyHvpEmpty, K6MatchesMixedLadderAndNonProportionalIsRefused) {
  FBF_REQUIRE_GPU();
  // ~5 s on the RTX 3080 and ~12 s on the gfx1200, its family's one case:
  // 32 energies and two products.
  const std::filesystem::path dir = kDataDir / kHvpCase;
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(dir);
  const HostCase u = random_direction(h, "all", 894);
  const HostCase v = displaced(symmetric_direction(h, 895), u, 0.5);

  const auto got = run_hvp(res, h, {u}, {v});
  ASSERT_TRUE(got) << got.error().message;
  const double predicted = bars_dot(got->second.front(), h, v);
  std::int64_t populated = 0;
  std::string per_class;
  for (std::size_t k = 0; k < 8; ++k) {
    populated += got->first.populated.front()[k];
    per_class += std::format(" {} {}", kClassNames[k], got->first.populated.front()[k]);
  }
  EXPECT_GT(populated, 0) << per_class;

  const auto energy = [&](const double a, const double b) {
    DeviceCase g = upload_case(res, displaced(displaced(h, u, a), v, b));
    const auto e = nevpt2::class_energies(res, *g.blocks, g.in, 0);
    EXPECT_TRUE(e) << e.error().message;
    return e ? e->total : std::nan("");
  };
  // D(t) at t = 2.5e-3 / 2^k, k < 8, and its two-level Richardson
  // extrapolants R_k (the t² and t⁴ terms cancelled); the R_k that moves
  // least to R_{k+1} is the ladder's value. Single-level Richardson cannot
  // reach 1e-7 here: its t⁴ term is still 1.6e-6 where the energy's
  // round-off, σ_E / t², takes over (t ≈ 2e-5).
  constexpr std::int32_t kRungs = 8;
  std::vector<double> d;
  for (std::int32_t k = 0; k < kRungs; ++k) {
    const double t = std::ldexp(2.5e-3, -k);
    d.push_back((energy(t, t) - energy(t, -t) - energy(-t, t) + energy(-t, -t)) / (4.0 * t * t));
  }
  const auto r2 = [&](const std::size_t k) {
    const double r = ((4.0 * d[k + 1]) - d[k]) / 3.0;
    const double r_next = ((4.0 * d[k + 2]) - d[k + 1]) / 3.0;
    return ((16.0 * r_next) - r) / 15.0;
  };
  std::size_t best = 0;
  double best_change = std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k + 3 < d.size(); ++k) {
    const double change = std::abs(r2(k + 1) - r2(k));
    if (change < best_change) {
      best_change = change;
      best = k;
    }
  }
  const double ladder = r2(best);
  const double miss = std::abs(ladder - predicted);
  EXPECT_LE(miss, 1e-7 * std::abs(predicted));
  std::cout << std::format("[ nevpt2   ] hvp K6: <v, HVP u> {:+.12e}, mixed ladder {:+.12e} (rung "
                           "{}, change {:.1e}), relative {:.2e}, worst rho {:.2e}, populated{}\n",
                           predicted, ladder, best, best_change / std::abs(predicted),
                           miss / std::abs(predicted), got->first.defect.front(), per_class);

  const HostCase b1 = random_direction(h, "B", 896);
  const HostCase b2 = random_direction(h, "B", 897);
  const auto refused = run_hvp(res, h, {b1}, {b2});
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::Numerical) << refused.error().message;
  EXPECT_NE(refused.error().message.find("class "), std::string::npos) << refused.error().message;
  EXPECT_NE(refused.error().message.find("perturber "), std::string::npos)
      << refused.error().message;
  std::cout << std::format("[ nevpt2   ] hvp refused: {}\n", refused.error().message);
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
  expect_same_bars(whole->second, batched->second, false, kBatchedBarsTolerance);
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(batched->first.energies.energies[k], whole->first.energies.energies[k], 1e-12)
        << kClassNames[k];
    EXPECT_NEAR(batched->first.homogeneity[k], whole->first.homogeneity[k], 1e-12)
        << kClassNames[k];
  }
}

// The shift of every virtual orbital energy that puts @p mode's denominator
// at a quarter of its own coupling |y| = |t (λ + Δ)|: amplitude ±4, an
// intruder. Sr's Δ is ε_r, so its denominators move by exactly the shift.
double intruder_shift(const nevpt2::PcMode &mode) {
  return (std::abs(mode.amplitude * mode.denominator) / 4.0) - mode.denominator;
}

// @p message with its computed numbers blanked. An intruder's message
// carries the mode's amplitude y / (λ + Δ), and y is the slab walk's: the
// energy and its adjoint reassociate its einsums, and at CAS(8,8), whose Sr
// keeps modes at 1e-10 of its largest (README "Conditioning"), a noise-led
// mode's y differs between them in the third digit.
std::string stable_part(const std::string &message) {
  static const std::regex computed(R"(-?\d\.\d+e[+-]\d+)");
  return std::regex_replace(message, computed, "#");
}

// A class the energy refuses is refused by the adjoint, with the energy's
// error: never a derivative with that class summed in as zero. Two states:
// zero RDMs, whose Sr metric has no positive eigenvalue; and an intruder, the
// golden's with every virtual orbital energy shifted to put Sr's strongest
// mode (largest |y / (λ + Δ)|) at a quarter of its coupling, |t| = 4.
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
  const auto base = pc_energies(res, golden);
  ASSERT_TRUE(base.has_value());
  const double shift = intruder_shift(base->strongest[0]);
  HostCase intruder = golden;
  for (double &e : intruder.e_virt) {
    e += shift;
  }
  for (const auto &[what, h, needle] :
       {std::tuple{"zero RDMs", &zero_rdms, "no positive eigenvalue"},
        std::tuple{"an intruder in Sr", &intruder, "has an intruder"}}) {
    const auto energy = pc_energies(res, *h);
    ASSERT_FALSE(energy) << what;
    DeviceCase g = upload_case(res, *h);
    const DeviceBars b = zero_bars(res, *h);
    const auto adj = nevpt2::pc_class_energies_adjoint(res, *g.blocks, g.in, 0, b.bars);
    ASSERT_FALSE(adj) << what;
    EXPECT_EQ(adj.error().kind, fb::ErrorKind::Numerical) << what;
    EXPECT_EQ(adj.error().kind, energy.error().kind) << what;
    EXPECT_EQ(stable_part(adj.error().message), stable_part(energy.error().message)) << what;
    EXPECT_NE(adj.error().message.find("Sr"), std::string::npos)
        << what << ": " << adj.error().message;
    EXPECT_NE(adj.error().message.find(needle), std::string::npos)
        << what << ": " << adj.error().message;
  }
}

// What is not refused (#874): no sign test, and no pole without a coupling.
// A perturber below its reference is a negative denominator, which an
// excited state meets (ethylene S0's π*² root, Sr at -0.12 Eh, README "The
// cut, and refusals"); and a denominator near zero whose mode the reference
// does not couple to is harmless, |t| stays small. On n2_ccpvdz_cas44_df,
// whose nearest Sr mode has y = 0 exactly: every virtual at -1000 Eh, every
// Sr denominator negative and Sr's energy then positive; and the virtuals
// shifted to put that mode at λ + Δ = -1e-6.
TEST(Nevpt2PcEnergy, NegativeAndUncoupledNearPoleAccepted) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const HostCase golden = load_host_case(kDataDir / "n2_ccpvdz_cas44_df");
  const auto base = pc_energies(res, golden);
  ASSERT_TRUE(base.has_value());
  const nevpt2::PcMode nearest = base->nearest[0];
  ASSERT_GT(nearest.denominator, 0.0);
  ASSERT_LT(std::abs(nearest.amplitude * nearest.denominator), 1e-12)
      << "the nearest Sr mode must be uncoupled for this case to test it";

  HostCase low = golden;
  std::ranges::fill(low.e_virt, -1000.0);
  const auto below = pc_energies(res, low);
  ASSERT_TRUE(below.has_value()) << below.error().message;
  EXPECT_LT(below->nearest[0].denominator, 0.0);
  EXPECT_LT(below->strongest[0].denominator, 0.0);
  EXPECT_GT(below->energies[0], 0.0);
  EXPECT_TRUE(std::isfinite(below->total));

  constexpr double kTarget = -1e-6;
  HostCase near = golden;
  for (double &e : near.e_virt) {
    e += kTarget - nearest.denominator;
  }
  const auto at_pole = pc_energies(res, near);
  ASSERT_TRUE(at_pole.has_value()) << at_pole.error().message;
  EXPECT_NEAR(at_pole->nearest[0].denominator, kTarget, 1e-12);
  EXPECT_LE(std::abs(at_pole->strongest[0].amplitude), 1.0);
  EXPECT_TRUE(std::isfinite(at_pole->total));
}

// @p count doubles at @p d, read back.
std::vector<double> download_host(fb::DeviceResources &res, const double *const d,
                                  const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// ---------------------------------------------------------------------------
// The SC-QD reverse pass
// ---------------------------------------------------------------------------

const std::filesystem::path kQdDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2";

// A QD case (data/qd_nevpt2/) on the host: what every pair shares in
// @ref shared, its RDMs empty, and each ordered pair's golden transition RDMs
// in @ref rdms[I n + J], only their RDM fields set and f3ca and f3ac on the
// diagonal alone. The same type holds a direction.
struct QdHostCase {
  std::size_t n = 0;
  HostCase shared;
  std::vector<HostCase> rdms;
};

QdHostCase load_qd_case(const std::filesystem::path &dir) {
  QdHostCase q;
  q.n = read_array(dir / "casci_energies.npy").data.size();
  q.shared = load_host_case(dir, false);
  q.rdms.resize(q.n * q.n);
  for (std::size_t i = 0; i < q.n; ++i) {
    for (std::size_t j = 0; j < q.n; ++j) {
      load_rdms(q.rdms[(i * q.n) + j],
                dir / ("pair_" + std::to_string(i) + "_" + std::to_string(j)), i == j);
    }
  }
  return q;
}

// Pair @p p's whole HostCase: the shared fields and its RDMs.
HostCase merged(const QdHostCase &q, const std::size_t p) {
  HostCase h = q.shared;
  h.dm1 = q.rdms[p].dm1;
  h.dm2 = q.rdms[p].dm2;
  h.dm3 = q.rdms[p].dm3;
  h.f3ca = q.rdms[p].f3ca;
  h.f3ac = q.rdms[p].f3ac;
  return h;
}

// @p x + @p t @p dx over every field of every pair.
QdHostCase displaced(const QdHostCase &x, const QdHostCase &dx, const double t) {
  QdHostCase out{.n = x.n, .shared = displaced(x.shared, dx.shared, t), .rdms = {}};
  for (std::size_t p = 0; p < x.rdms.size(); ++p) {
    out.rdms.push_back(displaced(x.rdms[p], dx.rdms[p], t));
  }
  return out;
}

// A seeded random direction over @p group's fields, random_direction's
// groups: "rdm" (and "all") moves every pair's RDMs, each entry a normal
// deviate scaled by kRdmScale of the largest magnitude of its ket's own array
// (a transition's own is far smaller where a symmetry nearly zeroes it), a
// state's own dm1, dm2 and dm3 reversal-symmetrized; a transition's are raw.
QdHostCase random_direction(const QdHostCase &x, const std::string &group,
                            const std::uint64_t seed) {
  QdHostCase d{.n = x.n, .shared = random_direction(x.shared, group, seed), .rdms = x.rdms};
  std::mt19937_64 engine(seed + 1);
  std::normal_distribution<double> normal;
  const bool on = group == "all" || group == "rdm";
  for (std::size_t p = 0; p < x.rdms.size(); ++p) {
    const HostCase &ket = x.rdms[((p % x.n) * x.n) + (p % x.n)];
    HostCase &dp = d.rdms[p];
    for (auto [y, from] :
         {std::pair{&dp.dm1, &ket.dm1}, std::pair{&dp.dm2, &ket.dm2}, std::pair{&dp.dm3, &ket.dm3},
          std::pair{&dp.f3ca, &ket.f3ca}, std::pair{&dp.f3ac, &ket.f3ac}}) {
      double scale = 0.0;
      for (const double v : *from) {
        scale = std::max(scale, std::abs(v));
      }
      for (double &v : *y) {
        v = on ? kRdmScale * scale * normal(engine) : 0.0;
      }
    }
    if (p / x.n == p % x.n) {
      for (std::vector<double> *rdm : {&dp.dm1, &dp.dm2, &dp.dm3}) {
        reversal_symmetrize(*rdm, x.shared.na);
      }
    }
  }
  return d;
}

// @p q on the device: the shared fields (state 0's RDMs riding along) and the
// n x n pair inputs, f3ca and f3ac null off the diagonal.
struct QdDeviceCase {
  DeviceCase shared;
  std::vector<fb::Buffers::device<double>> arrays;
  std::vector<nevpt2::EnergyInputs> pairs;
};

QdDeviceCase upload_qd(fb::DeviceResources &res, const QdHostCase &q) {
  QdDeviceCase g{.shared = upload_case(res, merged(q, 0)), .arrays = {}, .pairs = {}};
  const auto put = [&](const std::vector<double> &a) -> const double * {
    if (a.empty()) {
      return nullptr;
    }
    g.arrays.push_back(upload(res, a));
    return g.arrays.back().data();
  };
  for (const HostCase &r : q.rdms) {
    nevpt2::EnergyInputs in = g.shared.in;
    in.dm1 = put(r.dm1);
    in.dm2 = put(r.dm2);
    in.dm3 = put(r.dm3);
    in.f3ca = put(r.f3ca);
    in.f3ac = put(r.f3ac);
    g.pairs.push_back(in);
  }
  return g;
}

// L = Σ_IJ H̄_IJ H_IJ over sc_effective_hamiltonian at @p q.
double qd_objective(fb::DeviceResources &res, const QdHostCase &q,
                    const std::vector<double> &hbar) {
  QdDeviceCase g = upload_qd(res, q);
  const auto h = nevpt2::sc_effective_hamiltonian(res, *g.shared.blocks, g.pairs, 0);
  EXPECT_TRUE(h) << h.error().message;
  if (!h) {
    return std::nan("");
  }
  double sum = 0.0;
  for (std::size_t p = 0; p < h->size(); ++p) {
    sum += hbar[p] * (*h)[p].total;
  }
  return sum;
}

// A QD reverse pass's transition callback.
using Transition =
    std::function<fb::Status(std::int64_t, std::int64_t, const nevpt2::EnergyBars &)>;

// The bars of a QD reverse pass over @p q, every bar wanted, each zeroed
// first: per pair, [0] with the shared ones, the transitions' as handed over,
// each exactly once. @p pass runs it on the pairs, the n EnergyBars and the
// callback, and says whether it succeeded.
std::optional<std::vector<HostBars>>
qd_bars(fb::DeviceResources &res, const QdHostCase &q,
        const std::function<bool(const QdDeviceCase &, std::span<const nevpt2::EnergyBars>,
                                 const Transition &)> &pass) {
  QdDeviceCase g = upload_qd(res, q);
  const std::size_t n = q.n;
  const DeviceBars first = zero_bars(res, merged(q, 0));
  // States 1.., their own RDMs' bars alone, beside the first's shared ones.
  std::vector<DeviceBars> rest(n - 1);
  std::vector<nevpt2::EnergyBars> bars{first.bars};
  for (std::size_t j = 1; j < n; ++j) {
    const HostCase &own = q.rdms[(j * n) + j];
    DeviceBars &b = rest[j - 1];
    const auto zeros = [&](const std::size_t count) {
      b.arrays.push_back(upload(res, std::vector<double>(count, 0.0)));
      return b.arrays.back().data();
    };
    b.arrays.reserve(5);
    b.bars = first.bars;
    b.bars.dm1 = zeros(own.dm1.size());
    b.bars.dm2 = zeros(own.dm2.size());
    b.bars.dm3 = zeros(own.dm3.size());
    b.bars.f3ca = zeros(own.f3ca.size());
    b.bars.f3ac = zeros(own.f3ac.size());
    bars.push_back(b.bars);
  }
  std::vector<HostBars> host(n * n);
  std::vector<std::int32_t> handed(n * n, 0);
  const std::size_t na = q.shared.na;
  const Transition transition = [&](const std::int64_t bra, const std::int64_t ket,
                                    const nevpt2::EnergyBars &t) -> fb::Status {
    const auto p = static_cast<std::size_t>((bra * static_cast<std::int64_t>(n)) + ket);
    ++handed[p];
    EXPECT_EQ(t.f3ca, nullptr);
    EXPECT_EQ(t.lambda, nullptr);
    host[p].dm1 = download_host(res, t.dm1, na * na);
    host[p].dm2 = download_host(res, t.dm2, na * na * na * na);
    host[p].dm3 = download_host(res, t.dm3, na * na * na * na * na * na);
    return {};
  };
  if (!pass(g, bars, transition)) {
    return std::nullopt;
  }
  for (std::size_t p = 0; p < n * n; ++p) {
    EXPECT_EQ(handed[p], p / n == p % n ? 0 : 1) << "pair " << p;
  }
  host[0] = download_bars(res, first);
  for (std::size_t j = 1; j < n; ++j) {
    HostBars &h = host[(j * n) + j];
    const DeviceBars &b = rest[j - 1];
    h.dm1 = download_host(res, b.arrays[0]);
    h.dm2 = download_host(res, b.arrays[1]);
    h.dm3 = download_host(res, b.arrays[2]);
    h.f3ca = download_host(res, b.arrays[3]);
    h.f3ac = download_host(res, b.arrays[4]);
  }
  return host;
}

// sc_effective_hamiltonian_adjoint over @p q with H̄ @p hbar, as qd_bars.
std::optional<std::pair<nevpt2::ScEffectiveHamiltonianAdjoint, std::vector<HostBars>>>
run_qd_adjoint(fb::DeviceResources &res, const QdHostCase &q, const std::vector<double> &hbar,
               const std::int32_t batch) {
  std::optional<nevpt2::ScEffectiveHamiltonianAdjoint> adj;
  auto host = qd_bars(res, q,
                      [&](const QdDeviceCase &g, std::span<const nevpt2::EnergyBars> bars,
                          const Transition &transition) {
                        auto got = nevpt2::sc_effective_hamiltonian_adjoint(
                            res, *g.shared.blocks, g.pairs, hbar, batch, bars, transition);
                        EXPECT_TRUE(got) << got.error().message;
                        if (got) {
                          adj = std::move(*got);
                        }
                        return got.has_value();
                      });
  if (!host) {
    return std::nullopt;
  }
  return std::pair{std::move(*adj), std::move(*host)};
}

// Σ <bar, dx> over every field of every pair: the shared bars through pair
// (0, 0)'s, and each pair's RDMs'.
double bars_dot(const std::vector<HostBars> &bars, const QdHostCase &q, const QdHostCase &dq) {
  double sum = bars_dot(bars[0], merged(q, 0), merged(dq, 0));
  for (std::size_t p = 1; p < bars.size(); ++p) {
    sum += dot(bars[p].dm1, dq.rdms[p].dm1) + dot(bars[p].dm2, dq.rdms[p].dm2) +
           dot(bars[p].dm3, dq.rdms[p].dm3) + dot(bars[p].f3ca, dq.rdms[p].f3ca) +
           dot(bars[p].f3ac, dq.rdms[p].f3ac);
  }
  return sum;
}

// A seeded random H̄, n x n row-major: not symmetric, so the bars of (I, J)
// and (J, I) differ.
std::vector<double> random_hbar(const std::size_t n, const std::uint64_t seed) {
  std::mt19937_64 engine(seed);
  std::normal_distribution<double> normal;
  std::vector<double> hbar(n * n);
  for (double &x : hbar) {
    x = normal(engine);
  }
  return hbar;
}

// H₈, three states, roots 1 and 2 near-degenerate and every pair coupled, so
// every pair's bars are formed. The checks held to round-off run here: its
// bars repeat run to run to 3e-13 of their largest element (whole vs whole,
// n = 3, RTX 3080). Not every case's do: h2co_skew_ccpvdz_sa3's repeat to
// 9e-10 (h1e, h2e and state 2's RDMs; 1.3e-11 at n = 1) and
// h8_ring_sto3g_sa3's to 2.8e-8, already 1.9e-8 at n = 1, which is
// class_energies_adjoint's own arithmetic on that state.
const std::string kQdAdjointCase = "h8_sto3g_sa3";

// Eq. 40, Σ_IJ H̄_IJ H^k_IJ per class from the bars (each coupling is
// homogeneous of degree one in its transition norm and its ket's N and H),
// and Euler's identity, <Λ, B> + 2 Σ <h1e_v-bar, h1e_v> = 4 L: exact, so a
// mis-transcribed term or a wrong multiplier on either side of a pair fails
// them. And the elements are sc_effective_hamiltonian's.
class Nevpt2QdEnergyAdjointCases : public ::testing::TestWithParam<std::string> {};

TEST_P(Nevpt2QdEnergyAdjointCases, Eq40AndEulerFromTheBars) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const QdHostCase q = load_qd_case(kQdDataDir / GetParam());
  const std::vector<double> hbar = random_hbar(q.n, 698);
  const auto got = run_qd_adjoint(res, q, hbar, 0);
  ASSERT_TRUE(got.has_value());
  const auto &[adj, bars] = *got;
  QdDeviceCase g = upload_qd(res, q);
  const auto forward = nevpt2::sc_effective_hamiltonian(res, *g.shared.blocks, g.pairs, 0);
  ASSERT_TRUE(forward) << forward.error().message;
  ASSERT_EQ(adj.elements.size(), q.n * q.n);
  double objective = 0.0;
  std::array<double, 8> weighted{};
  for (std::size_t p = 0; p < q.n * q.n; ++p) {
    for (std::size_t k = 0; k < 8; ++k) {
      EXPECT_NEAR(adj.elements[p].energies[k], (*forward)[p].energies[k], 1e-12)
          << "element " << p << " " << kClassNames[k];
      EXPECT_NEAR(adj.elements[p].norms[k], (*forward)[p].norms[k], 1e-12)
          << "element " << p << " " << kClassNames[k];
      weighted[k] += hbar[p] * adj.elements[p].energies[k];
    }
    EXPECT_NEAR(adj.elements[p].total, (*forward)[p].total, 1e-12) << "element " << p;
    objective += hbar[p] * adj.elements[p].total;
  }
  double worst = 0.0;
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(adj.eq40[k], weighted[k], 1e-12 * std::max(1.0, std::abs(weighted[k])))
        << kClassNames[k];
    worst = std::max(worst, std::abs(adj.eq40[k] - weighted[k]));
  }
  const HostCase h = q.shared;
  const double euler = lambda_dot(bars[0].lambda, h, h) + 2.0 * dot(bars[0].h1e_v_sr, h.h1e_v_sr) +
                       2.0 * dot(bars[0].h1e_v_si, h.h1e_v_si) +
                       2.0 * dot(bars[0].h1e_v_sir, h.h1e_v_sir);
  EXPECT_NEAR(euler, 4.0 * objective, 1e-12 * std::abs(objective));
  std::cout << std::format("[ nevpt2   ] SC-QD reverse {}: worst Eq. 40 miss {:.2e}, Euler "
                           "{:+.2e} of L = {:+.6e}\n",
                           GetParam(), worst, euler - (4.0 * objective), objective);
}

// All six: the identities are exact wherever the energy is defined.
INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2QdEnergyAdjointCases,
                         ::testing::Values(std::string{"h2co_ccpvdz_sa2"},
                                           std::string{"h2co_ccpvdz_sa2_weighted"},
                                           std::string{"h2co_skew_ccpvdz_sa3"},
                                           std::string{"h6_631g_sa2"}, std::string{"h8_sto3g_sa3"},
                                           std::string{"h8_ring_sto3g_sa3"}),
                         [](const auto &info) { return info.param; });

// One state is class_energies_adjoint itself: its energies, Eq. 40 and every
// bar, to the atomics' round-off (kBatchedBarsTolerance's floor), at H̄ = 1.
TEST(Nevpt2QdEnergyAdjoint, OneStateIsClassEnergiesAdjoint) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const QdHostCase all = load_qd_case(kQdDataDir / kQdAdjointCase);
  const QdHostCase one{.n = 1, .shared = all.shared, .rdms = {all.rdms[0]}};
  const auto qd = run_qd_adjoint(res, one, {1.0}, 0);
  const auto sc = run_adjoint(res, merged(one, 0), 0);
  ASSERT_TRUE(qd.has_value());
  ASSERT_TRUE(sc.has_value());
  ASSERT_EQ(qd->first.elements.size(), 1U);
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(qd->first.elements[0].energies[k], sc->first.energies.energies[k], 1e-12)
        << kClassNames[k];
    EXPECT_NEAR(qd->first.elements[0].norms[k], sc->first.energies.norms[k], 1e-12)
        << kClassNames[k];
    EXPECT_NEAR(qd->first.eq40[k], sc->first.eq40[k], 1e-12) << kClassNames[k];
  }
  expect_same_bars(sc->second, qd->second[0], true, kBatchedBarsTolerance);
}

// Walked one slab row at a time (H₈ has two core and two virtual orbitals,
// so every walk splits), every bar, the transitions' handed over included,
// is the whole walk's to round-off.
TEST(Nevpt2QdEnergyAdjoint, BatchedBarsMatchWhole) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const QdHostCase q = load_qd_case(kQdDataDir / kQdAdjointCase);
  const std::vector<double> hbar = random_hbar(q.n, 698);
  const auto whole = run_qd_adjoint(res, q, hbar, 0);
  const auto batched = run_qd_adjoint(res, q, hbar, 1);
  ASSERT_TRUE(whole.has_value());
  ASSERT_TRUE(batched.has_value());
  expect_same_bars(whole->second[0], batched->second[0], true, kBatchedBarsTolerance);
  for (std::size_t p = 1; p < q.n * q.n; ++p) {
    const HostBars &a = whole->second[p];
    const HostBars &b = batched->second[p];
    const bool own = p / q.n == p % q.n;
    for (const auto &[name, x, y] :
         {std::tuple{"dm1", &a.dm1, &b.dm1}, std::tuple{"dm2", &a.dm2, &b.dm2},
          std::tuple{"dm3", &a.dm3, &b.dm3}, std::tuple{"f3ca", &a.f3ca, &b.f3ca},
          std::tuple{"f3ac", &a.f3ac, &b.f3ac}}) {
      ASSERT_EQ(x->size(), y->size()) << name << " pair " << p;
      if (!own && (std::string_view{name} == "f3ca" || std::string_view{name} == "f3ac")) {
        EXPECT_TRUE(x->empty()) << name << " pair " << p;
        continue;
      }
      double scale = 0.0;
      double worst = 0.0;
      for (std::size_t k = 0; k < x->size(); ++k) {
        scale = std::max(scale, std::abs((*x)[k]));
        worst = std::max(worst, std::abs((*x)[k] - (*y)[k]));
      }
      EXPECT_GT(scale, 0.0) << name << " pair " << p;
      EXPECT_LE(worst, kBatchedBarsTolerance * scale) << name << " pair " << p;
    }
  }
  for (std::size_t p = 0; p < q.n * q.n; ++p) {
    EXPECT_NEAR(batched->first.elements[p].total, whole->first.elements[p].total, 1e-12)
        << "element " << p;
  }
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(batched->first.eq40[k], whole->first.eq40[k], 1e-12) << kClassNames[k];
  }
}

// The bars against sc_effective_hamiltonian itself: L = Σ H̄_IJ H_IJ
// differenced along a seeded random direction of each group (B, h1e_v, the
// orbital energies, h1e and h2e, every pair's RDMs, and all of them), with
// SC's ratio test and Richardson extrapolant held to 1e-7 of Σ <bar, dx>.
//
// Every group on h8_sto3g_sa3; the orbital energies alone on the other five.
// There, along raw B, h or RDM directions, the SC energy is not smooth at the
// ladder's scale (1e-2 down to 2e-5): no run of ratios near 4, or a run that
// ends in a kink. That is SC's own, not the coupling's: one state alone,
// class_energies itself, does the same, e.g. h2co_skew_ccpvdz_sa3 along h
// (no ratio run, every state) and h8_ring_sto3g_sa3's state 0 along h (the
// extrapolant 100% off after four ratios at 4.00). A perturber whose norm is
// small (1e-12 here) has a Rayleigh quotient H/N that a raw direction moves
// by far more than its own size, toward the pole Δ + H/N = 0, and the
// coupling's numerator is the transition norm, not N. The orbital energies
// move Δ alone, and are smooth on every case.
class Nevpt2QdEnergyAdjointLadder
    : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(Nevpt2QdEnergyAdjointLadder, BarsMatchRichardsonLadder) {
  FBF_REQUIRE_GPU();
  const auto &[name, group] = GetParam();
  fb::DeviceResources res(0);
  const QdHostCase q = load_qd_case(kQdDataDir / name);
  const std::vector<double> hbar = random_hbar(q.n, 698);
  const QdHostCase dx = random_direction(q, group, 560);
  const auto got = run_qd_adjoint(res, q, hbar, 0);
  ASSERT_TRUE(got.has_value());
  const double predicted = bars_dot(got->second, q, dx);
  const Ladder l = richardson_ladder(
      [&](const double t) { return qd_objective(res, displaced(q, dx, t), hbar); }, 1e-2);
  ASSERT_GE(l.run, 3) << "ratios:" << l.ratios;
  EXPECT_NEAR(l.richardson, predicted, 1e-7 * std::abs(predicted)) << "ratios:" << l.ratios;
  std::cout << std::format("[ nevpt2   ] SC-QD reverse ladder {} {}: Σ<bar, dx> {:.12e}, "
                           "Richardson {:.12e}, relative {:.2e}, ratios{}\n",
                           name, group, predicted, l.richardson,
                           std::abs(l.richardson - predicted) / std::abs(predicted), l.ratios);
}

INSTANTIATE_TEST_SUITE_P(
    H8EveryGroup, Nevpt2QdEnergyAdjointLadder,
    ::testing::Combine(::testing::Values(std::string{"h8_sto3g_sa3"}),
                       ::testing::Values(std::string{"all"}, std::string{"B"}, std::string{"h1e_v"},
                                         std::string{"e"}, std::string{"h"}, std::string{"rdm"})),
    [](const auto &info) { return std::get<0>(info.param) + "_" + std::get<1>(info.param); });

INSTANTIATE_TEST_SUITE_P(
    OrbitalEnergies, Nevpt2QdEnergyAdjointLadder,
    ::testing::Combine(::testing::Values(std::string{"h2co_ccpvdz_sa2"},
                                         std::string{"h2co_ccpvdz_sa2_weighted"},
                                         std::string{"h2co_skew_ccpvdz_sa3"},
                                         std::string{"h6_631g_sa2"},
                                         std::string{"h8_ring_sto3g_sa3"}),
                       ::testing::Values(std::string{"e"})),
    [](const auto &info) { return std::get<0>(info.param) + "_" + std::get<1>(info.param); });

// An H̄ that is not n², a bars list that is not n long, or a state's bars that
// do not share the first's integral bars are InvalidConfig before anything is
// launched; malformed pairs are sc_effective_hamiltonian's refusals.
TEST(Nevpt2QdEnergyAdjoint, MalformedInputsAreInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const QdHostCase q = load_qd_case(kQdDataDir / "h6_631g_sa2");
  ASSERT_EQ(q.n, 2U);
  QdDeviceCase g = upload_qd(res, q);
  const DeviceBars b = zero_bars(res, merged(q, 0));
  const std::vector<nevpt2::EnergyBars> two{b.bars, b.bars};
  const std::vector<double> hbar(4, 1.0);
  const auto refused = [&](std::span<const nevpt2::EnergyInputs> pairs, std::span<const double> h,
                           std::span<const nevpt2::EnergyBars> bars) {
    const auto got =
        nevpt2::sc_effective_hamiltonian_adjoint(res, *g.shared.blocks, pairs, h, 0, bars, {});
    return !got && got.error().kind == fb::ErrorKind::InvalidConfig;
  };
  const std::span<const nevpt2::EnergyInputs> pairs{g.pairs};
  EXPECT_TRUE(refused(pairs.first(3), hbar, two));
  EXPECT_TRUE(refused(pairs, std::span{hbar}.first(3), two));
  EXPECT_TRUE(refused(pairs, hbar, std::span{two}.first(1)));
  std::vector<nevpt2::EnergyBars> unshared = two;
  unshared[1].lambda = nullptr;
  EXPECT_TRUE(refused(pairs, hbar, unshared));
}

// ---------------------------------------------------------------------------
// The PC-QD reverse pass
// ---------------------------------------------------------------------------

// pc_effective_hamiltonian_adjoint over @p q with H̄ @p hbar, as qd_bars.
std::optional<std::pair<nevpt2::PcEffectiveHamiltonianAdjoint, std::vector<HostBars>>>
run_pc_qd_adjoint(fb::DeviceResources &res, const QdHostCase &q, const std::vector<double> &hbar,
                  const std::int32_t batch) {
  std::optional<nevpt2::PcEffectiveHamiltonianAdjoint> adj;
  auto host = qd_bars(res, q,
                      [&](const QdDeviceCase &g, std::span<const nevpt2::EnergyBars> bars,
                          const Transition &transition) {
                        auto got = nevpt2::pc_effective_hamiltonian_adjoint(
                            res, *g.shared.blocks, g.pairs, hbar, batch, bars, transition);
                        EXPECT_TRUE(got) << got.error().message;
                        if (got) {
                          adj = std::move(*got);
                        }
                        return got.has_value();
                      });
  if (!host) {
    return std::nullopt;
  }
  return std::pair{std::move(*adj), std::move(*host)};
}

// pc_effective_hamiltonian at @p q.
fb::Result<std::vector<nevpt2::PcClassEnergies>> pc_qd_energies(fb::DeviceResources &res,
                                                                const QdHostCase &q) {
  QdDeviceCase g = upload_qd(res, q);
  return nevpt2::pc_effective_hamiltonian(res, *g.shared.blocks, g.pairs, 0);
}

// The homogeneity sums, Σ_IJ H̄_IJ E^k_IJ per class from the bars (each
// coupling is homogeneous of degree one in its ket's S, K and S₁ and its
// transition's S and S₁ together), and Euler's identity in (B, h1e_v²),
// <Λ, B> + 2 Σ <h1e_v-bar, h1e_v> = 4 L: exact, so a mis-transcribed term,
// a wrong multiplier on either row of a pair or a transition's bar on the
// wrong side of U fails them. And the elements are pc_effective_hamiltonian's.
class Nevpt2PcQdEnergyAdjointCases : public ::testing::TestWithParam<std::string> {};

TEST_P(Nevpt2PcQdEnergyAdjointCases, HomogeneityAndEulerFromTheBars) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const QdHostCase q = load_qd_case(kQdDataDir / GetParam());
  const std::vector<double> hbar = random_hbar(q.n, 703);
  const auto got = run_pc_qd_adjoint(res, q, hbar, 0);
  ASSERT_TRUE(got.has_value());
  const auto &[adj, bars] = *got;
  const auto forward = pc_qd_energies(res, q);
  ASSERT_TRUE(forward) << forward.error().message;
  ASSERT_EQ(adj.elements.size(), q.n * q.n);
  double objective = 0.0;
  std::array<double, 8> weighted{};
  for (std::size_t p = 0; p < q.n * q.n; ++p) {
    for (std::size_t k = 0; k < 8; ++k) {
      EXPECT_NEAR(adj.elements[p].energies[k], (*forward)[p].energies[k], 1e-12)
          << "element " << p << " " << kClassNames[k];
      EXPECT_EQ(adj.elements[p].dropped[k], (*forward)[p].dropped[k])
          << "element " << p << " " << kClassNames[k];
      weighted[k] += hbar[p] * adj.elements[p].energies[k];
    }
    EXPECT_NEAR(adj.elements[p].total, (*forward)[p].total, 1e-12) << "element " << p;
    objective += hbar[p] * adj.elements[p].total;
  }
  double worst = 0.0;
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(adj.homogeneity[k], weighted[k], 1e-12 * std::max(1.0, std::abs(weighted[k])))
        << kClassNames[k];
    worst = std::max(worst, std::abs(adj.homogeneity[k] - weighted[k]));
  }
  const HostCase h = q.shared;
  const double euler = lambda_dot(bars[0].lambda, h, h) + 2.0 * dot(bars[0].h1e_v_sr, h.h1e_v_sr) +
                       2.0 * dot(bars[0].h1e_v_si, h.h1e_v_si) +
                       2.0 * dot(bars[0].h1e_v_sir, h.h1e_v_sir);
  EXPECT_NEAR(euler, 4.0 * objective, 1e-12 * std::abs(objective));
  std::cout << std::format("[ nevpt2   ] PC-QD reverse {}: worst homogeneity miss {:.2e}, Euler "
                           "{:+.2e} of L = {:+.6e}\n",
                           GetParam(), worst, euler - (4.0 * objective), objective);
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2PcQdEnergyAdjointCases,
                         ::testing::Values(std::string{"h2co_ccpvdz_sa2"},
                                           std::string{"h2co_ccpvdz_sa2_weighted"},
                                           std::string{"h2co_skew_ccpvdz_sa3"},
                                           std::string{"h6_631g_sa2"}, std::string{"h8_sto3g_sa3"},
                                           std::string{"h8_ring_sto3g_sa3"}),
                         [](const auto &info) { return info.param; });

// H₈ on a ring, three states, every pair coupled (roots 0 and 2 by ~0.03 Eh),
// every root's every PC metric clearing the gap rule by >= 2e7: the checks
// held to round-off run here. Its bars repeat run to run (RTX 3080, ten runs,
// each bar against its own largest element) to 6.9e-13 for state 0 alone and
// 7.3e-14 for state 2, and the walk's (Λ, h1e_v's, the orbital energies') to
// 2.5e-13 at n = 3. State 1's do not: 2.2e-11 alone, its Sr keeping small
// modes, and so the sweep's bars of every pair it is in at n = 3 neither, as
// Nevpt2PcEnergyAdjoint.BatchedBarsMatchWhole has it at CAS(8,8). On
// h8_sto3g_sa3, whose root 0's PC passes only over round-off, its bars
// repeat to 1.4e-7 of their scale.
const std::string kPcQdAdjointCase = "h8_ring_sto3g_sa3";

// One state is pc_class_energies_adjoint itself (it is that call's n = 1, at
// H̄ = 1): its energies, dropped counts, homogeneity sums and every bar, the
// bars at kBatchedBarsTolerance, the standing rule (the largest run-to-run
// spread measured was 6.9e-13). States 0 and 2; state 1's spread is
// 2.2e-11 (above).
TEST(Nevpt2PcQdEnergyAdjoint, OneStateIsPcClassEnergiesAdjoint) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const QdHostCase all = load_qd_case(kQdDataDir / kPcQdAdjointCase);
  for (const std::size_t j : {std::size_t{0}, std::size_t{2}}) {
    const QdHostCase one{.n = 1, .shared = all.shared, .rdms = {all.rdms[(j * all.n) + j]}};
    const auto qd = run_pc_qd_adjoint(res, one, {1.0}, 0);
    const auto pc = run_pc_adjoint(res, merged(one, 0), 0);
    ASSERT_TRUE(qd.has_value());
    ASSERT_TRUE(pc.has_value());
    ASSERT_EQ(qd->first.elements.size(), 1U);
    for (std::size_t k = 0; k < 8; ++k) {
      EXPECT_NEAR(qd->first.elements[0].energies[k], pc->first.energies.energies[k], 1e-12)
          << "state " << j << " " << kClassNames[k];
      EXPECT_EQ(qd->first.elements[0].dropped[k], pc->first.energies.dropped[k])
          << "state " << j << " " << kClassNames[k];
      EXPECT_NEAR(qd->first.homogeneity[k], pc->first.homogeneity[k], 1e-12)
          << "state " << j << " " << kClassNames[k];
    }
    expect_same_bars(pc->second, qd->second[0], true, kBatchedBarsTolerance);
  }
}

// Walked one slab row at a time (one core and three virtual orbitals, so
// every walk with an external index splits), the walk's bars match the whole
// walk's at kBatchedBarsTolerance (the standing rule; 2.5e-13 measured), every
// element and the homogeneity sums at 1e-12, with the same dropped counts; each transition is handed over once. The sweep's bars, the
// transitions' among them, are not compared element by element: they carry
// state 1's spread (above), as PC's one-state BatchedBarsMatchWhole leaves
// them to its homogeneity sums.
TEST(Nevpt2PcQdEnergyAdjoint, BatchedBarsMatchWhole) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const QdHostCase q = load_qd_case(kQdDataDir / kPcQdAdjointCase);
  const std::vector<double> hbar = random_hbar(q.n, 703);
  const auto whole = run_pc_qd_adjoint(res, q, hbar, 0);
  const auto batched = run_pc_qd_adjoint(res, q, hbar, 1);
  ASSERT_TRUE(whole.has_value());
  ASSERT_TRUE(batched.has_value());
  expect_same_bars(whole->second[0], batched->second[0], false, kBatchedBarsTolerance);
  for (std::size_t p = 0; p < q.n * q.n; ++p) {
    for (std::size_t k = 0; k < 8; ++k) {
      EXPECT_NEAR(batched->first.elements[p].energies[k], whole->first.elements[p].energies[k],
                  1e-12)
          << "element " << p << " " << kClassNames[k];
      EXPECT_EQ(batched->first.elements[p].dropped[k], whole->first.elements[p].dropped[k])
          << "element " << p << " " << kClassNames[k];
    }
  }
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(batched->first.homogeneity[k], whole->first.homogeneity[k], 1e-12)
        << kClassNames[k];
  }
}

// L = Σ_IJ H̄_IJ H_IJ over pc_effective_hamiltonian at @p q.
double pc_qd_objective(fb::DeviceResources &res, const QdHostCase &q,
                       const std::vector<double> &hbar) {
  const auto h = pc_qd_energies(res, q);
  EXPECT_TRUE(h) << h.error().message;
  if (!h) {
    return std::nan("");
  }
  double sum = 0.0;
  for (std::size_t p = 0; p < h->size(); ++p) {
    sum += hbar[p] * (*h)[p].total;
  }
  return sum;
}

// The bars against pc_effective_hamiltonian itself: L = Σ H̄_IJ H_IJ
// differenced along each group of PC's ladder (Nevpt2PcEnergyAdjointLadder):
// its ratio test, its t0 = 1e-3 and SC's 1e-7. The groups are SC-QD's random
// directions, except that every state's own dm1-dm3 move along PC's active
// rotation, which keeps each S_J's spectrum; the transitions' dm1-dm3 move
// raw (they enter b_{I←J} alone, linearly, and no solve), and f3ca and f3ac
// raw. "h1e_v" is quadratic, so every difference is held to the bars.
//
// Every group on h2co_ccpvdz_sa2, h2co_ccpvdz_sa2_weighted and h6_631g_sa2;
// on the other three, B everywhere and e where it holds. Measured on the RTX
// 3080, what keeps the rest out: along h, the RDMs or all of them, a ket's PC
// is refused within the ladder's reach (K moved off annihilating S's
// near-null functions sends a kept mode's λ through -Δ, or the gap rule
// fails), and along h1e_v or e, L's run-to-run round-off reaches the finest
// rungs.
// - h2co_skew_ccpvdz_sa3: refused at t >= 2.5e-4 along h, 1e-3 along rdm
//   and 1.25e-4 along all; below, ratios 5.09 4.21 4.05 4.01 4.00 (h,
//   extrapolant 9.4e-9 off), 4.67 4.14 4.03 4.01 4.00 4.00 4.00 (rdm,
//   1.7e-9) and 4.62 4.14 4.03 4.01 (all, 1.9e-7, the run still settling).
//   Along h1e_v one run in six missed at its finest rung by 1.4e-7.
// - h8_sto3g_sa3 (its root 0's PC passes only over round-off): refused at
//   every rung along h and all, and at t >= 1.6e-5 along rdm. In five runs
//   h1e_v's finest differences missed by up to 1.9e-7 (two past 1e-7), and
//   e's ratio runs were three or four long, between 3.49 and 4.38.
// - h8_ring_sto3g_sa3: refused at t >= 2.5e-4 along h (then 5.97 4.37 4.09
//   4.02 4.01, 5.2e-9 off) and at t >= 6.3e-5 along rdm and all, state 1's
//   Sr denominator λ + Δ at -33 at t = 1e-3; then 31.9 5.15 4.24 (rdm) and
//   27.0 5.09 4.23 (all), a pole within ~1e-5 of the base point. Its h1e_v
//   share of L is 5.7e-6 against L = 0.16, so two fine rungs' differences
//   missed by 1.4e-7 and 5.7e-7.
class Nevpt2PcQdEnergyAdjointLadder
    : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(Nevpt2PcQdEnergyAdjointLadder, BarsMatchRichardsonLadder) {
  FBF_REQUIRE_GPU();
  const auto &[name, group] = GetParam();
  fb::DeviceResources res(0);
  const QdHostCase q = load_qd_case(kQdDataDir / name);
  const std::size_t n = q.n;
  const std::size_t na = q.shared.na;
  const std::vector<double> hbar = random_hbar(n, 703);
  QdHostCase straight = random_direction(q, group, 561);
  const bool rotate_rdms = group == "rdm" || group == "all";
  const std::vector<double> kappa = random_kappa(na, 562);
  const auto shape = [&](const std::size_t rank) { return std::vector<std::size_t>(rank, na); };
  const auto all_axes = [](const std::size_t rank) {
    std::vector<std::size_t> axes(rank);
    std::iota(axes.begin(), axes.end(), std::size_t{0});
    return axes;
  };
  constexpr std::array kRotated{std::pair{&HostCase::dm1, std::size_t{2}},
                                std::pair{&HostCase::dm2, std::size_t{4}},
                                std::pair{&HostCase::dm3, std::size_t{6}}};
  QdHostCase dx = straight;
  for (std::size_t j = 0; j < n; ++j) {
    const std::size_t jj = (j * n) + j;
    for (const auto [field, rank] : kRotated) {
      std::ranges::fill(straight.rdms[jj].*field, 0.0);
      dx.rdms[jj].*field =
          rotate_rdms ? rotation_tangent(q.rdms[jj].*field, shape(rank), all_axes(rank), kappa)
                      : std::vector<double>((q.rdms[jj].*field).size(), 0.0);
    }
  }
  const auto got = run_pc_qd_adjoint(res, q, hbar, 0);
  ASSERT_TRUE(got.has_value());
  const double predicted = bars_dot(got->second, q, dx);
  const auto energy = [&](const double t) {
    QdHostCase x = displaced(q, straight, t);
    if (rotate_rdms) {
      const std::vector<double> u = exp_kappa(kappa, na, t);
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t jj = (j * n) + j;
        for (const auto [field, rank] : kRotated) {
          x.rdms[jj].*field = rotated(q.rdms[jj].*field, shape(rank), all_axes(rank), u);
        }
      }
    }
    return pc_qd_objective(res, x, hbar);
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
  std::cout << std::format("[ nevpt2   ] PC-QD reverse ladder {} {}: Σ<bar, dx> {:.12e}, {} "
                           "{:.12e}, relative {:.2e}, ratios{}\n",
                           name, group, predicted, group == "h1e_v" ? "D(t0)" : "Richardson", got_d,
                           std::abs(got_d - predicted) / std::abs(predicted), l.ratios);
}

INSTANTIATE_TEST_SUITE_P(
    EveryGroup, Nevpt2PcQdEnergyAdjointLadder,
    ::testing::Combine(::testing::Values(std::string{"h2co_ccpvdz_sa2"},
                                         std::string{"h2co_ccpvdz_sa2_weighted"},
                                         std::string{"h6_631g_sa2"}),
                       ::testing::Values(std::string{"all"}, std::string{"B"}, std::string{"h1e_v"},
                                         std::string{"e"}, std::string{"h"}, std::string{"rdm"})),
    [](const auto &info) { return std::get<0>(info.param) + "_" + std::get<1>(info.param); });

INSTANTIATE_TEST_SUITE_P(
    SlabsAndOrbitalEnergies, Nevpt2PcQdEnergyAdjointLadder,
    ::testing::Values(std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"B"}},
                      std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"e"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"B"}},
                      std::tuple{std::string{"h8_ring_sto3g_sa3"}, std::string{"B"}},
                      std::tuple{std::string{"h8_ring_sto3g_sa3"}, std::string{"e"}}),
    [](const auto &info) { return std::get<0>(info.param) + "_" + std::get<1>(info.param); });

// A class the energy refuses is refused by the adjoint with the energy's
// error, naming the class and the state, never summed in as zero. On
// h8_sto3g_sa3: state 1's own RDMs zeroed (Sr's metric has no positive
// eigenvalue, state 1), and every virtual orbital energy shifted to make
// state 0's strongest Sr mode an intruder (|y / (λ + Δ)| = 4).
TEST(Nevpt2PcQdEnergyAdjoint, RefusesWhatTheEnergyRefuses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const QdHostCase golden = load_qd_case(kQdDataDir / "h8_sto3g_sa3");
  ASSERT_EQ(golden.n, 3U);
  QdHostCase zero_own = golden;
  HostCase &own1 = zero_own.rdms[golden.n + 1];
  for (std::vector<double> *rdm : {&own1.dm1, &own1.dm2, &own1.dm3, &own1.f3ca, &own1.f3ac}) {
    std::ranges::fill(*rdm, 0.0);
  }
  // State 0's Sr intruder, as RefusesWhatTheEnergyRefuses places it; state 0
  // is checked first, so its refusal is the one reported.
  const auto base = pc_qd_energies(res, golden);
  ASSERT_TRUE(base.has_value());
  const double shift = intruder_shift(base->front().strongest[0]);
  QdHostCase intruder = golden;
  for (double &e : intruder.shared.e_virt) {
    e += shift;
  }
  const std::vector<double> hbar = random_hbar(golden.n, 703);
  for (const auto &[what, q, needle] :
       {std::tuple{"state 1's RDMs zero", &zero_own, "state 1: Sr's metric has no positive"},
        std::tuple{"state 0's intruder in Sr", &intruder, "state 0: Sr has an intruder"}}) {
    const auto energy = pc_qd_energies(res, *q);
    ASSERT_FALSE(energy) << what;
    QdDeviceCase g = upload_qd(res, *q);
    const DeviceBars b = zero_bars(res, merged(*q, 0));
    const std::vector<nevpt2::EnergyBars> bars(q->n, b.bars);
    const auto adj =
        nevpt2::pc_effective_hamiltonian_adjoint(res, *g.shared.blocks, g.pairs, hbar, 0, bars, {});
    ASSERT_FALSE(adj) << what;
    EXPECT_EQ(adj.error().kind, fb::ErrorKind::Numerical) << what;
    EXPECT_EQ(adj.error().message, energy.error().message) << what;
    EXPECT_NE(adj.error().message.find(needle), std::string::npos)
        << what << ": " << adj.error().message;
  }
}

// The PC adjoint checks its inputs as SC's: an H̄ that is not n², a bars list
// that is not n long, a state's bars that do not share the first's integral
// bars, or malformed pairs are InvalidConfig before anything is launched.
TEST(Nevpt2PcQdEnergyAdjoint, MalformedInputsAreInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const QdHostCase q = load_qd_case(kQdDataDir / "h6_631g_sa2");
  ASSERT_EQ(q.n, 2U);
  QdDeviceCase g = upload_qd(res, q);
  const DeviceBars b = zero_bars(res, merged(q, 0));
  const std::vector<nevpt2::EnergyBars> two{b.bars, b.bars};
  const std::vector<double> hbar(4, 1.0);
  const auto refused = [&](std::span<const nevpt2::EnergyInputs> pairs, std::span<const double> h,
                           std::span<const nevpt2::EnergyBars> bars) {
    const auto got =
        nevpt2::pc_effective_hamiltonian_adjoint(res, *g.shared.blocks, pairs, h, 0, bars, {});
    return !got && got.error().kind == fb::ErrorKind::InvalidConfig;
  };
  const std::span<const nevpt2::EnergyInputs> pairs{g.pairs};
  EXPECT_TRUE(refused(pairs.first(3), hbar, two));
  EXPECT_TRUE(refused(pairs, std::span{hbar}.first(3), two));
  EXPECT_TRUE(refused(pairs, hbar, std::span{two}.first(1)));
  std::vector<nevpt2::EnergyBars> unshared = two;
  unshared[1].lambda = nullptr;
  EXPECT_TRUE(refused(pairs, hbar, unshared));
}

// ---------------------------------------------------------------------------
// The PC forward tangent
// ---------------------------------------------------------------------------

// The cut's round-off scale for a basis of size @p d, ε √d: S's null modes sit
// below 7.4 times it (pc.cpp's kPcNoise note), real dropped modes above.
double round_off(const std::size_t d) {
  return std::numeric_limits<double>::epsilon() * std::sqrt(static_cast<double>(d));
}

// The metric basis sizes, in PySCF's class order: Sr, Si n³, Sijr and Srsi n,
// Srs and Sij n², Sir 2 n²; Sijrs has no metric.
std::array<std::size_t, 8> metric_sizes(const std::size_t n) {
  return {n * n * n, n * n * n, 0, n, n, n * n, n * n, 2 * n * n};
}

// The duality directions: each input group alone, then all of them
// (random_direction's, seeds 810 on, an all-zero field passed as null).
const std::array<std::string, 6> kPcTangentGroups{"B", "h1e_v", "e", "h", "rdm", "all"};

// pc_class_energies_tangent over @p h along @p groups' directions, walked in
// batches of @p batch, with the kept span's rotation when @p span, against
// the bars @p bars: per direction Ė, Σ <bar, dx> and the miss relative to
// that sum's largest summand. The energies are checked against @p energies.
struct PcDuality {
  nevpt2::PcClassEnergiesTangent tangent;
  std::vector<double> dot, predicted, miss;
};

std::optional<PcDuality> pc_duality(fb::DeviceResources &res, const HostCase &h,
                                    const HostBars &bars, const nevpt2::PcClassEnergies &energies,
                                    const std::int32_t batch, const bool span,
                                    const std::vector<std::string> &groups,
                                    const std::string &what) {
  std::vector<HostCase> dx;
  for (const std::string &group : groups) {
    const auto at = std::ranges::find(kPcTangentGroups, group) - kPcTangentGroups.begin();
    dx.push_back(random_direction(h, group, 810 + static_cast<std::uint64_t>(at)));
  }
  DeviceCase g = upload_case(res, h);
  std::vector<DeviceCase> gd;
  gd.reserve(dx.size());
  std::vector<nevpt2::EnergyDirection> directions;
  for (const HostCase &d : dx) {
    gd.push_back(upload_case(res, d));
    directions.push_back(direction_of(gd.back(), d));
  }
  auto got = nevpt2::pc_class_energies_tangent(res, *g.blocks, g.in, batch, directions, span);
  EXPECT_TRUE(got) << what << ": " << got.error().message;
  if (!got) {
    return std::nullopt;
  }
  EXPECT_EQ(got->total_dot.size(), groups.size()) << what;
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_NEAR(got->energies.energies[k], energies.energies[k], 1e-10)
        << what << " " << kClassNames[k];
    EXPECT_EQ(got->energies.dropped[k], energies.dropped[k]) << what << " " << kClassNames[k];
  }
  PcDuality out{.tangent = std::move(*got), .dot = {}, .predicted = {}, .miss = {}};
  for (std::size_t u = 0; u < groups.size() && u < out.tangent.total_dot.size(); ++u) {
    const std::vector<double> summands = bars_summands(bars, h, dx[u]);
    const double predicted = std::accumulate(summands.begin(), summands.end(), 0.0);
    double scale = 0.0;
    for (const double x : summands) {
      scale = std::max(scale, std::abs(x));
    }
    const double dot = out.tangent.total_dot[u];
    out.dot.push_back(dot);
    out.predicted.push_back(predicted);
    out.miss.push_back(std::abs(dot - predicted) / scale);
    std::cout << std::format("[ nevpt2   ] PC tangent {} batch {}{} {}: Ė {:+.12e}, Σ<bar, dx> "
                             "{:+.12e}, miss / largest summand {:.2e}\n",
                             what, batch, span ? "" : " (span off)", groups[u], dot, predicted,
                             out.miss.back());
  }
  return out;
}

// Each class's dropped count, its largest cut σ and its smallest kept σ, over
// S's largest, for the log.
std::string pc_cuts(const nevpt2::PcClassEnergiesTangent &t) {
  std::string out;
  for (std::size_t k = 0; k < 8; ++k) {
    if (t.energies.dropped[k] > 0) {
      out += std::format(" {} {} cut {:.2e} kept {:.2e}", kClassNames[k], t.energies.dropped[k],
                         t.energies.cut[k], t.energies.kept[k]);
    }
  }
  return out;
}

// The issue's duality bar: Ė against Σ <bar, dx> to 1e-10 of that sum's
// largest summand.
constexpr double kPcDuality = 1e-10;

// The ε √d multiple below which a dropped mode is round-off's: twice the
// largest null mode measured (7.4, pc.cpp's kPcNoise note), far below the
// cut's own 100.
constexpr double kNullModes = 15.0;

// On the PC goldens every class's cut drops exact null modes alone, which K
// annihilates at an eigenstate: the fixed-span terms carry Ė. Walked whole,
// and in threes (a partial last slab).
class Nevpt2PcEnergyTangentNullModes : public ::testing::TestWithParam<std::int32_t> {};

TEST_P(Nevpt2PcEnergyTangentNullModes, DualToTheBars) {
  FBF_REQUIRE_GPU();
  const std::int32_t batch = GetParam();
  const std::string name = "n2_ccpvdz_cas44_df";
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(kDataDir / name);
  const auto adj = run_pc_adjoint(res, h, 0);
  ASSERT_TRUE(adj.has_value());
  const std::vector<std::string> groups(kPcTangentGroups.begin(), kPcTangentGroups.end());
  const auto got = pc_duality(res, h, adj->second, adj->first.energies, batch, true, groups, name);
  ASSERT_TRUE(got.has_value());
  for (std::size_t u = 0; u < got->miss.size(); ++u) {
    EXPECT_LE(got->miss[u], kPcDuality) << groups[u];
  }
  const std::array<std::size_t, 8> sizes = metric_sizes(h.na);
  std::int64_t dropped = 0;
  for (std::size_t k = 0; k < 8; ++k) {
    dropped += got->tangent.energies.dropped[k];
    if (got->tangent.energies.dropped[k] > 0) {
      EXPECT_LT(got->tangent.energies.cut[k], kNullModes * round_off(sizes[k])) << kClassNames[k];
    }
  }
  EXPECT_GT(dropped, 0);
  std::cout << std::format("[ nevpt2   ] PC tangent {} batch {}: worst {:.2e};{}\n", name, batch,
                           std::ranges::max(got->miss), pc_cuts(got->tangent));
}

INSTANTIATE_TEST_SUITE_P(Batches, Nevpt2PcEnergyTangentNullModes, ::testing::Values(0, 3),
                         [](const auto &info) { return "batch" + std::to_string(info.param); });

// A cut that drops real modes, with a gapped kept spectrum, from
// data/qd_nevpt2/'s h2co_skew_ccpvdz_sa3: the RDMs of the superposition
// |0⟩ + δ |1⟩ from its golden own and transition RDMs, dm_00 + δ (dm_01 +
// dm_10) + δ² dm_11 (dm1-dm3; f3ca and f3ac, given on the diagonal alone, as
// f3_00 + δ² f3_11), times 1 - kDeficit δ², δ = kSuperposition. Two things
// follow.
// - The trace deficit, (kDeficit - 1) δ² against ⟨ψ|ψ⟩ = 1 + δ², lifts the
//   hole metric's (Sij's hdm2) fully occupied null modes, whose δδ term no
//   longer cancels the RDMs': real eigenvalues of order 1e-14 to 1e-13 of
//   σ_max, below the cut. (Normalized, they stay at round-off; at a deficit
//   of 2 δ² some rise past the cut and sit there as kept modes at 2.7e-13,
//   PSB3's conditioning.)
// - ψ is not an eigenstate, so K no longer annihilates the dropped modes:
//   its part along them is of order δ, and the kept span's rotation
//   carries Ė at about 1e-5 of it, far above the bar. A pure state's own
//   null modes do not rise (they are the operators' identities: O|ψ⟩ spans
//   what O|0⟩ does), and an ensemble's lifted modes would carry K's part at
//   order σ_n, the rotation then below the bar.
// Every other class keeps its spectrum gapped (smallest kept σ 1.6e-4, Sir).
// PSB3's real modes instead run through the cut, its kept ones at 3e-13
// (SpanTermVisibleAtPsb3Twist).
constexpr double kSuperposition = 3e-7;
constexpr double kDeficit = 1.25;

HostCase superposition_case() {
  const QdHostCase q = load_qd_case(kQdDataDir / "h2co_skew_ccpvdz_sa3");
  HostCase h = merged(q, 0);
  const double d = kSuperposition;
  const std::array<std::pair<const HostCase *, double>, 3> terms{
      std::pair{&q.rdms[1], d}, std::pair{&q.rdms[q.n], d}, std::pair{&q.rdms[q.n + 1], d * d}};
  for (const auto &[x, w] : terms) {
    for (auto [y, a] :
         {std::pair{&h.dm1, &x->dm1}, std::pair{&h.dm2, &x->dm2}, std::pair{&h.dm3, &x->dm3},
          std::pair{&h.f3ca, &x->f3ca}, std::pair{&h.f3ac, &x->f3ac}}) {
      for (std::size_t k = 0; k < a->size(); ++k) {
        (*y)[k] += w * (*a)[k];
      }
    }
  }
  for (std::vector<double> *y : {&h.dm1, &h.dm2, &h.dm3, &h.f3ca, &h.f3ac}) {
    for (double &v : *y) {
      v *= 1.0 - (kDeficit * d * d);
    }
  }
  return h;
}

// The gate: duality at the issue's bar on the superposition, every direction; and
// the kept span's rotation visible, the tangent without it missing by more
// than ten bars along some direction. The fixture's point is checked too:
// some class cuts a mode above round-off, and every cutting class keeps its
// smallest mode far from the cut.
TEST(Nevpt2PcEnergyTangent, DualToTheBarsWithRealModesDropped) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const HostCase h = superposition_case();
  const auto adj = run_pc_adjoint(res, h, 0);
  ASSERT_TRUE(adj.has_value());
  const std::vector<std::string> groups(kPcTangentGroups.begin(), kPcTangentGroups.end());
  const std::string what = "h2co_skew_ccpvdz_sa3 superposition";
  const auto with = pc_duality(res, h, adj->second, adj->first.energies, 0, true, groups, what);
  const auto without = pc_duality(res, h, adj->second, adj->first.energies, 0, false, groups, what);
  ASSERT_TRUE(with.has_value());
  ASSERT_TRUE(without.has_value());
  for (std::size_t u = 0; u < with->miss.size(); ++u) {
    EXPECT_LE(with->miss[u], kPcDuality) << groups[u];
  }
  EXPECT_GT(std::ranges::max(without->miss), 10.0 * kPcDuality)
      << "the kept span's rotation is not visible";
  const std::array<std::size_t, 8> sizes = metric_sizes(h.na);
  double real = 0.0;
  double kept = 1.0;
  for (std::size_t k = 0; k < 8; ++k) {
    if (with->tangent.energies.dropped[k] > 0) {
      real = std::max(real, with->tangent.energies.cut[k] / round_off(sizes[k]));
      kept = std::min(kept, with->tangent.energies.kept[k]);
    }
  }
  EXPECT_GT(real, kNullModes) << "no class's cut drops a real mode";
  EXPECT_GT(kept, 1e-6) << "a kept mode sits near the cut";
  std::cout << std::format("[ nevpt2   ] PC tangent {}: worst {:.2e}, span off {:.2e}; largest "
                           "cut {:.1f} ε √d, smallest kept {:.2e};{}\n",
                           what, std::ranges::max(with->miss), std::ranges::max(without->miss),
                           real, kept, pc_cuts(with->tangent));
}

// The conditioning floor (EnergyInputs::pc_floor, #917) on a golden whose
// metrics are gapped: at kFloor it cuts real modes the default keeps, and the
// energy, its reverse pass and its tangent all read it, the same dropped
// counts and energies, each class's kept mode above the floor and its cut
// below, and the tangent dual to the bars at the issue's bar, the kept span's
// rotation (#862) carrying the real dropped modes as it does at the default.
constexpr double kFloor = 1e-3;

TEST(Nevpt2PcEnergyTangent, ConditioningFloorReadByEveryPass) {
  FBF_REQUIRE_GPU();
  const std::string name = "n2_ccpvdz_cas44_df";
  fb::DeviceResources res(0);
  const HostCase h = load_host_case(kDataDir / name);
  const auto base = pc_energies(res, h);
  ASSERT_TRUE(base) << base.error().message;
  DeviceCase g = upload_case(res, h);
  g.in.pc_floor = kFloor;
  const auto floored = nevpt2::pc_class_energies(res, *g.blocks, g.in, 0);
  ASSERT_TRUE(floored) << floored.error().message;
  const DeviceBars b = zero_bars(res, h);
  const auto adj = nevpt2::pc_class_energies_adjoint(res, *g.blocks, g.in, 0, b.bars);
  ASSERT_TRUE(adj) << adj.error().message;
  const HostBars bars = download_bars(res, b);
  const HostCase dx = random_direction(h, "all", 917);
  const DeviceCase gd = upload_case(res, dx);
  const std::array<nevpt2::EnergyDirection, 1> directions{direction_of(gd, dx)};
  const auto tangent = nevpt2::pc_class_energies_tangent(res, *g.blocks, g.in, 0, directions);
  ASSERT_TRUE(tangent) << tangent.error().message;

  std::int64_t more = 0;
  for (std::size_t k = 0; k < 8; ++k) {
    const std::string cls = kClassNames[k];
    EXPECT_GE(floored->dropped[k], base->dropped[k]) << cls;
    more += floored->dropped[k] - base->dropped[k];
    for (const nevpt2::PcClassEnergies *e : {&adj->energies, &tangent->energies}) {
      EXPECT_EQ(e->dropped[k], floored->dropped[k]) << cls;
      EXPECT_NEAR(e->energies[k], floored->energies[k], 1e-10) << cls;
      EXPECT_EQ(e->kept[k], floored->kept[k]) << cls;
    }
    if (k != 2) {
      // Every class but Sijrs solves a metric: its kept modes clear the floor.
      EXPECT_GT(floored->kept[k], kFloor) << cls;
      EXPECT_LE(floored->cut[k], kFloor) << cls;
      EXPECT_LE(base->kept[k], floored->kept[k]) << cls;
    }
  }
  EXPECT_GT(more, 0) << "the floor cut nothing the default keeps";
  EXPECT_GT(std::abs(floored->total - base->total), 1e-9) << "the floor moved no energy";
  const std::vector<double> summands = bars_summands(bars, h, dx);
  const double predicted = std::accumulate(summands.begin(), summands.end(), 0.0);
  double scale = 0.0;
  for (const double x : summands) {
    scale = std::max(scale, std::abs(x));
  }
  const double miss = std::abs(tangent->total_dot.front() - predicted) / scale;
  EXPECT_LE(miss, kPcDuality);
  std::cout << std::format("[ nevpt2   ] PC floor {:.0e} {}: {} more modes cut, E {:+.12f} "
                           "(default {:+.12f}); Ė against Σ<bar, dx> {:.2e};{}\n",
                           kFloor, name, more, floored->total, base->total, miss,
                           pc_cuts(*tangent));
}

// PSB3 twisted 90° about C13=C14 (nevpt2_psb3_gpu.cpp's kPsb3Twisted,
// PcQdActiveActiveGateAtTheTwist's DF-SA3-CASSCF(6,6)/STO-3G over
// cc-pVDZ-JKFIT), state 0's single-state class inputs at those orbitals,
// built by nevpt2::builders_tangent's value. Sr's and Si's cuts drop real
// modes there (#862).
constexpr std::string_view kPsb3Twisted = R"(14
PSB3, C13=C14 twisted 90 degrees
N     -0.0454402492     0.0522740911    -0.0000000000
C      1.1276796768     0.5920624626     0.0000000000
C      2.3467584180    -0.1268535537     0.0000000000
C      3.5352478218     0.5543991513     0.0000000000
C      4.2381819494     0.9573272501    -1.1898886726
C      5.4137881019     1.6311951607    -1.1318525401
H      1.1448444823     1.6750816790    -0.0000000000
H      2.3229137518    -1.2094227332    -0.0000000000
H      3.9874420661     0.8136009171     0.9516444335
H      3.7996110498     0.7059345067    -2.1474371187
H     -0.1682915976    -0.9466686957    -0.0000000000
H     -0.8778598767     0.6155991196    -0.0000000000
H      5.8725712624     1.8941737519    -0.1873080265
H      5.9322594717     1.9283875648    -2.0319280688
)";

// The twist's state-0 inputs, on the host, once per process.
const HostCase &psb3_twist() {
  static const HostCase kCase = [] {
    const fb::test::ScratchDir dir("nevpt2_pc_tangent_psb3");
    EXPECT_TRUE(fb::write_file(dir / "psb3.xyz", std::string{kPsb3Twisted}).has_value());
    const std::string toml =
        "[molecule]\nxyz = \"psb3.xyz\"\ncharge = 1\n[basis_set]\nbasis = \"sto-3g\"\n"
        "[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n"
        "[scf]\nmax_iterations = 400\n"
        "[casscf]\nncas = 6\nnelecas = 6\nnroots = 3\nconv_tol_grad = 1e-10\n";
    EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
    auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
    EXPECT_TRUE(calc.has_value()) << calc.error().message;
    if (!calc.has_value()) {
      return HostCase{};
    }
    fb::NodeResources node;
    const fb::Status ran = calc->run(node);
    EXPECT_TRUE(ran.has_value()) << ran.error().message;
    if (!ran.has_value()) {
      return HostCase{};
    }
    fb::DeviceResources &res = node.device(0);
    const auto &casscf = calc->get<fb::CasscfOutput>();
    const auto &eri = calc->get<fb::EriOutput>();
    const auto &core = calc->get<fb::CoreHamiltonianOutput>();
    if (!casscf.has_value() || !eri.has_value() || !core.has_value()) {
      ADD_FAILURE() << "the run left no CASSCF, ERI or core Hamiltonian";
      return HostCase{};
    }
    const auto &df = std::get<fb::DensityFitted>(eri->eri);
    const auto nao = static_cast<std::int32_t>(casscf->c.n());
    auto strings = fb::ci::DeviceStringSet::create(res, casscf->ncas, casscf->nelecas / 2);
    EXPECT_TRUE(strings.has_value()) << strings.error().message;
    if (!strings.has_value()) {
      return HostCase{};
    }
    // State 0's CI vector, the first of the three.
    const std::size_t ndet = casscf->vectors.size() / casscf->energies.size();
    const fb::Buffers::device<double> vector = upload(
        res, std::vector<double>(casscf->vectors.begin(),
                                 casscf->vectors.begin() + static_cast<std::ptrdiff_t>(ndet)));
    const auto built = nevpt2::builders_tangent(
        res, *strings, *strings, df.naux, nao, nao, df.b.data(), casscf->c.data(), core->h.data(),
        casscf->ncore, casscf->ncas, vector.data(), 0, nullptr, nullptr, nullptr);
    EXPECT_TRUE(built.has_value()) << built.error().message;
    if (!built.has_value()) {
      return HostCase{};
    }
    const nevpt2::BuilderOutputs &v = built->value;
    HostCase h;
    h.naux = static_cast<std::size_t>(v.blocks.naux);
    h.nc = static_cast<std::size_t>(v.blocks.ncore);
    h.na = static_cast<std::size_t>(v.blocks.ncas);
    h.nv = static_cast<std::size_t>(v.blocks.nvirt);
    const auto all = [&](const fb::Buffers::device<double> &d) {
      return download_host(res, d.data(), d.num_elements());
    };
    h.aa = all(v.blocks.aa);
    h.ca = all(v.blocks.ca);
    h.va = all(v.blocks.va);
    h.cv = all(v.blocks.cv);
    h.h1e = all(v.h1e);
    h.h2e = all(v.h2e);
    h.e_core = all(v.e_core);
    h.e_virt = all(v.e_virt);
    h.h1e_v_sr = all(v.h1e_v_sr);
    h.h1e_v_si = all(v.h1e_v_si);
    h.h1e_v_sir = all(v.h1e_v_sir);
    h.dm1 = all(v.dm1);
    h.dm2 = all(v.dm2);
    h.dm3 = all(v.dm3);
    h.f3ca = all(v.f3ca);
    h.f3ac = all(v.f3ac);
    return h;
  }();
  return kCase;
}

// PSB3's twist cannot carry the duality gate: its Sr and Si keep modes at
// about 4e-13 of σ_max, just above the cut, so the solve amplifies the
// einsums' atomic round-off in S and K by about 1e12, and along any direction
// that moves K both PC's bars and its tangent repeat run to run only to a few
// 1e-6 relative (#917; RTX 3080, since #916's warp pre-reduction, 2e-6 to
// 7e-6, against 1.4e-5 to 2.4e-5 before). So it checks visibility alone,
// statistically: over kSpanRepeats reverse passes and as many tangents, each
// with its sample deviation σ_a and σ_t along each direction, the two means
// must agree within kSpanSigmas √(σ_t² + σ_a²), and the tangent without the
// kept span's rotation must miss the bars' mean by more than 100 times that
// along the directions that move S (h moves K alone, so U̇ is zero there).
// The bound is each sample's own spread, not the means' (√kSpanRepeats
// tighter): the two are one bilinear form evaluated two ways, dual only to
// the solve's conditioning, so a residue of that order is not a failure.
// Five of a nine-degree-of-freedom deviation is far in the tail (Student's
// t, two-sided, about 7e-4 for the means alone); the earlier 3σ over five
// passes, with the tangent a single sample, failed 2 runs in 3 once #916 had
// shrunk σ_a and not σ_t.
constexpr std::int32_t kSpanRepeats = 10;
constexpr double kSpanSigmas = 5.0;

TEST(Nevpt2PcEnergyTangent, SpanTermVisibleAtPsb3Twist) {
  FBF_REQUIRE_GPU();
  // A check, not the gate (DualToTheBarsWithRealModesDropped is, and the
  // family's fast case): a CASSCF, ten reverse passes and eleven tangents,
  // about 7 s on the RTX 3080, its CASSCF 14 s on the gfx1200
  // (nevpt2_psb3_gpu.cpp).
  FBF_SKIP_IF_FAST_TESTS();
  const HostCase &h = psb3_twist();
  ASSERT_GT(h.na, 0U);
  fb::DeviceResources res(0);
  const std::vector<std::string> groups{"h", "rdm", "all"};
  std::vector<HostCase> dx;
  for (const std::string &group : groups) {
    const auto at = std::ranges::find(kPcTangentGroups, group) - kPcTangentGroups.begin();
    dx.push_back(random_direction(h, group, 810 + static_cast<std::uint64_t>(at)));
  }
  DeviceCase g = upload_case(res, h);
  std::vector<DeviceCase> gd;
  gd.reserve(dx.size());
  std::vector<nevpt2::EnergyDirection> directions;
  for (const HostCase &d : dx) {
    gd.push_back(upload_case(res, d));
    directions.push_back(direction_of(gd.back(), d));
  }
  // Each sample's mean and deviation.
  const auto stats = [](const std::vector<double> &x) {
    const double n = static_cast<double>(x.size());
    const double mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double var = 0.0;
    for (const double v : x) {
      var += (v - mean) * (v - mean);
    }
    return std::pair{mean, std::sqrt(var / (n - 1.0))};
  };
  std::vector<std::vector<double>> sums(groups.size());
  std::vector<std::vector<double>> dots(groups.size());
  std::optional<nevpt2::PcClassEnergiesTangent> last;
  for (std::int32_t rep = 0; rep < kSpanRepeats; ++rep) {
    const auto adj = run_pc_adjoint(res, h, 0);
    ASSERT_TRUE(adj.has_value());
    auto t = nevpt2::pc_class_energies_tangent(res, *g.blocks, g.in, 0, directions, true);
    ASSERT_TRUE(t) << t.error().message;
    ASSERT_EQ(t->total_dot.size(), groups.size());
    for (std::size_t u = 0; u < groups.size(); ++u) {
      const std::vector<double> summands = bars_summands(adj->second, h, dx[u]);
      sums[u].push_back(std::accumulate(summands.begin(), summands.end(), 0.0));
      dots[u].push_back(t->total_dot[u]);
    }
    last = std::move(*t);
  }
  const auto without =
      nevpt2::pc_class_energies_tangent(res, *g.blocks, g.in, 0, directions, false);
  ASSERT_TRUE(without) << without.error().message;
  const std::string what = "psb3_twist_sto3g";
  for (std::size_t u = 0; u < groups.size(); ++u) {
    const auto [mean_a, sigma_a] = stats(sums[u]);
    const auto [mean_t, sigma_t] = stats(dots[u]);
    const double spread = std::hypot(sigma_a, sigma_t);
    const double on = std::abs(mean_t - mean_a);
    const double off = std::abs(without->total_dot[u] - mean_a);
    EXPECT_LE(on, kSpanSigmas * spread)
        << groups[u] << ": Ė mean " << mean_t << ", Σ<bar, dx> mean " << mean_a;
    if (groups[u] != "h") {
      EXPECT_GT(off, 100.0 * kSpanSigmas * spread)
          << groups[u] << ": span off Ė " << without->total_dot[u];
    }
    std::cout << std::format("[ nevpt2   ] PC tangent {} {}: Σ<bar, dx> mean {:+.10e} σ_a {:.2e} "
                             "({:.1e} relative), Ė σ_t {:.2e}; with the span term {:.2f} "
                             "spreads off, without {:.1f}\n",
                             what, groups[u], mean_a, sigma_a, sigma_a / std::abs(mean_a), sigma_t,
                             on / spread, off / spread);
  }
  std::cout << std::format("[ nevpt2   ] PC tangent {}:{}\n", what, pc_cuts(*last));
}

// A class the energy refuses, the tangent refuses with the energy's error,
// never summing it as zero: each of the energy's refusals (README "The cut,
// and refusals"). Zero RDMs, whose Sr metric has no positive eigenvalue; an
// intruder, Sr's strongest mode at |t| = 4 (RefusesWhatTheEnergyRefuses'
// shift); and a denominator at round-off, every virtual shifted to put Sr's
// nearest mode at λ + Δ = 0.
TEST(Nevpt2PcEnergyTangent, RefusesWhatTheEnergyRefuses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const HostCase golden = load_host_case(kDataDir / kAdjointCase);
  HostCase zero_rdms = golden;
  for (std::vector<double> *rdm :
       {&zero_rdms.dm1, &zero_rdms.dm2, &zero_rdms.dm3, &zero_rdms.f3ca, &zero_rdms.f3ac}) {
    std::ranges::fill(*rdm, 0.0);
  }
  const auto base = pc_energies(res, golden);
  ASSERT_TRUE(base.has_value());
  HostCase intruder = golden;
  for (double &e : intruder.e_virt) {
    e += intruder_shift(base->strongest[0]);
  }
  HostCase pole = golden;
  for (double &e : pole.e_virt) {
    e -= base->nearest[0].denominator;
  }
  for (const auto &[what, h, needle] :
       {std::tuple{"zero RDMs", &zero_rdms, "no positive eigenvalue"},
        std::tuple{"an intruder in Sr", &intruder, "has an intruder"},
        std::tuple{"a pole in Sr", &pole, "at round-off"}}) {
    const auto energy = pc_energies(res, *h);
    ASSERT_FALSE(energy) << what;
    DeviceCase g = upload_case(res, *h);
    const HostCase dx = random_direction(*h, "all", 8100);
    DeviceCase gd = upload_case(res, dx);
    const std::array directions{direction_of(gd, dx)};
    const auto got = nevpt2::pc_class_energies_tangent(res, *g.blocks, g.in, 0, directions);
    ASSERT_FALSE(got) << what;
    EXPECT_EQ(got.error().kind, fb::ErrorKind::Numerical) << what;
    EXPECT_EQ(got.error().kind, energy.error().kind) << what;
    EXPECT_EQ(stable_part(got.error().message), stable_part(energy.error().message)) << what;
    EXPECT_NE(got.error().message.find("Sr"), std::string::npos)
        << what << ": " << got.error().message;
    EXPECT_NE(got.error().message.find(needle), std::string::npos)
        << what << ": " << got.error().message;
    std::cout << std::format("[ nevpt2   ] PC tangent refused {}: {}\n", what, got.error().message);
  }
}

} // namespace
