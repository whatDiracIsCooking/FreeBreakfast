// GPU cases for hessian::df_pc_nevpt2, the PC-NEVPT2 analytic nuclear
// Hessian, on this tree's own DF-CASSCF of data/nevpt2_hessian/'s fixtures
// (#803) through the registry, against their semi-numerical PC Hessians:
//
// - h2co_skew_sto3g (C1): every element within 2x the fixture's bar, each
//   part's largest element in bars (the visibility rule), and the
//   diagnostics, nothing refused;
// - nh3_cs_sto3g (Cs, symmetry-adapted coordinates): every element within
//   2x the bar and nothing refused (PC has no lifted null mode there);
// - h2co_neardeg_sto3g (a virtual pair 6.0e-3 Eh apart): the kappa_oo
//   measurement, every element within 2x the bar, nothing refused;
// - hcn_sto3g (C-infinity-v, PC's Sir nulls by symmetry): every element
//   within 2x the bar, no coordinate's Schur lift refused;
// - examples/nevpt2.toml's water (C2v, cc-pVDZ), symmetry-adapted (#952):
//   not refused, though its virtuals 17 and 18 are 1.4e-3 Eh apart and the
//   B2 vibration mixes them within 5.6e-4 bohr (SC is refused there); the B2
//   element against this tree's numerical ladder;
// - a C3v ammonia whose virtual e pairs are degenerate: refused, naming the
//   pairs;
// - a C2v formaldehyde CAS(6,6) whose Sr and Si keep under 1e-10: refused,
//   naming the class and its kept (#943);
// - HCN CAS(6,6), whose x and y coordinates lift Sir's null modes within
//   1e-3 bohr: refused, naming each coordinate, the class and t* (#943);
// - the input refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"
#include "symmetric_water.h"

namespace {

namespace fb = free_breakfast;

// The reference CASSCF's ||g|| (df_nevpt2_tangents_gpu.cpp's).
constexpr double kConvTolGrad = 1e-10;
// The issue's gate: every element within this many of the fixture's bars.
constexpr double kBars = 2.0;
// The visibility rule: each part has an element above this many bars.
constexpr double kVisibleBars = 10.0;
// Diagnostics: the CP residuals are df_casscf's (a converged solve sits near
// 1e-9), the asymmetry and the sum rule a fraction of the smallest bar.
constexpr double kCpResidual = 1e-8;
constexpr double kAsymmetry = 1e-9;
constexpr double kSumRule = 1e-8;

const fb::json &fixture(std::string_view name) {
  static std::map<std::string, fb::json, std::less<>> docs;
  if (const auto it = docs.find(name); it != docs.end()) {
    return it->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_hessian" /
                            std::format("{}.json", name));
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(std::string{name}, read.has_value() ? std::move(*read) : fb::json{})
      .first->second;
}

void upload(fb::DeviceResources &res, std::span<const double> host, double *d) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(d, host.data(), host.size_bytes(), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
}

// This tree's DF-CASSCF of a fixture's start (atoms, charge, basis, fitting
// set, active space) through the real registry in @p node.
std::optional<fb::Calculation> run_casscf(const fb::json &start, fb::NodeResources &node) {
  const fb::test::ScratchDir dir("df_pc_nevpt2_hessian");
  std::string xyz =
      std::format("{}\n{}\n", start.at("atoms").size(), start.at("name").get<std::string>());
  for (const fb::json &atom : start.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string active;
  for (const std::int32_t a : start.at("active").get<std::vector<std::int32_t>>()) {
    active += std::format("{}{}", active.empty() ? "" : ", ", a);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
      "[casscf]\nconv_tol_grad = {:g}\nncas = {}\nnelecas = {}\nactive = [{}]\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
      start.at("charge").get<std::int32_t>(), start.at("basis").at("name").get<std::string>(),
      kConvTolGrad, start.at("ncas").get<std::int32_t>(), start.at("nelecas").get<std::int32_t>(),
      active, start.at("aux_basis").at("name").get<std::string>());
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  if (!calc.has_value()) {
    return std::nullopt;
  }
  const fb::Status ran = calc->run(node);
  EXPECT_TRUE(ran.has_value()) << ran.error().message;
  if (!ran.has_value()) {
    return std::nullopt;
  }
  return std::move(*calc);
}

// One case: the registry's products and the one-root Solution of its
// canonical CASSCF.
struct Reference {
  std::unique_ptr<fb::NodeResources> node;
  std::optional<fb::Calculation> calc;
  std::optional<fb::mcscf::Solution> solution;
  std::int32_t ncore = 0;
  std::int32_t ncas = 0;

  [[nodiscard]] fb::DeviceResources &res() const { return node->device(0); }
  [[nodiscard]] const fb::Molecule &molecule() const {
    return calc->get<fb::MoleculeOutput>()->molecule;
  }
  [[nodiscard]] const fb::BasisSet &basis() const { return calc->get<fb::BasisSetOutput>()->basis; }
  [[nodiscard]] const fb::DensityFitted &fit() const {
    return std::get<fb::DensityFitted>(calc->get<fb::EriOutput>()->eri);
  }

  [[nodiscard]] fb::Result<fb::hessian::PcNevpt2Terms>
  hessian(std::span<const double> coordinates = {}) const {
    return hessian_of(*solution, coordinates);
  }
  [[nodiscard]] fb::Result<fb::hessian::PcNevpt2Terms>
  hessian_of(const fb::mcscf::Solution &s, std::span<const double> coordinates = {}) const {
    return fb::hessian::df_pc_nevpt2(res(), molecule(), basis(),
                                     calc->get<fb::PrimitivePairClassesOutput>()->classes,
                                     fit().aux, calc->get<fb::CoreHamiltonianOutput>()->h, fit().b,
                                     fit().l.data(), s, ncore, ncas, 0, coordinates);
  }
};

Reference reference(const fb::json &start) {
  Reference ref{.node = std::make_unique<fb::NodeResources>(),
                .calc = std::nullopt,
                .solution = std::nullopt};
  std::optional<fb::Calculation> calc = run_casscf(start, *ref.node);
  if (!calc.has_value()) {
    return ref;
  }
  ref.calc.emplace(std::move(*calc));
  fb::DeviceResources &res = ref.res();
  const auto &cas = ref.calc->get<fb::CasscfOutput>();
  EXPECT_TRUE(cas.has_value());
  if (!cas.has_value()) {
    return ref;
  }
  EXPECT_LT(cas->gradient_norm, kConvTolGrad);
  ref.ncore = cas->ncore;
  ref.ncas = cas->ncas;
  std::cout << std::format("[ casscf   ] {}: E {:.10f}, ||g|| {:.1e}\n",
                           start.at("name").get<std::string>(), cas->energy, cas->gradient_norm);
  ref.solution.emplace(fb::mcscf::Solution{
      .c = fb::SphericalMatrixAO(res, ref.basis()),
      .roots = {.energies = cas->energies,
                .spin_squares = cas->spin_squares,
                .vectors = fb::Buffers::device<double>(cas->vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas->energy,
      .gradient_norm = cas->gradient_norm,
      .gamma = cas->rdm1,
      .gamma2 = cas->rdm2,
      .root_gammas = cas->rdm1,
      .iterations = cas->iterations,
      .weights = cas->weights});
  upload(res, cas->vectors, ref.solution->roots.vectors.data());
  EXPECT_EQ(wwr::wwrMemcpyAsync(ref.solution->c.data(), cas->c.data(),
                                cas->c.num_elements() * sizeof(double),
                                wwr::wwrMemcpyDeviceToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return ref;
}

// A fixture's n x n matrix, [i][j], as its rows.
std::vector<std::vector<double>> matrix(const fb::json &m) {
  return m.get<std::vector<std::vector<double>>>();
}

// The rows @p keep of the fixture's coordinates, flattened row-major.
std::vector<double> coordinate_rows(const fb::json &doc, std::span<const std::size_t> keep) {
  const auto vectors = matrix(doc.at("coordinates").at("vectors"));
  std::vector<double> q;
  for (const std::size_t i : keep) {
    q.insert(q.end(), vectors[i].begin(), vectors[i].end());
  }
  return q;
}

// The largest |part| in bars of the fixture's elements, over @p keep's
// coordinates (part k x k column-major over them).
double in_bars(std::span<const double> part, const std::vector<std::vector<double>> &bar,
               std::span<const std::size_t> keep) {
  double worst = 0.0;
  const std::size_t k = keep.size();
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t i = 0; i < k; ++i) {
      worst = std::max(worst, std::abs(part[i + (k * j)]) / bar[keep[i]][keep[j]]);
    }
  }
  return worst;
}

// Every element of @p h over @p keep's coordinates against the fixture's,
// in bars; the worst miss.
double expect_within_bars(const fb::hessian::PcNevpt2Terms &h, const fb::json &scheme,
                          std::span<const std::size_t> keep, std::string_view what) {
  const auto want = matrix(scheme.at("hessian"));
  const auto bar = matrix(scheme.at("bar"));
  const std::size_t k = keep.size();
  EXPECT_EQ(h.hessian.size(), k * k) << what;
  double worst = 0.0;
  for (std::size_t j = 0; j < k && h.hessian.size() == k * k; ++j) {
    for (std::size_t i = 0; i < k; ++i) {
      const std::size_t a = keep[i];
      const std::size_t b = keep[j];
      const double miss = std::abs(h.hessian[i + (k * j)] - want[a][b]) / bar[a][b];
      worst = std::max(worst, miss);
      EXPECT_LE(miss, kBars) << what << ": element (" << a << ", " << b << ") "
                             << h.hessian[i + (k * j)] << " against " << want[a][b] << ", bar "
                             << bar[a][b];
    }
  }
  return worst;
}

std::vector<std::size_t> iota(std::size_t n) {
  std::vector<std::size_t> v(n);
  std::iota(v.begin(), v.end(), std::size_t{0});
  return v;
}

// #817's convention: a coordinate's Schur-lift step below this is refused.
constexpr double kLiftStep = 1e-3;

// The parts' largest elements in bars, printed; the smallest returned.
double print_parts(const fb::hessian::PcNevpt2Terms &h, const std::vector<std::vector<double>> &bar,
                   std::span<const std::size_t> keep, std::string_view tag) {
  double smallest = std::numeric_limits<double>::infinity();
  for (const auto &[name, part] :
       {std::pair{"E_cas", &h.e_cas}, std::pair{"Z d2g/dRdR", &h.z_lxy},
        std::pair{"Z L_Ax x_B", &h.z_lx}, std::pair{"Z third", &h.z_third},
        std::pair{"Z kappa_oo rows", &h.z_koo}, std::pair{"skeleton", &h.skeleton},
        std::pair{"<t, HVP t>", &h.hvp}, std::pair{"tau-kappa", &h.rotation_tau},
        std::pair{"kappa-kappa", &h.rotation_kappa}}) {
    const double bars = in_bars(*part, bar, keep);
    smallest = std::min(smallest, bars);
    std::cout << std::format("[ {:8} ] {:16} {:10.3e} bars (rule {:g})\n", tag, name, bars,
                             kVisibleBars);
  }
  return smallest;
}

// The diagnostics, printed: CP residual, Z-vector, asymmetry, sum rule,
// z^c, the smallest inactive gap, the smallest kept and Schur-lift step.
void print_diagnostics(const fb::hessian::PcNevpt2Terms &h, const fb::json &doc,
                       std::string_view tag) {
  double cp = 0.0;
  for (const double r : h.casscf.residuals) {
    cp = std::max(cp, r);
  }
  double kept = std::numeric_limits<double>::infinity();
  for (const double k : h.kept) {
    if (k > 0.0) {
      kept = std::min(kept, k);
    }
  }
  double step = std::numeric_limits<double>::infinity();
  for (const auto &steps : h.lift_steps) {
    for (const double s : steps) {
      step = std::min(step, s);
    }
  }
  std::cout << std::format("[ {:8} ] E_corr {:.10f} (fixture {:.10f}), CP residual {:.1e}, "
                           "Z-vector {:.1e}, asymmetry {:.1e}, sum rule {:.1e}, max |z^c| "
                           "{:.3e}, min gap {:.3e}, smallest kept {:.2e}, smallest t* {:.3e}\n",
                           tag, h.e_corr, doc.at("e_corr").at("pc").get<double>(), cp,
                           h.zvector_residual, h.asymmetry, h.translation, h.max_zc, h.min_gap,
                           kept, step);
  EXPECT_LE(cp, kCpResidual) << tag;
  EXPECT_LE(h.asymmetry, kAsymmetry) << tag;
  EXPECT_LE(h.translation, kSumRule) << tag;
  EXPECT_GE(step, kLiftStep) << tag;
}

TEST(HessianDfPcNevpt2, MatchesTheSkewFixture) {
  FBF_REQUIRE_GPU();
  const fb::json &doc = fixture("h2co_skew_sto3g");
  const Reference ref = reference(doc);
  ASSERT_TRUE(ref.solution.has_value());
  const auto keep = iota(3 * ref.basis().natoms());
  const auto start = std::chrono::steady_clock::now();
  auto h = ref.hessian(coordinate_rows(doc, keep));
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ASSERT_TRUE(h.has_value()) << h.error().message;
  const double worst = expect_within_bars(*h, doc.at("pc"), keep, "skew");
  std::cout << std::format("[ skew     ] worst element {:.3f} bars (allowed {:g}); {:.1f} s\n",
                           worst, kBars, seconds);
  // Visibility: every part has an element the fixture would see.
  EXPECT_GT(print_parts(*h, matrix(doc.at("pc").at("bar")), keep, "skew"), kVisibleBars);
  print_diagnostics(*h, doc, "skew");
}

TEST(HessianDfPcNevpt2, MatchesTheSymmetricFixture) {
  FBF_REQUIRE_GPU();
  const fb::json &doc = fixture("nh3_cs_sto3g");
  const Reference ref = reference(doc);
  ASSERT_TRUE(ref.solution.has_value());
  const auto keep = iota(3 * ref.basis().natoms());
  auto h = ref.hessian(coordinate_rows(doc, keep));
  ASSERT_TRUE(h.has_value()) << h.error().message;
  const double worst = expect_within_bars(*h, doc.at("pc"), keep, "nh3");
  std::cout << std::format("[ nh3      ] worst element {:.3f} bars (allowed {:g})\n", worst, kBars);
  print_diagnostics(*h, doc, "nh3");
}

// The kappa_oo measurement (#817): the virtual pair 6.0e-3 Eh apart makes
// kappa_oo^x = -f^(x) / (eps_p - eps_q) large, and the Hessian must carry it
// to the fixture's bar, whose ladder ran through 5.6e-5 Eh.
TEST(HessianDfPcNevpt2, MatchesTheNearDegenerateFixture) {
  FBF_REQUIRE_GPU();
  const fb::json &doc = fixture("h2co_neardeg_sto3g");
  const Reference ref = reference(doc);
  ASSERT_TRUE(ref.solution.has_value());
  const std::size_t n = 3 * ref.basis().natoms();
  const auto keep = iota(n);

  // kappa_oo's size: its core-core and virtual-virtual blocks (kappa_nr has
  // neither), over every coordinate.
  {
    const auto &classes = ref.calc->get<fb::PrimitivePairClassesOutput>()->classes;
    const auto &h_core = ref.calc->get<fb::CoreHamiltonianOutput>()->h;
    auto cp = fb::hessian::df_casscf_cp(ref.res(), ref.molecule(), ref.basis(), classes,
                                        ref.fit().aux, h_core, ref.fit().b, ref.fit().l.data(),
                                        *ref.solution, ref.ncore, ref.ncas);
    ASSERT_TRUE(cp.has_value()) << cp.error().message;
    const std::size_t nmo = ref.basis().nao_sphe();
    const auto first_virtual = static_cast<std::size_t>(ref.ncore + ref.ncas);
    double core = 0.0;
    double virt = 0.0;
    auto pairs = fb::hessian::df_nevpt2_tangents(
        ref.res(), ref.molecule(), ref.basis(), classes, ref.fit().aux, h_core, ref.fit().b,
        ref.fit().l.data(), *ref.solution, ref.ncore, ref.ncas, *cp, true,
        [&](std::int32_t, std::int32_t count, const double *, const double *, const double *,
            const double *kappa) -> fb::Status {
          std::vector<double> host(static_cast<std::size_t>(count) * nmo * nmo);
          EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), kappa, host.size() * sizeof(double),
                                        wwr::wwrMemcpyDeviceToHost, ref.res().stream()),
                    wwr::wwrSuccess);
          if (fb::Status synced = ref.res().sync(); !synced.has_value()) {
            return synced;
          }
          for (std::size_t x = 0; x < static_cast<std::size_t>(count); ++x) {
            for (std::size_t q = 0; q < nmo; ++q) {
              for (std::size_t p = 0; p < nmo; ++p) {
                const double v = std::abs(host[(x * nmo * nmo) + p + (nmo * q)]);
                if (p < static_cast<std::size_t>(ref.ncore) &&
                    q < static_cast<std::size_t>(ref.ncore)) {
                  core = std::max(core, v);
                } else if (p >= first_virtual && q >= first_virtual) {
                  virt = std::max(virt, v);
                }
              }
            }
          }
          return {};
        });
    ASSERT_TRUE(pairs.has_value()) << pairs.error().message;
    EXPECT_TRUE(pairs->empty()) << "a pair below the degenerate-pair rule";
    std::cout << std::format("[ neardeg  ] max |kappa_oo^x|: core-core {:.3e}, virtual-virtual "
                             "{:.3e} (virtual gap {:.3e} Eh)\n",
                             core, virt, doc.at("inactive_gaps").at("virtual").get<double>());
  }

  auto h = ref.hessian(coordinate_rows(doc, keep));
  ASSERT_TRUE(h.has_value()) << h.error().message;
  const double worst = expect_within_bars(*h, doc.at("pc"), keep, "neardeg");
  std::cout << std::format("[ neardeg  ] worst element {:.3f} bars (allowed {:g})\n", worst, kBars);
  print_parts(*h, matrix(doc.at("pc").at("bar")), keep, "neardeg");
  print_diagnostics(*h, doc, "neardeg");
}

TEST(HessianDfPcNevpt2, MatchesTheLinearFixture) {
  FBF_REQUIRE_GPU();
  const fb::json &doc = fixture("hcn_sto3g");
  const Reference ref = reference(doc);
  ASSERT_TRUE(ref.solution.has_value());
  const auto keep = iota(3 * ref.basis().natoms());
  auto h = ref.hessian(coordinate_rows(doc, keep));
  ASSERT_TRUE(h.has_value()) << h.error().message;
  const double worst = expect_within_bars(*h, doc.at("pc"), keep, "hcn");
  std::cout << std::format("[ hcn      ] worst element {:.3f} bars (allowed {:g})\n", worst, kBars);
  print_diagnostics(*h, doc, "hcn");
}

// The B2 element of the symmetric water, from the numerical [hessian]'s
// ladder of PC's analytic gradient along the B2 vibration (#952; RTX 3080,
// --preset default): 1.2850090393, ...0426, ...0413 at h = 0.004, 0.002,
// 0.001 bohr, and 1.2850090384, ...0392, ...0198 at h = 0.0005 down to
// 1.25e-4. Flat to its floor, 2.1e-7 at h = 0.004: PC does not see the
// inactive pair's mixing that makes SC's element 15.7 there.
constexpr double kWaterB2Ladder = 1.2850090413;
constexpr double kWaterB2Floor = 2.1e-7;

TEST(HessianDfPcNevpt2, SymmetricWaterIsNotRefused) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const Reference ref = reference(fb::test::symmetric_water());
  ASSERT_TRUE(ref.solution.has_value());
  ASSERT_EQ(fb::test::symmetric_water_coordinates().labels[5], "B2 vibration");
  const auto keep = iota(9);
  auto h = ref.hessian(fb::test::symmetric_water_rows(keep));
  ASSERT_TRUE(h.has_value()) << h.error().message;
  const double b2 = h->hessian[5 + (9 * 5)];
  std::cout << std::format("[ water    ] PC's B2 element {:.10f} (ladder {:.10f}), asymmetry "
                           "{:.1e}\n",
                           b2, kWaterB2Ladder, h->asymmetry);
  EXPECT_LE(std::abs(b2 - kWaterB2Ladder), kBars * kWaterB2Floor);
  EXPECT_LE(h->asymmetry, kAsymmetry);
}

// A C3v ammonia in 6-31G, its full valence CAS(8,7) (df_nevpt2_tangents_gpu.cpp's):
// its virtual e pairs are degenerate, and E displacements move f within them.
fb::json c3v_ammonia() {
  const double rho = std::sqrt((1.9 * 1.9) - (0.7 * 0.7));
  fb::json atoms = fb::json::array();
  atoms.push_back({{"symbol", "N"}, {"coord", {0.0, 0.0, 0.0}}});
  for (std::int32_t k = 0; k < 3; ++k) {
    const double phi = 2.0 * std::numbers::pi * k / 3.0;
    atoms.push_back({{"symbol", "H"}, {"coord", {rho * std::cos(phi), rho * std::sin(phi), -0.7}}});
  }
  return {{"name", "nh3_c3v_631g"},
          {"atoms", atoms},
          {"charge", 0},
          {"basis", {{"name", "6-31G"}}},
          {"aux_basis", {{"name", "cc-pVDZ-JKFIT"}}},
          {"ncas", 7},
          {"nelecas", 8},
          {"active", {1, 2, 3, 4, 5, 6, 7}}};
}

TEST(HessianDfPcNevpt2, DegenerateInactivePairIsRefused) {
  FBF_REQUIRE_GPU();
  const Reference ref = reference(c3v_ammonia());
  ASSERT_TRUE(ref.solution.has_value());
  const auto h = ref.hessian();
  ASSERT_FALSE(h.has_value()) << "a degenerate pair's kappa_oo was taken";
  EXPECT_EQ(h.error().kind, fb::ErrorKind::Numerical) << h.error().message;
  EXPECT_NE(h.error().message.find("hessian::df_pc_nevpt2"), std::string::npos)
      << h.error().message;
  EXPECT_NE(h.error().message.find("(coordinate "), std::string::npos) << h.error().message;
  EXPECT_NE(h.error().message.find(", orbitals "), std::string::npos) << h.error().message;
  EXPECT_NE(h.error().message.find(", gap "), std::string::npos) << h.error().message;
  std::cout << std::format("[ c3v      ] {}\n", h.error().message.substr(0, 300));
}

// The kept refusal (#943): a C2v formaldehyde in STO-3G with CAS(6,6) over
// orbitals 5 to 10. Sr's and Si's metrics have real modes between the
// default cut and 1e-10 of their largest, so their smallest kept
// eigenvalues are about 3e-12 and 4e-12: under the 1e-10 at which PC's
// second derivative is attainable (#917). At a pc_floor of 1e-10 they keep
// 2.5e-10 and 2.8e-10 and the call is not refused.
fb::json c2v_formaldehyde() {
  const auto bohr = [](double angstrom) { return angstrom / fb::kAngstromPerBohr<double>; };
  return {{"name", "h2co_c2v_sto3g_cas66"},
          {"atoms",
           {{{"symbol", "C"}, {"coord", {0.0, 0.0, 0.0}}},
            {{"symbol", "O"}, {"coord", {0.0, 0.0, bohr(1.21)}}},
            {{"symbol", "H"}, {"coord", {0.0, bohr(0.94), bohr(-0.59)}}},
            {{"symbol", "H"}, {"coord", {0.0, bohr(-0.94), bohr(-0.59)}}}}},
          {"charge", 0},
          {"basis", {{"name", "sto-3g"}}},
          {"aux_basis", {{"name", "cc-pVDZ-JKFIT"}}},
          {"ncas", 6},
          {"nelecas", 6},
          {"active", {5, 6, 7, 8, 9, 10}}};
}

TEST(HessianDfPcNevpt2, ThinKeptClassIsRefused) {
  FBF_REQUIRE_GPU();
  const Reference ref = reference(c2v_formaldehyde());
  ASSERT_TRUE(ref.solution.has_value());
  const auto h = ref.hessian();
  ASSERT_FALSE(h.has_value()) << "a class kept below 1e-10 was differentiated";
  const std::string &m = h.error().message;
  EXPECT_EQ(h.error().kind, fb::ErrorKind::Numerical) << m;
  EXPECT_NE(m.find("hessian::df_pc_nevpt2: Sr keeps "), std::string::npos) << m;
  EXPECT_NE(m.find("a pc_floor of 1e-10 or more cuts it"), std::string::npos) << m;
  // The value it names: Sr's kept, below the bar.
  const std::size_t at = m.find("Sr keeps ") + std::string_view{"Sr keeps "}.size();
  EXPECT_LT(std::stod(m.substr(at)), 1e-10) << m;
  std::cout << std::format("[ kept     ] {}\n", m.substr(0, 300));
}

// The Schur-lift refusal (#943): hcn_sto3g's geometry with CAS(6,6), the
// fixture's pi space and the sigma pair beside it. A transverse coordinate's
// CP response mixes sigma and pi inside the active space, so it breaks the
// axial symmetry that keeps Sir's null modes null, and lifts them: at the
// default cut each x and y coordinate's t* is about 2e-4 to 5e-4 bohr. The
// z coordinates keep the symmetry and lift nothing.
TEST(HessianDfPcNevpt2, NearbyBranchSwitchIsRefused) {
  FBF_REQUIRE_GPU();
  fb::json start = fixture("hcn_sto3g");
  start["name"] = "hcn_sto3g_cas66";
  start["ncas"] = 6;
  start["nelecas"] = 6;
  start["active"] = {4, 5, 6, 7, 8, 9};
  const Reference ref = reference(start);
  ASSERT_TRUE(ref.solution.has_value());
  const auto h = ref.hessian();
  ASSERT_FALSE(h.has_value()) << "a Schur lift within 1e-3 bohr was differentiated";
  const std::string &m = h.error().message;
  EXPECT_EQ(h.error().kind, fb::ErrorKind::Numerical) << m;
  EXPECT_NE(m.find("hessian::df_pc_nevpt2: the PC surface switches branch"), std::string::npos)
      << m;
  // Every transverse coordinate is named, with its class and a t* under the
  // bar; no axial one is.
  for (std::size_t x = 0; x < 9; ++x) {
    const std::string named = std::format("coordinate {} class Sir t* ", x);
    const std::size_t at = m.find(named);
    if (x % 3 == 2) {
      EXPECT_EQ(at, std::string::npos) << named << m;
      continue;
    }
    ASSERT_NE(at, std::string::npos) << named << m;
    EXPECT_LT(std::stod(m.substr(at + named.size())), kLiftStep) << named << m;
  }
  std::cout << std::format("[ lift     ] {}\n", m.substr(0, 600));
}

TEST(HessianDfPcNevpt2, Refusals) {
  FBF_REQUIRE_GPU();
  const Reference ref = reference(fixture("h2co_skew_sto3g"));
  ASSERT_TRUE(ref.solution.has_value());
  fb::DeviceResources &res = ref.res();
  const auto expect_refused = [](const auto &r, const char *what) {
    ASSERT_FALSE(r.has_value()) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what << ": " << r.error().message;
    EXPECT_NE(r.error().message.find("hessian::df_pc_nevpt2"), std::string::npos)
        << what << ": " << r.error().message;
  };
  const std::size_t n = 3 * ref.basis().natoms();
  // Not whole rows, and two rows that are not orthonormal.
  expect_refused(ref.hessian(std::vector<double>(n + 1, 0.0)), "a row and one more");
  std::vector<double> two(2 * n, 0.0);
  two[0] = 1.0;
  two[n] = 1.0;
  expect_refused(ref.hessian(two), "two rows along one coordinate");
  // A state average (QD's reference): two roots.
  fb::mcscf::Solution averaged{
      .c = fb::SphericalMatrixAO(res, ref.basis()),
      .roots = {.energies = {ref.solution->energy, ref.solution->energy + 0.1},
                .spin_squares = {0.0, 0.0},
                .vectors = fb::Buffers::device<double>(
                    2 * ref.solution->roots.vectors.num_elements(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = ref.solution->energy,
      .gradient_norm = ref.solution->gradient_norm,
      .gamma = ref.solution->gamma,
      .gamma2 = ref.solution->gamma2,
      .root_gammas = ref.solution->root_gammas,
      .iterations = {},
      .weights = {0.5, 0.5}};
  expect_refused(ref.hessian_of(averaged), "a state average");
}

} // namespace
