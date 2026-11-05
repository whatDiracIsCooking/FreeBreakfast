// The Nevpt2 step against PySCF's SC-NEVPT2 (data/nevpt2/nevpt2_df.json,
// tools/emit/nevpt2_reference.py), through the real registry: a config naming
// [casscf], a density-fitted [eri] and [nevpt2] plans the whole chain, runs
// our own DF-CASSCF from each data/casscf/casscf_df.json start, then SC-NEVPT2
// per root. Compared per root: each class's norm and energy and their sum,
// and each PC-NEVPT2 class energy and their sum against block2's on the same
// state (PySCF has no PC), with what must hold of PC beside SC: see kPcSlack.
// N2 is not among the cases: SC-NEVPT2 depends on the arbitrary split of its
// degenerate pi* virtual pairs by up to 5e-6 (data/README.md
// "nevpt2/nevpt2_df.json"), so the reference leaves it out.
// Then a state whose PC metric has no gap at its cut, which PC accepts
// (#853), a state PC refuses, which keeps its SC, and the refusal of an
// open-shell ScfOutput. The refusal of an
// integral-direct [eri], made when the config is read, is
// test/pipeline/config_host.cpp's.
//
// Then [nevpt2] quasi_degenerate = true (PipelineQdNevpt2): the H2CO cases of
// data/qd_nevpt2/ (tools/emit/qd_nevpt2_reference.py) and its two polished
// near-degenerate H8 cases through our own DF-SA-CASSCF from the same start,
// the SC- and PC-QD-NEVPT2 eigenvalues and vectors against theirs; PC-QD
// accepted through a gapless metric; and QD over one root refused by the
// step itself.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The CASSCF's convergence, as pipeline_casscf_gpu.cpp runs it.
constexpr double kConvTolGrad = 1e-9;
// pipeline_casscf_gpu.cpp's bar on each root's CASSCF energy.
constexpr double kCasscfEnergyTolerance = 1e-8;

// The tolerance on every class norm and energy and on E_corr, derived from
// how closely our CASSCF reaches the reference's polished orbitals.
//
// NEVPT2 is first order in the orbitals, where the CASSCF energy is second.
// data/README.md "nevpt2/nevpt2_df.json" measured how far PySCF's own NEVPT2
// moves between mc1step's unpolished orbitals and the polished ones, per unit
// of ||g||: 1.3e-9 / 1.5e-6 = 9e-4 (h2o_ccpvdz), 4.9e-8 / 1.0e-6 = 0.05 and
// 4.3e-8 / 6.5e-7 = 0.07 (the two h2co cases); take 0.07 for every case,
// h2o_ccpvdz_avas (not measured there) included. Each orbital set is within
// ~||g|| / (H's least eigenvalue) of the true minimum, ours at ||g|| <
// kConvTolGrad (asserted below) and the reference's at < 1e-10, so the
// orbital term is below 0.07 (1e-9 + 1e-10) ~ 8e-11. The CI vectors are each
// side's to a relative residual of 1e-12 (ci::SolverOptions; the reference's
// are dense), ~1e-10 hartree, and NEVPT2 is first order in them too. That is
// ~2e-10 together; 1e-9 leaves a factor five. Measured on an RTX 3080
// (CUDA) and an RX 9060 XT (HIP): ||g|| 6e-12 to 5e-10, each root's CASSCF
// energy within 1.3e-12 of the reference's, and every class and E_corr
// within 4e-11.
//
// Not the 1e-7 of nevpt2_energy_gpu.cpp, which is the demo's bar on
// identical inputs; and nothing here is loosened for a case. PC is first
// order in the orbitals and the CI vectors as SC is, so the same derivation
// sets its bar against block2's PC on the reference's state.
constexpr double kTolerance = 1e-9;

// PC is SC without the final contraction: each class minimizes the same
// Hylleraas functional over a span holding SC's one function per tuple, so
// no PC class lies above its SC class, and Sijrs, with no active index, is
// SC's own. Both are checked to this, which is only rounding: the two are
// computed from the same slabs, in the same order.
constexpr double kPcSlack = 1e-12;

// The fitting set both references were made over, compiled in.
constexpr std::string_view kDfAuxBasis = "cc-pVDZ-JKFIT";

struct Case {
  std::string_view name;
  // Under slow_test.h: every case but H2O cc-pVDZ, the cheapest.
  bool slow;
};

// ctest names a discovered case by this, so `ctest -R h2co_ccpvdz` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

// data/<dir>/<file>, read once per file.
const fb::json &reference(std::string_view dir, std::string_view file) {
  static std::map<std::string, fb::json, std::less<>> docs;
  const std::string key = std::format("{}/{}", dir, file);
  if (const auto it = docs.find(key); it != docs.end()) {
    return it->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / dir / file);
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(key, read.has_value() ? std::move(*read) : fb::json{}).first->second;
}

const fb::json &reference_case(std::string_view dir, std::string_view file, std::string_view name) {
  for (const fb::json &entry : reference(dir, file).at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in " << dir << "/" << file;
  static const fb::json kNone;
  return kNone;
}

// The reference's molecule written as an XYZ file in dir, and a config naming
// it with the given tables after it.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view tables) {
  // Bohr in the reference, Angstrom in an XYZ file.
  std::string xyz =
      std::format("{}\n{}\n", entry.at("atoms").size(), entry.at("name").get<std::string>());
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n{}",
      entry.at("charge").get<std::int32_t>(), entry.at("basis").get<std::string>(), tables);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The [casscf] table of the reference's start and weights.
std::string casscf_table(const fb::json &entry) {
  std::string table = std::format("[casscf]\nconv_tol_grad = {:g}\n", kConvTolGrad);
  if (!entry.at("avas").is_null()) {
    table += std::format("avas_threshold = {:.17g}\navas = [",
                         entry.at("avas").at("threshold").get<double>());
    const auto labels = entry.at("avas").at("labels").get<std::vector<std::string>>();
    for (std::size_t i = 0; i < labels.size(); ++i) {
      table += std::format("{}\"{}\"", i == 0 ? "" : ", ", labels[i]);
    }
    table += "]\n";
  } else {
    table += std::format("ncas = {}\nnelecas = {}\n", entry.at("ncas").get<std::int32_t>(),
                         entry.at("nelecas").get<std::int32_t>());
    if (!entry.at("active").is_null()) {
      const auto active = entry.at("active").get<std::vector<std::int32_t>>();
      table += "active = [";
      for (std::size_t i = 0; i < active.size(); ++i) {
        table += std::format("{}{}", i == 0 ? "" : ", ", active[i]);
      }
      table += "]\n";
    }
  }
  const auto weights = entry.at("weights").get<std::vector<double>>();
  table += std::format("nroots = {}\nweights = [", weights.size());
  for (std::size_t i = 0; i < weights.size(); ++i) {
    table += std::format("{}{:.17g}", i == 0 ? "" : ", ", weights[i]);
  }
  return table + "]\n";
}

// Each root of @p nevpt2 against the reference's @p roots: every class's SC
// norm and energy and their sum, and every PC class energy and their sum
// against block2's, to kTolerance; PC beside SC to kPcSlack. The CASSCF's
// root energies against @p casscf_energies.
void expect_roots(std::string_view label, const fb::json &roots,
                  const std::vector<double> &casscf_energies, const fb::CasscfOutput &casscf,
                  const fb::Nevpt2Output &nevpt2) {
  ASSERT_EQ(nevpt2.roots.size(), roots.size());
  ASSERT_EQ(casscf.energies.size(), roots.size());
  for (std::size_t r = 0; r < roots.size(); ++r) {
    const fb::json &root = roots[r];
    const fb::json &classes = root.at("classes");
    ASSERT_EQ(classes.size(), 8U);
    EXPECT_NEAR(casscf.energies[r], casscf_energies[r], kCasscfEnergyTolerance) << "root " << r;
    const fb::nevpt2::ClassEnergies &got = nevpt2.roots[r];
    double worst = 0.0;
    for (std::size_t k = 0; k < 8; ++k) {
      const std::string name = classes[k].at("name").get<std::string>();
      const double norm = classes[k].at("norm").get<double>();
      const double energy = classes[k].at("energy").get<double>();
      EXPECT_NEAR(got.norms[k], norm, kTolerance) << "root " << r << " " << name << " norm";
      EXPECT_NEAR(got.energies[k], energy, kTolerance) << "root " << r << " " << name;
      worst = std::max({worst, std::abs(got.norms[k] - norm), std::abs(got.energies[k] - energy)});
    }
    const double e_corr = root.at("e_corr").get<double>();
    EXPECT_NEAR(got.total, e_corr, kTolerance) << "root " << r << " E_corr";

    // PC against block2's, and beside SC: none above SC's, Sijrs SC's own.
    ASSERT_EQ(nevpt2.pc_roots.size(), roots.size());
    ASSERT_TRUE(nevpt2.pc_roots[r].has_value()) << "root " << r << ": " << nevpt2.pc_refusals[r];
    const fb::nevpt2::PcClassEnergies &pc = *nevpt2.pc_roots[r];
    const fb::json &pc_classes = root.at("pc_class_energies");
    ASSERT_EQ(pc_classes.size(), 8U);
    double pc_sum = 0.0;
    double pc_worst = 0.0;
    for (std::size_t k = 0; k < 8; ++k) {
      const std::string name = classes[k].at("name").get<std::string>();
      ASSERT_EQ(pc_classes[k].at("name").get<std::string>(), name);
      const double energy = pc_classes[k].at("energy").get<double>();
      EXPECT_NEAR(pc.energies[k], energy, kTolerance) << "root " << r << " PC " << name;
      EXPECT_LE(pc.energies[k], got.energies[k] + kPcSlack) << "root " << r << " PC " << name;
      pc_sum += pc.energies[k];
      pc_worst = std::max(pc_worst, std::abs(pc.energies[k] - energy));
    }
    const double e_pc = root.at("e_pc_total").get<double>();
    EXPECT_NEAR(pc.total, e_pc, kTolerance) << "root " << r << " PC E_corr";
    EXPECT_NEAR(pc.energies[2], got.energies[2], kPcSlack) << "root " << r << " PC Sijrs";
    EXPECT_EQ(pc.dropped[2], 0) << "root " << r << " PC Sijrs";
    EXPECT_NEAR(pc.total, pc_sum, kPcSlack) << "root " << r << " PC E_corr";
    // What kTolerance's derivation rests on, printed for the record.
    std::cout << std::format("[ nevpt2   ] {} root {}: |g| {:.2e}, CASSCF dE {:+.2e}, "
                             "E_corr {:.12f} dE_corr {:+.2e}, worst class {:.2e} (tol {:.0e})\n",
                             label, r, casscf.gradient_norm,
                             casscf.energies[r] - casscf_energies[r], got.total, got.total - e_corr,
                             worst, kTolerance);
    std::cout << std::format("[ nevpt2   ] {} root {}: PC E_corr {:.12f} dE_corr {:+.2e}, "
                             "worst class {:.2e} (PC - SC {:+.2e}), dropped {}\n",
                             label, r, pc.total, pc.total - e_pc, pc_worst, pc.total - got.total,
                             pc.dropped);
  }
}

class PipelineNevpt2 : public ::testing::TestWithParam<Case> {};

TEST_P(PipelineNevpt2, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  const Case &c = GetParam();
  if (c.slow) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &start = reference_case("casscf", "casscf_df.json", c.name);
  const fb::json &want = reference_case("nevpt2", "nevpt2_df.json", c.name);
  ASSERT_EQ(start.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  ASSERT_EQ(want.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);

  const fb::test::ScratchDir dir("pipeline_nevpt2");
  auto calc = fb::read_config(
      write_config(dir, start,
                   casscf_table(start) +
                       std::format("[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
                                   "[nevpt2]\n",
                                   kDfAuxBasis)),
      dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &casscf = calc->get<fb::CasscfOutput>();
  const auto &nevpt2 = calc->get<fb::Nevpt2Output>();
  ASSERT_TRUE(casscf.has_value() && nevpt2.has_value());
  EXPECT_LT(casscf->gradient_norm, kConvTolGrad);
  // Without quasi_degenerate, no QD product.
  EXPECT_TRUE(nevpt2->qd_couplings.empty() && nevpt2->qd_energies.empty() &&
              nevpt2->pc_qd_energies.empty() && nevpt2->pc_qd_refusal.empty());

  expect_roots(c.name, want.at("roots"), want.at("casscf_energies").get<std::vector<double>>(),
               *casscf, *nevpt2);
}

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineNevpt2,
                         ::testing::Values(Case{.name = "h2o_ccpvdz", .slow = false},
                                           Case{.name = "h2co_ccpvdz_sa2", .slow = true},
                                           Case{.name = "h2co_ccpvdz_sa2_weighted", .slow = true},
                                           Case{.name = "h2o_ccpvdz_avas", .slow = true}),
                         [](const auto &info) { return std::string(info.param.name); });

// Frozen core (#859): [nevpt2] frozen_core against data/nevpt2/
// nevpt2_df_frozen_core.json (tools/emit/nevpt2_reference.py --frozen-core:
// PySCF's class functions with the frozen orbitals sliced out of its fitted
// integrals, h1eff over the whole core, and block2's PC with frozen=n), through
// the same chain and at the same bars as MatchesPyscf: the frozen orbitals are
// a slice of what that test already holds, so its derivation carries over.
// One case per frozen_core value as the key spells it: H2O's "auto" (its O
// 1s), and H2CO's "auto" (C and O 1s) and 1 (the lowest, its O 1s).
struct FrozenCase {
  std::string_view name;
  std::string_view frozen_core;
  bool slow;
};

void PrintTo(const FrozenCase &c, std::ostream *os) {
  // ctest names a case by this: "auto" unquoted, so `ctest -R` can select it.
  *os << c.name << "_" << (c.frozen_core.starts_with('"') ? "auto" : c.frozen_core);
}

class PipelineNevpt2FrozenCore : public ::testing::TestWithParam<FrozenCase> {};

TEST_P(PipelineNevpt2FrozenCore, MatchesReference) {
  FBF_REQUIRE_GPU();
  const FrozenCase &c = GetParam();
  if (c.slow) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &start = reference_case("casscf", "casscf_df.json", c.name);
  const fb::json &want = reference_case("nevpt2", "nevpt2_df_frozen_core.json", c.name);
  ASSERT_EQ(want.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  const fb::json *variant = nullptr;
  for (const fb::json &v : want.at("variants")) {
    const fb::json &key = v.at("frozen_core");
    if ((key.is_string() ? std::format("\"{}\"", key.get<std::string>())
                         : std::to_string(key.get<std::int32_t>())) == c.frozen_core) {
      variant = &v;
    }
  }
  ASSERT_NE(variant, nullptr) << c.frozen_core;

  const fb::test::ScratchDir dir("pipeline_nevpt2_frozen_core");
  auto calc = fb::read_config(
      write_config(dir, start,
                   casscf_table(start) +
                       std::format("[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
                                   "[nevpt2]\nfrozen_core = {}\n",
                                   kDfAuxBasis, c.frozen_core)),
      dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &casscf = calc->get<fb::CasscfOutput>();
  const auto &nevpt2 = calc->get<fb::Nevpt2Output>();
  ASSERT_TRUE(casscf.has_value() && nevpt2.has_value());
  EXPECT_LT(casscf->gradient_norm, kConvTolGrad);
  EXPECT_EQ(nevpt2->frozen_core, variant->at("n_frozen").get<std::int32_t>());
  expect_roots(std::format("{} frozen_core {}", c.name, c.frozen_core), variant->at("roots"),
               want.at("casscf_energies").get<std::vector<double>>(), *casscf, *nevpt2);
}

INSTANTIATE_TEST_SUITE_P(
    Reference, PipelineNevpt2FrozenCore,
    ::testing::Values(FrozenCase{.name = "h2o_ccpvdz", .frozen_core = "\"auto\"", .slow = false},
                      FrozenCase{
                          .name = "h2co_ccpvdz_sa2", .frozen_core = "\"auto\"", .slow = true},
                      FrozenCase{.name = "h2co_ccpvdz_sa2", .frozen_core = "1", .slow = true}),
    [](const auto &info) {
      return std::format("{}_{}", info.param.name,
                         info.param.frozen_core.starts_with('"') ? "auto" : info.param.frozen_core);
    });

// PC through a metric spectrum with no gap at its cut (#853). HF in cc-pVDZ,
// CAS(10,7) over every electron: the F 1s is active and all but doubly
// occupied (2 - n ~1e-5), so Sr's metric (d = 343) has eigenvalues of its
// products with the other small occupations that run continuously from
// 5.8e-14 of its largest up, under 14 null modes at <= 4.6e-17. Until #853
// that was refused (smallest kept / largest dropped 1.7 against 1e3); those
// modes are real and carry no energy, and PC E_corr moves by 2.6e-9 Eh with
// the cut anywhere from 1e-15 to 1e-12 of the largest, from -0.1415607189 to
// -0.1415607163 (RTX 3080, ../../src/nevpt2/energy/README.md "PC-NEVPT2").
// So it is accepted, and E_corr must sit in that band: within kHfPcBand of its
// middle. The bar is the band's half-width with room for the CASSCF's
// convergence (first order in the orbitals, under 1e-9 at kConvTolGrad); a
// cut that moved up into 1e-11 .. 1e-10 would move E_corr by 3e-8 .. 1.7e-7.
// The small-molecule twin of PSB3's CAS(6,6).
constexpr double kHfPcCorr = -0.1415607176;
constexpr double kHfPcBand = 1e-8;
TEST(PipelineNevpt2, PcAcceptedThroughAContinuum) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_nevpt2");
  ASSERT_TRUE(
      fb::write_file(dir / "hf.xyz", "2\nhydrogen fluoride\nH 0 0 0\nF 0 0 0.917\n").has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"hf.xyz\"\n[basis_set]\nbasis = \"cc-pvdz\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
                  "[casscf]\nncas = 7\nnelecas = 10\nconv_tol_grad = {:g}\n[nevpt2]\n",
                  kDfAuxBasis, kConvTolGrad);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &nevpt2 = calc->get<fb::Nevpt2Output>();
  ASSERT_TRUE(nevpt2.has_value());

  // SC stands: every class computed, the correlation energy their sum.
  ASSERT_EQ(nevpt2->roots.size(), 1U);
  const fb::nevpt2::ClassEnergies &sc = nevpt2->roots[0];
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_TRUE(std::isfinite(sc.energies[k])) << k;
    EXPECT_LE(sc.energies[k], 0.0) << k;
  }
  EXPECT_NEAR(sc.total, std::ranges::fold_left(sc.energies, 0.0, std::plus{}), kPcSlack);
  EXPECT_LT(sc.total, -0.05);

  // PC stands: accepted, every class at or below SC's, E_corr in the band.
  ASSERT_EQ(nevpt2->pc_roots.size(), 1U);
  ASSERT_EQ(nevpt2->pc_refusals.size(), 1U);
  ASSERT_TRUE(nevpt2->pc_roots[0].has_value()) << nevpt2->pc_refusals[0];
  EXPECT_TRUE(nevpt2->pc_refusals[0].empty()) << nevpt2->pc_refusals[0];
  const fb::nevpt2::PcClassEnergies &pc = *nevpt2->pc_roots[0];
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_TRUE(std::isfinite(pc.energies[k])) << k;
    EXPECT_LE(pc.energies[k], sc.energies[k] + kPcSlack) << k;
  }
  EXPECT_NEAR(pc.total, std::ranges::fold_left(pc.energies, 0.0, std::plus{}), kPcSlack);
  EXPECT_NEAR(pc.total, kHfPcCorr, kHfPcBand);
  std::cout << std::format("[ nevpt2   ] HF CAS(10,7): SC E_corr {:.12f}; PC E_corr {:.12f} "
                           "(band {:.10f} +- {:.0e}), dropped {}\n",
                           sc.total, pc.total, kHfPcCorr, kHfPcBand, pc.dropped);
}

// A state whose PC is refused keeps its SC: water in STO-3G with CAS(2,1),
// its one active orbital doubly occupied, so Si's metric is zero and has no
// positive eigenvalue, the one refusal besides a non-positive denominator
// that the PC solve has (#853).
TEST(PipelineNevpt2, PcRefusalKeepsSc) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_nevpt2");
  ASSERT_TRUE(fb::write_file(dir / "water.xyz", "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\n"
                                                "H 0 -0.7572 -0.4692\n")
                  .has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"water.xyz\"\n[basis_set]\nbasis = \"sto-3g\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
                  "[casscf]\nncas = 1\nnelecas = 2\nconv_tol_grad = {:g}\n[nevpt2]\n",
                  kDfAuxBasis, kConvTolGrad);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &nevpt2 = calc->get<fb::Nevpt2Output>();
  ASSERT_TRUE(nevpt2.has_value());

  // SC stands: every class computed, the correlation energy their sum.
  ASSERT_EQ(nevpt2->roots.size(), 1U);
  const fb::nevpt2::ClassEnergies &sc = nevpt2->roots[0];
  for (std::size_t k = 0; k < 8; ++k) {
    EXPECT_TRUE(std::isfinite(sc.energies[k])) << k;
    EXPECT_LE(sc.energies[k], 0.0) << k;
  }
  EXPECT_NEAR(sc.total, std::ranges::fold_left(sc.energies, 0.0, std::plus{}), kPcSlack);
  EXPECT_LT(sc.total, -0.01);

  // PC is empty, and says which class refused.
  ASSERT_EQ(nevpt2->pc_roots.size(), 1U);
  ASSERT_EQ(nevpt2->pc_refusals.size(), 1U);
  EXPECT_FALSE(nevpt2->pc_roots[0].has_value());
  EXPECT_NE(nevpt2->pc_refusals[0].find("Si's metric has no positive eigenvalue"),
            std::string::npos)
      << nevpt2->pc_refusals[0];
  std::cout << std::format("[ nevpt2   ] water CAS(2,1): SC E_corr {:.12f}; PC {}\n", sc.total,
                           nevpt2->pc_refusals[0]);
}

// An open-shell ScfOutput, n_alpha != n_beta, is refused when the step runs,
// naming the step, before anything is queued; so is a molecule of
// multiplicity other than 1 over the closed-shell one. The rest of the chain
// is water's DF-CASSCF(4,4) in STO-3G; the open shell's orbitals are never
// read.
TEST(PipelineNevpt2, AnOpenShellOrANonSingletIsUnsupported) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_nevpt2");
  ASSERT_TRUE(fb::write_file(dir / "water.xyz", "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\n"
                                                "H 0 -0.7572 -0.4692\n")
                  .has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"water.xyz\"\n[basis_set]\nbasis = \"sto-3g\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
                  "[casscf]\nncas = 4\nnelecas = 4\n",
                  kDfAuxBasis);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const fb::BasisSet &basis = calc->get<fb::BasisSetOutput>()->basis;
  const fb::ScfOutput open{.energy = 0.0,
                           .iterations = 0,
                           .n_alpha = 6,
                           .n_beta = 4,
                           .mo_energies = {},
                           .c = fb::SphericalMatrixAO(res, basis),
                           .d = fb::SphericalMatrixAO(res, basis)};
  const auto refused = fb::Nevpt2Step{}.run(
      fb::StepInput<fb::Nevpt2Step>{*calc->get<fb::MoleculeOutput>(),
                                    *calc->get<fb::BasisSetOutput>(), *calc->get<fb::EriOutput>(),
                                    *calc->get<fb::CoreHamiltonianOutput>(), open,
                                    *calc->get<fb::CasscfOutput>()},
      res);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::Unsupported);
  EXPECT_NE(refused.error().message.find("[nevpt2] needs a closed-shell reference"),
            std::string::npos)
      << refused.error().message;

  const fb::Molecule &singlet = calc->get<fb::MoleculeOutput>()->molecule;
  const fb::MoleculeOutput triplet{fb::Molecule(res, singlet.atoms(), singlet.charge(), 3)};
  const auto not_singlet = fb::Nevpt2Step{}.run(
      fb::StepInput<fb::Nevpt2Step>{triplet, *calc->get<fb::BasisSetOutput>(),
                                    *calc->get<fb::EriOutput>(),
                                    *calc->get<fb::CoreHamiltonianOutput>(),
                                    *calc->get<fb::ScfOutput>(), *calc->get<fb::CasscfOutput>()},
      res);
  ASSERT_FALSE(not_singlet.has_value());
  EXPECT_EQ(not_singlet.error().kind, fb::ErrorKind::Unsupported);
  EXPECT_NE(not_singlet.error().message.find("[nevpt2] is implemented for a singlet only, and the "
                                             "molecule's multiplicity is 3"),
            std::string::npos)
      << not_singlet.error().message;

  // A frozen core past the CASSCF's (water's 3, CAS(4,4)), refused before
  // anything is queued; "auto" resolves to its O 1s and runs.
  const auto too_many = fb::Nevpt2Step{.frozen_core = 4}.run(
      fb::StepInput<fb::Nevpt2Step>{*calc->get<fb::MoleculeOutput>(),
                                    *calc->get<fb::BasisSetOutput>(), *calc->get<fb::EriOutput>(),
                                    *calc->get<fb::CoreHamiltonianOutput>(),
                                    *calc->get<fb::ScfOutput>(), *calc->get<fb::CasscfOutput>()},
      res);
  ASSERT_FALSE(too_many.has_value());
  EXPECT_EQ(too_many.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(too_many.error().message.find("[nevpt2] frozen_core is 4, and the CASSCF has 3 core "
                                          "orbitals"),
            std::string::npos)
      << too_many.error().message;
  const auto automatic = fb::Nevpt2Step{.frozen_core = fb::Nevpt2Step::kFrozenCoreAuto}.run(
      fb::StepInput<fb::Nevpt2Step>{*calc->get<fb::MoleculeOutput>(),
                                    *calc->get<fb::BasisSetOutput>(), *calc->get<fb::EriOutput>(),
                                    *calc->get<fb::CoreHamiltonianOutput>(),
                                    *calc->get<fb::ScfOutput>(), *calc->get<fb::CasscfOutput>()},
      res);
  ASSERT_TRUE(automatic.has_value()) << automatic.error().message;
  EXPECT_EQ(automatic->frozen_core, 1);

  // QD over this single-root CASSCF, the step driven without the config that
  // would have refused it.
  const auto one_root = fb::Nevpt2Step{.quasi_degenerate = true}.run(
      fb::StepInput<fb::Nevpt2Step>{*calc->get<fb::MoleculeOutput>(),
                                    *calc->get<fb::BasisSetOutput>(), *calc->get<fb::EriOutput>(),
                                    *calc->get<fb::CoreHamiltonianOutput>(),
                                    *calc->get<fb::ScfOutput>(), *calc->get<fb::CasscfOutput>()},
      res);
  ASSERT_FALSE(one_root.has_value());
  EXPECT_EQ(one_root.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(one_root.error().message.find("[nevpt2] quasi_degenerate couples the roots of a state "
                                          "average, and the CASSCF has 1 root"),
            std::string::npos)
      << one_root.error().message;
}

// QD-NEVPT2's bar on every SC and PC eigenvalue, the issue's ~1e-8, derived
// as kTolerance is. Each H_eff element is E_I δ_IJ + ½ (C_IJ + C_JI): E_I is
// our CASSCF root energy (a root's energy is not stationary in a state
// average) and each coupling is first order in the orbitals and the CI
// vectors as a class energy is. Per unit ||g||, measured between each H8
// case's unpolished golden (mc1step, ||g|| 1.7e-7 and 4.5e-7) and its
// polished copy: root energies 0.34, class energies 0.28, H_eff elements
// 0.20 (h8_sto3g_sa3; the ring 2e-3); H2CO's 0.07 is kTolerance's. Take 0.4:
// ours converged to kConvTolGrad and the goldens' polished to |g| < 1e-10,
// so an element is within 0.4 (1e-9 + 1e-10) ~ 4.4e-10, and ~1e-10 more from
// the CI vectors' residuals. An eigenvalue of a symmetric n x n matrix moves
// by at most ||δH||_2 <= n max |δH_IJ|, 1.6e-9 at n = 3; 1e-8 leaves a factor
// six. The unpolished goldens could not hold it: their own eigenvalues are
// 3.2e-8 (SC) and 2.9e-8 (PC) off the polished ones on h8_sto3g_sa3.
constexpr double kQdTolerance = 1e-8;
// The eigenvectors' bar: a component moves by about ||δH|| / gap, the
// smallest gap here H8's 19 mEh, so ~1e-7; 1e-6 is loose on purpose, since
// the vectors only check that the right states mix.
constexpr double kQdVectorTolerance = 1e-6;

// data/qd_nevpt2/qd_nevpt2.json's entry for @p name.
const fb::json &qd_reference(std::string_view name) {
  return reference_case("qd_nevpt2", "qd_nevpt2.json", name);
}

// data/qd_nevpt2/<name>/<array>.npy; column-major, as io.npy reads it.
std::vector<double> qd_array(std::string_view name, std::string_view array) {
  auto a = fb::read_npy(std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2" / name /
                        std::format("{}.npy", array));
  EXPECT_TRUE(a.has_value()) << name << "/" << array << ": " << (a ? "" : a.error().message);
  return a ? std::move(a->data) : std::vector<double>{};
}

// qd_nevpt2.json's entry as run.toml in @p dir: its molecule, basis and
// fitting set, and its state average in [casscf] from the Casci step's start,
// as the reference's starts from PySCF's default active space.
std::filesystem::path write_qd_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                      std::string_view nevpt2) {
  std::string xyz =
      std::format("{}\n{}\n", entry.at("atoms").size(), entry.at("name").get<std::string>());
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at(1).get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at(0).get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  const auto weights = entry.at("weights").get<std::vector<double>>();
  std::string weight_list;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weight_list += std::format("{}{:.17g}", i == 0 ? "" : ", ", weights[i]);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\nconv_tol_grad = {:g}\n{}",
      entry.at("basis").get<std::string>(), entry.at("aux_basis").at("name").get<std::string>(),
      entry.at("ncas").get<std::int32_t>(), entry.at("nelecas").get<std::int32_t>(), weights.size(),
      weight_list, kConvTolGrad, nevpt2);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// One QD variant against the goldens: each eigenvalue to kQdTolerance; the
// symmetrized H_eff's diagonal to it too, and its off-diagonal in magnitude
// (its sign is the two roots' CI phases'); each eigenvector's components in
// magnitude, for the same reason. The wanted H_eff is symmetric and the
// vectors column-major, state k's at [k n, (k + 1) n).
double expect_qd(std::string_view prefix, std::span<const double> heff,
                 std::span<const double> energies, std::span<const double> vectors,
                 std::span<const double> want_heff, std::span<const double> want_energies,
                 std::span<const double> want_vectors) {
  const std::size_t n = want_energies.size();
  EXPECT_EQ(energies.size(), n) << prefix;
  EXPECT_EQ(heff.size(), n * n) << prefix;
  EXPECT_EQ(vectors.size(), n * n) << prefix;
  if (energies.size() != n || heff.size() != n * n || vectors.size() != n * n ||
      want_heff.size() != n * n || want_vectors.size() != n * n) {
    return std::numeric_limits<double>::infinity();
  }
  double worst = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    EXPECT_NEAR(energies[k], want_energies[k], kQdTolerance) << prefix << "eigenvalue " << k;
    worst = std::max(worst, std::abs(energies[k] - want_energies[k]));
    for (std::size_t i = 0; i < n; ++i) {
      // Symmetric, so its storage order is moot; the vectors are columns in
      // both.
      const double h = heff[(i * n) + k];
      const double want = want_heff[(i * n) + k];
      if (i == k) {
        EXPECT_NEAR(h, want, kQdTolerance) << prefix << "heff " << i;
      } else {
        EXPECT_NEAR(std::abs(h), std::abs(want), kQdTolerance) << prefix << "heff " << i << k;
      }
      EXPECT_NEAR(std::abs(vectors[(k * n) + i]), std::abs(want_vectors[(k * n) + i]),
                  kQdVectorTolerance)
          << prefix << "vector " << k << " component " << i;
    }
  }
  return worst;
}

// std::string, not std::string_view: ctest names a case by printing it.
class PipelineQdNevpt2 : public ::testing::TestWithParam<std::string> {};

TEST_P(PipelineQdNevpt2, MatchesGoldens) {
  FBF_REQUIRE_GPU();
  // Each case is under a second on an RTX 3080, so all are in the fast tier.
  const std::string &name = GetParam();
  const fb::json &entry = qd_reference(name);
  const fb::test::ScratchDir dir("pipeline_qd_nevpt2");
  auto calc = fb::read_config(write_qd_config(dir, entry, "[nevpt2]\nquasi_degenerate = true\n"),
                              dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  ASSERT_TRUE(calc->step<fb::Nevpt2Step>()->quasi_degenerate);
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &casscf = calc->get<fb::CasscfOutput>();
  const auto &nevpt2 = calc->get<fb::Nevpt2Output>();
  ASSERT_TRUE(casscf.has_value() && nevpt2.has_value());
  EXPECT_LT(casscf->gradient_norm, kConvTolGrad);
  const auto want_casscf = entry.at("casscf_energies").get<std::vector<double>>();
  const std::size_t n = want_casscf.size();
  ASSERT_EQ(casscf->energies.size(), n);
  for (std::size_t r = 0; r < n; ++r) {
    EXPECT_NEAR(casscf->energies[r], want_casscf[r], kCasscfEnergyTolerance) << "root " << r;
  }

  // The QD diagonal is each root's SC on the shared orbitals: every class
  // against the goldens' class_energies, at the state-specific bar.
  ASSERT_EQ(nevpt2->qd_couplings.size(), n * n);
  const std::vector<double> class_energies = qd_array(name, "class_energies");
  ASSERT_EQ(class_energies.size(), n * 8);
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t k = 0; k < 8; ++k) {
      EXPECT_NEAR(nevpt2->qd_couplings[(r * n) + r].energies[k], class_energies[r + (n * k)],
                  kTolerance)
          << "root " << r << " class " << k;
    }
  }
  const double sc_worst = expect_qd("", nevpt2->qd_heff, nevpt2->qd_energies, nevpt2->qd_vectors,
                                    qd_array(name, "heff_sym"), qd_array(name, "eigenvalues"),
                                    qd_array(name, "eigenvectors"));
  EXPECT_TRUE(nevpt2->pc_qd_refusal.empty()) << nevpt2->pc_qd_refusal;
  ASSERT_EQ(nevpt2->pc_qd_couplings.size(), n * n);
  const double pc_worst =
      expect_qd("pc_", nevpt2->pc_qd_heff, nevpt2->pc_qd_energies, nevpt2->pc_qd_vectors,
                qd_array(name, "pc_heff_sym"), qd_array(name, "pc_eigenvalues"),
                qd_array(name, "pc_eigenvectors"));

  // The state-specific results are what they are without the option: where
  // nevpt2_df.json has the case, each root's SC E_corr against PySCF's.
  ASSERT_EQ(nevpt2->roots.size(), n);
  for (const fb::json &other : reference("nevpt2", "nevpt2_df.json").at("cases")) {
    if (other.at("name").get<std::string>() == name) {
      for (std::size_t r = 0; r < n; ++r) {
        EXPECT_NEAR(nevpt2->roots[r].total, other.at("roots")[r].at("e_corr").get<double>(),
                    kTolerance)
            << "root " << r << " state-specific E_corr";
      }
    }
  }
  std::cout << std::format("[ qd       ] {}: |g| {:.2e}, SC-QD {::.10f} (worst {:.2e}), PC-QD "
                           "{::.10f} (worst {:.2e}), tol {:.0e}\n",
                           name, casscf->gradient_norm, nevpt2->qd_energies, sc_worst,
                           nevpt2->pc_qd_energies, pc_worst, kQdTolerance);
}

// The skew case is C1, so no coupling vanishes; the two C2v ones couple
// roots of different symmetry, every off-diagonal zero. The two H8 cases
// (#688) are where QD visibly matters: data/qd_nevpt2/'s h8_sto3g_sa3 and
// h8_ring_sto3g_sa3, polished copies, whose couplings move the QD energies
// 1.7 and 6 mEh off the diagonal. PC solves both (#687).
INSTANTIATE_TEST_SUITE_P(Goldens, PipelineQdNevpt2,
                         ::testing::Values("h2co_skew_ccpvdz_sa3", "h2co_ccpvdz_sa2",
                                           "h2co_ccpvdz_sa2_weighted", "h8_sto3g_sa3_polished",
                                           "h8_ring_sto3g_sa3_polished"),
                         [](const auto &info) { return info.param; });

// Frozen-core QD (#859): [nevpt2] quasi_degenerate with frozen_core against
// data/qd_nevpt2/qd_nevpt2_frozen_core.json (qd_nevpt2_reference.py
// --frozen-core: the same states as qd_nevpt2.json's, the frozen orbitals
// sliced out of the fitted integrals, h1eff over the whole core), at
// MatchesGoldens' bars: the diagonal per class, SC's and PC's symmetrized
// H_eff, eigenvalues and eigenvectors. The skew H2CO is C1, every coupling
// live: "auto" (C and O 1s) and 1 (O 1s); the C2v one "auto".
struct QdFrozenCase {
  std::string name;
  std::string frozen_core;
};

void PrintTo(const QdFrozenCase &c, std::ostream *os) {
  // ctest names a case by this: "auto" unquoted, so `ctest -R` can select it.
  *os << c.name << "_" << (c.frozen_core.starts_with('"') ? "auto" : c.frozen_core);
}

class PipelineQdNevpt2FrozenCore : public ::testing::TestWithParam<QdFrozenCase> {};

TEST_P(PipelineQdNevpt2FrozenCore, MatchesReference) {
  FBF_REQUIRE_GPU();
  const QdFrozenCase &c = GetParam();
  const fb::json &entry = reference_case("qd_nevpt2", "qd_nevpt2_frozen_core.json", c.name);
  const fb::json *variant = nullptr;
  for (const fb::json &v : entry.at("variants")) {
    const fb::json &key = v.at("frozen_core");
    if ((key.is_string() ? std::format("\"{}\"", key.get<std::string>())
                         : std::to_string(key.get<std::int32_t>())) == c.frozen_core) {
      variant = &v;
    }
  }
  ASSERT_NE(variant, nullptr) << c.frozen_core;
  const fb::test::ScratchDir dir("pipeline_qd_nevpt2_frozen_core");
  auto calc = fb::read_config(
      write_qd_config(
          dir, entry,
          std::format("[nevpt2]\nquasi_degenerate = true\nfrozen_core = {}\n", c.frozen_core)),
      dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &casscf = calc->get<fb::CasscfOutput>();
  const auto &nevpt2 = calc->get<fb::Nevpt2Output>();
  ASSERT_TRUE(casscf.has_value() && nevpt2.has_value());
  EXPECT_LT(casscf->gradient_norm, kConvTolGrad);
  EXPECT_EQ(nevpt2->frozen_core, variant->at("n_frozen").get<std::int32_t>());
  const auto want_casscf = entry.at("casscf_energies").get<std::vector<double>>();
  const std::size_t n = want_casscf.size();
  ASSERT_EQ(casscf->energies.size(), n);
  for (std::size_t r = 0; r < n; ++r) {
    EXPECT_NEAR(casscf->energies[r], want_casscf[r], kCasscfEnergyTolerance) << "root " << r;
  }

  // The answers, flattened as expect_qd reads them: H_eff (symmetric) row
  // by row, the vectors state by state.
  const auto flat = [](const fb::json &rows) {
    std::vector<double> out;
    for (const fb::json &row : rows) {
      const auto values = row.get<std::vector<double>>();
      out.insert(out.end(), values.begin(), values.end());
    }
    return out;
  };
  const auto expect_variant = [&](std::string_view prefix, const fb::json &want,
                                  std::span<const fb::nevpt2::ClassEnergies> sc_couplings,
                                  std::span<const fb::nevpt2::PcClassEnergies> pc_couplings,
                                  std::span<const double> heff, std::span<const double> energies,
                                  std::span<const double> vectors) {
    const fb::json &classes = want.at("class_energies");
    for (std::size_t r = 0; r < n; ++r) {
      for (std::size_t k = 0; k < 8; ++k) {
        const double got = sc_couplings.empty() ? pc_couplings[(r * n) + r].energies[k]
                                                : sc_couplings[(r * n) + r].energies[k];
        EXPECT_NEAR(got, classes[r][k].get<double>(), kTolerance)
            << prefix << "root " << r << " class " << k;
      }
    }
    return expect_qd(prefix, heff, energies, vectors, flat(want.at("heff_sym")),
                     want.at("eigenvalues").get<std::vector<double>>(),
                     flat(want.at("eigenvectors")));
  };
  ASSERT_EQ(nevpt2->qd_couplings.size(), n * n);
  const double sc_worst = expect_variant("", variant->at("sc"), nevpt2->qd_couplings, {},
                                         nevpt2->qd_heff, nevpt2->qd_energies, nevpt2->qd_vectors);
  ASSERT_TRUE(variant->contains("pc")) << "the reference refused PC";
  EXPECT_TRUE(nevpt2->pc_qd_refusal.empty()) << nevpt2->pc_qd_refusal;
  ASSERT_EQ(nevpt2->pc_qd_couplings.size(), n * n);
  const double pc_worst =
      expect_variant("pc_", variant->at("pc"), {}, nevpt2->pc_qd_couplings, nevpt2->pc_qd_heff,
                     nevpt2->pc_qd_energies, nevpt2->pc_qd_vectors);
  std::cout << std::format("[ qd       ] {} frozen_core {}: |g| {:.2e}, SC-QD {::.10f} (worst "
                           "{:.2e}), PC-QD {::.10f} (worst {:.2e}), tol {:.0e}\n",
                           c.name, c.frozen_core, casscf->gradient_norm, nevpt2->qd_energies,
                           sc_worst, nevpt2->pc_qd_energies, pc_worst, kQdTolerance);
}

INSTANTIATE_TEST_SUITE_P(
    Reference, PipelineQdNevpt2FrozenCore,
    ::testing::Values(QdFrozenCase{.name = "h2co_skew_ccpvdz_sa3", .frozen_core = "\"auto\""},
                      QdFrozenCase{.name = "h2co_skew_ccpvdz_sa3", .frozen_core = "1"},
                      QdFrozenCase{.name = "h2co_ccpvdz_sa2", .frozen_core = "\"auto\""}),
    [](const auto &info) {
      return std::format("{}_{}", info.param.name,
                         info.param.frozen_core.starts_with('"') ? "auto" : info.param.frozen_core);
    });

// PC-QD through a metric spectrum with no gap at its cut (#853):
// PcAcceptedThroughAContinuum's HF CAS(10,7) averaged over three roots, its
// F 1s active in each, accepted where it was refused until #853. Three, not
// two: HF's lowest excited singlet is a 1Pi, so a second root is one
// component of a degenerate pair, and an average over it alone is not
// smooth where the pair meets (see ../../src/mcscf/solve/README.md, "A
// state average that splits a degenerate pair"): it took 23 to 44 macro
// iterations, and sometimes not 50 (#716). Over both components it is
// invariant under the rotation about the bond, and takes 6. Which basis the
// pair's CI vectors take is arbitrary, and QD-NEVPT2's eigenvalues depend on
// it (each state's own H_0), so nothing here compares them to a number: SC-QD
// and PC-QD are each held to what any basis gives.
TEST(PipelineQdNevpt2, PcQdAcceptedThroughAContinuum) {
  FBF_REQUIRE_GPU();
  constexpr std::size_t n = 3;
  const fb::test::ScratchDir dir("pipeline_qd_nevpt2");
  ASSERT_TRUE(
      fb::write_file(dir / "hf.xyz", "2\nhydrogen fluoride\nH 0 0 0\nF 0 0 0.917\n").has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"hf.xyz\"\n[basis_set]\nbasis = \"cc-pvdz\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
                  "[casscf]\nncas = 7\nnelecas = 10\nnroots = {}\nconv_tol_grad = {:g}\n"
                  "[nevpt2]\nquasi_degenerate = true\n",
                  kDfAuxBasis, n, kConvTolGrad);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &casscf = calc->get<fb::CasscfOutput>();
  const auto &nevpt2 = calc->get<fb::Nevpt2Output>();
  ASSERT_TRUE(casscf.has_value() && nevpt2.has_value());
  // 6 on the RTX 3080 in every run; mcscf_solve_gpu.cpp's bar.
  EXPECT_LE(casscf->iterations.size(), 20U);

  // SC-QD stands: H_eff symmetric, its diagonal each root's CASSCF energy
  // plus its own coupling, its eigenvalues summing to its trace.
  ASSERT_EQ(nevpt2->qd_couplings.size(), n * n);
  ASSERT_EQ(nevpt2->qd_heff.size(), n * n);
  ASSERT_EQ(nevpt2->qd_energies.size(), n);
  ASSERT_EQ(nevpt2->qd_vectors.size(), n * n);
  double trace = 0.0;
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t s = 0; s < r; ++s) {
      EXPECT_EQ(nevpt2->qd_heff[r + (s * n)], nevpt2->qd_heff[s + (r * n)]) << r << ", " << s;
    }
    EXPECT_NEAR(nevpt2->qd_heff[(r * n) + r],
                casscf->energies[r] + nevpt2->qd_couplings[(r * n) + r].total, kPcSlack);
    EXPECT_LT(nevpt2->qd_couplings[(r * n) + r].total, -0.05) << r;
    trace += nevpt2->qd_heff[(r * n) + r];
  }
  EXPECT_TRUE(std::ranges::is_sorted(nevpt2->qd_energies));
  EXPECT_NEAR(std::accumulate(nevpt2->qd_energies.begin(), nevpt2->qd_energies.end(), 0.0), trace,
              kPcSlack);

  // PC-QD stands, held as SC-QD is: H_eff symmetric, its diagonal each
  // root's CASSCF energy plus its own coupling, its eigenvalues ascending and
  // summing to its trace; each diagonal coupling at or below SC's.
  EXPECT_TRUE(nevpt2->pc_qd_refusal.empty()) << nevpt2->pc_qd_refusal;
  ASSERT_EQ(nevpt2->pc_qd_couplings.size(), n * n);
  ASSERT_EQ(nevpt2->pc_qd_heff.size(), n * n);
  ASSERT_EQ(nevpt2->pc_qd_energies.size(), n);
  ASSERT_EQ(nevpt2->pc_qd_vectors.size(), n * n);
  double pc_trace = 0.0;
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t s = 0; s < r; ++s) {
      EXPECT_EQ(nevpt2->pc_qd_heff[r + (s * n)], nevpt2->pc_qd_heff[s + (r * n)]) << r << ", " << s;
    }
    const double coupling = nevpt2->pc_qd_couplings[(r * n) + r].total;
    EXPECT_NEAR(nevpt2->pc_qd_heff[(r * n) + r], casscf->energies[r] + coupling, kPcSlack);
    EXPECT_LE(coupling, nevpt2->qd_couplings[(r * n) + r].total + kPcSlack) << r;
    pc_trace += nevpt2->pc_qd_heff[(r * n) + r];
  }
  EXPECT_TRUE(std::ranges::is_sorted(nevpt2->pc_qd_energies));
  EXPECT_NEAR(std::accumulate(nevpt2->pc_qd_energies.begin(), nevpt2->pc_qd_energies.end(), 0.0),
              pc_trace, kPcSlack);
  std::cout << std::format("[ qd       ] HF CAS(10,7) SA-3: {} macro iterations, |g| {:.2e}, SC-QD "
                           "{::.10f}; PC-QD {::.10f}\n",
                           casscf->iterations.size(), casscf->gradient_norm, nevpt2->qd_energies,
                           nevpt2->pc_qd_energies);
}

} // namespace
