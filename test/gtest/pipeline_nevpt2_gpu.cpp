// The Nevpt2 step against PySCF's SC-NEVPT2 (data/nevpt2/nevpt2_df.json,
// tools/emit/nevpt2_reference.py), through the real registry: a config naming
// [casscf], a density-fitted [eri] and [nevpt2] plans the whole chain, runs
// our own DF-CASSCF from each data/casscf/casscf_df.json start, then SC-NEVPT2
// per root. Compared per root: each class's norm and energy and their sum.
// N2 is not among the cases: SC-NEVPT2 depends on the arbitrary split of its
// degenerate pi* virtual pairs by up to 5e-6 (data/README.md
// "nevpt2/nevpt2_df.json"), so the reference leaves it out.
// Also the refusal of an integral-direct [eri], which needs no GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

static_assert(fb::PipelineStep<fb::Nevpt2Step>);
static_assert(fb::kRunShape<fb::Nevpt2Step> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::Nevpt2Step>,
                           fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::EriOutput,
                                    fb::CoreHamiltonianOutput, fb::CasscfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::Nevpt2Step>, fb::Nevpt2Output>);

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
// identical inputs; and nothing here is loosened for a case.
constexpr double kTolerance = 1e-9;

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

  const fb::json &roots = want.at("roots");
  const auto casscf_energies = want.at("casscf_energies").get<std::vector<double>>();
  ASSERT_EQ(nevpt2->roots.size(), roots.size());
  ASSERT_EQ(casscf->energies.size(), roots.size());
  for (std::size_t r = 0; r < roots.size(); ++r) {
    const fb::json &root = roots[r];
    const fb::json &classes = root.at("classes");
    ASSERT_EQ(classes.size(), 8U);
    EXPECT_NEAR(casscf->energies[r], casscf_energies[r], kCasscfEnergyTolerance) << "root " << r;
    const fb::nevpt2::ClassEnergies &got = nevpt2->roots[r];
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
    // What kTolerance's derivation rests on, printed for the record.
    std::cout << std::format("[ nevpt2   ] {} root {}: |g| {:.2e}, CASSCF dE {:+.2e}, "
                             "E_corr {:.12f} dE_corr {:+.2e}, worst class {:.2e} (tol {:.0e})\n",
                             c.name, r, casscf->gradient_norm,
                             casscf->energies[r] - casscf_energies[r], got.total,
                             got.total - e_corr, worst, kTolerance);
  }
}

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineNevpt2,
                         ::testing::Values(Case{.name = "h2o_ccpvdz", .slow = false},
                                           Case{.name = "h2co_ccpvdz_sa2", .slow = true},
                                           Case{.name = "h2co_ccpvdz_sa2_weighted", .slow = true},
                                           Case{.name = "h2o_ccpvdz_avas", .slow = true}),
                         [](const auto &info) { return std::string(info.param.name); });

// [nevpt2] over integral-direct ERIs, [eri] written so or filled in, is
// refused when the config is read, before the SCF and CASSCF it would follow.
// Nothing runs, so no GPU is needed.
TEST(PipelineNevpt2, IntegralDirectEriIsUnsupported) {
  const fb::test::ScratchDir dir("pipeline_nevpt2");
  for (const std::string_view eri : {"", "[eri]\nmode = \"integral_direct\"\n"}) {
    const std::string toml =
        std::format("[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"cc-pvdz\"\n"
                    "[casscf]\nncas = 4\nnelecas = 4\n{}[nevpt2]\n",
                    eri);
    const auto calc = fb::parse_config(toml, dir.path(), dir / "out");
    ASSERT_FALSE(calc.has_value()) << eri;
    EXPECT_EQ(calc.error().kind, fb::ErrorKind::Unsupported) << calc.error().message;
    EXPECT_NE(calc.error().message.find("[eri]"), std::string::npos) << calc.error().message;
  }
}

} // namespace
