// The Cis step against PySCF's TDA (data/cis/cis.json,
// tools/emit/cis_reference.py) to 1e-6 hartree: H2O in STO-3G and 6-31G
// exact, and H2O in cc-pVDZ density-fitted over cc-pVDZ-JKFIT against PySCF's
// DF RHF + TDA over the same set (cis::DfSigma), singlets and triplets, each
// through the real registry (a config naming [cis] plans the chain to the
// ScfOutput and the EriOutput) and through the step run again by hand over
// the same products. Each state's amplitudes are checked unit-norm and
// orthogonal to the others', each singlet's oscillator strength against the
// reference's to 1e-6 (H2O cc-pVDZ exact too, singlets only), and each
// triplet's f and transition dipole zero. HF in cc-pVTZ, past the dipole's d,
// still gives its energies, with f and the dipoles zero: against the
// reference integral-direct (slow, ~55 s, so skipped in fast mode), and
// density-fitted over cc-pVTZ-JKFIT in the fast set. Then the refusals: too few or too many states,
// a molecule that is not a closed-shell singlet, and an open-shell
// ScfOutput; and [cis] under a density-fitted [eri] planned when the config
// is read. Runs the device chain, so needs a GPU, but for the config case.

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
namespace cis = free_breakfast::cis;

static_assert(fb::PipelineStep<fb::CisStep>);
static_assert(fb::kRunShape<fb::CisStep> == fb::RunShape::Device);
static_assert(
    std::same_as<fb::StepInput<fb::CisStep>,
                 fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                          fb::PrimitivePairClassesOutput, fb::EriOutput, fb::ScfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::CisStep>, fb::CisOutput>);

// The issue's bar against cis.json's PySCF roots, which are good to ~1e-14,
// for the excitation energies and the reference energy.
constexpr double kEnergyTolerance = 1e-6;
// The Ritz vectors are orthonormal to the Davidson's own rounding.
constexpr double kOrthonormalTolerance = 1e-10;
// The issue's bar against cis.json's oscillator strengths, which PySCF's own
// and the dense A's eigenvectors agree on to 2e-11 or better here.
constexpr double kOscillatorTolerance = 1e-6;
// f is (2/3) ω |μ|² of the state's own ω and μ, to rounding.
constexpr double kRoundingTolerance = 1e-14;

// The SCF converged tightly, so the reference's error is not the 1e-6 the
// excitations are held to; integral-direct, every Fock build full too, which
// a fitted [eri] refuses to be told (its every build is full).
constexpr std::string_view kScf = "[scf]\nenergy_tolerance = 1e-11\ncommutator_tolerance = 1e-10\n";
constexpr std::string_view kDirectScf = "fock_rebuild_interval = 1\n";

const fb::json &reference_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "cis" / "cis.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in cis/cis.json";
  static const fb::json kNone;
  return kNone;
}

// The reference's molecule written as an XYZ file in dir, and a config naming
// it in its basis, fitted over the reference's aux basis when it has one,
// with the given tables after it.
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
  const fb::json &aux = entry.at("aux_basis");
  const std::string eri =
      aux.is_null() ? std::string{}
                    : std::format("[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
                                  aux.get<std::string>());
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n{}{}{}{}",
      entry.at("basis").get<std::string>(), eri, kScf, aux.is_null() ? kDirectScf : "", tables);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The step's input from a run calculation's products, the ScfOutput and the
// MoleculeOutput replaceable.
fb::StepInput<fb::CisStep> step_input(const fb::Calculation &calc,
                                      const fb::MoleculeOutput *molecule = nullptr,
                                      const fb::ScfOutput *scf = nullptr) {
  return fb::StepInput<fb::CisStep>{molecule != nullptr ? *molecule
                                                        : *calc.get<fb::MoleculeOutput>(),
                                    *calc.get<fb::BasisSetOutput>(),
                                    *calc.get<fb::SphericalTransformOutput>(),
                                    *calc.get<fb::PrimitivePairClassesOutput>(),
                                    *calc.get<fb::EriOutput>(),
                                    scf != nullptr ? *scf : *calc.get<fb::ScfOutput>()};
}

// The reference's roots of one spin: the excitation energies to 1e-6, the
// shape, and the amplitudes orthonormal. With the reference's oscillator
// strengths, each group's summed f to 1e-6 and every |μ|² that f; without
// them (triplets, and singlets past the dipole's d), f and μ exactly zero.
void expect_matches(const fb::CisOutput &out, const fb::json &entry, const fb::json &roots,
                    std::string_view what) {
  const auto want = roots.at("energies").get<std::vector<double>>();
  const auto n_occ = entry.at("occupied").get<std::int32_t>();
  const auto n_virt = entry.at("nao").get<std::int32_t>() - n_occ;
  EXPECT_EQ(out.n_occ, n_occ) << what;
  EXPECT_EQ(out.n_virt, n_virt) << what;
  EXPECT_GE(out.iterations, 1) << what;
  EXPECT_NEAR(out.reference_energy, entry.at("rhf_energy").get<double>(), kEnergyTolerance) << what;
  ASSERT_EQ(out.states.size(), want.size()) << what;
  const auto n_ov = static_cast<std::size_t>(n_occ) * static_cast<std::size_t>(n_virt);
  for (std::size_t r = 0; r < want.size(); ++r) {
    const fb::CisState &state = out.states[r];
    std::cout << what << " state " << r << ": " << state.excitation_energy - want[r]
              << " hartree\n";
    EXPECT_NEAR(state.excitation_energy, want[r], kEnergyTolerance) << what << " state " << r;
    if (r > 0) {
      EXPECT_LE(out.states[r - 1].excitation_energy, state.excitation_energy) << what;
    }
    if (roots.contains("oscillator_strengths")) {
      const auto &mu = state.transition_dipole;
      EXPECT_NEAR(state.oscillator_strength,
                  2.0 / 3.0 * state.excitation_energy *
                      ((mu[0] * mu[0]) + (mu[1] * mu[1]) + (mu[2] * mu[2])),
                  kRoundingTolerance)
          << what << " state " << r;
    } else {
      EXPECT_EQ(state.oscillator_strength, 0.0) << what << " state " << r;
      EXPECT_EQ(state.transition_dipole, (std::array<double, 3>{0.0, 0.0, 0.0}))
          << what << " state " << r;
    }
    ASSERT_EQ(state.amplitudes.size(), n_ov) << what << " state " << r;
    for (std::size_t s = 0; s <= r; ++s) {
      double overlap = 0.0;
      for (std::size_t k = 0; k < n_ov; ++k) {
        overlap += state.amplitudes[k] * out.states[s].amplitudes[k];
      }
      EXPECT_NEAR(overlap, r == s ? 1.0 : 0.0, kOrthonormalTolerance)
          << what << " states " << r << ", " << s;
    }
  }
  // A degenerate group's single f is undefined, its sum is not.
  if (roots.contains("oscillator_strengths")) {
    const auto f = roots.at("oscillator_strengths").get<std::vector<double>>();
    const fb::json &groups = roots.at("groups");
    ASSERT_EQ(groups.size(), f.size()) << what;
    for (std::size_t g = 0; g < f.size(); ++g) {
      double sum = 0.0;
      for (const fb::json &k : groups[g]) {
        sum += out.states[k.get<std::size_t>()].oscillator_strength;
      }
      std::cout << what << " group " << g << ": f " << sum - f[g] << "\n";
      EXPECT_NEAR(sum, f[g], kOscillatorTolerance) << what << " group " << g;
    }
  }
}

struct Case {
  std::string_view name;
  std::string_view spin;
};

// ctest names a discovered case by this, so `ctest -R h2o_631g` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name << "_" << c.spin;
}

class CisStepReference : public ::testing::TestWithParam<Case> {};

// The config runs the chain and the step; then the step runs again by hand
// over the same products, from a CisStep built in code. The h2o_ccpvdz_df
// cases run the chain fitted, so the step takes cis::DfSigma.
TEST_P(CisStepReference, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  // Exact K at f: ~55 s on the 3080. The fitted f-shell case below stands
  // for it in the fast set.
  if (GetParam().name == "hf_ccpvtz") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &entry = reference_case(GetParam().name);
  const std::string spin{GetParam().spin};
  const fb::json &roots = entry.at(spin == "triplet" ? "triplets" : "singlets");
  const auto n_states = static_cast<std::int32_t>(roots.at("energies").size());
  const fb::test::ScratchDir dir("cis_step");
  auto calc = fb::read_config(
      write_config(dir, entry,
                   std::format("[cis]\nn_states = {}\nspin = \"{}\"\n", n_states, spin)),
      dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  ASSERT_EQ(calc->step<fb::CisStep>()->options.n_states, n_states);
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &chain = calc->get<fb::CisOutput>();
  ASSERT_TRUE(chain.has_value());
  expect_matches(*chain, entry, roots, "config " + spin);

  fb::DeviceResources &res = node.device(0);
  const fb::CisStep step{
      .options = {.n_states = n_states,
                  .spin = spin == "triplet" ? cis::Spin::Triplet : cis::Spin::Singlet},
  };
  const auto by_hand = step.run(step_input(*calc), res);
  ASSERT_TRUE(by_hand.has_value()) << by_hand.error().message;
  expect_matches(*by_hand, entry, roots, "by hand " + spin);
}

INSTANTIATE_TEST_SUITE_P(Pyscf, CisStepReference,
                         ::testing::Values(Case{.name = "h2o_sto3g", .spin = "singlet"},
                                           Case{.name = "h2o_sto3g", .spin = "triplet"},
                                           Case{.name = "h2o_631g", .spin = "singlet"},
                                           Case{.name = "h2o_631g", .spin = "triplet"},
                                           Case{.name = "h2o_ccpvdz", .spin = "singlet"},
                                           Case{.name = "h2o_ccpvdz_df", .spin = "singlet"},
                                           Case{.name = "h2o_ccpvdz_df", .spin = "triplet"},
                                           Case{.name = "hf_ccpvtz", .spin = "singlet"}),
                         [](const auto &info) {
                           return std::format("{}_{}", info.param.name, info.param.spin);
                         });

// Past the dipole kernels' d (fluorine's cc-pVTZ f shell), the step still
// returns the singlets, with every f and μ zero rather than an error: HF in
// cc-pVTZ fitted over cc-pVTZ-JKFIT, the cheap way to an f shell. Its roots
// have no fitted reference, so only their shape is checked.
TEST(CisStep, PastTheDipolesDTheEnergiesStillReturn) {
  FBF_REQUIRE_GPU();
  fb::json entry = reference_case("hf_ccpvtz");
  entry["aux_basis"] = "cc-pvtz-jkfit";
  const fb::test::ScratchDir dir("cis_step");
  auto calc =
      fb::read_config(write_config(dir, entry, "[cis]\nn_states = 3\n"), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &out = calc->get<fb::CisOutput>();
  ASSERT_TRUE(out.has_value());
  ASSERT_EQ(out->states.size(), 3U);
  for (std::size_t r = 0; r < out->states.size(); ++r) {
    const fb::CisState &state = out->states[r];
    EXPECT_GT(state.excitation_energy, 0.0) << "state " << r;
    if (r > 0) {
      EXPECT_LE(out->states[r - 1].excitation_energy, state.excitation_energy);
    }
    EXPECT_EQ(state.oscillator_strength, 0.0) << "state " << r;
    EXPECT_EQ(state.transition_dipole, (std::array<double, 3>{0.0, 0.0, 0.0})) << "state " << r;
  }
}

// Every state there is, n_states = n_ov = 10 in H2O / STO-3G: the subspace
// clamped up to 2 n_states past n_ov, and the guess the whole space; and a
// max_subspace of 1 clamped up to 2 n_states. Both still the reference's
// singlets.
TEST(CisStep, TheSubspaceIsClamped) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_sto3g");
  const fb::test::ScratchDir dir("cis_step");
  auto calc = fb::read_config(write_config(dir, entry, ""), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  fb::DeviceResources &res = node.device(0);
  const auto want = entry.at("singlets").at("energies").get<std::vector<double>>();

  const auto every = fb::CisStep{.options = {.n_states = 10}}.run(step_input(*calc), res);
  ASSERT_TRUE(every.has_value()) << every.error().message;
  ASSERT_EQ(every->states.size(), 10U);
  for (std::size_t r = 0; r < want.size(); ++r) {
    EXPECT_NEAR(every->states[r].excitation_energy, want[r], kEnergyTolerance) << "state " << r;
  }

  const auto narrow =
      fb::CisStep{.options = {.n_states = 3}, .max_subspace = 1}.run(step_input(*calc), res);
  ASSERT_TRUE(narrow.has_value()) << narrow.error().message;
  ASSERT_EQ(narrow->states.size(), 3U);
  for (std::size_t r = 0; r < 3; ++r) {
    EXPECT_NEAR(narrow->states[r].excitation_energy, want[r], kEnergyTolerance) << "state " << r;
  }
}

void expect_refused(const fb::Result<fb::CisOutput> &refused, fb::ErrorKind kind,
                    std::string_view needle) {
  ASSERT_FALSE(refused.has_value()) << needle;
  EXPECT_EQ(refused.error().kind, kind) << refused.error().message;
  EXPECT_NE(refused.error().message.find(needle), std::string::npos) << refused.error().message;
}

// Each refused when the step runs, naming the step, before anything is
// allocated: n_states below 1 or past H2O / STO-3G's n_ov = 10; a triplet
// molecule; an odd electron count; an ScfOutput whose electrons are not the
// molecule's closed shell.
TEST(CisStep, Refusals) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_sto3g");
  const fb::test::ScratchDir dir("cis_step");
  auto calc = fb::read_config(write_config(dir, entry, ""), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};

  expect_refused(fb::CisStep{.options = {.n_states = 0}}.run(step_input(*calc), res),
                 fb::ErrorKind::InvalidConfig, "cis: n_states is 0, not at least 1");
  expect_refused(fb::CisStep{.options = {.n_states = 11}}.run(step_input(*calc), res),
                 fb::ErrorKind::InvalidConfig,
                 "cis: n_states is 11, past the 10 single excitations");

  const fb::Molecule &singlet = calc->get<fb::MoleculeOutput>()->molecule;
  const fb::MoleculeOutput triplet{fb::Molecule(res, singlet.atoms(), singlet.charge(), 3)};
  expect_refused(fb::CisStep{}.run(step_input(*calc, &triplet), res), fb::ErrorKind::InvalidConfig,
                 "cis: needs a closed-shell singlet reference, and the molecule has 10 electrons "
                 "at multiplicity 3");
  const fb::MoleculeOutput cation{fb::Molecule(res, singlet.atoms(), 1, 1)};
  expect_refused(fb::CisStep{}.run(step_input(*calc, &cation), res), fb::ErrorKind::InvalidConfig,
                 "the molecule has 9 electrons at multiplicity 1");

  const fb::BasisSet &basis = calc->get<fb::BasisSetOutput>()->basis;
  const fb::ScfOutput open{.energy = 0.0,
                           .iterations = 0,
                           .n_alpha = 6,
                           .n_beta = 4,
                           .mo_energies = {},
                           .c = fb::SphericalMatrixAO(res, basis),
                           .d = fb::SphericalMatrixAO(res, basis)};
  expect_refused(fb::CisStep{}.run(step_input(*calc, nullptr, &open), res),
                 fb::ErrorKind::InvalidConfig, "the ScfOutput has 6 alpha and 4 beta");
}

// [cis] under a density-fitted [eri] is planned when the config is read: the
// Cis step after an Eri step in fitting mode over the named aux basis, with
// nothing refused. Nothing runs, so no GPU is needed.
TEST(CisStep, DensityFittedEriIsPlanned) {
  const fb::test::ScratchDir dir("cis_step");
  const auto calc =
      fb::parse_config("[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"cc-pvdz\"\n"
                       "[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n[cis]\n",
                       dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  ASSERT_TRUE(calc->step<fb::CisStep>().has_value());
  const auto &eri = calc->step<fb::EriStep>();
  ASSERT_TRUE(eri.has_value());
  EXPECT_EQ(eri->mode, fb::EriMode::DensityFitting);
  EXPECT_EQ(eri->aux_basis, "cc-pvdz-jkfit");
}

} // namespace
