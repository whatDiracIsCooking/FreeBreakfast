// Host-only cases for the TOML front end: a config document -> the pipeline
// it asks for, with path resolution, filled-in producers, and every way a
// config can be refused. Nothing is run, so nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

constexpr std::string_view kWater = R"(
[molecule]
xyz = "water.xyz"
charge = -1
multiplicity = 2

[basis_set]
basis = "/basis/sto-3g.json"

[orthogonalizer]
eigenvalue_floor = 1e-8

[guess]
type = "sad"
)";

// The two tables every config that reaches a basis needs.
constexpr std::string_view kInputs =
    "[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"b.json\"\n";

std::string with_inputs(std::string_view tables) {
  return std::string(kInputs).append(tables);
}

void expect_config_error(std::string_view toml, std::string_view needle) {
  auto calc = fb::parse_config(toml, "");
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(calc.error().message.find(needle), std::string::npos) << calc.error().message;
}

TEST(PipelineConfig, EveryOptionIsRead) {
  auto calc = fb::parse_config(kWater, "/inputs");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::MoleculeStep &molecule = *calc->step<fb::MoleculeStep>();
  EXPECT_EQ(molecule.xyz, std::filesystem::path("/inputs/water.xyz"));
  EXPECT_EQ(molecule.charge, -1);
  EXPECT_EQ(molecule.multiplicity, 2);
  // An absolute path is kept as written.
  EXPECT_EQ(calc->step<fb::BasisSetStep>()->basis, "/basis/sto-3g.json");
  EXPECT_EQ(calc->step<fb::OrthogonalizerStep>()->eigenvalue_floor, 1e-8);
  EXPECT_EQ(calc->step<fb::GuessStep>()->type, fb::GuessType::Sad);
}

TEST(PipelineConfig, ARelativeBasisPathIsAnchoredToTheConfig) {
  auto calc =
      fb::parse_config("[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"b.json\"\n", "/inputs");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::BasisSetStep>()->basis, "/inputs/b.json");
}

TEST(PipelineConfig, ABuiltinBasisNameIsNotAPath) {
  // The name of a compiled-in set must survive the front end verbatim: joined
  // to base_dir it would become /inputs/sto-3g, a file nobody wrote.
  for (const std::string_view spelling : {"sto-3g", "STO-3G", "sto3g", "cc-pVTZ"}) {
    auto calc = fb::parse_config(
        std::format("[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"{}\"\n", spelling),
        "/inputs");
    ASSERT_TRUE(calc.has_value()) << spelling << ": " << calc.error().message;
    EXPECT_EQ(calc->step<fb::BasisSetStep>()->basis, spelling);
  }
}

TEST(PipelineConfig, AbsentOptionsKeepTheStepDefaults) {
  auto calc = fb::parse_config(with_inputs("[orthogonalizer]\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::MoleculeStep defaults{};
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->charge, defaults.charge);
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->multiplicity, defaults.multiplicity);
  EXPECT_EQ(calc->step<fb::OrthogonalizerStep>()->eigenvalue_floor,
            fb::OrthogonalizerStep{}.eigenvalue_floor);
  // No base directory: relative to the working directory.
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->xyz, std::filesystem::path("w.xyz"));
}

TEST(PipelineConfig, IntegerReadsAsDouble) {
  auto calc = fb::parse_config(with_inputs("[orthogonalizer]\neigenvalue_floor = 0\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::OrthogonalizerStep>()->eigenvalue_floor, 0.0);
}

TEST(PipelineConfig, ATableIsARequestAndItsProducersAreFilledIn) {
  auto calc = fb::parse_config(with_inputs("[core_hamiltonian]\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_TRUE(calc->step<fb::CoreHamiltonianStep>().has_value());
  EXPECT_TRUE(calc->step<fb::SphericalTransformStep>().has_value());
  EXPECT_TRUE(calc->step<fb::PrimitivePairClassesStep>().has_value());
  // Not upstream of anything asked for.
  EXPECT_FALSE(calc->step<fb::OverlapStep>().has_value());
  EXPECT_FALSE(calc->step<fb::OrthogonalizerStep>().has_value());
}

TEST(PipelineConfig, TheGuessNeedsNoDeviceStepPastTheMolecule) {
  auto calc = fb::parse_config(with_inputs("[guess]\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::GuessStep>()->type, fb::GuessStep{}.type);
  EXPECT_FALSE(calc->step<fb::OverlapStep>().has_value());
  EXPECT_FALSE(calc->step<fb::PrimitivePairClassesStep>().has_value());
}

TEST(PipelineConfig, TheScfFillsInTheWholeChain) {
  auto calc = fb::parse_config(with_inputs("[scf]\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_TRUE(calc->step<fb::OverlapStep>().has_value());
  EXPECT_TRUE(calc->step<fb::OrthogonalizerStep>().has_value());
  EXPECT_TRUE(calc->step<fb::CoreHamiltonianStep>().has_value());
  EXPECT_TRUE(calc->step<fb::GuessStep>().has_value());
  const fb::ScfStep defaults{};
  const fb::ScfStep &scf = *calc->step<fb::ScfStep>();
  EXPECT_EQ(scf.max_iterations, defaults.max_iterations);
  EXPECT_EQ(scf.energy_tolerance, defaults.energy_tolerance);
  EXPECT_EQ(scf.commutator_tolerance, defaults.commutator_tolerance);
  EXPECT_EQ(scf.diis_history, defaults.diis_history);
  EXPECT_EQ(scf.fock_rebuild_interval, defaults.fock_rebuild_interval);
  EXPECT_TRUE(scf.molden.empty());
}

TEST(PipelineConfig, EveryScfOptionIsRead) {
  auto calc = fb::parse_config(with_inputs("[scf]\nmax_iterations = 40\nenergy_tolerance = 1e-8\n"
                                           "commutator_tolerance = 1e-5\ndiis_history = 6\n"
                                           "fock_rebuild_interval = 1\nmolden = \"w.molden\"\n"),
                               "/run");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::ScfStep &scf = *calc->step<fb::ScfStep>();
  EXPECT_EQ(scf.max_iterations, 40);
  EXPECT_EQ(scf.energy_tolerance, 1e-8);
  EXPECT_EQ(scf.commutator_tolerance, 1e-5);
  EXPECT_EQ(scf.diis_history, 6);
  EXPECT_EQ(scf.fock_rebuild_interval, 1);
  EXPECT_EQ(scf.molden, std::filesystem::path("/run/w.molden"));
}

TEST(PipelineConfig, AMisspeltScfOptionIsInvalidConfig) {
  expect_config_error(with_inputs("[scf]\nmax_iter = 40\n"), "[scf] has no option 'max_iter'");
  expect_config_error(with_inputs("[scf]\nmax_iterations = 1.5\n"), "[scf] max_iterations");
}

TEST(PipelineConfig, OnlyWhatWasAskedForIsPlanned) {
  auto calc = fb::parse_config("[molecule]\nxyz = \"w.xyz\"\n", "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_TRUE(calc->step<fb::MoleculeStep>().has_value());
  EXPECT_FALSE(calc->step<fb::BasisSetStep>().has_value());
}

TEST(PipelineConfig, AFilledInStepMissingARequiredOptionIsInvalidConfig) {
  // [overlap] needs a basis, and [basis_set] cannot be read from nothing.
  expect_config_error("[molecule]\nxyz = \"w.xyz\"\n[overlap]\n", "[basis_set] needs 'basis'");
  expect_config_error("[basis_set]\nbasis = \"b.json\"\n", "[molecule] needs 'xyz'");
}

TEST(PipelineConfig, MissingRequiredOptionIsInvalidConfig) {
  expect_config_error("[molecule]\ncharge = 0\n[basis_set]\nbasis = \"b.json\"\n", "'xyz'");
}

TEST(PipelineConfig, AConfigNamingNoStepIsInvalidConfig) {
  expect_config_error("", "names no step");
}

TEST(PipelineConfig, UnknownStepIsInvalidConfig) {
  expect_config_error(with_inputs("[mp2]\nfrozen_core = true\n"), "unknown step [mp2]");
}

TEST(PipelineConfig, AStepThatIsNotATableIsInvalidConfig) {
  expect_config_error("molecule = \"w.xyz\"\n", "write it as [molecule]");
}

TEST(PipelineConfig, MisspeltOptionIsInvalidConfig) {
  expect_config_error(
      "[molecule]\nxyz = \"w.xyz\"\nmultiplicty = 3\n[basis_set]\nbasis = \"b.json\"\n",
      "'multiplicty'");
  expect_config_error(with_inputs("[orthogonalizer]\nfloor = 1e-8\n"), "'floor'");
  // A step with no options has no keys to give.
  expect_config_error(with_inputs("[overlap]\nscreen = 1e-12\n"),
                      "[overlap] has no option 'screen'");
}

TEST(PipelineConfig, AnUnknownGuessTypeIsInvalidConfig) {
  expect_config_error(with_inputs("[guess]\ntype = \"core\"\n"), "[guess] has no type 'core'");
  expect_config_error(with_inputs("[guess]\ntype = \"SAD\"\n"), "[guess] has no type 'SAD'");
}

TEST(PipelineConfig, TheLocalizationModeAndItsOptionsAreRead) {
  auto calc =
      fb::parse_config(with_inputs("[localization]\nmode = \"pipek_mezey\"\n"
                                   "max_sweeps = 7\nexponent = 4\nmolden = \"pm.molden\"\n"),
                       "/run");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::LocalizationStep &step = *calc->step<fb::LocalizationStep>();
  EXPECT_EQ(step.mode, fb::LocalizationMode::PipekMezey);
  EXPECT_EQ(step.pipek_mezey.max_sweeps, 7);
  EXPECT_EQ(step.pipek_mezey.exponent, fb::PipekMezeyExponent::Four);
  EXPECT_EQ(step.molden, std::filesystem::path("/run/pm.molden"));
  // An empty table is Pipek-Mezey with its defaults.
  auto plain = fb::parse_config(with_inputs("[localization]\n"), "");
  ASSERT_TRUE(plain.has_value()) << plain.error().message;
  EXPECT_EQ(plain->step<fb::LocalizationStep>()->mode, fb::LocalizationMode::PipekMezey);
  EXPECT_EQ(plain->step<fb::LocalizationStep>()->pipek_mezey.exponent, fb::PipekMezeyExponent::Two);
  EXPECT_TRUE(plain->step<fb::LocalizationStep>()->molden.empty());
}

TEST(PipelineConfig, AnUnknownLocalizationModeIsInvalidConfig) {
  expect_config_error(with_inputs("[localization]\nmode = \"foster_boys\"\n"),
                      "[localization] has no mode 'foster_boys'");
  expect_config_error(with_inputs("[localization]\nmode = 1\n"), "[localization] mode");
  expect_config_error(with_inputs("[localization]\nexponent = 3\n"),
                      "[localization] exponent is 3");
}

TEST(PipelineConfig, EveryCasciOptionIsReadAndTheScfIsFilledIn) {
  auto calc = fb::parse_config(
      with_inputs("[casci]\nncas = 4\nnelecas = 4\nnroots = 3\nactive = [2, 4, 5, 7]\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::CasciStep &casci = *calc->step<fb::CasciStep>();
  EXPECT_EQ(casci.ncas, 4);
  EXPECT_EQ(casci.nelecas, 4);
  EXPECT_EQ(casci.nroots, 3);
  EXPECT_EQ(casci.active, (std::vector<std::int32_t>{2, 4, 5, 7}));
  EXPECT_TRUE(calc->step<fb::ScfStep>().has_value());
  EXPECT_TRUE(calc->step<fb::CoreHamiltonianStep>().has_value());
  EXPECT_FALSE(calc->step<fb::LocalizationStep>().has_value());

  auto plain = fb::parse_config(with_inputs("[casci]\nncas = 2\nnelecas = 2\n"), "");
  ASSERT_TRUE(plain.has_value()) << plain.error().message;
  EXPECT_EQ(plain->step<fb::CasciStep>()->nroots, 1);
  EXPECT_TRUE(plain->step<fb::CasciStep>()->active.empty());
}

TEST(PipelineConfig, AMalformedCasciTableIsInvalidConfig) {
  expect_config_error(with_inputs("[casci]\nnelecas = 4\n"), "[casci] needs 'ncas'");
  expect_config_error(with_inputs("[casci]\nncas = 4\n"), "[casci] needs 'nelecas'");
  expect_config_error(with_inputs("[casci]\nncas = 4\nnelecas = 4\nnroot = 2\n"),
                      "[casci] has no option 'nroot'");
  expect_config_error(with_inputs("[casci]\nncas = 4.5\nnelecas = 4\n"), "[casci] ncas");
  expect_config_error(with_inputs("[casci]\nncas = 2\nnelecas = 2\nactive = 3\n"),
                      "[casci] active is not an array");
  expect_config_error(with_inputs("[casci]\nncas = 2\nnelecas = 2\nactive = [3, 4.0]\n"),
                      "[casci] active is not an array of 32-bit integers");
}

// The ranges need the molecule's electrons and the basis's orbitals, so
// orbital_order checks them when the step runs: here, water's 10 electrons
// over STO-3G's 7 orbitals.
TEST(PipelineConfig, TheCasciOrbitalOrderPutsCoreActiveRest) {
  const auto order = [](std::string_view table) {
    auto calc = fb::parse_config(with_inputs(table), "");
    EXPECT_TRUE(calc.has_value()) << calc.error().message;
    return calc->step<fb::CasciStep>()->orbital_order(10, 7);
  };
  auto frontier = order("[casci]\nncas = 4\nnelecas = 4\n");
  ASSERT_TRUE(frontier.has_value()) << frontier.error().message;
  EXPECT_EQ(*frontier, (std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 6}));
  auto explicit_set = order("[casci]\nncas = 2\nnelecas = 2\nactive = [5, 1]\n");
  ASSERT_TRUE(explicit_set.has_value()) << explicit_set.error().message;
  EXPECT_EQ(*explicit_set, (std::vector<std::int32_t>{0, 2, 3, 4, 5, 1, 6}));
  auto all = order("[casci]\nncas = 7\nnelecas = 10\n");
  ASSERT_TRUE(all.has_value()) << all.error().message;
  EXPECT_EQ(*all, (std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 6}));
}

TEST(PipelineConfig, AnImpossibleCasciActiveSpaceIsRefusedByOrbitalOrder) {
  const auto expect_refused = [](std::string_view table, std::string_view needle) {
    auto calc = fb::parse_config(with_inputs(table), "");
    ASSERT_TRUE(calc.has_value()) << calc.error().message;
    const auto order = calc->step<fb::CasciStep>()->orbital_order(10, 7);
    ASSERT_FALSE(order.has_value());
    EXPECT_EQ(order.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(order.error().message.find(needle), std::string::npos) << order.error().message;
  };
  // ncore + ncas > nmo: 3 core and 5 active over 7.
  expect_refused("[casci]\nncas = 5\nnelecas = 4\n", "ncore 3 + ncas 5 exceeds the 7 orbitals");
  // nelecas > 2 ncas.
  expect_refused("[casci]\nncas = 2\nnelecas = 6\n", "nelecas in [0, min(2 ncas, 10)]");
  expect_refused("[casci]\nncas = 0\nnelecas = 0\n", "ncas >= 1");
  // An odd core electron count.
  expect_refused("[casci]\nncas = 4\nnelecas = 3\n", "odd core electron count");
  // A bad or duplicate index, or the wrong number of them.
  expect_refused("[casci]\nncas = 2\nnelecas = 2\nactive = [4, 7]\n",
                 "active orbital 7 is not in [0, 7)");
  expect_refused("[casci]\nncas = 2\nnelecas = 2\nactive = [-1, 4]\n",
                 "active orbital -1 is not in [0, 7)");
  expect_refused("[casci]\nncas = 2\nnelecas = 2\nactive = [4, 4]\n",
                 "active orbital 4 is named twice");
  expect_refused("[casci]\nncas = 2\nnelecas = 2\nactive = [4, 5, 6]\n",
                 "active names 3 orbitals, not ncas 2");
}

TEST(PipelineConfig, WrongTypedOptionIsInvalidConfigNamingIt) {
  expect_config_error(
      "[molecule]\nxyz = \"w.xyz\"\ncharge = 1.5\n[basis_set]\nbasis = \"b.json\"\n",
      "[molecule] charge");
  expect_config_error("[molecule]\nxyz = 3\n[basis_set]\nbasis = \"b.json\"\n", "[molecule] xyz");
  expect_config_error(
      "[molecule]\nxyz = \"w.xyz\"\ncharge = 3000000000\n[basis_set]\nbasis = \"b.json\"\n",
      "[molecule] charge");
  expect_config_error(with_inputs("[orthogonalizer]\neigenvalue_floor = \"small\"\n"),
                      "[orthogonalizer] eigenvalue_floor");
  expect_config_error(with_inputs("[guess]\ntype = 1\n"), "[guess] type");
}

TEST(PipelineConfig, FromLoadsTheProductRelativeToTheConfig) {
  auto calc = fb::parse_config(with_inputs("[guess]\nfrom = \"d.npy\"\n"), "/inputs");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_FALSE(calc->step<fb::GuessStep>().has_value());
  EXPECT_EQ(calc->from<fb::GuessStep>(), std::filesystem::path("/inputs/d.npy"));
  // What the guess's load reads is still planned.
  EXPECT_TRUE(calc->step<fb::BasisSetStep>().has_value());
}

TEST(PipelineConfig, ALoadedGuessIsNotFilledInForTheScf) {
  auto calc = fb::parse_config(with_inputs("[guess]\nfrom = \"d.npy\"\n[scf]\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_TRUE(calc->step<fb::ScfStep>().has_value());
  EXPECT_FALSE(calc->step<fb::GuessStep>().has_value());
  EXPECT_EQ(calc->from<fb::GuessStep>(), std::filesystem::path("d.npy"));
  // Without a filled-in guess, nothing loads it.
  auto unloaded = fb::parse_config(with_inputs("[scf]\n"), "");
  ASSERT_TRUE(unloaded.has_value()) << unloaded.error().message;
  EXPECT_FALSE(unloaded->from<fb::GuessStep>().has_value());
}

TEST(PipelineConfig, FromTakesNoOtherOption) {
  expect_config_error(with_inputs("[guess]\nfrom = \"d.npy\"\ntype = \"sad\"\n"),
                      "[guess] loads 'from' a file, so takes no option 'type'");
  expect_config_error(with_inputs("[guess]\nfrom = 1\n"), "[guess] from");
}

TEST(PipelineConfig, AStepWithNoLoadCannotBeLoaded) {
  expect_config_error(with_inputs("[overlap]\nfrom = \"s.npy\"\n"),
                      "[overlap] cannot be loaded 'from' a file");
}

TEST(PipelineConfig, MalformedTomlIsIoError) {
  auto calc = fb::parse_config("[molecule\n", "");
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::IO);
}

TEST(PipelineConfig, ReadResolvesAgainstTheFilesDirectory) {
  const fb::test::ScratchDir dir("pipeline-config");
  std::ofstream(dir / "run.toml") << kWater;
  auto calc = fb::read_config(dir / "run.toml");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->xyz, dir / "water.xyz");
}

TEST(PipelineConfig, ReadOfMissingFileIsIoError) {
  const fb::test::ScratchDir dir("pipeline-config");
  auto calc = fb::read_config(dir / "absent.toml");
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::IO);
}

} // namespace
