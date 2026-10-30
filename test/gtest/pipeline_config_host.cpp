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

// An output directory apart from every input directory the cases use. Nothing
// is created: parse_config never touches the disk.
const std::filesystem::path kOut = "/out";

std::string with_inputs(std::string_view tables) {
  return std::string(kInputs).append(tables);
}

void expect_config_error(std::string_view toml, std::string_view needle) {
  auto calc = fb::parse_config(toml, "", kOut);
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(calc.error().message.find(needle), std::string::npos) << calc.error().message;
}

TEST(PipelineConfig, EveryOptionIsRead) {
  auto calc = fb::parse_config(kWater, "/inputs", kOut);
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

TEST(PipelineConfig, ARelativeBasisPathIsAnchoredToTheInputDirectory) {
  auto calc = fb::parse_config("[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"b.json\"\n",
                               "/inputs", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::BasisSetStep>()->basis, "/inputs/b.json");
}

TEST(PipelineConfig, ABuiltinBasisNameIsNotAPath) {
  // The name of a compiled-in set must survive the front end verbatim: joined
  // to base_dir it would become /inputs/sto-3g, a file nobody wrote.
  for (const std::string_view spelling : {"sto-3g", "STO-3G", "sto3g", "cc-pVTZ"}) {
    auto calc = fb::parse_config(
        std::format("[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"{}\"\n", spelling),
        "/inputs", kOut);
    ASSERT_TRUE(calc.has_value()) << spelling << ": " << calc.error().message;
    EXPECT_EQ(calc->step<fb::BasisSetStep>()->basis, spelling);
  }
}

TEST(PipelineConfig, ABuiltinAuxBasisNameIsNotAPath) {
  // BasisSetStep's rule over the fitting sets: a compiled-in name is kept
  // verbatim, spelt however it was written.
  for (const std::string_view spelling : {"cc-pvdz-jkfit", "cc-pVTZ-JKFIT", "ccpvdzjkfit"}) {
    auto calc =
        fb::parse_config(with_inputs(std::format(
                             "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n", spelling)),
                         "/inputs", kOut);
    ASSERT_TRUE(calc.has_value()) << spelling << ": " << calc.error().message;
    EXPECT_EQ(calc->step<fb::EriStep>()->aux_basis, spelling);
  }
  // An orbital set's name is no fitting set's, so it reads as a path.
  auto orbital = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz\"\n"), "/inputs", kOut);
  ASSERT_TRUE(orbital.has_value()) << orbital.error().message;
  EXPECT_EQ(orbital->step<fb::EriStep>()->aux_basis, "/inputs/cc-pvdz");
}

TEST(PipelineConfig, AnAuxBasisPathIsAnchoredToTheInputDirectory) {
  auto relative =
      fb::parse_config(with_inputs("[eri]\nmode = \"density_fitting\"\naux_basis = \"fit.json\"\n"),
                       "/inputs", kOut);
  ASSERT_TRUE(relative.has_value()) << relative.error().message;
  EXPECT_EQ(relative->step<fb::EriStep>()->aux_basis, "/inputs/fit.json");
  // An absolute path is kept as written.
  auto absolute = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\naux_basis = \"/fits/fit.json\"\n"), "/inputs",
      kOut);
  ASSERT_TRUE(absolute.has_value()) << absolute.error().message;
  EXPECT_EQ(absolute->step<fb::EriStep>()->aux_basis, "/fits/fit.json");
}

TEST(PipelineConfig, TheEriFillsInWhatItReads) {
  auto calc = fb::parse_config(with_inputs("[eri]\n"), "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_TRUE(calc->step<fb::MoleculeStep>().has_value());
  EXPECT_TRUE(calc->step<fb::BasisSetStep>().has_value());
  EXPECT_TRUE(calc->step<fb::PrimitivePairClassesStep>().has_value());
  // three_center and the metric build their own transforms.
  EXPECT_FALSE(calc->step<fb::SphericalTransformStep>().has_value());
  EXPECT_FALSE(calc->step<fb::ScfStep>().has_value());
}

TEST(PipelineConfig, TheEriIsIntegralDirectByDefault) {
  // Written empty, by name, or left out and filled in for the SCF.
  for (std::string_view tables : {"[eri]\n", "[eri]\nmode = \"integral_direct\"\n", "[scf]\n"}) {
    auto calc = fb::parse_config(with_inputs(tables), "", kOut);
    ASSERT_TRUE(calc.has_value()) << tables << ": " << calc.error().message;
    const fb::EriStep &eri = *calc->step<fb::EriStep>();
    EXPECT_EQ(eri.mode, fb::EriMode::IntegralDirect) << tables;
    EXPECT_TRUE(eri.aux_basis.empty()) << tables;
  }
}

TEST(PipelineConfig, DensityFittingLeavesADefaultAuxBasisToTheStep) {
  // The default needs the basis's name, which only loading it gives, so the
  // front end leaves aux_basis empty and EriStep::run picks it.
  auto calc = fb::parse_config(with_inputs("[eri]\nmode = \"density_fitting\"\n[scf]\n"), "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::EriStep>()->mode, fb::EriMode::DensityFitting);
  EXPECT_TRUE(calc->step<fb::EriStep>()->aux_basis.empty());
}

TEST(PipelineConfig, TheErisOptionsAreRefusedOutOfPlace) {
  expect_config_error(with_inputs("[eri]\nmode = \"ri\"\n"),
                      "[eri] has no mode 'ri' (known: 'integral_direct', 'density_fitting')");
  expect_config_error(with_inputs("[eri]\nmode = \"direct\"\n"), "[eri] has no mode 'direct'");
  expect_config_error(with_inputs("[eri]\nmode = 1\n"), "[eri] mode is not a string");
  expect_config_error(with_inputs("[eri]\naux_basis = \"cc-pvdz-jkfit\"\n"),
                      "[eri] aux_basis needs mode = 'density_fitting'");
  expect_config_error(
      with_inputs("[eri]\nmode = \"integral_direct\"\naux_basis = \"cc-pvdz-jkfit\"\n"),
      "[eri] aux_basis needs mode = 'density_fitting'");
  expect_config_error(with_inputs("[eri]\nmode = \"density_fitting\"\naux_basis = \"\"\n"),
                      "[eri] aux_basis is empty");
  expect_config_error(with_inputs("[eri]\nmode = \"density_fitting\"\naux_basis = 3\n"),
                      "[eri] aux_basis is not a string");
  expect_config_error(with_inputs("[eri]\naux = \"cc-pvtz-jkfit\"\n"), "[eri] has no option 'aux'");
}

TEST(PipelineConfig, TheErisOldHomesAreRefused) {
  // The ERI choice is the pipeline's, in [eri]: [scf] and [casscf] take none
  // of it, and [density_fitting] is gone.
  for (std::string_view key : {"eri = \"density_fitting\"\n", "aux_basis = \"cc-pvdz-jkfit\"\n"}) {
    const std::string name = std::string(key.substr(0, key.find(' ')));
    expect_config_error(with_inputs(std::format("[scf]\n{}", key)),
                        std::format("[scf] has no option '{}'", name));
    expect_config_error(with_inputs(std::format("[casscf]\nncas = 2\nnelecas = 2\n{}", key)),
                        std::format("[casscf] has no option '{}'", name));
  }
  expect_config_error(with_inputs("[density_fitting]\naux_basis = \"cc-pvdz-jkfit\"\n"),
                      "unknown step [density_fitting]");
}

TEST(PipelineConfig, TheFockRebuildIntervalIsRefusedUnderDensityFitting) {
  // fock_rebuild_interval is the integral-direct build's, whichever table
  // comes first.
  for (std::string_view tables :
       {"[eri]\nmode = \"density_fitting\"\n[scf]\nfock_rebuild_interval = 1\n",
        "[scf]\nfock_rebuild_interval = 1\n[eri]\nmode = \"density_fitting\"\n"}) {
    expect_config_error(with_inputs(tables),
                        "[scf] fock_rebuild_interval is the integral-direct Fock build's");
  }
  // Integral-direct, written or filled in, reads it.
  for (std::string_view tables :
       {"[scf]\nfock_rebuild_interval = 1\n", "[eri]\nmode = \"integral_direct\"\n[scf]\n"
                                              "fock_rebuild_interval = 1\n"}) {
    auto calc = fb::parse_config(with_inputs(tables), "", kOut);
    ASSERT_TRUE(calc.has_value()) << calc.error().message;
    EXPECT_EQ(calc->step<fb::ScfStep>()->fock_rebuild_interval, 1);
  }
}

TEST(PipelineConfig, OneEriServesTheScfAndTheCasscf) {
  // One [eri] for the pipeline: the SCF and the CASSCF read the same step's
  // product, so there is no per-step choice to make.
  auto calc = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pVDZ-JKFIT\"\n"
                  "[casscf]\nncas = 2\nnelecas = 2\n"),
      "/inputs", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_TRUE(calc->step<fb::ScfStep>().has_value());
  EXPECT_TRUE(calc->step<fb::CasscfStep>().has_value());
  EXPECT_EQ(calc->step<fb::EriStep>()->mode, fb::EriMode::DensityFitting);
  EXPECT_EQ(calc->step<fb::EriStep>()->aux_basis, "cc-pVDZ-JKFIT");
  // A [casscf] alone fills [eri] in, integral-direct.
  auto plain = fb::parse_config(with_inputs("[casscf]\nncas = 2\nnelecas = 2\n"), "", kOut);
  ASSERT_TRUE(plain.has_value()) << plain.error().message;
  EXPECT_EQ(plain->step<fb::EriStep>()->mode, fb::EriMode::IntegralDirect);
}

TEST(PipelineConfig, AnAuxBasisFileIsNotInTheOutputDirectory) {
  auto calc = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\naux_basis = \"../out/fit.json\"\n"), "/run",
      "/out");
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(calc.error().message.find("is in the output directory '/out'"), std::string::npos)
      << calc.error().message;
  // A name is no file, wherever the output goes.
  auto name = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n"), "/run",
      "/out");
  EXPECT_TRUE(name.has_value()) << name.error().message;
}

TEST(PipelineConfig, AbsentOptionsKeepTheStepDefaults) {
  auto calc = fb::parse_config(with_inputs("[orthogonalizer]\n"), "", kOut);
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
  auto calc = fb::parse_config(with_inputs("[orthogonalizer]\neigenvalue_floor = 0\n"), "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::OrthogonalizerStep>()->eigenvalue_floor, 0.0);
}

TEST(PipelineConfig, ATableIsARequestAndItsProducersAreFilledIn) {
  auto calc = fb::parse_config(with_inputs("[core_hamiltonian]\n"), "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_TRUE(calc->step<fb::CoreHamiltonianStep>().has_value());
  EXPECT_TRUE(calc->step<fb::SphericalTransformStep>().has_value());
  EXPECT_TRUE(calc->step<fb::PrimitivePairClassesStep>().has_value());
  // Not upstream of anything asked for.
  EXPECT_FALSE(calc->step<fb::OverlapStep>().has_value());
  EXPECT_FALSE(calc->step<fb::OrthogonalizerStep>().has_value());
}

TEST(PipelineConfig, TheGuessNeedsNoDeviceStepPastTheMolecule) {
  auto calc = fb::parse_config(with_inputs("[guess]\n"), "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::GuessStep>()->type, fb::GuessStep{}.type);
  EXPECT_FALSE(calc->step<fb::OverlapStep>().has_value());
  EXPECT_FALSE(calc->step<fb::PrimitivePairClassesStep>().has_value());
}

TEST(PipelineConfig, TheScfFillsInTheWholeChain) {
  auto calc = fb::parse_config(with_inputs("[scf]\n"), "", kOut);
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
  // Always written, under the step's own name.
  EXPECT_EQ(scf.molden, kOut / "scf.molden");
}

TEST(PipelineConfig, EveryScfOptionIsRead) {
  auto calc = fb::parse_config(with_inputs("[scf]\nmax_iterations = 40\nenergy_tolerance = 1e-8\n"
                                           "commutator_tolerance = 1e-5\ndiis_history = 6\n"
                                           "fock_rebuild_interval = 1\n"),
                               "/run", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::ScfStep &scf = *calc->step<fb::ScfStep>();
  EXPECT_EQ(scf.max_iterations, 40);
  EXPECT_EQ(scf.energy_tolerance, 1e-8);
  EXPECT_EQ(scf.commutator_tolerance, 1e-5);
  EXPECT_EQ(scf.diis_history, 6);
  EXPECT_EQ(scf.fock_rebuild_interval, 1);
}

TEST(PipelineConfig, AnOutputIsNamedByTheRunNotTheConfig) {
  // The four steps that write a Molden file, each under its own kName.
  auto calc = fb::parse_config(
      with_inputs(
          "[localization]\n[casci]\nncas = 2\nnelecas = 2\n[casscf]\nncas = 2\nnelecas = 2\n"),
      "/run", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::ScfStep>()->molden, kOut / "scf.molden");
  EXPECT_EQ(calc->step<fb::LocalizationStep>()->molden, kOut / "localization.molden");
  EXPECT_EQ(calc->step<fb::CasciStep>()->molden, kOut / "casci.molden");
  EXPECT_EQ(calc->step<fb::CasscfStep>()->molden, kOut / "casscf.molden");
  for (std::string_view step : {"scf", "localization"}) {
    expect_config_error(with_inputs(std::format("[{}]\nmolden = \"x.molden\"\n", step)),
                        std::format("[{}] has no option 'molden'", step));
  }
  expect_config_error(with_inputs("[casci]\nncas = 2\nnelecas = 2\nmolden = \"x.molden\"\n"),
                      "[casci] has no option 'molden'");
  expect_config_error(with_inputs("[casscf]\nncas = 2\nnelecas = 2\nmolden = \"x.molden\"\n"),
                      "[casscf] has no option 'molden'");
}

TEST(PipelineConfig, TheOutputDirectoryIsNotTheInputDirectory) {
  // Compared as the directories they name: '.', '' and the working directory
  // spelt out are one, and a trailing separator or '..' changes nothing.
  const std::filesystem::path cwd = std::filesystem::current_path();
  const std::array<std::pair<std::filesystem::path, std::filesystem::path>, 5> same{{
      {"", "."},
      {".", cwd},
      {"/run", "/run/"},
      {"/run/in", "/run/out/../in"},
      {"in", cwd / "in"},
  }};
  for (const auto &[in, out] : same) {
    auto calc = fb::parse_config(with_inputs("[scf]\n"), in, out);
    ASSERT_FALSE(calc.has_value()) << in << " vs " << out;
    EXPECT_EQ(calc.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(calc.error().message.find("the input and output directories are both"),
              std::string::npos)
        << calc.error().message;
  }
  // A directory inside the other is not the same one: the default output
  // directory sits inside the default input directory.
  auto nested = fb::parse_config(with_inputs("[scf]\n"), ".", "fbf_output");
  EXPECT_TRUE(nested.has_value()) << nested.error().message;
}

TEST(PipelineConfig, NoInputFileIsInTheOutputDirectory) {
  // Each input key, reaching the output directory past a different input one.
  const std::array<std::string, 3> tables{
      "[molecule]\nxyz = \"/out/w.xyz\"\n[basis_set]\nbasis = \"b.json\"\n[scf]\n",
      "[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"../out/b.json\"\n[scf]\n",
      with_inputs("[guess]\nfrom = \"/out/scf_density.npy\"\n[scf]\n"),
  };
  for (const std::string &table : tables) {
    auto calc = fb::parse_config(table, "/run", "/out/");
    ASSERT_FALSE(calc.has_value()) << table;
    EXPECT_EQ(calc.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(calc.error().message.find("is in the output directory '/out'"), std::string::npos)
        << calc.error().message;
  }
  // Below it is apart: outputs are written into it, not under it.
  auto below = fb::parse_config(
      "[molecule]\nxyz = \"/out/prev/w.xyz\"\n[basis_set]\nbasis = \"sto-3g\"\n[scf]\n", "/run",
      kOut);
  EXPECT_TRUE(below.has_value()) << below.error().message;
}

TEST(PipelineConfig, AMisspeltScfOptionIsInvalidConfig) {
  expect_config_error(with_inputs("[scf]\nmax_iter = 40\n"), "[scf] has no option 'max_iter'");
  expect_config_error(with_inputs("[scf]\nmax_iterations = 1.5\n"), "[scf] max_iterations");
}

TEST(PipelineConfig, OnlyWhatWasAskedForIsPlanned) {
  auto calc = fb::parse_config("[molecule]\nxyz = \"w.xyz\"\n", "", kOut);
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
  expect_config_error(with_inputs("[overlap]\nscreen = 1e-12\n"),
                      "[overlap] has no option 'screen'");
  // A step with no options has no keys to give.
  expect_config_error(with_inputs("[spherical_transform]\nscreen = 1e-12\n"),
                      "[spherical_transform] has no option 'screen'");
}

TEST(PipelineConfig, WriteNpyNamesTheOverlapsOwnFile) {
  // Off unless [overlap] asks, and a filled-in overlap is read from an empty
  // table, so planning it for another step writes nothing.
  auto calc = fb::parse_config(with_inputs("[overlap]\nwrite_npy = true\n"), "/run", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::OverlapStep>()->npy, kOut / "overlap.npy");
  for (std::string_view table : {"[overlap]\n", "[overlap]\nwrite_npy = false\n", "[scf]\n"}) {
    auto off = fb::parse_config(with_inputs(table), "/run", kOut);
    ASSERT_TRUE(off.has_value()) << off.error().message;
    EXPECT_TRUE(off->step<fb::OverlapStep>()->npy.empty()) << table;
  }
  expect_config_error(with_inputs("[overlap]\nwrite_npy = 1\n"),
                      "[overlap] write_npy is not a boolean");
  // A config names no output, so not this one either.
  expect_config_error(with_inputs("[overlap]\nnpy = \"s.npy\"\n"), "[overlap] has no option 'npy'");
}

TEST(PipelineConfig, AnUnknownGuessTypeIsInvalidConfig) {
  expect_config_error(with_inputs("[guess]\ntype = \"core\"\n"), "[guess] has no type 'core'");
  expect_config_error(with_inputs("[guess]\ntype = \"SAD\"\n"), "[guess] has no type 'SAD'");
}

TEST(PipelineConfig, TheLocalizationModeAndItsOptionsAreRead) {
  auto calc = fb::parse_config(with_inputs("[localization]\nmode = \"pipek_mezey\"\n"
                                           "max_sweeps = 7\nexponent = 4\n"),
                               "/run", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::LocalizationStep &step = *calc->step<fb::LocalizationStep>();
  EXPECT_EQ(step.mode, fb::LocalizationMode::PipekMezey);
  EXPECT_EQ(step.pipek_mezey.max_sweeps, 7);
  EXPECT_EQ(step.pipek_mezey.exponent, fb::localization::PmExponent::Four);
  // An empty table is Pipek-Mezey with its defaults.
  auto plain = fb::parse_config(with_inputs("[localization]\n"), "", kOut);
  ASSERT_TRUE(plain.has_value()) << plain.error().message;
  EXPECT_EQ(plain->step<fb::LocalizationStep>()->mode, fb::LocalizationMode::PipekMezey);
  EXPECT_EQ(plain->step<fb::LocalizationStep>()->pipek_mezey.exponent,
            fb::localization::PmExponent::Two);
  // IBO is Knizia's exponent four unless the table names another.
  auto ibo = fb::parse_config(with_inputs("[localization]\nmode = \"ibo\"\n"), "", kOut);
  ASSERT_TRUE(ibo.has_value()) << ibo.error().message;
  EXPECT_EQ(ibo->step<fb::LocalizationStep>()->mode, fb::LocalizationMode::Ibo);
  EXPECT_EQ(ibo->step<fb::LocalizationStep>()->pipek_mezey.exponent,
            fb::localization::PmExponent::Four);
  auto ibo2 =
      fb::parse_config(with_inputs("[localization]\nmode = \"ibo\"\nexponent = 2\n"), "", kOut);
  ASSERT_TRUE(ibo2.has_value()) << ibo2.error().message;
  EXPECT_EQ(ibo2->step<fb::LocalizationStep>()->pipek_mezey.exponent,
            fb::localization::PmExponent::Two);
}

TEST(PipelineConfig, LivvoIsReadAndNamesItsOwnMolden) {
  // In either mode; the LIVVOs' Molden file is named by the run, beside the
  // localized orbitals', and only when there are LIVVOs to write.
  for (std::string_view mode : {"pipek_mezey", "ibo"}) {
    auto calc = fb::parse_config(
        with_inputs(std::format("[localization]\nmode = \"{}\"\nlivvo = true\n", mode)), "/run",
        kOut);
    ASSERT_TRUE(calc.has_value()) << calc.error().message;
    const fb::LocalizationStep &step = *calc->step<fb::LocalizationStep>();
    EXPECT_TRUE(step.livvo) << mode;
    EXPECT_EQ(step.livvo_molden, kOut / "localization_livvo.molden") << mode;
    EXPECT_EQ(step.molden, kOut / "localization.molden") << mode;
  }
  for (std::string_view table : {"[localization]\n", "[localization]\nlivvo = false\n"}) {
    auto calc = fb::parse_config(with_inputs(table), "/run", kOut);
    ASSERT_TRUE(calc.has_value()) << calc.error().message;
    EXPECT_FALSE(calc->step<fb::LocalizationStep>()->livvo) << table;
    EXPECT_TRUE(calc->step<fb::LocalizationStep>()->livvo_molden.empty()) << table;
  }
  expect_config_error(with_inputs("[localization]\nlivvo = 1\n"),
                      "[localization] livvo is not a boolean");
  expect_config_error(with_inputs("[localization]\nlivvo = \"true\"\n"),
                      "[localization] livvo is not a boolean");
  // A config names no output, so not this one either.
  expect_config_error(with_inputs("[localization]\nlivvo = true\nlivvo_molden = \"x.molden\"\n"),
                      "[localization] has no option 'livvo_molden'");
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
      with_inputs("[casci]\nncas = 4\nnelecas = 4\nnroots = 3\nactive = [2, 4, 5, 7]\n"), "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::CasciStep &casci = *calc->step<fb::CasciStep>();
  EXPECT_EQ(casci.ncas, 4);
  EXPECT_EQ(casci.nelecas, 4);
  EXPECT_EQ(casci.nroots, 3);
  EXPECT_EQ(casci.active, (std::vector<std::int32_t>{2, 4, 5, 7}));
  EXPECT_TRUE(calc->step<fb::ScfStep>().has_value());
  EXPECT_TRUE(calc->step<fb::CoreHamiltonianStep>().has_value());
  EXPECT_FALSE(calc->step<fb::LocalizationStep>().has_value());

  auto plain = fb::parse_config(with_inputs("[casci]\nncas = 2\nnelecas = 2\n"), "", kOut);
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

TEST(PipelineConfig, TheCasciAvasOptionsAreReadInPlaceOfAnActiveSpace) {
  auto calc = fb::parse_config(with_inputs("[casci]\navas = [\"O 2p\", \"H 1s\"]\n"
                                           "avas_threshold = 0.3\nnroots = 2\n"),
                               "base", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::CasciStep &casci = *calc->step<fb::CasciStep>();
  EXPECT_EQ(casci.avas, (std::vector<std::string>{"O 2p", "H 1s"}));
  EXPECT_EQ(casci.avas_threshold, 0.3);
  EXPECT_EQ(casci.nroots, 2);
  EXPECT_EQ(casci.ncas, 0);
  EXPECT_EQ(casci.nelecas, 0);

  auto plain = fb::parse_config(with_inputs("[casci]\navas = [\"N 2p\"]\n"), "", kOut);
  ASSERT_TRUE(plain.has_value()) << plain.error().message;
  EXPECT_EQ(plain->step<fb::CasciStep>()->avas_threshold, 0.2);
}

TEST(PipelineConfig, AMalformedCasciAvasIsInvalidConfig) {
  expect_config_error(with_inputs("[casci]\navas = [\"O 2p\"]\nncas = 3\n"),
                      "[casci] avas picks the active space, so takes no 'ncas'");
  expect_config_error(with_inputs("[casci]\navas = [\"O 2p\"]\nnelecas = 4\n"),
                      "[casci] avas picks the active space, so takes no 'nelecas'");
  expect_config_error(with_inputs("[casci]\navas = [\"O 2p\"]\nactive = [1]\n"),
                      "[casci] avas picks the active space, so takes no 'active'");
  expect_config_error(with_inputs("[casci]\nncas = 2\nnelecas = 2\navas_threshold = 0.3\n"),
                      "[casci] avas_threshold needs 'avas'");
  expect_config_error(with_inputs("[casci]\navas = []\n"), "[casci] avas names no AO label");
  expect_config_error(with_inputs("[casci]\navas = \"O 2p\"\n"),
                      "[casci] avas is not an array of strings");
  expect_config_error(with_inputs("[casci]\navas = [\"O 2p\", 2]\n"),
                      "[casci] avas is not an array of strings");
  expect_config_error(with_inputs("[casci]\navas = [\"O 2p\"]\navas_threshold = \"x\"\n"),
                      "[casci] avas_threshold is not a number");
}

// The ranges need the molecule's electrons and the basis's orbitals, so
// orbital_order checks them when the step runs: here, water's 10 electrons
// over STO-3G's 7 orbitals.
TEST(PipelineConfig, TheCasciOrbitalOrderPutsCoreActiveRest) {
  const auto order = [](std::string_view table) {
    auto calc = fb::parse_config(with_inputs(table), "", kOut);
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
    auto calc = fb::parse_config(with_inputs(table), "", kOut);
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

TEST(PipelineConfig, EveryCasscfOptionIsReadAndTheScfIsFilledIn) {
  auto calc = fb::parse_config(
      with_inputs("[casscf]\nncas = 3\nnelecas = 4\nnroots = 2\nactive = [5, 6, 7]\n"
                  "weights = [0.75, 0.25]\nconv_tol = 1e-9\nconv_tol_grad = 1e-5\n"
                  "max_cycle = 20\n"),
      "base", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::CasscfStep &casscf = *calc->step<fb::CasscfStep>();
  EXPECT_EQ(casscf.ncas, 3);
  EXPECT_EQ(casscf.nelecas, 4);
  EXPECT_EQ(casscf.nroots, 2);
  EXPECT_EQ(casscf.active, (std::vector<std::int32_t>{5, 6, 7}));
  EXPECT_EQ(casscf.weights, (std::vector<double>{0.75, 0.25}));
  EXPECT_EQ(casscf.conv_tol, 1e-9);
  EXPECT_EQ(casscf.conv_tol_grad, 1e-5);
  EXPECT_EQ(casscf.max_cycle, 20);
  EXPECT_TRUE(calc->step<fb::ScfStep>().has_value());
  EXPECT_FALSE(calc->step<fb::CasciStep>().has_value());

  // The defaults are mcscf::solve's; empty weights are equal ones.
  auto plain = fb::parse_config(with_inputs("[casscf]\navas = [\"O 2p\"]\n"), "", kOut);
  ASSERT_TRUE(plain.has_value()) << plain.error().message;
  const fb::CasscfStep &defaults = *plain->step<fb::CasscfStep>();
  const fb::mcscf::Options options;
  EXPECT_EQ(defaults.avas, (std::vector<std::string>{"O 2p"}));
  EXPECT_EQ(defaults.nroots, 1);
  EXPECT_TRUE(defaults.weights.empty());
  EXPECT_EQ(defaults.conv_tol, options.conv_tol);
  EXPECT_EQ(defaults.conv_tol_grad, options.conv_tol_grad);
  EXPECT_EQ(defaults.max_cycle, options.max_cycle);

  // An integer weight reads as a number, as every float option does.
  auto whole =
      fb::parse_config(with_inputs("[casscf]\nncas = 2\nnelecas = 2\nweights = [1]\n"), "", kOut);
  ASSERT_TRUE(whole.has_value()) << whole.error().message;
  EXPECT_EQ(whole->step<fb::CasscfStep>()->weights, (std::vector<double>{1.0}));
}

TEST(PipelineConfig, AMalformedCasscfTableIsInvalidConfig) {
  // The selection's refusals, shared with [casci].
  expect_config_error(with_inputs("[casscf]\nnelecas = 4\n"), "[casscf] needs 'ncas'");
  expect_config_error(with_inputs("[casscf]\nncas = 4\n"), "[casscf] needs 'nelecas'");
  expect_config_error(with_inputs("[casscf]\navas = [\"O 2p\"]\nncas = 3\n"),
                      "[casscf] avas picks the active space, so takes no 'ncas'");
  expect_config_error(with_inputs("[casscf]\nncas = 2\nnelecas = 2\navas_threshold = 0.3\n"),
                      "[casscf] avas_threshold needs 'avas'");
  expect_config_error(with_inputs("[casscf]\navas = []\n"), "[casscf] avas names no AO label");
  expect_config_error(with_inputs("[casscf]\nncas = 2\nnelecas = 2\nactive = [3, 4.0]\n"),
                      "[casscf] active is not an array of 32-bit integers");
  expect_config_error(with_inputs("[casscf]\nncas = 4\nnelecas = 4\nnroot = 2\n"),
                      "[casscf] has no option 'nroot'");

  // Its own.
  const auto cas = [](std::string_view keys) {
    return with_inputs(std::format("[casscf]\nncas = 3\nnelecas = 4\n{}", keys));
  };
  expect_config_error(cas("weights = 0.5\n"), "[casscf] weights is not an array of numbers");
  expect_config_error(cas("weights = [0.5, \"x\"]\nnroots = 2\n"),
                      "[casscf] weights is not an array of numbers");
  expect_config_error(cas("weights = [0.5, 0.5]\n"),
                      "[casscf] weights has 2 entries, not nroots 1");
  expect_config_error(cas("nroots = 2\nweights = [1.0]\n"),
                      "[casscf] weights has 1 entries, not nroots 2");
  expect_config_error(cas("nroots = 2\nweights = [1.25, -0.25]\n"),
                      "[casscf] weights has a negative entry");
  expect_config_error(cas("nroots = 2\nweights = [0.5, 0.4]\n"),
                      "[casscf] weights sum to 0.9, not 1");
  expect_config_error(cas("conv_tol = 0.0\n"),
                      "[casscf] conv_tol and conv_tol_grad must be positive");
  expect_config_error(cas("conv_tol_grad = -1e-6\n"),
                      "[casscf] conv_tol and conv_tol_grad must be positive");
  expect_config_error(cas("conv_tol = nan\n"),
                      "[casscf] conv_tol and conv_tol_grad must be positive");
  expect_config_error(cas("max_cycle = 0\n"), "[casscf] max_cycle is 0, below 1");
  expect_config_error(cas("max_cycle = 1.5\n"), "[casscf] max_cycle is not a 32-bit integer");
}

TEST(PipelineConfig, EveryFciOptionIsReadAndTheScfIsFilledIn) {
  auto calc = fb::parse_config(with_inputs("[fci]\nnroots = 3\n"), "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::FciStep>()->nroots, 3);
  EXPECT_TRUE(calc->step<fb::ScfStep>().has_value());
  EXPECT_TRUE(calc->step<fb::CoreHamiltonianStep>().has_value());
  EXPECT_FALSE(calc->step<fb::CasciStep>().has_value());

  auto plain = fb::parse_config(with_inputs("[fci]\n"), "", kOut);
  ASSERT_TRUE(plain.has_value()) << plain.error().message;
  EXPECT_EQ(plain->step<fb::FciStep>()->nroots, 1);
}

TEST(PipelineConfig, AMalformedFciTableIsInvalidConfig) {
  // No active space: FCI is every orbital.
  expect_config_error(with_inputs("[fci]\nncas = 4\n"), "[fci] has no option 'ncas'");
  expect_config_error(with_inputs("[fci]\nnroots = 1.5\n"), "[fci] nroots");
}

TEST(PipelineConfig, EveryCisOptionIsReadAndTheScfIsFilledIn) {
  auto calc = fb::parse_config(with_inputs("[cis]\nn_states = 3\nspin = \"triplet\"\n"
                                           "residual_tolerance = 1e-8\nmax_iterations = 50\n"
                                           "max_subspace = 12\ncluster_gap = 0.05\n"),
                               "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::CisStep &cis = *calc->step<fb::CisStep>();
  EXPECT_EQ(cis.options.n_states, 3);
  EXPECT_EQ(cis.options.spin, fb::cis::Spin::Triplet);
  EXPECT_EQ(cis.options.davidson.residual_tolerance, 1e-8);
  EXPECT_EQ(cis.options.davidson.max_iterations, 50);
  EXPECT_EQ(cis.max_subspace, 12);
  EXPECT_EQ(cis.options.cluster_gap, 0.05);
  EXPECT_TRUE(calc->step<fb::ScfStep>().has_value());
  EXPECT_TRUE(calc->step<fb::EriStep>().has_value());
  EXPECT_FALSE(calc->step<fb::CasciStep>().has_value());

  // The defaults are cis::Options' own.
  auto plain = fb::parse_config(with_inputs("[cis]\n"), "", kOut);
  ASSERT_TRUE(plain.has_value()) << plain.error().message;
  const fb::CisStep &defaults = *plain->step<fb::CisStep>();
  const fb::cis::Options options;
  EXPECT_EQ(defaults.options.n_states, options.n_states);
  EXPECT_EQ(defaults.options.spin, fb::cis::Spin::Singlet);
  EXPECT_EQ(defaults.options.davidson.residual_tolerance, options.davidson.residual_tolerance);
  EXPECT_EQ(defaults.options.davidson.max_iterations, options.davidson.max_iterations);
  EXPECT_EQ(defaults.max_subspace, fb::CisStep{}.max_subspace);
  EXPECT_EQ(defaults.options.cluster_gap, 0.0);
  EXPECT_FALSE(defaults.options.ris.has_value());
  EXPECT_EQ(defaults.options.method, fb::cis::Method::Full);
}

// method = "ris" is TDA-ris, at cis::Options' θs unless theta_j or theta_k
// says; it is independent of the RIS preconditioner, so ris = true may stand
// beside it.
TEST(PipelineConfig, TheCisMethodIsRead) {
  auto calc = fb::parse_config(
      with_inputs("[cis]\nmethod = \"ris\"\ntheta_j = 0.3\ntheta_k = 0.15\nris = true\n"), "",
      kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::cis::Options &options = calc->step<fb::CisStep>()->options;
  EXPECT_EQ(options.method, fb::cis::Method::Ris);
  EXPECT_EQ(options.theta_j, 0.3);
  EXPECT_EQ(options.theta_k, 0.15);
  EXPECT_TRUE(options.ris.has_value());

  auto plain = fb::parse_config(with_inputs("[cis]\nmethod = \"ris\"\n"), "", kOut);
  ASSERT_TRUE(plain.has_value()) << plain.error().message;
  const fb::cis::Options &defaults = plain->step<fb::CisStep>()->options;
  EXPECT_EQ(defaults.method, fb::cis::Method::Ris);
  EXPECT_EQ(defaults.theta_j, fb::cis::Options{}.theta_j);
  EXPECT_EQ(defaults.theta_k, fb::cis::Options{}.theta_k);
  EXPECT_FALSE(defaults.ris.has_value());

  auto full = fb::parse_config(with_inputs("[cis]\nmethod = \"full\"\n"), "", kOut);
  ASSERT_TRUE(full.has_value()) << full.error().message;
  EXPECT_EQ(full->step<fb::CisStep>()->options.method, fb::cis::Method::Full);
}

TEST(PipelineConfig, TheCisRisOptionsAreRead) {
  auto calc = fb::parse_config(
      with_inputs("[cis]\nris = true\nris_theta = 0.35\nris_shells = \"spd\"\n"), "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const auto &ris = calc->step<fb::CisStep>()->options.ris;
  ASSERT_TRUE(ris.has_value());
  EXPECT_EQ(ris->theta, 0.35);
  EXPECT_EQ(ris->shells, fb::cis::RisShells::Spd);

  // ris alone is RisOptions' defaults; ris = false is none.
  auto plain = fb::parse_config(with_inputs("[cis]\nris = true\n"), "", kOut);
  ASSERT_TRUE(plain.has_value()) << plain.error().message;
  const auto &defaults = plain->step<fb::CisStep>()->options.ris;
  ASSERT_TRUE(defaults.has_value());
  EXPECT_EQ(defaults->theta, fb::cis::RisOptions{}.theta);
  EXPECT_EQ(defaults->shells, fb::cis::RisShells::Sp);
  auto off = fb::parse_config(with_inputs("[cis]\nris = false\n"), "", kOut);
  ASSERT_TRUE(off.has_value()) << off.error().message;
  EXPECT_FALSE(off->step<fb::CisStep>()->options.ris.has_value());
  EXPECT_FALSE(defaults->exchange);
  EXPECT_EQ(defaults->k_max_inner_iters, fb::cis::RisOptions{}.k_max_inner_iters);
  EXPECT_EQ(defaults->k_inner_tol, fb::cis::RisOptions{}.k_inner_tol);

  // RIS-JK and its GMRES's budget.
  auto jk = fb::parse_config(with_inputs("[cis]\nris = true\nris_exchange = true\n"
                                         "ris_k_max_inner_iters = 25\nris_k_inner_tol = 1e-4\n"),
                             "", kOut);
  ASSERT_TRUE(jk.has_value()) << jk.error().message;
  const auto &exchange = jk->step<fb::CisStep>()->options.ris;
  ASSERT_TRUE(exchange.has_value());
  EXPECT_TRUE(exchange->exchange);
  EXPECT_EQ(exchange->k_max_inner_iters, 25);
  EXPECT_EQ(exchange->k_inner_tol, 1e-4);
}

TEST(PipelineConfig, AMalformedCisTableIsInvalidConfig) {
  expect_config_error(with_inputs("[cis]\nnroots = 3\n"), "[cis] has no option 'nroots'");
  expect_config_error(with_inputs("[cis]\nspin = \"quintet\"\n"), "[cis] has no spin 'quintet'");
  expect_config_error(with_inputs("[cis]\nspin = 1\n"), "[cis] spin is not a string");
  expect_config_error(with_inputs("[cis]\nn_states = 0\n"), "[cis] n_states is 0, below 1");
  expect_config_error(with_inputs("[cis]\nn_states = 1.5\n"), "[cis] n_states");
  expect_config_error(with_inputs("[cis]\nresidual_tolerance = 0\n"),
                      "[cis] residual_tolerance must be positive");
  expect_config_error(with_inputs("[cis]\nmax_iterations = 0\n"),
                      "[cis] max_iterations is 0, below 1");
  expect_config_error(with_inputs("[cis]\ncluster_gap = -0.5\n"),
                      "[cis] cluster_gap must not be negative");
  expect_config_error(with_inputs("[cis]\nris = 1\n"), "[cis] ris is not a boolean");
  expect_config_error(with_inputs("[cis]\nmethod = \"rpa\"\n"), "[cis] has no method 'rpa'");
  expect_config_error(with_inputs("[cis]\ntheta_j = 0.3\n"),
                      "[cis] theta_j needs method = \"ris\"");
  expect_config_error(with_inputs("[cis]\nmethod = \"full\"\ntheta_k = 0.3\n"),
                      "[cis] theta_k needs method = \"ris\"");
  expect_config_error(with_inputs("[cis]\nmethod = \"ris\"\ntheta_j = 0\n"),
                      "[cis] theta_j must be finite and positive");
  expect_config_error(with_inputs("[cis]\nmethod = \"ris\"\ntheta_k = -0.2\n"),
                      "[cis] theta_k must be finite and positive");
  expect_config_error(with_inputs("[cis]\nris_theta = 0.3\n"), "[cis] ris_theta needs ris = true");
  expect_config_error(with_inputs("[cis]\nris = false\nris_shells = \"s\"\n"),
                      "[cis] ris_shells needs ris = true");
  expect_config_error(with_inputs("[cis]\nris = true\nris_shells = \"spdf\"\n"),
                      "[cis] has no ris_shells 'spdf'");
  expect_config_error(with_inputs("[cis]\nris = true\nris_theta = 0\n"),
                      "[cis] ris_theta must be finite and positive");
  expect_config_error(with_inputs("[cis]\nris = true\ncluster_gap = 0.1\n"),
                      "[cis] ris and cluster_gap each pick the preconditioner");
  expect_config_error(with_inputs("[cis]\nris_exchange = true\n"),
                      "[cis] ris_exchange needs ris = true");
  expect_config_error(with_inputs("[cis]\nris = true\nris_exchange = 1\n"),
                      "[cis] ris_exchange is not a boolean");
  expect_config_error(with_inputs("[cis]\nris = true\nris_k_max_inner_iters = 5\n"),
                      "[cis] ris_k_max_inner_iters needs ris_exchange = true");
  expect_config_error(with_inputs("[cis]\nris = true\nris_exchange = false\n"
                                  "ris_k_inner_tol = 1e-3\n"),
                      "[cis] ris_k_inner_tol needs ris_exchange = true");
  expect_config_error(with_inputs("[cis]\nris = true\nris_exchange = true\n"
                                  "ris_k_max_inner_iters = -1\n"),
                      "[cis] ris_k_max_inner_iters is -1, below 0");
  expect_config_error(with_inputs("[cis]\nris = true\nris_exchange = true\n"
                                  "ris_k_inner_tol = 0\n"),
                      "[cis] ris_k_inner_tol must be positive");
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

TEST(PipelineConfig, FromLoadsTheProductRelativeToTheInputDirectory) {
  auto calc = fb::parse_config(with_inputs("[guess]\nfrom = \"d.npy\"\n"), "/inputs", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_FALSE(calc->step<fb::GuessStep>().has_value());
  EXPECT_EQ(calc->from<fb::GuessStep>(), std::filesystem::path("/inputs/d.npy"));
  // What the guess's load reads is still planned.
  EXPECT_TRUE(calc->step<fb::BasisSetStep>().has_value());
}

TEST(PipelineConfig, ALoadedGuessIsNotFilledInForTheScf) {
  auto calc = fb::parse_config(with_inputs("[guess]\nfrom = \"d.npy\"\n[scf]\n"), "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_TRUE(calc->step<fb::ScfStep>().has_value());
  EXPECT_FALSE(calc->step<fb::GuessStep>().has_value());
  EXPECT_EQ(calc->from<fb::GuessStep>(), std::filesystem::path("d.npy"));
  // Without a filled-in guess, nothing loads it.
  auto unloaded = fb::parse_config(with_inputs("[scf]\n"), "", kOut);
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
  auto calc = fb::parse_config("[molecule\n", "", kOut);
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::IO);
}

TEST(PipelineConfig, ReadResolvesAgainstTheInputDirectoryAndMakesTheOutputOne) {
  const fb::test::ScratchDir dir("pipeline-config");
  std::filesystem::create_directory(dir / "config");
  std::ofstream(dir / "config/run.toml") << kWater << "[scf]\n";
  // Not the config file's directory: the run's.
  auto calc = fb::read_config(dir / "config/run.toml", dir / "in", dir / "out/a");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->xyz, dir / "in/water.xyz");
  EXPECT_EQ(calc->step<fb::ScfStep>()->molden, dir / "out/a/scf.molden");
  EXPECT_TRUE(std::filesystem::is_directory(dir / "out/a"));
}

TEST(PipelineConfig, ReadOfMissingFileIsIoError) {
  const fb::test::ScratchDir dir("pipeline-config");
  auto calc = fb::read_config(dir / "absent.toml", dir.path(), dir / "out");
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::IO);
}

TEST(PipelineConfig, AnOutputDirectoryThatCannotBeMadeIsIoError) {
  const fb::test::ScratchDir dir("pipeline-config");
  std::ofstream(dir / "run.toml") << kWater;
  std::ofstream(dir / "taken") << "a file, not a directory";
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "taken");
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::IO);
  EXPECT_NE(calc.error().message.find("cannot create the output directory"), std::string::npos)
      << calc.error().message;
}

} // namespace
