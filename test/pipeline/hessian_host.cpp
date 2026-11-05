// Host-only cases for the Hessian step: its contract (static_asserts, as
// steps_host.cpp holds every step's), and its table read by the TOML front
// end -- the method (analytic by default) and the displacement, the chain
// copied in from the config's own tables, every refusal (the analytic method
// over integral-direct ERIs among them), and beside an [optimize], the
// molecule it reads loaded from the final.xyz that step writes, so it runs
// after it; the CASSCF chain copied in beside a [casscf_gradient], the
// analytic method there the root's (beside an [optimize] too), and its
// refusals; the ladder, its pairs and coordinates (#812), on by default
// beside a numerical [nevpt2_gradient], and their refusals; the analytic
// method beside a [nevpt2_gradient] (#818), its coordinates, what it
// refuses before running, and examples/nevpt2_hessian.toml parsed; and the
// guard on a displaced CASSCF, check_displaced_casscf, fed an active orbital
// swapped out and roots swapped. Nothing is run, so nothing touches a
// device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable; // ScratchDir

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// A Node step, as each method runs Pipelines of its own on the node, reading
// the pipeline's molecule: whichever step makes it, it runs after. The
// product carries the harmonic vibrations beside the Hessian.
static_assert(fb::PipelineStep<fb::HessianStep>);
static_assert(fb::kRunShape<fb::HessianStep> == fb::RunShape::Node);
static_assert(std::same_as<fb::StepInput<fb::HessianStep>, fb::View<fb::MoleculeOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::HessianStep>, fb::HessianOutput>);
static_assert(!fb::LoadableStep<fb::HessianStep>);
static_assert(std::same_as<decltype(fb::HessianOutput::vibrations), fb::geometry::Vibrations>);

constexpr std::string_view kInputs =
    "[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"b.json\"\n";

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

// [hessian] reads its own keys and copies in the chain it reruns, each step
// as its own table configures it; the molecule is the pipeline's, read
// from the file, and nothing else is filled in for it.
TEST(PipelineHessian, TheStepReadsItsKeysAndCopiesItsChain) {
  auto calc = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n"
                  "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n"
                  "[hessian]\nmethod = \"numerical\"\ndisplacement = 5e-4\n"),
      "/in", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::HessianStep &hessian = *calc->step<fb::HessianStep>();
  EXPECT_EQ(hessian.method, fb::HessianMethod::Numerical);
  EXPECT_EQ(hessian.displacement, 5e-4);
  EXPECT_EQ(hessian.json_path, kOut / "hessian.json");
  // The chain, read from the tables; a step with no table at its defaults.
  EXPECT_EQ(hessian.basis_set.basis, "/in/b.json");
  EXPECT_EQ(hessian.eri.mode, fb::EriMode::DensityFitting);
  EXPECT_EQ(hessian.eri.aux_basis, "cc-pvdz-jkfit");
  EXPECT_EQ(hessian.scf.energy_tolerance, 1e-12);
  EXPECT_EQ(hessian.scf.commutator_tolerance, 1e-10);
  EXPECT_EQ(hessian.orthogonalizer.eigenvalue_floor, fb::OrthogonalizerStep{}.eigenvalue_floor);
  EXPECT_EQ(hessian.guess.type, fb::GuessType::Sad);
  // The molecule it reads is the config's, run from its file.
  ASSERT_TRUE(calc->step<fb::MoleculeStep>().has_value());
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->xyz, std::filesystem::path("/in/w.xyz"));
  EXPECT_FALSE(calc->from<fb::MoleculeStep>().has_value());

  // The default step, and the chain's steps at their defaults with no tables.
  auto defaults = fb::parse_config(with_inputs("[hessian]\nmethod = \"numerical\"\n"), "", kOut);
  ASSERT_TRUE(defaults.has_value()) << defaults.error().message;
  const fb::HessianStep &plain = *defaults->step<fb::HessianStep>();
  EXPECT_EQ(plain.displacement, fb::HessianStep{}.displacement);
  EXPECT_EQ(plain.eri.mode, fb::EriMode::IntegralDirect);
  EXPECT_EQ(plain.scf.energy_tolerance, fb::ScfStep{}.energy_tolerance);
  // A table is a request: only the molecule it reads is filled in.
  EXPECT_FALSE(defaults->step<fb::ScfStep>().has_value());
  EXPECT_TRUE(defaults->step<fb::MoleculeStep>().has_value());

  // With no method, the analytic one, over the density-fitted chain it needs.
  auto analytic =
      fb::parse_config(with_inputs("[eri]\nmode = \"density_fitting\"\n[hessian]\n"), "", kOut);
  ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
  EXPECT_EQ(analytic->step<fb::HessianStep>()->method, fb::HessianMethod::Analytic);
  EXPECT_EQ(analytic->step<fb::HessianStep>()->eri.mode, fb::EriMode::DensityFitting);
  auto named = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\n[hessian]\nmethod = \"analytic\"\n"), "",
      kOut);
  ASSERT_TRUE(named.has_value()) << named.error().message;
  EXPECT_EQ(named->step<fb::HessianStep>()->method, fb::HessianMethod::Analytic);
}

TEST(PipelineHessian, AMalformedTableIsInvalidConfig) {
  expect_config_error(with_inputs("[hessian]\nmethod = \"finite\"\n"),
                      "[hessian] has no method 'finite' (known: 'analytic', 'numerical')");
  // The displacement is the numerical method's alone.
  expect_config_error(
      with_inputs("[eri]\nmode = \"density_fitting\"\n[hessian]\ndisplacement = 1e-3\n"),
      "[hessian] displacement is the numerical method's, and method is 'analytic'");
  expect_config_error(with_inputs("[hessian]\nmethod = \"numerical\"\nstep = 1e-3\n"),
                      "[hessian] has no option 'step'");
  for (const std::string_view h : {"0", "-1e-3", "nan", "inf"}) {
    expect_config_error(
        with_inputs(std::format("[hessian]\nmethod = \"numerical\"\ndisplacement = {}\n", h)),
        "[hessian] displacement must be finite and positive");
  }
  expect_config_error(with_inputs("[hessian]\nmethod = \"numerical\"\ndisplacement = \"1e-3\"\n"),
                      "[hessian] displacement is not a number");
  // The chain is rerun at every displacement, so none of it is loaded.
  expect_config_error(
      with_inputs("[hessian]\nmethod = \"numerical\"\n[guess]\nfrom = \"d.npy\"\n"),
      "[hessian] reruns [guess] at every geometry, so it cannot be loaded 'from' a file");
  // The molecule it reads comes from somewhere.
  expect_config_error("[basis_set]\nbasis = \"b.json\"\n[hessian]\nmethod = \"numerical\"\n",
                      "[molecule] needs 'xyz'");
}

// The analytic Hessian is density-fitted only: over integral-direct ERIs,
// the default [eri] or one written so, it is refused as Unsupported, naming
// both ways out, before anything runs.
TEST(PipelineHessian, TheAnalyticMethodRefusesIntegralDirectEris) {
  for (const std::string_view eri : {"", "[eri]\nmode = \"integral_direct\"\n"}) {
    auto calc = fb::parse_config(with_inputs(std::format("{}[hessian]\n", eri)), "", kOut);
    ASSERT_FALSE(calc.has_value());
    EXPECT_EQ(calc.error().kind, fb::ErrorKind::Unsupported);
    EXPECT_EQ(calc.error().message,
              "[hessian] method 'analytic' is density-fitted only, and [eri] mode is "
              "'integral_direct': set [eri] mode = 'density_fitting', or [hessian] method = "
              "'numerical'");
  }
}

// Beside an [optimize], the molecule the Hessian reads is the one loaded
// from final.xyz, which reads OptimizeOutput: the Hessian is ordered after
// the optimization and runs at the optimum. Both steps hold the same chain.
TEST(PipelineHessian, BesideAnOptimizeItReadsTheOptimum) {
  auto calc = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\n[scf]\nenergy_tolerance = 1e-12\n"
                  "[optimize]\n[hessian]\n"),
      "/in", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_FALSE(calc->step<fb::MoleculeStep>().has_value());
  EXPECT_EQ(calc->from<fb::MoleculeStep>(), kOut / "final.xyz");
  static_assert(std::same_as<fb::LoadInput<fb::MoleculeStep>, fb::View<fb::OptimizeOutput>>);
  const fb::HessianStep &hessian = *calc->step<fb::HessianStep>();
  const fb::OptimizeStep &optimize = *calc->step<fb::OptimizeStep>();
  EXPECT_EQ(hessian.basis_set.basis, optimize.basis_set.basis);
  EXPECT_EQ(hessian.eri.mode, optimize.eri.mode);
  EXPECT_EQ(hessian.scf.energy_tolerance, optimize.scf.energy_tolerance);
}

// Beside a [casscf_gradient] the numerical method differences that root's
// gradient: [casscf] and it are copied in with the rest of the chain.
TEST(PipelineHessian, BesideACasscfGradientItCopiesTheCasscfChain) {
  auto calc = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\n"
                  "[casscf]\nncas = 4\nnelecas = 4\nactive = [2, 3, 5, 6]\nnroots = 2\n"
                  "weights = [0.7, 0.3]\nconv_tol_grad = 1e-9\n"
                  "[casscf_gradient]\nroot = 1\n[hessian]\nmethod = \"numerical\"\n"),
      "/in", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::HessianStep &hessian = *calc->step<fb::HessianStep>();
  ASSERT_TRUE(hessian.casscf_gradient.has_value());
  EXPECT_EQ(hessian.casscf_gradient->root, 1);
  EXPECT_EQ(hessian.casscf.ncas, 4);
  EXPECT_EQ(hessian.casscf.nelecas, 4);
  EXPECT_EQ(hessian.casscf.active, (std::vector<std::int32_t>{2, 3, 5, 6}));
  EXPECT_EQ(hessian.casscf.nroots, 2);
  EXPECT_EQ(hessian.casscf.weights, (std::vector<double>{0.7, 0.3}));
  EXPECT_EQ(hessian.casscf.conv_tol_grad, 1e-9);
  // No CI guess from the config: the step sets the reference's per displaced
  // solve, and ci_guess is no TOML key.
  EXPECT_TRUE(hessian.casscf.ci_guess.empty());
  expect_config_error(with_inputs("[eri]\nmode = \"density_fitting\"\n"
                                  "[casscf]\nncas = 4\nnelecas = 4\nci_guess = [1.0]\n"),
                      "[casscf] has no option 'ci_guess'");
  // The outer [casscf] and [casscf_gradient] still run, as tables are requests.
  EXPECT_TRUE(calc->step<fb::CasscfStep>().has_value());
  EXPECT_TRUE(calc->step<fb::CasscfGradientStep>().has_value());

  // Without one, the RHF chain: no CASSCF gradient set.
  auto rhf = fb::parse_config(with_inputs("[hessian]\nmethod = \"numerical\"\n"), "", kOut);
  ASSERT_TRUE(rhf.has_value()) << rhf.error().message;
  EXPECT_FALSE(rhf->step<fb::HessianStep>()->casscf_gradient.has_value());
}

// Beside a [casscf_gradient] the analytic method -- the default -- is that
// root's analytic Hessian, over the same chain; beside an [optimize] too,
// which then holds the same CASSCF chain, and the Hessian reads the optimum.
TEST(PipelineHessian, BesideACasscfGradientTheAnalyticMethodIsTheRootsHessian) {
  constexpr std::string_view kCasscf =
      "[eri]\nmode = \"density_fitting\"\n"
      "[casscf]\nncas = 4\nnelecas = 4\nactive = [2, 3, 5, 6]\nnroots = 2\n"
      "weights = [0.5, 0.5]\n[casscf_gradient]\nroot = 1\n";
  for (const std::string_view method : {"", "method = \"analytic\"\n"}) {
    auto calc =
        fb::parse_config(with_inputs(std::format("{}[hessian]\n{}", kCasscf, method)), "", kOut);
    ASSERT_TRUE(calc.has_value()) << calc.error().message;
    const fb::HessianStep &hessian = *calc->step<fb::HessianStep>();
    EXPECT_EQ(hessian.method, fb::HessianMethod::Analytic);
    ASSERT_TRUE(hessian.casscf_gradient.has_value());
    EXPECT_EQ(hessian.casscf_gradient->root, 1);
    EXPECT_EQ(hessian.casscf.nroots, 2);
    EXPECT_EQ(hessian.casscf.weights, (std::vector<double>{0.5, 0.5}));
    EXPECT_EQ(hessian.eri.mode, fb::EriMode::DensityFitting);
  }
  auto optimized =
      fb::parse_config(with_inputs(std::format("{}[optimize]\n[hessian]\n", kCasscf)), "/in", kOut);
  ASSERT_TRUE(optimized.has_value()) << optimized.error().message;
  EXPECT_EQ(optimized->from<fb::MoleculeStep>(), kOut / "final.xyz");
  const fb::HessianStep &hessian = *optimized->step<fb::HessianStep>();
  const fb::OptimizeStep &optimize = *optimized->step<fb::OptimizeStep>();
  EXPECT_EQ(hessian.method, fb::HessianMethod::Analytic);
  ASSERT_TRUE(hessian.casscf_gradient.has_value());
  ASSERT_TRUE(optimize.casscf_gradient.has_value());
  EXPECT_EQ(hessian.casscf_gradient->root, optimize.casscf_gradient->root);
  EXPECT_EQ(hessian.casscf.active, optimize.casscf.active);
}

// The CASSCF chain, by either method, is density-fitted only, and its
// Hessian is one energy's: [gradient] beside it is refused.
TEST(PipelineHessian, TheCasscfChainRefusesIntegralDirectErisAndASecondGradient) {
  constexpr std::string_view kCasscf = "[casscf]\nncas = 4\nnelecas = 4\n[casscf_gradient]\n";
  auto analytic = fb::parse_config(with_inputs(std::format("{}[hessian]\n", kCasscf)), "", kOut);
  ASSERT_FALSE(analytic.has_value());
  EXPECT_EQ(analytic.error().kind, fb::ErrorKind::Unsupported);
  EXPECT_EQ(analytic.error().message,
            "[hessian] method 'analytic' is density-fitted only, and [eri] mode is "
            "'integral_direct': set [eri] mode = 'density_fitting', or [hessian] method = "
            "'numerical'");
  for (const std::string_view method : {"", "method = \"numerical\"\n"}) {
    expect_config_error(
        with_inputs(std::format("[eri]\nmode = \"density_fitting\"\n{}[gradient]\n[hessian]\n{}",
                                kCasscf, method)),
        "[hessian] is the Hessian of the energy of [gradient] or of [casscf_gradient], and the "
        "config names both");
  }
  // The chain's [casscf] is read as the pipeline reads it: no space, refused.
  expect_config_error(with_inputs("[eri]\nmode = \"density_fitting\"\n[casscf_gradient]\n"
                                  "[hessian]\n"),
                      "[casscf] needs 'ncas'");
}

// The ladder and the bilinearity check (#812): off for the RHF and CASSCF
// chains unless asked, on beside a [nevpt2_gradient] with every pair (the
// step caps it by the molecule's size); the keys read as written, the
// coordinates flattened row by row.
TEST(PipelineHessian, TheLadderAndItsPairsAreReadAndDefaultOnForNevpt2) {
  constexpr std::string_view kCas = "[eri]\nmode = \"density_fitting\"\n"
                                    "[casscf]\nncas = 2\nnelecas = 2\n";
  const auto hessian_of = [&](std::string_view tables) {
    auto calc = fb::parse_config(with_inputs(tables), "", kOut);
    EXPECT_TRUE(calc.has_value()) << calc.error().message;
    return calc.has_value() ? *calc->step<fb::HessianStep>() : fb::HessianStep{};
  };
  // The RHF and CASSCF chains: no ladder, no pairs, the Cartesians.
  for (const std::string_view chain :
       {std::string_view{""}, std::string_view{"[casscf_gradient]\n"}}) {
    const fb::HessianStep plain =
        hessian_of(std::format("{}{}[hessian]\nmethod = \"numerical\"\n", kCas, chain));
    EXPECT_FALSE(plain.ladder);
    ASSERT_TRUE(plain.pairs.has_value());
    EXPECT_TRUE(plain.pairs->empty());
    EXPECT_TRUE(plain.coordinates.empty());
  }
  // The CASSCF chain, asked.
  const fb::HessianStep asked = hessian_of(
      std::format("{}[casscf_gradient]\n[hessian]\nmethod = \"numerical\"\nladder = true\n"
                  "pairs = [[0, 1], [4, 2]]\n",
                  kCas));
  EXPECT_TRUE(asked.ladder);
  ASSERT_TRUE(asked.pairs.has_value());
  EXPECT_EQ(*asked.pairs, (std::vector<std::array<std::int32_t, 2>>{{0, 1}, {4, 2}}));
  // The NEVPT2 chain: the ladder over every pair by default; a list of
  // pairs, or none, as written; no ladder, no pairs.
  constexpr std::string_view kNevpt2 = "[nevpt2_gradient]\n[hessian]\nmethod = \"numerical\"\n";
  const fb::HessianStep nevpt2 = hessian_of(std::format("{}{}", kCas, kNevpt2));
  EXPECT_TRUE(nevpt2.ladder);
  EXPECT_FALSE(nevpt2.pairs.has_value());
  const fb::HessianStep listed = hessian_of(std::format("{}{}pairs = [[7, 8]]\n", kCas, kNevpt2));
  EXPECT_TRUE(listed.ladder);
  EXPECT_EQ(listed.pairs, (std::vector<std::array<std::int32_t, 2>>{{7, 8}}));
  const fb::HessianStep none = hessian_of(std::format("{}{}pairs = []\n", kCas, kNevpt2));
  ASSERT_TRUE(none.pairs.has_value());
  EXPECT_TRUE(none.pairs->empty());
  const fb::HessianStep off = hessian_of(std::format("{}{}ladder = false\n", kCas, kNevpt2));
  EXPECT_FALSE(off.ladder);
  ASSERT_TRUE(off.pairs.has_value());
  EXPECT_TRUE(off.pairs->empty());
  // Coordinates, row-major.
  const fb::HessianStep turned = hessian_of(
      std::format("{}{}coordinates = [[0.6, 0.8, 0], [-0.8, 0.6, 0], [0, 0, 1]]\n", kCas, kNevpt2));
  EXPECT_EQ(turned.coordinates, (std::vector<double>{0.6, 0.8, 0, -0.8, 0.6, 0, 0, 0, 1}));
}

TEST(PipelineHessian, AMalformedLadderIsRefused) {
  constexpr std::string_view kCas = "[eri]\nmode = \"density_fitting\"\n"
                                    "[casscf]\nncas = 2\nnelecas = 2\n";
  // The numerical method's keys, under the analytic one.
  for (const std::string_view key : {"ladder = true", "pairs = [[0, 1]]", "coordinates = [[1]]"}) {
    expect_config_error(with_inputs(std::format("{}[casscf_gradient]\n[hessian]\n{}\n", kCas, key)),
                        "is the numerical method's, and method is 'analytic'");
  }
  constexpr std::string_view kNevpt2 = "[nevpt2_gradient]\n[hessian]\nmethod = \"numerical\"\n";
  for (const std::string_view pair : {"[1, 1]", "[-1, 2]"}) {
    expect_config_error(with_inputs(std::format("{}{}pairs = [{}]\n", kCas, kNevpt2, pair)),
                        "is not two distinct coordinates");
  }
  expect_config_error(with_inputs(std::format("{}{}pairs = [[1, 2, 3]]\n", kCas, kNevpt2)),
                      "[hessian] pairs is not an array of integer pairs");
  expect_config_error(with_inputs(std::format("{}{}coordinates = [[1, 0], [0]]\n", kCas, kNevpt2)),
                      "[hessian] coordinates are 2 rows, and one has 1 numbers");
  expect_config_error(with_inputs(std::format("{}{}coordinates = [1, 0]\n", kCas, kNevpt2)),
                      "[hessian] coordinates is not an array of arrays of numbers");
  expect_config_error(with_inputs(std::format("{}{}coordinates = []\n", kCas, kNevpt2)),
                      "[hessian] coordinates is empty");
  // Pairs are judged against the ladder's bars.
  expect_config_error(
      with_inputs(std::format("{}{}ladder = false\npairs = [[0, 1]]\n", kCas, kNevpt2)),
      "[hessian] pairs are judged against the ladder's error bars: set ladder = true");
  expect_config_error(with_inputs(std::format("{}[casscf_gradient]\n[hessian]\nmethod = "
                                              "\"numerical\"\npairs = [[0, 1]]\n",
                                              kCas)),
                      "[hessian] pairs are judged against the ladder's error bars");
  // The RHF chain has no restart to measure the noise by.
  auto rhf =
      fb::parse_config(with_inputs("[hessian]\nmethod = \"numerical\"\nladder = true\n"), "", kOut);
  ASSERT_FALSE(rhf.has_value());
  EXPECT_EQ(rhf.error().kind, fb::ErrorKind::Unsupported);
  EXPECT_NE(rhf.error().message.find("the RHF chain has no such start"), std::string::npos)
      << rhf.error().message;
}

// Beside a [nevpt2_gradient] the analytic method -- the default -- is that
// variant's analytic Hessian over the same chain: no ladder and no pairs,
// and coordinates read, the one analytic Hessian that takes them.
TEST(PipelineHessian, BesideANevpt2GradientTheAnalyticMethodIsTheVariantsHessian) {
  constexpr std::string_view kCas = "[eri]\nmode = \"density_fitting\"\n"
                                    "[casscf]\nncas = 2\nnelecas = 2\n";
  for (const std::string_view variant : {"sc", "pc"}) {
    for (const std::string_view method : {"", "method = \"analytic\"\n"}) {
      auto calc = fb::parse_config(
          with_inputs(std::format("{}[nevpt2_gradient]\nvariant = \"{}\"\n[hessian]\n{}", kCas,
                                  variant, method)),
          "", kOut);
      ASSERT_TRUE(calc.has_value()) << calc.error().message;
      const fb::HessianStep &hessian = *calc->step<fb::HessianStep>();
      EXPECT_EQ(hessian.method, fb::HessianMethod::Analytic);
      ASSERT_TRUE(hessian.nevpt2_gradient.has_value());
      EXPECT_EQ(hessian.nevpt2_gradient->variant,
                variant == "pc" ? fb::Nevpt2Variant::Pc : fb::Nevpt2Variant::Sc);
      EXPECT_FALSE(hessian.casscf_gradient.has_value());
      EXPECT_EQ(hessian.casscf.ncas, 2);
      EXPECT_FALSE(hessian.ladder);
      ASSERT_TRUE(hessian.pairs.has_value());
      EXPECT_TRUE(hessian.pairs->empty());
      EXPECT_TRUE(hessian.coordinates.empty());
    }
  }
  auto turned = fb::parse_config(
      with_inputs(std::format("{}[nevpt2_gradient]\n[hessian]\ncoordinates = [[0.6, 0.8, 0], "
                              "[-0.8, 0.6, 0], [0, 0, 1]]\n",
                              kCas)),
      "", kOut);
  ASSERT_TRUE(turned.has_value()) << turned.error().message;
  EXPECT_EQ(turned->step<fb::HessianStep>()->coordinates,
            (std::vector<double>{0.6, 0.8, 0, -0.8, 0.6, 0, 0, 0, 1}));
  // The numerical method's keys stay its own.
  for (const std::string_view key : {"ladder = true", "pairs = [[0, 1]]", "displacement = 0.03"}) {
    expect_config_error(with_inputs(std::format("{}[nevpt2_gradient]\n[hessian]\n{}\n", kCas, key)),
                        "is the numerical method's, and method is 'analytic'");
  }
}

// What hessian::df_sc_nevpt2 and df_pc_nevpt2 cannot take, refused when the
// config is read, before anything runs: a state-averaged reference and an
// integral-direct [eri] (Unsupported), a frozen core (InvalidConfig) and a
// quasi-degenerate gradient (Unsupported).
TEST(PipelineHessian, TheAnalyticNevpt2HessianRefusesWhatItCannotTake) {
  const auto refused = [](std::string_view tables, fb::ErrorKind kind, std::string_view says) {
    auto calc = fb::parse_config(with_inputs(tables), "", kOut);
    ASSERT_FALSE(calc.has_value()) << says;
    EXPECT_EQ(calc.error().kind, kind) << calc.error().message;
    EXPECT_NE(calc.error().message.find(says), std::string::npos) << calc.error().message;
  };
  constexpr std::string_view kFitted = "[eri]\nmode = \"density_fitting\"\n";
  constexpr std::string_view kOne = "[casscf]\nncas = 2\nnelecas = 2\n";
  for (const std::string_view variant : {"sc", "pc"}) {
    const std::string gradient = std::format("[nevpt2_gradient]\nvariant = \"{}\"\n", variant);
    refused(std::format("{}[casscf]\nncas = 2\nnelecas = 2\nnroots = 2\n{}[hessian]\n", kFitted,
                        gradient),
            fb::ErrorKind::Unsupported,
            "[nevpt2_gradient] differentiates a single-state CASSCF, and [casscf] nroots is 2");
    for (const std::string_view eri : {"", "[eri]\nmode = \"integral_direct\"\n"}) {
      refused(std::format("{}{}{}[hessian]\n", eri, kOne, gradient), fb::ErrorKind::Unsupported,
              "'integral_direct'");
    }
    for (const std::string_view frozen : {"1", "\"auto\""}) {
      refused(std::format("{}{}{}frozen_core = {}\n[hessian]\n", kFitted, kOne, gradient, frozen),
              fb::ErrorKind::InvalidConfig,
              "[hessian] is the all-electron NEVPT2 energy's, and [nevpt2_gradient] frozen_core is "
              "not 0");
    }
    refused(std::format("{}[casscf]\nncas = 2\nnelecas = 2\nnroots = 2\n{}"
                        "quasi_degenerate = true\n[hessian]\n",
                        kFitted, gradient),
            fb::ErrorKind::Unsupported,
            "[hessian] is a state-specific NEVPT2 energy's, and [nevpt2_gradient] is "
            "quasi_degenerate");
  }
}

// examples/nevpt2_hessian.toml, as the binary reads it: the analytic SC
// Hessian over its single-state, density-fitted, all-electron CASSCF.
TEST(PipelineHessian, TheNevpt2HessianExampleParses) {
  const std::filesystem::path examples{FBF_EXAMPLES_DIR};
  // read_config makes the output directory.
  const fb::test::ScratchDir out("pipeline-hessian-example");
  auto calc = fb::read_config(examples / "nevpt2_hessian.toml", examples, out.path());
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::HessianStep &hessian = *calc->step<fb::HessianStep>();
  EXPECT_EQ(hessian.method, fb::HessianMethod::Analytic);
  ASSERT_TRUE(hessian.nevpt2_gradient.has_value());
  EXPECT_EQ(hessian.nevpt2_gradient->variant, fb::Nevpt2Variant::Sc);
  EXPECT_EQ(hessian.casscf.nroots, 1);
  EXPECT_EQ(hessian.casscf.active, (std::vector<std::int32_t>{6, 7, 8, 9}));
  EXPECT_EQ(hessian.eri.mode, fb::EriMode::DensityFitting);
  EXPECT_EQ(hessian.json_path, out / "hessian.json");
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->xyz, examples / "h2co_skew.xyz");
}

// check_displaced_casscf over a non-orthonormal AO basis, S = diag(1, 4, 1,
// 1), two active orbitals and two roots of three determinants: a solve that
// followed the reference passes; an active orbital swapped for one outside
// the space, or the roots exchanged, is Numerical, naming which.
TEST(PipelineHessian, TheDisplacedCasscfGuardRefusesASwappedOrbitalOrRoot) {
  constexpr std::size_t kNao = 4;
  const std::vector<double> s{1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  // e_0 and e_1 / 2: orthonormal in S.
  const std::vector<double> start{1, 0, 0, 0, 0, 0.5, 0, 0};
  // The same space, its orbitals exchanged and turned a little: each still in
  // it. Without S, e_1 / 2's weight would be 1/16.
  const double c = std::cos(0.01);
  const double t = std::sin(0.01);
  const std::vector<double> followed{t, 0.5 * c, 0, 0, c, -0.5 * t, 0, 0};
  const std::vector<double> reference{1, 0, 0, 0, 1, 0};
  const std::vector<double> roots{c, t, 0, -t, c, 0};
  EXPECT_TRUE(fb::check_displaced_casscf(kNao, s, start, followed, reference, roots, 2));

  // Orbital 1 swapped for e_2, outside the space.
  const std::vector<double> swapped{t, 0.5 * c, 0, 0, 0, 0, 1, 0};
  const fb::Status orbital =
      fb::check_displaced_casscf(kNao, s, start, swapped, reference, roots, 2);
  ASSERT_FALSE(orbital.has_value());
  EXPECT_EQ(orbital.error().kind, fb::ErrorKind::Numerical);
  EXPECT_NE(orbital.error().message.find("active orbital 1 of 2 left the reference's active "
                                         "space: its weight there is 0.000e+00"),
            std::string::npos)
      << orbital.error().message;

  // The two roots exchanged (a sign is no change).
  const std::vector<double> exchanged{0, 1, 0, -1, 0, 0};
  const fb::Status root =
      fb::check_displaced_casscf(kNao, s, start, followed, reference, exchanged, 2);
  ASSERT_FALSE(root.has_value());
  EXPECT_EQ(root.error().kind, fb::ErrorKind::Numerical);
  EXPECT_NE(root.error().message.find("root 0 of 2 is not the reference's root 0"),
            std::string::npos)
      << root.error().message;
  const std::vector<double> flipped{-1, 0, 0, 0, -1, 0};
  EXPECT_TRUE(fb::check_displaced_casscf(kNao, s, start, followed, reference, flipped, 2));

  // Spans that do not fit each other.
  const fb::Status sizes =
      fb::check_displaced_casscf(kNao, s, start, followed, reference, {roots.data(), 5}, 2);
  ASSERT_FALSE(sizes.has_value());
  EXPECT_EQ(sizes.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
