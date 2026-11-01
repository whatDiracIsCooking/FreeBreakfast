// Host-only cases for the Hessian step: its contract (static_asserts, as
// steps_host.cpp holds every step's), and its table read by the TOML front
// end -- the method (analytic by default) and the displacement, the chain
// copied in from the config's own tables, every refusal (the analytic method
// over integral-direct ERIs among them), and beside an [optimize], the
// molecule it reads loaded from the final.xyz that step writes, so it runs
// after it; the CASSCF chain copied in beside a [casscf_gradient], and its
// refusals; and the guard on a displaced CASSCF, check_displaced_casscf, fed
// an active orbital swapped out and roots swapped. Nothing is run, so
// nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

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
  // The outer [casscf] and [casscf_gradient] still run, as tables are requests.
  EXPECT_TRUE(calc->step<fb::CasscfStep>().has_value());
  EXPECT_TRUE(calc->step<fb::CasscfGradientStep>().has_value());

  // Without one, the RHF chain: no CASSCF gradient set.
  auto rhf = fb::parse_config(with_inputs("[hessian]\nmethod = \"numerical\"\n"), "", kOut);
  ASSERT_TRUE(rhf.has_value()) << rhf.error().message;
  EXPECT_FALSE(rhf->step<fb::HessianStep>()->casscf_gradient.has_value());
}

// The CASSCF chain is the numerical method's alone, and one gradient is
// differenced.
TEST(PipelineHessian, TheCasscfChainRefusesTheAnalyticMethodAndASecondGradient) {
  constexpr std::string_view kCasscf = "[eri]\nmode = \"density_fitting\"\n"
                                       "[casscf]\nncas = 4\nnelecas = 4\n[casscf_gradient]\n";
  auto analytic = fb::parse_config(with_inputs(std::format("{}[hessian]\n", kCasscf)), "", kOut);
  ASSERT_FALSE(analytic.has_value());
  EXPECT_EQ(analytic.error().kind, fb::ErrorKind::Unsupported);
  EXPECT_EQ(analytic.error().message,
            "[hessian] method 'analytic' is the RHF's, and [casscf_gradient] asks for a CASSCF "
            "root's: set [hessian] method = 'numerical'");
  expect_config_error(
      with_inputs(std::format("{}[gradient]\n[hessian]\nmethod = \"numerical\"\n", kCasscf)),
      "[hessian] differences the gradient of [gradient] or of [casscf_gradient], and the config "
      "names both");
  // The chain's [casscf] is read as the pipeline reads it: no space, refused.
  expect_config_error(with_inputs("[eri]\nmode = \"density_fitting\"\n[casscf_gradient]\n"
                                  "[hessian]\nmethod = \"numerical\"\n"),
                      "[casscf] needs 'ncas'");
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
