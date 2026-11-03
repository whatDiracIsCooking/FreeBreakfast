// Host-only cases for the Nevpt2Nac step: its contract (static_asserts, as
// steps_host.cpp holds every step's), and its table read by the TOML front
// end -- the pairs (every pair by default), the variant and the two options,
// nevpt2_nac.json in the output directory, the CASSCF and the ERIs filled
// in, and every refusal at config read: a malformed table, an integral-direct
// [eri] written or filled in, a [casscf] of one root, a pair past H_eff's
// states, and an open-shell molecule. Nothing is run, so nothing touches a
// device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

// A Device step over the CASSCF and everything the QD coupling reads beside
// it, as [nevpt2_gradient]'s; the product carries one nevpt2::QdNac per pair.
static_assert(fb::PipelineStep<fb::Nevpt2NacStep>);
static_assert(fb::kRunShape<fb::Nevpt2NacStep> == fb::RunShape::Device);
static_assert(std::same_as<
              fb::StepInput<fb::Nevpt2NacStep>,
              fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput,
                       fb::EriOutput, fb::CoreHamiltonianOutput, fb::ScfOutput, fb::CasscfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::Nevpt2NacStep>, fb::Nevpt2NacOutput>);
static_assert(!fb::LoadableStep<fb::Nevpt2NacStep>);
static_assert(
    std::same_as<decltype(fb::Nevpt2NacOutput::couplings), std::vector<fb::nevpt2::QdNac>>);

constexpr std::string_view kInputs =
    "[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"b.json\"\n";
constexpr std::string_view kFitted = "[eri]\nmode = \"density_fitting\"\n";
// A state average of three roots, at the default (equal) weights.
constexpr std::string_view kSa3 = "[casscf]\nncas = 3\nnelecas = 4\nnroots = 3\n";

const std::filesystem::path kOut = "/out";

std::string with_inputs(std::string_view tables) {
  return std::string(kInputs).append(tables);
}

void expect_error(std::string_view toml, fb::ErrorKind kind, std::string_view needle) {
  auto calc = fb::parse_config(toml, "", kOut);
  ASSERT_FALSE(calc.has_value()) << toml;
  EXPECT_EQ(calc.error().kind, kind) << calc.error().message;
  EXPECT_NE(calc.error().message.find(needle), std::string::npos) << calc.error().message;
}

// [nevpt2_nac] reads its pairs, variant and options, writes nevpt2_nac.json
// into the output directory, and fills in the CASSCF, the SCF and the core
// Hamiltonian it reads. Left out, the pairs are every pair and the variant
// SC; the step resolves the pairs once it knows the CASSCF's roots.
TEST(PipelineNevpt2Nac, TheStepReadsItsPairsVariantAndOptionsAndWritesItsJson) {
  auto calc =
      fb::parse_config(with_inputs(std::format("{}{}[nevpt2_nac]\n", kFitted, kSa3)), "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::Nevpt2NacStep &step = *calc->step<fb::Nevpt2NacStep>();
  EXPECT_TRUE(step.pairs.empty());
  EXPECT_EQ(step.variant, fb::Nevpt2Variant::Sc);
  EXPECT_FALSE(step.options.mult_ediff);
  EXPECT_FALSE(step.options.use_etfs);
  EXPECT_EQ(step.json_path, kOut / "nevpt2_nac.json");
  EXPECT_TRUE(calc->step<fb::CasscfStep>().has_value());
  EXPECT_TRUE(calc->step<fb::ScfStep>().has_value());
  EXPECT_TRUE(calc->step<fb::CoreHamiltonianStep>().has_value());
  EXPECT_TRUE(calc->step<fb::PrimitivePairClassesStep>().has_value());
  // Nothing else is planned for it: H_eff and R are the step's own.
  EXPECT_FALSE(calc->step<fb::Nevpt2Step>().has_value());

  auto written = fb::parse_config(
      with_inputs(std::format("{}{}[nevpt2_nac]\npairs = [[2, 1], [0, 2]]\nvariant = \"pc\"\n"
                              "mult_ediff = true\nuse_etfs = true\n",
                              kFitted, kSa3)),
      "", kOut);
  ASSERT_TRUE(written.has_value()) << written.error().message;
  const fb::Nevpt2NacStep &chosen = *written->step<fb::Nevpt2NacStep>();
  EXPECT_EQ(chosen.pairs, (std::vector<std::array<std::int32_t, 2>>{{2, 1}, {0, 2}}));
  EXPECT_EQ(chosen.variant, fb::Nevpt2Variant::Pc);
  EXPECT_TRUE(chosen.options.mult_ediff);
  EXPECT_TRUE(chosen.options.use_etfs);

  auto sc = fb::parse_config(
      with_inputs(std::format("{}{}[nevpt2_nac]\nvariant = \"sc\"\n", kFitted, kSa3)), "", kOut);
  ASSERT_TRUE(sc.has_value()) << sc.error().message;
  EXPECT_EQ(sc->step<fb::Nevpt2NacStep>()->variant, fb::Nevpt2Variant::Sc);

  // Unequal weights are read: the coupling takes any weights.
  auto weighted =
      fb::parse_config(with_inputs(std::format("{}[casscf]\nncas = 3\nnelecas = 4\nnroots = 2\n"
                                               "weights = [0.6, 0.4]\n[nevpt2_nac]\n",
                                               kFitted)),
                       "", kOut);
  ASSERT_TRUE(weighted.has_value()) << weighted.error().message;
}

TEST(PipelineNevpt2Nac, AMalformedTableIsInvalidConfig) {
  const std::string head = std::format("{}{}", kFitted, kSa3);
  for (const auto &[table, needle] :
       {std::pair{"pairs = [[0, 1]]\nstate = 1\n", "[nevpt2_nac] has no option 'state'"},
        std::pair{"quasi_degenerate = true\n", "[nevpt2_nac] has no option 'quasi_degenerate'"},
        std::pair{"scheme = \"sc\"\n", "[nevpt2_nac] has no option 'scheme'"},
        std::pair{"json_path = \"n.json\"\n", "[nevpt2_nac] has no option 'json_path'"},
        std::pair{"pairs = []\n", "[nevpt2_nac] pairs is empty"},
        std::pair{"pairs = [0, 1]\n", "[nevpt2_nac] pairs is not an array of integer pairs"},
        std::pair{"pairs = [[0, 1, 2]]\n", "[nevpt2_nac] pairs is not an array of integer pairs"},
        std::pair{"pairs = [[1, 1]]\n", "[nevpt2_nac] pair [1, 1] is not two distinct QD states"},
        std::pair{"pairs = [[-1, 0]]\n", "[nevpt2_nac] pair [-1, 0] is not two distinct QD states"},
        std::pair{"variant = \"nevpt2\"\n",
                  "[nevpt2_nac] has no variant 'nevpt2' (known: 'sc', 'pc')"},
        std::pair{"variant = 1\n", "[nevpt2_nac] variant is not a string"},
        std::pair{"mult_ediff = 1\n", "[nevpt2_nac] mult_ediff is not a boolean"},
        std::pair{"use_etfs = \"yes\"\n", "[nevpt2_nac] use_etfs is not a boolean"}}) {
    expect_error(with_inputs(std::format("{}[nevpt2_nac]\n{}", head, table)),
                 fb::ErrorKind::InvalidConfig, needle);
  }
  // A filled-in [casscf] is read from an empty table, which names no space.
  expect_error(with_inputs(std::format("{}[nevpt2_nac]\n", kFitted)), fb::ErrorKind::InvalidConfig,
               "[casscf] needs 'ncas'");
}

// Across tables, before anything runs: a coupling needs two roots of the
// CASSCF, and a pair names QD states H_eff has, one per root.
TEST(PipelineNevpt2Nac, ACasscfWithoutTheStatesIsInvalidConfig) {
  expect_error(
      with_inputs(std::format("{}[casscf]\nncas = 3\nnelecas = 4\n[nevpt2_nac]\n", kFitted)),
      fb::ErrorKind::InvalidConfig,
      "[nevpt2_nac] couples QD states of a state average, and [casscf] nroots is 1");
  expect_error(
      with_inputs(std::format("{}{}[nevpt2_nac]\npairs = [[0, 1], [2, 3]]\n", kFitted, kSa3)),
      fb::ErrorKind::InvalidConfig,
      "[nevpt2_nac] pair [2, 3] names a QD state past H_eff's 3, one per [casscf] root");
}

// Well formed, and not what the QD coupling takes: Unsupported, naming the
// table to change.
TEST(PipelineNevpt2Nac, IntegralDirectErisAndOpenShellsAreUnsupported) {
  // An integral-direct [eri], written or filled in.
  for (const std::string_view eri : {"", "[eri]\nmode = \"integral_direct\"\n"}) {
    expect_error(with_inputs(std::format("{}{}[nevpt2_nac]\n", eri, kSa3)),
                 fb::ErrorKind::Unsupported,
                 "[nevpt2_nac] runs over density fitting only, and [eri] mode is "
                 "'integral_direct'");
  }
  expect_error(std::format("[molecule]\nxyz = \"w.xyz\"\nmultiplicity = 3\n[basis_set]\nbasis = "
                           "\"b.json\"\n{}{}[nevpt2_nac]\n",
                           kFitted, kSa3),
               fb::ErrorKind::Unsupported,
               "[nevpt2_nac] is implemented for a closed-shell singlet only, and [molecule] "
               "multiplicity is 3");
}

} // namespace
