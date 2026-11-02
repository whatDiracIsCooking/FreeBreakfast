// Host-only cases for the Nac step: its contract (static_asserts, as
// steps_host.cpp holds every step's), and its table read by the TOML front
// end -- the pairs (every pair by default) and the two options, nac.json in
// the output directory, the CASSCF and the ERIs filled in, and every refusal:
// a malformed table, an integral-direct [eri] written or filled in, a
// [casscf] of one root, a pair past its roots, and unequal weights. Nothing
// is run, so nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

// A Device step over the CASSCF and everything df_nac reads beside it, as
// [casscf_gradient]'s; the product carries one mcscf::Nac per pair.
static_assert(fb::PipelineStep<fb::NacStep>);
static_assert(fb::kRunShape<fb::NacStep> == fb::RunShape::Device);
static_assert(
    std::same_as<fb::StepInput<fb::NacStep>,
                 fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput,
                          fb::EriOutput, fb::CoreHamiltonianOutput, fb::CasscfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::NacStep>, fb::NacOutput>);
static_assert(!fb::LoadableStep<fb::NacStep>);
static_assert(std::same_as<decltype(fb::NacOutput::couplings), std::vector<fb::mcscf::Nac>>);

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

// [nac] reads its pairs and options, writes nac.json into the output
// directory, and fills in the CASSCF and the core Hamiltonian it reads.
// Left out, the pairs are every pair, which the step resolves once it knows
// the CASSCF's roots.
TEST(PipelineNac, TheStepReadsItsPairsAndOptionsAndWritesItsJson) {
  auto calc = fb::parse_config(with_inputs(std::format("{}{}[nac]\n", kFitted, kSa3)), "", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::NacStep &step = *calc->step<fb::NacStep>();
  EXPECT_TRUE(step.pairs.empty());
  EXPECT_FALSE(step.options.mult_ediff);
  EXPECT_FALSE(step.options.use_etfs);
  EXPECT_EQ(step.json_path, kOut / "nac.json");
  EXPECT_TRUE(calc->step<fb::CasscfStep>().has_value());
  EXPECT_TRUE(calc->step<fb::CoreHamiltonianStep>().has_value());
  EXPECT_TRUE(calc->step<fb::PrimitivePairClassesStep>().has_value());

  auto written = fb::parse_config(
      with_inputs(std::format("{}{}[nac]\npairs = [[1, 0], [0, 2]]\nmult_ediff = true\n"
                              "use_etfs = true\n",
                              kFitted, kSa3)),
      "", kOut);
  ASSERT_TRUE(written.has_value()) << written.error().message;
  const fb::NacStep &chosen = *written->step<fb::NacStep>();
  EXPECT_EQ(chosen.pairs, (std::vector<std::array<std::int32_t, 2>>{{1, 0}, {0, 2}}));
  EXPECT_TRUE(chosen.options.mult_ediff);
  EXPECT_TRUE(chosen.options.use_etfs);

  // Equal weights written out are the default's.
  auto equal =
      fb::parse_config(with_inputs(std::format("{}{}weights = [0.5, 0.5]\nnroots = 2\n[nac]\n",
                                               kFitted, "[casscf]\nncas = 3\nnelecas = 4\n")),
                       "", kOut);
  ASSERT_TRUE(equal.has_value()) << equal.error().message;
}

TEST(PipelineNac, AMalformedTableIsInvalidConfig) {
  const std::string head = std::format("{}{}", kFitted, kSa3);
  for (const auto &[table, needle] :
       {std::pair{"pairs = [[0, 1]]\nroot = 1\n", "[nac] has no option 'root'"},
        std::pair{"json_path = \"n.json\"\n", "[nac] has no option 'json_path'"},
        std::pair{"pairs = []\n", "[nac] pairs is empty"},
        std::pair{"pairs = [0, 1]\n", "[nac] pairs is not an array of integer pairs"},
        std::pair{"pairs = [[0, 1, 2]]\n", "[nac] pairs is not an array of integer pairs"},
        std::pair{"pairs = [[0, 1.5]]\n", "[nac] pairs is not an array of 32-bit integer pairs"},
        std::pair{"pairs = \"all\"\n", "[nac] pairs is not an array of integer pairs"},
        std::pair{"pairs = [[1, 1]]\n", "[nac] pair [1, 1] is not two distinct roots"},
        std::pair{"pairs = [[-1, 0]]\n", "[nac] pair [-1, 0] is not two distinct roots"},
        std::pair{"mult_ediff = 1\n", "[nac] mult_ediff is not a boolean"},
        std::pair{"use_etfs = \"yes\"\n", "[nac] use_etfs is not a boolean"}}) {
    expect_error(with_inputs(std::format("{}[nac]\n{}", head, table)), fb::ErrorKind::InvalidConfig,
                 needle);
  }
  // A filled-in [casscf] is read from an empty table, which names no space.
  expect_error(with_inputs(std::format("{}[nac]\n", kFitted)), fb::ErrorKind::InvalidConfig,
               "[casscf] needs 'ncas'");
}

// Across tables, before anything runs: a coupling needs two roots of the
// CASSCF, and a pair names roots it has.
TEST(PipelineNac, ACasscfWithoutTheRootsIsInvalidConfig) {
  expect_error(with_inputs(std::format("{}[casscf]\nncas = 3\nnelecas = 4\n[nac]\n", kFitted)),
               fb::ErrorKind::InvalidConfig,
               "[nac] couples roots of a state average, and [casscf] nroots is 1");
  expect_error(with_inputs(std::format("{}{}[nac]\npairs = [[0, 1], [2, 3]]\n", kFitted, kSa3)),
               fb::ErrorKind::InvalidConfig,
               "[nac] pair [2, 3] names a root past [casscf] nroots 3");
}

// Well formed, and not what df_nac takes: Unsupported, naming the table to
// change.
TEST(PipelineNac, IntegralDirectErisAndUnequalWeightsAreUnsupported) {
  // An integral-direct [eri], written or filled in.
  for (const std::string_view eri : {"", "[eri]\nmode = \"integral_direct\"\n"}) {
    expect_error(with_inputs(std::format("{}{}[nac]\n", eri, kSa3)), fb::ErrorKind::Unsupported,
                 "[nac] runs over density fitting only, and [eri] mode is 'integral_direct'");
  }
  expect_error(with_inputs(std::format("{}[casscf]\nncas = 3\nnelecas = 4\nnroots = 2\n"
                                       "weights = [0.6, 0.4]\n[nac]\n",
                                       kFitted)),
               fb::ErrorKind::Unsupported,
               "[nac] takes a state average at equal weights only, and [casscf] weights range "
               "from 0.4 to 0.6");
}

} // namespace
