// The Fci step against PySCF's fci.FCI (data/fci/fci.json,
// tools/emit/fci_step_reference.py), through the real registry: a config
// naming [fci] plans the whole chain to the ScfOutput and solves the full CI over
// every orbital on the GPU sigma. Compared: each root's total energy and
// <S^2>, and, for a root of no degenerate level, its natural occupations.
// Both are invariant under any rotation of the orbitals, so they depend on
// neither SCF's convergence nor the orbitals' signs. Every 1-RDM is symmetric
// with trace N. A case at a multiplicity other than 1 is an FCI of that spin
// over the RHF orbitals: the Scf step refuses such a molecule, so the chain
// runs as the singlet and the step again by hand with the molecule at the
// multiplicity. Then a space past the device's memory, refused before it is
// allocated. Runs the device chain, so needs a GPU.

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

static_assert(fb::PipelineStep<fb::FciStep>);
static_assert(fb::kRunShape<fb::FciStep> == fb::RunShape::Device);
static_assert(std::same_as<
              fb::StepInput<fb::FciStep>,
              fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                       fb::PrimitivePairClassesOutput, fb::CoreHamiltonianOutput, fb::ScfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::FciStep>, fb::FciOutput>);

// The issue's bar, for the energies and <S^2>.
constexpr double kEnergyTolerance = 1e-8;
// The natural occupations move to first order in the CI vector.
constexpr double kOccupationTolerance = 1e-7;

const fb::json &reference() {
  static const fb::json doc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "fci" / "fci.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return doc;
}

const fb::json &reference_case(std::string_view name) {
  for (const fb::json &entry : reference().at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in fci/fci.json";
  static const fb::json kNone;
  return kNone;
}

// The reference's molecule written as an XYZ file in dir, and a config naming
// it in the given basis with the given tables after it.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view basis, std::string_view tables) {
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
      entry.at("charge").get<std::int32_t>(), basis, tables);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The eigenvalues of a symmetric n x n matrix, descending: cyclic Jacobi.
std::vector<double> eigenvalues(std::vector<double> a, std::size_t n) {
  for (std::int32_t sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        off += a[(p * n) + q] * a[(p * n) + q];
      }
    }
    if (off < 1e-30) {
      break;
    }
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        const double apq = a[(p * n) + q];
        if (apq == 0.0) {
          continue;
        }
        const double theta = (a[(q * n) + q] - a[(p * n) + p]) / (2.0 * apq);
        const double t =
            std::copysign(1.0, theta) / (std::abs(theta) + std::sqrt((theta * theta) + 1.0));
        const double c = 1.0 / std::sqrt((t * t) + 1.0);
        const double s = t * c;
        for (std::size_t k = 0; k < n; ++k) {
          const double akp = a[(k * n) + p];
          const double akq = a[(k * n) + q];
          a[(k * n) + p] = (c * akp) - (s * akq);
          a[(k * n) + q] = (s * akp) + (c * akq);
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double apk = a[(p * n) + k];
          const double aqk = a[(q * n) + k];
          a[(p * n) + k] = (c * apk) - (s * aqk);
          a[(q * n) + k] = (s * apk) + (c * aqk);
        }
      }
    }
  }
  std::vector<double> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = a[(i * n) + i];
  }
  std::ranges::sort(values, std::greater<>{});
  return values;
}

struct Case {
  std::string_view name;
  // Under slow_test.h: H2O 6-31G's 1.66M determinants.
  bool slow;
};

// ctest names a discovered case by this, so `ctest -R h2o_631g` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

class PipelineFci : public ::testing::TestWithParam<Case> {};

TEST_P(PipelineFci, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  if (GetParam().slow) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &entry = reference_case(GetParam().name);
  const fb::json &roots = entry.at("roots");
  const fb::test::ScratchDir dir("pipeline_fci");
  auto calc = fb::read_config(write_config(dir, entry, entry.at("basis").get<std::string>(),
                                           std::format("[fci]\nnroots = {}\n", roots.size())),
                              dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &chain = calc->get<fb::FciOutput>();
  ASSERT_TRUE(chain.has_value());

  // The step again by hand over the same products, but the molecule's
  // multiplicity, when the reference's is not 1.
  std::optional<fb::FciOutput> by_hand;
  if (const auto multiplicity = entry.value("multiplicity", std::int32_t{1}); multiplicity != 1) {
    fb::DeviceResources &res = node.device(0);
    const fb::ScopedDeviceIndex on{res.dev_idx()};
    const fb::Molecule &singlet = calc->get<fb::MoleculeOutput>()->molecule;
    const fb::MoleculeOutput molecule{
        fb::Molecule(res, singlet.atoms(), singlet.charge(), multiplicity)};
    auto ran_by_hand = calc->step<fb::FciStep>()->run(
        fb::StepInput<fb::FciStep>{
            molecule, *calc->get<fb::BasisSetOutput>(), *calc->get<fb::SphericalTransformOutput>(),
            *calc->get<fb::PrimitivePairClassesOutput>(), *calc->get<fb::CoreHamiltonianOutput>(),
            *calc->get<fb::ScfOutput>()},
        res);
    ASSERT_TRUE(ran_by_hand.has_value()) << ran_by_hand.error().message;
    by_hand.emplace(std::move(*ran_by_hand));
  }
  const fb::FciOutput &fci = by_hand ? *by_hand : *chain;

  EXPECT_EQ(fci.nmo, entry.at("nmo").get<std::int32_t>());
  EXPECT_EQ(fci.n_electrons, entry.at("electrons").get<std::int32_t>());
  const auto n = static_cast<std::size_t>(fci.nmo);
  ASSERT_EQ(fci.energies.size(), roots.size());
  ASSERT_EQ(fci.spin_squares.size(), roots.size());
  ASSERT_EQ(fci.rdm1.size(), roots.size() * n * n);
  for (std::size_t r = 0; r < roots.size(); ++r) {
    const fb::json &root = roots.at(r);
    EXPECT_NEAR(fci.energies[r], root.at("energy").get<double>(), kEnergyTolerance) << "root " << r;
    EXPECT_NEAR(fci.spin_squares[r], root.at("s2").get<double>(), kEnergyTolerance) << "root " << r;

    const std::vector<double> gamma(fci.rdm1.begin() + static_cast<std::ptrdiff_t>(r * n * n),
                                    fci.rdm1.begin() +
                                        static_cast<std::ptrdiff_t>((r + 1) * n * n));
    double trace = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
      trace += gamma[(t * n) + t];
      for (std::size_t u = 0; u < n; ++u) {
        EXPECT_NEAR(gamma[(t * n) + u], gamma[(u * n) + t], 1e-12) << "root " << r;
      }
    }
    EXPECT_NEAR(trace, fci.n_electrons, 1e-10) << "root " << r;
    if (root.at("degenerate").get<bool>()) {
      continue; // its 1-RDM depends on the rotation within the level
    }
    const auto expected = root.at("natural_occupations").get<std::vector<double>>();
    const std::vector<double> ours = eigenvalues(gamma, n);
    ASSERT_EQ(ours.size(), expected.size());
    for (std::size_t k = 0; k < n; ++k) {
      EXPECT_NEAR(ours[k], expected[k], kOccupationTolerance) << "root " << r << ", " << k;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineFci,
                         ::testing::Values(Case{.name = "h2o_sto3g", .slow = false},
                                           Case{.name = "n2_sto3g", .slow = false},
                                           Case{.name = "h2o_631g", .slow = true},
                                           Case{.name = "h2o_sto3g_triplet", .slow = false}),
                         [](const auto &info) { return std::string(info.param.name); });

// H2O in cc-pVDZ: 10 electrons in 24 orbitals, C(24, 5)^2 = 1.8e9
// determinants, whose sigma scratch alone is terabytes. Refused when the step
// runs, before anything of it is allocated, and the error is the step's
// product.
TEST(PipelineFci, ASpacePastTheDeviceMemoryIsRefusedUpFront) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_sto3g");
  const fb::test::ScratchDir dir("pipeline_fci");
  auto calc =
      fb::read_config(write_config(dir, entry, "cc-pvdz", "[fci]\n"), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::Allocation);
  EXPECT_NE(ran.error().message.find("fci: 10 electrons in 24 orbitals"), std::string::npos)
      << ran.error().message;
  ASSERT_TRUE(calc->get<fb::ScfOutput>().has_value());
  EXPECT_FALSE(calc->get<fb::FciOutput>().has_value());
}

// An open-shell ScfOutput, n_alpha != n_beta, is refused when the step runs,
// naming the step, before the memory estimate its electron count would feed.
// The rest of the chain is water's in STO-3G; the open shell's orbitals are
// never read.
TEST(PipelineFci, AnOpenShellScfOutputIsUnsupported) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_sto3g");
  const fb::test::ScratchDir dir("pipeline_fci");
  auto calc =
      fb::read_config(write_config(dir, entry, "sto-3g", "[fci]\n"), dir.path(), dir / "out");
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
  const auto refused = calc->step<fb::FciStep>()->run(
      fb::StepInput<fb::FciStep>{*calc->get<fb::MoleculeOutput>(), *calc->get<fb::BasisSetOutput>(),
                                 *calc->get<fb::SphericalTransformOutput>(),
                                 *calc->get<fb::PrimitivePairClassesOutput>(),
                                 *calc->get<fb::CoreHamiltonianOutput>(), open},
      res);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::Unsupported);
  EXPECT_NE(refused.error().message.find("fci: only a closed-shell reference"), std::string::npos)
      << refused.error().message;
}

// A multiplicity that does not fit the electrons is refused when the step
// runs, naming it, before the memory estimate: 2S of the other parity from
// water's 10 electrons, N_alpha above its 7 orbitals, or 2S above the
// electrons. The step run by hand with the molecule at each.
TEST(PipelineFci, AMultiplicityThatDoesNotFitIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_sto3g");
  const fb::test::ScratchDir dir("pipeline_fci");
  auto calc =
      fb::read_config(write_config(dir, entry, "sto-3g", "[fci]\n"), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const fb::Molecule &singlet = calc->get<fb::MoleculeOutput>()->molecule;
  for (const std::int32_t multiplicity : {2, 11, 13}) {
    const fb::MoleculeOutput molecule{
        fb::Molecule(res, singlet.atoms(), singlet.charge(), multiplicity)};
    const auto refused = calc->step<fb::FciStep>()->run(
        fb::StepInput<fb::FciStep>{
            molecule, *calc->get<fb::BasisSetOutput>(), *calc->get<fb::SphericalTransformOutput>(),
            *calc->get<fb::PrimitivePairClassesOutput>(), *calc->get<fb::CoreHamiltonianOutput>(),
            *calc->get<fb::ScfOutput>()},
        res);
    ASSERT_FALSE(refused.has_value()) << "multiplicity " << multiplicity;
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(refused.error().message.find(
                  std::format("fci: multiplicity {} does not fit 10", multiplicity)),
              std::string::npos)
        << refused.error().message;
  }
}

} // namespace
