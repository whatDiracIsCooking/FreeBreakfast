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
// reference integral-direct (slow, ~74 s on gfx1200 and ~33 s on the 3080, so
// skipped in fast mode), and density-fitted over cc-pVTZ-JKFIT in the fast
// set. With the clustered
// preconditioner on ([cis] cluster_gap), with RIS-J ([cis] ris), and with
// RIS-JK ([cis] ris_exchange), H2O STO-3G integral-direct and cc-pVDZ fitted
// still match the reference to 1e-6. TDA-ris ([cis] method = "ris") is held
// to the reference's own TDA-ris over the same two RIS fits, never to full
// CIS: H2O cc-pVDZ over the exact RHF, singlets and triplets, the energies to
// 1e-6 and the singlets' f to 1e-5, and benzene cc-pVDZ over the
// cc-pVDZ-JKFIT RHF, timed beside the full DF-CIS over the same products. Then the refusals: too few or too many states, a negative
// cluster_gap, one beside ris, a TDA-ris θ that is not positive, a molecule
// that is not a closed-shell singlet, and an open-shell ScfOutput. Runs the
// device chain, so needs a GPU. The step's contract, and [cis] under a
// density-fitted [eri] planned when the config is read, are test/pipeline's
// (steps_host.cpp, config_host.cpp).

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

// The issue's bar against cis.json's PySCF roots, which are good to ~1e-14,
// for the excitation energies and the reference energy.
constexpr double kEnergyTolerance = 1e-6;
// The Ritz vectors are orthonormal to the Davidson's own rounding.
constexpr double kOrthonormalTolerance = 1e-10;
// The issue's bar against cis.json's oscillator strengths, which PySCF's own
// and the dense A's eigenvectors agree on to 2e-11 or better here.
constexpr double kOscillatorTolerance = 1e-6;
// TDA-ris's bar for the oscillator strengths, against the dense RIS A's
// eigenvectors: the issue's 1e-5.
constexpr double kRisOscillatorTolerance = 1e-5;
// CODATA 2018, for the benzene case's printed errors only.
constexpr double kEvPerHartree = 27.211386245988;
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
// strengths, each group's summed f to f_tolerance and every |μ|² that f;
// without them (triplets, and singlets past the dipole's d), f and μ exactly
// zero.
void expect_matches(const fb::CisOutput &out, const fb::json &entry, const fb::json &roots,
                    std::string_view what, double f_tolerance = kOscillatorTolerance) {
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
      EXPECT_NEAR(sum, f[g], f_tolerance) << what << " group " << g;
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
// cases run the chain fitted, so the step takes cis::DfSigma. hf_ccpvtz runs
// the chain only: its by-hand rerun was a second exact-K-at-f Davidson,
// ~54 s on gfx1200 and ~23 s on the 3080, holding nothing the other cases'
// reruns do not.
TEST_P(CisStepReference, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  // Exact K at f: ~74 s on gfx1200, ~33 s on the 3080. The fitted f-shell
  // case below stands for it in the fast set.
  const bool exact_f = GetParam().name == "hf_ccpvtz";
  if (exact_f) {
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
  if (exact_f) {
    return;
  }

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

// The clustered preconditioner, [cis] cluster_gap = 0.1, through the config
// and then by hand for both spins, over H2O STO-3G integral-direct
// (cis::Sigma) and cc-pVDZ fitted (cis::DfSigma). Fewer states than the
// reference's, so the guess, 2 n_states unit vectors, is not the whole
// space and the preconditioner runs; the gap groups at least one pair of the
// guess's Δε, checked from the ScfOutput's orbital energies. Each root against
// the reference's to 1e-6, the bar unchanged.
TEST(CisStep, ClusteredMatchesPyscf) {
  FBF_REQUIRE_GPU();
  constexpr double kGap = 0.1;
  constexpr std::int32_t kStates = 3;
  for (const std::string_view name : {"h2o_sto3g", "h2o_ccpvdz_df"}) {
    const fb::json &entry = reference_case(name);
    const fb::test::ScratchDir dir("cis_step");
    auto calc = fb::read_config(
        write_config(dir, entry,
                     std::format("[cis]\nn_states = {}\ncluster_gap = {}\n", kStates, kGap)),
        dir.path(), dir / "out");
    ASSERT_TRUE(calc.has_value()) << calc.error().message;
    ASSERT_EQ(calc->step<fb::CisStep>()->options.cluster_gap, kGap);
    fb::NodeResources node;
    const fb::Status ran = calc->run(node);
    ASSERT_TRUE(ran.has_value()) << name << ": " << ran.error().message;

    // The guess's 2 n_states lowest Δε, of which the gap must cluster some.
    const auto &eps = calc->get<fb::ScfOutput>()->mo_energies;
    const auto n_occ = entry.at("occupied").get<std::size_t>();
    std::vector<double> delta;
    for (std::size_t i = 0; i < n_occ; ++i) {
      for (std::size_t a = n_occ; a < eps.size(); ++a) {
        delta.push_back(eps[a] - eps[i]);
      }
    }
    std::ranges::sort(delta);
    std::int32_t pairs = 0;
    for (std::size_t g = 1; g < 2 * kStates; ++g) {
      pairs += delta[g] - delta[g - 1] <= kGap ? 1 : 0;
    }
    ASSERT_GT(pairs, 0) << name;
    ASSERT_LT(2 * static_cast<std::size_t>(kStates), delta.size()) << name;

    const auto check = [&](const fb::CisOutput &out, const fb::json &roots, std::string_view what) {
      const auto want = roots.at("energies").get<std::vector<double>>();
      ASSERT_EQ(out.states.size(), static_cast<std::size_t>(kStates)) << what;
      for (std::size_t r = 0; r < out.states.size(); ++r) {
        std::cout << name << " " << what << " state " << r << ": "
                  << out.states[r].excitation_energy - want[r] << " hartree, " << out.iterations
                  << " iterations\n";
        EXPECT_NEAR(out.states[r].excitation_energy, want[r], kEnergyTolerance)
            << name << " " << what << " state " << r;
      }
    };
    const auto &chain = calc->get<fb::CisOutput>();
    ASSERT_TRUE(chain.has_value());
    check(*chain, entry.at("singlets"), "config singlet");

    fb::DeviceResources &res = node.device(0);
    for (const cis::Spin spin : {cis::Spin::Singlet, cis::Spin::Triplet}) {
      const bool triplet = spin == cis::Spin::Triplet;
      const fb::CisStep step{
          .options = {.n_states = kStates, .spin = spin, .cluster_gap = kGap},
      };
      const auto by_hand = step.run(step_input(*calc), res);
      ASSERT_TRUE(by_hand.has_value()) << name << ": " << by_hand.error().message;
      check(*by_hand, entry.at(triplet ? "triplets" : "singlets"),
            triplet ? "by hand triplet" : "by hand singlet");
    }
  }
}

// The RIS-J preconditioner, [cis] ris = true, through the config and then by
// hand for both spins, over H2O STO-3G integral-direct (cis::Sigma) and
// cc-pVDZ fitted (cis::DfSigma), each with its own RIS fit beside the
// [eri]'s. Fewer states than the reference's, so the guess is not the whole
// space and the preconditioner runs. Each root against the reference's to
// 1e-6, the bar unchanged: RIS changes the correction, not the operator the
// Davidson converges on.
TEST(CisStep, RisMatchesPyscf) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kStates = 3;
  for (const std::string_view name : {"h2o_sto3g", "h2o_ccpvdz_df"}) {
    const fb::json &entry = reference_case(name);
    const fb::test::ScratchDir dir("cis_step");
    auto calc = fb::read_config(
        write_config(dir, entry, std::format("[cis]\nn_states = {}\nris = true\n", kStates)),
        dir.path(), dir / "out");
    ASSERT_TRUE(calc.has_value()) << calc.error().message;
    ASSERT_TRUE(calc->step<fb::CisStep>()->options.ris.has_value());
    fb::NodeResources node;
    const fb::Status ran = calc->run(node);
    ASSERT_TRUE(ran.has_value()) << name << ": " << ran.error().message;

    const auto check = [&](const fb::CisOutput &out, const fb::json &roots, std::string_view what) {
      const auto want = roots.at("energies").get<std::vector<double>>();
      ASSERT_EQ(out.states.size(), static_cast<std::size_t>(kStates)) << what;
      for (std::size_t r = 0; r < out.states.size(); ++r) {
        std::cout << name << " " << what << " state " << r << ": "
                  << out.states[r].excitation_energy - want[r] << " hartree, " << out.iterations
                  << " iterations\n";
        EXPECT_NEAR(out.states[r].excitation_energy, want[r], kEnergyTolerance)
            << name << " " << what << " state " << r;
      }
    };
    const auto &chain = calc->get<fb::CisOutput>();
    ASSERT_TRUE(chain.has_value());
    check(*chain, entry.at("singlets"), "config singlet");

    fb::DeviceResources &res = node.device(0);
    for (const cis::Spin spin : {cis::Spin::Singlet, cis::Spin::Triplet}) {
      const bool triplet = spin == cis::Spin::Triplet;
      const fb::CisStep step{
          .options = {.n_states = kStates, .spin = spin, .ris = cis::RisOptions{}},
      };
      const auto by_hand = step.run(step_input(*calc), res);
      ASSERT_TRUE(by_hand.has_value()) << name << ": " << by_hand.error().message;
      check(*by_hand, entry.at(triplet ? "triplets" : "singlets"),
            triplet ? "by hand triplet" : "by hand singlet");
    }
  }
}

// RIS-JK, [cis] ris_exchange = true beside ris, at its default GMRES budget,
// the same way over the same two: the exchange changes the correction again,
// and still not the operator.
TEST(CisStep, RisExchangeMatchesPyscf) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kStates = 3;
  for (const std::string_view name : {"h2o_sto3g", "h2o_ccpvdz_df"}) {
    const fb::json &entry = reference_case(name);
    const fb::test::ScratchDir dir("cis_step");
    auto calc = fb::read_config(
        write_config(
            dir, entry,
            std::format("[cis]\nn_states = {}\nris = true\nris_exchange = true\n", kStates)),
        dir.path(), dir / "out");
    ASSERT_TRUE(calc.has_value()) << calc.error().message;
    const auto &ris = calc->step<fb::CisStep>()->options.ris;
    ASSERT_TRUE(ris.has_value());
    ASSERT_TRUE(ris->exchange);
    fb::NodeResources node;
    const fb::Status ran = calc->run(node);
    ASSERT_TRUE(ran.has_value()) << name << ": " << ran.error().message;

    const auto check = [&](const fb::CisOutput &out, const fb::json &roots, std::string_view what) {
      const auto want = roots.at("energies").get<std::vector<double>>();
      ASSERT_EQ(out.states.size(), static_cast<std::size_t>(kStates)) << what;
      for (std::size_t r = 0; r < out.states.size(); ++r) {
        std::cout << name << " " << what << " state " << r << ": "
                  << out.states[r].excitation_energy - want[r] << " hartree, " << out.iterations
                  << " iterations\n";
        EXPECT_NEAR(out.states[r].excitation_energy, want[r], kEnergyTolerance)
            << name << " " << what << " state " << r;
      }
    };
    const auto &chain = calc->get<fb::CisOutput>();
    ASSERT_TRUE(chain.has_value());
    check(*chain, entry.at("singlets"), "config singlet");

    fb::DeviceResources &res = node.device(0);
    for (const cis::Spin spin : {cis::Spin::Singlet, cis::Spin::Triplet}) {
      const bool triplet = spin == cis::Spin::Triplet;
      const fb::CisStep step{
          .options = {.n_states = kStates, .spin = spin, .ris = cis::RisOptions{.exchange = true}},
      };
      const auto by_hand = step.run(step_input(*calc), res);
      ASSERT_TRUE(by_hand.has_value()) << name << ": " << by_hand.error().message;
      check(*by_hand, entry.at(triplet ? "triplets" : "singlets"),
            triplet ? "by hand triplet" : "by hand singlet");
    }
  }
}

// TDA-ris, [cis] method = "ris", over H2O cc-pVDZ's exact RHF: the config
// runs the chain with an integral-direct [eri] the operator does not read,
// then the step runs again by hand for both spins. Against the reference's
// TDA-ris over the same two RIS fits, the energies to 1e-6 and the singlets'
// f to 1e-5; never against full CIS.
TEST(CisStep, TdaRisMatchesReference) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_ccpvdz_ris");
  const fb::json &singlets = entry.at("singlets");
  const auto n_states = static_cast<std::int32_t>(singlets.at("energies").size());
  const fb::test::ScratchDir dir("cis_step");
  auto calc = fb::read_config(
      write_config(dir, entry, std::format("[cis]\nn_states = {}\nmethod = \"ris\"\n", n_states)),
      dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  ASSERT_EQ(calc->step<fb::CisStep>()->options.method, cis::Method::Ris);
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  ASSERT_TRUE(std::holds_alternative<fb::IntegralDirect>(calc->get<fb::EriOutput>()->eri));
  const auto &chain = calc->get<fb::CisOutput>();
  ASSERT_TRUE(chain.has_value());
  expect_matches(*chain, entry, singlets, "config singlet", kRisOscillatorTolerance);

  fb::DeviceResources &res = node.device(0);
  for (const cis::Spin spin : {cis::Spin::Singlet, cis::Spin::Triplet}) {
    const bool triplet = spin == cis::Spin::Triplet;
    const fb::json &roots = entry.at(triplet ? "triplets" : "singlets");
    const fb::CisStep step{
        .options = {.n_states = static_cast<std::int32_t>(roots.at("energies").size()),
                    .spin = spin,
                    .method = cis::Method::Ris},
    };
    const auto by_hand = step.run(step_input(*calc), res);
    ASSERT_TRUE(by_hand.has_value()) << by_hand.error().message;
    expect_matches(*by_hand, entry, roots, triplet ? "by hand triplet" : "by hand singlet",
                   kRisOscillatorTolerance);
  }
}

// Benzene cc-pVDZ over the cc-pVDZ-JKFIT RHF: TDA-ris's singlets against the
// reference's TDA-ris, the energies to 1e-6 and f to 1e-5, and the full
// DF-CIS over the same products, its energies against benzene_ccpvdz_df's to
// 1e-6 (its f is #511's), each step's best of three runs timed. Prints each
// TDA-ris root's error against the full DF-CIS reference, which is RIS's
// approximation error and is not held to anything. Under a second on the
// RTX 3080, so in the fast set.
TEST(CisStep, BenzeneTdaRisMatchesReference) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("benzene_ccpvdz_df_ris");
  const fb::json &full = reference_case("benzene_ccpvdz_df");
  const fb::json &roots = entry.at("singlets");
  const auto n_states = static_cast<std::int32_t>(roots.at("energies").size());
  const fb::test::ScratchDir dir("cis_step");
  auto calc = fb::read_config(write_config(dir, entry, ""), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  ASSERT_TRUE(std::holds_alternative<fb::DensityFitted>(calc->get<fb::EriOutput>()->eri));
  fb::DeviceResources &res = node.device(0);

  // Each run is the whole step: Δε, the sigma's create (TDA-ris's two fits,
  // or the MO blocks of [eri]'s B), the Davidson and the dipoles.
  const auto timed = [&](const cis::Method method, const std::int32_t states) {
    const fb::CisStep step{.options = {.n_states = states, .method = method}};
    fb::Result<fb::CisOutput> out = step.run(step_input(*calc), res);
    double best = std::numeric_limits<double>::infinity();
    for (std::int32_t rep = 0; rep < 3 && out.has_value(); ++rep) {
      const auto start = std::chrono::steady_clock::now();
      out = step.run(step_input(*calc), res);
      const std::chrono::duration<double> took = std::chrono::steady_clock::now() - start;
      best = std::min(best, took.count());
    }
    return std::pair{std::move(out), best};
  };
  const auto [ris, ris_seconds] = timed(cis::Method::Ris, n_states);
  ASSERT_TRUE(ris.has_value()) << ris.error().message;
  expect_matches(*ris, entry, roots, "tda-ris singlet", kRisOscillatorTolerance);
  const fb::json &full_roots = full.at("singlets");
  const auto [cis_full, full_seconds] =
      timed(cis::Method::Full, static_cast<std::int32_t>(full_roots.at("energies").size()));
  ASSERT_TRUE(cis_full.has_value()) << cis_full.error().message;
  const auto want = full_roots.at("energies").get<std::vector<double>>();
  ASSERT_EQ(cis_full->states.size(), want.size());
  for (std::size_t r = 0; r < want.size(); ++r) {
    EXPECT_NEAR(cis_full->states[r].excitation_energy, want[r], kEnergyTolerance)
        << "full df-cis state " << r;
  }
  for (std::size_t r = 0; r < std::min(ris->states.size(), want.size()); ++r) {
    std::cout << std::format("benzene state {}: tda-ris {:.10f}, df-cis {:.10f}, error {:+.6f} "
                             "hartree ({:+.4f} eV)\n",
                             r, ris->states[r].excitation_energy, want[r],
                             ris->states[r].excitation_energy - want[r],
                             (ris->states[r].excitation_energy - want[r]) * kEvPerHartree);
  }
  std::cout << std::format("benzene wall clock: tda-ris {:.3f} s ({} iterations), full df-cis "
                           "{:.3f} s ({} iterations)\n",
                           ris_seconds, ris->iterations, full_seconds, cis_full->iterations);
}

void expect_refused(const fb::Result<fb::CisOutput> &refused, fb::ErrorKind kind,
                    std::string_view needle) {
  ASSERT_FALSE(refused.has_value()) << needle;
  EXPECT_EQ(refused.error().kind, kind) << refused.error().message;
  EXPECT_NE(refused.error().message.find(needle), std::string::npos) << refused.error().message;
}

// Each refused when the step runs, naming the step, before anything is
// allocated: n_states below 1 or past H2O / STO-3G's n_ov = 10; a negative
// cluster_gap, and a positive one beside ris; a TDA-ris θ not positive; a triplet
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
  expect_refused(fb::CisStep{.options = {.cluster_gap = -0.5}}.run(step_input(*calc), res),
                 fb::ErrorKind::InvalidConfig, "cis: cluster_gap is -0.5, not at least 0");
  expect_refused(fb::CisStep{.options = {.cluster_gap = 0.1, .ris = cis::RisOptions{}}}.run(
                     step_input(*calc), res),
                 fb::ErrorKind::InvalidConfig, "cis: ris and a cluster_gap of 0.1 each pick");
  expect_refused(fb::CisStep{.options = {.method = cis::Method::Ris, .theta_j = 0.0}}.run(
                     step_input(*calc), res),
                 fb::ErrorKind::InvalidConfig, "cis: theta_j is 0, not finite and positive");
  expect_refused(fb::CisStep{.options = {.method = cis::Method::Ris, .theta_k = -1.0}}.run(
                     step_input(*calc), res),
                 fb::ErrorKind::InvalidConfig, "cis: theta_k is -1, not finite and positive");

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

} // namespace
