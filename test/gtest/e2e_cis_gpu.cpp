// Benzene in cc-pVDZ end to end, the CIS milestone's capstone: a Calculation
// read from a TOML runs the molecule, the basis, the density-fitted [eri] over
// cc-pVDZ-JKFIT, the RHF and the Cis step with cis::DfSigma and RIS-J ([cis]
// ris); then the step runs again by hand over the same products with the
// diagonal preconditioner and with RIS-JK. n_ov = 21 · 93 = 1953, far past a
// dense diagonalization of A. Each run's lowest five singlets (six roots, the
// last level an E pair) against cis.json's PySCF DF RHF + TDA over the same
// aux set (benzene_ccpvdz_df) to 1e-6 hartree, and their oscillator
// strengths, summed per degenerate group, to 1e-5; the RHF energy to 1e-6.
// Each run's Davidson iteration count is printed, for the PR that compares
// the three preconditioners.
//
// The exact reference (benzene_ccpvdz) is not run: the integral-direct sigma
// takes the same three runs to 282 s on the RTX 3080, past the 120 s an entry
// is given (test/CMakeLists.txt).
//
// Slow, so skipped in fast mode, and RUN_SERIAL (test/gtest/CMakeLists.txt).
// Needs a GPU.

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

// The issue's bars against cis.json's PySCF roots, which agree with a dense
// diagonalization of A to 3e-14 (energies) and 7e-8 (benzene's bright E1u
// pair's f).
constexpr double kEnergyTolerance = 1e-6;
constexpr double kOscillatorTolerance = 1e-5;
// The Ritz vectors are orthonormal to the Davidson's own rounding.
constexpr double kOrthonormalTolerance = 1e-10;
// f is (2/3) ω |μ|² of the state's own ω and μ, to rounding.
constexpr double kRoundingTolerance = 1e-14;

// The SCF converged tightly, so the reference's error is not the 1e-6 the
// excitations are held to; integral-direct, every Fock build full too.
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

// The reference's molecule as an XYZ file in dir, and a config naming it in
// its basis, fitted over the reference's aux basis when it has one, with the
// [cis] table after it.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view cis_table) {
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
      entry.at("basis").get<std::string>(), eri, kScf, aux.is_null() ? kDirectScf : "", cis_table);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The reference's singlets: the shape, the reference energy, the excitation
// energies ascending, the amplitudes orthonormal, each f its own (2/3) ω |μ|²,
// and each degenerate group's summed f.
void expect_matches(const fb::CisOutput &out, const fb::json &entry, std::string_view what) {
  const fb::json &roots = entry.at("singlets");
  const auto want = roots.at("energies").get<std::vector<double>>();
  const auto n_occ = entry.at("occupied").get<std::int32_t>();
  const auto n_virt = entry.at("nao").get<std::int32_t>() - n_occ;
  EXPECT_EQ(out.n_occ, n_occ) << what;
  EXPECT_EQ(out.n_virt, n_virt) << what;
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
    const auto &mu = state.transition_dipole;
    EXPECT_NEAR(state.oscillator_strength,
                2.0 / 3.0 * state.excitation_energy *
                    ((mu[0] * mu[0]) + (mu[1] * mu[1]) + (mu[2] * mu[2])),
                kRoundingTolerance)
        << what << " state " << r;
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
  const auto f = roots.at("oscillator_strengths").get<std::vector<double>>();
  const fb::json &groups = roots.at("groups");
  ASSERT_EQ(groups.size(), f.size()) << what;
  for (std::size_t g = 0; g < f.size(); ++g) {
    double sum = 0.0;
    for (const fb::json &k : groups[g]) {
      sum += out.states[k.get<std::size_t>()].oscillator_strength;
    }
    std::cout << what << " group " << g << ": f " << sum << ", " << sum - f[g] << " off\n";
    EXPECT_NEAR(sum, f[g], kOscillatorTolerance) << what << " group " << g;
  }
}

// The config runs the chain to the Cis step with RIS-J; then the step runs by
// hand over the same products with the diagonal preconditioner and with
// RIS-JK. Every run against the reference, and the three iteration counts
// printed on one line.
TEST(E2eCis, BenzeneCcpvdzDensityFitted) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  constexpr std::string_view kName = "benzene_ccpvdz_df";
  const fb::json &entry = reference_case(kName);
  // The reference's whole groups: the lowest five levels, six roots.
  const auto n_states = static_cast<std::int32_t>(entry.at("singlets").at("energies").size());
  const fb::test::ScratchDir dir("e2e_cis");
  auto calc = fb::read_config(
      write_config(dir, entry, std::format("[cis]\nn_states = {}\nris = true\n", n_states)),
      dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  ASSERT_TRUE(calc->step<fb::CisStep>()->options.ris.has_value());
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << kName << ": " << ran.error().message;
  const auto &ris_j = calc->get<fb::CisOutput>();
  ASSERT_TRUE(ris_j.has_value());
  expect_matches(*ris_j, entry, std::format("{} RIS-J (config)", kName));

  const fb::StepInput<fb::CisStep> in{*calc->get<fb::MoleculeOutput>(),
                                      *calc->get<fb::BasisSetOutput>(),
                                      *calc->get<fb::SphericalTransformOutput>(),
                                      *calc->get<fb::PrimitivePairClassesOutput>(),
                                      *calc->get<fb::EriOutput>(),
                                      *calc->get<fb::ScfOutput>()};
  fb::DeviceResources &res = node.device(0);

  const auto diagonal = fb::CisStep{.options = {.n_states = n_states}}.run(in, res);
  ASSERT_TRUE(diagonal.has_value()) << kName << " diagonal: " << diagonal.error().message;
  expect_matches(*diagonal, entry, std::format("{} diagonal", kName));

  const auto ris_jk =
      fb::CisStep{.options = {.n_states = n_states, .ris = cis::RisOptions{.exchange = true}}}.run(
          in, res);
  ASSERT_TRUE(ris_jk.has_value()) << kName << " RIS-JK: " << ris_jk.error().message;
  expect_matches(*ris_jk, entry, std::format("{} RIS-JK", kName));

  std::cout << kName << " Davidson iterations, " << n_states << " roots: diagonal "
            << diagonal->iterations << ", RIS-J " << ris_j->iterations << ", RIS-JK "
            << ris_jk->iterations << "\n";
}

} // namespace
