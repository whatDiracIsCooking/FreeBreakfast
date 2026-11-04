// PC-QD-NEVPT2 where the PC metrics have no gap at their cut (#853): PSB3
// (the penta-2,4-dieniminium cation), DF-SA3-CASSCF(6,6)/cc-pVDZ over
// cc-pVDZ-JKFIT at its ground-state CASSCF minimum, through the real
// registry. Every root's Sr and Si metric runs continuously through the cut
// (src/nevpt2/energy/README.md, "PC-NEVPT2"), which the gap rule refused
// until #853. data/nevpt2_psb3/psb3_fc.json holds this tree's own central
// differences of PC-QD state 0's energy (RTX 3080, --preset default, every
// point re-converged; steps 2e-3, 1e-3 and 5e-4 bohr, Richardson over the
// two finest, its error bar the distance from the Richardson over the two
// coarser): the analytic gradient must sit within max(2 x that bar,
// 3 sigma_E / h_min) on every component, sigma_E the energies' measured noise
// (src/nevpt2/energy/README.md, "PC-NEVPT2"). That is the check that the PC
// reverse pass, which drops the rank-projector terms, stays the energy's
// derivative when real modes are cut. The state is accepted at all (the rule),
// and its energy is the differences' reference point's.
//
// One run is about 7 s on the RTX 3080, unloaded; under the tier's load it is
// past slow_test.h's ten seconds, and pipeline_nevpt2_gpu.cpp's HF CAS(10,7)
// (PcAcceptedThroughAContinuum) is the fast case of the same rule.

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

// The differences were taken at this CASSCF convergence.
constexpr double kConvTolGrad = 1e-10;
// The state's energy against the differences' reference point: the same
// config on the same card, apart by the CASSCF's convergence and the
// einsums' atomics.
constexpr double kEnergyTolerance = 1e-9;

TEST(Nevpt2Psb3, PcQdGradientMatchesNumerical) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const auto read =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_psb3" / "psb3_fc.json");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  const fb::json &ref = *read;

  const fb::test::ScratchDir dir("nevpt2_psb3");
  std::string xyz = std::format("{}\nPSB3\n", ref.at("atoms").size());
  for (const fb::json &atom : ref.at("atoms")) {
    const auto coord = atom.at("angstrom").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0], coord[1], coord[2]);
  }
  ASSERT_TRUE(fb::write_file(dir / "psb3.xyz", xyz).has_value());
  std::string active;
  for (const auto t : ref.at("active").get<std::vector<std::int32_t>>()) {
    active += std::format("{}{}", active.empty() ? "" : ", ", t);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"psb3.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n[scf]\nmax_iterations = 400\n"
      "[casscf]\nncas = 6\nnelecas = 6\nnroots = 3\nactive = [{}]\nconv_tol_grad = {:g}\n"
      "[nevpt2_gradient]\nquasi_degenerate = true\nvariant = \"pc\"\nstate = {}\n",
      ref.at("charge").get<std::int32_t>(), ref.at("basis").get<std::string>(),
      ref.at("aux_basis").get<std::string>(), active, kConvTolGrad,
      ref.at("state").get<std::int32_t>());
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto written = fb::read_json(dir / "out" / "nevpt2_gradient.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  const fb::json &out = *written;

  EXPECT_NEAR(out.at("energy").get<double>(), ref.at("energy").get<double>(), kEnergyTolerance);
  const auto total = out.at("total").get<std::vector<std::vector<double>>>();
  const auto richardson = ref.at("richardson").get<std::vector<std::vector<double>>>();
  const auto bar = ref.at("error_bar").get<std::vector<std::vector<double>>>();
  const double floor = ref.at("noise_floor").get<double>();
  ASSERT_EQ(total.size(), richardson.size());
  ASSERT_EQ(bar.size(), richardson.size());
  double worst = 0.0;
  for (std::size_t a = 0; a < total.size(); ++a) {
    for (std::size_t k = 0; k < 3; ++k) {
      const double miss = std::abs(total[a][k] - richardson[a][k]);
      const double allowed = std::max(2.0 * bar[a][k], floor);
      EXPECT_LE(miss, allowed) << "atom " << a << " axis " << k << ": analytic " << total[a][k]
                               << ", numerical " << richardson[a][k] << " +- " << bar[a][k];
      worst = std::max(worst, miss / allowed);
    }
  }
  std::cout << std::format("[ psb3     ] PC-QD state 0: E {:.10f}, worst |analytic - numerical| "
                           "{:.2f} of its allowance\n",
                           out.at("energy").get<double>(), worst);
}

} // namespace
