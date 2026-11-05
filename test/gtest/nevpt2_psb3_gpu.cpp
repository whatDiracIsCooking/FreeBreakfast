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
//
// PcQdActiveActiveGateAtTheTwist (#862) is the fast one here: PSB3 twisted to
// 90° about C13=C14, in STO-3G, where PC-QD state 0's Lagrangian was not
// stationary in the active-active rotations until the reverse pass took the
// kept span's rotation into the dropped modes (src/nevpt2/energy/README.md,
// "The PC reverse pass"). About 3 s on the RTX 3080 and 14 s on the gfx1200,
// it is the cheapest PC-QD check at the twist.

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

// PSB3 with C13=C14 twisted rigidly to 90° at its FC length (#853's 90°
// geometry), in angstrom.
constexpr std::string_view kPsb3Twisted = R"(14
PSB3, C13=C14 twisted 90 degrees
N     -0.0454402492     0.0522740911    -0.0000000000
C      1.1276796768     0.5920624626     0.0000000000
C      2.3467584180    -0.1268535537     0.0000000000
C      3.5352478218     0.5543991513     0.0000000000
C      4.2381819494     0.9573272501    -1.1898886726
C      5.4137881019     1.6311951607    -1.1318525401
H      1.1448444823     1.6750816790    -0.0000000000
H      2.3229137518    -1.2094227332    -0.0000000000
H      3.9874420661     0.8136009171     0.9516444335
H      3.7996110498     0.7059345067    -2.1474371187
H     -0.1682915976    -0.9466686957    -0.0000000000
H     -0.8778598767     0.6155991196    -0.0000000000
H      5.8725712624     1.8941737519    -0.1873080265
H      5.9322594717     1.9283875648    -2.0319280688
)";

// The gate's active-active residue (max |X^L - X^Lᵀ| on that block), the
// Lagrangian's stationarity in the active-active rotations. Measured on the
// RTX 3080: 3.5e-10 before #862's fix (three runs), 3.3e-12 to 7.3e-12 after
// (four runs); SC-QD state 1 at the same geometry 8.5e-13.
constexpr double kActiveActiveTolerance = 3e-11;

// PC-QD state 0 at the twist, DF-SA3-CASSCF(6,6)/STO-3G over cc-pVDZ-JKFIT:
// Sr's and Si's metrics drop real modes there, and the transition rows'
// share of them (of order √σ, the own rows' σ) moved the gradient with the
// kept span's rotation, which the reverse pass used to hold still. The gate
// names the defect: no other block moved.
TEST(Nevpt2Psb3, PcQdActiveActiveGateAtTheTwist) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("nevpt2_psb3_twist");
  ASSERT_TRUE(fb::write_file(dir / "psb3.xyz", std::string{kPsb3Twisted}).has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"psb3.xyz\"\ncharge = 1\n[basis_set]\nbasis = \"sto-3g\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n"
                  "[scf]\nmax_iterations = 400\n"
                  "[casscf]\nncas = 6\nnelecas = 6\nnroots = 3\nconv_tol_grad = {:g}\n"
                  "[nevpt2_gradient]\nquasi_degenerate = true\nvariant = \"pc\"\nstate = 0\n",
                  kConvTolGrad);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto written = fb::read_json(dir / "out" / "nevpt2_gradient.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  const fb::json &gate = written->at("asymmetry");

  const double active_active = gate.at("active_active").get<double>();
  EXPECT_LE(active_active, kActiveActiveTolerance);
  std::cout << std::format("[ psb3     ] PC-QD state 0 at 90°, STO-3G: gate active_active {:.2e} "
                           "(bar {:.0e}), active_core {:.2e}, virtual_active {:.2e}\n",
                           active_active, kActiveActiveTolerance,
                           gate.at("active_core").get<double>(),
                           gate.at("virtual_active").get<double>());
}

} // namespace
