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
// "The PC reverse pass"). It runs at pc_floor 1e-10 (#1020), where that
// defect reads about 1500 bars. About 3 s on the RTX 3080 and 14 s on the
// gfx1200, it is the cheapest PC-QD check at the twist.

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
// Lagrangian's stationarity in the active-active rotations. PC is invariant
// under them only in exact arithmetic: S and K carry round-off r that does
// not rotate with the basis, so the residue is -<∂L/∂(S, K), [G, r]>, G the
// rotation's generator, and the bars along Sr's and Si's smallest kept
// modes amplify r (#1000). No other block sees it: the Z-vector absorbs r
// there. A CASSCF stopping point or an atomic order re-rolls r, so the
// residue varies from run to run while active_core and virtual_active
// repeat.
//
// The floor (#1020). At the default cut, max(1e-13, 100 ε √d) of σ_max, Sr
// and Si keep modes from 3.5e-13 (ket 1's Sr) up, and their spectra run on
// through 1e-9 with no gap, so no floor stands out of the spectrum itself.
// kTwistPcFloor is the pipeline's PC [hessian] floor (#923), the smallest
// at which PSB3's second-order ladder is clean (src/nevpt2/energy/README.md,
// "The conditioning floor"): every kept mode is then 1.07e-10 of σ_max or
// more, 300 times the default's smallest. It drops 19 to 48 modes per ket
// where the default drops 10 to 27, and moves E_0 by 8.5e-11 Eh
// (-245.4285464001 to -245.4285463148): a different, nearby functional
// whose Lagrangian must be just as stationary, which is all the gate asks.
//
// The bar is the residue's sensitivity to r times r's bound, at that floor.
// kRoundOffSensitivity is the residue per unit relative r: symmetric
// Gaussian r added to every own S and K before the solve, entries η max|S|
// and η max|K|, gave 2.2e-12 to 7.8e-12 at η = 1e-14 (six draws), 6.1e-11
// and 6.5e-11 at 1e-13, 6.0e-10 and 6.5e-10 at 1e-12 (gfx1200), so linear
// at 6e2 per unit η above r's own share; the largest, 7.8e2, is taken. At
// the default cut the same η = 1e-14 gave 2.3e-11 to 4.7e-11 (4.7e3, #1000).
// r's bound is √N ε, N = 400 x 36 the longest reduction into K: f3ca's and
// f3ac's consume, over the 400 determinants of CAS(6,6) and the n² of each
// W element (src/nevpt2/rdm/README.md, "Design"). The bar is 2.1e-11; with
// r as it comes the residue measured 2.1e-12.
//
// #862's defect, the kept span's rotation not taken, is far out of reach at
// the floor: 3.19e-8 with the term switched off (gfx1200), against 3.48e-10
// at the default cut, past even the library's own 1e-8 gate, so the run is
// refused (Numerical, naming the active-active block): about 1500 bars. The
// floor drops real modes (σ to 1e-10) whose transition rows the term
// carries. SC-QD at the same geometry is 8.5e-13.
constexpr double kTwistPcFloor = 1e-10;
constexpr double kRoundOffSensitivity = 7.8e2;
constexpr double kLongestReduction = 400.0 * 36.0;
const double kActiveActiveTolerance =
    kRoundOffSensitivity * std::sqrt(kLongestReduction) * std::numeric_limits<double>::epsilon();

// PC-QD state 0 at the twist, DF-SA3-CASSCF(6,6)/STO-3G over cc-pVDZ-JKFIT:
// Sr's and Si's metrics drop real modes there, and the transition rows'
// share of them (of order √σ, the own rows' σ) moved the gradient with the
// kept span's rotation, which the reverse pass used to hold still. The gate
// names the defect: no other block moved.
TEST(Nevpt2Psb3, PcQdActiveActiveGateAtTheTwist) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("nevpt2_psb3_twist");
  ASSERT_TRUE(fb::write_file(dir / "psb3.xyz", std::string{kPsb3Twisted}).has_value());
  // The steps as a config would give them, but for the floor, which no TOML
  // key sets on a QD [nevpt2_gradient].
  fb::CasscfStep casscf;
  casscf.ncas = 6;
  casscf.nelecas = 6;
  casscf.nroots = 3;
  casscf.conv_tol_grad = kConvTolGrad;
  auto calc = fb::Calculation::create(
      {fb::MoleculeStep{.xyz = dir / "psb3.xyz", .charge = 1}, fb::BasisSetStep{.basis = "sto-3g"},
       fb::EriStep{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pvdz-jkfit"},
       fb::ScfStep{.max_iterations = 400}, casscf,
       fb::Nevpt2GradientStep{.variant = fb::Nevpt2Variant::Pc,
                              .quasi_degenerate = true,
                              .state = 0,
                              .pc_floor = kTwistPcFloor}},
      []<class S>() -> fb::Result<S> { return S{}; });
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &out = calc->get<fb::Nevpt2GradientOutput>();
  ASSERT_TRUE(out.has_value()) << out.error().message;
  const auto &g = std::get<fb::nevpt2::QdPcGradient>(out->gradient);
  // X^L's residues: core_core, active_active, virtual_virtual, active_core,
  // virtual_core, virtual_active.
  const std::array<double, 6> &gate = g.lagrangian.asymmetry;
  const double active_active = gate[1];
  EXPECT_LE(active_active, kActiveActiveTolerance);
  std::cout << std::format("[ psb3     ] PC-QD state 0 at 90°, STO-3G, pc_floor {:g}: gate "
                           "active_active {:.2e} (bar {:.2e}), active_core {:.2e}, "
                           "virtual_active {:.2e}\n",
                           kTwistPcFloor, active_active, kActiveActiveTolerance, gate[3], gate[5]);
}

} // namespace
