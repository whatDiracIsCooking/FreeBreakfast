// GPU case for the Nac step through the real registry: a TOML config read,
// planned and run by Calculation -- the DF-RHF, the equal-weight DF-SA-CASSCF
// from its orbitals and the couplings between its roots -- its nac.json
// against PySCF's DF SA-CASSCF couplings (data/casscf_nac/casscf_nac.json,
// the h2co_ccpvdz_sa2 case: H2CO, SA-2 CAS(4,3), cc-pVDZ / cc-pVDZ-JKFIT):
// the roots' energies, then h_IJ, d^CSF and d_IJ of the one default pair
// (0, 1) to 1e-8, as mcscf_nac_gpu.cpp holds mcscf::df_nac, up to the sign
// s_0 s_1 the two CI phases carry. Then the step run again by hand over the
// same products with pairs = [[1, 0]] and use_etfs: the reverse coupling and
// the option, against the reference's (1, 0) under the same sign; and a pair
// of equal roots, refused before anything runs.
//
// One [eri] serves the whole pipeline, so the CASSCF starts from the DF-RHF's
// orbitals where the reference's starts from the exact RHF's: the same
// active orbitals, and the same converged DF-SA-CASSCF, as in
// pipeline_casscf_gradient_gpu.cpp. Runs the device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

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

// The issue's bar, as mcscf_nac_gpu.cpp holds mcscf::df_nac.
constexpr double kNacTolerance = 1e-8;
constexpr double kEnergyTolerance = 1e-8;
// The CASSCF's convergence, as mcscf_nac_gpu.cpp's: the couplings are the
// CASSCF's only at a stationary point.
constexpr double kConvTolGrad = 1e-10;

const fb::json &reference_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read =
        fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_nac" / "casscf_nac.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf_nac/casscf_nac.json";
  static const fb::json kNone;
  return kNone;
}

// The reference's coupling (bra, ket).
const fb::json &reference_coupling(const fb::json &entry, std::int32_t bra, std::int32_t ket) {
  for (const fb::json &coupling : entry.at("couplings")) {
    if (coupling.at("bra").get<std::int32_t>() == bra &&
        coupling.at("ket").get<std::int32_t>() == ket) {
      return coupling;
    }
  }
  ADD_FAILURE() << "no coupling (" << bra << ", " << ket << ")";
  static const fb::json kNone;
  return kNone;
}

// An n_atom x 3 array of rows, flat.
std::vector<double> flat(const fb::json &rows) {
  std::vector<double> out;
  for (const auto &row : rows.get<std::vector<std::vector<double>>>()) {
    out.insert(out.end(), row.begin(), row.end());
  }
  return out;
}

// max_i |ours_i - sign want_i|, each element also EXPECTed to kNacTolerance.
double compare(std::span<const double> ours, std::span<const double> want, double sign,
               std::string_view what) {
  EXPECT_EQ(ours.size(), want.size()) << what;
  double error = 0.0;
  for (std::size_t i = 0; i < std::min(ours.size(), want.size()); ++i) {
    EXPECT_NEAR(ours[i], sign * want[i], kNacTolerance) << what << " " << i;
    error = std::max(error, std::abs(ours[i] - (sign * want[i])));
  }
  return error;
}

// The reference's molecule as molecule.xyz in @p dir (bohr there, Angstrom
// here), and run.toml naming it in the reference's basis, over its fitting
// set, with the reference's state average in [casscf] (its start the Casci
// step's, as the reference's) and @p nac after it.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view nac) {
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
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nconv_tol_grad = {:g}\n{}",
      entry.at("charge").get<std::int32_t>(), entry.at("basis").get<std::string>(),
      entry.at("aux_basis").at("name").get<std::string>(), entry.at("ncas").get<std::int32_t>(),
      entry.at("nelecas").get<std::int32_t>(), entry.at("weights").size(), kConvTolGrad, nac);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

TEST(PipelineNac, JsonMatchesPyscf) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2co_ccpvdz_sa2");
  const auto start_time = std::chrono::steady_clock::now();
  const fb::test::ScratchDir dir("pipeline_nac");
  auto calc = fb::read_config(write_config(dir, entry, "[nac]\n"), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  const auto written = fb::read_json(dir / "out" / "nac.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_FALSE(written->at("mult_ediff").get<bool>());
  EXPECT_FALSE(written->at("use_etfs").get<bool>());
  const auto energies = written->at("energies").get<std::vector<double>>();
  const auto want_energies = entry.at("energies").get<std::vector<double>>();
  ASSERT_EQ(energies.size(), want_energies.size());
  for (std::size_t r = 0; r < energies.size(); ++r) {
    EXPECT_NEAR(energies[r], want_energies[r], kEnergyTolerance) << "root " << r;
  }
  // The default pairs: every I < J, here the one.
  ASSERT_EQ(written->at("couplings").size(), 1U);
  const fb::json &ours = written->at("couplings").at(0);
  EXPECT_EQ(ours.at("bra").get<std::int32_t>(), 0);
  EXPECT_EQ(ours.at("ket").get<std::int32_t>(), 1);
  const fb::json &want = reference_coupling(entry, 0, 1);
  EXPECT_NEAR(ours.at("energy_gap").get<double>(), want.at("energy_gap").get<double>(),
              kEnergyTolerance);
  EXPECT_GT(ours.at("zvector").at("iterations").get<std::int32_t>(), 0);
  // s_0 s_1 from h's overlap with the reference.
  const std::vector<double> h = flat(ours.at("h"));
  const std::vector<double> want_h = flat(want.at("h"));
  const double sign =
      std::inner_product(h.begin(), h.end(), want_h.begin(), 0.0) < 0.0 ? -1.0 : 1.0;
  const double h_error = compare(h, want_h, sign, "h");
  const double csf_error = compare(flat(ours.at("csf")), flat(want.at("csf")), sign, "csf");
  const double total_error = compare(flat(ours.at("total")), flat(want.at("total")), sign, "total");

  // The product is what the file holds.
  const auto &out = calc->get<fb::NacOutput>();
  ASSERT_TRUE(out.has_value());
  ASSERT_EQ(out->pairs.size(), 1U);
  EXPECT_EQ(out->pairs[0], (std::array<std::int32_t, 2>{0, 1}));
  ASSERT_EQ(out->couplings.size(), 1U);
  EXPECT_EQ(out->couplings[0].total, flat(ours.at("total")));

  // The step by hand over the same products: the reverse pair, with
  // electron-translation factors, against the reference's (1, 0), whose CI
  // phases carry the same s_0 s_1.
  const fb::StepInput<fb::NacStep> in{*calc->get<fb::MoleculeOutput>(),
                                      *calc->get<fb::BasisSetOutput>(),
                                      *calc->get<fb::PrimitivePairClassesOutput>(),
                                      *calc->get<fb::EriOutput>(),
                                      *calc->get<fb::CoreHamiltonianOutput>(),
                                      *calc->get<fb::CasscfOutput>()};
  fb::DeviceResources &res = node.device(0);
  const auto reverse = fb::NacStep{.pairs = {{1, 0}}, .options = {.use_etfs = true}}.run(in, res);
  ASSERT_TRUE(reverse.has_value()) << reverse.error().message;
  ASSERT_EQ(reverse->couplings.size(), 1U);
  EXPECT_EQ(reverse->pairs[0], (std::array<std::int32_t, 2>{1, 0}));
  const fb::json &want_reverse = reference_coupling(entry, 1, 0);
  const fb::mcscf::Nac &d = reverse->couplings[0];
  const double etf_error = compare(d.total, flat(want_reverse.at("etf")), sign, "etf (1, 0)");
  compare(d.numerator, flat(want_reverse.at("h")), sign, "h (1, 0)");
  compare(d.csf, flat(want_reverse.at("csf")), sign, "csf (1, 0)");

  // A pair of equal roots, refused before anything runs.
  const auto refused = fb::NacStep{.pairs = {{1, 1}}}.run(in, res);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig) << refused.error().message;

  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start_time;
  std::cout << std::format("[ nac ] h2co_ccpvdz_sa2 (0, 1): h {:.2e}, csf {:.2e}, total {:.2e}; "
                           "(1, 0) etf {:.2e} ({:.2f} s)\n",
                           h_error, csf_error, total_error, etf_error, seconds.count());
}

} // namespace
