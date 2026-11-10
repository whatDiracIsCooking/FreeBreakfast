// GPU cases for hessian::df_qd_eigenvector_response: term (ii) of the
// QD-NEVPT2 nuclear Hessian (docs/nevpt2/nevpt2.tex Eq. qd-hess), the QD
// eigenvector response, from the interstate couplings h_PQ alone, on
// data/qd_nevpt2/'s cases from this tree's own DF-SA-CASSCF through the
// registry:
//
// - h2co_skew_ccpvdz_sa3 (C1, three roots at equal weights), SC and PC: the
//   one several-W call over every trace-zero W^PQ returns h_PQ equal to the
//   per-pair nevpt2::qd_sc_coupling / qd_pc_coupling, each converged on its
//   own, within kCouplingTol on every component. Term (ii)'s value and
//   dR_P/dR_A are closed forms over those couplings, gated through the whole
//   Hessian (#965, #966); their size, the sum rule and the gauge are printed.
// - h2co_ccpvdz_sa2 made degenerate (root 1's CASSCF energy moved so that
//   H_eff's diagonal is equal; its off-diagonal vanishes by C2v symmetry):
//   Numerical naming both QD states, SC and PC, from the energies alone.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.mcscf;
import free_breakfast.nevpt2;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "test_workspaces.h"

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bar, absolute (Eh/bohr): the several-W call's h_PQ against the
// per-pair call's, two independently converged Z-vector solves at 1e-10.
constexpr double kCouplingTol = 1e-9;
// The reference SA-CASSCF's ||g|| and |Delta E|.
constexpr double kConvTolGrad = 1e-10;
constexpr double kConvTol = 1e-12;

double max_abs(std::span<const double> v) {
  return std::ranges::fold_left(
      v, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

// One case's DF-SA-CASSCF through the registry, and the solution the QD
// calls take.
struct QdCase {
  std::unique_ptr<fb::NodeResources> node;
  std::optional<fb::Calculation> calc;
  std::optional<fb::mcscf::Solution> solution;
  std::int32_t ncore = 0;
  std::int32_t ncas = 0;

  [[nodiscard]] fb::DeviceResources &res() const { return node->device(0); }
  [[nodiscard]] const fb::Molecule &molecule() const {
    return calc->get<fb::MoleculeOutput>()->molecule;
  }
  [[nodiscard]] const fb::BasisSet &basis() const { return calc->get<fb::BasisSetOutput>()->basis; }
  [[nodiscard]] std::span<const fb::PrimitivePairClass> classes() const {
    return calc->get<fb::PrimitivePairClassesOutput>()->classes;
  }
  [[nodiscard]] const fb::DensityFitted &fit() const {
    return std::get<fb::DensityFitted>(calc->get<fb::EriOutput>()->eri);
  }
  [[nodiscard]] const fb::SphericalMatrixAO &h() const {
    return calc->get<fb::CoreHamiltonianOutput>()->h;
  }

  [[nodiscard]] fb::Result<fb::hessian::QdEigenvectorResponse> response(const bool pc) const {
    return fb::hessian::df_qd_eigenvector_response(res(), molecule(), basis(), classes(), fit().aux,
                                                   h(), fit().b, fit().l.data(), *solution, ncore,
                                                   ncas, pc);
  }

  // qd_sc_coupling's (or qd_pc_coupling's) h_PQ, eigenvalues and residual.
  struct Coupling {
    std::vector<double> total;
    std::vector<double> eigenvalues;
    double residual = 0.0;
  };
  [[nodiscard]] fb::Result<Coupling> coupling(const std::int32_t p, const std::int32_t q,
                                              const bool pc) const {
    const auto of = [](const auto &g) -> fb::Result<Coupling> {
      if (!g) {
        return std::unexpected(g.error());
      }
      return Coupling{.total = g->total,
                      .eigenvalues = g->lagrangian.eigenvalues,
                      .residual = g->lagrangian.multipliers.residual};
    };
    if (pc) {
      return of(fb::nevpt2::qd_pc_coupling(res(), molecule(), basis(), classes(), fit().aux, h(),
                                           fit().b, fit().l.data(), *solution, ncore, ncas, p, q,
                                           0));
    }
    return of(fb::nevpt2::qd_sc_coupling(res(), molecule(), basis(), classes(), fit().aux, h(),
                                         fit().b, fit().l.data(), *solution, ncore, ncas, p, q, 0));
  }

  // H_eff's correlation elements C, n x n [M n + N], SC's or PC's.
  [[nodiscard]] std::vector<double> totals(const bool pc) const {
    std::vector<double> out;
    const fb::Status got = fb::nevpt2::qd_inputs(
        res(), molecule(), basis(), fit().aux, h(), fit().b, *solution, ncore, ncas,
        [&](const fb::nevpt2::MoBlocks &blocks,
            const std::span<const fb::nevpt2::EnergyInputs> pairs) -> fb::Status {
          if (pc) {
            const auto e = fb::nevpt2::pc_effective_hamiltonian(res(), blocks, pairs, 0);
            if (!e) {
              return std::unexpected(e.error());
            }
            std::ranges::transform(*e, std::back_inserter(out),
                                   &fb::nevpt2::PcClassEnergies::total);
          } else {
            const auto e = fb::nevpt2::sc_effective_hamiltonian(res(), blocks, pairs, 0);
            if (!e) {
              return std::unexpected(e.error());
            }
            std::ranges::transform(*e, std::back_inserter(out), &fb::nevpt2::ClassEnergies::total);
          }
          return {};
        });
    EXPECT_TRUE(got.has_value()) << got.error().message;
    return out;
  }
};

void load(QdCase &c, const std::string &name) {
  const auto doc =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2" / "qd_nevpt2.json");
  ASSERT_TRUE(doc.has_value()) << doc.error().message;
  fb::json entry;
  for (const fb::json &e : doc->at("cases")) {
    if (e.at("name").get<std::string>() == name) {
      entry = e;
    }
  }
  ASSERT_FALSE(entry.is_null()) << "no case " << name;
  const auto weights = entry.at("weights").get<std::vector<double>>();
  c.node = std::make_unique<fb::NodeResources>(
      fb::NodeResourcesConfig{.workspaces = fb::test::kWorkspaces});
  const fb::test::ScratchDir dir("df_qd_eigenvector_response");
  std::string xyz = std::format("{}\n{}\n", entry.at("atoms").size(), name);
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at(1).get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at(0).get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  ASSERT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string listed;
  for (const double w : weights) {
    listed += std::format("{}{:.17g}", listed.empty() ? "" : ", ", w);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\nconv_tol = {:g}\n"
      "conv_tol_grad = {:g}\n",
      entry.at("basis").get<std::string>(), entry.at("aux_basis").at("name").get<std::string>(),
      entry.at("ncas").get<std::int32_t>(), entry.at("nelecas").get<std::int32_t>(), weights.size(),
      listed, kConvTol, kConvTolGrad);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::Status ran = calc->run(*c.node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  c.calc.emplace(std::move(*calc));
  fb::DeviceResources &r = c.res();
  const auto &cas = c.calc->get<fb::CasscfOutput>();
  ASSERT_TRUE(cas.has_value());
  ASSERT_LT(cas->gradient_norm, kConvTolGrad);
  c.ncore = cas->ncore;
  c.ncas = cas->ncas;
  const auto n2 = static_cast<std::size_t>(c.ncas) * static_cast<std::size_t>(c.ncas);
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t root = 0; root < weights.size(); ++root) {
    for (std::size_t i = 0; i < n2; ++i) {
      gamma[i] += cas->weights[root] * cas->rdm1[(root * n2) + i];
    }
  }
  c.solution.emplace(fb::mcscf::Solution{
      .c = fb::SphericalMatrixAO(r, c.basis()),
      .roots = {.energies = cas->energies,
                .spin_squares = cas->spin_squares,
                .vectors = fb::Buffers::device<double>(cas->vectors.size(), r.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas->energy,
      .gradient_norm = cas->gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas->rdm2,
      .root_gammas = cas->rdm1,
      .iterations = cas->iterations,
      .weights = cas->weights});
  ASSERT_EQ(wwr::wwrMemcpyAsync(c.solution->roots.vectors.data(), cas->vectors.data(),
                                cas->vectors.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                r.stream()),
            wwr::wwrSuccess);
  ASSERT_EQ(wwr::wwrMemcpyAsync(c.solution->c.data(), cas->c.data(),
                                cas->c.num_elements() * sizeof(double),
                                wwr::wwrMemcpyDeviceToDevice, r.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(r.sync().has_value());
}

// The several-W call against the per-pair couplings, every pair P < Q and
// every component, within kCouplingTol; then term (ii)'s diagnostics.
void expect_couplings_match(const std::string &name, const bool pc) {
  QdCase c;
  load(c, name);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  const std::string label = std::format("{} {}", pc ? "PC" : "SC", name);
  const auto got = c.response(pc);
  ASSERT_TRUE(got.has_value()) << label << ": " << got.error().message;
  const std::size_t n = got->eigenvalues.size();
  const std::size_t nx = 3 * static_cast<std::size_t>(c.basis().natoms());
  ASSERT_EQ(got->couplings.size(), n * (n - 1) / 2 * nx) << label;
  ASSERT_EQ(got->term.size(), n * nx * nx) << label;
  ASSERT_EQ(got->dr.size(), n * n * nx) << label;
  EXPECT_LE(got->max_residual, 1e-10) << label;

  std::size_t pair = 0;
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = p + 1; q < n; ++q, ++pair) {
      const auto one = c.coupling(static_cast<std::int32_t>(p), static_cast<std::int32_t>(q), pc);
      ASSERT_TRUE(one.has_value())
          << label << " (" << p << ", " << q << "): " << one.error().message;
      ASSERT_EQ(one->total.size(), nx) << label;
      for (std::size_t s = 0; s < n; ++s) {
        EXPECT_NEAR(one->eigenvalues[s], got->eigenvalues[s], 1e-12) << label << " E_" << s;
      }
      const std::span<const double> list(got->couplings.data() + (pair * nx), nx);
      double miss = 0.0;
      for (std::size_t x = 0; x < nx; ++x) {
        EXPECT_NEAR(list[x], one->total[x], kCouplingTol)
            << label << " h_" << p << q << " component " << x;
        miss = std::max(miss, std::abs(list[x] - one->total[x]));
      }
      std::cout << std::format("[ hessian  ] qd term (ii) {} h_{}{}: max |h| {:.3e}, list vs "
                               "pair {:.2e} (bar {:g}), residuals {:.1e} / {:.1e}\n",
                               label, p, q, max_abs(list), miss, kCouplingTol, got->max_residual,
                               one->residual);
    }
  }

  // Diagnostics, never gated: term (ii)'s size per state, its sum rule, and
  // dR_P's gauge R_P^T dR_P = 0.
  for (std::size_t p = 0; p < n; ++p) {
    double gauge = 0.0;
    for (std::size_t x = 0; x < nx; ++x) {
      double d = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        d += got->eigenvectors[(p * n) + i] * got->dr[(p * n * nx) + i + (n * x)];
      }
      gauge = std::max(gauge, std::abs(d));
    }
    std::cout << std::format("[ hessian  ] qd term (ii) {} state {}: E {:.10f}, max |term| "
                             "{:.3e}, max |dR| {:.3e}, gauge {:.1e}\n",
                             label, p, got->eigenvalues[p], got->largest[p],
                             max_abs(std::span<const double>(got->dr).subspan(p * n * nx, n * nx)),
                             gauge);
  }
  std::cout << std::format("[ hessian  ] qd term (ii) {}: QD gap {:.4f}, sum rule {:.1e}\n", label,
                           got->qd_gap, got->sum_rule);
}

TEST(HessianDfQdEigenvectorResponse, ScListMatchesPairs) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  expect_couplings_match("h2co_skew_ccpvdz_sa3", false);
}

// PC's half of the acceptance, a family of its own so that it is its
// family's representative and the gate runs both halves (slow_test.h, "split
// the mode off"; #1161): 1.7 s on an idle RTX 3080 and 2.1 s on the gfx1200,
// as SC's 1.7 s and 2.0 to 3.2 s, past the line for a case that is not one.
TEST(HessianDfQdPcEigenvectorResponse, ListMatchesPairs) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  expect_couplings_match("h2co_skew_ccpvdz_sa3", true);
}

// h2co_ccpvdz_sa2 with root 1's CASSCF energy moved to make H_eff's
// diagonal equal (its off-diagonal vanishes by C2v symmetry): E_0 = E_1, a
// Numerical error naming both QD states, SC and PC.
TEST(HessianDfQdEigenvectorResponse, RefusesDegeneratePair) {
  FBF_REQUIRE_GPU();
  QdCase c;
  load(c, "h2co_ccpvdz_sa2");
  ASSERT_FALSE(HasFatalFailure());
  const std::vector<double> energies = c.solution->roots.energies;
  for (const bool pc : {false, true}) {
    const std::vector<double> totals = c.totals(pc);
    ASSERT_EQ(totals.size(), 4U);
    c.solution->roots.energies = energies;
    c.solution->roots.energies[1] = energies[0] + totals[0] - totals[3];
    const auto got = c.response(pc);
    ASSERT_FALSE(got.has_value()) << (pc ? "PC" : "SC");
    EXPECT_EQ(got.error().kind, fb::ErrorKind::Numerical) << got.error().message;
    EXPECT_NE(got.error().message.find("hessian::df_qd_eigenvector_response"), std::string::npos)
        << got.error().message;
    EXPECT_NE(got.error().message.find("QD states 0 and 1"), std::string::npos)
        << got.error().message;
    std::cout << std::format("[ hessian  ] qd term (ii) {} refuses: {}\n", pc ? "PC" : "SC",
                             got.error().message);
  }
}

} // namespace
