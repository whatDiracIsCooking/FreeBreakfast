// The step footprints of the RHF chain (#1140), of CASCI and FCI (#1049), of
// CASSCF (#1151), of the RHF gradient (#1050) and of the CASSCF gradient and
// couplings (#1160), and mcscf::solve's own and the CASSCF readers' routines,
// each held to the pool peak of one
// run of its step by test/support/pool_footprint.h's two bars, sound and
// tight. The dimensions are plan_shape's over a parsed water config, so the
// dims pass is exercised end to end, and each step is the one the config
// planned. Water in cc-pVDZ (s, p, d: U is not the identity), integral-direct
// and density-fitted over cc-pVDZ-JKFIT, the Scf closed-shell and, for the
// cation's doublet, open-shell.
//
// Each step runs on a fresh DeviceResources, its inputs on another, so its
// pool holds nothing else. It runs once to grow the workspaces, which on a
// fresh DeviceResources reach exactly the footprint's, and then again,
// measured, the workspaces already grown, as the pipeline grows them before
// the first step: the pool's peak is then the step's persistent and scratch.
// CASCI over an explicit space, over AVAS's pick (the footprint at the space
// it picked) and over a carried orbitals file; FCI under a pool cap that
// leaves its sigma the least scratch it runs with, and unbounded; and
// CASSCF's loop and step, integral-direct and density-fitted; the RHF
// gradient, integral-direct and density-fitted; and, over a density-fitted
// SA-2 CASSCF, the routines its gradient and couplings call and the
// CasscfGradient and Nac steps, at equal and unequal weights.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.core.nontransferrable;
import free_breakfast.io.base;
import free_breakfast.io.npy;
import free_breakfast.ci;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.shape;
import free_breakfast.mcscf;
import free_breakfast.integrals.eri.density_fitting.jk; // df_exchange_footprint
import free_breakfast.response;
import free_breakfast.gradient;   // one_electron_terms
import free_breakfast.integrals;  // casscf_integrals, overlap_half_gradient
import free_breakfast.recipes.lu; // getrf_lwork
import free_breakfast.nevpt2;     // the NEVPT2 routines the NEVPT2 steps call

#include "pool_footprint.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
namespace fbt = free_breakfast::test;

constexpr std::string_view kWaterXyz = R"(3
water
O  0.0  0.0     0.1173
H  0.0  0.7572 -0.4692
H  0.0 -0.7572 -0.4692
)";

// Benzene, C-C 1.39 A and C-H 1.09 A, in the xy plane.
constexpr std::string_view kBenzeneXyz = R"(12
benzene
C   1.3900  0.0000 0.0
C   0.6950  1.2038 0.0
C  -0.6950  1.2038 0.0
C  -1.3900  0.0000 0.0
C  -0.6950 -1.2038 0.0
C   0.6950 -1.2038 0.0
H   2.4800  0.0000 0.0
H   1.2400  2.1477 0.0
H  -1.2400  2.1477 0.0
H  -2.4800  0.0000 0.0
H  -1.2400 -2.1477 0.0
H   1.2400 -2.1477 0.0
)";

// A molecule, its basis, and the ERIs over it.
struct System {
  std::string_view xyz = kWaterXyz;
  std::string_view basis = "cc-pvdz";
  fb::EriMode mode = fb::EriMode::IntegralDirect;
  // The cation's doublet, an open shell, in place of the neutral singlet.
  bool cation = false;
  // TOML tables after [scf]: a CI step's.
  std::string_view tables{};
};

// The step the plan configured, or its defaults where the planner filled none.
template <class S>
S planned(const fb::Calculation &calc) {
  return calc.step<S>().value_or(S{});
}

// Water's plan, its shape, and every product the chain makes before the Scf,
// on their own DeviceResources.
struct Plan {
  fbt::ScratchDir dir{"pipeline_step_footprint"};
  fb::DeviceResources inputs{0};
  std::optional<fb::Calculation> calc;
  std::optional<fb::PlanShape> shape;
  std::optional<fb::MoleculeOutput> molecule;
  std::optional<fb::BasisSetOutput> basis;
  std::optional<fb::SphericalTransformOutput> transform;
  std::optional<fb::PrimitivePairClassesOutput> pairs;
  std::optional<fb::EriOutput> eri;
  std::optional<fb::OverlapOutput> overlap;
  std::optional<fb::OrthogonalizerOutput> x;
  std::optional<fb::CoreHamiltonianOutput> h;
  std::optional<fb::GuessOutput> guess;
  std::optional<fb::ScfOutput> scf;

  explicit Plan(const System &system) {
    EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", system.xyz).has_value());
    std::filesystem::create_directories(dir / "out");
    const std::string toml =
        std::format("[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\nmultiplicity = {}\n"
                    "[basis_set]\nbasis = \"{}\"\n{}[scf]\n{}",
                    system.cation ? 1 : 0, system.cation ? 2 : 1, system.basis,
                    system.mode == fb::EriMode::DensityFitting
                        ? "[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n"
                        : "",
                    system.tables);
    auto parsed = fb::parse_config(toml, dir.path(), dir / "out");
    EXPECT_TRUE(parsed.has_value()) << parsed.error().message;
    calc.emplace(std::move(parsed).value());
    auto shaped = fb::plan_shape(*calc, inputs);
    EXPECT_TRUE(shaped.has_value()) << shaped.error().message;
    shape.emplace(std::move(shaped).value());
  }

  [[nodiscard]] const fb::FootprintDimensions &dims() const { return shape->dims; }

  // The chain up to (and not including) the Scf, on inputs, in the planner's
  // order.
  void run_chain() {
    molecule.emplace(planned<fb::MoleculeStep>(*calc).run(fb::View<>{}, inputs).value());
    basis.emplace(
        planned<fb::BasisSetStep>(*calc).run(fb::View<fb::MoleculeOutput>{*molecule}).value());
    transform.emplace(planned<fb::SphericalTransformStep>(*calc)
                          .run(fb::View<fb::BasisSetOutput>{*basis}, inputs)
                          .value());
    pairs.emplace(planned<fb::PrimitivePairClassesStep>(*calc)
                      .run(fb::View<fb::BasisSetOutput>{*basis}, inputs)
                      .value());
    auto computed = planned<fb::EriStep>(*calc).run(eri_view(), inputs);
    ASSERT_TRUE(computed) << computed.error().message;
    eri.emplace(std::move(*computed));
    overlap.emplace(planned<fb::OverlapStep>(*calc).run(overlap_view(), inputs).value());
    x.emplace(planned<fb::OrthogonalizerStep>(*calc)
                  .run(fb::View<fb::BasisSetOutput, fb::OverlapOutput>{*basis, *overlap}, inputs)
                  .value());
    h.emplace(planned<fb::CoreHamiltonianStep>(*calc).run(core_view(), inputs).value());
    guess.emplace(run_guess());
  }

  [[nodiscard]] fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>
  eri_view() const {
    return fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
        *molecule, *basis, *pairs};
  }
  [[nodiscard]] fb::View<fb::BasisSetOutput, fb::SphericalTransformOutput,
                         fb::PrimitivePairClassesOutput>
  overlap_view() const {
    return fb::View<fb::BasisSetOutput, fb::SphericalTransformOutput,
                    fb::PrimitivePairClassesOutput>{*basis, *transform, *pairs};
  }
  [[nodiscard]] fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                         fb::PrimitivePairClassesOutput>
  core_view() const {
    return fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                    fb::PrimitivePairClassesOutput>{*molecule, *basis, *transform, *pairs};
  }
  [[nodiscard]] fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                         fb::EriOutput, fb::OverlapOutput, fb::OrthogonalizerOutput,
                         fb::CoreHamiltonianOutput, fb::GuessOutput>
  scf_view() const {
    return fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                    fb::EriOutput, fb::OverlapOutput, fb::OrthogonalizerOutput,
                    fb::CoreHamiltonianOutput, fb::GuessOutput>{
        *molecule, *basis, *transform, *eri, *overlap, *x, *h, *guess};
  }

  // The chain through the Scf, on inputs.
  void run_through_scf() {
    run_chain();
    auto converged = planned<fb::ScfStep>(*calc).run(scf_view(), inputs);
    ASSERT_TRUE(converged) << converged.error().message;
    scf.emplace(std::move(*converged));
  }

  // What a CASCI or FCI step reads.
  [[nodiscard]] fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                         fb::PrimitivePairClassesOutput, fb::CoreHamiltonianOutput, fb::ScfOutput>
  ci_view() const {
    return fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                    fb::PrimitivePairClassesOutput, fb::CoreHamiltonianOutput, fb::ScfOutput>{
        *molecule, *basis, *transform, *pairs, *h, *scf};
  }

  // The Guess step, which runs in a subflow.
  [[nodiscard]] fb::GuessOutput run_guess() const {
    fb::tf::Executor executor(2);
    fb::tf::Taskflow flow;
    fb::Result<fb::GuessOutput> out = fb::err_config("the guess task did not run");
    flow.emplace([&](fb::tf::Subflow &subflow) {
      out = planned<fb::GuessStep>(*calc).run(
          fb::View<fb::MoleculeOutput, fb::BasisSetOutput>{*molecule, *basis}, subflow);
    });
    executor.run(flow).wait();
    EXPECT_TRUE(out) << out.error().message;
    return std::move(*out);
  }
};

// @p run, a step's, held to @p fp on a fresh DeviceResources, with @p live
// allocations live at its peak: run once to grow the workspaces to exactly
// the footprint's, then measured.
// With @p workspace_bound the device workspace is held to at most the
// footprint's, and the gap logged: a Davidson's or the augmented Hessian's
// syevd is counted at its subspace's largest order, which a solve that
// converges first never reaches.
template <typename Run>
void expect_step_footprint(const fb::MemoryFootprint &fp, const std::size_t live, Run &&run,
                           std::optional<fbt::FootprintSlack> slack = {},
                           bool workspace_bound = false,
                           std::source_location location = std::source_location::current()) {
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const std::size_t device_before = res.device_workspace(0).num_elements();
  const std::size_t pinned_before = res.pinned_bytes();
  run(res);
  const std::size_t grown = res.device_workspace(0).num_elements();
  if (workspace_bound) {
    EXPECT_LE(grown, std::max(device_before, fp.device.workspace))
        << "the device workspace, from line " << location.line();
    std::cout << std::format("[ footprint ] workspace {} of the bound {}\n", grown,
                             fp.device.workspace);
  } else {
    EXPECT_EQ(grown, std::max(device_before, fp.device.workspace))
        << "the device workspace, from line " << location.line();
  }
  EXPECT_EQ(res.pinned_bytes(), std::max(pinned_before, fp.pinned.peak()))
      << "the pinned memory, from line " << location.line();
  const fbt::PoolPeak peak = fbt::measure_pool_peak(*res.device(), [&] { run(res); });
  std::cout << std::format(
      "[ footprint ] line {}: measured {} (G {}), footprint {} + workspace {}\n", location.line(),
      peak.used, peak.granularity, fp.device.persistent + fp.device.scratch, fp.device.workspace);
  fbt::expect_footprint(peak, {.persistent = fp.device.persistent, .scratch = fp.device.scratch},
                        live, slack, location);
}

// A step's Result, which must hold its product.
template <class R>
void expect_ran(const R &ran) {
  EXPECT_TRUE(ran) << ran.error().message;
}

TEST(StepFootprint, SetupSteps) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  Plan plan(System{});
  plan.run_chain();
  const fb::FootprintDimensions &dims = plan.dims();

  // Z and the positions.
  const auto molecule = planned<fb::MoleculeStep>(*plan.calc);
  expect_step_footprint(molecule.footprint(dims), 2, [&](fb::DeviceResources &res) {
    expect_ran(molecule.run(fb::View<>{}, res));
  });
  // Host-shaped.
  const auto basis = planned<fb::BasisSetStep>(*plan.calc);
  EXPECT_EQ(basis.footprint(dims), fb::MemoryFootprint{});
  expect_step_footprint(basis.footprint(dims), 0, [&](fb::DeviceResources &) {
    expect_ran(basis.run(fb::View<fb::MoleculeOutput>{*plan.molecule}));
  });
  // U.
  const auto transform = planned<fb::SphericalTransformStep>(*plan.calc);
  ASSERT_GT(transform.footprint(dims).device.persistent, 0U);
  expect_step_footprint(transform.footprint(dims), 1, [&](fb::DeviceResources &res) {
    expect_ran(transform.run(fb::View<fb::BasisSetOutput>{*plan.basis}, res));
  });
  // The six classes' 20 fields each and the 11 tables; the classes sort in the
  // device workspace.
  const auto pairs = planned<fb::PrimitivePairClassesStep>(*plan.calc);
  expect_step_footprint(pairs.footprint(dims), (6 * 20) + 11, [&](fb::DeviceResources &res) {
    expect_ran(pairs.run(fb::View<fb::BasisSetOutput>{*plan.basis}, res));
  });
  // Host-shaped.
  const auto guess = planned<fb::GuessStep>(*plan.calc);
  EXPECT_EQ(guess.footprint(dims), fb::MemoryFootprint{});
  expect_step_footprint(guess.footprint(dims), 0,
                        [&](fb::DeviceResources &) { (void)plan.run_guess(); });
}

TEST(StepFootprint, OneElectronSteps) {
  FBF_REQUIRE_GPU();
  Plan plan(System{});
  plan.run_chain();
  const fb::FootprintDimensions &dims = plan.dims();

  // The pair's Cartesian and spherical matrices and its conversion scratch.
  const auto overlap = planned<fb::OverlapStep>(*plan.calc);
  expect_step_footprint(overlap.footprint(dims), 3, [&](fb::DeviceResources &res) {
    expect_ran(overlap.run(plan.overlap_view(), res));
  });
  const auto h = planned<fb::CoreHamiltonianStep>(*plan.calc);
  expect_step_footprint(h.footprint(dims), 3, [&](fb::DeviceResources &res) {
    expect_ran(h.run(plan.core_view(), res));
  });
  // X, U and the eigenvalues.
  const auto x = planned<fb::OrthogonalizerStep>(*plan.calc);
  expect_step_footprint(x.footprint(dims), 3, [&](fb::DeviceResources &res) {
    expect_ran(
        x.run(fb::View<fb::BasisSetOutput, fb::OverlapOutput>{*plan.basis, *plan.overlap}, res));
  });
}

// J's classes are sized to the records coulomb_pair_classes's prune keeps at
// the step's tau, which only the device can say; the footprint counts every
// mmd record, its bound at tau = 0. The plan's own Eri, built from the same
// records at the same tau, says how many were kept.
TEST(StepFootprint, EriIntegralDirect) {
  FBF_REQUIRE_GPU();
  Plan plan(System{});
  plan.run_chain();
  const fb::FootprintDimensions &dims = plan.dims();
  std::vector<std::vector<std::size_t>> kept(dims.mmd_records.size(),
                                             std::vector<std::size_t>(dims.mmd_records.size()));
  for (const fb::CoulombPairClass &cls : std::get<fb::IntegralDirect>(plan.eri->eri).j_classes) {
    kept[static_cast<std::size_t>(cls.la)][static_cast<std::size_t>(cls.lb)] =
        static_cast<std::size_t>(cls.n_rec);
  }
  const std::size_t pruned =
      fb::coulomb_pair_classes_footprint(dims.mmd_records).device.persistent -
      fb::coulomb_pair_classes_footprint(kept).device.persistent;
  const auto eri = planned<fb::EriStep>(*plan.calc);
  // At the peak, K's: J's 6 classes' 31 fields each and at most the 9 ordered
  // classes' 30 fields each; the gathers sort in the device workspace.
  expect_step_footprint(
      eri.footprint(dims), (6 * 31) + (9 * 30),
      [&](fb::DeviceResources &res) { expect_ran(eri.run(plan.eri_view(), res)); },
      fbt::FootprintSlack{.bytes = pruned,
                          .reason = "J's classes hold the records the tau = 1e-13 prune keeps; "
                                    "the footprint, every mmd record"});
}

TEST(StepFootprint, EriDensityFitted) {
  FBF_REQUIRE_GPU();
  Plan plan(System{.mode = fb::EriMode::DensityFitting});
  plan.run_chain();
  const auto eri = planned<fb::EriStep>(*plan.calc);
  ASSERT_EQ(eri.mode, fb::EriMode::DensityFitting);
  // At the peak: L, B and the three-center tensor's four buffers.
  expect_step_footprint(eri.footprint(plan.dims()), 6, [&](fb::DeviceResources &res) {
    expect_ran(eri.run(plan.eri_view(), res));
  });
}

struct ScfCase {
  std::string_view name;
  System system;
  // Live at the peak.
  std::size_t live;
};

class ScfStepFootprint : public ::testing::TestWithParam<ScfCase> {};

// Under slow_test.h: the integral-direct ROHF, the family's one case past
// 1.5 s.
TEST_P(ScfStepFootprint, MatchesThePoolPeak) {
  FBF_REQUIRE_GPU();
  if (GetParam().name == "OpenDirect") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  Plan plan(GetParam().system);
  plan.run_chain();
  const auto scf = planned<fb::ScfStep>(*plan.calc);
  ASSERT_EQ(plan.dims().n_alpha != plan.dims().n_beta, GetParam().system.cation);
  const fb::MemoryFootprint fp = scf.footprint(plan.dims());
  if (GetParam().name == "BenzeneMinimalDf") {
    // K's workspace, not syevd's, is the step's, at n_alpha = 21 columns: the
    // first build's guess factor, of a rank past 21 (a factor of all its
    // columns at once grows the workspace past this), is taken 21 at a time.
    ASSERT_EQ(fp.device.workspace,
              fb::df_exchange_footprint(36, static_cast<std::int32_t>(plan.dims().naux), 21)
                  .device.workspace);
  }
  expect_step_footprint(fp, GetParam().live, [&](fb::DeviceResources &res) {
    expect_ran(scf.run(plan.scf_view(), res));
  });
}

constexpr fb::EriMode kDf = fb::EriMode::DensityFitting;

// ScfState's 8 buffers and the DIIS history's 5 throughout; the integral-direct
// builder's d_built, dd and g (3 each) and k, and an open shell's da_built and
// db_built; the density-fitted one's J and K, and the guess's V and
// eigenvalues; an open shell's OpenShellState, 5. Benzene in STO-3G over
// cc-pVDZ-JKFIT is where the density-fitted K, not syevd, sets the workspace.
INSTANTIATE_TEST_SUITE_P(
    Shells, ScfStepFootprint,
    ::testing::Values(ScfCase{"ClosedDirect", System{}, 8 + 5 + 8},
                      ScfCase{"OpenDirect", System{.cation = true}, 8 + 5 + 10 + 5},
                      ScfCase{"ClosedDf", System{.mode = kDf}, 8 + 5 + 4},
                      ScfCase{"OpenDf", System{.mode = kDf, .cation = true}, 8 + 5 + 4 + 5},
                      ScfCase{"BenzeneMinimalDf",
                              System{.xyz = kBenzeneXyz, .basis = "sto-3g", .mode = kDf},
                              8 + 5 + 4}),
    [](const ::testing::TestParamInfo<ScfCase> &info) { return std::string(info.param.name); });

// @p dims with mmd's record counts replaced by those J's classes keep at
// coulomb_pair_classes' default tau over @p plan's classes: what a step's
// J classes hold, where its footprint counts every record.
fb::FootprintDimensions with_kept_records(const Plan &plan, fb::FootprintDimensions dims) {
  fb::DeviceResources res{0};
  const auto j_classes = fb::coulomb_pair_classes(res, plan.pairs->classes);
  EXPECT_TRUE(j_classes);
  for (std::vector<std::size_t> &row : dims.mmd_records) {
    std::ranges::fill(row, 0);
  }
  for (const fb::CoulombPairClass &cls : j_classes.value()) {
    dims.mmd_records[static_cast<std::size_t>(cls.la)][static_cast<std::size_t>(cls.lb)] =
        static_cast<std::size_t>(cls.n_rec);
  }
  return dims;
}

// The tight bar's slack for the J prune: the footprint's peak at every mmd
// record less its peak at the records J keeps, exactly, and 0 when the J
// classes are not live at the step's peak.
fbt::FootprintSlack prune_slack(const fb::MemoryFootprint &every, const fb::MemoryFootprint &kept) {
  return fbt::FootprintSlack{
      .bytes = (every.device.persistent + every.device.scratch) -
               (kept.device.persistent + kept.device.scratch),
      .reason = "J's classes hold the records the tau = 1e-13 prune keeps; the footprint, every "
                "mmd record"};
}

// At the CI's peak, the Davidson's: CasciOutput::c; the one string set, the
// sigma's 3 tables, the diagonal; ci::solve's 8 (guesses, shifted diagonal,
// values, vectors, picks, the empty P-space pair, S^2's table) and the
// Davidson's 11; the sigma's scratch.
constexpr std::size_t kCasciLive = 1 + 1 + 3 + 1 + 8 + 11 + 1;

// Water in cc-pVDZ, CAS(8, 8) by index, two roots, with the Molden file's
// natural-orbital eigenproblem: the CI's Davidson, over 2,485 packed
// determinants, is the peak, past the Hamiltonian's. The workspace is the
// Davidson's syevd at its 32-column subspace, which this solve fills.
TEST(StepFootprint, CasciExplicit) {
  FBF_REQUIRE_GPU();
  Plan plan(System{.tables = "[casci]\nncas = 8\nnelecas = 8\nnroots = 2\n"});
  plan.run_through_scf();
  const fb::FootprintDimensions &dims = plan.dims();
  ASSERT_EQ(dims.ncas, 8U);
  ASSERT_EQ(dims.nelecas, 8U);
  // Set on the step, as a caller would: the config has no molden key.
  auto casci = planned<fb::CasciStep>(*plan.calc);
  casci.molden = plan.dir / "casci.molden";
  const fb::MemoryFootprint fp = casci.footprint(dims);
  expect_step_footprint(
      fp, kCasciLive, [&](fb::DeviceResources &res) { expect_ran(casci.run(plan.ci_view(), res)); },
      prune_slack(fp, casci.footprint(with_kept_records(plan, dims))));
}

// AVAS over O 2p: the footprint at the space AVAS picked, which plan_shape's
// bound (6 orbitals, half filling) need not be.
TEST(StepFootprint, CasciAvas) {
  FBF_REQUIRE_GPU();
  Plan plan(System{.tables = "[casci]\navas = [\"O 2p\"]\n"});
  plan.run_through_scf();
  const auto casci = planned<fb::CasciStep>(*plan.calc);
  ASSERT_NE(plan.dims().minao, nullptr);
  const auto picked = casci.run(plan.ci_view(), plan.inputs);
  ASSERT_TRUE(picked) << picked.error().message;
  fb::FootprintDimensions dims = plan.dims();
  std::cout << std::format("[ footprint ] AVAS picked CAS({}, {}), the bound CAS({}, {})\n",
                           picked->nelecas, picked->ncas, dims.nelecas, dims.ncas);
  dims.ncas = static_cast<std::size_t>(picked->ncas);
  dims.nelecas = static_cast<std::size_t>(picked->nelecas);
  const fb::MemoryFootprint fp = casci.footprint(dims);
  expect_step_footprint(
      fp, kCasciLive, [&](fb::DeviceResources &res) { expect_ran(casci.run(plan.ci_view(), res)); },
      prune_slack(fp, casci.footprint(with_kept_records(plan, dims))));
}

// The Scf's own orbitals carried in as an orbitals file: S, the work, M, X,
// and the overlap and the two Löwdin orthogonalizations.
TEST(StepFootprint, CasciCarriedOrbitals) {
  FBF_REQUIRE_GPU();
  Plan plan(System{.tables = "[casci]\nncas = 4\nnelecas = 4\norbitals = \"c.npy\"\n"});
  plan.run_through_scf();
  const std::size_t nao = plan.dims().nao;
  fb::NpyArray c{.shape = {nao, nao}, .data = std::vector<double>(nao * nao)};
  EXPECT_EQ(wwr::wwrMemcpyAsync(c.data.data(), plan.scf->c.data(), sizeof(double) * nao * nao,
                                wwr::wwrMemcpyDeviceToHost, plan.inputs.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(plan.inputs.sync());
  ASSERT_TRUE(fb::write_npy(plan.dir / "c.npy", c));
  const auto casci = planned<fb::CasciStep>(*plan.calc);
  const fb::FootprintDimensions &dims = plan.dims();
  const fb::MemoryFootprint fp = casci.footprint(dims);
  expect_step_footprint(
      fp, kCasciLive, [&](fb::DeviceResources &res) { expect_ran(casci.run(plan.ci_view(), res)); },
      prune_slack(fp, casci.footprint(with_kept_records(plan, dims))));
}

// N2 in STO-3G, 14 electrons in 10 orbitals, two roots: 14,400 determinants,
// 7,260 in the symmetric layout. Its plan, footprint and the sigma's scratch,
// the least (one string's G, the footprint's) and G for the whole space.
struct FciPlan {
  static constexpr std::string_view kN2 = R"(2
nitrogen
N 0.0 0.0  0.55
N 0.0 0.0 -0.55
)";
  Plan plan{System{.xyz = kN2, .basis = "sto-3g", .tables = "[fci]\nnroots = 2\n"}};
  fb::FciStep fci;
  fb::MemoryFootprint fp;
  std::size_t least = fb::ci::sigma_apply_bytes(10, 120, 120, fb::ci::Layout::symmetric, 0);
  std::size_t whole = fb::ci::sigma_apply_bytes(10, 120, 120, fb::ci::Layout::symmetric);

  FciPlan() {
    plan.run_through_scf();
    EXPECT_EQ(plan.dims().ncas, 10U);
    EXPECT_EQ(plan.dims().nelecas, 14U);
    fci = planned<fb::FciStep>(*plan.calc);
    fp = fci.footprint(plan.dims());
    EXPECT_LT(least, whole);
  }
};

// At the CI's peak: the one string set, the sigma's 3 tables, the diagonal,
// ci::solve's 8, the Davidson's 11 and the sigma's scratch.
constexpr std::size_t kFciLive = 1 + 3 + 1 + 8 + 11 + 1;

// Under a pool cap of the footprint's peak, the 64 MiB of fragmentation
// headroom the step leaves (the pool reserves in chunks, and the cap bounds
// what it reserves), and the pool's rounding of every allocation live at its
// peak, the sigma takes the least scratch it runs with and the run is held to
// the footprint itself. A block of strings is 120 rows of 56 doubles, 53,760
// bytes, past the rounding, so the cap leaves no room for a second string's.
TEST(StepFootprint, FciAtTheLeast) {
  FBF_REQUIRE_GPU();
  const FciPlan f;
  std::size_t granularity = 0;
  {
    fb::DeviceResources probe{0};
    granularity = fbt::pool_granularity(*probe.device());
  }
  const std::size_t rounding = granularity * (kFciLive + 16);
  ASSERT_LT(rounding, std::size_t{120} * 56 * sizeof(double));
  const std::uint64_t cap = f.fp.device.peak() + fb::kPoolHeadroomBytes + rounding;
  fb::DeviceResources res{0, fb::DeviceResources::kDefaultPoolReleaseThreshold, 0, cap};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  expect_ran(f.fci.run(f.plan.ci_view(), res));
  EXPECT_EQ(res.device_workspace(0).num_elements(), f.fp.device.workspace);
  const fbt::PoolPeak peak =
      fbt::measure_pool_peak(*res.device(), [&] { expect_ran(f.fci.run(f.plan.ci_view(), res)); });
  std::cout << std::format("[ footprint ] capped at {}: measured {} (G {}), footprint {} + "
                           "workspace {}\n",
                           cap, peak.used, peak.granularity,
                           f.fp.device.persistent + f.fp.device.scratch, f.fp.device.workspace);
  fbt::expect_footprint(
      peak, {.persistent = f.fp.device.persistent, .scratch = f.fp.device.scratch}, kFciLive);
}

// Unbounded, the sigma takes G for the whole space, past the footprint's
// least by exactly the difference.
TEST(StepFootprint, FciUnbounded) {
  FBF_REQUIRE_GPU();
  const FciPlan f;
  fb::MemoryFootprint unbounded = f.fp;
  unbounded.device.scratch += f.whole - f.least;
  expect_step_footprint(unbounded, kFciLive, [&](fb::DeviceResources &res) {
    expect_ran(f.fci.run(f.plan.ci_view(), res));
  });
}

// What the Casscf step reads.
fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput, fb::EriOutput,
         fb::CoreHamiltonianOutput, fb::ScfOutput>
casscf_view(const Plan &plan) {
  return fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                  fb::EriOutput, fb::CoreHamiltonianOutput, fb::ScfOutput>{
      *plan.molecule, *plan.basis, *plan.transform, *plan.eri, *plan.h, *plan.scf};
}

// Water in cc-pVDZ, the SA-2 CAS(4, 4) over 2 core orbitals, converged loosely:
// what the loop holds does not depend on how far it goes, once a second macro
// iteration runs on a sigma a product has grown (the steady state its
// footprint counts), and the tighter defaults only add iterations of it.
constexpr std::string_view kCasscfTable =
    "[casscf]\nncas = 4\nnelecas = 4\nnroots = 2\nconv_tol = 1e-6\nconv_tol_grad = 1e-3\n";

struct SolveCase {
  std::string_view name;
  fb::mcscf::DensityFitting fitting;
  // Live at the peak.
  std::size_t live;
  // Past slow_test.h's 1.5 s: the integral-direct J and K of each macro
  // iteration's integrals and Fock.
  bool slow;
};

class McscfSolveFootprint : public ::testing::TestWithParam<SolveCase> {};

// mcscf::solve from the selected space, the step's call alone: integral-
// direct; the Hessian fitted over B, the rest integral-direct over classes
// built here (the Eri step makes one or the other); and every build fitted.
// The workspace is held to its bound: the augmented Hessian's syevd is counted
// at its subspace's largest order, and the CI's at 16 nroots columns.
TEST_P(McscfSolveFootprint, MatchesThePoolPeak) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("Full");
  if (GetParam().slow) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::mcscf::DensityFitting fitting = GetParam().fitting;
  const bool fitted = fitting != fb::mcscf::DensityFitting::Direct;
  Plan plan(System{.mode = fitted ? fb::EriMode::DensityFitting : fb::EriMode::IntegralDirect,
                   .tables = kCasscfTable});
  plan.run_through_scf();
  const fb::FootprintDimensions &dims = plan.dims();
  ASSERT_GT(dims.skew_eig_lwork, 0U);
  const auto casscf = planned<fb::CasscfStep>(*plan.calc);
  const auto space =
      casscf.select(plan.inputs, plan.molecule->molecule, plan.basis->basis, *plan.scf);
  ASSERT_TRUE(space) << space.error().message;
  std::vector<fb::CoulombPairClass> own_j;
  std::vector<fb::OrderedPairClass> own_k;
  std::span<fb::CoulombPairClass> j_classes;
  std::span<fb::OrderedPairClass> k_classes;
  const fb::Buffers::device<double> *b = nullptr;
  std::int32_t naux = 0;
  if (const auto *direct = std::get_if<fb::IntegralDirect>(&plan.eri->eri)) {
    j_classes = direct->j_classes;
    k_classes = direct->k_classes;
  } else {
    const auto &fit = std::get<fb::DensityFitted>(plan.eri->eri);
    b = &fit.b;
    naux = fit.naux;
    if (fitting == fb::mcscf::DensityFitting::ApproxHessian) {
      own_j = fb::coulomb_pair_classes(plan.inputs, plan.pairs->classes).value();
      own_k = fb::ordered_pair_classes(plan.inputs, plan.basis->basis, plan.pairs->classes).value();
      j_classes = own_j;
      k_classes = own_k;
    }
  }
  fb::mcscf::Options options;
  options.density_fitting = fitting;
  options.conv_tol = casscf.conv_tol;
  options.conv_tol_grad = casscf.conv_tol_grad;
  const fb::MemoryFootprint fp =
      fb::mcscf::solve_footprint(plan.basis->basis, space->ncore, space->ncas, space->nelecas,
                                 plan.molecule->molecule.multiplicity(), casscf.nroots, options,
                                 naux, dims.syevd_lwork, dims.skew_eig_lwork);
  expect_step_footprint(
      fp, GetParam().live,
      [&](fb::DeviceResources &res) {
        const auto solved =
            fb::mcscf::solve(res, plan.basis->basis, plan.molecule->molecule, plan.h->h,
                             plan.transform->u, j_classes, k_classes, space->c, space->ncore,
                             space->ncas, space->nelecas, casscf.nroots, {}, options, b, naux);
        ASSERT_TRUE(solved) << solved.error().message;
      },
      {}, true);
}

// At the peak, the loop's: the one string set, the sigma's 3 tables and its
// scratch, C twice, the diagonal, gamma, Gamma, the step, kappa, its
// eigenvectors and eigenvalues, the rotation and the augmented Hessian's 10,
// and the last roots: 25; the integrals' ppaa, papa, h_eff and (tu|vw), 4;
// the Fock's 6; the Hessian's 9, and its J and K's 3 (integral-direct) or 2;
// the relaxed Hessian's 9 and H'(x)'s sigma's 4; and the fitted product's
// scratch, 1.
INSTANTIATE_TEST_SUITE_P(
    Fittings, McscfSolveFootprint,
    ::testing::Values(
        SolveCase{"Direct", fb::mcscf::DensityFitting::Direct, 25 + 4 + 6 + 12 + 13, true},
        SolveCase{"ApproxHessian", fb::mcscf::DensityFitting::ApproxHessian,
                  25 + 4 + 6 + 11 + 13 + 1, true},
        SolveCase{"Full", fb::mcscf::DensityFitting::Full, 25 + 4 + 6 + 11 + 13 + 1, false}),
    [](const ::testing::TestParamInfo<SolveCase> &info) { return std::string(info.param.name); });

// The Casscf step: its footprint at the explicit space, mcscf::solve_footprint's
// loop between the selection and the tail, the run on a fresh DeviceResources.
// The workspace is held to its bound, as McscfSolveFootprint's. Live at the
// peak: the selection's C beside the loop's, as McscfSolveFootprint counts it.
void expect_casscf_footprint(const fb::EriMode mode) {
  Plan plan(System{.mode = mode, .tables = kCasscfTable});
  plan.run_through_scf();
  const fb::FootprintDimensions &dims = plan.dims();
  ASSERT_EQ(dims.ncas, 4U);
  ASSERT_EQ(dims.nelecas, 4U);
  auto casscf = planned<fb::CasscfStep>(*plan.calc);
  casscf.molden = plan.dir / "casscf.molden";
  const std::size_t hessian = mode == fb::EriMode::DensityFitting ? 11 + 1 : 12;
  expect_step_footprint(
      casscf.footprint(dims), 1 + 25 + 4 + 6 + hessian + 13,
      [&](fb::DeviceResources &res) { expect_ran(casscf.run(casscf_view(plan), res)); }, {}, true);
}

// Under slow_test.h: the integral-direct J and K of every macro iteration.
TEST(StepFootprint, CasscfDirect) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_casscf_footprint(fb::EriMode::IntegralDirect);
}

// Under slow_test.h too, at about 1.6 s: McscfSolveFootprint's Full, its
// family's representative, holds the loop in the fast set.
TEST(StepFootprint, CasscfDensityFitted) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_casscf_footprint(fb::EriMode::DensityFitting);
}

// What the Gradient step reads.
fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput, fb::EriOutput,
         fb::ScfOutput>
gradient_view(const Plan &plan) {
  return fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput,
                  fb::EriOutput, fb::ScfOutput>{*plan.molecule, *plan.basis, *plan.pairs, *plan.eri,
                                                *plan.scf};
}

// The Gradient step over water's RHF: at the integral-direct one's peak, D
// and the J gradient's U, pair, J classes and their prune's scratch; at the
// fitted one's, D, -W, gmmwich's diagonal and work and a one-electron
// derivative call's seven.
void expect_gradient_footprint(const fb::EriMode mode, const std::size_t live) {
  Plan plan(System{.mode = mode, .tables = "[gradient]\n"});
  plan.run_through_scf();
  const fb::FootprintDimensions &dims = plan.dims();
  ASSERT_EQ(dims.aux != nullptr, mode == fb::EriMode::DensityFitting);
  const auto gradient = planned<fb::GradientStep>(*plan.calc);
  const fb::MemoryFootprint fp = gradient.footprint(dims);
  EXPECT_EQ(fp.device.persistent, 0U);
  expect_step_footprint(fp, live, [&](fb::DeviceResources &res) {
    expect_ran(gradient.run(gradient_view(plan), res));
  });
}

TEST(StepFootprint, GradientDirect) {
  FBF_REQUIRE_GPU();
  expect_gradient_footprint(fb::EriMode::IntegralDirect, 1 + 4 + (6 * 31) + 8);
}

TEST(StepFootprint, GradientDensityFitted) {
  FBF_REQUIRE_GPU();
  expect_gradient_footprint(fb::EriMode::DensityFitting, 4 + 7);
}

// The CASSCF gradient's and couplings' routines (#1160), over water's SA-2
// CAS(4, 4), every build fitted (kCasscfTable), converged once for the suite;
// and the CasscfGradient and Nac steps that read it. Each routine runs on a
// fresh DeviceResources, its inputs (the plan's chain, the CASSCF's
// mcscf::Solution and, for zvector, the CI-relaxed Hessian) on the plan's.
// The CASSCF's product as the mcscf::Solution the readers build from it, on
// @p res: gamma the weighted average of the roots' 1-RDMs, the weights
// carried.
fb::mcscf::Solution solution_of(fb::DeviceResources &res, const fb::BasisSet &basis,
                                const fb::CasscfOutput &cas) {
  std::vector<double> weights = cas.weights;
  const auto n2 = static_cast<std::size_t>(cas.ncas) * static_cast<std::size_t>(cas.ncas);
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < weights.size(); ++r) {
    for (std::size_t i = 0; i < n2; ++i) {
      gamma[i] += weights[r] * cas.rdm1[(r * n2) + i];
    }
  }
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, basis),
      .roots = {.energies = cas.energies,
                .spin_squares = cas.spin_squares,
                .vectors = fb::Buffers::device<double>(cas.vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas.energy,
      .gradient_norm = cas.gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas.rdm2,
      .root_gammas = cas.rdm1,
      .iterations = cas.iterations,
      .weights = std::move(weights),
  };
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                cas.vectors.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                cas.c.num_elements() * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync());
  return solution;
}

// @p count doubles of @p from on the host.
std::vector<double> to_host(fb::DeviceResources &res, const double *from, std::size_t count) {
  std::vector<double> to(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(to.data(), from, count * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync());
  return to;
}

// The CI-relaxed Hessian at a solution, as df_gradient and df_nac build it,
// on @p res: kept on the heap, as its members point into their neighbours.
struct Relaxed {
  std::optional<fb::ci::DeviceStringSet> strings;
  std::optional<fb::ci::Rdm> rdm;
  std::optional<fb::CasscfIntegrals> ints;
  std::optional<fb::response::Fock> fock;
  std::optional<fb::ci::Sigma> sigma;
  std::optional<fb::response::Hessian> hessian;
  std::optional<fb::response::RelaxedHessian> relaxed;
};

// The SA-2 CAS(4, 4) and its readers' inputs, made once for a suite.
struct SaCasscf {
  Plan plan{
      System{.mode = fb::EriMode::DensityFitting, .tables = std::string_view{kSaCasscfTables}}};
  std::optional<fb::CasscfOutput> cas;
  std::optional<fb::mcscf::Solution> solution;
  // The same CASSCF at unequal weights, converged when first asked for: what
  // makes the Z-vector carry rotations among the roots.
  std::optional<fb::CasscfOutput> skewed_cas;
  std::optional<fb::mcscf::Solution> skewed_solution;
  static constexpr std::string_view kSaCasscfTables =
      "[casscf]\nncas = 4\nnelecas = 4\nnroots = 2\nconv_tol = 1e-6\nconv_tol_grad = 1e-3\n"
      "[casscf_gradient]\nroot = 1\n[nac]\n";

  SaCasscf() {
    plan.run_through_scf();
    auto ran = planned<fb::CasscfStep>(*plan.calc).run(casscf_view(plan), plan.inputs);
    EXPECT_TRUE(ran) << ran.error().message;
    cas.emplace(std::move(*ran));
    solution.emplace(solution_of(plan.inputs, basis(), *cas));
  }

  // The CASSCF at weights 0.7 and 0.3, and its solution.
  [[nodiscard]] const fb::mcscf::Solution &skewed() {
    if (!skewed_solution) {
      auto casscf = planned<fb::CasscfStep>(*plan.calc);
      casscf.weights = {0.7, 0.3};
      auto ran = casscf.run(casscf_view(plan), plan.inputs);
      EXPECT_TRUE(ran) << ran.error().message;
      skewed_cas.emplace(std::move(*ran));
      skewed_solution.emplace(solution_of(plan.inputs, basis(), *skewed_cas));
    }
    return *skewed_solution;
  }

  [[nodiscard]] const fb::Molecule &molecule() const { return plan.molecule->molecule; }
  [[nodiscard]] const fb::BasisSet &basis() const { return plan.basis->basis; }
  [[nodiscard]] std::span<const fb::PrimitivePairClass> classes() const {
    return plan.pairs->classes;
  }
  [[nodiscard]] const fb::SphericalMatrixAO &h() const { return plan.h->h; }
  [[nodiscard]] const fb::DensityFitted &fit() const {
    return std::get<fb::DensityFitted>(plan.eri->eri);
  }
  [[nodiscard]] std::int32_t ncore() const { return cas->ncore; }
  [[nodiscard]] std::int32_t ncas() const { return cas->ncas; }
  [[nodiscard]] std::int32_t nelecas() const { return cas->nelecas; }
  [[nodiscard]] std::int32_t nmo() const { return static_cast<std::int32_t>(basis().nao_sphe()); }
  [[nodiscard]] std::int32_t naux() const { return fit().naux; }
  [[nodiscard]] std::int32_t pairs() const {
    return (ncore() * (nmo() - ncore())) + (ncas() * (nmo() - ncore() - ncas()));
  }
  [[nodiscard]] std::int32_t dim() const {
    return static_cast<std::int32_t>(cas->vectors.size() / cas->energies.size());
  }

  // The CI-relaxed Hessian at @p sol, its weights the solution's, on the
  // plan's DeviceResources.
  [[nodiscard]] std::unique_ptr<Relaxed> relaxed(const fb::mcscf::Solution &sol) {
    fb::DeviceResources &res = plan.inputs;
    auto out = std::make_unique<Relaxed>();
    out->strings.emplace(fb::ci::DeviceStringSet::create(res, ncas(), nelecas() / 2).value());
    out->rdm.emplace(fb::ci::Rdm::create(res, *out->strings, *out->strings).value());
    out->ints.emplace(
        fb::casscf_integrals(res, basis(), molecule(), h(), fit().b, naux(), sol.c, ncore(), ncas())
            .value());
    out->fock.emplace(fb::response::fock(res, molecule(), h(), fit().b, naux(), sol.c, ncore(),
                                         ncas(), out->ints->ppaa, out->ints->papa, sol.gamma,
                                         sol.gamma2)
                          .value());
    out->sigma.emplace(fb::ci::Sigma::create(res, *out->strings, *out->strings).value());
    const auto m = static_cast<std::size_t>(ncas()) * static_cast<std::size_t>(ncas());
    EXPECT_TRUE(out->sigma->set_integrals(res, out->ints->hamiltonian.e_core,
                                          to_host(res, out->ints->hamiltonian.h_eff.data(), m),
                                          to_host(res, out->ints->hamiltonian.eri.data(), m * m)));
    out->hessian.emplace(fb::response::Hessian::create(res, h(), fit().b, naux(), sol.c, ncore(),
                                                       ncas(), out->ints->ppaa, out->ints->papa,
                                                       *out->fock)
                             .value());
    const auto mo = static_cast<std::size_t>(nmo());
    fb::response::ResponseIntegrals response{mo,
                                             static_cast<std::size_t>(ncore()),
                                             static_cast<std::size_t>(ncas()),
                                             to_host(res, out->fock->fock_inactive.data(), mo * mo),
                                             to_host(res, out->ints->ppaa.data(), mo * mo * m),
                                             to_host(res, out->ints->papa.data(), mo * mo * m)};
    auto made = fb::response::RelaxedHessian::create(res, *out->hessian, *out->sigma, *out->rdm,
                                                     *out->strings, *out->strings, sol.roots,
                                                     sol.weights, std::move(response));
    EXPECT_TRUE(made) << made.error().message;
    out->relaxed.emplace(std::move(*made));
    return out;
  }
};

// The SA-2 CAS(4, 4), converged at the suite's first case that asks for it
// (after its FBF_REQUIRE_GPU) and freed at the suite's end.
class CasscfReaderFootprint : public ::testing::Test {
protected:
  static void TearDownTestSuite() { sa_.reset(); }
  static SaCasscf &sa() {
    if (!sa_) {
      sa_ = std::make_unique<SaCasscf>();
    }
    return *sa_;
  }

private:
  static inline std::unique_ptr<SaCasscf> sa_;
};

// One root of @p sol as a solution of its own, on @p res: what df_gradient
// takes without a response (the state-specific gradient). Its RDMs' values do
// not matter to what it allocates; their sizes are checked.
fb::mcscf::Solution one_root(fb::DeviceResources &res, const fb::BasisSet &basis,
                             const fb::CasscfOutput &cas) {
  const auto n2 = static_cast<std::size_t>(cas.ncas) * static_cast<std::size_t>(cas.ncas);
  const std::size_t dim = cas.vectors.size() / cas.energies.size();
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, basis),
      .roots = {.energies = {cas.energies[0]},
                .spin_squares = {cas.spin_squares[0]},
                .vectors = fb::Buffers::device<double>(dim, res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas.energies[0],
      .gradient_norm = cas.gradient_norm,
      .gamma = std::vector<double>(cas.rdm1.begin(), cas.rdm1.begin() + std::ptrdiff_t(n2)),
      .gamma2 = cas.rdm2,
      .root_gammas = std::vector<double>(cas.rdm1.begin(), cas.rdm1.begin() + std::ptrdiff_t(n2)),
      .iterations = cas.iterations,
      .weights = {1.0},
  };
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                dim * sizeof(double), wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                cas.c.num_elements() * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync());
  return solution;
}

// df_gradient at root 1 of the state average: the Z-vector's Lagrangian, the
// orbital response and the terms. Live at the peak, the Lagrangian's solve:
// the integrals' 4 and the Fock's 6, kappa, the strings' 2 and the RDMs, the
// sigma's 3 tables and its scratch, the Hessian's 11, the relaxed Hessian's 9
// and H'(x)'s sigma's 4, the right-hand side's Fock's 6, the ZVector's 2, its
// work 2, pcg's 1 and the product's 1.
TEST_F(CasscfReaderFootprint, DfGradient) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  SaCasscf &s = sa();
  const fb::DensityFitted &fit = s.fit();
  expect_step_footprint(fb::mcscf::df_gradient_footprint(s.basis(), fit.aux, s.ncore(), s.ncas(),
                                                         s.nelecas(), 1, 2, false),
                        4 + 6 + 1 + 3 + 4 + 11 + 13 + 6 + 6, [&](fb::DeviceResources &res) {
                          EXPECT_TRUE(fb::mcscf::df_gradient(
                              res, s.molecule(), s.basis(), s.classes(), fit.aux, s.h(), fit.b,
                              fit.l.data(), *s.solution, s.ncore(), s.ncas(), 1));
                        });
}

// The routines the gradient and the couplings share, each held alone: the
// transition 1-RDM, the transition orbital gradient, the root-pair
// multipliers, zeta's terms (one pair, zeta nonzero), the overlap's half
// gradient and the one-electron terms at a caller's D and -W.
TEST_F(CasscfReaderFootprint, SharedRoutines) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  SaCasscf &s = sa();
  fb::DeviceResources &in = s.plan.inputs;
  const fb::mcscf::Solution &sol = *s.solution;
  const auto m = static_cast<std::size_t>(s.ncas()) * static_cast<std::size_t>(s.ncas());
  const auto dim = static_cast<std::size_t>(s.dim());
  auto strings = fb::ci::DeviceStringSet::create(in, s.ncas(), s.nelecas() / 2);
  ASSERT_TRUE(strings);
  auto rdm = fb::ci::Rdm::create(in, *strings, *strings);
  ASSERT_TRUE(rdm);
  ASSERT_EQ(static_cast<std::size_t>(rdm->size()), dim);
  const double *const bra = sol.roots.vectors.data();
  const double *const ket = bra + dim;
  fb::Buffers::device<double> rdms(m + (m * m), in.device());
  ASSERT_TRUE(rdm->trans_rdm12(in, ket, bra, rdms.data(), rdms.data() + m));
  const std::vector<double> gamma = to_host(in, rdms.data(), m);
  const std::vector<double> gamma2 = to_host(in, rdms.data() + m, m * m);
  const fb::CasscfIntegrals ints =
      fb::casscf_integrals(in, s.basis(), s.molecule(), s.h(), s.fit().b, s.naux(), sol.c,
                           s.ncore(), s.ncas())
          .value();

  // One chunk of D, the workspace alone.
  expect_step_footprint(
      fb::ci::trans_rdm1_footprint(s.ncas(), s.dim()), 0,
      [&](fb::DeviceResources &res) { EXPECT_TRUE(rdm->trans_rdm1(res, bra, ket, rdms.data())); });
  // Both Focks' six members each, and the second's 2 P_c, J and K.
  const fb::MemoryFootprint transition =
      fb::mcscf::transition_orbital_gradient_footprint(s.nmo(), s.naux(), s.ncore(), s.ncas());
  EXPECT_GT(transition.device.persistent, 0U);
  expect_step_footprint(transition, 6 + 6 + 3, [&](fb::DeviceResources &res) {
    EXPECT_TRUE(fb::mcscf::transition_orbital_gradient(res, s.molecule(), s.h(), s.fit().b,
                                                       s.naux(), sol.c, s.ncore(), s.ncas(),
                                                       ints.ppaa, ints.papa, gamma, gamma2));
  });
  // The overlaps.
  fb::Buffers::device<double> y_bar(2 * dim, in.device());
  EXPECT_EQ(wwr::wwrMemsetAsync(y_bar.data(), 0, sizeof(double) * 2 * dim, in.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(in.sync());
  expect_step_footprint(
      fb::mcscf::root_pair_multipliers_footprint(2), 1, [&](fb::DeviceResources &res) {
        EXPECT_TRUE(fb::mcscf::root_pair_multipliers(res, sol.roots, sol.weights, y_bar.data()));
      });
  // The orbital part and the RDMs, and the pair's transition gradient's.
  const std::vector<double> zeta{0.25};
  expect_step_footprint(
      fb::mcscf::zeta_terms_footprint(s.nmo(), s.naux(), s.ncore(), s.ncas(), s.dim()),
      2 + 6 + 6 + 3, [&](fb::DeviceResources &res) {
        EXPECT_TRUE(fb::mcscf::zeta_terms(res, s.molecule(), s.h(), s.fit().b, s.naux(), sol,
                                          s.ncore(), s.ncas(), ints.ppaa, ints.papa, *rdm, zeta));
      });
  // U, the pair's three, the slabs, their spherical copies and the rows.
  fb::SphericalMatrixAO a(in, s.basis());
  EXPECT_EQ(wwr::wwrMemsetAsync(a.data(), 0, sizeof(double) * a.num_elements(), in.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(in.sync());
  expect_step_footprint(fb::overlap_half_gradient_footprint(s.basis()), 7,
                        [&](fb::DeviceResources &res) {
                          EXPECT_TRUE(fb::overlap_half_gradient(res, s.basis(), s.classes(), a));
                        });
  // One derivative call's seven.
  expect_step_footprint(fb::gradient::one_electron_terms_footprint(s.basis()), 7,
                        [&](fb::DeviceResources &res) {
                          fb::gradient::RhfTerms g;
                          EXPECT_TRUE(fb::gradient::one_electron_terms(res, s.molecule(), s.basis(),
                                                                       s.classes(), a, a, g));
                        });
}

// zvector over the CI-relaxed Hessian at @p sol, g the Fock's at root 0's
// 1-RDM; at unequal weights (@p skew) with a CI right-hand side and a rotation
// one, so the rotations' matrices are taken too. The products' sigma growth is
// the warm-up run's, as its owner keeps it.
void expect_zvector_footprint(SaCasscf &s, const fb::mcscf::Solution &sol, const bool skew) {
  fb::DeviceResources &in = s.plan.inputs;
  const auto m = static_cast<std::size_t>(s.ncas()) * static_cast<std::size_t>(s.ncas());
  const fb::MemoryFootprint product =
      fb::response::hessian_apply_footprint(s.nmo(), s.naux(), s.ncore(), s.ncas());
  const std::unique_ptr<Relaxed> r = s.relaxed(sol);
  const std::vector<double> root_gamma(sol.root_gammas.begin(),
                                       sol.root_gammas.begin() + std::ptrdiff_t(m));
  const fb::response::Fock own =
      fb::response::fock(in, s.molecule(), s.h(), s.fit().b, s.naux(), sol.c, s.ncore(), s.ncas(),
                         r->ints->ppaa, r->ints->papa, root_gamma, sol.gamma2)
          .value();
  fb::Buffers::device<double> ci_rhs(2 * static_cast<std::size_t>(s.dim()), in.device());
  EXPECT_EQ(
      wwr::wwrMemsetAsync(ci_rhs.data(), 0, sizeof(double) * ci_rhs.num_elements(), in.stream()),
      wwr::wwrSuccess);
  ASSERT_TRUE(in.sync());
  const std::vector<double> theta_rhs{1e-3};
  // z_o and z_c, -b and the rest, the overlaps, pcg's vectors and the
  // product's scratch.
  expect_step_footprint(
      fb::response::zvector_footprint(s.pairs(), s.dim(), 2, product, skew, skew), 6,
      [&](fb::DeviceResources &res) {
        const auto z =
            skew ? fb::response::zvector(res, *r->relaxed, *r->hessian, sol.roots,
                                         own.gradient.data(), ci_rhs.data(), 1e-10, 200, theta_rhs)
                 : fb::response::zvector(res, *r->relaxed, *r->hessian, sol.roots,
                                         own.gradient.data());
        ASSERT_TRUE(z) << z.error().message;
        EXPECT_EQ(std::ranges::any_of(z->theta, [](double t) { return t != 0.0; }), skew);
      });
}

TEST_F(CasscfReaderFootprint, Zvector) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_zvector_footprint(sa(), *sa().solution, false);
}

// The other calls df_gradient_footprint covers: one root, no response; a mix
// summing to 0; and
// df_transition_gradient at the pair's transition RDMs, without multipliers
// and with given ones and an extra adjoint (values that do not matter).
TEST_F(CasscfReaderFootprint, DfGradientForms) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  SaCasscf &s = sa();
  fb::DeviceResources &in = s.plan.inputs;
  const fb::DensityFitted &fit = s.fit();
  const auto fp = [&](std::int32_t nroots, bool rotating, fb::mcscf::GradientForm form,
                      bool multipliers = false, bool extra = false) {
    return fb::mcscf::df_gradient_footprint(s.basis(), fit.aux, s.ncore(), s.ncas(), s.nelecas(), 1,
                                            nroots, rotating, form, multipliers, extra);
  };
  const auto call = [&](fb::DeviceResources &res, const fb::mcscf::Solution &sol,
                        std::span<const double> mix) {
    EXPECT_TRUE(fb::mcscf::df_gradient(res, s.molecule(), s.basis(), s.classes(), fit.aux, s.h(),
                                       fit.b, fit.l.data(), sol, s.ncore(), s.ncas(), mix));
  };
  {
    const fb::mcscf::Solution single = one_root(in, s.basis(), *s.cas);
    const std::vector<double> mix{1.0};
    expect_step_footprint(fp(1, false, fb::mcscf::GradientForm::Mix), 24,
                          [&](fb::DeviceResources &res) { call(res, single, mix); });
  }
  const std::vector<double> zero{0.5, -0.5};
  expect_step_footprint(fp(2, false, fb::mcscf::GradientForm::ZeroMix), 70,
                        [&](fb::DeviceResources &res) { call(res, *s.solution, zero); });

  const fb::mcscf::Solution &sol = *s.solution;
  const auto m = static_cast<std::size_t>(s.ncas()) * static_cast<std::size_t>(s.ncas());
  const auto dim = static_cast<std::size_t>(s.dim());
  const std::unique_ptr<Relaxed> r = s.relaxed(sol);
  fb::Buffers::device<double> rdms(m + (m * m), in.device());
  ASSERT_TRUE(r->rdm->trans_rdm12(in, sol.roots.vectors.data() + dim, sol.roots.vectors.data(),
                                  rdms.data(), rdms.data() + m));
  const std::vector<double> gamma = to_host(in, rdms.data(), m);
  const std::vector<double> gamma2 = to_host(in, rdms.data() + m, m * m);
  const auto transition = [&](fb::DeviceResources &res, const fb::response::ZVector *z,
                              const fb::mcscf::MoAdjoint *extra) {
    EXPECT_TRUE(fb::mcscf::df_transition_gradient(res, s.molecule(), s.basis(), s.classes(),
                                                  fit.aux, s.h(), fit.b, fit.l.data(), sol,
                                                  s.ncore(), s.ncas(), gamma, gamma2, z, extra));
  };
  expect_step_footprint(fp(2, false, fb::mcscf::GradientForm::Transition), 24,
                        [&](fb::DeviceResources &res) { transition(res, nullptr, nullptr); });
  const fb::Buffers::device<double> g =
      fb::mcscf::transition_orbital_gradient(in, s.molecule(), s.h(), fit.b, s.naux(), sol.c,
                                             s.ncore(), s.ncas(), r->ints->ppaa, r->ints->papa,
                                             gamma, gamma2)
          .value();
  const fb::response::ZVector z =
      fb::response::zvector(in, *r->relaxed, *r->hessian, sol.roots, g.data()).value();
  const auto nn = static_cast<std::size_t>(s.nmo()) * static_cast<std::size_t>(s.nmo());
  fb::Buffers::device<double> adjoint(nn + (static_cast<std::size_t>(s.naux()) * nn) + nn,
                                      in.device());
  EXPECT_EQ(
      wwr::wwrMemsetAsync(adjoint.data(), 0, sizeof(double) * adjoint.num_elements(), in.stream()),
      wwr::wwrSuccess);
  ASSERT_TRUE(in.sync());
  const fb::mcscf::MoAdjoint extra{.h_bar = adjoint.data(),
                                   .lambda = adjoint.data() + nn,
                                   .x = adjoint.data() + nn +
                                        (static_cast<std::size_t>(s.naux()) * nn)};
  expect_step_footprint(fp(2, false, fb::mcscf::GradientForm::Transition, true, true), 40,
                        [&](fb::DeviceResources &res) { transition(res, &z, &extra); });
}

// df_nac for the pair (0, 1) at @p sol: its setup, the pair's RDMs, the
// Z-vector, h_IJ and d^CSF; @p skew says the weights are unequal, and the
// Z-vector rotates the roots. Live at the peak, h_IJ's terms beside the setup
// and the Z-vector.
void expect_df_nac_footprint(SaCasscf &s, const fb::mcscf::Solution &sol, const bool skew) {
  const fb::DensityFitted &fit = s.fit();
  expect_step_footprint(
      fb::mcscf::df_nac_footprint(s.basis(), fit.aux, s.ncore(), s.ncas(), s.nelecas(), 1, 2, skew),
      80, [&](fb::DeviceResources &res) {
        EXPECT_TRUE(fb::mcscf::df_nac(res, s.molecule(), s.basis(), s.classes(), fit.aux, s.h(),
                                      fit.b, fit.l.data(), sol, s.ncore(), s.ncas(), 0, 1,
                                      fb::mcscf::NacOptions{}));
      });
}

TEST_F(CasscfReaderFootprint, DfNac) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_df_nac_footprint(sa(), *sa().solution, false);
}

// The combination: nac_combination_terms alone, then the combination's df_nac,
// over T coupling the two roots and a D^X with one core-virtual element.
TEST_F(CasscfReaderFootprint, DfNacCombination) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  SaCasscf &s = sa();
  const fb::DensityFitted &fit = s.fit();
  const std::vector<double> t{0.0, 0.3, -0.3, 0.0};
  const auto mo = static_cast<std::size_t>(s.nmo());
  std::vector<double> d_x(mo * mo, 0.0);
  d_x[((mo - 1) * mo) + 0] = 0.1;
  // The strings, the RDMs, and a half gradient's: a, the work, A, U, the pair
  // and the slabs.
  expect_step_footprint(
      fb::mcscf::nac_combination_terms_footprint(s.basis(), s.ncas(), s.nelecas(), 1, true),
      2 + 1 + 3 + 7, [&](fb::DeviceResources &res) {
        EXPECT_TRUE(fb::mcscf::nac_combination_terms(res, s.molecule(), s.basis(), s.classes(),
                                                     fit.aux, fit.b, *s.solution, s.ncore(),
                                                     s.ncas(), t, d_x));
      });
  expect_step_footprint(fb::mcscf::df_nac_combination_footprint(s.basis(), fit.aux, s.ncore(),
                                                                s.ncas(), s.nelecas(), 1, 2, false,
                                                                true),
                        80, [&](fb::DeviceResources &res) {
                          EXPECT_TRUE(fb::mcscf::df_nac(res, s.molecule(), s.basis(), s.classes(),
                                                        fit.aux, s.h(), fit.b, fit.l.data(),
                                                        *s.solution, s.ncore(), s.ncas(), t, d_x));
                        });
}

// The steps at plan_shape's dimensions: the CASSCF's two roots, its explicit
// space; over the CASSCF at its own equal weights, and at the skewed ones
// with the dimensions' flag set as plan_shape sets it for them
// (PipelineShape.CasscfRootsAndWeights). Live at the peak, the upload's two
// beside their call's.
template <class Step>
void expect_reader_step_footprint(SaCasscf &s, const bool skew, const std::size_t live) {
  fb::FootprintDimensions dims = s.plan.dims();
  ASSERT_EQ(dims.casscf_nroots, 2U);
  ASSERT_FALSE(dims.casscf_unequal_weights);
  ASSERT_EQ(dims.ncas, 4U);
  ASSERT_EQ(dims.nelecas, 4U);
  dims.casscf_unequal_weights = skew;
  if (skew) {
    (void)s.skewed();
  }
  const fb::CasscfOutput &cas = skew ? *s.skewed_cas : *s.cas;
  const auto step = planned<Step>(*s.plan.calc);
  const fb::MemoryFootprint fp = step.footprint(dims);
  EXPECT_EQ(fp.device.persistent, 0U);
  expect_step_footprint(fp, live, [&](fb::DeviceResources &res) {
    expect_ran(step.run(
        fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput,
                 fb::EriOutput, fb::CoreHamiltonianOutput, fb::CasscfOutput>{
            *s.plan.molecule, *s.plan.basis, *s.plan.pairs, *s.plan.eri, *s.plan.h, cas},
        res));
  });
}

constexpr std::size_t kGradientStepLive = 2 + 4 + 6 + 1 + 3 + 4 + 11 + 13 + 6 + 6;
constexpr std::size_t kNacStepLive = 2 + 80;

TEST_F(CasscfReaderFootprint, CasscfGradientStep) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_reader_step_footprint<fb::CasscfGradientStep>(sa(), false, kGradientStepLive);
}

TEST_F(CasscfReaderFootprint, NacStep) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_reader_step_footprint<fb::NacStep>(sa(), false, kNacStepLive);
}

// Everything above that unequal weights change, over the CASSCF converged at
// weights 0.7 and 0.3: zvector with the rotations, df_gradient at a root,
// df_nac, and both steps.
TEST_F(CasscfReaderFootprint, UnequalWeights) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  SaCasscf &s = sa();
  const fb::mcscf::Solution &sol = s.skewed();
  const fb::DensityFitted &fit = s.fit();
  expect_zvector_footprint(s, sol, true);
  expect_step_footprint(fb::mcscf::df_gradient_footprint(s.basis(), fit.aux, s.ncore(), s.ncas(),
                                                         s.nelecas(), 1, 2, true),
                        kGradientStepLive - 2, [&](fb::DeviceResources &res) {
                          EXPECT_TRUE(fb::mcscf::df_gradient(
                              res, s.molecule(), s.basis(), s.classes(), fit.aux, s.h(), fit.b,
                              fit.l.data(), sol, s.ncore(), s.ncas(), 1));
                        });
  expect_df_nac_footprint(s, sol, true);
  expect_reader_step_footprint<fb::CasscfGradientStep>(s, true, kGradientStepLive);
  expect_reader_step_footprint<fb::NacStep>(s, true, kNacStepLive);
}

// The CIS and Localization steps (#1159), each run on the Scf of its parsed
// water config, at plan_shape's dimensions: CIS's Davidson subspace (40
// columns, past nao) is in syevd's table, and MINAO is placed for IBO and the
// LIVVOs. CIS's workspace is held to at most the footprint's: its Davidson's
// syevd is counted at the subspace's largest order, which a solve that
// converges first never reaches. Each step's product is on the host (CIS) or
// its C, U and LIVVOs (Localization).

// What a CIS step reads.
fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
         fb::PrimitivePairClassesOutput, fb::EriOutput, fb::ScfOutput>
cis_view(const Plan &plan) {
  return fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                  fb::PrimitivePairClassesOutput, fb::EriOutput, fb::ScfOutput>{
      *plan.molecule, *plan.basis, *plan.transform, *plan.pairs, *plan.eri, *plan.scf};
}

// What a Localization step reads.
fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::OverlapOutput, fb::ScfOutput>
localization_view(const Plan &plan) {
  return fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::OverlapOutput, fb::ScfOutput>{
      *plan.molecule, *plan.basis, *plan.overlap, *plan.scf};
}

// Integral-direct singlets: at the Davidson's peak, the orbital energies, Δε,
// the guess, θ, the roots' values and vectors; the sigma's two pairs (three
// each), its other matrix and half transform; and the Davidson's 11. Under
// slow_test.h: the integral-direct sigma's columns run one at a time, past
// 1.5 s; the density-fitted and TDA-ris cases stay in the fast set.
TEST(StepFootprint, CisDirect) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  Plan plan(System{.tables = "[cis]\nn_states = 3\n"});
  plan.run_through_scf();
  const auto cis = planned<fb::CisStep>(*plan.calc);
  const fb::FootprintDimensions &dims = plan.dims();
  ASSERT_GT(dims.syevd_lwork.size(), static_cast<std::size_t>(cis.max_subspace));
  expect_step_footprint(
      cis.footprint(dims), 6 + 8 + 11,
      [&](fb::DeviceResources &res) { expect_ran(cis.run(cis_view(plan), res)); }, {}, true);
}

// Density-fitted, with clusters at a gap wide enough that the guess's six
// entries are one (cluster_footprint's most): at the Davidson's peak the six,
// the three MO blocks, the clusters' indices, eigenvalues and eigenvectors,
// and the Davidson's 11.
TEST(StepFootprint, CisDensityFittedClustered) {
  FBF_REQUIRE_GPU();
  Plan plan(System{.mode = fb::EriMode::DensityFitting,
                   .tables = "[cis]\nn_states = 3\ncluster_gap = 100.0\n"});
  plan.run_through_scf();
  const auto cis = planned<fb::CisStep>(*plan.calc);
  expect_step_footprint(
      cis.footprint(plan.dims()), 6 + 3 + 3 + 11,
      [&](fb::DeviceResources &res) { expect_ran(cis.run(cis_view(plan), res)); }, {}, true);
}

// TDA-ris with RIS-JK's preconditioner, over the integral-direct Eri: getrf's
// work at the RIS basis's order is plan_shape's, the solver's own query there
// (0 on HIP, whose getrf takes no work). The peak is the sigma's J
// fit's three-centre tensor: the four before the preconditioner, its 11
// buffers, the roots' two, the three MO blocks, the fit's metric pair (U is
// the identity over s and p) and B, and three_center's four.
TEST(StepFootprint, CisRis) {
  FBF_REQUIRE_GPU();
  Plan plan(System{.tables = "[cis]\nn_states = 3\nmethod = \"ris\"\nris = true\n"
                             "ris_exchange = true\n"});
  plan.run_through_scf();
  const auto cis = planned<fb::CisStep>(*plan.calc);
  ASSERT_TRUE(cis.options.ris.has_value());
  const auto ris_aux =
      fb::cis::ris_basis(plan.dims().atoms, cis.options.ris->theta, cis.options.ris->shells);
  ASSERT_TRUE(ris_aux) << ris_aux.error().message;
  const auto lwork = fb::getrf_lwork(plan.inputs, static_cast<std::int32_t>(ris_aux->nao_sphe()));
  ASSERT_TRUE(lwork) << lwork.error().message;
  ASSERT_EQ(plan.dims().getrf_lwork, *lwork);
  expect_step_footprint(
      cis.footprint(plan.dims()), 4 + 11 + 2 + 3 + 2 + 1 + 4,
      [&](fb::DeviceResources &res) { expect_ran(cis.run(cis_view(plan), res)); }, {}, true);
}

// Pipek-Mezey, closed shell: at the sort, the product's C and U, and the
// keys, the permutation and the permuted C and U.
TEST(StepFootprint, LocalizationPipekMezey) {
  FBF_REQUIRE_GPU();
  Plan plan(System{.tables = "[localization]\n"});
  plan.run_through_scf();
  const auto localization = planned<fb::LocalizationStep>(*plan.calc);
  EXPECT_EQ(plan.dims().minao, nullptr);
  expect_step_footprint(localization.footprint(plan.dims()), 2 + 4, [&](fb::DeviceResources &res) {
    expect_ran(localization.run(localization_view(plan), res));
  });
}

// The cation's doublet: the doubly and singly occupied spaces each swept on
// their own, an open shell's U block held beside them.
TEST(StepFootprint, LocalizationOpenShell) {
  FBF_REQUIRE_GPU();
  Plan plan(System{.cation = true, .tables = "[localization]\n"});
  plan.run_through_scf();
  const auto localization = planned<fb::LocalizationStep>(*plan.calc);
  ASSERT_NE(plan.dims().n_alpha, plan.dims().n_beta);
  expect_step_footprint(localization.footprint(plan.dims()), 2 + 1 + 4,
                        [&](fb::DeviceResources &res) {
                          expect_ran(localization.run(localization_view(plan), res));
                        });
}

// IBO with the LIVVOs and both Molden files: MINAO is plan_shape's, and
// syevd's table reaches its order. The peak is the IAOs' first overlap,
// cc-pVDZ against MINAO: the IAOs and the product's three, S12 and S2, and
// overlap_mixed_spherical's four buffers beside its records' fields (20 a
// class) and 11 tables. The workspace is held
// to at most the footprint's, which counts the near-singular S1's fallback
// (orthogonalize at nao), which water's S1 does not take.
TEST(StepFootprint, LocalizationIboLivvo) {
  FBF_REQUIRE_GPU();
  Plan plan(System{.tables = "[localization]\nmode = \"ibo\"\nlivvo = true\n"});
  plan.run_through_scf();
  const auto localization = planned<fb::LocalizationStep>(*plan.calc);
  ASSERT_NE(plan.dims().minao, nullptr);
  ASSERT_GT(plan.dims().syevd_lwork.size(), plan.dims().minao->nao_sphe());
  const std::size_t n_classes =
      fb::mixed_primitive_pair_classes(plan.inputs, *plan.dims().basis, *plan.dims().minao)
          .value()
          .size();
  expect_step_footprint(
      localization.footprint(plan.dims()), 1 + 3 + 2 + 4 + (20 * n_classes) + 11,
      [&](fb::DeviceResources &res) { expect_ran(localization.run(localization_view(plan), res)); },
      {}, true);
}

// The NEVPT2 steps (#1052) and the routines their footprints compose, over
// water's CAS(4, 4), every build fitted: the SA-2 CASSCF the Nevpt2 step,
// the QD gradient and the couplings read, and the single-state one the
// state-specific gradients read, each converged once for its suite. Every
// step runs on a pool capped at its footprint, the headroom and the pool's
// rounding, so each adaptive part (nevpt2::rdms' and rdms_adjoint's tiles,
// the classes' slab batch) takes the least it runs with, as its footprint
// counts it. PC's kept ranks are the data's: a step footprint counts each
// class at the rank that holds the most, and the test's slack is the
// difference the run's ranks make, from the library footprints at them.

// A CASSCF the NEVPT2 steps read, and its mcscf::Solution on the plan's
// DeviceResources.
struct Nevpt2Reference {
  Plan plan;
  std::optional<fb::CasscfOutput> cas;
  std::optional<fb::mcscf::Solution> solution;

  explicit Nevpt2Reference(const std::string_view tables)
      : plan(System{.mode = fb::EriMode::DensityFitting, .tables = tables}) {
    plan.run_through_scf();
    auto ran = planned<fb::CasscfStep>(*plan.calc).run(casscf_view(plan), plan.inputs);
    EXPECT_TRUE(ran) << ran.error().message;
    cas.emplace(std::move(*ran));
    solution.emplace(solution_of(plan.inputs, plan.basis->basis, *cas));
  }

  [[nodiscard]] const fb::FootprintDimensions &dims() const { return plan.dims(); }
  [[nodiscard]] const fb::DensityFitted &fit() const {
    return std::get<fb::DensityFitted>(plan.eri->eri);
  }
  [[nodiscard]] std::int32_t nmo() const { return static_cast<std::int32_t>(plan.dims().nao); }
  [[nodiscard]] std::int32_t nvirt() const { return nmo() - cas->ncore - cas->ncas; }
  [[nodiscard]] std::int32_t nroots() const {
    return static_cast<std::int32_t>(cas->energies.size());
  }
  [[nodiscard]] fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::EriOutput,
                         fb::CoreHamiltonianOutput, fb::ScfOutput, fb::CasscfOutput>
  nevpt2_view() const {
    return fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::EriOutput,
                    fb::CoreHamiltonianOutput, fb::ScfOutput, fb::CasscfOutput>{
        *plan.molecule, *plan.basis, *plan.eri, *plan.h, *plan.scf, *cas};
  }
  [[nodiscard]] fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput,
                         fb::EriOutput, fb::CoreHamiltonianOutput, fb::ScfOutput, fb::CasscfOutput>
  reader_view() const {
    return fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput,
                    fb::EriOutput, fb::CoreHamiltonianOutput, fb::ScfOutput, fb::CasscfOutput>{
        *plan.molecule, *plan.basis, *plan.pairs, *plan.eri, *plan.h, *plan.scf, *cas};
  }
  // Every state's PC class counts, as qd_pc_energy drops them on the
  // solution's own orbitals: what the effective Hamiltonians' kets keep.
  [[nodiscard]] std::vector<std::array<std::int64_t, 8>> qd_dropped() {
    fb::DeviceResources &res = plan.inputs;
    auto alpha = fb::ci::DeviceStringSet::create(res, cas->ncas, cas->nelecas / 2).value();
    auto beta = fb::ci::DeviceStringSet::create(res, cas->ncas, cas->nelecas / 2).value();
    const auto e =
        fb::nevpt2::qd_pc_energy(res, alpha, beta, fit().naux, nmo(), nmo(), fit().b.data(),
                                 solution->c.data(), plan.h->h.data(), cas->ncore, cas->ncas,
                                 solution->weights, solution->roots.vectors.data(), 0);
    EXPECT_TRUE(e) << e.error().message;
    std::vector<std::array<std::int64_t, 8>> out;
    for (std::int32_t j = 0; j < nroots() && e; ++j) {
      out.push_back((*e)[static_cast<std::size_t>((j * nroots()) + j)].dropped);
    }
    return out;
  }
};

// The SA-2 CASSCF, every NEVPT2 step configured over it, and the
// single-state one, converged at the suite's first case that asks for each.
class Nevpt2ReaderFootprint : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    sa_.reset();
    ss_.reset();
  }
  static Nevpt2Reference &sa() {
    if (!sa_) {
      sa_ = std::make_unique<Nevpt2Reference>(
          "[casscf]\nncas = 4\nnelecas = 4\nnroots = 2\nconv_tol = 1e-8\nconv_tol_grad = 1e-6\n"
          "[nevpt2]\nquasi_degenerate = true\n[nevpt2_gradient]\nquasi_degenerate = true\n"
          "[nevpt2_nac]\n");
    }
    return *sa_;
  }
  static Nevpt2Reference &ss() {
    if (!ss_) {
      ss_ = std::make_unique<Nevpt2Reference>(
          "[casscf]\nncas = 4\nnelecas = 4\nconv_tol = 1e-8\nconv_tol_grad = 1e-6\n"
          "[nevpt2_gradient]\n");
    }
    return *ss_;
  }

private:
  static inline std::unique_ptr<Nevpt2Reference> sa_;
  static inline std::unique_ptr<Nevpt2Reference> ss_;
};

// @p run, a step's or a routine's, held to @p fp with every adaptive part at
// its least: on a fresh DeviceResources whose pool is capped at the
// footprint's peak, the 64 MiB of fragmentation headroom an adaptive part
// leaves of the budget, and the pool's rounding of the allocations live at
// the peak (FciAtTheLeast's cap); once to grow the workspaces to exactly the
// footprint's, then measured, with @p live allocations live at its peak.
template <typename Run>
void expect_least_footprint(const fb::MemoryFootprint &fp, const std::size_t live, Run &&run,
                            std::optional<fbt::FootprintSlack> slack = {},
                            std::source_location location = std::source_location::current()) {
  std::size_t granularity = 0;
  {
    fb::DeviceResources probe{0};
    granularity = fbt::pool_granularity(*probe.device());
  }
  const std::uint64_t cap = fp.device.peak() + fb::kPoolHeadroomBytes + (granularity * (live + 16));
  fb::DeviceResources res{0, fb::DeviceResources::kDefaultPoolReleaseThreshold, 0, cap};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  run(res);
  EXPECT_EQ(res.device_workspace(0).num_elements(), fp.device.workspace)
      << "the device workspace, from line " << location.line();
  EXPECT_EQ(res.pinned_bytes(), fp.pinned.peak())
      << "the pinned memory, from line " << location.line();
  const fbt::PoolPeak peak = fbt::measure_pool_peak(*res.device(), [&] { run(res); });
  std::cout << std::format(
      "[ footprint ] line {}: capped at {}: measured {} (G {}), footprint {} + workspace {}\n",
      location.line(), cap, peak.used, peak.granularity, fp.device.persistent + fp.device.scratch,
      fp.device.workspace);
  fbt::expect_footprint(peak, {.persistent = fp.device.persistent, .scratch = fp.device.scratch},
                        live, slack, location);
  EXPECT_LE(res.device()->pool_usage().used_high, cap) << "from line " << location.line();
}

// The difference @p most - @p at (their device peaks) as a footprint slack:
// PC's kept ranks are the data's, and @p most counts each class at the rank
// that holds the most.
fbt::FootprintSlack rank_slack(const std::size_t bytes) {
  return {.bytes = bytes,
          .reason = "PC's kept ranks are the data's: the footprint counts each class at the rank "
                    "that holds the most, the run keeps fewer (the library footprints' difference "
                    "at the run's ranks)"};
}

std::size_t peak_gap(const fb::MemoryFootprint &most, const fb::MemoryFootprint &at) {
  return most.device.peak() > at.device.peak() ? most.device.peak() - at.device.peak() : 0;
}

// Live at the peak of each case's measured run (test/support/pool_footprint.h's
// A), counted off a trace of its pool allocations. The Nevpt2 step's, at a
// root's PC Sr solve: the integrals, the strings, the CI vectors, the
// per-root arrays, the Fock, the MO blocks, the identity, Sr's matrices,
// its solve and U, T and T₁; with QD, at the diagonal walk's.
constexpr std::size_t kNevpt2StepLive = 51;
constexpr std::size_t kNevpt2QdStepLive = 62;
// The effective Hamiltonians and the PT2 densities alone.
constexpr std::size_t kScDiagonalLive = 19;
constexpr std::size_t kScPairLive = 8;
constexpr std::size_t kPcDiagonalLive = 24;
constexpr std::size_t kPcPairLive = 9;
constexpr std::size_t kScDensityLive = 25;
constexpr std::size_t kPcDensityLive = 20;
// The gradient's routines: sc_adjoint's and pc_adjoint's at the bars beside
// the forward's arrays, zc_adjoint's every buffer, the QD adjoints' at the
// walk beside every pair's RDMs.
constexpr std::size_t kScAdjointLive = 55;
constexpr std::size_t kPcAdjointLive = 55;
constexpr std::size_t kZcAdjointLive = 18;
constexpr std::size_t kQdScAdjointLive = 84;
constexpr std::size_t kQdPcAdjointLive = 182;
// The couplings', SC's at the df_gradient pass beside the Lagrangian, PC's
// at its adjoint's walk.
constexpr std::size_t kCouplingLive = 49;
constexpr std::size_t kPcCouplingLive = 186;

// The Nevpt2 step over the SA-2 CASSCF, SC and PC per root, at the least:
// its footprint, the PC classes counted at their whole bases, past the run
// by the ranks it keeps, the unconstrained run's (the same data). And the SC
// and PC energies at the least equal the unconstrained run's to 1e-10 Eh:
// the tile and the batch reorder sums only.
TEST_F(Nevpt2ReaderFootprint, Nevpt2Step) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  Nevpt2Reference &s = sa();
  ASSERT_EQ(s.dims().casscf_nroots, 2U);
  auto step = planned<fb::Nevpt2Step>(*s.plan.calc);
  step.quasi_degenerate = false;
  const fb::MemoryFootprint fp = step.footprint(s.dims());
  EXPECT_EQ(fp.device.persistent, 0U);

  // Unconstrained: the budget whole.
  fb::Result<fb::Nevpt2Output> whole = fb::err_config("not run");
  {
    fb::DeviceResources res{0};
    whole = step.run(s.nevpt2_view(), res);
    ASSERT_TRUE(whole) << whole.error().message;
  }
  const std::int32_t ncore = s.cas->ncore;
  const std::int32_t ncas = s.cas->ncas;
  const auto &lwork = s.dims().syevd_lwork;
  const fb::MemoryFootprint most =
      fb::nevpt2::pc_class_energies_footprint(ncore, ncas, s.nvirt(), -1, lwork);
  std::size_t slack = 0;
  for (const auto &pc : whole->pc_roots) {
    ASSERT_TRUE(pc.has_value());
    slack = std::max(slack, peak_gap(most, fb::nevpt2::pc_class_energies_footprint(
                                               ncore, ncas, s.nvirt(), -1, lwork, pc->dropped)));
  }

  std::optional<fb::Nevpt2Output> least;
  expect_least_footprint(
      fp, kNevpt2StepLive,
      [&](fb::DeviceResources &res) {
        auto out = step.run(s.nevpt2_view(), res);
        ASSERT_TRUE(out) << out.error().message;
        least.emplace(std::move(*out));
      },
      rank_slack(slack));
  ASSERT_TRUE(least.has_value());
  for (std::size_t r = 0; r < whole->roots.size(); ++r) {
    EXPECT_NEAR(least->roots[r].total, whole->roots[r].total, 1e-10) << "SC root " << r;
    ASSERT_TRUE(least->pc_roots[r].has_value());
    EXPECT_NEAR(least->pc_roots[r]->total, whole->pc_roots[r]->total, 1e-10) << "PC root " << r;
  }
}

// With quasi_degenerate: the roots' walks, then the state-averaged inputs,
// both effective Hamiltonians' diagonal walk with PC's kets kept, a pair's
// transition RDMs and couplings, and the two eigensolves. The slack is the
// largest of the ranks' differences over the PC calls the step makes: each
// root's classes, the diagonal walk, and a pair's walk beside the kets kept.
// The QD energies at the least equal the unconstrained run's to 1e-10 Eh.
TEST_F(Nevpt2ReaderFootprint, Nevpt2StepQuasiDegenerate) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  Nevpt2Reference &s = sa();
  const auto step = planned<fb::Nevpt2Step>(*s.plan.calc);
  ASSERT_TRUE(step.quasi_degenerate);
  const fb::MemoryFootprint fp = step.footprint(s.dims());
  EXPECT_EQ(fp.device.persistent, 0U);
  fb::Result<fb::Nevpt2Output> whole = fb::err_config("not run");
  {
    fb::DeviceResources res{0};
    whole = step.run(s.nevpt2_view(), res);
    ASSERT_TRUE(whole) << whole.error().message;
  }
  const std::int32_t ncore = s.cas->ncore;
  const std::int32_t ncas = s.cas->ncas;
  const std::int32_t n = s.nroots();
  const auto &lwork = s.dims().syevd_lwork;
  std::size_t slack = 0;
  const fb::MemoryFootprint most =
      fb::nevpt2::pc_class_energies_footprint(ncore, ncas, s.nvirt(), -1, lwork);
  for (const auto &pc : whole->pc_roots) {
    ASSERT_TRUE(pc.has_value());
    slack = std::max(slack, peak_gap(most, fb::nevpt2::pc_class_energies_footprint(
                                               ncore, ncas, s.nvirt(), -1, lwork, pc->dropped)));
  }
  std::vector<std::array<std::int64_t, 8>> dropped;
  for (std::int32_t j = 0; j < n; ++j) {
    dropped.push_back(whole->pc_qd_couplings[static_cast<std::size_t>((j * n) + j)].dropped);
  }
  const auto diagonal = [&](const std::span<const std::array<std::int64_t, 8>> d) {
    return fb::nevpt2::pc_effective_hamiltonian_footprint(ncore, ncas, s.nvirt(), n, true, 0, -1,
                                                          lwork, true, d);
  };
  const auto pair = [&](const std::span<const std::array<std::int64_t, 8>> d) {
    return fb::nevpt2::pc_effective_hamiltonian_footprint(ncore, ncas, s.nvirt(), n, false, 2, -1,
                                                          lwork, true, d);
  };
  slack = std::max({slack, peak_gap(diagonal({}), diagonal(dropped)),
                    diagonal({}).device.persistent - diagonal(dropped).device.persistent +
                        peak_gap(pair({}), pair(dropped))});

  std::optional<fb::Nevpt2Output> least;
  expect_least_footprint(
      fp, kNevpt2QdStepLive,
      [&](fb::DeviceResources &res) {
        auto out = step.run(s.nevpt2_view(), res);
        ASSERT_TRUE(out) << out.error().message;
        least.emplace(std::move(*out));
      },
      rank_slack(slack));
  ASSERT_TRUE(least.has_value());
  ASSERT_EQ(least->qd_energies.size(), whole->qd_energies.size());
  for (std::size_t k = 0; k < whole->qd_energies.size(); ++k) {
    EXPECT_NEAR(least->qd_energies[k], whole->qd_energies[k], 1e-10) << "SC-QD state " << k;
    EXPECT_NEAR(least->pc_qd_energies[k], whole->pc_qd_energies[k], 1e-10) << "PC-QD state " << k;
  }
}

// The Nevpt2Gradient step, @p variant, state-specific over the single-state
// CASSCF or with @p quasi_degenerate state 0 of the SA-2 one, at the least:
// the gradient's footprint and the Solution's upload, its device arrays the
// product's. A PC footprint counts its classes at the rank that holds the
// most: the slack is the library gradient footprint's difference at the
// run's ranks.
void expect_nevpt2_gradient_step(Nevpt2Reference &s, const fb::Nevpt2Variant variant,
                                 const bool quasi_degenerate, const std::size_t live) {
  auto step = planned<fb::Nevpt2GradientStep>(*s.plan.calc);
  step.variant = variant;
  step.quasi_degenerate = quasi_degenerate;
  step.state = 0;
  const fb::MemoryFootprint fp = step.footprint(s.dims());
  EXPECT_GT(fp.device.persistent, 0U);
  std::size_t slack = 0;
  if (variant == fb::Nevpt2Variant::Pc) {
    const fb::BasisSet &basis = s.plan.basis->basis;
    const fb::BasisSet &aux = s.fit().aux;
    const std::int32_t ncore = s.cas->ncore;
    const std::int32_t ncas = s.cas->ncas;
    const auto nelecas = s.cas->nelecas;
    const auto &lwork = s.dims().syevd_lwork;
    if (quasi_degenerate) {
      const auto at = [&](const std::span<const std::array<std::int64_t, 8>> d) {
        return fb::nevpt2::qd_pc_gradient_footprint(basis, aux, ncore, ncas, nelecas, 1, s.nroots(),
                                                    false, 0, lwork, 0, d);
      };
      slack = peak_gap(at({}), at(s.qd_dropped()));
    } else {
      fb::DeviceResources &in = s.plan.inputs;
      auto strings = fb::ci::DeviceStringSet::create(in, ncas, nelecas / 2).value();
      const auto e =
          fb::nevpt2::pc_energy(in, strings, strings, s.fit().naux, s.nmo(), s.nmo(),
                                s.fit().b.data(), s.solution->c.data(), s.plan.h->h.data(), ncore,
                                ncas, s.solution->roots.vectors.data(), 0);
      ASSERT_TRUE(e) << e.error().message;
      const std::array<std::array<std::int64_t, 8>, 1> dropped{e->dropped};
      const auto at = [&](const std::span<const std::array<std::int64_t, 8>> d) {
        return fb::nevpt2::pc_gradient_footprint(basis, aux, ncore, ncas, nelecas, 1, 0, lwork, 0,
                                                 d);
      };
      slack = peak_gap(at({}), at(dropped));
    }
  }
  expect_least_footprint(
      fp, live, [&](fb::DeviceResources &res) { expect_ran(step.run(s.reader_view(), res)); },
      slack > 0 ? std::optional{rank_slack(slack)} : std::nullopt);
}

// The gradient steps': the state-specific ones at df_gradient's pass beside
// the Lagrangian's arrays, the QD SC one there too, the QD PC one at its
// adjoint's walk.
constexpr std::size_t kNevpt2GradientLive = 75;
constexpr std::size_t kNevpt2QdGradientLive = 51;
constexpr std::size_t kNevpt2QdPcGradientLive = 188;

TEST_F(Nevpt2ReaderFootprint, Nevpt2GradientStep) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_nevpt2_gradient_step(ss(), fb::Nevpt2Variant::Sc, false, kNevpt2GradientLive);
}

TEST_F(Nevpt2ReaderFootprint, Nevpt2GradientStepPc) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_nevpt2_gradient_step(ss(), fb::Nevpt2Variant::Pc, false, kNevpt2GradientLive);
}

TEST_F(Nevpt2ReaderFootprint, Nevpt2GradientStepQuasiDegenerate) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_nevpt2_gradient_step(sa(), fb::Nevpt2Variant::Sc, true, kNevpt2QdGradientLive);
}

TEST_F(Nevpt2ReaderFootprint, Nevpt2GradientStepQuasiDegeneratePc) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_nevpt2_gradient_step(sa(), fb::Nevpt2Variant::Pc, true, kNevpt2QdPcGradientLive);
}

// The Nevpt2Nac step, @p variant, over the SA-2 CASSCF's one pair: the
// upload, H_eff, then the pair's coupling. PC's slack, as the gradient's:
// the step's two PC phases, H_eff's walk and the coupling, at the
// rank that holds the most and at the run's.
void expect_nevpt2_nac_step(Nevpt2Reference &s, const fb::Nevpt2Variant variant,
                            const std::size_t live) {
  auto step = planned<fb::Nevpt2NacStep>(*s.plan.calc);
  step.variant = variant;
  const fb::MemoryFootprint fp = step.footprint(s.dims());
  EXPECT_EQ(fp.device.persistent, 0U);
  std::size_t slack = 0;
  if (variant == fb::Nevpt2Variant::Pc) {
    const fb::BasisSet &basis = s.plan.basis->basis;
    const fb::BasisSet &aux = s.fit().aux;
    const std::int32_t ncore = s.cas->ncore;
    const std::int32_t ncas = s.cas->ncas;
    const auto nelecas = s.cas->nelecas;
    const std::int32_t n = s.nroots();
    const auto &lwork = s.dims().syevd_lwork;
    const std::vector<std::array<std::int64_t, 8>> dropped = s.qd_dropped();
    const auto phases = [&](const std::span<const std::array<std::int64_t, 8>> d) {
      const fb::MemoryFootprint heff = fb::nevpt2::qd_inputs_footprint(
          s.fit().naux, s.nmo(), ncore, ncas, nelecas, 1, n,
          drop(fb::nevpt2::pc_effective_hamiltonian_footprint(ncore, ncas, s.nvirt(), n, true,
                                                              n * (n - 1), 0, lwork, false, d)));
      const fb::MemoryFootprint pair = fb::nevpt2::qd_pc_nac_footprint(
          basis, aux, ncore, ncas, nelecas, 1, n, false, 0, lwork, d);
      return std::max(heff.device.peak(), pair.device.peak());
    };
    slack = phases({}) - phases(dropped);
  }
  expect_least_footprint(
      fp, live, [&](fb::DeviceResources &res) { expect_ran(step.run(s.reader_view(), res)); },
      slack > 0 ? std::optional{rank_slack(slack)} : std::nullopt);
}

// The coupling steps', at their pair's coupling's peak.
constexpr std::size_t kNevpt2NacLive = 51;
constexpr std::size_t kNevpt2PcNacLive = 188;

TEST_F(Nevpt2ReaderFootprint, Nevpt2NacStep) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_nevpt2_nac_step(sa(), fb::Nevpt2Variant::Sc, kNevpt2NacLive);
}

TEST_F(Nevpt2ReaderFootprint, Nevpt2NacStepPc) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_nevpt2_nac_step(sa(), fb::Nevpt2Variant::Pc, kNevpt2PcNacLive);
}

// The routines the NEVPT2 steps' footprints compose, each held alone at the
// run's PC ranks (so exactly, no slack): over the SA-2 CASSCF's
// qd_inputs, the effective Hamiltonians' diagonal walk with the kets kept
// and a pair's walk over them, SC's and PC's, and the PT2 transition
// densities over every pair.
TEST_F(Nevpt2ReaderFootprint, EffectiveHamiltonians) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  Nevpt2Reference &s = sa();
  const std::int32_t ncore = s.cas->ncore;
  const std::int32_t ncas = s.cas->ncas;
  const std::int32_t n = s.nroots();
  const auto &lwork = s.dims().syevd_lwork;
  const std::vector<std::array<std::int64_t, 8>> dropped = s.qd_dropped();
  const std::vector<double> t{0.0, 0.5, -0.5, 0.0};
  ASSERT_TRUE(fb::nevpt2::qd_inputs(
      s.plan.inputs, s.plan.molecule->molecule, s.plan.basis->basis, s.fit().aux, s.plan.h->h,
      s.fit().b, *s.solution, ncore, ncas,
      [&](const fb::nevpt2::MoBlocks &blocks,
          const std::span<const fb::nevpt2::EnergyInputs> pairs) -> fb::Status {
        // The diagonal alone, and the pair (0, 1) beside its transpose.
        std::vector<fb::nevpt2::EnergyInputs> diagonal(pairs.begin(), pairs.end());
        std::vector<fb::nevpt2::EnergyInputs> off(pairs.begin(), pairs.end());
        for (std::int32_t p = 0; p < n * n; ++p) {
          fb::nevpt2::EnergyInputs &omit = p / n == p % n ? off[p] : diagonal[p];
          omit.dm1 = omit.dm2 = omit.dm3 = omit.f3ca = omit.f3ac = nullptr;
        }
        fb::nevpt2::ScAmplitudes sc_kets;
        fb::nevpt2::PcAmplitudes pc_kets;
        // The integrals' identities, the classes' blocks, slabs and tables.
        expect_step_footprint(
            fb::nevpt2::sc_effective_hamiltonian_footprint(ncore, ncas, s.nvirt(), s.cas->nelecas,
                                                           n, true, 0, 0),
            kScDiagonalLive, [&](fb::DeviceResources &res) {
              EXPECT_TRUE(fb::nevpt2::sc_effective_hamiltonian(res, blocks, diagonal, 0, &sc_kets));
            });
        expect_step_footprint(
            fb::nevpt2::sc_effective_hamiltonian_footprint(ncore, ncas, s.nvirt(), s.cas->nelecas,
                                                           n, false, 2, 0),
            kScPairLive, [&](fb::DeviceResources &res) {
              EXPECT_TRUE(fb::nevpt2::sc_effective_hamiltonian(res, blocks, off, 0, &sc_kets));
            });
        // Each class's solves and kets, the kept U's the call's product.
        const fb::MemoryFootprint kept = fb::nevpt2::pc_effective_hamiltonian_footprint(
            ncore, ncas, s.nvirt(), n, true, 0, 0, lwork, true, dropped);
        EXPECT_GT(kept.device.persistent, 0U);
        expect_step_footprint(kept, kPcDiagonalLive, [&](fb::DeviceResources &res) {
          fb::nevpt2::PcAmplitudes local;
          EXPECT_TRUE(fb::nevpt2::pc_effective_hamiltonian(res, blocks, diagonal, 0, &local));
        });
        EXPECT_TRUE(
            fb::nevpt2::pc_effective_hamiltonian(s.plan.inputs, blocks, diagonal, 0, &pc_kets));
        expect_step_footprint(
            fb::nevpt2::pc_effective_hamiltonian_footprint(ncore, ncas, s.nvirt(), n, false, 2, 0,
                                                           lwork, true, dropped),
            kPcPairLive, [&](fb::DeviceResources &res) {
              EXPECT_TRUE(fb::nevpt2::pc_effective_hamiltonian(res, blocks, off, 0, &pc_kets));
            });
        // The walk over every pair, then the weighted RDMs, the blocks and a
        // class's coefficients (or amplitudes) and slabs; the density.
        expect_step_footprint(
            fb::nevpt2::sc_pt2_transition_density_footprint(ncore, ncas, s.nvirt(), s.cas->nelecas,
                                                            n, 0),
            kScDensityLive, [&](fb::DeviceResources &res) {
              EXPECT_TRUE(fb::nevpt2::sc_pt2_transition_density(res, blocks, pairs, t, 0));
            });
        expect_step_footprint(
            fb::nevpt2::pc_pt2_transition_density_footprint(ncore, ncas, s.nvirt(), n, 0, lwork,
                                                            dropped),
            kPcDensityLive, [&](fb::DeviceResources &res) {
              EXPECT_TRUE(fb::nevpt2::pc_pt2_transition_density(res, blocks, pairs, t, 0));
            });
        return {};
      }));
}

// The gradient's routines, each alone at its least (rdms' and rdms_adjoint's
// tiles at one determinant): over the single-state CASSCF, sc_adjoint,
// pc_adjoint and zc_adjoint; over the SA-2 one, qd_sc_adjoint and
// qd_pc_adjoint at a W.
TEST_F(Nevpt2ReaderFootprint, GradientRoutines) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  for (Nevpt2Reference *ref : {&ss(), &sa()}) {
    Nevpt2Reference &s = *ref;
    fb::DeviceResources &in = s.plan.inputs;
    const std::int32_t ncore = s.cas->ncore;
    const std::int32_t ncas = s.cas->ncas;
    const std::int32_t nmo = s.nmo();
    const std::int32_t naux = s.fit().naux;
    const auto &lwork = s.dims().syevd_lwork;
    const fb::mcscf::Solution &sol = *s.solution;
    const double *const b = s.fit().b.data();
    const double *const c = sol.c.data();
    const double *const h = s.plan.h->h.data();
    auto alpha = fb::ci::DeviceStringSet::create(in, ncas, s.cas->nelecas / 2).value();
    auto beta = fb::ci::DeviceStringSet::create(in, ncas, s.cas->nelecas / 2).value();
    if (s.nroots() == 1) {
      const auto e = fb::nevpt2::pc_energy(in, alpha, beta, naux, nmo, nmo, b, c, h, ncore, ncas,
                                           sol.roots.vectors.data(), 0);
      ASSERT_TRUE(e) << e.error().message;
      const std::array<std::array<std::int64_t, 8>, 1> dropped{e->dropped};
      // The forward's arrays and their bars, the walk, the builders' reverse.
      expect_least_footprint(fb::nevpt2::sc_adjoint_footprint(naux, nmo, nmo, ncore, ncas, 2, 2, 0),
                             kScAdjointLive, [&](fb::DeviceResources &res) {
                               EXPECT_TRUE(fb::nevpt2::sc_adjoint(res, alpha, beta, naux, nmo, nmo,
                                                                  b, c, h, ncore, ncas,
                                                                  sol.roots.vectors.data(), 0));
                             });
      expect_least_footprint(
          fb::nevpt2::pc_adjoint_footprint(naux, nmo, nmo, ncore, ncas, 2, 2, 0, lwork, 0, dropped),
          kPcAdjointLive, [&](fb::DeviceResources &res) {
            EXPECT_TRUE(fb::nevpt2::pc_adjoint(res, alpha, beta, naux, nmo, nmo, b, c, h, ncore,
                                               ncas, sol.roots.vectors.data(), 0));
          });
      const fb::nevpt2::ScAdjoint corr =
          fb::nevpt2::sc_adjoint(in, alpha, beta, naux, nmo, nmo, b, c, h, ncore, ncas,
                                 sol.roots.vectors.data(), 0)
              .value();
      const fb::nevpt2::ZcMultipliers zc =
          fb::nevpt2::zc_multipliers(in, nmo, ncore, ncas, corr.a.data(), corr.epsilon.data())
              .value();
      // The sigma's tables and apply scratch, D, h, B and Φ's arrays.
      expect_least_footprint(fb::nevpt2::zc_adjoint_footprint(naux, nmo, nmo, ncas, 2, 2, 1),
                             kZcAdjointLive, [&](fb::DeviceResources &res) {
                               EXPECT_TRUE(fb::nevpt2::zc_adjoint(
                                   res, alpha, beta, naux, nmo, nmo, b, c, h, ncore, ncas,
                                   std::array{1.0}, sol.roots.vectors.data(), zc.z.data()));
                             });
      continue;
    }
    const std::vector<double> w{0.6, 0.2, 0.2, 0.4};
    const std::vector<std::array<std::int64_t, 8>> dropped = s.qd_dropped();
    // The arrays and pair RDMs, the bars, the walk with its transitions, the
    // builders' reverse.
    expect_least_footprint(
        fb::nevpt2::qd_sc_adjoint_footprint(naux, nmo, nmo, ncore, ncas, 2, 2, 2, 0),
        kQdScAdjointLive, [&](fb::DeviceResources &res) {
          EXPECT_TRUE(fb::nevpt2::qd_sc_adjoint(res, alpha, beta, naux, nmo, nmo, b, c, h, ncore,
                                                ncas, sol.weights, sol.roots.vectors.data(), w, 0));
        });
    expect_least_footprint(
        fb::nevpt2::qd_pc_adjoint_footprint(naux, nmo, nmo, ncore, ncas, 2, 2, 2, 0, lwork, 0,
                                            dropped),
        kQdPcAdjointLive, [&](fb::DeviceResources &res) {
          EXPECT_TRUE(fb::nevpt2::qd_pc_adjoint(res, alpha, beta, naux, nmo, nmo, b, c, h, ncore,
                                                ncas, sol.weights, sol.roots.vectors.data(), w, 0));
        });
  }
}

// The couplings between the SA-2 CASSCF's two QD states, each alone at its
// least: the interstate coupling (the state form's Lagrangian at W^PQ, the
// core's Fock held, the passes at a mix summing to 0), SC's and PC's; and
// the derivative coupling qd_sc_nac and qd_pc_nac, the coupling, D^PT2 and
// the two combinations one at a time.
TEST_F(Nevpt2ReaderFootprint, Couplings) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  Nevpt2Reference &s = sa();
  const fb::BasisSet &basis = s.plan.basis->basis;
  const fb::BasisSet &aux = s.fit().aux;
  const std::int32_t ncore = s.cas->ncore;
  const std::int32_t ncas = s.cas->ncas;
  const std::int32_t nelecas = s.cas->nelecas;
  const std::int32_t n = s.nroots();
  const auto &lwork = s.dims().syevd_lwork;
  const std::vector<std::array<std::int64_t, 8>> dropped = s.qd_dropped();
  const auto call = [&](auto &&routine) {
    return [&, routine](fb::DeviceResources &res) {
      const auto out = routine(res);
      EXPECT_TRUE(out) << out.error().message;
    };
  };
  const fb::Molecule &molecule = s.plan.molecule->molecule;
  const auto &classes = s.plan.pairs->classes;
  const fb::SphericalMatrixAO &h = s.plan.h->h;
  const fb::DensityFitted &fit = s.fit();
  expect_least_footprint(
      fb::nevpt2::qd_sc_coupling_footprint(basis, aux, ncore, ncas, nelecas, 1, n, false, 0, lwork),
      kCouplingLive, call([&](fb::DeviceResources &res) {
        return fb::nevpt2::qd_sc_coupling(res, molecule, basis, classes, aux, h, fit.b,
                                          fit.l.data(), *s.solution, ncore, ncas, 0, 1, 0);
      }));
  expect_least_footprint(fb::nevpt2::qd_pc_coupling_footprint(basis, aux, ncore, ncas, nelecas, 1,
                                                              n, false, 0, lwork, dropped),
                         kPcCouplingLive, call([&](fb::DeviceResources &res) {
                           return fb::nevpt2::qd_pc_coupling(res, molecule, basis, classes, aux, h,
                                                             fit.b, fit.l.data(), *s.solution,
                                                             ncore, ncas, 0, 1, 0);
                         }));
  expect_least_footprint(
      fb::nevpt2::qd_sc_nac_footprint(basis, aux, ncore, ncas, nelecas, 1, n, false, 0, lwork),
      kCouplingLive, call([&](fb::DeviceResources &res) {
        return fb::nevpt2::qd_sc_nac(res, molecule, basis, classes, aux, h, fit.b, fit.l.data(),
                                     *s.solution, ncore, ncas, 0, 1, 0);
      }));
  expect_least_footprint(fb::nevpt2::qd_pc_nac_footprint(basis, aux, ncore, ncas, nelecas, 1, n,
                                                         false, 0, lwork, dropped),
                         kPcCouplingLive, call([&](fb::DeviceResources &res) {
                           return fb::nevpt2::qd_pc_nac(res, molecule, basis, classes, aux, h,
                                                        fit.b, fit.l.data(), *s.solution, ncore,
                                                        ncas, 0, 1, 0);
                         }));
}

} // namespace
