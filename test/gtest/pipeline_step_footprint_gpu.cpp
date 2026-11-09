// The step footprints of the RHF chain (#1140), of CASCI and FCI (#1049) and
// of CASSCF (#1151), and mcscf::solve's own, each held to the pool peak of one
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
// CASSCF's loop and step, integral-direct and density-fitted.

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

} // namespace
