// Pre-flight over parsed configs: water's RHF chain in cc-pVDZ held to a
// max_device_mem_per_gpu below its footprint's peak, refused before the first
// step, and one above it, run with the pool inside the cap in workspaces
// sized at the plan's maximum; a capped RHF -> CASSCF the same; the pre-warm
// refusing a cap the card cannot supply; and a CASCI under AVAS narrowed to
// the active space the cap fits. Then the nested steps, an RHF [optimize] and
// a semi-numerical [hessian], over the RHF and over a CASSCF gradient, and a
// CASSCF [optimize], refused below their peak and run at it. Every cap is set
// from the plan's own fold, so the cases hold on any card. Uncapped,
// pre-flight sizes the workspaces alone.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.core.nontransferrable;
import free_breakfast.io.base;
import free_breakfast.pipeline.shape;

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

constexpr std::string_view kRhf = R"(
[molecule]
xyz = "water.xyz"
[basis_set]
basis = "cc-pvdz"
[scf]
)";

// The CASCI's AVAS labels: O's 2p and each H's 1s, 5 MINAO AOs, so a worst
// case of 10 orbitals.
constexpr std::string_view kAvasCasci = R"(
[molecule]
xyz = "water.xyz"
[basis_set]
basis = "cc-pvdz"
[casci]
avas = ["O 2p", "H 1s"]
)";

constexpr std::uint64_t kReserve = fb::DeviceResources::kDeviceReserve;

// @p tables parsed, the geometry written where they resolve it and the
// output directory made, as read_config makes it.
fb::Calculation parse(const fb::test::ScratchDir &dir, std::string_view tables) {
  EXPECT_TRUE(fb::write_file(dir / "water.xyz", kWaterXyz).has_value());
  std::filesystem::create_directories(dir / "out");
  auto calc = fb::parse_config(tables, dir.path(), dir / "out");
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  return std::move(*calc);
}

// @p calc's fold at @p dims.
fb::PlanFootprint fold(const fb::Calculation &calc, const fb::FootprintDimensions &dims) {
  std::vector<fb::MemoryFootprint> footprints;
  for (const fb::StepFootprint &step : calc.footprints(dims)) {
    footprints.push_back(step.footprint);
  }
  return fb::plan_footprint(footprints);
}

// The [memory] table capping one device at @p device bytes.
std::string memory(std::uint64_t device) {
  return std::format("[memory]\nmax_device_mem_per_gpu = {}\n", device);
}

// What the DeviceResources holds in its pool from construction, its info
// flag: pre-flight's room takes it off the cap, as no step counts it.
constexpr std::size_t kOwnPool = fb::DeviceResources::kOwnPoolBytes;

// The device cap whose room is @p peak exactly: the reserve, the headroom,
// the DeviceResources' own pool bytes, and on a pool that rounds reuse an
// eighth of the peak above it. One byte less is a room of peak - 1.
std::uint64_t cap_for(std::size_t peak, bool rounds_reuse) {
  const std::size_t rounding = rounds_reuse ? fb::math::idivup(peak, fbt::kPoolReuseDivisor) : 0;
  return kReserve + fb::kPoolHeadroomBytes + kOwnPool + peak + rounding;
}

// Below the peak: refused naming the step that reaches it, and no step runs.
TEST(Preflight, CapBelowThePeakIsRefusedBeforeTheFirstStep) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_preflight_below");
  fb::NodeResources node; // as main builds it: pre-flight sizes the workspaces
  const bool rounds_reuse = fbt::pool_rounds_reuse(node.device(0).dev_idx());
  fb::PlanFootprint plan;
  std::string peak_step;
  {
    const fb::Calculation uncapped = parse(dir, kRhf);
    const fb::PlanShape shape = fb::plan_shape(uncapped, node.device(0)).value();
    plan = fold(uncapped, shape.dims);
    peak_step = uncapped.footprints(shape.dims)[plan.device_peak_step].name;
  }
  const std::size_t peak = plan.footprint.device.peak();
  // A room of peak - 1 bytes: one byte short of the peak's cap.
  fb::Calculation calc = parse(dir, std::string(kRhf) + memory(cap_for(peak, rounds_reuse) - 1));
  const fb::Status checked = fb::preflight(calc, node);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().kind, fb::ErrorKind::InvalidConfig);
  const std::string &message = checked.error().message;
  EXPECT_NE(
      message.find(std::format("[{}] reaches the plan's device peak of {} bytes", peak_step, peak)),
      std::string::npos)
      << message;
  // main stops at the refusal: nothing ran, so no product was built.
  EXPECT_FALSE(calc.get<fb::MoleculeOutput>().has_value());
}

// @p tables under a device cap at twice its fold's peak and a pinned cap of
// 64 MiB, run after pre-flight: the devices carry the caps and workspaces at
// the plan's maximum, rounded up, held for the run, so nothing in it grows
// them (a borrow past them would be refused) and the run's largest borrows
// are within them; the pool stays inside its cap. @p Product is what the run
// builds last.
template <class Product>
void expect_capped_run_inside(std::string_view name, std::string_view tables) {
  const fb::test::ScratchDir dir(name);
  fb::NodeResources node; // as main builds it: pre-flight sizes the workspaces
  EXPECT_EQ(node.device(0).workspaces(), fb::MemoryFootprint{});
  fb::PlanFootprint plan;
  {
    const fb::Calculation uncapped = parse(dir, tables);
    plan = fold(uncapped, fb::plan_shape(uncapped, node.device(0)).value().dims);
  }
  const std::uint64_t device_cap = cap_for(2 * plan.footprint.device.peak(), true);
  constexpr std::size_t kPinnedCap = std::size_t{64} << 20;
  fb::Calculation calc = parse(dir, std::string(tables) + memory(device_cap) +
                                        std::format("max_pinned_mem_per_gpu = {}\n", kPinnedCap));
  const fb::Status checked = fb::preflight(calc, node);
  ASSERT_TRUE(checked.has_value()) << checked.error().message;

  fb::DeviceResources &res = node.device(0);
  const std::uint64_t pool_cap = device_cap - kReserve;
  EXPECT_EQ(res.pool_cap(), pool_cap);
  EXPECT_EQ(res.pinned_cap(), kPinnedCap);
  constexpr std::size_t kAlign = fb::DeviceResources::kWorkspaceAlignment;
  const fb::MemoryFootprint workspaces = res.workspaces();
  EXPECT_EQ(workspaces.device.workspace,
            fb::math::align_up(plan.footprint.device.workspace, kAlign));
  EXPECT_EQ(workspaces.pinned.workspace,
            fb::math::align_up(plan.footprint.pinned.workspace, kAlign));
  EXPECT_EQ(res.pinned_bytes(), workspaces.pinned.workspace);

  res.device()->reset_pool_high_water();
  const fb::Status ran = calc.run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  EXPECT_TRUE(calc.get<Product>().has_value());
  const fb::PoolUsage usage = res.device()->pool_usage();
  std::cout << std::format("[ preflight ] {}: workspaces {} and {} bytes, largest borrows {} and "
                           "{}, pool cap {}, UsedMemHigh {}\n",
                           name, workspaces.device.workspace, workspaces.pinned.workspace,
                           res.workspace_high_water().device.workspace,
                           res.workspace_high_water().pinned.workspace, pool_cap, usage.used_high);
  EXPECT_LE(usage.used_high, pool_cap);
  EXPECT_LE(usage.reserved_high, pool_cap);
  EXPECT_EQ(res.workspaces(), workspaces);
  EXPECT_LE(res.workspace_high_water().device.workspace, workspaces.device.workspace);
  EXPECT_LE(res.workspace_high_water().pinned.workspace, workspaces.pinned.workspace);
}

// Above the peak, with a pinned cap too: water's RHF.
TEST(Preflight, CapAboveThePeakRunsInsideIt) {
  FBF_REQUIRE_GPU();
  expect_capped_run_inside<fb::ScfOutput>("pipeline_preflight_above", kRhf);
}

// The fitted water's RHF -> CAS(4,4) in STO-3G, the same (#1123): the
// workspaces sized for the CASSCF from the first step.
constexpr std::string_view kCasscf = R"(
[molecule]
xyz = "water.xyz"
[basis_set]
basis = "sto-3g"
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"
[casscf]
ncas = 4
nelecas = 4
)";

TEST(Preflight, CappedCasscfRunsInsideIt) {
  FBF_REQUIRE_GPU();
  expect_capped_run_inside<fb::CasscfOutput>("pipeline_preflight_casscf", kCasscf);
}

// Uncapped, pre-flight checks nothing against a cap but still builds the
// devices with workspaces at the plan's maximum, uncapped, and the run borrows
// within them.
TEST(Preflight, UncappedSizesTheWorkspaces) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_preflight_uncapped");
  fb::NodeResources node; // as main builds it
  fb::PlanFootprint plan;
  {
    const fb::Calculation uncapped = parse(dir, kCasscf);
    plan = fold(uncapped, fb::plan_shape(uncapped, node.device(0)).value().dims);
  }
  fb::Calculation calc = parse(dir, kCasscf);
  const fb::Status checked = fb::preflight(calc, node);
  ASSERT_TRUE(checked.has_value()) << checked.error().message;
  fb::DeviceResources &res = node.device(0);
  EXPECT_EQ(res.pool_cap(), 0U);
  EXPECT_EQ(res.pinned_cap(), 0U);
  constexpr std::size_t kAlign = fb::DeviceResources::kWorkspaceAlignment;
  EXPECT_EQ(res.workspaces().device.workspace,
            fb::math::align_up(plan.footprint.device.workspace, kAlign));
  EXPECT_EQ(res.workspaces().pinned.workspace,
            fb::math::align_up(plan.footprint.pinned.workspace, kAlign));
  const fb::Status ran = calc.run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  EXPECT_TRUE(calc.get<fb::CasscfOutput>().has_value());
  EXPECT_LE(res.workspace_high_water().device.workspace, res.workspaces().device.workspace);
}

// Half the card held outside the capped pool, by a Device of its own: a cap
// of the whole card fits the plan, but the pre-warm finds the card cannot
// supply it, and says how much is free.
TEST(Preflight, PreWarmRefusesACapTheCardCannotSupply) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_preflight_prewarm");
  fb::NodeResources node; // as main builds it: pre-flight sizes the workspaces
  const std::int32_t dev_idx = node.device(0).dev_idx();
  const std::size_t total = node.device(0).device()->props().totalGlobalMem;
  const auto other = std::make_shared<fb::Device>(dev_idx);
  const fb::Buffers::device<std::byte> held(total / 2, other);
  ASSERT_EQ(wwr::extension::sync(other->stream()), wwr::wwrSuccess);

  fb::Calculation calc = parse(dir, std::string(kRhf) + memory(total));
  const fb::Status checked = fb::preflight(calc, node);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().kind, fb::ErrorKind::Allocation);
  const std::string &message = checked.error().message;
  EXPECT_NE(message.find("bytes free"), std::string::npos) << message;
  EXPECT_NE(message.find(std::format("the pool's cap of {} bytes", total - kReserve)),
            std::string::npos)
      << message;
}

// A CASCI under AVAS whose worst case, 10 orbitals, does not fit: the cap is
// set to the fold at a smaller space, so pre-flight narrows avas_max_ncas to
// it, and warns. The fold grows with the space; the case picks the largest
// space below 10 whose fold is below the next one's.
TEST(Preflight, AvasIsNarrowedToTheSpaceTheCapFits) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_preflight_avas");
  fb::NodeResources node; // as main builds it: pre-flight sizes the workspaces
  const bool rounds_reuse = fbt::pool_rounds_reuse(node.device(0).dev_idx());
  std::vector<std::size_t> peaks;
  {
    const fb::Calculation uncapped = parse(dir, kAvasCasci);
    // Held, not copied out of: the dimensions point at the shape's bases.
    fb::PlanShape shape = fb::plan_shape(uncapped, node.device(0)).value();
    fb::FootprintDimensions &dims = shape.dims;
    ASSERT_EQ(dims.ncas, 10U);
    peaks.push_back(0); // no space of 0 orbitals is picked
    for (std::size_t ncas = 1; ncas <= 10; ++ncas) {
      dims.ncas = ncas;
      dims.nelecas = ncas;
      peaks.push_back(fold(uncapped, dims).footprint.device.peak());
    }
  }
  ASSERT_TRUE(std::ranges::is_sorted(peaks));
  std::size_t fits = 9;
  while (fits > 0 && peaks[fits] == peaks[fits + 1]) {
    --fits;
  }
  ASSERT_GT(fits, 0U) << "the CASCI never reaches the plan's peak";

  fb::Calculation calc =
      parse(dir, std::string(kAvasCasci) + memory(cap_for(peaks[fits], rounds_reuse)));
  fb::log::reset_warning_count();
  const fb::Status checked = fb::preflight(calc, node);
  ASSERT_TRUE(checked.has_value()) << checked.error().message;
  EXPECT_EQ(calc.step<fb::CasciStep>()->avas_max_ncas, static_cast<std::int32_t>(fits));
  EXPECT_GE(fb::log::warning_count(), 1U);
}

// The nested steps (#1054): an RHF [optimize] and a semi-numerical [hessian]
// over water in STO-3G, integral-direct, each the plan's peak step. A cap
// one byte below the peak's room is refused naming the step before the first
// step runs; a cap whose room is the peak exactly (the cap less the
// reserve, the headroom and DeviceResources::kOwnPoolBytes) runs every
// geometry inside it, the pool's UsedMemHigh read over the whole run, and held to the peak
// itself too (beside a rounding pool's reuse bound; above DeviceResources'
// own pool bytes, its syevd flag, the one pool allocation no step makes),
// since water's peak
// is far under the cap's 64 MiB of headroom.
constexpr std::string_view kNestedOptimize = R"(
[molecule]
xyz = "water.xyz"
[basis_set]
basis = "sto-3g"
[optimize]
)";

constexpr std::string_view kNestedHessian = R"(
[molecule]
xyz = "water.xyz"
[basis_set]
basis = "sto-3g"
[hessian]
method = "numerical"
)";

// A semi-numerical [hessian] over a CASSCF gradient (#1204): a CAS(2,2) of
// the fitted water in STO-3G, its reference chain to [casscf] then 18
// displaced chains from the reference's carried orbitals.
constexpr std::string_view kNestedCasscfHessian = R"(
[molecule]
xyz = "water.xyz"
[basis_set]
basis = "sto-3g"
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"
[casscf]
ncas = 2
nelecas = 2
[casscf_gradient]
[hessian]
method = "numerical"
)";

// A QD [hessian] (#1209): an analytic SC-QD-NEVPT2 Hessian over the fitted
// water's CAS(6,7) in cc-pVDZ, 21 roots. Its ensemble's edge solves a CI over
// 22 roots, so syevd's work is counted through 16 * 22 = 352 of the CI's
// 630 packed singlet strings, past the CASSCF's 16 * 21 = 336 and PC's
// n_act^3 = 343: the least active space and roots whose edge reaches past
// both.
constexpr std::string_view kNestedQdHessian = R"(
[molecule]
xyz = "water.xyz"
[basis_set]
basis = "cc-pvdz"
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"
[casscf]
ncas = 7
nelecas = 6
nroots = 21
[nevpt2_gradient]
variant = "sc"
quasi_degenerate = true
state = 1
[hessian]
)";

// An [optimize] over a CASSCF gradient (#1212): a CAS(2,2) of the fitted
// water in STO-3G, the first chain from the SCF and every later one from the
// carried orbitals. (A followed SA-2 root's walk here stops in the CASSCF
// gradient's Z-vector, whose Hessian is not positive there; the followed
// forms are StepFootprint.Optimize*'s.)
constexpr std::string_view kNestedCasscfOptimize = R"(
[molecule]
xyz = "water.xyz"
[basis_set]
basis = "sto-3g"
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"
[casscf]
ncas = 2
nelecas = 2
[casscf_gradient]
[optimize]
)";

// @p tables' folded plan, and the step reaching its device peak, on @p node's
// device 0.
std::pair<fb::PlanFootprint, std::string>
nested_plan(const fb::test::ScratchDir &dir, fb::NodeResources &node, std::string_view tables) {
  const fb::Calculation uncapped = parse(dir, tables);
  const fb::PlanShape shape = fb::plan_shape(uncapped, node.device(0)).value();
  const fb::PlanFootprint plan = fold(uncapped, shape.dims);
  // The workspace, which differs most between the backends: syevd's own
  // query is the solver's to say.
  std::cout << std::format("[ preflight ] syevd's work at nao = {}: {} bytes, at 3 natoms = {}: "
                           "{} bytes\n",
                           shape.dims.nao, sizeof(double) * shape.dims.syevd_lwork[shape.dims.nao],
                           3 * shape.dims.natoms,
                           3 * shape.dims.natoms < shape.dims.syevd_lwork.size()
                               ? sizeof(double) * shape.dims.syevd_lwork[3 * shape.dims.natoms]
                               : 0);
  return {plan, std::string(uncapped.footprints(shape.dims)[plan.device_peak_step].name)};
}

void expect_nested_refused_below(std::string_view tables, std::string_view step) {
  const fb::test::ScratchDir dir("pipeline_preflight_nested_below");
  fb::NodeResources node; // as main builds it: pre-flight sizes the workspaces
  const bool rounds_reuse = fbt::pool_rounds_reuse(node.device(0).dev_idx());
  const auto [plan, peak_step] = nested_plan(dir, node, tables);
  const std::size_t peak = plan.footprint.device.peak();
  EXPECT_EQ(peak_step, step);
  // One byte past the cap the at-the-peak case runs under.
  fb::Calculation calc = parse(dir, std::string(tables) + memory(cap_for(peak, rounds_reuse) - 1));
  const fb::Status checked = fb::preflight(calc, node);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().kind, fb::ErrorKind::InvalidConfig);
  const std::string &message = checked.error().message;
  EXPECT_NE(
      message.find(std::format("[{}] reaches the plan's device peak of {} bytes", step, peak)),
      std::string::npos)
      << message;
  EXPECT_FALSE(calc.get<fb::MoleculeOutput>().has_value());
}

template <class Product>
void expect_nested_runs_at_the_peak(std::string_view tables) {
  const fb::test::ScratchDir dir("pipeline_preflight_nested_at");
  fb::NodeResources node; // as main builds it: pre-flight sizes the workspaces
  const bool rounds_reuse = fbt::pool_rounds_reuse(node.device(0).dev_idx());
  const fb::PlanFootprint plan = nested_plan(dir, node, tables).first;
  const std::size_t peak = plan.footprint.device.peak();
  const std::uint64_t device_cap = cap_for(peak, rounds_reuse);
  fb::Calculation calc = parse(dir, std::string(tables) + memory(device_cap));
  const fb::Status checked = fb::preflight(calc, node);
  ASSERT_TRUE(checked.has_value()) << checked.error().message;

  fb::DeviceResources &res = node.device(0);
  const std::uint64_t pool_cap = device_cap - kReserve;
  const std::size_t device_workspace = res.workspaces().device.workspace;
  EXPECT_EQ(device_workspace, plan.footprint.device.workspace);
  // Before the first step the pool holds the workspace and
  // DeviceResources' own pool bytes, its syevd flag made with it: nothing of
  // the plan's but the workspace, which its peak counts.
  const std::uint64_t held = res.device()->pool_usage().used_current;
  EXPECT_EQ(held, device_workspace + kOwnPool);
  res.device()->reset_pool_high_water();
  const fb::Status ran = calc.run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  EXPECT_TRUE(calc.get<Product>().has_value());
  const fb::PoolUsage usage = res.device()->pool_usage();
  std::cout << std::format("[ preflight ] peak {} bytes (workspace {}), pool cap {}, held before "
                           "the run {}, UsedMemHigh {}, ReservedMemHigh {}\n",
                           peak, plan.footprint.device.workspace, pool_cap, held, usage.used_high,
                           usage.reserved_high);
  EXPECT_LE(usage.used_high, pool_cap);
  EXPECT_LE(usage.reserved_high, pool_cap);
  // And the fold is sound over every geometry: what the run held above
  // DeviceResources' own pool bytes is at most the peak, beside the pool's reuse rounding where it
  // rounds (pool_footprint.h).
  EXPECT_LE(usage.used_high - kOwnPool,
            peak + (rounds_reuse ? fb::math::idivup(peak, fbt::kPoolReuseDivisor) : 0));
  // Every geometry's chain ran in the workspace pre-flight sized.
  EXPECT_LE(res.workspace_high_water().device.workspace, device_workspace);
}

TEST(Preflight, NestedOptimizeBelowThePeakIsRefusedBeforeTheFirstStep) {
  FBF_REQUIRE_GPU();
  expect_nested_refused_below(kNestedOptimize, "optimize");
}

TEST(Preflight, NestedOptimizeAtThePeakRunsInsideIt) {
  FBF_REQUIRE_GPU();
  expect_nested_runs_at_the_peak<fb::OptimizeOutput>(kNestedOptimize);
}

TEST(Preflight, NestedHessianBelowThePeakIsRefusedBeforeTheFirstStep) {
  FBF_REQUIRE_GPU();
  expect_nested_refused_below(kNestedHessian, "hessian");
}

TEST(Preflight, NestedHessianAtThePeakRunsInsideIt) {
  FBF_REQUIRE_GPU();
  expect_nested_runs_at_the_peak<fb::HessianOutput>(kNestedHessian);
}

// The semi-numerical CASSCF [hessian] (#1204), its displaced chains and the
// reference held to the fold. The run at the peak is 19 CAS(2,2) chains,
// about 1.0 s on the RTX 3080.
TEST(Preflight, NestedCasscfHessianBelowThePeakIsRefusedBeforeTheFirstStep) {
  FBF_REQUIRE_GPU();
  expect_nested_refused_below(kNestedCasscfHessian, "hessian");
}

TEST(Preflight, NestedCasscfHessianAtThePeakRunsInsideIt) {
  FBF_REQUIRE_GPU();
  expect_nested_runs_at_the_peak<fb::HessianOutput>(kNestedCasscfHessian);
}

// Pre-flight folds the QD [hessian]'s footprint, which counts its edge's CI
// solve, over a syevd table that reaches that solve's subspace: refused
// cleanly below the peak, where a table through the CASSCF's order alone
// aborted in the footprint (#1209). Nothing runs.
TEST(Preflight, NestedQdHessianBelowThePeakIsRefusedBeforeTheFirstStep) {
  FBF_REQUIRE_GPU();
  expect_nested_refused_below(kNestedQdHessian, "hessian");
}

// The CASSCF [optimize] (#1212), its evaluations held to the fold.
TEST(Preflight, NestedCasscfOptimizeBelowThePeakIsRefusedBeforeTheFirstStep) {
  FBF_REQUIRE_GPU();
  expect_nested_refused_below(kNestedCasscfOptimize, "optimize");
}

TEST(Preflight, NestedCasscfOptimizeAtThePeakRunsInsideIt) {
  FBF_REQUIRE_GPU();
  expect_nested_runs_at_the_peak<fb::OptimizeOutput>(kNestedCasscfOptimize);
}

} // namespace
