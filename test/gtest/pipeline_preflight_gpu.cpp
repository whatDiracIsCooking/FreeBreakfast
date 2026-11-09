// Pre-flight over parsed configs: water's RHF chain in cc-pVDZ held to a
// max_device_mem_per_gpu below its footprint's peak, refused before the first
// step, and one above it, run with the pool inside the cap and the
// workspaces never grown past the pre-grow; the pre-warm refusing a cap the
// card cannot supply; and a CASCI under AVAS narrowed to the active space
// the cap fits. Every cap is set from the plan's own fold, so the cases hold
// on any card.

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
    EXPECT_TRUE(step.footprint.has_value()) << step.name;
    footprints.push_back(step.footprint.value_or(fb::MemoryFootprint{}));
  }
  return fb::plan_footprint(footprints);
}

// The [memory] table capping one device at @p device bytes.
std::string memory(std::uint64_t device) {
  return std::format("[memory]\nmax_device_mem_per_gpu = {}\n", device);
}

// The device cap whose room is @p peak: the reserve, the headroom, and on a
// pool that rounds reuse an eighth of the peak above it.
std::uint64_t cap_for(std::size_t peak, bool rounds_reuse) {
  const std::size_t rounding = rounds_reuse ? fb::math::idivup(peak, fbt::kPoolReuseDivisor) : 0;
  return kReserve + fb::kPoolHeadroomBytes + peak + rounding;
}

// Below the peak: refused naming the step that reaches it, and no step runs.
TEST(Preflight, CapBelowThePeakIsRefusedBeforeTheFirstStep) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_preflight_below");
  fb::NodeResources node;
  fb::PlanFootprint plan;
  std::string peak_step;
  {
    const fb::Calculation uncapped = parse(dir, kRhf);
    const fb::PlanShape shape = fb::plan_shape(uncapped, node.device(0)).value();
    plan = fold(uncapped, shape.dims);
    peak_step = uncapped.footprints(shape.dims)[plan.device_peak_step].name;
  }
  const std::size_t peak = plan.footprint.device.peak();
  // A room of peak - 1 bytes, and less where the pool rounds reuse.
  fb::Calculation calc =
      parse(dir, std::string(kRhf) + memory(kReserve + fb::kPoolHeadroomBytes + peak - 1));
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

// Above the peak, with a pinned cap too: the devices carry the caps, the
// workspaces are pre-grown to the plan's maximum and never grow in the run,
// and the pool stays inside its cap.
TEST(Preflight, CapAboveThePeakRunsInsideIt) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_preflight_above");
  fb::NodeResources node;
  fb::PlanFootprint plan;
  {
    const fb::Calculation uncapped = parse(dir, kRhf);
    plan = fold(uncapped, fb::plan_shape(uncapped, node.device(0)).value().dims);
  }
  const std::uint64_t device_cap = cap_for(2 * plan.footprint.device.peak(), true);
  constexpr std::size_t kPinnedCap = std::size_t{64} << 20;
  fb::Calculation calc = parse(dir, std::string(kRhf) + memory(device_cap) +
                                        std::format("max_pinned_mem_per_gpu = {}\n", kPinnedCap));
  const fb::Status checked = fb::preflight(calc, node);
  ASSERT_TRUE(checked.has_value()) << checked.error().message;

  fb::DeviceResources &res = node.device(0);
  const std::uint64_t pool_cap = device_cap - kReserve;
  EXPECT_EQ(res.pool_cap(), pool_cap);
  EXPECT_EQ(res.pinned_cap(), kPinnedCap);
  const std::size_t device_workspace = res.device_workspace(0).num_elements();
  const std::size_t pinned_workspace = res.pinned_workspace("test", 0).value().size();
  EXPECT_GE(device_workspace, plan.footprint.device.workspace);
  EXPECT_GE(pinned_workspace, plan.footprint.pinned.workspace);

  res.device()->reset_pool_high_water();
  const fb::Status ran = calc.run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  EXPECT_TRUE(calc.get<fb::ScfOutput>().has_value());
  const fb::PoolUsage usage = res.device()->pool_usage();
  EXPECT_LE(usage.used_high, pool_cap);
  EXPECT_LE(usage.reserved_high, pool_cap);
  EXPECT_EQ(res.device_workspace(0).num_elements(), device_workspace);
  EXPECT_EQ(res.pinned_workspace("test", 0).value().size(), pinned_workspace);
}

// Half the card held outside the capped pool, by a Device of its own: a cap
// of the whole card fits the plan, but the pre-warm finds the card cannot
// supply it, and says how much is free.
TEST(Preflight, PreWarmRefusesACapTheCardCannotSupply) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_preflight_prewarm");
  fb::NodeResources node;
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
  fb::NodeResources node;
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

} // namespace
