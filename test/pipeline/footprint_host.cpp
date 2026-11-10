// The plan fold over made-up steps: each step's footprint is read from its
// member at the plan's dimensions and its own options, the plan is
// folded in plan order, and the peak and the step reaching it are held to the
// values worked out beside each case, exactly. No step runs, so nothing
// touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.types;
import free_breakfast.pipeline.footprint;

namespace {

namespace fb = free_breakfast;

// A setup step: a product of nao^2 doubles, built with two more as scratch,
// and a little pinned staging.
struct SetupStep {
  static constexpr std::string_view kName = "setup";
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &dims) const {
    return {.device = {.persistent = dims.nao * dims.nao * 8,
                       .scratch = 2 * dims.nao * dims.nao * 8,
                       .workspace = 0},
            .pinned = {.persistent = 0, .scratch = 0, .workspace = 64}};
  }
};

// The big one: a naux x nao^2 tensor kept, with a scratch of nroots of them
// (an option) and a solver's workspace.
struct TensorStep {
  static constexpr std::string_view kName = "tensor";
  std::size_t nroots = 2;
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &dims) const {
    const std::size_t tensor = dims.naux * dims.nao * dims.nao * 8;
    return {.device = {.persistent = tensor, .scratch = nroots * tensor, .workspace = 4096},
            .pinned = {.persistent = 0, .scratch = 0, .workspace = 0}};
  }
};

// A later step: a small product, its scratch on the pinned side (a bounce
// buffer of naux doubles per atom) above anything the others take there.
struct OffloadStep {
  static constexpr std::string_view kName = "offload";
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &dims) const noexcept {
    return {.device = {.persistent = dims.nao * 8, .scratch = 0, .workspace = 1024},
            .pinned = {.persistent = dims.naux * 8,
                       .scratch = dims.natoms * dims.naux * 8,
                       .workspace = 0}};
  }
};

// A host fake.
struct HostStep {
  static constexpr std::string_view kName = "host";
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &) const { return {}; }
};

const fb::FootprintDimensions kDims{.nao = 10, .naux = 30, .natoms = 3};

TEST(PipelineFootprint, FoldMatchesHandComputedPeak) {
  const std::array<std::string_view, 4> names{SetupStep::kName, TensorStep::kName,
                                              OffloadStep::kName, HostStep::kName};
  const std::array<fb::MemoryFootprint, 4> plan{
      SetupStep{}.footprint(kDims), TensorStep{.nroots = 3}.footprint(kDims),
      OffloadStep{}.footprint(kDims), HostStep{}.footprint(kDims)};
  const fb::PlanFootprint folded = fb::plan_footprint(plan);

  // Device, persistent: setup 800, tensor 24000, offload 80, host 0.
  // Held at each step, without the workspace:
  //   setup   800 + 2*10*10*8 = 2400
  //   tensor  800 + 24000 + 3*24000 = 96800
  //   offload 800 + 24000 + 80 = 24880
  //   host    24880
  // plus the plan's maximum workspace, 4096: 100896, at the tensor step.
  EXPECT_EQ(folded.footprint.device.peak(), std::size_t{96800 + 4096});
  EXPECT_EQ(names[folded.device_peak_step], "tensor");
  EXPECT_EQ(folded.footprint.device.persistent, std::size_t{24880});
  EXPECT_EQ(folded.footprint.device.workspace, std::size_t{4096});

  // Pinned, persistent: only offload's 240. Held: 0, 0, 240 + 3*30*8 = 960,
  // 240; plus the setup's workspace, 64: 1024, at the offload step.
  EXPECT_EQ(folded.footprint.pinned.peak(), std::size_t{960 + 64});
  EXPECT_EQ(names[folded.pinned_peak_step], "offload");
}

// Two steps holding the same at their own peaks: the first one is named.
TEST(PipelineFootprint, FoldNamesTheFirstStepToReachThePeak) {
  const std::array<fb::MemoryFootprint, 3> plan{
      fb::MemoryFootprint{.device = {.persistent = 0, .scratch = 50, .workspace = 0}},
      fb::MemoryFootprint{.device = {.persistent = 10, .scratch = 40, .workspace = 0}},
      fb::MemoryFootprint{.device = {.persistent = 0, .scratch = 40, .workspace = 0}}};
  const fb::PlanFootprint folded = fb::plan_footprint(plan);
  EXPECT_EQ(folded.footprint.device.peak(), std::size_t{50});
  EXPECT_EQ(folded.device_peak_step, std::size_t{0});
}

TEST(PipelineFootprint, EmptyPlanHoldsNothing) {
  const fb::PlanFootprint folded = fb::plan_footprint({});
  EXPECT_EQ(folded.footprint, fb::MemoryFootprint{});
  EXPECT_EQ(folded.device_peak_step, std::size_t{0});
  EXPECT_EQ(folded.pinned_peak_step, std::size_t{0});
}

} // namespace
