// Pre-flight over made-up steps: a Pipeline's footprints, listed in plan
// order, folded and held to the [memory] caps by check_plan, and the node's
// RAM held to the pinned cap by check_pinned_node. Each peak is held to the
// value worked out beside it, exactly, and each cap one byte below it is
// refused naming the step. The device steps are planned, never run, so
// nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.pipeline;

namespace {

namespace fb = free_breakfast;

struct A {};
struct B {};
struct C {};
struct D {};

// The root: a small product, built with a little scratch, and pinned
// staging. Loadable, with the same footprint on its load.
struct RootStep {
  static constexpr std::string_view kName = "root";
  fb::Result<A> run(fb::View<> /*in*/, fb::DeviceResources & /*res*/) const { return A{}; }
  static fb::Result<A> load(const std::filesystem::path & /*from*/, fb::View<> /*in*/) {
    return A{};
  }
  fb::MemoryFootprint footprint(const fb::FootprintDimensions & /*dims*/) const {
    return {.device = {.persistent = 100, .scratch = 50, .workspace = 0},
            .pinned = {.persistent = 0, .scratch = 0, .workspace = 16}};
  }
};

// A big product with a solver's workspace, and the pinned peak: a kept
// pinned product and a bounce buffer.
struct BigStep {
  static constexpr std::string_view kName = "big";
  fb::Result<B> run(fb::View<A> /*in*/, fb::DeviceResources & /*res*/) const { return B{}; }
  fb::MemoryFootprint footprint(const fb::FootprintDimensions & /*dims*/) const {
    return {.device = {.persistent = 1000, .scratch = 4000, .workspace = 256},
            .pinned = {.persistent = 10, .scratch = 200, .workspace = 0}};
  }
};

// Adaptive: its scratch is the least it runs with, and one more unit of its
// adaptive part is 300 bytes. The device peak.
struct AdaptiveStep {
  static constexpr std::string_view kName = "adaptive";
  fb::Result<C> run(fb::View<B> /*in*/, fb::DeviceResources & /*res*/) const { return C{}; }
  fb::MemoryFootprint footprint(const fb::FootprintDimensions & /*dims*/) const {
    return {.device = {.persistent = 200, .scratch = 5000, .workspace = 64},
            .pinned = {.persistent = 0, .scratch = 40, .workspace = 32}};
  }
  std::size_t adaptive_unit(const fb::FootprintDimensions & /*dims*/) const { return 300; }
};

// A host fake: it holds nothing of either kind.
struct HostStep {
  static constexpr std::string_view kName = "host";
  fb::Result<D> run(fb::View<A> /*in*/) const { return D{}; }
  fb::MemoryFootprint footprint(const fb::FootprintDimensions & /*dims*/) const { return {}; }
};

using Calc = fb::Pipeline<RootStep, BigStep, AdaptiveStep, HostStep>;

const fb::FootprintDimensions kDims{};

// root, big, adaptive and host, in that plan order.
std::vector<fb::StepFootprint> chain() {
  auto calc = Calc::create({RootStep{}, BigStep{}, AdaptiveStep{}, HostStep{}});
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  return calc->footprints(kDims);
}

// Device: persistent root 100, big 1000, adaptive 200, host 0; the plan's
// workspace max(0, 256, 64, 0) = 256. Held at each step, persistent up to it
// plus its scratch plus that workspace:
//   root      100 + 50 + 256                 =  406
//   big       1100 + 4000 + 256              = 5356
//   adaptive  1300 + 5000 + 256              = 6556   the device peak
//   host      1300 + 0 + 256                 = 1556
// Pinned: persistent big 10; workspace max(16, 0, 32, 0) = 32:
//   root      0 + 0 + 32 = 32
//   big       10 + 200 + 32 = 242                     the pinned peak
//   adaptive  10 + 40 + 32 = 82
//   host      10 + 0 + 32 = 42
constexpr std::size_t kDevicePeak = 6556;
constexpr std::size_t kPinnedPeak = 242;

TEST(Preflight, FootprintsListThePlanInOrder) {
  const std::vector<fb::StepFootprint> steps = chain();
  ASSERT_EQ(steps.size(), 4U);
  EXPECT_EQ(steps[0].name, "root");
  EXPECT_EQ(steps[1].name, "big");
  EXPECT_EQ(steps[2].name, "adaptive");
  EXPECT_EQ(steps[3].name, "host");
  EXPECT_EQ(steps[1].footprint, BigStep{}.footprint(kDims));
  EXPECT_FALSE(steps[1].adaptive_unit.has_value());
  EXPECT_EQ(steps[2].adaptive_unit, std::optional<std::size_t>{300});
}

TEST(Preflight, CapAtThePeakPasses) {
  const auto plan = fb::check_plan(chain(), {.max_device_mem_per_gpu = 1 << 30,
                                             .device_room = kDevicePeak,
                                             .max_pinned_mem_per_gpu = kPinnedPeak});
  ASSERT_TRUE(plan.has_value()) << plan.error().message;
  EXPECT_EQ(plan->footprint.device.peak(), kDevicePeak);
  EXPECT_EQ(plan->device_peak_step, 2U);
  EXPECT_EQ(plan->footprint.device.workspace, 256U);
  EXPECT_EQ(plan->footprint.pinned.peak(), kPinnedPeak);
  EXPECT_EQ(plan->pinned_peak_step, 1U);
  EXPECT_EQ(plan->footprint.pinned.workspace, 32U);
}

TEST(Preflight, DeviceCapOneByteBelowThePeakNamesTheStep) {
  const auto plan = fb::check_plan(chain(), {.max_device_mem_per_gpu = 1 << 30,
                                             .device_room = kDevicePeak - 1,
                                             .max_pinned_mem_per_gpu = kPinnedPeak});
  ASSERT_FALSE(plan.has_value());
  EXPECT_EQ(plan.error().kind, fb::ErrorKind::InvalidConfig);
  const std::string &message = plan.error().message;
  EXPECT_NE(message.find("[adaptive]"), std::string::npos) << message;
  EXPECT_NE(message.find("6556 bytes"), std::string::npos) << message;
  EXPECT_NE(message.find("6555 bytes"), std::string::npos) << message;
  EXPECT_NE(message.find("max_device_mem_per_gpu = 1073741824"), std::string::npos) << message;
}

TEST(Preflight, PinnedCapOneByteBelowThePeakNamesTheStep) {
  const auto plan = fb::check_plan(chain(), {.max_pinned_mem_per_gpu = kPinnedPeak - 1});
  ASSERT_FALSE(plan.has_value());
  EXPECT_EQ(plan.error().kind, fb::ErrorKind::InvalidConfig);
  const std::string &message = plan.error().message;
  EXPECT_NE(message.find("[big]"), std::string::npos) << message;
  EXPECT_NE(message.find("242 bytes"), std::string::npos) << message;
  EXPECT_NE(message.find("max_pinned_mem_per_gpu = 241"), std::string::npos) << message;
}

// At the device peak the adaptive step, which reaches it, has nothing above
// its least: it fits, but at its narrowest, and that is a warning. With one
// more unit of room it is not.
TEST(Preflight, AdaptiveStepAtItsLeastWarns) {
  fb::log::reset_warning_count();
  ASSERT_TRUE(fb::check_plan(chain(), {.device_room = kDevicePeak}).has_value());
  EXPECT_EQ(fb::log::warning_count(), 1U);

  fb::log::reset_warning_count();
  ASSERT_TRUE(fb::check_plan(chain(), {.device_room = kDevicePeak + 300}).has_value());
  EXPECT_EQ(fb::log::warning_count(), 0U);
}

// A loaded product is as resident as a computed one: the loaded step reports
// its footprint, from a default step.
TEST(Preflight, LoadedStepReportsItsFootprint) {
  auto calc = Calc::create({fb::Load<RootStep>{"a"}, BigStep{}});
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const std::vector<fb::StepFootprint> steps = calc->footprints(kDims);
  ASSERT_EQ(steps.size(), 2U);
  EXPECT_EQ(steps[0].name, "root");
  EXPECT_EQ(steps[0].footprint, RootStep{}.footprint(kDims));
}

// Two devices of 100 pinned bytes each fit 200 bytes of MemAvailable, and not
// 199; the refusal names both.
TEST(Preflight, PinnedCapTimesDevicesMustFitMemAvailable) {
  EXPECT_TRUE(fb::check_pinned_node(100, 2, 200).has_value());
  const fb::Status refused = fb::check_pinned_node(100, 2, 199);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
  const std::string &message = refused.error().message;
  EXPECT_NE(message.find("2 devices"), std::string::npos) << message;
  EXPECT_NE(message.find("max_pinned_mem_per_gpu = 100 bytes"), std::string::npos) << message;
  EXPECT_NE(message.find("MemAvailable of 199 bytes"), std::string::npos) << message;
  EXPECT_TRUE(fb::check_pinned_node(100, 0, 0).has_value());
}

} // namespace
