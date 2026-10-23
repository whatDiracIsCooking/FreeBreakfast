// Pipeline<Steps...> over made-up steps: the registry checks (static_asserts),
// planning, and running on a node with no devices. The device-shaped steps
// here are planned but never run, so nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.pipeline;

namespace {

namespace fb = free_breakfast;

struct A {
  std::int32_t value = 0;
};
struct B {
  std::int32_t value = 0;
};
struct C {
  std::int32_t value = 0;
};
struct D {
  std::int32_t value = 0;
};
struct E {
  std::int32_t value = 0;
};
struct F {
  std::size_t devices = 0;
};
struct P {};
struct Q {};

// A root step: a, or a failure.
struct RootStep {
  static constexpr std::string_view kName = "root";
  std::int32_t value = 1;
  bool fail = false;
  fb::Result<A> run(fb::View<> /*in*/) const {
    if (fail) {
      return fb::err_numerical("root failed");
    }
    return A{.value = value};
  }
};

struct DoubleStep {
  static constexpr std::string_view kName = "double";
  fb::Result<B> run(fb::View<A> in) const { return B{.value = 2 * in.get<A>().value}; }
};

struct SumStep {
  static constexpr std::string_view kName = "sum";
  fb::Result<C> run(fb::View<A, B> in) const {
    return C{.value = in.get<A>().value + in.get<B>().value};
  }
};

// Host-parallel inside: c + 1 + 2, from two subtasks.
struct SubflowStep {
  static constexpr std::string_view kName = "subflow";
  fb::Result<D> run(fb::View<C> in, fb::tf::Subflow &subflow) const {
    std::int32_t one = 0;
    std::int32_t two = 0;
    subflow.emplace([&one] { one = 1; });
    subflow.emplace([&two] { two = 2; });
    subflow.join(); // the subtasks hold references into this frame
    return D{.value = in.get<C>().value + one + two};
  }
};

struct DeviceStep {
  static constexpr std::string_view kName = "device";
  fb::Result<E> run(fb::View<A> in, fb::DeviceResources & /*res*/) const {
    return E{.value = in.get<A>().value};
  }
};

struct NodeStep {
  static constexpr std::string_view kName = "node";
  fb::Result<F> run(fb::View<A> /*in*/, fb::NodeResources &node) const {
    return F{.devices = node.device_count()};
  }
};

using Calc = fb::Pipeline<RootStep, DoubleStep, SumStep, SubflowStep, DeviceStep, NodeStep>;

// --- The registry -----------------------------------------------------------

template <class... Steps>
concept FormsPipeline = requires { typename fb::Pipeline<Steps...>; };

// A second producer of A, and a second step named "root".
struct OtherRootStep {
  static constexpr std::string_view kName = "other_root";
  fb::Result<A> run(fb::View<> /*in*/) const { return A{}; }
};
struct SameNameStep {
  static constexpr std::string_view kName = "root";
  fb::Result<B> run(fb::View<> /*in*/) const { return B{}; }
};

static_assert(FormsPipeline<>);
static_assert(FormsPipeline<RootStep, DoubleStep>);
static_assert(FormsPipeline<RootStep, DoubleStep, SumStep, SubflowStep, DeviceStep, NodeStep>);
// Nothing in the registry produces A.
static_assert(!FormsPipeline<DoubleStep>);
static_assert(!FormsPipeline<RootStep, OtherRootStep>);
static_assert(!FormsPipeline<RootStep, SameNameStep>);

template <class Pipe, class T>
concept CanGet = requires(const Pipe &pipe) { pipe.template get<T>(); };
static_assert(CanGet<Calc, A>);
static_assert(CanGet<Calc, F>);
static_assert(!CanGet<Calc, P>);
static_assert(std::same_as<decltype(std::declval<const Calc &>().get<A>()), const fb::Result<A> &>);

// Each reads the other's product: a registry the compiler accepts, and a
// cycle create refuses.
struct PStep {
  static constexpr std::string_view kName = "p";
  fb::Result<P> run(fb::View<Q> /*in*/) const { return P{}; }
};
struct QStep {
  static constexpr std::string_view kName = "q";
  fb::Result<Q> run(fb::View<P> /*in*/) const { return Q{}; }
};

// --- Planning and running ---------------------------------------------------

// The host half of a node: no devices, two workers.
fb::NodeResourcesConfig host_only() {
  return {.devices = std::vector<std::int32_t>{}, .num_threads = 2};
}

void expect_error(const fb::Status &status, fb::ErrorKind kind, std::string_view needle) {
  ASSERT_FALSE(status.has_value());
  EXPECT_EQ(status.error().kind, kind);
  EXPECT_NE(status.error().message.find(needle), std::string::npos) << status.error().message;
}

template <class T>
void expect_error(const fb::Result<T> &result, fb::ErrorKind kind, std::string_view needle) {
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().kind, kind);
  EXPECT_NE(result.error().message.find(needle), std::string::npos) << result.error().message;
}

TEST(Pipeline, RunsEachStepAfterItsProducers) {
  auto calc = Calc::create({RootStep{.value = 3}, DoubleStep{}, SumStep{}, SubflowStep{}});
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node(host_only());
  ASSERT_TRUE(calc->run(node).has_value());
  ASSERT_TRUE(calc->get<C>().has_value());
  EXPECT_EQ(calc->get<A>()->value, 3);
  EXPECT_EQ(calc->get<B>()->value, 6);
  EXPECT_EQ(calc->get<C>()->value, 9);
  EXPECT_EQ(calc->get<D>()->value, 12);
}

TEST(Pipeline, TheOrderStepsAreGivenInDoesNotMatter) {
  auto calc = Calc::create({SumStep{}, DoubleStep{}, RootStep{.value = 3}});
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node(host_only());
  ASSERT_TRUE(calc->run(node).has_value());
  EXPECT_EQ(calc->get<C>()->value, 9);
}

TEST(Pipeline, ANodeStepRunsWithoutADevice) {
  auto calc = Calc::create({RootStep{}, NodeStep{}});
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node(host_only());
  ASSERT_TRUE(calc->run(node).has_value());
  EXPECT_EQ(calc->get<F>()->devices, 0U);
}

TEST(Pipeline, MissingProducerIsInvalidConfigNamingBothSteps) {
  auto calc = Calc::create({SumStep{}, RootStep{}});
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(calc.error().message.find("[sum] reads the product of [double]"), std::string::npos)
      << calc.error().message;
}

TEST(Pipeline, AStepGivenTwiceIsInvalidConfig) {
  auto calc = Calc::create({RootStep{}, RootStep{}});
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(calc.error().message.find("[root]"), std::string::npos) << calc.error().message;
}

TEST(Pipeline, ACycleIsInvalidConfigNamingItsSteps) {
  auto calc = fb::Pipeline<PStep, QStep>::create({PStep{}, QStep{}});
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(calc.error().message.find("[p], [q]"), std::string::npos) << calc.error().message;
}

TEST(Pipeline, InsertFillsInEveryMissingProducer) {
  std::vector<std::string_view> inserted;
  auto calc = Calc::create({SumStep{}}, [&]<class S>() -> fb::Result<S> {
    inserted.push_back(S::kName);
    if constexpr (std::same_as<S, RootStep>) {
      return RootStep{.value = 5};
    } else {
      return S{};
    }
  });
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  // Registry order, whatever order they were found missing in.
  EXPECT_EQ(inserted, (std::vector<std::string_view>{"root", "double"}));
  ASSERT_TRUE(calc->step<RootStep>().has_value());
  EXPECT_EQ(calc->step<RootStep>()->value, 5);
  EXPECT_FALSE(calc->step<SubflowStep>().has_value());
  fb::NodeResources node(host_only());
  ASSERT_TRUE(calc->run(node).has_value());
  EXPECT_EQ(calc->get<C>()->value, 15);
}

TEST(Pipeline, AnInsertErrorIsReturnedAsItIs) {
  auto calc = Calc::create({DoubleStep{}}, []<class S>() -> fb::Result<S> {
    return fb::err_io("cannot make [{}]", S::kName);
  });
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::IO);
  EXPECT_EQ(calc.error().message, "cannot make [root]");
}

TEST(Pipeline, AFailedStepFailsItsConsumersAndRunReportsIt) {
  auto calc = Calc::create({RootStep{.fail = true}, DoubleStep{}, SumStep{}});
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node(host_only());
  expect_error(calc->run(node), fb::ErrorKind::Numerical, "root failed");
  // Downstream steps did not run; each holds the failed input's error.
  expect_error(calc->get<B>(), fb::ErrorKind::Numerical, "root failed");
  expect_error(calc->get<C>(), fb::ErrorKind::Numerical, "root failed");
}

TEST(Pipeline, AnUnplannedOrUnrunProductIsInvalidConfig) {
  auto calc = Calc::create({RootStep{}});
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  expect_error(calc->get<A>(), fb::ErrorKind::InvalidConfig, "[root] has not run");
  fb::NodeResources node(host_only());
  ASSERT_TRUE(calc->run(node).has_value());
  EXPECT_TRUE(calc->get<A>().has_value());
  expect_error(calc->get<B>(), fb::ErrorKind::InvalidConfig, "[double] has not run");
}

TEST(Pipeline, ASecondRunIsUnsupported) {
  auto calc = Calc::create({RootStep{}});
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node(host_only());
  ASSERT_TRUE(calc->run(node).has_value());
  expect_error(calc->run(node), fb::ErrorKind::Unsupported, "runs once");
}

TEST(Pipeline, ADeviceStepWithNoDeviceFailsBeforeAnyStepRuns) {
  auto calc = Calc::create({RootStep{}, DeviceStep{}});
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node(host_only());
  expect_error(calc->run(node), fb::ErrorKind::GpuRuntime, "[device] needs a GPU device");
  expect_error(calc->get<A>(), fb::ErrorKind::InvalidConfig, "has not run");
}

} // namespace
