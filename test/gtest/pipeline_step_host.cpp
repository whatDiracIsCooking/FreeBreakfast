// The pipeline step contract over made-up steps. Almost everything here is a
// static_assert, verified by compiling this file; only the last cases run.
// Device-shaped steps name the DeviceResources and NodeResources types and
// never construct one, so nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;

namespace {

namespace fb = free_breakfast;

struct A {
  std::int32_t value = 0;
};
struct B {
  double value = 0.0;
};
// Products own device buffers, so a move-only product must be accepted.
struct MoveOnly {
  std::unique_ptr<std::int32_t> value;
};

// Templated on their arguments, so an ill-formed one evaluates to false
// instead of failing the file.
template <class... Ts>
concept FormsView = requires { typename fb::View<Ts...>; };
template <class V, class T>
concept CanGet = requires(const V &view) { view.template get<T>(); };

// --- View -------------------------------------------------------------------

static_assert(FormsView<>);
static_assert(FormsView<A, B>);
static_assert(FormsView<MoveOnly>);
static_assert(!FormsView<A, A>);
static_assert(!FormsView<A, B, A>);
static_assert(!FormsView<const A>);
static_assert(!FormsView<A &>);
static_assert(!FormsView<void>);

static_assert(CanGet<fb::View<A>, A>);
static_assert(!CanGet<fb::View<A>, B>);
static_assert(CanGet<fb::View<A, B>, B>);
static_assert(!CanGet<fb::View<>, A>);
// Read-only: get hands out a const reference.
static_assert(std::same_as<decltype(std::declval<const fb::View<A> &>().get<A>()), const A &>);

// --- One well-formed step of each shape -------------------------------------

struct HostStep {
  static constexpr std::string_view kName = "host";

  double scale = 2.0;

  fb::Result<B> run(fb::View<A> in) const { return B{.value = scale * in.get<A>().value}; }
};

struct HostSubflowStep {
  static constexpr std::string_view kName = "host_subflow";
  fb::Result<B> run(fb::View<A> in, fb::tf::Subflow &subflow) const;
};

struct DeviceStep {
  static constexpr std::string_view kName = "device";
  fb::Result<MoveOnly> run(fb::View<A, B> in, fb::DeviceResources &res) const;
};

struct DeviceSubflowStep {
  static constexpr std::string_view kName = "device_subflow";
  fb::Result<B> run(fb::View<A> in, fb::DeviceResources &res, fb::tf::Subflow &subflow) const;
};

struct NodeStep {
  static constexpr std::string_view kName = "node";
  fb::Result<B> run(fb::View<A> in, fb::NodeResources &node) const;
};

// A root step reads nothing; its options are its inputs.
struct RootStep {
  static constexpr std::string_view kName = "root";

  std::int32_t value = 7;

  fb::Result<A> run(fb::View<> /*in*/) const { return A{.value = value}; }
};

// noexcept and a view taken by const reference read the same.
struct NoexceptByReferenceStep {
  static constexpr std::string_view kName = "noexcept_by_reference";
  fb::Result<B> run(const fb::View<A> &in) const noexcept;
};

static_assert(fb::PipelineStep<HostStep>);
static_assert(fb::PipelineStep<HostSubflowStep>);
static_assert(fb::PipelineStep<DeviceStep>);
static_assert(fb::PipelineStep<DeviceSubflowStep>);
static_assert(fb::PipelineStep<NodeStep>);
static_assert(fb::PipelineStep<RootStep>);
static_assert(fb::PipelineStep<NoexceptByReferenceStep>);

static_assert(fb::kRunShape<HostStep> == fb::RunShape::Host);
static_assert(fb::kRunShape<HostSubflowStep> == fb::RunShape::HostSubflow);
static_assert(fb::kRunShape<DeviceStep> == fb::RunShape::Device);
static_assert(fb::kRunShape<DeviceSubflowStep> == fb::RunShape::DeviceSubflow);
static_assert(fb::kRunShape<NodeStep> == fb::RunShape::Node);
static_assert(fb::kRunShape<RootStep> == fb::RunShape::Host);

// What it reads and writes, read off run.
static_assert(std::same_as<fb::StepInput<DeviceStep>, fb::View<A, B>>);
static_assert(std::same_as<fb::StepOutput<DeviceStep>, MoveOnly>);
static_assert(std::same_as<fb::StepInput<RootStep>, fb::View<>>);
static_assert(std::same_as<fb::StepInput<NoexceptByReferenceStep>, fb::View<A>>);

// --- Rejected steps ---------------------------------------------------------

struct NoName {
  fb::Result<B> run(fb::View<A> in) const;
};
static_assert(!fb::PipelineStep<NoName>);

struct EmptyName {
  static constexpr std::string_view kName = "";
  fb::Result<B> run(fb::View<A> in) const;
};
static_assert(!fb::PipelineStep<EmptyName>);

// A name that is not a constant cannot be checked unique at compile time.
struct RuntimeName {
  static inline std::string_view kName = "runtime";
  fb::Result<B> run(fb::View<A> in) const;
};
static_assert(!fb::PipelineStep<RuntimeName>);

struct NoRun {
  static constexpr std::string_view kName = "no_run";
};
static_assert(!fb::PipelineStep<NoRun>);

struct InputNotAView {
  static constexpr std::string_view kName = "input_not_a_view";
  fb::Result<B> run(std::tuple<A> in) const;
};
static_assert(!fb::PipelineStep<InputNotAView>);

struct ReadsOwnOutput {
  static constexpr std::string_view kName = "reads_own_output";
  fb::Result<B> run(fb::View<A, B> in) const;
};
static_assert(!fb::PipelineStep<ReadsOwnOutput>);

struct OutputNotAProduct {
  static constexpr std::string_view kName = "output_not_a_product";
  fb::Result<const B> run(fb::View<A> in) const;
};
static_assert(!fb::PipelineStep<OutputNotAProduct>);

struct ReturnsBareOutput {
  static constexpr std::string_view kName = "returns_bare_output";
  B run(fb::View<A> in) const;
};
static_assert(!fb::PipelineStep<ReturnsBareOutput>);

struct NonConstRun {
  static constexpr std::string_view kName = "non_const_run";
  fb::Result<B> run(fb::View<A> in);
};
static_assert(!fb::PipelineStep<NonConstRun>);

struct NoShape {
  static constexpr std::string_view kName = "no_shape";
  fb::Result<B> run(fb::View<A> in, std::int32_t &unknown) const;
};
static_assert(!fb::PipelineStep<NoShape>);

// Exact resource types only: a const DeviceResources& is not a shape.
struct ConstDeviceResources {
  static constexpr std::string_view kName = "const_device_resources";
  fb::Result<B> run(fb::View<A> in, const fb::DeviceResources &res) const;
};
static_assert(!fb::PipelineStep<ConstDeviceResources>);

// An overload set has no one signature to read.
struct OverloadedRun {
  static constexpr std::string_view kName = "overloaded_run";
  fb::Result<B> run(fb::View<A> in) const;
  fb::Result<B> run(fb::View<A> in, fb::DeviceResources &res) const;
};
static_assert(!fb::PipelineStep<OverloadedRun>);

struct TemplateRun {
  static constexpr std::string_view kName = "template_run";
  template <class V>
  fb::Result<B> run(V in) const;
};
static_assert(!fb::PipelineStep<TemplateRun>);

// --- Running ----------------------------------------------------------------

TEST(PipelineStep, ViewGetReturnsTheBoundObject) {
  const A a{.value = 3};
  const B b{.value = 0.5};
  const fb::View<A, B> view{a, b};

  EXPECT_EQ(&view.get<A>(), &a);
  EXPECT_EQ(&view.get<B>(), &b);
}

TEST(PipelineStep, HostStepRunsOnAHandBuiltView) {
  const A a{.value = 3};
  const HostStep step{.scale = 1.5};

  const fb::Result<B> out = step.run(fb::View<A>{a});

  ASSERT_TRUE(out.has_value());
  EXPECT_DOUBLE_EQ(out->value, 4.5);
}

TEST(PipelineStep, RootStepReadsOnlyItsOptions) {
  const RootStep step{};

  const fb::Result<A> out = step.run(fb::View<>{});

  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->value, 7);
}

} // namespace
