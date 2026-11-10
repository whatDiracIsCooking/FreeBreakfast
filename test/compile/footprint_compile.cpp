// Compile-time checks of the footprint vocabulary: core.footprint's kinds are
// distinct types that do not compose with each other, and the footprint is a
// requirement of pipeline.step's PipelineStep (#1122): a made-up step with a
// well-formed `MemoryFootprint footprint(const FootprintDimensions &) const`
// is a PipelineStep, and the same step without one, or with one malformed in
// any one way (non-const, static, overloaded, a template, another return or
// parameter type, a data member), is not. No member is defined: the concept
// never calls one.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.pipeline.footprint;
import free_breakfast.pipeline.step;

namespace {

namespace fb = free_breakfast;

using Device = fb::Footprint<wwr::extension::MemoryKind::Device>;
using Pinned = fb::Footprint<wwr::extension::MemoryKind::Pinned>;

// ---------------------------------------------------------------------------
// Footprint<K>: the kind is in the type
// ---------------------------------------------------------------------------

static_assert(!std::same_as<Device, Pinned>);
static_assert(!std::convertible_to<Device, Pinned>);
static_assert(!std::convertible_to<Pinned, Device>);

template <class A, class B>
concept Thenable = requires(const A &a, const B &b) { fb::then(a, b); };

static_assert(Thenable<Device, Device>);
static_assert(Thenable<Pinned, Pinned>);
static_assert(Thenable<fb::MemoryFootprint, fb::MemoryFootprint>);
static_assert(!Thenable<Device, Pinned>);
static_assert(!Thenable<Pinned, Device>);
static_assert(std::same_as<decltype(fb::drop(Pinned{})), Pinned>);

// ---------------------------------------------------------------------------
// The footprint member, as PipelineStep requires it
// ---------------------------------------------------------------------------

struct Made {};

// A made-up step, but for its footprint: a non-empty kName and one const run
// over the device. Each model below adds the footprint member, or misses it
// in one way.
struct StepBase {
  static constexpr std::string_view kName = "made";
  fb::Result<Made> run(fb::View<> in, fb::DeviceResources &res) const;
};

struct WellFormed : StepBase {
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &) const;
};

struct WellFormedNoexcept : StepBase {
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &) const noexcept;
};

struct NoFootprint : StepBase {};

struct NonConst : StepBase {
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &);
};

struct Static : StepBase {
  static fb::MemoryFootprint footprint(const fb::FootprintDimensions &);
};

struct Overloaded : StepBase {
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &) const;
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &, std::size_t nroots) const;
};

struct Template : StepBase {
  template <class D>
  fb::MemoryFootprint footprint(const D &) const;
};

struct WrongReturn : StepBase {
  Device footprint(const fb::FootprintDimensions &) const;
};

struct OptionalReturn : StepBase {
  std::optional<fb::MemoryFootprint> footprint(const fb::FootprintDimensions &) const;
};

struct WrongParameter : StepBase {
  fb::MemoryFootprint footprint(fb::FootprintDimensions) const;
};

struct DataMember : StepBase {
  fb::MemoryFootprint footprint;
};

static_assert(fb::PipelineStep<WellFormed>);
static_assert(fb::PipelineStep<WellFormedNoexcept>);

static_assert(!fb::PipelineStep<NoFootprint>);
static_assert(!fb::PipelineStep<NonConst>);
static_assert(!fb::PipelineStep<Static>);
static_assert(!fb::PipelineStep<Overloaded>);
static_assert(!fb::PipelineStep<Template>);
static_assert(!fb::PipelineStep<WrongReturn>);
static_assert(!fb::PipelineStep<OptionalReturn>);
static_assert(!fb::PipelineStep<WrongParameter>);
static_assert(!fb::PipelineStep<DataMember>);

// ---------------------------------------------------------------------------
// The plan fold is a constant expression too
// ---------------------------------------------------------------------------

constexpr std::array<fb::MemoryFootprint, 2> kPlan{
    fb::MemoryFootprint{.device = {.persistent = 10, .scratch = 5, .workspace = 0}},
    fb::MemoryFootprint{.device = {.persistent = 20, .scratch = 7, .workspace = 3}}};
static_assert(fb::plan_footprint(kPlan).footprint.device.peak() == 10 + 20 + 7 + 3);
static_assert(fb::plan_footprint(kPlan).device_peak_step == 1);

} // namespace
