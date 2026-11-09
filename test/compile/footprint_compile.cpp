// Compile-time checks of the footprint vocabulary: core.footprint's kinds are
// distinct types that do not compose with each other, and pipeline.footprint's
// step_footprint recognises a well-formed footprint member and nothing else.
//
// The detection itself is unexported (#1122 deletes it), so it is checked
// through step_footprint, the one exported template that uses it, evaluated
// at compile time: a model with a well-formed member gives its footprint, and
// each model that misses it in one way gives none. Only the well-formed
// models' members are defined, since only they are called.

import std;
import free_breakfast.core;
import free_breakfast.pipeline.footprint;

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
// The footprint member, as step_footprint detects it
// ---------------------------------------------------------------------------

constexpr fb::MemoryFootprint kWellFormed{
    .device = {.persistent = 1, .scratch = 2, .workspace = 3},
    .pinned = {.persistent = 4, .scratch = 5, .workspace = 6}};

struct WellFormed {
  constexpr fb::MemoryFootprint footprint(const fb::FootprintDimensions &) const {
    return kWellFormed;
  }
};

struct WellFormedNoexcept {
  constexpr fb::MemoryFootprint footprint(const fb::FootprintDimensions &) const noexcept {
    return kWellFormed;
  }
};

struct NoFootprint {};

struct NonConst {
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &);
};

struct Static {
  static fb::MemoryFootprint footprint(const fb::FootprintDimensions &);
};

struct Overloaded {
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &) const;
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &, std::size_t nroots) const;
};

struct Template {
  template <class D>
  fb::MemoryFootprint footprint(const D &) const;
};

struct WrongReturn {
  Device footprint(const fb::FootprintDimensions &) const;
};

struct WrongParameter {
  fb::MemoryFootprint footprint(fb::FootprintDimensions) const;
};

struct DataMember {
  fb::MemoryFootprint footprint;
};

template <class S>
constexpr bool kDetected = fb::step_footprint(S{}, fb::FootprintDimensions{}).has_value();

static_assert(kDetected<WellFormed>);
static_assert(kDetected<WellFormedNoexcept>);
static_assert(*fb::step_footprint(WellFormed{}, fb::FootprintDimensions{}) == kWellFormed);

static_assert(!kDetected<NoFootprint>);
static_assert(!kDetected<NonConst>);
static_assert(!kDetected<Static>);
static_assert(!kDetected<Overloaded>);
static_assert(!kDetected<Template>);
static_assert(!kDetected<WrongReturn>);
static_assert(!kDetected<WrongParameter>);
static_assert(!kDetected<DataMember>);

// ---------------------------------------------------------------------------
// The plan fold is a constant expression too
// ---------------------------------------------------------------------------

constexpr std::array<fb::MemoryFootprint, 2> kPlan{
    fb::MemoryFootprint{.device = {.persistent = 10, .scratch = 5, .workspace = 0}},
    fb::MemoryFootprint{.device = {.persistent = 20, .scratch = 7, .workspace = 3}}};
static_assert(fb::plan_footprint(kPlan).footprint.device.peak() == 10 + 20 + 7 + 3);
static_assert(fb::plan_footprint(kPlan).device_peak_step == 1);

} // namespace
