// Compile-time checks of tables as concepts (#1674): a step's table is its
// kTable, or its kName when it declares none, and its keys its kKeys, or none;
// several steps may share one table, and two steps of one table declaring the
// same key is a registry pipeline.pipeline's Pipeline refuses to instantiate.
// No member is defined: neither the concepts nor the constraint calls one.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.pipeline.footprint;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.pipeline;

namespace {

namespace fb = free_breakfast;

struct Atoms {};
struct Built {};
struct Other {};

// A table's head, its kName the table's name, and a step of the same table
// reading its own keys and producing what the head reads.
struct Head {
  static constexpr std::string_view kName = "table";
  static constexpr std::array<std::string_view, 2> kKeys{"charge", "multiplicity"};
  fb::Result<Built> run(fb::View<Atoms> in, fb::DeviceResources &res) const;
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &) const;
};

struct Member {
  static constexpr std::string_view kName = "member";
  static constexpr std::string_view kTable = "table";
  static constexpr std::array<std::string_view, 1> kKeys{"file"};
  fb::Result<Atoms> run(fb::View<> in) const;
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &) const;
};

// As Member, but declaring the head's 'charge' too.
struct Clashing {
  static constexpr std::string_view kName = "member";
  static constexpr std::string_view kTable = "table";
  static constexpr std::array<std::string_view, 2> kKeys{"file", "charge"};
  fb::Result<Atoms> run(fb::View<> in) const;
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &) const;
};

// The same key in a table of its own: no clash.
struct Apart {
  static constexpr std::string_view kName = "apart";
  static constexpr std::array<std::string_view, 1> kKeys{"charge"};
  fb::Result<Other> run(fb::View<Built> in) const;
  fb::MemoryFootprint footprint(const fb::FootprintDimensions &) const;
};

// A kTable that is empty, or not a constant, is no table.
struct EmptyTable : Member {
  static constexpr std::string_view kTable = "";
};
struct RuntimeTable : Member {
  static inline std::string_view kTable = "table";
};

// A registry Pipeline can be instantiated over.
template <class... Steps>
concept Plannable = requires { typename fb::Pipeline<Steps...>; };

// ---------------------------------------------------------------------------
// The table and keys a step declares, or its defaults
// ---------------------------------------------------------------------------

static_assert(fb::PipelineStep<Head>);
static_assert(fb::PipelineStep<Member>);
static_assert(fb::kStepTable<Head> == "table");
static_assert(fb::kStepTable<Member> == "table");
static_assert(fb::kStepTable<Apart> == "apart");
static_assert(fb::kStepKeys<Member>.size() == 1 && fb::kStepKeys<Member>[0] == "file");
static_assert(!fb::PipelineStep<EmptyTable>);
static_assert(!fb::PipelineStep<RuntimeTable>);

// ---------------------------------------------------------------------------
// One table, several steps: their keys must be disjoint
// ---------------------------------------------------------------------------

static_assert(Plannable<Member, Head>);
static_assert(Plannable<Member, Head, Apart>);
static_assert(!Plannable<Clashing, Head>);
static_assert(!Plannable<Clashing, Head, Apart>);

} // namespace
