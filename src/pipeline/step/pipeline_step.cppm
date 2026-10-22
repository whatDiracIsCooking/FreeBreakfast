/// @file
/// @brief `:pipeline_step`: the concept a step type satisfies, and what it
///        reads, writes and holds, all read off its `run` signature.

export module free_breakfast.pipeline.step:pipeline_step;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import :view;
import :run_shape;

// Unexported, which in a module is the privacy; in an interface partition so
// the exported templates below can reach them from an importer's TU.
namespace free_breakfast {

template <class V>
constexpr bool kIsView = false;
template <class... Ts>
constexpr bool kIsView<View<Ts...>> = true;

template <class T, class V>
constexpr bool kViewContains = false;
template <class T, class... Ts>
constexpr bool kViewContains<T, View<Ts...>> = (std::same_as<T, Ts> || ...);

// The five resource lists a run may take after its view, exactly: anything
// else (a tf::FlowBuilder&, a const DeviceResources&) has no shape.
template <class... Resources>
constexpr std::optional<RunShape> kShapeOfResources = std::nullopt;
template <>
constexpr std::optional<RunShape> kShapeOfResources<> = RunShape::Host;
template <>
constexpr std::optional<RunShape> kShapeOfResources<tf::Subflow &> = RunShape::HostSubflow;
template <>
constexpr std::optional<RunShape> kShapeOfResources<DeviceResources &> = RunShape::Device;
template <>
constexpr std::optional<RunShape> kShapeOfResources<DeviceResources &, tf::Subflow &> =
    RunShape::DeviceSubflow;
template <>
constexpr std::optional<RunShape> kShapeOfResources<NodeResources &> = RunShape::Node;

// The parts of a `const` member function's type. Taking &S::run is what
// rejects an overloaded or template run: there is no one signature to read.
// A non-const run is a different member-pointer type and matches nothing.
template <class F>
struct RunSignature {};
template <class S, class Return, class Input, class... Resources, bool kNoexcept>
struct RunSignature<Return (S::*)(Input, Resources...) const noexcept(kNoexcept)> {
  using InputView = std::remove_cvref_t<Input>;
  using ReturnType = Return;
  static constexpr std::optional<RunShape> kShape = kShapeOfResources<Resources...>;
};

template <class S>
using RunSignatureOf = RunSignature<decltype(&S::run)>;

// Result<T> is std::expected<T, Error>: a bare T, or another expected, has no
// output.
template <class R>
struct ResultValue {};
template <class T>
struct ResultValue<Result<T>> {
  using type = T;
};

// kName is the step's TOML table name, so it must be a non-empty constant: the
// pipeline checks names unique across its steps at compile time.
template <class S>
concept StepName = requires { typename std::bool_constant<std::string_view{S::kName}.empty()>; } &&
                   !std::string_view{S::kName}.empty();

} // namespace free_breakfast

export namespace free_breakfast {

/// @brief The `View` step `S` reads: its `run`'s first parameter.
template <class S>
using StepInput = typename RunSignatureOf<S>::InputView;

/// @brief The product step `S` writes: the `T` of its `run`'s `Result<T>`.
template <class S>
using StepOutput = typename ResultValue<typename RunSignatureOf<S>::ReturnType>::type;

/// @brief A pipeline step: a non-empty `kName` and one `const` `run`, whose
///        signature declares everything else.
///
/// ```cpp
/// struct OrthogonalizerStep {
///   static constexpr std::string_view kName = "orthogonalizer";
///   double eigenvalue_floor = kOrthogonalizeEigenvalueFloor<double>;  // an option
///   Result<Orthogonalizer> run(View<BasisSet, Overlap> in, DeviceResources &res) const;
/// };
/// ```
///
/// - The first parameter is a `View<Ts...>`: what the step reads.
/// - The return type is `Result<T>`, `T` a product not among the `Ts`: the
///   one product it writes (reading its own output is a cycle the compiler
///   can already see).
/// - The remaining parameters are exactly one of the five `RunShape`
///   resource lists.
/// - `run` is `const` (a step is configuration, not changed by running),
///   neither overloaded nor a template, and may be `noexcept`.
///
/// Options are the step's own data members. Contracts the types cannot check
/// are in README.md: a device `run` only enqueues on `res.stream()`, a host
/// `run` never touches the device, and a subflow holding references into the
/// task's frame is joined before `run` returns.
template <class S>
concept PipelineStep =
    requires {
      typename StepInput<S>;
      typename StepOutput<S>;
    } && kIsView<StepInput<S>> && Product<StepOutput<S>> &&
    !kViewContains<StepOutput<S>, StepInput<S>> && RunSignatureOf<S>::kShape.has_value() &&
    StepName<S>;

/// @brief The `RunShape` of step `S`'s `run`.
template <PipelineStep S>
constexpr RunShape kRunShape = *RunSignatureOf<S>::kShape;

} // namespace free_breakfast
