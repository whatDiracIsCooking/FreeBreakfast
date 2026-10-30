/// @file
/// @brief `Pipeline<Steps...>`: a set of steps planned into a task graph,
///        run on a `NodeResources`, and the products it built.
///
/// A class template over a registry of step types, so a test plans over a
/// few made-up steps and the real registry is one instantiation
/// (`pipeline.config`'s `Calculation`). `create` is the planner: it wires each
/// step's `View` to the steps producing its types and refuses a pipeline that
/// cannot run, before anything touches a device. `run` is the scheduling. See
/// README.md.

export module free_breakfast.pipeline.pipeline;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.core.wwr;
import free_breakfast.pipeline.step;

#include "diagnostics.h"

// Unexported, which in a module is the privacy; in the interface so the
// exported template below can reach them from an importer's TU.
namespace free_breakfast {

// How many of Steps produce T.
template <class T, class... Steps>
constexpr size_t kProducerCount = (size_t{0} + ... + size_t{std::same_as<StepOutput<Steps>, T>});

// The position in Steps of the step producing T; sizeof...(Steps) if none.
template <class T, class... Steps>
constexpr size_t kProducerIndex = [] {
  size_t i = 0;
  (void)((std::same_as<StepOutput<Steps>, T> ? false : (++i, true)) && ...);
  return i;
}();

// What a step's View reads, as positions in Steps.
template <class V, class... Steps>
struct InputProducers {};
template <class... Ts, class... Steps>
struct InputProducers<View<Ts...>, Steps...> {
  static constexpr bool kAllProduced = ((kProducerCount<Ts, Steps...> == 1) && ...);
  static constexpr std::array<size_t, sizeof...(Ts)> kIndices{kProducerIndex<Ts, Steps...>...};
};

// What step S's load reads, as positions in Steps; none for a step with no
// load, which is never planned as loaded.
template <class S, class... Steps>
constexpr std::span<const size_t> load_producers() {
  if constexpr (LoadableStep<S>) {
    return InputProducers<LoadInput<S>, Steps...>::kIndices;
  } else {
    return {};
  }
}

template <class S, class... Steps>
constexpr bool kLoadInputsProduced = [] {
  if constexpr (LoadableStep<S>) {
    return InputProducers<LoadInput<S>, Steps...>::kAllProduced;
  } else {
    return true;
  }
}();

// Where step S runs when it is loaded; its run's shape for a step with no load.
template <class S>
constexpr RunShape load_shape() {
  if constexpr (LoadableStep<S>) {
    return kLoadShape<S>;
  } else {
    return kRunShape<S>;
  }
}

// The alternatives of Pipeline::Step: every step, then Load<S> for each step
// with a load, so a Load of one without fails to compile.
template <class S, bool = LoadableStep<S>>
struct LoadAlternative {
  using type = std::tuple<>;
};
template <class S>
struct LoadAlternative<S, true> {
  using type = std::tuple<Load<S>>;
};
template <class Tuple>
struct VariantOver {};
template <class... Ts>
struct VariantOver<std::tuple<Ts...>> {
  using type = std::variant<Ts...>;
};
template <class... Steps>
using StepVariant = typename VariantOver<decltype(std::tuple_cat(
    std::declval<std::tuple<Steps...>>(),
    std::declval<typename LoadAlternative<Steps>::type>()...))>::type;

template <class... Steps>
constexpr bool kUniqueStepNames = [] {
  const std::array<std::string_view, sizeof...(Steps)> names{Steps::kName...};
  for (size_t i = 0; i < names.size(); ++i) {
    for (size_t j = i + 1; j < names.size(); ++j) {
      if (names[i] == names[j]) {
        return false;
      }
    }
  }
  return true;
}();

// A registry a Pipeline can plan over: steps with unique names, one producer
// per product, and a producer for everything any step's run or load reads. A
// pipeline built from it may still leave a producer out, which create reports.
template <class... Steps>
concept StepRegistry = (PipelineStep<Steps> && ...) && kUniqueStepNames<Steps...> &&
                       ((kProducerCount<StepOutput<Steps>, Steps...> == 1) && ...) &&
                       (InputProducers<StepInput<Steps>, Steps...>::kAllProduced && ...) &&
                       (kLoadInputsProduced<Steps, Steps...> && ...);

// The shapes whose run takes a DeviceResources: they need a device, and the
// pipeline waits on it after them.
constexpr bool needs_device(RunShape shape) {
  return shape == RunShape::Device || shape == RunShape::DeviceSubflow;
}

// The shapes on the device chain: run one after another, in plan order.
constexpr bool on_device_chain(RunShape shape) {
  return needs_device(shape) || shape == RunShape::Node;
}

} // namespace free_breakfast

export namespace free_breakfast {

/// @brief A set of steps from the registry `Steps`, planned into a task graph
/// by @ref create, run once by @ref run, and holding what each step produced.
///
/// ```cpp
/// using Calc = Pipeline<MoleculeStep, BasisSetStep, OverlapStep, ...>;
/// Calc calc = FBF_TRY(Calc::create({MoleculeStep{.xyz = "w.xyz"}, ...}));
/// FBF_TRY(calc.run(node));
/// if (const Result<OverlapOutput> &s = calc.get<OverlapOutput>()) { ... s->s ... }
/// ```
///
/// The registry is checked at compile time: unique `kName`s, one producer per
/// product, and a producer in `Steps` for every type any step reads. Which of
/// them a pipeline holds is decided at run time (from a config file), so a
/// missing producer, a step given twice and a cycle are @ref create's errors.
///
/// **Scheduling** (`run`). Every step is one task. A step runs after the steps
/// producing its inputs. Steps that hold a device (`Device`, `DeviceSubflow`,
/// `Node`) are chained in plan order, so one drives the device at a time; a
/// host step runs beside them as soon as its inputs exist. A device step runs
/// on device 0 under a `ScopedDeviceIndex`, and its stream is waited on
/// after it, so its product is finished when a consumer, host or device, reads
/// it.
///
/// **Failure.** A step whose input failed does not run: its slot holds the
/// first failed input's error. `run` returns the first error in plan order,
/// which is a step's own, whatever the schedule.
///
/// **Nesting.** A step may run a Pipeline of its own on the same node
/// (OptimizeStep): run from an executor worker, the flow is co-run on it.
///
/// **Free order.** The products are destroyed in reverse plan order with
/// their device current, after the run has waited on every device step.
///
/// Movable until it runs, not copyable, not assignable.
template <PipelineStep... Steps>
  requires StepRegistry<Steps...>
class Pipeline {
  static constexpr size_t kStepCount = sizeof...(Steps);
  template <size_t I>
  using StepAt = std::tuple_element_t<I, std::tuple<Steps...>>;

public:
  /// @brief One step of the registry, as the front end hands it over: an `S`
  /// to run, or a `Load<S>` to read its product instead, for an `S` with a
  /// `load`.
  using Step = StepVariant<Steps...>;

  /// @brief Plan @p steps: wire each one's `View` to its producers, and order
  /// them. A loaded step is wired by its `load`'s `View`, so what only its
  /// `run` reads is not planned for it. ErrorKind::InvalidConfig, naming the
  /// steps, for a step given twice (to run or to load), a step reading a
  /// product no step in @p steps writes, or a cycle.
  [[nodiscard]] static Result<Pipeline> create(std::vector<Step> steps) {
    std::nullptr_t strict = nullptr;
    return plan(std::move(steps), strict);
  }

  /// @brief As @ref create, but a missing producer `S` is made by
  /// `insert.template operator()<S>()`, a `Result<S>`, and planned in turn.
  /// Its error is returned as it is.
  ///
  /// The front end's hook for filling in a requested step's producers: the
  /// config file inserts a step by reading it from an empty table.
  template <class Insert>
  [[nodiscard]] static Result<Pipeline> create(std::vector<Step> steps, Insert &&insert) {
    return plan(std::move(steps), insert);
  }

  Pipeline(Pipeline &&) noexcept = default;
  Pipeline(const Pipeline &) = delete;
  Pipeline &operator=(const Pipeline &) = delete;
  Pipeline &operator=(Pipeline &&) = delete;

  /// @brief Destroys the products in reverse plan order, with the device
  /// they live on current.
  ~Pipeline() {
    std::optional<ScopedDeviceIndex> on;
    if (dev_idx_) {
      on.emplace(*dev_idx_);
    }
    for (const size_t i : std::views::reverse(order_)) {
      (void)with_index(i, [&]<size_t I>() -> Status {
        std::get<I>(outputs_).reset();
        return {};
      });
    }
  }

  /// @brief Run every planned step on @p node, wait for all of them, and
  /// return the first error in plan order.
  ///
  /// Device steps use `node.device(0)`; with none, a pipeline holding one is
  /// ErrorKind::GpuRuntime before any step runs. A second `run` is
  /// ErrorKind::Unsupported: re-running a pipeline waits for something to
  /// need it.
  [[nodiscard]] Status run(NodeResources &node) {
    if (ran_) {
      return err_unsupported("a pipeline runs once");
    }
    DeviceResources *res = nullptr;
    for (const size_t i : order_) {
      if (needs_device(shape(i))) {
        if (node.device_count() == 0) {
          return err_gpu("[{}] needs a GPU device, and none is visible", kNames[i]);
        }
        res = &node.device(0);
        dev_idx_ = res->dev_idx();
        break;
      }
    }
    ran_ = true;

    tf::Taskflow flow;
    std::array<tf::Task, kStepCount> tasks{};
    std::optional<tf::Task> chain_tail;
    for (const size_t i : order_) {
      (void)with_index(i, [&]<size_t I>() -> Status {
        tasks[I] = emplace_task<I>(flow, node, res);
        return {};
      });
      for (const size_t j : inputs(i)) {
        tasks[j].precede(tasks[i]);
      }
      if (on_device_chain(shape(i))) {
        if (chain_tail) {
          chain_tail->precede(tasks[i]);
        }
        chain_tail = tasks[i];
      }
    }
    // A pipeline run from inside a step of another (OptimizeStep's chain) is
    // on one of the executor's workers, and blocking it on wait() would hold
    // a worker the flow may need: corun runs the flow on it instead.
    if (node.executor().this_worker_id() >= 0) {
      node.executor().corun(flow);
    } else {
      node.executor().run(flow).wait();
    }

    for (const size_t i : order_) {
      if (const Error *error = failure(i)) {
        return std::unexpected(*error);
      }
    }
    return {};
  }

  /// @brief The product of type `T`: its value, its step's error, or, for a
  /// product this pipeline did not plan or has not run, ErrorKind::InvalidConfig.
  template <class T>
    requires(std::same_as<T, StepOutput<Steps>> || ...)
  [[nodiscard]] const Result<T> &get() const {
    constexpr size_t kI = kProducerIndex<T, Steps...>;
    if (const auto &slot = std::get<kI>(outputs_)) {
      return *slot;
    }
    static const Result<T> not_run = err_config("[{}] has not run in this pipeline", kNames[kI]);
    return not_run;
  }

  /// @brief The planned step of type `S`, as given or filled in; empty if the
  /// pipeline does not hold one, or loads its product.
  template <class S>
    requires(std::same_as<S, Steps> || ...)
  [[nodiscard]] const std::optional<S> &step() const {
    return std::get<std::optional<S>>(steps_);
  }

  /// @brief Where the product of step `S` is loaded from; empty if the
  /// pipeline does not load it.
  template <class S>
    requires(std::same_as<S, Steps> || ...) && LoadableStep<S>
  [[nodiscard]] const std::optional<std::filesystem::path> &from() const {
    return from_[kProducerIndex<StepOutput<S>, Steps...>];
  }

private:
  static constexpr std::array<std::string_view, kStepCount> kNames{Steps::kName...};
  static constexpr std::array<RunShape, kStepCount> kShapes{kRunShape<Steps>...};
  static constexpr std::array<std::span<const size_t>, kStepCount> kInputs{
      std::span<const size_t>{InputProducers<StepInput<Steps>, Steps...>::kIndices}...};
  static constexpr std::array<std::span<const size_t>, kStepCount> kLoadInputs{
      load_producers<Steps, Steps...>()...};
  static constexpr std::array<RunShape, kStepCount> kLoadShapes{load_shape<Steps>()...};

  Pipeline() = default;

  // create's body. Insert is std::nullptr_t when a missing producer is an
  // error.
  template <class Insert>
  static Result<Pipeline> plan(std::vector<Step> steps, Insert &insert) {
    Pipeline pipeline;
    std::array<bool, kStepCount> planned{};
    for (Step &step : steps) {
      FBF_TRY(std::visit([&](auto &s) { return pipeline.take(s, planned); }, step));
    }
    // Close over the producers, in registry order, so an insertion error is
    // the same whichever order the steps came in.
    for (bool grew = true; grew;) {
      grew = false;
      for (size_t i = 0; i < kStepCount; ++i) {
        if (!planned[i]) {
          continue;
        }
        for (const size_t j : pipeline.inputs(i)) {
          if (planned[j]) {
            continue;
          }
          if constexpr (std::same_as<Insert, std::nullptr_t>) {
            return err_config("[{}] reads the product of [{}], which is not in the pipeline",
                              kNames[i], kNames[j]);
          } else {
            FBF_TRY(with_index(j, [&]<size_t J>() -> Status {
              std::get<J>(pipeline.steps_) = FBF_TRY(insert.template operator()<StepAt<J>>());
              return {};
            }));
          }
          planned[j] = grew = true;
        }
      }
    }
    pipeline.order_ = FBF_TRY(pipeline.plan_order(planned));
    return pipeline;
  }

  // Take step S into the plan, to run.
  template <class S>
  Status take(S &step, std::array<bool, kStepCount> &planned) {
    constexpr size_t kI = kProducerIndex<StepOutput<S>, Steps...>;
    FBF_TRY(plan_once(kI, planned));
    std::get<kI>(steps_) = std::move(step);
    return {};
  }

  // Take step S into the plan, to load its product.
  template <class S>
  Status take(Load<S> &load, std::array<bool, kStepCount> &planned) {
    constexpr size_t kI = kProducerIndex<StepOutput<S>, Steps...>;
    FBF_TRY(plan_once(kI, planned));
    from_[kI] = std::move(load.from);
    return {};
  }

  static Status plan_once(size_t i, std::array<bool, kStepCount> &planned) {
    if (planned[i]) {
      return err_config("[{}] is given twice", kNames[i]);
    }
    planned[i] = true;
    return {};
  }

  // What step i reads, and where it runs: its load's if it is loaded, its
  // run's otherwise.
  std::span<const size_t> inputs(size_t i) const { return from_[i] ? kLoadInputs[i] : kInputs[i]; }
  RunShape shape(size_t i) const { return from_[i] ? kLoadShapes[i] : kShapes[i]; }

  // Call f.template operator()<I>() for the runtime index i < kStepCount.
  template <class F>
  static Status with_index(size_t i, F &&f) {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
      Status status;
      (void)((i == Is ? (status = f.template operator()<Is>(), false) : true) && ...);
      return status;
    }(std::make_index_sequence<kStepCount>{});
  }

  // The planned steps in an order where each follows its producers: Kahn's,
  // lowest registry position first, so the plan is the same for any input
  // order. Steps left over form a cycle.
  Result<std::vector<size_t>> plan_order(const std::array<bool, kStepCount> &planned) const {
    std::vector<size_t> order;
    std::array<bool, kStepCount> placed{};
    for (bool grew = true; grew;) {
      grew = false;
      for (size_t i = 0; i < kStepCount; ++i) {
        if (!planned[i] || placed[i] ||
            !std::ranges::all_of(inputs(i), [&](size_t j) { return placed[j]; })) {
          continue;
        }
        order.push_back(i);
        placed[i] = grew = true;
        break; // restart, so the lowest ready position always goes first
      }
    }
    if (order.size() != static_cast<size_t>(std::ranges::count(planned, true))) {
      std::string cycle;
      for (size_t i = 0; i < kStepCount; ++i) {
        if (planned[i] && !placed[i]) {
          cycle += std::format("{}[{}]", cycle.empty() ? "" : ", ", kNames[i]);
        }
      }
      return err_config("the steps {} read each other's products in a cycle", cycle);
    }
    return order;
  }

  // Step i's error, or nullptr if it produced its product.
  const Error *failure(size_t i) const {
    const Error *error = nullptr;
    (void)with_index(i, [&]<size_t I>() -> Status {
      const auto &slot = std::get<I>(outputs_);
      if (slot && !slot->has_value()) {
        error = &slot->error();
      }
      return {};
    });
    return error;
  }

  // Step I's View over its producers' products, or the first failed one's
  // error.
  template <class... Ts>
  Result<View<Ts...>> bind(std::type_identity<View<Ts...>> /*view*/) const {
    const Error *error = nullptr;
    (void)(((error = failure(kProducerIndex<Ts, Steps...>)) == nullptr) && ...);
    if (error != nullptr) {
      return std::unexpected(*error);
    }
    return View<Ts...>{**std::get<kProducerIndex<Ts, Steps...>>(outputs_)...};
  }

  // make(resources...) on the resources kShape names. A device one runs
  // under a ScopedDeviceIndex, and its stream is waited on after.
  template <class T, RunShape kShape, class Make, class... Subflow>
  static Result<T> on_resources(const Make &make, NodeResources &node, DeviceResources *res,
                                Subflow &...subflow) {
    if constexpr (kShape == RunShape::Host || kShape == RunShape::HostSubflow) {
      return make(subflow...);
    } else if constexpr (needs_device(kShape)) {
      const ScopedDeviceIndex on{res->dev_idx()};
      auto out = make(*res, subflow...);
      if (out) {
        FBF_TRY(res->sync());
      }
      return out;
    } else {
      static_assert(kShape == RunShape::Node);
      auto out = make(node);
      for (DeviceResources &device : node.devices()) {
        const ScopedDeviceIndex on{device.dev_idx()};
        FBF_TRY(device.sync());
      }
      return out;
    }
  }

  // Step I's task, writing its slot with make on the resources kShape names.
  template <size_t I, RunShape kShape, class Make>
  tf::Task emplace_shaped(tf::Taskflow &flow, NodeResources &node, DeviceResources *res,
                          Make make) {
    using T = StepOutput<StepAt<I>>;
    auto &slot = std::get<I>(outputs_);
    if constexpr (kShape == RunShape::HostSubflow || kShape == RunShape::DeviceSubflow) {
      return flow.emplace([&slot, &node, res, make](tf::Subflow &subflow) {
        slot = on_resources<T, kShape>(make, node, res, subflow);
      });
    } else {
      return flow.emplace(
          [&slot, &node, res, make] { slot = on_resources<T, kShape>(make, node, res); });
    }
  }

  // Step I as one task of flow, after its inputs: its load if it is loaded,
  // its run otherwise.
  template <size_t I>
  tf::Task emplace_task(tf::Taskflow &flow, NodeResources &node, DeviceResources *res) {
    using S = StepAt<I>;
    using T = StepOutput<S>;
    if constexpr (LoadableStep<S>) {
      if (from_[I]) {
        return emplace_shaped<I, kLoadShape<S>>(
            flow, node, res, [this](auto &...resources) -> Result<T> {
              const auto view = FBF_TRY(bind(std::type_identity<LoadInput<S>>{}));
              return S::load(*from_[I], view, resources...);
            });
      }
    }
    return emplace_shaped<I, kRunShape<S>>(
        flow, node, res, [this](auto &...resources) -> Result<T> {
          const auto view = FBF_TRY(bind(std::type_identity<StepInput<S>>{}));
          return std::get<I>(steps_)->run(view, resources...);
        });
  }

  std::tuple<std::optional<Steps>...> steps_;
  // Each loaded step's path, in place of the step.
  std::array<std::optional<std::filesystem::path>, kStepCount> from_;
  std::tuple<std::optional<Result<StepOutput<Steps>>>...> outputs_;
  // The planned steps' registry positions, each after its producers.
  std::vector<size_t> order_;
  // The device the products live on, once a device step has run.
  std::optional<int32_t> dev_idx_;
  bool ran_ = false;
};

} // namespace free_breakfast
