# Pipeline: the planner

`Pipeline<Steps...>`: a set of steps planned into a task graph, run once on a
`NodeResources`, and the products it built.

## Module

`free_breakfast.pipeline.pipeline`, target `free_breakfast::pipeline::pipeline`.
Templates only, so no implementation unit. It imports `pipeline.step`,
`core.resources` (the node, and Taskflow through it) and `core.wwr`
(`ScopedDeviceIndex`), and **no step module**: a registry of made-up steps
compiles none of the real ones. The real registry is `pipeline.config`'s
`Calculation`.

```cpp
template <PipelineStep... Steps>
  requires StepRegistry<Steps...>
class Pipeline {
public:
  using Step = std::variant<Steps..., Load<S>...>;  // a Load<S> per S with a load

  static Result<Pipeline> create(std::vector<Step> steps);                  // strict
  template <class Insert>
  static Result<Pipeline> create(std::vector<Step> steps, Insert &&insert); // fill in producers

  Status run(NodeResources &node);

  template <class T> const Result<T> &get() const;          // T a product of Steps
  template <class S> const std::optional<S> &step() const;  // S one of Steps
  template <class S> const std::optional<std::filesystem::path> &from() const;  // S loadable
};
```

## The registry, checked at compile time

`Steps` must have unique `kName`s, exactly one producer per product, and a
producer for every type any step's `View` reads, its `run`'s or its `load`'s. A list that breaks one does
not form a `Pipeline` type. So a type → producer lookup is a constant, and
each step's edges are known before any config is read.

What the registry cannot know is which of its steps a given pipeline holds,
since that comes from a config file. That is `create`'s job.

## `create`: the planner

- **One instance per step.** A step given twice is `InvalidConfig`, and so is
  a step given both to run and to load. Two of one kind would need labels
  (orchestration section 9).
- **Run or load.** A `Load<S>{path}` plans `S` to read its product with
  `S::load` instead of making it with `S::run`. Its edges and its shape are
  `load`'s, so what only `run` reads is not filled in for it, and a step that
  runs on the device may load on the host, beside the chain. A step with no
  `load` has no `Load<S>` alternative in `Step`, so asking for one does not
  compile (`step/README.md`).
- **A producer for every input.** Strict `create` refuses a step whose input
  has no producer in the list (`[sum] reads the product of [double], which is
  not in the pipeline`). With `insert`, a missing producer `S` is made by
  `insert.template operator()<S>()` (a `Result<S>`) and planned in turn, until
  nothing is missing. Producers are visited in registry order, so the result
  and any insertion error are the same whatever order the steps came in. An
  insertion error is returned as it is. The config uses this to read a
  filled-in step from an empty table (`config/README.md`).
- **No cycles.** The registry admits two steps that each read the other's
  product. `create` orders the steps by Kahn's algorithm, lowest registry
  position first. Steps left over form a cycle and are named in an
  `InvalidConfig`.

Everything is decided before a device is touched: `main` plans before it
builds the node.

## `run`: the scheduling

One `tf::Taskflow`, one task per step, run on `node.executor()`.

| Edge | Ordering |
|---|---|
| producer → consumer | a Taskflow edge |
| device step → next device step | a Taskflow edge, in plan order: the **device chain** |

- **Device steps are chained** (`Device`, `DeviceSubflow`, `Node`): one drives
  the device at a time, which `DeviceResources`' one-driver rule needs and
  which the shared class streams need (orchestration section 7). Host steps
  (`Host`, `HostSubflow`) run beside the chain as soon as their inputs exist.
- **A device step runs on `node.device(0)`** under a `ScopedDeviceIndex`. With
  no device, a pipeline holding one is `GpuRuntime` before any step runs. One
  device for now.
- **The pipeline waits on `res.stream()` after every device step.** A product
  is therefore finished when any consumer, host or device, reads it, and a
  device → host edge needs no event. The wait is small next to the work the
  steps do (diagonalizations, and later Fock builds), and a failed wait is
  the step's error. A `Node` step is followed by a wait on every device. A
  step that enqueues on streams of its own waits on them itself
  (`PrimitivePairClasses`).
- **A subflow step's task is a subflow task**: its `run` gets the `tf::Subflow&`
  and joins it explicitly (`step/README.md`).

**Failure.** A task checks its inputs first. If one failed, the step does not
run, and its slot holds that input's error, unchanged. `run` returns the first
error in plan order. That is always a step's own error, since everything
upstream of it succeeded, so the error is the same whatever the schedule.

**Once.** A second `run` is `Unsupported`: nothing yet re-runs a flow
(geom-opt will).

## Products

`get<T>()` returns the slot of `T`'s producer: the product, the step's error,
or, for a product this pipeline did not plan or has not run, `InvalidConfig`.
Products are immutable, so it hands out `const Result<T>&`.

`step<S>()` returns the planned step, as given or filled in, so a caller can
read the options a product was made with. A loaded step has no options, so
`step<S>()` is empty for it, and `from<S>()` is the path its product is read
from.

**Free order.** The destructor destroys the products in reverse plan order,
with their device current. The run has already waited on every device step.
Products hold their `Device` by `shared_ptr`, so a `Pipeline` may outlive the
`NodeResources` it ran on (`main` builds the node after planning).

A `Pipeline` is movable until it runs (tasks capture `this`), and neither
copyable nor assignable.

## Tests

- `test/gtest/pipeline_pipeline_host.cpp`, over made-up steps, host-only:
  - registries that do not form a `Pipeline`;
  - running in dependency order, whatever order the steps came in;
  - a subflow step, and a `Node` step on a node with no devices;
  - a missing producer, a step given twice, a cycle;
  - insertion, and an insertion error;
  - a loaded step: standing in for its run, planned by its load's inputs and
    shape, not filled in for, given twice, and a failed load failing its
    consumers;
  - a failed step failing its consumers;
  - an unplanned product, a second `run`, and a device step with no device.
- `test/gtest/pipeline_calculation_gpu.cpp`: the real registry from a config
  file, against the same steps called by hand, and a failed root step failing
  every product after it.

## Files

| File | Role |
|------|------|
| `pipeline.cppm` | Module interface: `Pipeline`, and the unexported registry checks |
| `CMakeLists.txt` | The module library |
