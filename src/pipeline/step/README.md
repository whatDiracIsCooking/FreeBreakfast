# Pipeline step

The contract a pipeline step is written against. A step is a name and one
`run`, and the `run` signature declares the rest: what the step reads, the one
product it writes, and where it runs.

## Module

`free_breakfast.pipeline.step`, target `free_breakfast::pipeline::step`.
It imports `core.types`, `core.error_handling` (`Result`) and `core.resources`
(`DeviceResources`, `NodeResources`, and through them `tf::Subflow`), and
nothing from integrals or linalg: a leaf, so a new method never recompiles it.
Templates only, so there is no implementation unit.

| Export | What |
|---|---|
| `View<Ts...>` | A step's read-only inputs, one `const T&` per declared type |
| `RunShape` | Where a step runs: the five resource lists a `run` may take |
| `PipelineStep<S>` | The concept every step satisfies |
| `StepInput<S>`, `StepOutput<S>`, `kRunShape<S>` | What `S` reads, writes and holds, read off its `run` |

## A step

```cpp
struct OrthogonalizerStep {
  static constexpr std::string_view kName = "orthogonalizer";

  double eigenvalue_floor = kOrthogonalizeEigenvalueFloor<double>;  // an option

  Result<Orthogonalizer> run(View<BasisSet, Overlap> in, DeviceResources &res) const;
};
```

| Part | Requirement | Why |
|---|---|---|
| `kName` | a non-empty constant convertible to `std::string_view` | the TOML table name; the pipeline checks names unique across its steps at compile time, so it must be a constant |
| `run`'s first parameter | a `View<Ts...>`, by value or reference: `StepInput<S>` | the graph's edges |
| `run`'s return type | `Result<T>`, `T` a product not among the `Ts`: `StepOutput<S>` is `T` | one result slot per step, so an error belongs to exactly one product; a step reading its own output is a cycle the compiler can already see |
| `run`'s other parameters | exactly one of the five `RunShape` lists: `kRunShape<S>` | where the step runs (below) |
| `run` itself | `const`, not overloaded, not a template; may be `noexcept` | `const`: a step is configuration, not changed by running. One signature, so there is one thing to read |

**The signature is the declaration.** There is no `Requires` or `Output`
alias beside `run`. Each would only restate a type that is already in the
signature, and could disagree with it. That is the same reason there is no
`domain` constant (see `RunShape` below). The pipeline reads the types back
through `StepInput`, `StepOutput` and `kRunShape`, which take apart the type of
`&S::run`. Taking that address is also what rejects an overloaded or template
`run`.

**Options are the step's own data members**, defaulted. A root step takes
`View<>`, and its options are its inputs: atoms, a basis stamp.

**No central store.** A step never holds a writable store. Its one write is
the product `run` returns, which the pipeline stores in the step's own slot.
Its reads are the `View`, which the pipeline builds from its producers' slots.

## Products and `View`

A product is any non-`const`, move-constructible object type. It need not be
copyable, since products own device buffers. `View<Overlap, Overlap>`,
`View<const A>`, `View<A&>` and `View<void>` do not compile.

**Immutable once published.** `View::get` hands out `const T&` only.

**A product never borrows from another product.** A `View` holds pointers into
the producers' slots and lives for one `run` call. A product owns its data, or
shares it as `std::shared_ptr<const ...>`.

**Only what was declared.** `get<T>()` is constrained to the view's own `Ts`,
so an undeclared read is a compile error rather than a run-time lookup that
fails. That is the whole defence against the god-object context. Because it is
a `requires` clause and not a failure in the body, the negative test is a plain
`static_assert`:

```cpp
template <class V, class T>
concept CanGet = requires(const V &view) { view.template get<T>(); };
static_assert(!CanGet<View<A>, B>);
```

The concept wrapper is not decoration. A `requires` expression that is not in a
template, and whose requirement is ill-formed, makes the program ill-formed
rather than evaluating to `false`.

## `RunShape`: the signature says where a step runs

`run`'s resource parameters are the declaration, so they cannot disagree with
what the step uses: `DeviceResources&` is the capability to use the device, and
a step not handed one cannot reach it.

| `run(View, ...)` | `RunShape` | Scheduling | Example |
|---|---|---|---|
| nothing extra | `Host` | free (host) | `BasisSet` (`place`) |
| `tf::Subflow&` | `HostSubflow` | free (host), parallel inside | `GuessStep` (SAD, a subflow per stamp) |
| `DeviceResources&` | `Device` | on that device's chain | `PrimitivePairClasses`, `Overlap`, `Orthogonalizer` |
| `DeviceResources&, tf::Subflow&` | `DeviceSubflow` | on that device's chain, host-parallel inside | none yet: `PrimitivePairClasses`, were its records written in parallel |
| `NodeResources&` | `Node` | holds every device's chain | a multi-GPU J build |

The types must match exactly. A `tf::FlowBuilder&` or a
`const DeviceResources&` is not a shape, so each table row stays one
spelling.

Each step asks for the narrowest resource that does the job. `Node` is the
escape hatch: safe, and paid for only in concurrency.

### Contracts the types cannot check

- A device `run` only enqueues on `res.stream()`. It does not synchronize the
  host and does not select the device: the pipeline opens a
  `ScopedDeviceIndex` around it and waits on `res.stream()` after it
  (`pipeline/README.md`).
- A host `run` never touches the device. A host product the GPU needs is
  uploaded by the device step that consumes it.
- With a device resource and a subflow, one thread drives a device at a time
  inside the step too: the step's own task, or one subtask at a time per
  device. The other subtasks do host work only.
- A subflow holding references into the task's frame is joined explicitly
  before `run` returns (SAD's lesson).

## Files

Each partition holds one unexported block of helpers, then one exported block.
The partitions fall where the helpers' dependencies do: the signature helpers
in `:pipeline_step` name both `View` and `RunShape`, so those two are declared
first, in partitions of their own.

| File | Role |
|------|------|
| `step.cppm` | Primary interface: re-exports the three partitions |
| `view.cppm` | `:view`: `View`, and what may be a product |
| `run_shape.cppm` | `:run_shape`: `RunShape` |
| `pipeline_step.cppm` | `:pipeline_step`: `PipelineStep`, `StepInput`, `StepOutput`, `kRunShape`, and the signature reading behind them |
| `CMakeLists.txt` | The module library |

Tests are in `test/gtest/pipeline_step_host.cpp`: made-up steps, mostly
`static_assert`s, with no device.
