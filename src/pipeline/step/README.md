# Pipeline step

The contract a pipeline step is written against. A step is a name and one
`run`, and the `run` signature declares the rest: what the step reads, the one
product it writes, and where it runs. A step may also have a static `load`,
which reads that product back from a file instead, declared the same way.

## Module

`free_breakfast.pipeline.step`, target `free_breakfast::pipeline::step`.
It imports `core.types`, `core.error_handling` (`Result`) and `core.resources`
(`DeviceResources`, `NodeResources`, and through them `tf::Subflow`), and
nothing from integrals or recipes: a leaf, so a new method never recompiles it.
Templates only, so there is no implementation unit.

| Export | What |
|---|---|
| `View<Ts...>` | A step's read-only inputs, one `const T&` per declared type |
| `RunShape` | Where a step runs: the five resource lists a `run` may take |
| `PipelineStep<S>` | The concept every step satisfies |
| `StepInput<S>`, `StepOutput<S>`, `kRunShape<S>` | What `S` reads, writes and holds, read off its `run` |
| `LoadableStep<S>` | A step with a `load` |
| `LoadInput<S>`, `kLoadShape<S>` | What `S`'s `load` reads and holds |
| `Load<S>` | A request to load `S`'s product from a path, handed to the pipeline in place of an `S` |

## A step

```cpp
struct OrthogonalizerStep {
  static constexpr std::string_view kName = "orthogonalizer";

  double eigenvalue_floor = kOrthogonalizeEigenvalueFloor<double>;  // an option

  Result<OrthogonalizerOutput> run(View<BasisSetOutput, OverlapOutput> in, DeviceResources &res) const;
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
`View<>`, and its options are its inputs: `MoleculeStep`'s XYZ path, charge and
multiplicity.

**No central store.** A step never holds a writable store. Its one write is
the product `run` returns, which the pipeline stores in the step's own slot.
Its reads are the `View`, which the pipeline builds from its producers' slots.

## Products and `View`

A product is any non-`const`, move-constructible object type. It need not be
copyable, since products own device buffers. `View<OverlapOutput, OverlapOutput>`,
`View<const A>`, `View<A&>` and `View<void>` do not compile.

**Every product is its step's own `XOutput` struct**, `XStep` writing
`XOutput`, even when it holds only a domain object (`BasisSetOutput{basis}`).
A `View` finds a product by type, and the pipeline allows one producer per
type, so a step returning a bare `BasisSet` would take that type for the whole
graph: an auxiliary basis, or a geometry step's new `Molecule`, could never be a
second product. The wrapper keeps the domain types free.

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
| `DeviceResources&` | `Device` | on that device's chain | `PrimitivePairClassesOutput`, `OverlapOutput`, `OrthogonalizerOutput` |
| `DeviceResources&, tf::Subflow&` | `DeviceSubflow` | on that device's chain, host-parallel inside | none yet: `PrimitivePairClassesOutput`, were its records written in parallel |
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

## `load`: the product read back instead of made

```cpp
struct GuessStep {
  ...
  static Result<GuessOutput> load(const std::filesystem::path &from, View<BasisSetOutput> in);
};
```

| Part | Requirement | Why |
|---|---|---|
| first parameter | exactly `const std::filesystem::path &` | what to read; the front end resolves it |
| second parameter | a `View<Ts...>`, not holding the product: `LoadInput<S>` | what loading checks the file against, and so the edges a loaded step is planned by |
| return type | `run`'s, `Result<StepOutput<S>>` | one product per step, made either way, so the registry keeps one producer per product |
| other parameters | one of the five `RunShape` lists: `kLoadShape<S>`, not necessarily `run`'s | a product made on the device may be read on the host |
| `load` itself | `static`, not overloaded, not a template; may be `noexcept` | a step's options configure `run`, and a loaded product never ran |

**Loading is a mode of the step, not another step.** A second step
producing the same type would be a second producer, which the registry
refuses, and every loadable product would double the step types. With `load`
on the step, a loadable product costs one static member.

**The pipeline plans a loaded step by its `load`.** Its edges are `load`'s
`View`, so whatever only `run` reads is not planned for it: a loaded `[guess]`
needs the `BasisSet`, not SAD's `Molecule` atoms. `Load<S>` carries only the
path, since the step's options configure a `run` that does not happen.

A front end hands over `Load<S>{path}` in place of an `S`. It is constrained
to `LoadableStep<S>`, so a step with no `load` cannot be asked to load.

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
| `pipeline_step.cppm` | `:pipeline_step`: `PipelineStep`, `StepInput`, `StepOutput`, `kRunShape`, their `load` counterparts and `Load`, and the signature reading behind them |
| `CMakeLists.txt` | The module library |

Tests are in `test/gtest/pipeline_step_host.cpp`: made-up steps, mostly
`static_assert`s, with no device.
