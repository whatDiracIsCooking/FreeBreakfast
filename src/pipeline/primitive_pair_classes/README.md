# Pipeline: PrimitivePairClasses

Build the basis's McMurchie-Davidson primitive-pair records on the device, once,
for every integral step to launch over.

## Module

`free_breakfast.pipeline.primitive_pair_classes`, target
`free_breakfast::pipeline::primitive_pair_classes`. It re-exports
`free_breakfast.integrals.mmd`, so an importer can name `PrimitivePairClass`
without a second import.

## The step

```cpp
struct PrimitivePairClassesOutput {
  std::vector<PrimitivePairClass> classes;  // ascending (la, lb), each on its own stream
};

struct PrimitivePairClassesStep {
  static constexpr std::string_view kName = "primitive_pair_classes";

  Result<PrimitivePairClassesOutput> run(View<BasisSetOutput> in, DeviceResources &res) const;
};
```

A thin wrapper over `integrals.mmd`'s `primitive_pair_classes`: one class per
(la, lb) the basis reaches, records formed on the device in descending zeta
order, each with its bounds. A failed copy, launch or wait is `GpuRuntime`,
exactly as there.

- **The coefficients are the basis's own.** `primitive_pair_classes` takes them
  as given, so the `BasisSet` step places a primitive-normalized stamp: the
  BSE coefficients are for normalized primitives.
- **`Device`-shaped.** Section 11 expected `DeviceSubflow`, a subflow for
  writing the records in parallel on the host. `primitive_pair_classes` forms
  them on the device instead (the host only lists shells and shell pairs), so
  the step asks for the narrower shape.
- **No options.** The config fills it in when a step it plans reads its
  product; an empty table asks for it directly (`pipeline.config`).
- **One product, not one per class.** The classes are read together by every
  build, so a type per class would only add names.

## Which streams it uses

A device `run` should only enqueue on `res.stream()` (`step/README.md`), and
the pipeline waits on `res.stream()` after it. This step enqueues nothing
there: each class is built on a new `DeviceStream` of its own, and every build
launches that class on that same stream, which is what orders a build after its
records.

So **every class's stream is waited on before the step returns**, which the
pipeline's wait on `res.stream()` would not cover. `primitive_pair_classes`
does that wait itself (its shell tables are its own, freed when it returns), so
the step adds none. A consumer may then read a class from any stream. It costs
one wait per class, once per geometry, against builds that each take far
longer.

## Tests

`test/gtest/pipeline_primitive_pair_classes_gpu.cpp`:

- The step's contract is checked by `static_assert`, in `test/pipeline/steps_host.cpp`.
- The step's classes are checked against a direct `primitive_pair_classes` of
  the same basis: the same (la, lb) and record count per class, and the same
  records, every field bit for bit.
- An empty basis is checked to give no classes.

## Files

| File | Role |
|------|------|
| `primitive_pair_classes.cppm` | Module interface: `PrimitivePairClassesOutput`, `PrimitivePairClassesStep` |
| `CMakeLists.txt` | The module library |
