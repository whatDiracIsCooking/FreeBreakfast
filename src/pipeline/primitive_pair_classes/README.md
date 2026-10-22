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
struct PrimitivePairClasses {
  std::vector<PrimitivePairClass> classes;  // ascending (la, lb), each on its own stream
};

struct PrimitivePairClassesStep {
  static constexpr std::string_view kName = "primitive_pair_classes";

  Result<PrimitivePairClasses> run(View<BasisSet> in, DeviceResources &res) const;
};
```

A thin wrapper over `integrals.mmd`'s `primitive_pair_classes`: one class per
(la, lb) the basis reaches, records in descending zeta order, each record's
bounds formed on the device. A failed copy is `GpuRuntime`, exactly as there.

- **The coefficients are the basis's own.** `primitive_pair_classes` takes them
  as given, so the `BasisSet` step places a primitive-normalized stamp: the
  BSE coefficients are for normalized primitives.
- **`Device`-shaped.** Section 11 expected `DeviceSubflow`, a subflow for
  writing the records in parallel on the host. `primitive_pair_classes` writes
  them on one thread, so the step asks for the narrower shape. It widens when
  the record writing does.
- **No options**, so no TOML table: `main` constructs it.
- **One product, not one per class.** The classes are read together by every
  build, so a type per class would only add names.

## Which streams it uses

A device `run` should only enqueue on `res.stream()` (`step/README.md`). This
step enqueues nothing there: each class is built on a new `DeviceStream` of its
own, and every build launches that class on that same stream, which is what
orders a build after its records. It never synchronizes the host.

That ordering lives in the class, not in the pipeline's device chain, which
holds while every consumer launches a class on its own stream. A consumer that
reads a class's device side from another stream must first wait on the class's
stream.

## Tests

`test/gtest/pipeline_primitive_pair_classes_gpu.cpp`:

- The step's contract is checked by `static_assert`.
- The step's classes are checked against a direct `primitive_pair_classes` of
  the same basis: the same (la, lb) and record count per class, and the same
  device-side records, field by field.
- An empty basis is checked to give no classes.

## Files

| File | Role |
|------|------|
| `primitive_pair_classes.cppm` | Module interface: `PrimitivePairClasses`, `PrimitivePairClassesStep` |
| `CMakeLists.txt` | The module library |
