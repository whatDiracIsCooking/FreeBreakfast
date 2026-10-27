# Pipeline: Overlap

Build the spherical AO overlap matrix S.

## Module

`free_breakfast.pipeline.overlap`, target `free_breakfast::pipeline::overlap`.
It re-exports `free_breakfast.recipes.matrix_ao` and the two steps whose products
it reads, so an importer can name everything in `run`'s signature without a
second import.

## The step

```cpp
struct OverlapOutput {
  SphericalMatrixAO s;  // nao_sphe square
};

struct OverlapStep {
  static constexpr std::string_view kName = "overlap";

  Result<OverlapOutput> run(View<BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput> in,
                      DeviceResources &res) const;
};
```

A thin wrapper over `integrals.one_electron`'s `overlap` and `recipes.matrix_ao`'s
`cartesian_to_spherical`. Its errors are theirs: a failed launch or BLAS call is
`GpuRuntime`, and a class above f is `InvalidConfig`.

- **Spherical across the step boundary** (section 8 of the design). The kernels
  build Cartesian S into a transient `MatrixAOPair` over the shared U, which
  `cartesian_to_spherical` carries onto the spherical layout. Only the spherical
  matrix is moved out and published. The Cartesian matrix and the pair's
  scratch are freed when `run` returns, on `res.stream()`'s order, so nothing
  waits for them.
- **`Device`-shaped.** Everything is queued on `res.stream()`; each class's
  launch is forked from it and joined back (`launch_each`). No host
  synchronization.
- **`BasisSet` is read for the orders only**: the pair is sized from it.
- **No options.** The config fills it in when a step it plans reads its
  product; an empty table asks for it directly (`pipeline.config`).

## Tests

`test/gtest/pipeline_overlap_gpu.cpp`:

- The step's contract is checked by `static_assert`.
- The step's S is checked, element by element, against a direct `overlap` then
  `cartesian_to_spherical` over the same basis.
- S of a BSE basis read through the `BasisSet` step, with one primitive per
  shell, is checked to have a unit diagonal. That holds only if the step's
  primitives are normalized and U carries the d shell onto normalized solid
  harmonics.

## Files

| File | Role |
|------|------|
| `overlap.cppm` | Module interface: `OverlapOutput`, `OverlapStep` |
| `CMakeLists.txt` | The module library |
