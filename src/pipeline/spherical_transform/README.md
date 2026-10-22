# Pipeline: SphericalTransform

Upload the basis's Cartesian-to-spherical transform U, once, for every
AO-matrix step to share.

## Module

`free_breakfast.pipeline.spherical_transform`, target
`free_breakfast::pipeline::spherical_transform`. It re-exports
`free_breakfast.linalg.matrix_ao`, since every consumer of the product builds a
`MatrixAOPair` over it.

## The step

```cpp
struct SphericalTransform {
  std::shared_ptr<const Buffers::device<double>> u;  // nullptr: U is the identity
};

struct SphericalTransformStep {
  static constexpr std::string_view kName = "spherical_transform";

  Result<SphericalTransform> run(View<BasisSet> in, DeviceResources &res) const;
};
```

A thin wrapper over `linalg.matrix_ao`'s `upload_spherical_transform_matrix`:
U is `nao_cart x nao_sphe`, column-major, and `nullptr` when no shell is above p
(or the basis is empty), exactly as there. A failed upload is `GpuRuntime`.

- **The product is the shared buffer.** Section 8's rule is that every matrix
  product is spherical and Cartesian is a step's internal working form. The
  integral steps (`Overlap`, `CoreHamiltonian`) build Cartesian in a transient
  `MatrixAOPair` over this U and publish only the spherical matrix. U is
  never written after the upload, so those steps may share it across streams.
- **`Device`-shaped**, because U lives on the device.
- **No options**, so the step has no TOML table: `main` constructs it.

## The device contract it does not keep

A device `run` should only enqueue on `res.stream()` and never synchronize the
host (`step/README.md`). `upload_spherical_transform_matrix` synchronizes
`res.stream()` once, when there is a U to upload, so that the host staging copy
can be released. That is one small blocking copy per basis, on a step that runs
once per calculation. It is kept until the pipeline (stage 3) has an event to
hand out instead, as for `Molecule`'s nuclei.

## Tests

`test/gtest/pipeline_spherical_transform_gpu.cpp`:

- The step's contract is checked by `static_assert`.
- The step's U is checked, element by element, against a direct
  `upload_spherical_transform_matrix` of the same basis.
- A basis with no shell above p is checked to give `nullptr`.

## Files

| File | Role |
|------|------|
| `spherical_transform.cppm` | Module interface: `SphericalTransform`, `SphericalTransformStep` |
| `CMakeLists.txt` | The module library |
