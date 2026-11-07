# Pipeline: SphericalTransform

Upload the basis's Cartesian-to-spherical transform U, once, for every
AO-matrix step to share.

## Module

`free_breakfast.pipeline.spherical_transform`, target
`free_breakfast::pipeline::spherical_transform`. It re-exports
`free_breakfast.recipes.matrix_ao`, since every consumer of the product builds a
`MatrixAOPair` over it.

## The step

```cpp
struct SphericalTransformOutput {
  std::shared_ptr<const Buffers::device<double>> u;  // nullptr: U is the identity
};

struct SphericalTransformStep {
  static constexpr std::string_view kName = "spherical_transform";

  std::filesystem::path npy{};  // U written here as .npy if not empty

  Result<SphericalTransformOutput> run(View<BasisSetOutput> in, DeviceResources &res) const;
};
```

A thin wrapper over `recipes.matrix_ao`'s `upload_spherical_transform_matrix`:
U is `nao_cart x nao_sphe`, column-major, and `nullptr` when no shell is above p
(or the basis is empty), exactly as there. A failed upload is `GpuRuntime`; an
`npy` file that cannot be written is `write_npy`'s.

- **The product is the shared buffer.** Section 8's rule is that every matrix
  product is spherical and Cartesian is a step's internal working form. The
  integral steps (`Overlap`, `CoreHamiltonian`) build Cartesian in a transient
  `MatrixAOPair` over this U and publish only the spherical matrix. U is
  never written after the upload, so those steps may share it across streams.
- **`Device`-shaped**, because U lives on the device.
- **`npy`, off by default.** If not empty, U is written there as a float64
  `.npy`, `nao_cart x nao_sphe` in Fortran order, which a plain `np.load`
  reads. It is written from the host build, `basis`'s
  `spherical_transform_matrix`, not downloaded: that needs no device sync, and
  it gives an explicit identity where the product is `nullptr`, so the file
  always exists when asked for. The config sets it to `spherical_transform.npy`
  in the output directory for `[spherical_transform] write_npy = true`
  (`pipeline.config`). The config fills the step in when a step it plans reads
  its product, from an empty table, so U is not written unless the table asks.
- **`run` lives in `spherical_transform.cpp`**, so the host U and the `.npy`
  writer stay private to the module.

## The device contract it does not keep

A device `run` should only enqueue on `res.stream()` and never synchronize the
host (`step/README.md`). `upload_spherical_transform_matrix` synchronizes
`res.stream()` once, when there is a U to upload, so that the host staging copy
can be released. That is one small blocking copy per basis, on a step that runs
once per calculation. The staging copy is released inside `run`, so the
pipeline's own wait after the step cannot replace it; it stays until events
replace those waits, as for `Molecule`'s nuclei.

## Tests

`test/gtest/pipeline_spherical_transform_gpu.cpp`:

- The step's contract is checked by `static_assert`, in `test/pipeline/steps_host.cpp`.
- The step's U is checked, element by element, against a direct
  `upload_spherical_transform_matrix` of the same basis.
- A basis with no shell above p is checked to give `nullptr`.
- The `.npy` file is checked against the step's U, and against an explicit
  identity for a basis with no shell above p.

## Files

| File | Role |
|------|------|
| `spherical_transform.cppm` | Module interface: `SphericalTransformOutput`, `SphericalTransformStep` |
| `spherical_transform.cpp` | `SphericalTransformStep::run`, and the `.npy` write |
| `CMakeLists.txt` | The module library |
