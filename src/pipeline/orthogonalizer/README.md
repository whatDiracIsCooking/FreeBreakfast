# Pipeline: Orthogonalizer

Build the symmetric (Loewdin) orthogonalizer X = S^{-1/2} of the spherical
overlap.

## Module

`free_breakfast.pipeline.orthogonalizer`, target
`free_breakfast::pipeline::orthogonalizer`. It re-exports
`free_breakfast.linalg.matrix_ao` and `free_breakfast.pipeline.overlap`, so an
importer can name everything in `run`'s signature without a second import.

## The step

```cpp
struct Orthogonalizer {
  SphericalMatrixAO x;  // nao_sphe square
};

struct OrthogonalizerStep {
  static constexpr std::string_view kName = "orthogonalizer";

  double eigenvalue_floor = kOrthogonalizeEigenvalueFloor<double>;  // 1e-10

  Result<Orthogonalizer> run(View<BasisSet, Overlap> in, DeviceResources &res) const;
};
```

A thin wrapper over `linalg.orthogonalize`'s `orthogonalize<double>`, with the
same errors: a non-converged eigensolve is `Numerical`, and a failed runtime,
solver or BLAS call is `GpuRuntime`. An overlap eigenvalue at or below
`eigenvalue_floor` drops its mode (it contributes 0 to X). That is not an error.

- **`BasisSet` is read for the order only.** A `MatrixAO` is sized from a
  basis, so X is too. Reading the basis is an honest edge, and cheaper than a
  second way to construct a `MatrixAO`.
- **X is a `SphericalMatrixAO`**: it is `nao_sphe` square even when modes are
  dropped, since a dropped mode is a zero column of U Lambda^{-1/2}, not a
  smaller matrix.
- **One option, `eigenvalue_floor`**, read from an optional `[orthogonalizer]`
  table (`pipeline.config`). It is the first `double` option.

## The device contract it does not keep

A device `run` should only enqueue on `res.stream()` and never synchronize the
host (`step/README.md`). `orthogonalize` synchronizes `res.stream()` once, to
read syevd's convergence flag and the extreme eigenvalues. That is what makes a
non-converged solve an error here rather than a wrong X downstream. It is kept
until the pipeline (stage 3) has an event to hand out instead, as for
`Molecule`'s nuclei.

Its scratch (U, the scaled U, the eigenvalues and syevd's workspace) is carved
from `res.device_workspace()`, which grows if it has to and is never part of a
product.

## Tests

`test/gtest/pipeline_orthogonalizer_gpu.cpp`:

- The step's contract is checked by `static_assert`.
- The step's X is checked, element by element, against a direct
  `orthogonalize` of the same S.
- X^T S X is checked to be the identity on a well-conditioned S.
- A floor above every eigenvalue is checked to drop every mode (X = 0).

## Files

| File | Role |
|------|------|
| `orthogonalizer.cppm` | Module interface: `Orthogonalizer`, `OrthogonalizerStep` |
| `CMakeLists.txt` | The module library |
