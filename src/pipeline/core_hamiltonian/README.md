# Pipeline: CoreHamiltonian

Build the spherical core Hamiltonian H = T + V: the kinetic energy plus the
attraction to the molecule's nuclei.

## Module

`free_breakfast.pipeline.core_hamiltonian`, target
`free_breakfast::pipeline::core_hamiltonian`. It re-exports
`free_breakfast.linalg.matrix_ao` and the two steps whose products it reads, so
an importer can name everything in `run`'s signature without a second import.

## The step

```cpp
struct CoreHamiltonian {
  SphericalMatrixAO h;  // nao_sphe square
};

struct CoreHamiltonianStep {
  static constexpr std::string_view kName = "core_hamiltonian";

  Result<CoreHamiltonian> run(View<Molecule, BasisSet, SphericalTransform, PrimitivePairClasses> in,
                              DeviceResources &res) const;
};
```

A thin wrapper over `integrals.one_electron`'s `core_hamiltonian` and
`linalg.matrix_ao`'s `cartesian_to_spherical`, with their errors: a class above
d is `InvalidConfig` (refused before anything is launched), and a failed launch
or BLAS call is `GpuRuntime`.

- **One product, `CoreHamiltonian`, not T and V** (design section 8).
  `core_hamiltonian` forms T + V in place, with no T or V matrix of its own. T
  and V become products only when a consumer needs one, such as a virial ratio
  or an energy breakdown.
- **Spherical across the step boundary**, as for `Overlap`. The kernels build
  Cartesian H into a transient `MatrixAOPair` over the shared U, and only the
  spherical matrix is moved out and published.
- **`Molecule` is read for its device nuclei**, which V's kernels read. The
  Molecule step's own device contract is what makes them safe to read from every
  class's stream (`molecule/README.md`).
- **`Device`-shaped.** Everything is queued on `res.stream()`; each class's
  launch is forked from it and joined back. No host synchronization.
- **No options.** The config fills it in when a step it plans reads its
  product; an empty table asks for it directly (`pipeline.config`).

## Tests

`test/gtest/pipeline_core_hamiltonian_gpu.cpp`:

- The step's contract is checked by `static_assert`.
- The step's H is checked, element by element, against a direct
  `core_hamiltonian` then `cartesian_to_spherical` for water.
- A basis with an f shell is checked to come back as `InvalidConfig`, through
  the step's `Result`.

## Files

| File | Role |
|------|------|
| `core_hamiltonian.cppm` | Module interface: `CoreHamiltonian`, `CoreHamiltonianStep` |
| `CMakeLists.txt` | The module library |
