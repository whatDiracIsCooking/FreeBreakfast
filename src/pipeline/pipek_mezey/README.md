# Pipeline: PipekMezey

Localize the SCF's occupied orbitals by Pipek-Mezey.

## Module

`free_breakfast.pipeline.pipek_mezey`, target
`free_breakfast::pipeline::pipek_mezey`. It re-exports
`free_breakfast.pipeline.scf`, `free_breakfast.localization.pm` and
`free_breakfast.core.wwr`, so an importer can name the product, the options and
everything in `run`'s signature without a second import.

## The step

```cpp
struct PipekMezey {
  Buffers::device<double> c;  // nao_sphe x n_occ, column-major: C_occ U
  Buffers::device<double> u;  // n_occ x n_occ
  int32_t n_occ;              // electron_count() / 2
  PipekMezeyInfo info;        // sweeps, functional before and after
};

struct PipekMezeyStep {
  static constexpr std::string_view kName = "pipek_mezey";

  PipekMezeyOptions options;  // max_sweeps, tolerances, exponent

  Result<PipekMezey> run(View<Molecule, BasisSet, Overlap, Scf> in, DeviceResources &res) const;
};
```

A thin wrapper over `localization.pm`'s `pipek_mezey_localize<double>`, with
the same errors. `Scf::c` is column-major with the occupied orbitals first, so
its first `n_occ` columns are passed as they are: no transpose and no copy.
The Scf has already refused an open shell, so `n_occ` is half the electrons.

The options are read from a `[pipek_mezey]` table (`pipeline.config`):
`max_sweeps`, `functional_tolerance`, `rotation_gain_threshold`, and
`exponent`, 2 or 4.

Like `Orthogonalizer`, it synchronizes `res.stream()` inside `run`: once a
sweep, to read the functional for the convergence test.

## Tests

`test/gtest/pipeline_pipek_mezey_gpu.cpp`: the step's contract by
`static_assert`, and water in STO-3G through the real registry from a config
naming `[pipek_mezey]`, for both exponents: the functional rises, the localized
orbitals are S-orthonormal, and they give back the SCF's density.

## Files

| File | Role |
|------|------|
| `pipek_mezey.cppm` | Module interface: `PipekMezey`, `PipekMezeyStep` |
| `CMakeLists.txt` | The module library |
