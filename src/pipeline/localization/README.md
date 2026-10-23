# Pipeline: Localization

Localize the SCF's occupied orbitals, by the method a `LocalizationMode`
names. Pipek-Mezey is the only one so far; Foster-Boys is to follow.

## Module

`free_breakfast.pipeline.localization`, target
`free_breakfast::pipeline::localization`. It re-exports
`free_breakfast.pipeline.scf`, `free_breakfast.localization.pipek_mezey` and
`free_breakfast.core.wwr`, so an importer can name the product, the options and
everything in `run`'s signature without a second import.

## The step

```cpp
enum class LocalizationMode : uint8_t { PipekMezey };

struct Localization {
  Buffers::device<double> c;  // nao_sphe x n_occ, column-major: C_occ U
  Buffers::device<double> u;  // n_occ x n_occ
  int32_t n_occ;              // electron_count() / 2
  PipekMezeyInfo info;        // sweeps, functional before and after
};

struct LocalizationStep {
  static constexpr std::string_view kName = "localization";

  LocalizationMode mode = LocalizationMode::PipekMezey;
  PipekMezeyOptions pipek_mezey;  // max_sweeps, tolerances, exponent

  Result<Localization> run(View<Molecule, BasisSet, Overlap, Scf> in, DeviceResources &res) const;
};
```

A thin wrapper over the method's localizer, with the same errors: for
`PipekMezey`, `localization.pipek_mezey`'s `pipek_mezey_localize`. `Scf::c` is
column-major with the occupied orbitals first, so its first `n_occ` columns
are passed as they are: no transpose and no copy. The Scf has already refused
an open shell, so `n_occ` is half the electrons.

The options are read from a `[localization]` table (`pipeline.config`):
`mode`, `"pipek_mezey"` by default and the only value so far, then
Pipek-Mezey's own `max_sweeps`, `functional_tolerance`,
`rotation_gain_threshold`, and `exponent`, 2 or 4.

Like `Orthogonalizer`, it synchronizes `res.stream()` inside `run`: once a
sweep, to read the functional for the convergence test.

## Tests

`test/gtest/pipeline_localization_gpu.cpp`: the step's contract by
`static_assert`, and water in STO-3G and cc-pVDZ and NH4+ in cc-pVDZ through
the real registry from a config naming `[localization]`, for both Pipek-Mezey
exponents, against PySCF (`test/gtest/data/localization/`): the functional to
1e-8 and each orbital's atomic populations to 1e-5; and the localized orbitals
are S-orthonormal and give back the SCF's density.

## Files

| File | Role |
|------|------|
| `localization.cppm` | Module interface: `LocalizationMode`, `Localization`, `LocalizationStep` |
| `CMakeLists.txt` | The module library |
