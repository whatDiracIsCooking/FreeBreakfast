# Pipeline: Localization

Localize the SCF's occupied orbitals, by the method a `LocalizationMode`
names: Pipek-Mezey, or intrinsic bond orbitals.

## Module

`free_breakfast.pipeline.localization`, target
`free_breakfast::pipeline::localization`. It re-exports
`free_breakfast.pipeline.scf`, `free_breakfast.localization.ibo` (and through
it `free_breakfast.localization.pipek_mezey`),
`free_breakfast.analysis.population`, `free_breakfast.analysis.bond` and
`free_breakfast.core.wwr`, so an importer can name the product, the options and
everything in `run`'s signature without a second import.

## The step

```cpp
enum class LocalizationMode : uint8_t { PipekMezey, Ibo };

struct Localization {
  Buffers::device<double> c;  // nao_sphe x n_occ, column-major: C_occ U
  Buffers::device<double> u;  // n_occ x n_occ
  int32_t n_occ;              // electron_count() / 2
  PipekMezeyInfo info;        // sweeps, functional before and after
  std::vector<double> populations{}; // IBO only: natoms x n_occ IAO populations of the IBOs
  std::vector<double> charges{};     // IBO only: the IAO partial charges, one per atom
  std::vector<BondOrbital> bonds{};  // IBO only: each IBO's centers and their s/p/d character
};

struct LocalizationStep {
  static constexpr std::string_view kName = "localization";

  LocalizationMode mode = LocalizationMode::PipekMezey;
  PipekMezeyOptions pipek_mezey;  // max_sweeps, tolerances, exponent
  std::filesystem::path molden{}; // if set, the localized orbitals are written here

  Result<Localization> run(View<Molecule, BasisSet, Overlap, Scf> in, DeviceResources &res) const;
};
```

A thin wrapper over the method's localizer, with the same errors: for
`PipekMezey`, `localization.pipek_mezey`'s `pipek_mezey_localize`; for `Ibo`,
`localization.ibo`'s `ibo_localize`, over MINAO (`resolve_basis("minao")`)
placed over the Molecule here, so an element MINAO lacks is `place`'s error.
Both take `pipek_mezey`'s options, since IBO is the same sweep over IAO
populations. `Scf::c` is
column-major with the occupied orbitals first, so its first `n_occ` columns
are passed as they are: no transpose and no copy. The Scf has already refused
an open shell, so `n_occ` is half the electrons.

**IBO mode also reports the IAO charges and the bonds.** `ibo_localize` hands
back the IAOs it ran over, and `analysis.population`'s `orbital_weights`
gives each IBO's weight on each of them, once. From those:
`populations`, each IBO's population on each atom (their fourth powers sum
to `info.functional`); `charges`, Knizia's eq 3; and `bonds`,
`analysis.bond`'s reading of each IBO as a lone pair, bond or multi-center
bond. The charges and the bonds are logged at info level, one atom or IBO
per line:

```
localization: IAO charge   0 O  -0.470630
localization: IBO   1 bond  O0 61.8% (s 22% p 78%)  H2 38.2% (s 100%)
localization: IBO   3 lone  O0 100.0% (s 56% p 44%)
```

(Water in STO-3G.)

("3c+" marks three centers or more.) Pipek-Mezey has no IAOs, and leaves all
three empty.

The options are read from a `[localization]` table (`pipeline.config`):
`mode`, `"pipek_mezey"` by default or `"ibo"`, then the sweep's
`max_sweeps`, `functional_tolerance`, `rotation_gain_threshold`, and
`exponent`, 2 or 4 (2 by default for `"pipek_mezey"`, Knizia's 4 for
`"ibo"`), and `molden`.

**`molden` writes the localized orbitals.** When it is set, `c` and `u` are
downloaded and written by `io.molden`'s `write_molden`: the `n_occ` localized
orbitals only, each at occupation 2. They are not eigenvectors of F, so each
is given its Fock expectation value, `<i|F|i> = sum_k U_ki^2 eps_k` over the
SCF's occupied orbital energies. It is the `[localization] molden` TOML key,
relative to the config file. A failed write fails the step.

Like `Orthogonalizer`, it synchronizes `res.stream()` inside `run`: once a
sweep, to read the functional for the convergence test.

## Tests

`test/gtest/pipeline_localization_gpu.cpp`: the step's contract by
`static_assert`, and water in STO-3G and cc-pVDZ and NH4+ in cc-pVDZ through
the real registry from a config naming `[localization]`, for both Pipek-Mezey
exponents, against PySCF (`test/gtest/data/localization/`): the functional to
1e-8 and each orbital's atomic populations to 1e-5; and the localized orbitals
are S-orthonormal and give back the SCF's density. The same for `mode = "ibo"`
against PySCF's IBOs, plus the IAOs' orthonormality, span and partial charges
(`localization/ibo`'s README), and the step's `populations` reproducing L and
its `charges` PySCF's (`analysis/population`'s README), and its `bonds`
reading each molecule's lone pairs and X-H bonds (`analysis/bond`'s README). With `molden` set, the
file holds `n_occ` orbitals at occupation 2, and their energies sum to the
SCF's occupied ones (U is orthogonal).

## Files

| File | Role |
|------|------|
| `localization.cppm` | Module interface: `LocalizationMode`, `Localization`, `LocalizationStep` |
| `localization.cpp` | `LocalizationStep::run`, and the Molden file |
| `CMakeLists.txt` | The module library |
