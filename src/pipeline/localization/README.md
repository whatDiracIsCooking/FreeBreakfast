# Pipeline: Localization

Localize the SCF's occupied orbitals, by the method a `LocalizationMode`
names: Pipek-Mezey, or intrinsic bond orbitals. In either mode it can also
build the localized intrinsic valence virtual orbitals (LIVVOs).

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

struct LocalizationOutput {
  Buffers::device<double> c;  // nao_sphe x n_occ, column-major: C_occ U
  Buffers::device<double> u;  // n_occ x n_occ
  int32_t n_occ;              // ScfOutput::n_alpha
  localization::Info info;        // sweeps, functional before and after
  std::vector<double> populations{}; // IBO only: natoms x n_occ IAO populations of the IBOs
  std::vector<double> charges{};     // IBO only: the IAO partial charges, one per atom
  std::vector<BondOrbital> bonds{};  // IBO only: each IBO's centers and their s/p/d character
  Buffers::device<double> livvo;     // LIVVO only: nao_sphe x n_vvo, column-major
  int32_t n_vvo = 0;                 // LIVVO only: MINAO's nao_sphe - n_occ
  localization::Info livvo_info{};       // LIVVO only: their sweeps and functional
  std::vector<BondOrbital> livvo_bonds{};   // LIVVO only: each LIVVO's centers and sigma/pi kind
  std::vector<double> valence_character{};  // LIVVO only: t_val of each canonical virtual
};

struct LocalizationStep {
  static constexpr std::string_view kName = "localization";

  LocalizationMode mode = LocalizationMode::PipekMezey;
  localization::PmOptions pipek_mezey;  // max_sweeps, tolerances, exponent
  std::filesystem::path molden{}; // if set, the localized orbitals are written here
  bool livvo = false;             // the LIVVOs too, in either mode
  std::filesystem::path livvo_molden{}; // if set (needs livvo), the LIVVOs are written here

  Result<LocalizationOutput> run(View<MoleculeOutput, BasisSetOutput, OverlapOutput, ScfOutput> in, DeviceResources &res) const;
};
```

A thin wrapper over the method's localizer, with the same errors: for
`PipekMezey`, `localization.pipek_mezey`'s `localization::pm_localize`; for `Ibo`,
`localization.ibo`'s `localization::ibo_localize`, over MINAO (`resolve_basis("minao")`)
placed over the Molecule here, so an element MINAO lacks is `place`'s error.
Both take `pipek_mezey`'s options, since IBO is the same sweep over IAO
populations. `ScfOutput::c` is
column-major with the occupied orbitals first, so its first `n_occ` columns
are passed as they are: no transpose and no copy. `n_occ` is the ScfOutput's
`n_alpha`, and an open shell, `n_alpha != n_beta`, is `Unsupported` before
anything is allocated.

**The localized orbitals come out in ascending order of energy.** The
localizer's own order is whatever its Jacobi sweeps leave, so the step sorts
`c` and `u` by each orbital's Fock expectation value, `<i|F|i> = sum_k U_ki^2
eps_k`, the diagonal of `Uᵀ diag(eps_occ) U` (`recipes.blas`'s `gmmwich`). It
uses `recipes.sort`'s stable `merge_sort`, so orbitals of equal energy keep
the localizer's order, and its `gather_columns` permutes the columns of both,
so `c = C_occ u` still holds. It runs before anything reads the orbitals, so
`populations`, `bonds`, the log and the Molden file all use the sorted order:
cores first, then bonds, then lone pairs up to the HOMO.

**IBO mode also reports the IAO charges and the bonds.** `localization::ibo_localize` hands
back the IAOs it ran over, and `analysis.population`'s `orbital_weights`
gives each IBO's weight on each of them, once. From those:
`populations`, each IBO's population on each atom (their fourth powers sum
to `info.functional`); `charges`, Knizia's eq 3; and `bonds`,
`analysis.bond`'s reading of each IBO as a lone pair, bond or multi-center
bond. The charges and the bonds are logged at info level, one atom or IBO
per line:

```
localization: IAO charge   0 O  -0.470624
localization: IBO   0 sigma lone  O0 100.0% (s 99% p 1%)
localization: IBO   1 sigma bond  O0 61.8% (s 22% p 78%)  H1 38.2% (s 100%)
localization: IBO   3 sigma lone  O0 100.0% (s 56% p 44%)
localization: IBO   4 pi    lone  O0 100.0% (p 100%)
```

(Water in STO-3G, in the energy order above: the O 1s, the two O-H bonds,
then the lone pairs, the p one the HOMO.)

("3c+" marks three centers or more.) Pipek-Mezey has no IAOs, and leaves all
three empty.

The options are read from a `[localization]` table (`pipeline.config`):
`mode`, `"pipek_mezey"` by default or `"ibo"`, then the sweep's
`max_sweeps`, `functional_tolerance`, `rotation_gain_threshold`, and
`exponent`, 2 or 4 (2 by default for `"pipek_mezey"`, Knizia's 4 for
`"ibo"`), and `livvo`, `false` by default.

**`molden` writes the localized orbitals.** When it is set, `c` and `u` are
downloaded and written by `io.molden`'s `write_molden`: the `n_occ` localized
orbitals at occupation 2, then the SCF's canonical virtuals at occupation 0,
so the file holds every MO, as viewers and readers expect. The localization
mixes only occupied orbitals, so the set is still orthonormal and spans the
same space. The localized orbitals are not eigenvectors of F, so each is
given its Fock expectation value, `<i|F|i> = sum_k U_ki^2 eps_k` over the
SCF's occupied orbital energies; F is block-diagonal between occupied and
virtual, so each virtual keeps its own `eps_a`. The localized orbitals are
already sorted by that energy, and each is a weighted mean of occupied ones,
so it is at most the HOMO's: the whole file ascends. It is not a TOML key: `pipeline/config` sets
it to `localization.molden` in the run's output directory. A failed write
fails the step.

**`livvo` adds the LIVVOs**, in either mode, from
`localization.livvo` (its README has the formulas). MINAO is then resolved
and placed whichever mode runs, and the IAOs are built once: `localization::ibo_localize`
hands them back in IBO mode, and `intrinsic_atomic_orbitals` builds them from
the canonical occupied orbitals in Pipek-Mezey mode (the IAOs depend only on
the occupied space). From them, `valence_virtual_orbitals` and then
`localization::livvo_localize` give `livvo`, its `n_vvo = m - n_occ` columns. The sweep takes
`pipek_mezey`'s sweep count and tolerances at Knizia's exponent four, whatever
exponent the occupied orbitals ran at. Then:

- `livvo_bonds`: `bond_orbitals` over the LIVVOs' IAO weights
  (`orbital_weights`), with each orbital's sigma/pi kind;
- `valence_character`: `valence_character` of each of the SCF's canonical
  virtuals, in its order; they sum to `n_vvo`.

All logged at info level: the LIVVOs' sweeps and functional, one line per
LIVVO through the same reader as the IBOs', and one per virtual:

```
localization: 2 LIVVOs in 4 sweeps, L 0.119750 -> 0.438610
localization: LIVVO   0 sigma bond  H1 67.6% (s 100%)  O0 32.4% (s 28% p 72%)
localization: LIVVO   1 sigma bond  H2 67.6% (s 100%)  O0 32.4% (s 28% p 72%)
localization: virtual   5 eps +0.185476 t_val 0.7231
localization: virtual   6 eps +0.256181 t_val 0.5357
```

(Water in cc-pVDZ, Pipek-Mezey mode.)

`populations`, `charges` and `bonds` stay **IBO only** even when LIVVO places
MINAO in Pipek-Mezey mode: they describe the occupied IBOs.

**`livvo_molden` writes the LIVVOs**, at occupation 0. Each is given its Fock
expectation value over the canonical virtuals, `sum_a (C_vir^T S L)_al^2
eps_a`: the LIVVOs lie wholly in the virtual space, so that is all of
`<l|F|l>`, a weighted mean of the virtual energies. C_vir^T S L is two gemms
over the device workspace, then downloaded. `pipeline/config` sets it to
`localization_livvo.molden` in the run's output directory when `livvo` is set,
and to nothing otherwise; a hand-built step setting it without `livvo` is
refused, `InvalidConfig`, before anything runs.

Like `OrthogonalizerStep`, it synchronizes `res.stream()` inside `run`: once a
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
reading each molecule's lone pairs and X-H bonds (`analysis/bond`'s README).
In both modes, water in cc-pVDZ: the localized orbitals ascend in
`<i|F|i>`, and `c` is still `C_occ u` column for column. With `molden` set, the
file holds every MO, `n_occ` at occupation 2 and the rest at 0, in ascending
order; the occupied energies sum to the SCF's occupied ones (U is orthogonal)
and each virtual's is the SCF's own.
With `livvo = true`, in both modes, water in cc-pVDZ: two LIVVOs, each a
sigma bond on O and one hydrogen, a different one each, the virtuals' valence
characters each in [0, 1] and summing to 2, and `populations`, `charges` and
`bonds` empty in Pipek-Mezey mode; the LIVVO Molden file holds `n_vvo`
orbitals at occupation 0 with energies inside the canonical virtual range;
and a hand-built `livvo_molden` without `livvo` is `InvalidConfig`.

## Files

| File | Role |
|------|------|
| `localization.cppm` | Module interface: `LocalizationMode`, `LocalizationOutput`, `LocalizationStep` |
| `localization.cpp` | `LocalizationStep::run`, and the two Molden files |
| `CMakeLists.txt` | The module library |
