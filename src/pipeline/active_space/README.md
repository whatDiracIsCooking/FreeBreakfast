# Pipeline: active space

The active-space selection the steps over the ScfOutput's orbitals share (`Casci`;
`Casscf` to come): by index, or by AVAS over MINAO, and the rotation of the
ScfOutput's orbitals it gives. Not a step: no `kName`, no product, no `[table]`.

## Module

`free_breakfast.pipeline.active_space`, target
`free_breakfast::pipeline::active_space`. It re-exports
`free_breakfast.pipeline.scf` (`Molecule`, `BasisSet`, `ScfOutput`,
`SphericalMatrixAO`). `select` is defined in the implementation unit, so
AVAS's `io.basis` (MINAO), `overlap_mixed_spherical` and `diagonalize` are
PRIVATE links.

## The selection

```cpp
struct ActiveSpace {
  int32_t ncore, ncas, nelecas;  // nelecas = N - 2 ncore
  std::vector<double> r;         // R, nmo x nmo, column-major, orthogonal
  SphericalMatrixAO c;           // C R, AO x MO: core, active, rest
};

struct ActiveSpaceSelection {
  int32_t ncas = 0;                  // required without avas
  int32_t nelecas = 0;               // required without avas
  std::vector<int32_t> active;       // 0-based MO indices; empty: the frontier orbitals
  std::vector<std::string> avas;     // MINAO AO labels; not empty: AVAS picks the above
  double avas_threshold = 0.2;

  Result<std::vector<int32_t>> orbital_order(int32_t n_electrons, int32_t nmo) const;
  Result<ActiveSpace> select(DeviceResources &res, const Molecule &molecule,
                             const BasisSet &basis, const ScfOutput &scf) const;
};
```

A step's config **derives** from `ActiveSpaceSelection` (`struct CasciStep :
ActiveSpaceSelection`), so its fields read as the step's own (`step.ncas`), and
the TOML keys are the same in every table that takes them:
`pipeline/config`'s `read_active_space` reads them once for each step.

`select`, on `res.stream()`, builds the orbitals as a rotation R of the
ScfOutput's: `ActiveSpace::c` = C R, one gemm. By index, R is `orbital_order`'s
permutation, so C R's columns are C's exactly (each element is one C element
times 1 plus exact zeros). `orbital_order` gives the ScfOutput's orbital for each
column: the core, the (N − nelecas) / 2 lowest orbitals not active,
ascending; the active ones, as `active` names them or the ncas above the
core; the rest, ascending. That is PySCF's `sort_mo`, except that `active` is
0-based (`sort_mo`'s default base is 1). Under `avas`, R is AVAS's (below).
R is returned as well as C R: a natural-orbital Molden file rotates R's
active block, and the energy it prints per orbital is Σ_k R_kj² ε_k.

**An open shell** (ROHF, `n_alpha > n_beta`) is taken by index. N is
`n_alpha + n_beta`, the ScfOutput's first `n_beta` orbitals are doubly
occupied and the next `n_alpha − n_beta` singly, and the order must keep
them apart: every core orbital among the doubly occupied ones, and every
singly occupied one active. Either broken is `InvalidConfig`, naming the
orbital. With both kept the active space holds `n_alpha − ncore` alpha and
`n_beta − ncore` beta electrons, whose difference is the SCF's 2S, as
PySCF's `mcscf.CASCI(mf, ncas, (n_α, n_β))` over its ROHF has it. The frontier
default keeps both whenever `nelecas` ≥ `n_alpha − n_beta` (the core stops at
`n_beta`) and ncore + ncas ≥ `n_alpha` (the active space reaches the last
singly occupied orbital). A closed
shell's order is `orbital_order`'s alone, as before: `sort_mo` does not ask
whether a core it fills reaches past the occupied orbitals, and neither does
this.

## AVAS

`avas` names MINAO AOs (Sayfutyarova et al., JCTC 13, 4063 (2017)), and the
selection is PySCF's `mcscf.avas` at its defaults: ncore 0, no IAOs,
`openshell_option` 2, canonicalized.

**An open shell** (ROHF, `n_alpha > n_beta`) is taken as `openshell_option`
2 takes it: the occupied block is the first `n_alpha` orbitals, the singly
occupied ones projected together with the doubly occupied, so either kind
can land in the core or the active space by its weight, and nelecas is N −
2 ncore, `n_alpha − ncore` alpha and `n_beta − ncore` beta. So the core is
not checked to be doubly occupied, as the index path's is: a core orbital
can mix in a singly occupied one (CH₂'s 1a1 takes 0.017 of its 3a1), and
the Casci and Casscf steps then treat it as doubly occupied, as PySCF's
`mcscf.CASCI` over the AVAS orbitals does. PySCF's option 3 (the singly
occupied orbitals active outright, only the doubly occupied projected) is
not offered: one option is PySCF's default, and a second would be a key
nothing asks for yet. A core that took more than `n_beta` orbitals would
leave a negative beta count; the Casci step's and `mcscf::solve`'s
multiplicity checks refuse that (2S above nelecas).

- **The labels** are `<symbol> <n><l>`: every component of that shell of
  MINAO on every atom of the element, n counted from l + 1 over the atom's
  shells of that l in the file's order, as PySCF's `ao_labels` counts them
  (MINAO's oxygen is 1s, 2s, 2p). That is the form PySCF's own examples use;
  its labels are regular expressions over `ao_labels`, which this does not
  try to be. A label naming no AO of the molecule is refused rather than
  ignored: it is a typo or a missing element, never a request for nothing.
- **The projector.** MINAO is `resolve_basis("minao")`, placed over the
  molecule; S21 (labelled AO by MO) and S2 (labelled AO by labelled AO) are
  the labelled rows of `overlap_mixed_spherical`'s MINAO-by-basis S times C,
  and of its MINAO-by-MINAO S. P = S21ᵀ S2⁻¹ S21, through S2's
  eigendecomposition. Neither basis's AO normalization matters: P is
  invariant to scaling a labelled AO.
- **The selection.** P's occupied block and its virtual block are each
  diagonalized; an orbital of weight ≥ `avas_threshold` is active, an
  occupied one below it core, a virtual one below it the rest. So nelecas =
  N − 2 ncore. The weights are invariant to rotations within the occupied and
  within the virtual orbitals, so a degenerate set of the ScfOutput's is never split.
- **Canonical spaces.** Core, active and the rest are each rotated to
  diagonalize the ScfOutput's Fock matrix within them, diag(ε) in the MO basis, as
  PySCF's `canonicalize` does (its `symmetrize` is a no-op without point
  group symmetry). The energy is invariant to that within core and active;
  the virtual space's rotation only orders the published orbitals.
- **Where it runs.** The overlaps, and S(MINAO, basis) C, are built on the
  device. Everything after them is over a few dozen orbitals, so it is
  assembled on the host and each of the six symmetric eigenproblems (S2, the
  two blocks of P, the three spaces' Fock) solved by `diagonalize`.

What it settled:

- **The range checks are `orbital_order`'s, not the config's.** Whether an
  active space fits needs the molecule's electrons and the basis's orbitals,
  which a config does not have until the steps run. `orbital_order` takes them
  as integers, so the checks are testable on the host, and `select` calls it
  first. A config reads only types: `ncas` and `nelecas` required, integers;
  `active` an array of integers. Or, in their place, `avas` a non-empty array
  of strings, refused beside `ncas`, `nelecas` or `active`, with
  `avas_threshold` a number. The labels are parsed, and the threshold's range
  checked, when `select` runs, against the molecule's MINAO.
- **The orbitals are published,** not only their order: AVAS rotates the
  ScfOutput's orbitals rather than picking them, and an RDM needs the orbitals it
  is over.
- **Host synchronizations:** none by index; under `avas`, one for the
  overlaps and C and one per eigenproblem.

Errors, each message prefixed `active space:`: under `avas`, InvalidConfig
for a label not of the form, a label naming no MINAO AO of the molecule, a
threshold outside (0, 1), `ncas`, `nelecas` or `active` set beside it, or no
orbital selected; Numerical for an S2 not positive definite. Otherwise
`orbital_order`'s InvalidConfig (ncas below 1; nelecas negative, above 2 ncas
or above N; an odd core electron count; ncore + ncas above nmo; `active` not
ncas distinct indices in [0, nmo)), and for an open shell by index
InvalidConfig for a core orbital not doubly occupied or a singly occupied one
not active; `overlap_mixed_spherical`'s and
`diagonalize`'s own errors; GpuRuntime for a failed copy or gemm.

## Tests

Through the Casci step, which is the selection's one caller so far:
`test/gtest/pipeline_casci_gpu.cpp` checks the index path's C R against the
ScfOutput's columns in `orbital_order`'s order, exactly, and the AVAS cases' ncore,
ncas and nelecas against PySCF's `mcscf.avas` (`casci/README.md` has the
cases), over ROHF orbitals too, a doublet and a triplet, whose active
orbitals are compared up to sign as well; and the open shell's two refusals
by index, over a hand-built ScfOutput. `pipeline_config_host.cpp` has the `[casci]` selection keys and every
`orbital_order` refusal.

## Files

| File | Role |
|------|------|
| `active_space.cppm` | Module interface: `ActiveSpace`, `ActiveSpaceSelection` |
| `active_space.cpp` | `orbital_order`, the AVAS selection, and `select`: R and C R |
| `CMakeLists.txt` | The module library |
