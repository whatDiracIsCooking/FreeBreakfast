# Pipeline: Casci

Complete active space CI over the SCF's orbitals: the lowest singlets of an
active space, named by index or picked by AVAS over MINAO, its Hamiltonian
built integral-direct, the CI solved over the GPU sigma, each root's 1-RDM,
and the lowest root's natural orbitals as Molden.

## Module

`free_breakfast.pipeline.casci`, target `free_breakfast::pipeline::casci`. It
re-exports `free_breakfast.pipeline.scf` (which re-exports every product in
`run`'s signature) and `free_breakfast.recipes.davidson` (`DavidsonInfo`, in
the product). `run` is defined in the implementation unit, so
`integrals.active_space`, J's and K's classes, the `ci` modules, and AVAS's
`io.basis` (MINAO), `overlap_mixed_spherical` and `diagonalize`, and
`io.molden`, are PRIVATE links.

## The step

```cpp
struct Casci {
  int32_t ncore, ncas, nelecas;
  SphericalMatrixAO c;               // AO x MO: C R, columns core, active, rest
  std::vector<double> energies;      // per root, total (hartree)
  std::vector<double> spin_squares;  // <S^2> per root
  std::vector<double> vectors;       // root r at [r dim, (r + 1) dim)
  std::vector<double> rdm1;          // root r at [r ncas^2, (r + 1) ncas^2), row-major
  DavidsonInfo davidson;
};

struct CasciStep {
  static constexpr std::string_view kName = "casci";

  int32_t ncas = 0;             // required
  int32_t nelecas = 0;          // required
  int32_t nroots = 1;
  std::vector<int32_t> active;  // 0-based MO indices; empty: the frontier orbitals
  std::vector<std::string> avas;     // MINAO AO labels; not empty: AVAS picks the above
  double avas_threshold = 0.2;
  std::filesystem::path molden{};    // the lowest root's natural orbitals, if set

  Result<std::vector<int32_t>> orbital_order(int32_t n_electrons, int32_t nmo) const;
  Result<Casci> run(View<Molecule, BasisSet, SphericalTransform, PrimitivePairClasses,
                         CoreHamiltonian, Scf> in,
                    DeviceResources &res) const;
};
```

`run`, all on `res.stream()`:

1. **The orbitals**, as a rotation R of the Scf's: `Casci::c` = C R, one
   gemm. By index, R is `orbital_order`'s permutation, so C R's columns are
   C's exactly (each element is one C element times 1 plus exact zeros).
   `orbital_order` gives the Scf's orbital for each column:
   the core, the (N − nelecas) / 2 lowest orbitals not active, ascending; the
   active ones, as `active` names them or the ncas above the core; the rest,
   ascending. That is PySCF's `sort_mo`, except that `active` is 0-based
   (`sort_mo`'s default base is 1). The Scf's C is copied column by column into
   that order, and published as `Casci::c`: the 1-RDMs are over its columns
   [ncore, ncore + ncas). Under `avas`, R is AVAS's (below).
2. **The Hamiltonian.** `active_space_hamiltonian` over those orbitals, at its
   own default screens (`tau_j` 1e-14; see `integrals/active_space/`), from J
   and K classes built from `PrimitivePairClasses` per run, as the Scf does.
   E_core holds the core's energy and the nuclear repulsion, so every energy
   is a total.
3. **The CI.** h_eff and (tu|vw) are read back and reordered row-major,
   `CiSigma` (the GPU sigma) is built over the strings of nelecas / 2
   electrons of each spin, its diagonal taken on the device, and `ci_solve`
   finds the `nroots` lowest roots at its defaults: spin penalty 1 hartree,
   so they are singlets, and residual 1e-12 relative.
4. **The 1-RDMs**, one `rdm1` per root, on the host.
5. **The Molden file**, if `molden` is set: the lowest root's natural
   orbitals, C R with R's active block rotated to its 1-RDM's eigenvectors
   by descending occupation; the core at occupation 2, the rest at 0. Each
   orbital's energy is its expectation in the Scf's Fock matrix,
   Σ_k R_kj² ε_k, as the Localization step's Molden file gives its orbitals.

## AVAS

`avas` names MINAO AOs (Sayfutyarova et al., JCTC 13, 4063 (2017)), and the
selection is PySCF's `mcscf.avas` at its defaults for a closed shell: ncore 0,
no IAOs, `openshell_option` 2 (which for a closed shell is no option at all),
canonicalized.

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
  within the virtual orbitals, so a degenerate set of the Scf's is never split.
- **Canonical spaces.** Core, active and the rest are each rotated to
  diagonalize the Scf's Fock matrix within them, diag(ε) in the MO basis, as
  PySCF's `canonicalize` does (its `symmetrize` is a no-op without point
  group symmetry). The energy is invariant to that within core and active;
  the virtual space's rotation only orders the published orbitals.
- **Where it runs.** The overlaps, and S(MINAO, basis) C, are built on the
  device. Everything after them is over a few dozen orbitals, so it is assembled on the host and each
  of the six symmetric eigenproblems (S2, the two blocks of P, the three
  spaces' Fock) solved by `diagonalize`.

What it settled:

- **Singlets only.** The Scf is closed-shell, so N is even, and with an even
  core count nelecas is too: the sector is N_α = N_β, whose lowest spin is 0.
  A triplet CASCI is the sector N_α − N_β = 2, a `spin` option, once a caller
  needs it.
- **The range checks are `orbital_order`'s, not the config's.** Whether an
  active space fits needs the molecule's electrons and the basis's orbitals,
  which a config does not have until the steps run. `orbital_order` takes them
  as integers, so the checks are testable on the host, and `run` calls it
  first. A config reads only types: `ncas` and `nelecas` required,
  integers; `nroots` an integer; `active` an array of integers. Or, in their
  place, `avas` a non-empty array of strings, refused beside `ncas`,
  `nelecas` or `active`, with `avas_threshold` a number; and `molden` a path.
  The labels are parsed, and the threshold's range checked, when the step
  runs, against the molecule's MINAO.
- **The orbitals are published,** not only their order: the AVAS selection
  (#228) rotates the Scf's orbitals rather than picking them, and a 1-RDM
  needs the orbitals it is over.
- **Host synchronizations:** `active_space_hamiltonian`'s own, one to read the
  integrals back, one for the diagonal, and `ci_solve`'s, several per Davidson
  iteration (the penalty and the preconditioner are on the host); under
  `avas`, one for the overlaps and C and one per eigenproblem; with
  `molden`, one for C and one for the 1-RDM's eigenproblem.

Errors: under `avas`, InvalidConfig for a label not of the form, a label
naming no MINAO AO of the molecule, a threshold outside (0, 1), `ncas`,
`nelecas` or `active` set beside it, or no orbital selected; Numerical for an
S2 not positive definite. Otherwise `orbital_order`'s InvalidConfig (ncas below 1; nelecas negative, above
2 ncas or above N; an odd core electron count; ncore + ncas above nmo; `active`
not ncas distinct indices in [0, nmo)); `ci_solve`'s InvalidConfig for
`nroots` outside [1, dim], MaxIterationsExceeded, or Numerical for a root of
the wrong spin; the module calls' own errors (`overlap_mixed_spherical`'s,
`diagonalize`'s, `write_molden`'s among them); GpuRuntime for a failed copy or gemm.

## Tests

`test/gtest/pipeline_casci_gpu.cpp`, against PySCF's `mcscf.CASCI` over its own
RHF in this tree's basis files (`test/gtest/data/casci/`,
`tools/emit_casci_reference.py`), through the real registry from a config
naming `[casci]`, its Scf at the defaults: H₂O STO-3G and
cc-pVDZ CAS(4,4) and N₂ cc-pVDZ CAS(6,6) on the frontier orbitals, and H₂O
cc-pVDZ CAS(4,4) on an explicit, non-contiguous set. No active space splits a
degenerate set (N₂'s π pairs go in whole). For the lowest three singlets:

- the energy and ⟨S²⟩ to 1e-8;
- the natural occupations to 1e-7, for each root of no degenerate level;
- each 1-RDM symmetric with trace nelecas, and each CI vector of unit norm;
- the orbital energies to 1e-6, since the frontier pick is by index, and
  `Casci::c` the Scf's columns in `orbital_order`'s order, exactly.

The AVAS cases are against PySCF's `mcscf.avas` with the same labels and
threshold, over this tree's MINAO, then `mcscf.CASCI`: H₂O cc-pVDZ
`["O 2p", "H 1s"]` (CAS(8,6)), and N₂ cc-pVDZ `["N 2s", "N 2p"]` (CAS(10,8))
and `["N 2p"]` at 0.5 (CAS(8,6)). N₂'s CAS(10,8) compares its lowest singlet
only: `ci_solve` at its defaults converges the ground state in 36 Davidson
iterations, but no block of 2 to 5 roots within its 100. Compared as above, with ncore, ncas and
nelecas equal to PySCF's; the orbital checks are the index path's only. The
reference refuses a case with a weight within 1e-3 of the threshold. With
`molden`, H₂O's natural orbitals read back as `io_molden_host.cpp` reads a
file: every orbital written, at occupation 2, PySCF's natural occupations,
then 0.

Also an active space past the orbitals, and an AVAS label naming no MINAO AO,
refused when the step runs. The
`[casci]` parsing (the `avas` keys too) and every `orbital_order` refusal are in
`pipeline_config_host.cpp`, and an end-to-end run against the step called by
hand in `pipeline_calculation_gpu.cpp`.

## Files

| File | Role |
|------|------|
| `casci.cppm` | Module interface: `Casci`, `CasciStep` |
| `casci.cpp` | `orbital_order`, the AVAS selection, and `run`: C R, the Hamiltonian, the CI solve, the 1-RDMs, the Molden file |
| `CMakeLists.txt` | The module library |
