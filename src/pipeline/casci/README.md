# Pipeline: Casci

Complete active space CI over the SCF's canonical orbitals: the lowest
singlets of an active space, its Hamiltonian built integral-direct, the CI
solved over the GPU sigma, and each root's 1-RDM.

## Module

`free_breakfast.pipeline.casci`, target `free_breakfast::pipeline::casci`. It
re-exports `free_breakfast.pipeline.scf` (which re-exports every product in
`run`'s signature) and `free_breakfast.recipes.davidson` (`DavidsonInfo`, in
the product). `run` is defined in the implementation unit, so
`integrals.active_space`, J's and K's classes and the `ci` modules are PRIVATE
links.

## The step

```cpp
struct Casci {
  int32_t ncore, ncas, nelecas;
  SphericalMatrixAO c;               // AO x MO: the Scf's C, columns core, active, rest
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

  Result<std::vector<int32_t>> orbital_order(int32_t n_electrons, int32_t nmo) const;
  Result<Casci> run(View<Molecule, BasisSet, SphericalTransform, PrimitivePairClasses,
                         CoreHamiltonian, Scf> in,
                    DeviceResources &res) const;
};
```

`run`, all on `res.stream()`:

1. **The orbitals.** `orbital_order` gives the Scf's orbital for each column:
   the core, the (N − nelecas) / 2 lowest orbitals not active, ascending; the
   active ones, as `active` names them or the ncas above the core; the rest,
   ascending. That is PySCF's `sort_mo`, except that `active` is 0-based
   (`sort_mo`'s default base is 1). The Scf's C is copied column by column into
   that order, and published as `Casci::c`: the 1-RDMs are over its columns
   [ncore, ncore + ncas).
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
  integers; `nroots` an integer; `active` an array of integers.
- **The orbitals are published,** not only their order: the AVAS selection
  (#228) rotates the Scf's orbitals rather than picking them, and a 1-RDM
  needs the orbitals it is over.
- **Host synchronizations:** `active_space_hamiltonian`'s own, one to read the
  integrals back, one for the diagonal, and `ci_solve`'s, several per Davidson
  iteration (the penalty and the preconditioner are on the host).

Errors: `orbital_order`'s InvalidConfig (ncas below 1; nelecas negative, above
2 ncas or above N; an odd core electron count; ncore + ncas above nmo; `active`
not ncas distinct indices in [0, nmo)); `ci_solve`'s InvalidConfig for
`nroots` outside [1, dim], MaxIterationsExceeded, or Numerical for a root of
the wrong spin; the module calls' own errors; GpuRuntime for a failed copy.

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

Also an active space past the orbitals refused when the step runs. The
`[casci]` parsing and every `orbital_order` refusal are in
`pipeline_config_host.cpp`, and an end-to-end run against the step called by
hand in `pipeline_calculation_gpu.cpp`.

## Files

| File | Role |
|------|------|
| `casci.cppm` | Module interface: `Casci`, `CasciStep` |
| `casci.cpp` | `orbital_order`, and `run`: the column reorder, the Hamiltonian, the CI solve, the 1-RDMs |
| `CMakeLists.txt` | The module library |
