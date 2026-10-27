# Pipeline: Casci

Complete active space CI over the SCF's orbitals: the lowest singlets of an
active space, named by index or picked by AVAS over MINAO, its Hamiltonian
built integral-direct, the CI solved over the GPU sigma, each root's 1-RDM,
and the lowest root's natural orbitals as Molden.

## Module

`free_breakfast.pipeline.casci`, target `free_breakfast::pipeline::casci`. It
re-exports `free_breakfast.pipeline.scf` (which re-exports every product in
`run`'s signature), `free_breakfast.pipeline.active_space` (the selection,
`CasciStep`'s base), and `free_breakfast.recipes.davidson` (`DavidsonInfo`, in
the product). `run` is defined in the implementation unit, so
`integrals.active_space`, J's and K's classes, the `ci` modules,
`diagonalize` and `io.molden` are PRIVATE links.

**The active-space selection** -- `ncas`, `nelecas`, `active`, `avas`,
`avas_threshold`, `orbital_order`, AVAS over MINAO, and the rotation C R -- is
`pipeline/active_space`'s, shared with the steps that take one; see
[`active_space/README.md`](../active_space/README.md). This step adds the CI.

## The step

```cpp
struct CasciOutput {
  int32_t ncore, ncas, nelecas;
  SphericalMatrixAO c;               // AO x MO: C R, columns core, active, rest
  std::vector<double> energies;      // per root, total (hartree)
  std::vector<double> spin_squares;  // <S^2> per root
  std::vector<double> vectors;       // root r at [r dim, (r + 1) dim)
  std::vector<double> rdm1;          // root r at [r ncas^2, (r + 1) ncas^2), row-major
  DavidsonInfo davidson;
};

struct CasciStep : ActiveSpaceSelection {  // ncas, nelecas, active, avas, avas_threshold
  static constexpr std::string_view kName = "casci";

  int32_t nroots = 1;
  std::filesystem::path molden{};    // the lowest root's natural orbitals, if set

  Result<CasciOutput> run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput,
                         CoreHamiltonianOutput, ScfOutput> in,
                    DeviceResources &res) const;
};
```

`run`, all on `res.stream()`:

1. **The orbitals**: `ActiveSpaceSelection::select`'s `ActiveSpace`, C R
   with R a permutation by index or AVAS's rotation
   ([`active_space/README.md`](../active_space/README.md)), published as
   `CasciOutput::c`: the 1-RDMs are over its columns [ncore, ncore + ncas).
2. **The Hamiltonian.** `active_space_hamiltonian` over those orbitals, at its
   own default screens (`tau_j` 1e-14; see `integrals/active_space/`), from J
   and K classes built from `PrimitivePairClassesOutput` per run, as the Scf step does.
   E_core holds the core's energy and the nuclear repulsion, so every energy
   is a total.
3. **The CI.** h_eff and (tu|vw) are read back and reordered row-major,
   `ci::Sigma` (the GPU sigma) is built over the strings of nelecas / 2
   electrons of each spin, its diagonal taken on the device, and `ci::solve`
   finds the `nroots` lowest roots at its defaults: spin penalty 1 hartree,
   so they are singlets, and residual 1e-12 relative.
4. **The 1-RDMs**, one `ci::Sigma::rdm1` per root, on the device.
5. **The Molden file**, if `molden` is set: the lowest root's natural
   orbitals, C R with R's active block rotated to its 1-RDM's eigenvectors
   by descending occupation; the core at occupation 2, the rest at 0. Each
   orbital's energy is its expectation in the ScfOutput's Fock matrix,
   Σ_k R_kj² ε_k, as the Localization step's Molden file gives its orbitals.

What it settled:

- **Singlets only.** The SCF is closed-shell, so N is even, and with an even
  core count nelecas is too: the sector is N_α = N_β, whose lowest spin is 0.
  A triplet CASCI is the sector N_α − N_β = 2, a `spin` option, once a caller
  needs it.
- **The range checks are the selection's,** when the step runs; a config
  reads only types: the selection's keys as `active_space/README.md` says,
  and `nroots` an integer. `molden` is not a key: `pipeline/config` sets it
  to `casci.molden` in the run's output directory.
- **Host synchronizations:** the selection's (under `avas`),
  `active_space_hamiltonian`'s own, one to read the integrals back, and
  `ci::solve`'s: its guesses' Gram-Schmidt dots, a few per Davidson iteration
  to read the Ritz values and residual norms, and one for the energies and
  ⟨S²⟩s (the diagonal, the guesses, the penalty, the preconditioner and the
  roots stay on the device); one for the 1-RDMs and the vectors; with
  `molden`, one for C and one for the 1-RDM's eigenproblem.

Errors: `ActiveSpaceSelection::select`'s (`active_space/README.md`);
`ci::solve`'s InvalidConfig for `nroots` outside [1, dim],
MaxIterationsExceeded, or Numerical for a root of the wrong spin; the module
calls' own errors (`diagonalize`'s and `write_molden`'s among them);
GpuRuntime for a failed copy or gemm.

## Tests

`test/gtest/pipeline_casci_gpu.cpp`, against PySCF's `mcscf.CASCI` over its own
RHF in this tree's basis files (`test/gtest/data/casci/`,
`tools/emit_casci_reference.py`), through the real registry from a config
naming `[casci]`, its Scf step at the defaults: H₂O STO-3G and
cc-pVDZ CAS(4,4) and N₂ cc-pVDZ CAS(6,6) on the frontier orbitals, and H₂O
cc-pVDZ CAS(4,4) on an explicit, non-contiguous set. No active space splits a
degenerate set (N₂'s π pairs go in whole). For the lowest three singlets:

- the energy and ⟨S²⟩ to 1e-8;
- the natural occupations to 1e-7, for each root of no degenerate level;
- each 1-RDM symmetric with trace nelecas, and each CI vector of unit norm;
- the orbital energies to 1e-6, since the frontier pick is by index, and
  `CasciOutput::c` the ScfOutput's columns in `orbital_order`'s order, exactly.

The AVAS cases are against PySCF's `mcscf.avas` with the same labels and
threshold, over this tree's MINAO, then `mcscf.CASCI`: H₂O cc-pVDZ
`["O 2p", "H 1s"]` (CAS(8,6)), and N₂ cc-pVDZ `["N 2s", "N 2p"]` (CAS(10,8))
and `["N 2p"]` at 0.5 (CAS(8,6)). N₂'s CAS(10,8) compares its lowest singlet
only: `ci::solve` at its defaults converges the ground state in 36 Davidson
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
| `casci.cppm` | Module interface: `CasciOutput`, `CasciStep` |
| `casci.cpp` | `run`: the selection, the Hamiltonian, the CI solve, the 1-RDMs, the Molden file |
| `CMakeLists.txt` | The module library |
