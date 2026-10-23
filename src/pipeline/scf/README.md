# Pipeline: Scf

Iterate closed-shell restricted Hartree-Fock to self-consistency, from the
`Guess`, with Pulay DIIS on the Fock matrix.

## Module

`free_breakfast.pipeline.scf`, target `free_breakfast::pipeline::scf`. It
re-exports `free_breakfast.linalg.matrix_ao` and the `orthogonalizer`,
`core_hamiltonian` and `guess` steps, which between them re-export every other
product in `run`'s signature. `run` is defined in the implementation unit, so
J, K, DIIS and the BLAS calls it uses are PRIVATE links.

## The step

```cpp
struct Scf {
  double energy;                    // total: electronic + nuclear repulsion (hartree)
  int32_t iterations;               // Fock builds
  std::vector<double> mo_energies;  // ascending, host
  SphericalMatrixAO c;              // AO x MO
  SphericalMatrixAO d;              // spin-summed, 2 C_occ C_occ^T
};

struct ScfStep {
  static constexpr std::string_view kName = "scf";

  int32_t max_iterations = 100;
  double energy_tolerance = 1e-10;     // |dE| between two Fock builds
  double commutator_tolerance = 1e-7;  // ||X^T (F D S - S D F) X||_F
  int32_t diis_history = 8;

  Result<Scf> run(View<Molecule, BasisSet, SphericalTransform, PrimitivePairClasses,
                       Overlap, Orthogonalizer, CoreHamiltonian, Guess> in,
                  DeviceResources &res) const;
};
```

Each iteration, all on `res.stream()`:

1. **Fock build**: D's Cartesian form (`spherical_to_cartesian`), J and K from
   it (`coulomb`, `exchange` at their default screening), G = J - K/2 carried
   onto the spherical layout, F = H + G.
2. **Energy**: E = tr(D (H + F))/2 + E_nuc.
3. **Residual**: e = X^T (F D S - S D F) X, and its Frobenius norm. S D F is
   (F D S)^T for symmetric matrices, so it is one product and a `geam`.
4. **Converged** when |dE| < `energy_tolerance` and the norm is under
   `commutator_tolerance`, from the second iteration on.
5. Otherwise **DIIS** extrapolates F in place from the (F, e) history.
6. **Diagonalize** F' = X^T F X (`syevd`), C = X C', and D = 2 C_occ C_occ^T.

On convergence step 5 is skipped: the published orbitals are the canonical
ones of the Fock matrix as built, not of an extrapolation, and D is theirs.
The energy is that of the density the last Fock matrix was built from.

What it settled:

- **One coarse step owns the loop** (design section 14). D, F, the DIIS
  history and the J and K classes are its private state, not products. The
  classes are rebuilt from `PrimitivePairClasses` per run
  (`coulomb_pair_classes`, `ordered_pair_classes`), as design section 8 has
  them until their scratch is refactored away.
- **`syevd`'s work is `res.device_workspace()`'s**, fetched in the
  diagonalization and dropped when it returns, since nothing reads it after.
  Everything else the loop holds lives across calls (DIIS writes into `tmp`,
  `eps` is read at convergence), so it stays in the step's own buffers. The
  Orthogonalizer has already grown the workspace past `syevd`'s size for the
  same n, so in a pipeline the SCF adds no allocation for it.
- **The first device consumer of `Guess`**, so the host-to-device upload is
  here. The planner syncs after every device step and a host step has finished
  when its task does, so the Taskflow edge the `View` gives is all the ordering
  the upload needs.
- **The guess is rescaled to the molecule's electron count**: SAD places
  sum Z electrons whatever the charge, so a cation or anion's guess is scaled by
  N / sum Z before the first Fock build.
- **Closed shell only.** An odd electron count or a multiplicity other than 1
  is `Unsupported`.
- **Host synchronizations**: the energy's two dots and the residual norm read
  host scalars, DIIS reads its singular flag, and `syevd`'s flag is fetched:
  five per iteration. Batching them into one read is a later optimization.
- **A dropped overlap mode** (an eigenvalue under the orthogonalizer's floor)
  leaves a zero column in X, so F' has a spurious zero eigenvalue that the
  aufbau fill can pick. A canonical orthogonalizer with fewer columns than AOs
  is the fix, once a basis needs it.

Errors: an option out of range, more electron pairs than orbitals, or a guess
of the wrong size is `InvalidConfig`; a class above d is J's and K's
`InvalidConfig`; no convergence within `max_iterations` is
`MaxIterationsExceeded`; a non-converged eigensolve is `Numerical`; a failed
runtime, BLAS, solver or launch call is `GpuRuntime`.

## Tests

`test/gtest/pipeline_scf_gpu.cpp`, against PySCF's RHF in
`test/gtest/data/scf/rhf.json` (`tools/emit_scf_reference.py`), which ran over
this tree's own `data/basis` files:

- The step's contract is checked by `static_assert`.
- H2O in STO-3G and cc-pVDZ, and NH4+ (charged, so the guess is rescaled) in
  cc-pVDZ, each over the setup steps and the guess run by hand: the energy to
  1e-8 Eh and every orbital energy to 1e-6 Eh of PySCF's, and the invariants
  C^T S C = I, tr(D S) = N and D S D = 2 D.
- A config naming only `[scf]` plans the whole chain, and its energy matches
  PySCF's.
- An open shell is `Unsupported`, too few iterations `MaxIterationsExceeded`,
  and an option out of range `InvalidConfig`.
- The `[scf]` config cases are in `pipeline_config_host.cpp`.

## Files

| File | Role |
|------|------|
| `scf.cppm` | Module interface: `Scf`, `ScfStep` |
| `scf.cpp` | `ScfStep::run`: the Fock build, the energy, the residual, the eigensolve |
| `CMakeLists.txt` | The module library |
