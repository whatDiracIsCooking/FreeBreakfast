# Pipeline: Scf

Iterate closed-shell restricted Hartree-Fock to self-consistency, from the
`GuessOutput`, with Pulay DIIS on the Fock matrix. The two-electron part is
integral-direct, with incremental Fock builds, or density-fitted over an aux
basis (`aux_basis`, below).

## Module

`free_breakfast.pipeline.scf`, target `free_breakfast::pipeline::scf`. It
re-exports `free_breakfast.recipes.matrix_ao` and the `orthogonalizer`,
`core_hamiltonian` and `guess` steps, which between them re-export every other
product in `run`'s signature. `run` is defined in the implementation unit, so
J, K, DIIS, the BLAS calls it uses, and under density fitting the
`DensityFitting` step and DF J and K are PRIVATE links.

## The step

```cpp
struct ScfOutput {
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
  std::string aux_basis{};             // empty: direct; else DF over this fitting set (name or path)
  int32_t fock_rebuild_interval = 20;  // every this many builds, one is full; 1: all full
  double tau_j = 1e-10;                // J's screen, coulomb's tau; not a TOML key
  std::filesystem::path molden{};      // if set, the converged orbitals are written here

  Result<ScfOutput> run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput,
                       OverlapOutput, OrthogonalizerOutput, CoreHamiltonianOutput, GuessOutput> in,
                  DeviceResources &res) const;
};
```

Each iteration, all on `res.stream()`:

1. **Fock build**: density-fitted under `aux_basis` (below), or
   integral-direct, full or incremental (below): the Cartesian form
   (`spherical_to_cartesian`) of D or of D - D_built, J and K from it
   (`coulomb` at `tau_j`, `exchange` at its default screening), G = J - K/2 carried
   onto the spherical layout, then F = H + G (full) or F += G (incremental).
   D_built becomes D.
2. **Energy**: E = tr(D (H + F))/2 + E_nuc.
3. **Residual**: e = X^T (F D S - S D F) X, and its Frobenius norm. S D F is
   (F D S)^T for symmetric matrices, so it is one product and a `geam`.
4. **Converged** when |dE| < `energy_tolerance` and the norm is under
   `commutator_tolerance`, from the second iteration on, **on a full build**.
   Once the norm is under `commutator_tolerance`, every build is full.
5. Otherwise **DIIS** extrapolates F from the (F, e) history into a copy; the
   as-built F stays the incremental builds' accumulator. On the incremental
   build that switches to full ones, the history is then dropped.
6. **Diagonalize** F' = X^T F X (`recipes.diagonalize`), C = X C', and D = 2 C_occ C_occ^T.

On convergence step 5 is skipped: the published orbitals are the canonical
ones of the Fock matrix as built, not of an extrapolation, and D is theirs.
The energy is that of the density the last Fock matrix was built from.

What it settled:

- **The loop and the Fock build are apart.** `:loop`'s `rhf_loop` is the
  RHF iteration, written against a Fock builder with two members: `build`,
  which makes F = H + G(D) from the density and the orbitals it came from
  (null on the first build, whose D is the guess) and says whether that F is
  exact for D, so may be taken as converged; and `reset_diis`, told the
  commutator norm, which says whether DIIS's history is to be dropped. The
  incremental builds, `fock_rebuild_interval`, the switch to full builds and
  the DIIS reset are the direct builder's policy (`DirectFockBuilder` in
  `scf.cpp`); the guess upload and rescale, convergence and the Molden write
  are the loop's, so a second builder gets them unchanged. The orbitals are
  passed because a density-fitted K is built from `C_occ`, not from D. The
  loop is a template in an internal partition, not a public one.

- **Incremental Fock builds, by default.** G is linear in D, so
  F(D) = F(D_built) + G(D - D_built). J's screen (`Dmax · Ip · Iq`) and K's
  (`Ip · Iq · F_ab`, `F_ab <= max|D|`, and the reach cut at `G_max · max|D|`)
  both scale with the density they are handed, and the difference shrinks by
  orders of magnitude as the SCF converges, so the late builds skip most of
  their quartets. Each build's skipped terms are each under `tau`, but an
  incremental F accumulates every build's since the last full one: hence a
  full build every `fock_rebuild_interval` (and the first), and convergence
  taken only on a full build, so the published energy and orbitals are a
  full F's, as before. **Once the commutator is converged, every build is
  full.** By then D - D_built is small enough that the screening error is a
  good part of its G, and each incremental build adds a fresh one: on
  H2O/STO-3G the energy wandered 4.5e-10 Eh a build and never met
  `energy_tolerance`. A full F's error is a fixed function of D, so the loop
  keeps a fixed point. **The DIIS history is reset at that switch**, once:
  its F's are the drifted ones, and their error is as large as the residuals
  by then. Resetting any earlier restarts DIIS where it is still needed.
  `fock_rebuild_interval = 1` is the old loop, every build full, kept for an
  A/B.

  Measured on an RTX 3080, cc-pVDZ, from the first Fock build to converged,
  every run to the same energy within 1e-12 Eh:

  | | all full (`= 1`) | incremental, no reset | incremental (this) |
  |---|---|---|---|
  | gly5 (379 AOs) | 19 builds, 268 s | 4 + 18, 221 s | 3 full + 18, 208 s (-23%) |
  | gly10 (734 AOs) | 19 builds, 1062 s | 4 + 18, 857 s (-19%) | not run |

  A late incremental build is 3-3.5x cheaper than a full one. Switching at
  10x `commutator_tolerance` without the reset tied (209 s on gly5); at 10x
  and 100x with it, 226 and 234 s.

- **Density fitting is an ERI option of this step, not a step of its own.**
  With `aux_basis` set, the step builds B = L⁻¹ (P|mn) itself by calling
  `DensityFittingStep::run` over the molecule, basis and pair classes its
  `View` already holds, and `DfFockBuilder` makes F = H + J − K/2 from it:
  J from D (`df_coulomb`), K from the occupied orbitals the density came
  from (`df_exchange`, which takes a factor C of D = 2 C Cᵀ). B is the run's
  own and freed when the loop returns. A second step producing `ScfOutput`
  would need the planner to admit two producers of one product, and
  `ScfOutput` is what localization, CASCI, CASSCF and FCI read; and the step
  does not read `DensityFittingOutput`, because a `View` has no optional
  entries. The standalone `DensityFitting` step is unchanged.
  **Every DF build is full**, so `fock_rebuild_interval` and `tau_j` are not
  read and DIIS is never reset: DF's cost is dense BLAS, which a small ΔD
  does not shrink, and K from orbitals is not a function of ΔD. **The first
  build has no orbitals**: its D is the (rescaled) SAD guess, a sum of atomic
  densities with non-negative occupations, so positive semidefinite. Its K
  is from D's own factor: `syevd(D) = V Λ Vᵀ` (`recipes.diagonalize`), and
  C̃ = V (Λ/2)^{1/2} over the eigenvalues above **1e-12 of the largest**
  (`kDensityFactorFloor`), scaled in place by a `dgmm`. What that drops is
  syevd's round-off, about n·ε of the largest, so K is exact for the guess.
  A density loaded `from` a file need not be PSD; its negative part is left
  out of the first K with a warning, which moves only the first iterate.

  Measured on an RTX 3080 (both cuda slots held, the card otherwise idle),
  gly5 / cc-pVDZ (379 AOs) over cc-pVDZ-JKFIT, `free-breakfast` on a config
  naming `[molecule]`, `[basis_set]` and `[scf]`:

  | | builds | first build's log line to converged | whole run |
  |---|---|---|---|
  | direct (the 208 s above, rerun the same day) | 21 | 180.8 s | 193.7 s |
  | density-fitted | 19 | 5.9 s (0.33 s a build) | 13.2 s |

  The log line lands when the first build is done, so that column leaves the
  first build out; with it, the DF loop is about 6.2 s against the direct
  loop's 208 s.

  B itself (metric, three-center, fit) takes about 6.4 s of the DF run: a
  config planning only `[density_fitting]` ran in 6.85 s against 0.42 s for
  its setup steps alone. The step, B included, is about 12.6 s, some 15x
  faster than the direct SCF. The DF energy, −1110.135779508 Eh, is 2.0 mEh above
  the direct one, −1110.137794144 Eh: that is the fit's error at gly5 in
  cc-pVDZ-JKFIT, not the SCF's.

- **J's screen is an option, `tau_j`, K's is not.** At the default, 1e-10,
  J's screening error is in the converged energy and orbitals: 2.6e-10 Eh on
  N₂/cc-pVDZ, which `test/gtest/active_space_gpu.cpp`'s 1e-10 comparisons
  cannot absorb, so that test runs at 0. K's default screen (1e-13) does not
  bind there, so K has no option. `tau_j` is not a `[scf]` TOML key: that
  would be a change to `pipeline/config`.
- **One coarse step owns the loop** (design section 14). D, F, the DIIS
  history and the J and K classes are its private state, not products. The
  classes are rebuilt from `PrimitivePairClassesOutput` per run
  (`coulomb_pair_classes`, `ordered_pair_classes`), as design section 8 has
  them until their scratch is refactored away.
- **`syevd`'s work is `res.device_workspace()`'s**: `recipes.diagonalize`
  takes it per call and drops it when it returns, since nothing reads it after.
  Everything else the loop holds lives across calls (DIIS writes into `tmp`,
  `eps` is read at convergence), so it stays in the step's own buffers. The
  the Orthogonalizer step has already grown the workspace to `syevd`'s size for the
  same n, so in a pipeline the SCF adds no allocation for it.
- **The first device consumer of `GuessOutput`**, so the host-to-device upload is
  here. The planner syncs after every device step and a host step has finished
  when its task does, so the Taskflow edge the `View` gives is all the ordering
  the upload needs.
- **The guess is rescaled to the molecule's electron count**: SAD places
  sum Z electrons whatever the charge, so a cation or anion's guess is scaled by
  N / sum Z before the first Fock build.
- **`molden` writes on convergence.** When it is set, the converged C is
  downloaded and written by `io.molden`'s `write_molden`: every MO, the
  `nocc` lowest at occupation 2. It is not a TOML key: `pipeline/config` sets
  it to `scf.molden` in the run's output directory. Writing it is part of the
  step, so a failed write fails the step.
- **Closed shell only.** An odd electron count or a multiplicity other than 1
  is `Unsupported`.
- **Host synchronizations**: the energy's two dots and the residual norm read
  host scalars, DIIS reads its singular flag, and `syevd`'s flag is fetched:
  five per iteration. Batching them into one read is a later optimization.
- **A dropped overlap mode** (an eigenvalue under the orthogonalizer's floor)
  leaves a zero column in X, so F' has a spurious zero eigenvalue that the
  aufbau fill can pick. A canonical orthogonalizer with fewer columns than AOs
  is the fix, once a basis needs it.

Errors: under `aux_basis`, building B is `DensityFittingStep::run`'s errors
(an unreadable aux file `IO`, an unknown or incomplete set `InvalidConfig`, B
past the device's free memory `Unsupported`, naming the bytes); an option out of range, more electron pairs than orbitals, or a guess
of the wrong size is `InvalidConfig`; a class above f is J's and K's
`InvalidConfig`; no convergence within `max_iterations` is
`MaxIterationsExceeded`; a non-converged eigensolve is `Numerical`; a failed
runtime, BLAS, solver or launch call is `GpuRuntime`; a `molden` file that
cannot be written is `write_molden`'s error, after the SCF has converged.

## Tests

`test/gtest/pipeline_scf_gpu.cpp`, against PySCF's RHF in
`test/gtest/data/scf/rhf.json` (`tools/emit_scf_reference.py`), which ran over
this tree's own `data/basis` files:

- The step's contract is checked by `static_assert`.
- Incremental builds (the default) against `fock_rebuild_interval = 1` on
  H2O/cc-pVDZ: the energy to 1e-9 Eh and the orbital energies to 1e-7 Eh.
- H2O in STO-3G and cc-pVDZ, NH4+ (charged, so the guess is rescaled) in
  cc-pVDZ, and CO in cc-pVTZ (f shells on both atoms, off every axis), each over the setup steps and the guess run by hand: the energy to
  1e-8 Eh and every orbital energy to 1e-6 Eh of PySCF's, and the invariants
  C^T S C = I, tr(D S) = N and D S D = 2 D.
- A config naming only `[scf]` plans the whole chain, for H2O/cc-pVDZ and
  CO/cc-pVTZ, and its energy and orbital energies match PySCF's to the same
  tolerances.
- An open shell is `Unsupported`, too few iterations `MaxIterationsExceeded`,
  and an option out of range `InvalidConfig`.
- `molden` set writes a file a Molden reader takes: the header, one `[MO]`
  entry per orbital, `nocc` of them at occupation 2.
- The `[scf]` config cases are in `pipeline_config_host.cpp`.

`test/gtest/pipeline_df_scf_gpu.cpp`, against PySCF's DF-RHF over the same
orbital and fitting sets (`test/gtest/data/density_fitting/`,
`tools/emit_df_reference.py`):

- H2O and NH4+ in cc-pVDZ / cc-pVDZ-JKFIT and H2O in cc-pVTZ /
  cc-pVTZ-JKFIT (f orbitals, g fitting functions), each over the setup steps
  and the guess run by hand: the energy to 1e-8 Eh and every orbital energy
  to 1e-6 Eh, and the invariants C^T S C = I, tr(D S) = N and D S D = 2 D.
- An unknown aux basis is the fit's `IO`, passed through.
- The cc-pVTZ case launches three_center's (ff|g) class, whose stack takes
  most of a 10 GB card for the life of the process, so it is registered
  `RUN_SERIAL` in `test/gtest/CMakeLists.txt`, with the two other (ff|g)
  cases.
- A config whose `[scf]` selects density fitting plans the whole chain, with
  no `DensityFitting` step, and runs it to PySCF's energy and orbital
  energies.

## Files

| File | Role |
|------|------|
| `scf.cppm` | Module interface: `ScfStep`; re-exports `:output` |
| `scf_output.cppm` | Interface partition `:output`: `ScfOutput`, so that `:loop` can name it |
| `scf_loop.cppm` | Internal partition `:loop`: `rhf_loop`, the energy, the residual, convergence, DIIS, the eigensolve and the Molden write, against a Fock builder |
| `scf.cpp` | `ScfStep::run`: the checks; `DirectFockBuilder`, the integral-direct Fock build and its full/incremental policy; and `DfFockBuilder`, the density-fitted one, with the first build's K from the guess's factor |
| `CMakeLists.txt` | The module library |
