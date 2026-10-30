# Pipeline: Scf

Iterate restricted Hartree-Fock to self-consistency, from the `GuessOutput`,
with Pulay DIIS on the Fock matrix: closed-shell RHF, or ROHF at a
multiplicity above 1. The two-electron part is the `EriOutput`'s
(`pipeline/eri`): integral-direct, with incremental Fock builds, or
density-fitted, closed or open shell.

## Module

`free_breakfast.pipeline.scf`, target `free_breakfast::pipeline::scf`. It
re-exports `free_breakfast.recipes.matrix_ao` and the `orthogonalizer`,
`core_hamiltonian`, `guess` and `eri` steps, which between them re-export every
other product in `run`'s signature. `run` is defined in the implementation
unit, so J, K, DIIS, the BLAS calls it uses, and DF J and K are PRIVATE
links.

## The step

```cpp
struct ScfOutput {
  double energy;                    // total: electronic + nuclear repulsion (hartree)
  int32_t iterations;               // Fock builds
  int32_t n_alpha;                  // occupied in alpha: the first n_alpha columns of c
  int32_t n_beta;                   // occupied in beta; == n_alpha for RHF
  std::vector<double> mo_energies;  // host; ascending (ROHF: within each occupation)
  SphericalMatrixAO c;              // AO x MO
  SphericalMatrixAO d;              // spin-summed, 2 C_occ C_occ^T; ROHF: Da + Db
};

struct ScfStep {
  static constexpr std::string_view kName = "scf";

  int32_t max_iterations = 100;
  double energy_tolerance = 1e-10;     // |dE| between two Fock builds
  double commutator_tolerance = 1e-7;  // ||X^T (F D S - S D F) X||_F
  int32_t diis_history = 8;
  int32_t fock_rebuild_interval = 20;  // every this many builds, one is full; 1: all full
  double tau_j = 1e-10;                // J's screen, coulomb's tau; not a TOML key
  std::filesystem::path molden{};      // if set, the converged orbitals are written here

  Result<ScfOutput> run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, EriOutput,
                       OverlapOutput, OrthogonalizerOutput, CoreHamiltonianOutput, GuessOutput> in,
                  DeviceResources &res) const;
};
```

Each iteration, all on `res.stream()`:

1. **Fock build**: density-fitted under a `DensityFitted` ERI (below), or
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

An open shell (ROHF, n_alpha > n_beta) changes steps 1, 2 and 6 only:

1. The builder makes **Fa = H + J(D) − K(Da)** and **Fb = H + J(D) − K(Db)**,
   D = Da + Db, and the loop assembles the **Roothaan effective Fock matrix**
   F from them (`roothaan_fock`): PySCF's `get_roothaan_fock` but for the
   open-open block. With Fc = (Fa + Fb)/2, **Fo = (Fa + 3 Fb)/4**, the
   projectors Pc = Db S, Po = (Da − Db) S and Pv = I − Da S,
   A = ½ (Pcᵀ Fc Pc + Poᵀ Fo Po + Pvᵀ Fc Pv) + Poᵀ Fb Pc + Poᵀ Fa Pv +
   Pvᵀ Fc Pc, F = A + Aᵀ; PySCF has Fc where this has Fo (below). It is made
   from the densities, so the first build, whose Da = Db = D/2 are the guess
   split per spin, needs no C.
2. E = tr((Da + Db) H + Da Fa + Db Fb)/2 + E_nuc.
6. Da = C[:, :n_alpha] C[:, :n_alpha]ᵀ, Db over n_beta, and D = Da + Db.

DIIS and the residual are unchanged, on (F, Da + Db). On convergence the
published orbitals and energies are PySCF's (`publish_open`). F differs from
PySCF's matrix in the open block alone, so F's canonical orbitals with the
open ones, columns [n_beta, n_alpha), rotated among themselves to
diagonalize Fc there (`canonicalize_open`) carry PySCF's ROHF `mo_energy`,
every one, by occupation class. If those ascend, PySCF's matrix orders the
classes as aufbau does, and the published orbitals are its own canonical
ones, as before #476: its last Roothaan step, the closer one wherever Fc is
a fair curvature (CH₂'s IBO functional moved 1.5e-8 against PySCF's with
F's step, over `pipeline_localization_gpu`'s 1e-8). If they do not ascend,
aufbau on PySCF's matrix would move a localized hole, so F's orbitals stand
and the open energies keep their columns, below a core one. A closed shell
allocates none of the open-shell state and runs none of it, the second K
included, so its path is the RHF loop's operation for operation.

**Why the open block is not PySCF's** (issue #476). The diagonal blocks of a
Roothaan matrix are a canonicalization: they move no stationary point, and
neither the energy nor the commutator reads them. What they set is where the
open eigenvalues sit against the core and virtual ones, which is the Roothaan
step's estimate of the energy's curvature along a core-open rotation (a beta
electron moving, coupled through Fb) and an open-virtual one (an alpha
electron, through Fa). On a cation whose hole localizes on one of several
similar lone pairs, PySCF's Fc puts the open orbital *below* doubly occupied
ones: glycylglycine⁺/STO-3G's minimum has it at −0.480 Eh under three at
−0.458 to −0.444. That minimum is then no aufbau state of the matrix, every
diagonalization moves the hole to another lone pair, and DIIS cannot settle:
density-fitted gly5⁺/cc-pVDZ's |[F,D]| swung between 0.07 and 1.1 for 150
iterations. PySCF's own ROHF fails the same way from either default guess
(`minao`, `atom`); with a level shift of 0.1 to 1.0, ADIIS or EDIIS it
stalls too, or settles on a stationary point 63 mEh higher (gly3⁺/STO-3G at
0.1). Only its second-order solver (`.newton()`) reached the minimum, which
is internally stable. The SAD guess's D/2 split is not the cause: PySCF's
iteration, started from the converged minimum itself, leaves it at once. Fb in
the open block fixes the core-open gap but inverts the open-virtual one on
NO and CH₂/cc-pVTZ. A blend (1 − w) Fa + w Fb, run in PySCF over twenty
open shells (OH, NH₂, CH₂, H₂O⁺, CH₂/cc-pVTZ, O₂, NO, CN, CH₃, allyl, HCO,
NO₂, the N atom's quartet, C₆H₆⁺, glyoxal⁺, (H₂O)₂⁺, glycine⁺ in cc-pVDZ and
gly2-3-5⁺ in STO-3G), reaches every minimum at w = 0.7 and 0.8; 0.6 misses
two glycine cations and 0.9 misses NO. w = 3/4 is the middle
(`kOpenBlockBetaWeight`). On an RTX 3080 (CUDA), from the SAD guess:

| | before (Fc) | after (Fo) |
|---|---|---|
| gly5⁺/cc-pVDZ, density-fitted | not converged in 150 | 21 iterations, −1109.868083304 Eh (PySCF `.newton()`: −1109.868083304) |
| gly5⁺/cc-pVDZ, integral-direct | not converged (31 seen in #476) | 25 iterations, −1109.870090498 Eh, 392 s |
| gly2⁺/STO-3G, integral-direct | not converged in 150 | 22 iterations |
| gly5⁺/STO-3G, integral-direct | not run | 28 iterations |

The reference cases ("Tests") converge to the same energies, to 1e-12 Eh, in
0 to 2 more iterations: OH 8 and 13, NH₂ 13 → 14, CH₂ 14 → 15, H₂O⁺
13 → 14, CH₂/cc-pVTZ 14 → 16 integral-direct, and 0 to 2 more
density-fitted.

On convergence step 5 is skipped: the published orbitals are the canonical
ones of the Fock matrix as built, not of an extrapolation, and D is theirs.
The energy is that of the density the last Fock matrix was built from.

What it settled:

- **The loop and the Fock build are apart.** `:loop`'s `scf_loop` is the
  restricted iteration, written against a Fock builder with two members:
  `build`, which makes F = H + G(D) from the density and the orbitals it came
  from (null on the first build, whose D is the guess), or for an open shell
  Fa and Fb from the spin densities as well (the loop's open-shell state,
  null for a closed shell), and says whether that F is exact for D, so may be
  taken as converged; and `reset_diis`, told the
  commutator norm, which says whether DIIS's history is to be dropped. The
  incremental builds, `fock_rebuild_interval`, the switch to full builds and
  the DIIS reset are the direct builder's policy (`DirectFockBuilder` in
  `scf.cpp`); the guess upload and rescale, convergence and the Molden write
  are the loop's, so a second builder gets them unchanged; so is the open
  shell's Roothaan assembly. The orbitals are
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

  **An open shell's direct build** makes Fa = H + J(D) − K(Da) and
  Fb = H + J(D) − K(Db): one `coulomb` of D and one `exchange` per spin
  density, each K at weight 1 (the closed shell's D = 2 C Cᵀ takes K/2). Its
  incremental build adds J(D − D_built) to both and K(Ds − Ds_built) to each,
  tracking D, Da and Db as built; which builds are full, the switch to full
  builds and the DIIS reset are the closed shell's policy, unchanged, so both
  spins' builds are full together. J is built into the Cartesian K buffer and
  kept there for both spins; each spin's K goes to `g` and its G = J − K to
  `dd`, which J and K have read by then. A closed shell allocates neither
  spin's built density and takes the closed-shell build, operation for
  operation. The two K calls are 87.5% of a gly5⁺/cc-pVDZ build and 99.3%
  of CH₂/cc-pVTZ's; one pass over both densities would pay, and what it
  needs is the exchange README's ("Not done").

  Measured on an RTX 3080, cc-pVDZ, from the first Fock build to converged,
  every run to the same energy within 1e-12 Eh:

  | | all full (`= 1`) | incremental, no reset | incremental (this) |
  |---|---|---|---|
  | gly5 (379 AOs) | 19 builds, 268 s | 4 + 18, 221 s | 3 full + 18, 208 s (-23%) |
  | gly10 (734 AOs) | 19 builds, 1062 s | 4 + 18, 857 s (-19%) | not run |

  A late incremental build is 3-3.5x cheaper than a full one. Switching at
  10x `commutator_tolerance` without the reset tied (209 s on gly5); at 10x
  and 100x with it, 226 and 234 s.

- **The ERIs are the pipeline's, read from `EriOutput`.** One `[eri]`
  (`pipeline/eri`) chooses them for the whole pipeline and its step builds
  them once; this step branches on the alternative. Under `DensityFitted`,
  `DfFockBuilder` makes F = H + J − K/2 from its B: J from D
  (`df_coulomb`), K from the occupied orbitals the density came from
  (`df_exchange`, which takes a factor C of D = 2 C Cᵀ). For an open shell
  it makes Fa and Fb the same way, K(Da) and K(Db) being half of
  `df_exchange` over C's first n_alpha and n_beta columns: a second K. The
  first build's Da = Db = D/2, so its K(Da) = K(Db) is the closed shell's
  K(D)/2, from the guess's one factor, below. Under
  `IntegralDirect`, `DirectFockBuilder` launches J and K over its classes.
  The step builds neither: B and the classes outlive the SCF for the
  CASSCF (and every later reader) to use.
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
  config planning only the density-fitting step ran in 6.85 s against 0.42 s for
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
- **One coarse step owns the loop** (design section 14). D, F and the DIIS
  history are its private state, not products. The J and K classes are the
  `EriOutput`'s, shared, and their scratch is written by one build at a time
  because device steps run one at a time (`pipeline/eri/README.md`).
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
  n_beta lowest at occupation 2 and the next n_alpha − n_beta at 1. It is not a TOML key: `pipeline/config` sets
  it to `scf.molden` in the run's output directory. Writing it is part of the
  step, so a failed write fails the step.
- **ROHF takes either ERI.** n_alpha − n_beta = multiplicity − 1, so
  N and the multiplicity must agree in parity, or it is `InvalidConfig`.
  PySCF's `get_occ` takes the core orbitals lowest in its matrix's spectrum
  and the open ones as the lowest of the rest by their alpha energy; here
  both are aufbau in the iterated F's spectrum, whose open block is Fo. They
  agree on every reference case PySCF converges, which it found internally
  stable; on glycylglycine⁺ PySCF's rule is what keeps it from the minimum.
- **Host synchronizations**: the energy's two dots (three for an open shell)
  and the residual norm read host scalars, DIIS reads its singular flag, and
  `syevd`'s flag is fetched: five per iteration (six). Batching them into one read is a later optimization.
- **A dropped overlap mode** (an eigenvalue under the orthogonalizer's floor)
  leaves a zero column in X, so F' has a spurious zero eigenvalue that the
  aufbau fill can pick. A canonical orthogonalizer with fewer columns than AOs
  is the fix, once a basis needs it.

Errors: an option out of range, an electron count of the wrong parity for the
multiplicity, more alpha electrons than orbitals, or a guess of the wrong size
is `InvalidConfig`; a class above f is J's and K's
`InvalidConfig`; no convergence within `max_iterations` is
`MaxIterationsExceeded`; a non-converged eigensolve is `Numerical`; a failed
runtime, BLAS, solver or launch call is `GpuRuntime`; a `molden` file that
cannot be written is `write_molden`'s error, after the SCF has converged.

## Tests

`test/gtest/pipeline_scf_gpu.cpp`, against PySCF's RHF in
`test/gtest/data/scf/rhf.json` (`tools/emit/scf_reference.py`), which ran over
this tree's own `data/basis` files:

- The step's contract is checked by `static_assert`, in `test/pipeline/steps_host.cpp`.
- Incremental builds (the default) against `fock_rebuild_interval = 1` on
  H2O/cc-pVDZ: the energy to 1e-9 Eh and the orbital energies to 1e-7 Eh.
- H2O in STO-3G and cc-pVDZ, NH4+ (charged, so the guess is rescaled) in
  cc-pVDZ, and CO in cc-pVTZ (f shells on both atoms, off every axis), each over the setup steps and the guess run by hand: the energy to
  1e-8 Eh and every orbital energy to 1e-6 Eh of PySCF's, and the invariants
  C^T S C = I, tr(D S) = N and D S D = 2 D.
- A config naming only `[scf]` plans the whole chain, for H2O/cc-pVDZ and
  CO/cc-pVTZ, and its energy and orbital energies match PySCF's to the same
  tolerances.
- A multiplicity of the wrong
  parity (or below 1, or above N + 1) is `InvalidConfig`, too few iterations
  `MaxIterationsExceeded`, and an option out of range `InvalidConfig`.
- `molden` set writes a file a Molden reader takes: the header, one `[MO]`
  entry per orbital, `nocc` of them at occupation 2.
- The `[scf]` config cases are in `test/pipeline/config_host.cpp`.

Against PySCF's ROHF in `test/gtest/data/scf/rohf.json`, the orbital energies
being its Roothaan effective Fock matrix's eigenvalues, integral-direct:

- Doublets OH (STO-3G and cc-pVDZ), NH2 and H2O+ (charged) and triplet CH2
  in cc-pVDZ, and CH2 in cc-pVTZ (under `FBF_SKIP_IF_FAST_TESTS`): the
  energy to 1e-8 Eh, every orbital energy to 1e-6 Eh, n_alpha and n_beta,
  C^T S C = I, and for each spin density tr(Ds S) = n_s and Ds S Ds = Ds,
  with D = Da + Db.
- Glycylglycine⁺ in STO-3G (`glygly_cation_sto3g`, 53 AOs, 2.5 s): the
  localized hole of issue #476, which ran 150 iterations unconverged on
  PySCF's open block and takes 22 now. PySCF's ROHF cannot converge it, so
  its reference is `rohf_localized_hole.json`: the emitter iterates it on
  the same Fo (`OpenBlockROHF`), checks the energy against PySCF's
  `.newton()` to 1e-10 Eh and its stability, and records PySCF's canonical
  orbital energies by occupation class. Its open one (−0.480 Eh) sits below
  the top three core ones and is compared in its column; the file is its
  own because `test/pytest/test_rohf_reference.py` holds `rohf.json`'s
  orbital energies ascending. The energy is 6e-9 Eh off at the default `tau_j` (2e-12 at 0), J's
  screening over a larger molecule than the others.
- Incremental builds against `fock_rebuild_interval = 1` on OH/cc-pVDZ: the
  energy to 1e-9 Eh.

`test/gtest/pipeline_df_scf_gpu.cpp`, against PySCF's DF-RHF over the same
orbital and fitting sets (`test/gtest/data/density_fitting/`,
`tools/emit/df_reference.py`):

- H2O and NH4+ in cc-pVDZ / cc-pVDZ-JKFIT and H2O in cc-pVTZ /
  cc-pVTZ-JKFIT (f orbitals, g fitting functions), each over the setup steps,
  the fitted `EriOutput` and the guess run by hand: the energy to 1e-8 Eh and
  every orbital energy to 1e-6 Eh, and the invariants C^T S C = I,
  tr(D S) = N and D S D = 2 D.
- The cc-pVTZ case launches three_center's (ff|g) and (dd|g) classes, whose
  per-thread stack the CUDA runtime reserves for the life of the process:
  about 1.4 GB of a 10 GB card, (dd|g)'s 13,856 B being the family's worst
  frame (`integrals/eri/density_fitting/three_center/README.md`, "The
  stack"). It was 7.3 GB before FelizJueves's low-stack eri3 cut (ff|g) from
  70,232 B to 6,320 B, and the case was `RUN_SERIAL` then. It peaks at about
  1.5 GB now and shares the card like any other case.
- A config whose `[eri]` selects density fitting plans the whole chain and
  runs it to PySCF's energy and orbital energies.

Against PySCF's DF-ROHF (`test/gtest/data/density_fitting/df_rohf.json`), the
orbital energies being its Roothaan effective Fock matrix's eigenvalues:

- Doublets OH, NH2 and H2O+ (charged) and triplet CH2 in cc-pVDZ /
  cc-pVDZ-JKFIT, and CH2 in cc-pVTZ / cc-pVTZ-JKFIT: the energy to 1e-8 Eh,
  every orbital energy to 1e-6 Eh, n_alpha and n_beta, C^T S C = I, and for
  each spin density tr(Ds S) = n_s and Ds S Ds = Ds, with D = Da + Db.
- A config at `multiplicity = 2` (H2O+) plans and runs DF-ROHF to the same
  tolerances, and its Molden file has n_beta orbitals at occupation 2 and
  n_alpha − n_beta at 1.

## Files

| File | Role |
|------|------|
| `scf.cppm` | Module interface: `ScfStep`; re-exports `:output` |
| `scf_output.cppm` | Interface partition `:output`: `ScfOutput`, so that `:loop` can name it |
| `scf_loop.cppm` | Internal partition `:loop`: `scf_loop`, the Roothaan effective Fock matrix, the open shell's published PySCF orbitals, the energy, the residual, convergence, DIIS, the eigensolve and the Molden write, against a Fock builder |
| `scf.cpp` | `ScfStep::run`: the checks; `DirectFockBuilder`, the integral-direct Fock build, closed or open shell, and its full/incremental policy; and `DfFockBuilder`, the density-fitted one, closed or open shell, with the first build's K from the guess's factor |
| `CMakeLists.txt` | The module library |
