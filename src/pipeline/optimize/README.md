# Pipeline: Optimize

A geometry optimization: `../../geometry/`'s `geometry::optimize` (BFGS under a
trust radius, the rigid-body motions projected out, Gaussian's four criteria)
over the energy and gradient of the chain from the molecule to `[gradient]`
(RHF) or to `[casscf_gradient]` (a DF-CASSCF root, below), rerun as a nested
`Pipeline` at every geometry. Every other step in the config then runs at the
optimum.

## Module

`free_breakfast.pipeline.optimize`, target `free_breakfast::pipeline::optimize`.
It re-exports `free_breakfast.geometry` (`geometry::Options` is an option),
`free_breakfast.pipeline.gradient` and `free_breakfast.pipeline.casscf_gradient`,
which re-export every step the two chains rerun, each a member. Its product, `OptimizeOutput`, is declared in
`pipeline.molecule` (below).

## The step

```cpp
struct OptimizeOutput {   // in pipeline.molecule
  Molecule molecule;      // the optimized geometry
  double energy;          // the SCF's energy there, or the CASSCF root's
};

struct OptimizeStep {
  static constexpr std::string_view kName = "optimize";

  geometry::Options options{};  // thresholds, max_steps, trust radius, ...

  // The chain, copied in from the config's own tables by pipeline/config.
  MoleculeStep molecule{};      // its xyz is the start
  BasisSetStep basis_set{};
  EriStep eri{};
  OrthogonalizerStep orthogonalizer{};
  GuessStep guess{};
  ScfStep scf{};
  CasscfStep casscf{};                                  // read only under casscf_gradient
  std::optional<CasscfGradientStep> casscf_gradient{};  // set: the CASSCF chain

  std::filesystem::path trajectory{};  // optimize.xyz
  std::filesystem::path final_xyz{};   // final.xyz
  std::filesystem::path history{};     // optimize.json

  Result<OptimizeOutput> run(View<> in, NodeResources &node) const;
};
```

`run` reads the start from `molecule.xyz` and hands its coordinates (bohr) to
`geometry::optimize`. Each evaluation plans and runs one `Pipeline` over

```
molecule -> basis_set -> spherical_transform -> primitive_pair_classes -> eri
  -> overlap -> orthogonalizer -> core_hamiltonian -> guess -> scf -> gradient
```

at the trial geometry, and returns the `ScfOutput`'s energy and the
`GradientOutput`'s total. The pipeline is freed before the next evaluation, so
the device holds one geometry's products at a time. Its first error -- an SCF
that does not converge, `[gradient]`'s refusal of an open shell -- stops the
optimization and is returned as it is.

- **SAD at every geometry.** The guess is the configured `[guess]`, made
  afresh each time; reusing the previous geometry's orbitals is a later
  optimization.
- **The chain writes nothing.** `scf.molden` is cleared in its copy (and
  `casscf.molden` and `casscf_gradient.json_path` in the CASSCF chain's), and the
  option-less steps (spherical transform, pair classes, overlap, core
  Hamiltonian, gradient) run at their defaults, without `overlap.npy` or
  `gradient.json`. Those files are the outer pipeline's.
- **Any `[eri]`** (the RHF chain). The chain holds the config's `EriStep` as it is, so the
  ERIs are whatever `[eri]` asks for, density-fitted or integral-direct
  (`[gradient]` takes both, #517), with no branch here.
- **`Node`-shaped**, because an evaluation runs a `Pipeline` on the node. The
  nested `run` happens on an executor worker, so it co-runs its flow there
  rather than blocking the worker (`../pipeline/README.md`). The step's
  product lives on device 0.
- **Outputs**, each skipped if its path is empty: `optimize.xyz`, every
  evaluated geometry in order (Angstrom), its comment line the energy and
  `(rejected)` for a step the energy rose on; `final.xyz`, the optimum; and
  `optimize.json`:

  ```json
  {"energy": -76.027032783,
   "iterations": [{"energy": ..., "max_force": ..., "rms_force": ...,
                   "max_displacement": ..., "rms_displacement": ...,
                   "trust_radius": ..., "accepted": true}, ...]}
  ```

  one entry per evaluation, the first at the start; `geometry::Iteration`'s
  fields, `../../geometry/README.md`.

## The CASSCF chain

With `casscf_gradient` set (pipeline/config sets it when the config has a
`[casscf_gradient]`), the energy optimized is that step's root of a
state-specific DF-CASSCF, and the chain is

```
first:  molecule -> ... -> guess -> scf -> casscf -> casscf_gradient
later:  molecule -> ... -> eri -> overlap -> core_hamiltonian
          -> carried orbitals -> casscf -> casscf_gradient
```

each evaluation returning the `CasscfOutput`'s energy of `casscf_gradient.root`
and the `CasscfGradientOutput`'s total.

- **The active space is picked once.** At the first geometry the CASSCF
  starts from the SCF's orbitals and `[casscf]`'s selection (by index or
  AVAS). At every later one there is no SCF: a private step of this module,
  standing where the `ScfStep` stood, hands the CASSCF the last evaluation's
  converged orbitals, core, active and virtual, Loewdin-orthonormalized in the
  new overlap, C (CᵀSC)^-1/2 -- the orthonormal set nearest them -- and the
  CasscfStep takes them by index as they come (its `ncas` and `nelecas` the
  CASSCF's, no `active`, no `avas`). Re-picking at every geometry could swap
  an orbital in or out of the space and jump to another CASSCF solution
  mid-optimization. "Last" is the last *evaluated* geometry, a rejected one
  included: it is the trial point nearest the next.
- **That ScfOutput is not an SCF's.** Only the CasscfStep reads it, and only
  its orbitals, their energies (the last `CasscfOutput`'s `mo_energies`) and
  the occupation counts (the first SCF's). Its `energy` is NaN and its `d` is
  zero.
- **Density fitting only.** `[casscf_gradient]` takes nothing else, so an
  integral-direct `eri` is refused before anything runs, here and by
  pipeline/config.
- **State-specific in practice.** The chain takes `[casscf]` as configured,
  but `mcscf::df_gradient` refuses a state average of more than one root
  today, and that error stops the optimization at its first evaluation.
- **The outer CASSCF starts afresh.** After the optimization the outer
  `[casscf]` runs once more at the optimum from the outer SCF and its own
  pick. For a space that pick finds again it is the same root (the GPU case
  below checks it); one that a fresh pick would not find is not carried out
  of the optimization.

## Why a nested Pipeline, not a looping planner

`geometry::optimize` owns its loop and calls back for each energy and
gradient. A planner that looped would have to invert that: run the chain,
hand a step the result, take a new geometry back, and run again, with a
protocol between the planner and that one step. A nested `Pipeline` needs
none of it. The callback plans the chain with the steps the config already
configured and runs it once, which is what a `Pipeline` already does. The
planner learns only that a nested `run` co-runs its flow.

## How the steps after it see the optimum

A product has one producer, and every step reads `MoleculeOutput`, so the
steps after an `[optimize]` cannot read its product in place of the
molecule's. Instead the config **loads the molecule from the final.xyz this
step writes**: `MoleculeStep::load` reads `View<OptimizeOutput>`, which puts
it after the optimization, and takes the charge and multiplicity the file does
not carry from the optimized molecule. Every step reading the molecule --
`[localization]`, the outer `[scf]`, a second `[gradient]` -- then runs at the
optimum, through the planner's one existing way of making a product with
other edges. That is why `OptimizeOutput` is declared in `pipeline.molecule`:
the load names it, and this module imports every step that imports
`pipeline.molecule`.

The chain's own root is not `MoleculeStep` but a private step of this module
building the `Molecule` from the trial atoms in memory. A registry holding
`MoleculeStep` must hold a producer of what its `load` reads, so `MoleculeStep`
in the chain would drag `OptimizeStep` into its own chain.

The outer `[scf]` (and every table the chain also reads) then runs once more
at the optimum, one SCF more than the optimization: its products, the
optimized orbitals among them, are the outer pipeline's.

## Config

```toml
[optimize]
max_force = 4.5e-4        # hartree / bohr, Gaussian's defaults ...
rms_force = 3.0e-4
max_displacement = 1.8e-3 # bohr
rms_displacement = 1.2e-3
max_steps = 100           # evaluations after the first
```

The chain is the config's own `[molecule]`, `[basis_set]`, `[eri]`,
`[orthogonalizer]`, `[guess]` and `[scf]` tables, each at its defaults if
absent, and with a `[casscf_gradient]` in the config, `[casscf]` and it too;
one loading `from` a file is refused, since the chain is rerun at every
geometry, and so is a `[casscf_gradient]` beside a `[gradient]`.
`config/README.md` has the rules.

## Tests

- The step's contract and `MoleculeStep`'s load, by `static_assert`, in
  `test/pipeline/steps_host.cpp`; `[optimize]`'s parsing, the chain copied in,
  the molecule loaded from `final.xyz` and the refusals in
  `test/pipeline/config_host.cpp`.
- `test/pipeline/config_host.cpp` also: the CASSCF chain copied in, and its
  refusals (no `[casscf]` space, `[gradient]` beside it, an integral-direct
  `[eri]`).
- `test/gtest/pipeline_optimize_gpu.cpp`: DF-RHF / cc-pVDZ / cc-pVDZ-JKFIT
  from PySCF's starts (`test/gtest/data/gradient/gradient.json`'s
  `optimized`), the SCF converged tight and `[optimize]` at Gaussian's
  Opt=Tight thresholds (at the defaults ethanol is 3.9e-3 Angstrom off). The
  final geometry is held to PySCF's minimum to 1e-4 Angstrom after a
  least-squares alignment and the energy to 1e-8 Eh; H₂O's evaluations after
  the first to fewer than 15, ethanol's to fewer than 25 (22 today), until a
  model starting Hessian in `geometry` (#536) shortens the identity-started
  BFGS. PySCF's optimizer was Newton's, so its path is not compared. H₂O carries a `[localization]`, which must run
  at the optimum: the loaded molecule matches the optimized one, and the outer
  SCF's energy the optimizer's. Anti ethanol runs in about 7 s, inside the
  fast tier.
- The same file, over integral-direct ERIs (#517's demonstration, with
  `fock_rebuild_interval = 1` so every SCF build is full): H₂O from the same
  start, held to convergence (its last evaluation's largest force under the
  threshold) in fewer than 15 steps. Its minimum is the exact RHF's, so the
  DF reference does not apply.

- The same file, the CASSCF chain: H₂O's state-specific DF-CASSCF(4,4) over
  its σ O-H orbitals ([2, 3, 5, 6]), cc-pVDZ / cc-pVDZ-JKFIT, from an
  off-symmetry start (`test/gtest/data/casscf_gradient/casscf_optimize.json`,
  PySCF's minimum by Newton's method over `df.grad.casscf`), at Opt=Tight's
  thresholds and `conv_tol_grad` 1e-9. The final geometry is held to PySCF's
  to 1e-4 bohr after alignment, the energy to 1e-8 Eh, the first evaluation's
  energy to PySCF's at the start, the evaluations after the first to fewer
  than 15 (9 today), and the outer CASSCF at the optimum to the optimized
  energy. It runs in about 2 s, inside the fast tier. Beside it, a hand-built CASSCF chain over
  integral-direct ERIs, refused by the step.

## Files

| File | Role |
|------|------|
| `optimize.cppm` | Module interface: `OptimizeStep` |
| `optimize.cpp` | `run`: the chain per evaluation, its private root and carried-orbitals steps, and the three files |
| `CMakeLists.txt` | The module library |
