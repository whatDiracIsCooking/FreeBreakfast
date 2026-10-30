# Pipeline: Optimize

A geometry optimization: `../../geometry/`'s `geometry::optimize` (BFGS under a
trust radius, the rigid-body motions projected out, Gaussian's four criteria)
over the energy and gradient of the chain from the molecule to `[gradient]`,
rerun as a nested `Pipeline` at every geometry. Every other step in the config
then runs at the optimum.

## Module

`free_breakfast.pipeline.optimize`, target `free_breakfast::pipeline::optimize`.
It re-exports `free_breakfast.geometry` (`geometry::Options` is an option) and
`free_breakfast.pipeline.gradient`, which re-exports every step the chain
reruns, each a member. Its product, `OptimizeOutput`, is declared in
`pipeline.molecule` (below).

## The step

```cpp
struct OptimizeOutput {   // in pipeline.molecule
  Molecule molecule;      // the optimized geometry
  double energy;          // the SCF energy there
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
- **The chain writes nothing.** `scf.molden` is cleared in its copy, and the
  option-less steps (spherical transform, pair classes, overlap, core
  Hamiltonian, gradient) run at their defaults, without `overlap.npy` or
  `gradient.json`. Those files are the outer pipeline's.
- **Any `[eri]`.** The chain holds the config's `EriStep` as it is, so the
  ERIs are whatever `[eri]` asks for. `[gradient]` refuses integral-direct
  ERIs today; once it takes them (#517) `[optimize]` runs over them with no
  change here.
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
absent; one loading `from` a file is refused, since the chain is rerun at
every geometry. `config/README.md` has the rules.

## Tests

- The step's contract and `MoleculeStep`'s load, by `static_assert`, in
  `test/pipeline/steps_host.cpp`; `[optimize]`'s parsing, the chain copied in,
  the molecule loaded from `final.xyz` and the refusals in
  `test/pipeline/config_host.cpp`.
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

## Files

| File | Role |
|------|------|
| `optimize.cppm` | Module interface: `OptimizeStep` |
| `optimize.cpp` | `run`: the chain per evaluation, its private root, and the three files |
| `CMakeLists.txt` | The module library |
