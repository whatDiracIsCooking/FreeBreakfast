# Pipeline: Optimize

A geometry optimization: `../../geometry/`'s `geometry::optimize` (BFGS under a
trust radius, the rigid-body motions projected out, Gaussian's four criteria)
over the energy and gradient of the chain from the molecule to `[gradient]`
(RHF), to `[casscf_gradient]` (a DF-CASSCF root, followed from geometry to
geometry over a state average, below) or to
`[nevpt2_gradient]` (that CASSCF's SC- or PC-NEVPT2, or one SC- or
PC-QD-NEVPT2 state of its state average, below), rerun as a nested
`Pipeline` at every geometry; or, with `meci = [I, J]`, the minimum-energy
conical intersection of two roots of a DF-SA-CASSCF, `geometry::optimize_meci`
over one `mcscf::df_intersection` per geometry (below). Every other step in the
config then runs at the optimum.

## Module

`free_breakfast.pipeline.optimize`, target `free_breakfast::pipeline::optimize`.
It re-exports `free_breakfast.geometry` (`geometry::Options` is an option),
`free_breakfast.pipeline.gradient`, `free_breakfast.pipeline.casscf_gradient`
and `free_breakfast.pipeline.nevpt2_gradient`, which re-export every step the
three chains rerun, each a member. Its product, `OptimizeOutput`, is declared in
`pipeline.molecule` (below).

## The step

```cpp
struct OptimizeOutput {   // in pipeline.molecule
  Molecule molecule;      // the optimized geometry
  double energy;          // the SCF's energy there, the CASSCF root's, or its NEVPT2's;
                          // under meci the pair's mean (E_I + E_J) / 2
};

inline constexpr double kStateOverlapFloor = 0.9;
struct RootMatch {
  int32_t root;    // this geometry's root J matched to the last root I
  double overlap;  // S_IJ over row I's 2-norm, signed
};
// Each row of S over its 2-norm (below)
Result<std::vector<double>> normalized_overlaps(std::span<const double> root_overlaps);
// One match per last root, one to one; Numerical below the floor for a
// root in `followed` (below)
Result<std::vector<RootMatch>> match_roots(std::span<const double> root_overlaps,
                                           std::span<const int32_t> followed = {});
// The smaller singular value of rows and columns {i, j} of the normalized
// S, or Numerical below the floor (below, "The MECI")
Result<double> pair_overlap(std::span<const double> root_overlaps, int32_t i, int32_t j);
// |sum_I u_I s_I u'_pi(I)|, or Numerical below the floor (below)
Result<double> qd_state_overlap(std::span<const double> root_overlaps,
                                std::span<const double> previous,
                                std::span<const double> current);

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
  CasscfStep casscf{};                                  // read only under one below
  std::optional<CasscfGradientStep> casscf_gradient{};  // set: the CASSCF chain
  std::optional<Nevpt2GradientStep> nevpt2_gradient{};  // set: the NEVPT2 chain
  std::optional<std::array<int32_t, 2>> meci{};         // set: the MECI of roots I, J

  std::filesystem::path trajectory{};  // optimize.xyz
  std::filesystem::path final_xyz{};   // final.xyz
  std::filesystem::path history{};     // optimize.json

  Result<OptimizeOutput> run(View<> in, NodeResources &node) const;
  Status check_meci() const;  // meci's refusals, here and in pipeline/config
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
  fields, `../../geometry/README.md`. Over a state-averaged root each entry
  also has `state_overlap` (null at the first) and `root`, the root
  differentiated there; over a QD state, `state_overlap` and `heff_gap`
  (below).

## The CASSCF chain

With `casscf_gradient` set (pipeline/config sets it when the config has a
`[casscf_gradient]`), the energy optimized is that step's root of the
DF-CASSCF, state-specific or a state average's (below), and the chain is

```
first:  molecule -> ... -> guess -> scf -> casscf -> casscf_gradient
later:  molecule -> ... -> eri -> overlap -> core_hamiltonian
          -> carried orbitals -> casscf -> casscf_gradient
```

each evaluation returning the `CasscfOutput`'s energy of the root and the
`CasscfGradientOutput`'s total. The pipeline stops at the CasscfStep, and the
`CasscfGradientStep` is run on its products once the root is known.

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
- **A state-averaged root is followed** (below).
- **The outer CASSCF starts afresh.** After the optimization the outer
  `[casscf]` runs once more at the optimum from the outer SCF and its own
  pick. For a space that pick finds again it is the same root (the GPU case
  below checks it); one that a fresh pick would not find is not carried out
  of the optimization.

## The state-averaged root

Over a `[casscf]` of more than one root, `casscf_gradient.root` names the
root at the start. At every evaluation after the first:

1. this CASSCF's roots are matched to the last evaluation's by
   `mcscf::state_overlaps` (through the AO cross overlap of the two
   geometries) and `match_roots`: each last root I to the root J of largest
   |S_IJ|, which must be one to one;
2. the root followed is the one the last root matched, and the gradient is
   taken of it, not of the index: the CASSCF's roots are ordered by energy,
   so after two cross the configured index would be the other surface;
3. its overlap, S_IJ over the 2-norm of row I, is logged
   (`optimize: CASSCF root I followed to root J, matched overlap ...`, the
   raw |S_IJ| beside it) and written to `optimize.json` with the root; below
   `kStateOverlapFloor` (0.9, a turn of about 26° in one step) the
   evaluation fails, `Numerical`, and so does the optimization.

**Why the row is normalized.** S_IJ is the overlap of whole determinants,
the core's among them, at two geometries: it falls with the step's length for
every root alike, to about 0.93 at a 0.05 bohr step and 0.76 at 0.1 bohr on
formaldehyde's C-O (PySCF, `tools/emit/sa_casscf_optimize_reference.py`'s
start), which is not a root turning and would refuse an ordinary step under
the trust radius. Over row I's norm it is the cosine between J and root I's
part in this geometry's roots: the share of the followed root that stayed
in J. The matching itself, by largest |S_IJ| per row, is the same either way,
and so is the QD chain's, which takes only its sign.

`normalized_overlaps` is that normalized S, which `match_roots` matches
over; a check of several roots at once (a pair's 2 × 2 block, #774) reads
its block from it rather than normalizing again. `match_roots` takes the
roots to hold to the floor as a list. A state-specific CASSCF has one root
and nothing to match: its entries carry neither key.

## The MECI

With `meci = [I, J]` the step finds the minimum-energy conical intersection
(MECI) of roots I and J of the `[casscf]`, a state average, by
`../../geometry/`'s `geometry::optimize_meci` (Bearpark–Robb–Schlegel
gradient projection: a Newton step on the gap along x = g_J − g_I, BFGS on
the mean energy in the intersection space, converged on the gap
`options.max_gap` beside the four usual criteria). The chain is the CASSCF
chain's, ending at `[casscf]`, with the orbitals carried and the active space
picked once, as above. At every geometry one `mcscf::df_intersection` gives
g_I, g_J and h_IJ together (one CI-relaxed Hessian, one block Z-vector solve);
`optimize_meci` reads only h_IJ's direction, and re-takes the lower and upper
of the pair by energy at every evaluation, so I and J may come in either
order.

**The pair is followed by its space, not by its roots.** At every evaluation
after the first, the CASSCF's roots are overlapped with the last
evaluation's (`mcscf::state_overlaps`), each row normalized
(`normalized_overlaps`, as for a single root), and `pair_overlap` takes the
2 × 2 block of rows and columns {I, J}: its smaller singular value is how
much of the last pair's two-root space this geometry's pair keeps. Near a
crossing the two roots mix and swap freely, which `match_roots` would refuse
and which is no loss here; the block's singular values fall only when the
pair's space turns into a third root's. Below `kStateOverlapFloor` the
evaluation fails, `Numerical`, naming the two roots and the overlap. It is
written to `optimize.json` as `pair_overlap` (null at the first) and logged.

**Refusals**, by `check_meci`, in `run` before anything runs and in
pipeline/config once `[casscf]` is copied in: a gradient member set beside
`meci`, a `[casscf]` of one root, I = J, or a root past `nroots`
(`InvalidConfig`); unequal weights on I and J (`Unsupported`: the rotation
between two roots of unequal weight divides by their gap, singular at the
crossing, `df_intersection`'s own refusal); an integral-direct `[eri]`
(`Unsupported`). A `[gradient]` or `[casscf_gradient]` beside it is a later
step, run at the MECI; a `[nevpt2_gradient]` beside it makes the MECI the
QD-NEVPT2's (below).

**Outputs.** `OptimizeOutput::energy` is the pair's mean at the MECI.
`optimize.json` has the mean as `energy`, with `lower`, `upper` and `gap`,
at the head and per evaluation, where each entry also has the four norms,
the trust radius, `branching_dimension` (2, or 1 where h_IJ had nothing left
after x) and `pair_overlap`, and no `accepted` (`optimize_meci` rejects no
step). Each `optimize.xyz` comment holds the mean and the gap, and
`final.xyz`'s too. `examples/meci.toml` is ethylene's.

The CASCI's roots outside the average are not followed: a third root
crossing into S1 on the way is caught by `pair_overlap`, not avoided. The
test's case keeps it 0.30 Eh clear (the fixture's `s2_margin`).

## The QD-NEVPT2 MECI

With `meci = [P, Q]` beside a quasi-degenerate `[nevpt2_gradient]`, P and Q
are SC- or PC-QD-NEVPT2 states (H_eff's eigenvalues, ascending; the table's
`variant`), and the step finds their MECI the same way, over the same
CASSCF chain. At every geometry one `nevpt2::qd_sc_intersection` (or
`qd_pc_intersection`, `../../nevpt2/intersection/`) gives g_P, g_Q and h_PQ
at one set of H_eff eigenvectors, with no refusal of a degenerate pair, and
the QD energies; they go to `optimize_meci` as `df_intersection`'s do. The
weights may be unequal (the intersection's only division by a gap is by the
CASSCF roots', never by E_Q − E_P). `[nevpt2_gradient]`'s `frozen_core`
is the intersection's too: resolved at each geometry's CASSCF
(`Nevpt2Step::resolve_frozen_core`) and passed as
`QdIntersectionOptions::nfrozen`, so the MECI is the frozen-core
functional's, H_eff, g_P, g_Q and h_PQ alike (#875). A QD state optimized
without `meci` takes it through the step itself.

**The pair is followed by its two-state space**, through the CASSCF roots:
`pair_overlap(root_overlaps, previous, current, P, Q)` takes the 2 × 2 block
M_ab = u_a(last)ᵀ S u_b(this), a and b each P or Q, S the row-normalized
`state_overlaps` and u H_eff's eigenvectors, and its smaller singular value
must stay at or above `kStateOverlapFloor` (`Numerical` below, naming the
states). P and Q, and the CASSCF roots under them, may swap and mix freely:
no root is matched.

**Refusals** (`check_meci`, in `run` and in pipeline/config): a
`[nevpt2_gradient]` that is not quasi-degenerate (one state, none to cross)
or names a `state` (`InvalidConfig`); a `[casscf]` of one root, P = Q, or a
state past `nroots` (`InvalidConfig`); an open shell or an integral-direct
`[eri]` (`Unsupported`). The `[nevpt2_gradient]` is the chain's, and is not
run as a later step: at the MECI it would differentiate one of two
degenerate states, which it refuses.

**Outputs** are the CASSCF MECI's, and each `optimize.json` entry also
holds the CASSCF's reported `gradient_norm` and `ci_residual_norm` (its
largest Davidson residual, `CasscfOutput::ci_residual_norm`), `heff_gap`
(|E_Q − E_P|), `third_state_margin` (E_{Q+1} − E_Q, Q the higher of the
pair; null when there is none), `root_gap` (the CASSCF's smallest), and the
intersection's two gap sensitivities `gap_sensitivity_orbital` (s_o) and
`gap_sensitivity_ci` (s_c). The head, beside `energy`, `lower`, `upper` and
`gap`, holds the branching space at the final geometry, the last
evaluation's (`optimize_meci`'s crossing is its last evaluation): `g_p`,
`g_q` and `h_pq`, each n_atom rows of three (hartree / bohr), the
intersection's own g_P, g_Q and h_PQ unchanged (P and Q as `meci` names
them, h_PQ neither normalized nor re-signed), and `nroots`, the state
average's size. They are head-only: every evaluation's would triple the
file for vectors no reader needs before the crossing. At the crossing u_P
and u_Q are any orthonormal pair of the two-state space, so g_P, g_Q and
h_PQ are one choice of it; the plane span{g_Q − g_P, h_PQ} and
(g_P + g_Q)/2 are not (`../../nevpt2/intersection/README.md`).

**The gap's error bound is computable from the file alone.** The reported
gap's distance from the exact surface's is at most

    s_o ‖g‖ + s_c √n ρ
      = iterations[-1].gap_sensitivity_orbital × iterations[-1].gradient_norm
      + iterations[-1].gap_sensitivity_ci × √nroots × iterations[-1].ci_residual_norm

(`../../nevpt2/intersection/README.md`, "The gap sensitivities"), and any
entry's own five give the bound at its geometry. So a config that wants the
gap to 1e-7 sets `[casscf] conv_tol_grad` well under it:
`examples/qd_meci.toml` uses 1e-10.

## The NEVPT2 chain

With `nevpt2_gradient` set instead (pipeline/config sets it when the config
has a `[nevpt2_gradient]`), the energy optimized is the SC- or PC-NEVPT2 total
energy E_cas + E_corr of the single-state DF-CASSCF, and the chain is the
CASSCF chain's with `nevpt2_gradient` in place of `casscf_gradient`:

```
first:  molecule -> ... -> guess -> scf -> casscf -> nevpt2_gradient
later:  molecule -> ... -> eri -> overlap -> core_hamiltonian
          -> carried orbitals -> casscf -> nevpt2_gradient
```

each evaluation returning the root's `CasscfOutput` energy plus the
`Nevpt2GradientOutput`'s E_corr, and its total. Everything the CASSCF chain
says above holds: the active space is picked once and the orbitals carried
on, and the chain writes nothing (`nevpt2_gradient.json_path` is cleared in
its copy). At most one of the two gradients is set: both, by hand, is
`InvalidConfig` before anything runs, as pipeline/config refuses a config
naming both.

- **A refusal stops it.** Every refusal of the step -- a state average, an
  open shell, and the numerical ones `nevpt2::sc_gradient` or
  `nevpt2::pc_gradient` makes at run, a degenerate inactive pair whose
  multipliers are not determined and a class PC refuses (a metric with no
  positive eigenvalue, a non-positive λ + Δ) among them -- is the evaluation's error.
  `geometry::optimize` returns its callback's first error, so the
  optimization ends there with the step's message as it is, at whichever
  geometry it is met; it is never stepped through or retried at a shorter
  step. The density-fitting refusal is made here too, before anything runs.
- **Converge the CASSCF.** The gradient is the NEVPT2 energy's only as far
  as the CASSCF converged (`../nevpt2_gradient/README.md`), so a config
  optimizing to Opt=Tight tightens `[casscf]` `conv_tol_grad` (1e-9 in the
  GPU case below).
- **Watch the inactive gaps.** SC-NEVPT2 is not invariant to rotations
  among near-degenerate canonical core or virtual orbitals, so its energy
  curves sharply where two of them nearly cross (`nuclear_gradient.tex`
  §Hazards; `test/gtest/data/README.md`'s note on symmetric H₂O). The
  optimizer steps through that under its trust radius; only a pair the
  gradient cannot determine is refused.

## The QD-NEVPT2 state

With `nevpt2_gradient.quasi_degenerate` set, the `[casscf]` is a state
average and the energy optimized is E_K, the effective Hamiltonian's
eigenvalue `nevpt2_gradient.state` (ascending), over the `QdScGradient`'s or
`QdPcGradient`'s `total`. The chain is the NEVPT2 chain's, the state-averaged
CASSCF's orbitals carried on.

**The state is followed, not its index alone.** At every evaluation after
the first:

1. this CASSCF's roots are matched to the last evaluation's by
   `mcscf::state_overlaps` (through the AO cross overlap of the two
   geometries): each last root I to the root of largest |S_IJ|, which must
   be one to one, with that overlap's sign;
2. u_K, H_eff's eigenvector K over the roots (recovered from the
   Lagrangian's W = u_K u_Kᵀ), is carried into that matched basis and
   overlapped with the last u_K: `qd_state_overlap`, |Σ_I u_I s_I u'_π(I)|,
   s_I the sign of `match_roots`' overlap;
3. the overlap is logged (`optimize: QD state K matched overlap ...`, with
   H_eff's smallest gap and the CASSCF's root gap) and kept for
   `optimize.json`; below `kStateOverlapFloor` (0.9, a turn of about 26°
   in one step) the evaluation fails, `Numerical`, and so does the
   optimization.

The index K differentiated is never changed. H_eff's eigenvalues are
continuous and kept ascending, so the state at K keeps its identity until
two of them nearly cross; that is exactly where its u turns quickly and the
overlap falls. The CASSCF roots, by contrast, may swap order freely: the
matching carries u across a swap. So the optimization stays on the target
state's surface, or refuses, and never steps onto another one. "Last" is the
last evaluated geometry, a rejected one included, as for the carried
orbitals.

Every refusal of `[nevpt2_gradient]` stops it as in the NEVPT2 chain, H_eff
eigenvalue K within `mcscf::kRootGapFloor` of another among them.

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
# meci = [0, 1]           # the MECI of these two [casscf] roots instead
# max_gap = 1e-5          # hartree, the MECI's fifth criterion (meci only)
```

The chain is the config's own `[molecule]`, `[basis_set]`, `[eri]`,
`[orthogonalizer]`, `[guess]` and `[scf]` tables, each at its defaults if
absent, and with a `[casscf_gradient]` or a `[nevpt2_gradient]` in the
config, `[casscf]` and it too; one loading `from` a file is refused, since
the chain is rerun at every geometry, and so are any two of `[gradient]`,
`[casscf_gradient]` and `[nevpt2_gradient]`. With `meci`, `[casscf]` is
copied in and no gradient table; `check_meci`'s refusals are made there.
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
- `test/pipeline/config_host.cpp`: the NEVPT2 chain copied in, and its
  refusals (`[gradient]` or `[casscf_gradient]` beside it, an
  integral-direct `[eri]`, PC).
- `test/gtest/pipeline_optimize_gpu.cpp`, the NEVPT2 chain: the skew H₂O
  of `test/gtest/data/nevpt2_gradient/nevpt2_gradient.json`, SC-NEVPT2 over
  its DF-CASSCF(4,4) [2, 3, 5, 6], cc-pVDZ / cc-pVDZ-JKFIT, at Opt=Tight's
  thresholds and `conv_tol_grad` 1e-9, converged (its last evaluation's
  largest force under max_force) in fewer than 15 steps, to the geometry
  `nevpt2_optimize.json` records within 1e-5 bohr after alignment, where
  PySCF's DF-NEVPT2 single point is the optimized energy to 1e-9 Eh and
  PySCF's own numerical gradient is under max_force too. That geometry is
  this step's optimum: no reference program has an SC-NEVPT2 gradient to
  find its own (`tools/emit/nevpt2_optimize_reference.py`). The first
  evaluation is the start's PySCF energy to 1e-9, and the outer
  `[nevpt2_gradient]`, from a fresh SCF and pick at the optimum, the
  optimized energy. 13 steps in about 4 s, inside the fast tier. Beside it: a
  hand-edited state-averaged chain, which the step refuses after the first
  SCF and CASSCF, ending the run with that exact message and no file
  written; and both gradients set, or an integral-direct `eri`, refused
  before anything runs.
- `test/pipeline/optimize_host.cpp`: `qd_state_overlap` follows a state
  turned a little, through a sign flip and through a swap of the CASSCF
  roots (where an overlap by index would be 0); refuses, `Numerical`, an
  overlap just below `kStateOverlapFloor` (the floor itself is kept) and a
  30° turn, and roots that do not match one to one; and sizes that do not
  agree, `InvalidConfig`. `config_host.cpp`: an `[optimize]` over a
  quasi-degenerate `[nevpt2_gradient]` is copied in.
- `test/gtest/pipeline_optimize_gpu.cpp`, the QD state: formaldehyde's
  SC-QD-NEVPT2 state 1 (n → π*) over its DF-SA2-CASSCF(4,3)
  (`test/gtest/data/qd_nevpt2_gradient/h2co_ccpvdz_sa2.json`'s start),
  cc-pVDZ / cc-pVDZ-JKFIT, at Opt=Tight's thresholds: converged in fewer
  than 15 steps (10 today), every overlap after the first in
  `optimize.json` at or above the floor, H_eff's gap printed per step
  (0.095 at the least, no crossing), and the outer `[nevpt2_gradient]`, a
  fresh single point at the optimum, with its largest component under
  `max_force` and its rms under `rms_force`. The geometry stays planar, so
  the two roots are of different irreps and each overlap is 1 to rounding;
  the turning state is the host test's. About 4 s, inside the fast tier.

- `test/pipeline/optimize_host.cpp`: `normalized_overlaps` divides each row
  by its norm and refuses a row of zeros; `match_roots` follows a root through a
  swap of the CASSCF roots under a common decay of every overlap, its
  overlap normalized by its row and its phase signed; keeps a root turned
  20° and refuses, `Numerical`, one turned 30° (0.866, below the floor) while
  the unfollowed matching still succeeds; and refuses sizes that are no
  n × n, or a followed root past n, `InvalidConfig`.
- `test/gtest/pipeline_optimize_gpu.cpp`, the state-averaged root:
  formaldehyde's S1 (n → π*), root 1 of its DF-SA2-CASSCF(4,3), cc-pVDZ /
  cc-pVDZ-JKFIT, from `test/gtest/data/casscf_nac/`'s pyramidalized,
  twisted `h2co_skew_ccpvdz_sa3` start, at Opt=Tight's thresholds and
  `conv_tol_grad` 1e-9, to PySCF's pyramidal minimum
  (`test/gtest/data/casscf_gradient/sa_casscf_optimize.json`). A planar start
  would keep C₂ᵥ and reach the planar saddle, where the reference would
  agree, so the start is not planar, and the alignment allows a reflection
  (the minimum's two mirror images). The bars are derived from the fixture:
  the geometry within δ = (F_test + F_ref)/k_min, F_test the test's
  `max_force` and F_ref and k_min the reference's last force and least
  projected curvature, and the energy within (F_test + F_ref)·δ; every entry's `state_overlap` at or above the floor, and its
  `root` 1. The evaluations after the first are held under 25 (18 today),
  a regression guard until a model starting Hessian (#536).
- `test/pipeline/optimize_host.cpp`, the MECI pair: `pair_overlap` keeps a
  pair whose roots swap order, with either phase, under a common decay
  ([[0, ±1], [±1, 0]]), and a pair turned 45° within itself, where
  `match_roots` refuses; keeps a pair one of whose roots turned 20° into a
  third root, at cos 20°, and refuses, `Numerical`, one turned 30° (0.866,
  below the floor); and refuses sizes that are no n × n, I = J and roots
  past n, `InvalidConfig`. `config_host.cpp`: `meci` and `max_gap` parsed,
  `[casscf]` copied in and a `[casscf_gradient]` beside it left a later
  step; and every refusal (`meci` not two distinct roots or not integers,
  `max_gap` without `meci` or not positive, one root, a root past `nroots`,
  `[nevpt2_gradient]` beside it, unequal weights, an integral-direct `[eri]`
  written or filled in).
- `test/gtest/pipeline_optimize_gpu.cpp`, the MECI: ethylene's S0/S1 over
  its DF-SA2-CASSCF(2,2), cc-pVDZ / cc-pVDZ-JKFIT, from
  `test/gtest/data/casscf_meci/`'s start (the second CH₂ twisted 80° and
  pyramidalized; from the Cs start the search finds a seam saddle), at
  Opt=Tight's four thresholds and `max_gap` 1e-7. The bars are derived from
  the fixture, with F the test's `max_force` plus the reference's 1e-6, Δ the
  test's gap plus the reference's 1e-8, and s_min the gap's smaller slope in
  the branching plane (twice the smaller singular value of [x/2; h_01] at the
  fixture's MECI, 0.1413), not min(x_norm, 2·h_norm), which is not
  reproducible and never smaller. A fresh single point at `final.xyz` (the
  outer `[casscf]` from a fresh RHF and pick, then one `df_intersection`)
  has a gap within 1e-7 and a largest projected force (#771's projector,
  written again in the test) within `max_force`; the geometry, after
  alignment, is within δ = F/k_min + Δ/s_min (4.7e-4 bohr) of the fixture's
  MECI, and the mean energy within gbar_plane·Δ/s_min + F²/k_min (5.9e-8 Eh)
  of its; every `pair_overlap` after the first is at or above the floor.
  The first evaluation's energies and a fresh start's g_0, g_1 and h_01 (up
  to sign) against the fixture's, and the step count, are printed, not held.
  42 steps in about 15 s on the 3080, the family's cheapest, so in the fast
  tier.
- `test/pipeline/optimize_host.cpp`, the QD pair: `pair_overlap` over
  H_eff's eigenvectors keeps a swapped P/Q block ([[0, ±1], [±1, 0]]) at
  every phase, the same swap seen through swapped CASSCF roots, and a pair
  turned 45° within itself; keeps one turned 20° into a third state, at cos
  20°, and refuses one turned 30°, `Numerical`; and refuses eigenvectors of
  the wrong size or states not two distinct ones of n, `InvalidConfig`.
  `config_host.cpp`: the QD chain copied in, SC and PC, the outer
  `[nevpt2_gradient]` taken out, unequal weights allowed; and every refusal
  (a `state`, a single-state `[nevpt2_gradient]`, one root, P = Q, a state
  past `nroots`, an open shell, an integral-direct `[eri]` written or
  filled in).
- `test/gtest/pipeline_optimize_gpu.cpp`, the QD-NEVPT2 MECI (#785):
  ethylene's states 0 and 1 over its DF-SA3-CASSCF(2,2) (every singlet of
  the CAS, equal weights), cc-pVDZ / cc-pVDZ-JKFIT, from
  `test/gtest/data/casscf_meci/`'s `meci.atoms` (the SA2 CASSCF MECI,
  twisted off Cs), at Opt=Tight's four thresholds, `max_gap` 1e-7 and
  `[casscf] conv_tol_grad` 1e-10. Each claim runs once against its derived
  bar. **C1**: at a fresh single point at `final.xyz` (the outer `[casscf]`
  from a fresh RHF, then one `qd_*_intersection`), QD states 0 and 1 are
  within Δ + Σ_{opt, fresh} (s_o‖g‖ + s_c√3·ρ), ‖g‖ and ρ the reported
  `gradient_norm` and `ci_residual_norm` (the optimizer's last entry's and
  the fresh CASSCF's), s_o and s_c the fresh point's, with that convergence
  term at or below Δ/10. **C2**: at the same point, `nevpt2::numerical_heff`
  (#783) along an orthonormal basis of the 12 internal directions (steps 1e-3,
  5e-4, 2.5e-4 bohr, Richardson; 72 single points) gives ḡ, x and h, and
  max |P ḡ| in Cartesians (#771's projector) is within
  F + 2e(1 + 2|ḡ| / min(|x|, 2|h|)), e the oracle's largest error bar, with
  the oracle's term at or below F. **C3**: E₂ − E₁ ≥ 1e-2 there. SC and PC
  each run C1–C3 in the slow tier: at 31 steps and about 25 s the SC
  optimization is past slow_test.h's ten seconds and dearer than the
  CASSCF MECI, so the MECI family's fast-tier member stays that one. Every entry's pair
  overlap and third-state margin, the step count, the mean energy, the RMS
  displacement from the start and the wall times are printed, not held. On
  the 3080: SC 31 steps in 24.5 s, its fresh gap 3.6e-9 against a convergence term of 1.6e-12;
  the oracle 72 points in 12.4 s, max |P ḡ| 3.9e-6 against 1.51e-5 (its
  own term 1.2e-7); PC 24 steps in 18.6 s. s_c is ~1e-17 here: the three
  roots span the CAS's singlets, so the CI multipliers vanish.
- `test/gtest/pipeline_optimize_gpu.cpp`, the QD MECI's file (#844): skew
  formaldehyde's SC-QD-NEVPT2 states 0 and 1 over its DF-SA3-CASSCF(4,3)
  (`h2co_skew_ccpvdz_sa3`), every threshold loose so the one evaluation is
  the final geometry, the start written at `final.xyz`'s twelve decimals.
  Every entry has `gap_sensitivity_orbital` and `gap_sensitivity_ci`, the
  head `nroots` and `g_p`, `g_q`, `h_pq` as n_atom rows of three; one
  `qd_sc_intersection` over the outer `[casscf]` at that geometry is the
  product, and the three vectors are held to it at 1e-9 of their largest
  magnitude, the two sensitivities at 1e-9 of s_o + s_c: separately
  converged Z-vectors' reproducibility (`../../nevpt2/intersection/README.md`).
  The pair is apart there, so the vectors are fixed but for h_PQ's sign: the
  pair's relative phase follows the CASSCF's CI vector signs, which a
  separate run need not repeat, so h_PQ is held up to that one sign. The
  bound s_o ‖g‖ + s_c √n ρ is formed from the file and printed. Fast tier.

## Files

| File | Role |
|------|------|
| `optimize.cppm` | Module interface: `OptimizeStep`, `kStateOverlapFloor` and the four following functions |
| `optimize.cpp` | `run`: the chain per evaluation, its private root and carried-orbitals steps, the MECI's `df_intersection` and `pair_overlap`, `check_meci`, and the three files |
| `CMakeLists.txt` | The module library |
