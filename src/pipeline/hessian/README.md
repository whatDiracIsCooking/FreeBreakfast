# Pipeline: Hessian

The closed-shell RHF Cartesian Hessian at the pipeline's molecule -- or a
DF-CASSCF root's, state-specific or one root of a state average (below,
"The CASSCF chain"), or the SC- or PC-NEVPT2 energy's on a single-state
DF-CASSCF (below, "The NEVPT2 chain") -- its harmonic
frequencies and normal modes, and (analytic RHF) the IR intensities. Two
methods:

- **analytic**, the default: `hessian::df_rhf` (`../../hessian/`) at the SCF
  of one run of the chain from the molecule to `[scf]`, a nested `Pipeline`;
  beside a `[casscf_gradient]`, `hessian::df_casscf` or
  `hessian::df_sacasscf` at the CASSCF of one run of the chain to
  `[casscf]`; beside a `[nevpt2_gradient]`, `hessian::df_sc_nevpt2` or
  `hessian::df_pc_nevpt2` there. Density-fitted only.
- **numerical**: central differences of the analytic gradient of the chain
  from the molecule to `[gradient]`, rerun as a nested `Pipeline` at each of
  the 6N displaced geometries, as `[optimize]` reruns it at each evaluation
  (`../optimize/README.md`). Over either ERI kind, and the reference the
  analytic Hessian is tested against.

Either way the Hessian is symmetrized, checked, and handed to
`geometry::harmonic_analysis` (`../../geometry/`).

## Module

`free_breakfast.pipeline.hessian`, target `free_breakfast::pipeline::hessian`.
It re-exports `free_breakfast.geometry`, whose `Vibrations` the product
holds, and `free_breakfast.pipeline.gradient`,
`free_breakfast.pipeline.casscf_gradient` and
`free_breakfast.pipeline.nevpt2_gradient`, which re-export every step the
three chains rerun, each a member, and the `MoleculeOutput` `run` reads.

## The step

```cpp
enum class HessianMethod : uint8_t { Analytic, Numerical };

struct HessianOutput {
  std::vector<double> hessian;    // 3N x 3N, row-major, x = 3 A + k, symmetrized
  double asymmetry;               // max |H - H^T| before symmetrizing
  double translational_sum;       // max over x, l of |sum_B H[x][3 B + l]|
  geometry::Vibrations vibrations;  // frequencies (cm^-1) and normal modes
  std::vector<double> dipole_derivative;  // 3N x 3, (x, c) at 3 x + c; analytic only
  std::vector<double> intensities;        // km/mol per frequency; analytic only
};

struct HessianStep {
  static constexpr std::string_view kName = "hessian";

  HessianMethod method{HessianMethod::Analytic};
  double displacement{1e-3};      // h, bohr: the numerical method's
  std::vector<double> coordinates{};  // n rows of n, orthonormal; empty: Cartesian
                                      // (numerical, or the analytic NEVPT2)
  bool ladder{false};                 // the h, h/2, h/4 ladder (CASSCF chains)
  std::optional<std::vector<std::array<int32_t, 2>>> pairs{std::in_place};
                                      // bilinearity; nullopt: every pair
  static constexpr size_t kMaxEveryPairCoordinates = 12;

  // The chain, copied in from the config's own tables by pipeline/config.
  BasisSetStep basis_set{};
  EriStep eri{};
  OrthogonalizerStep orthogonalizer{};
  GuessStep guess{};
  ScfStep scf{};
  CasscfStep casscf{};                                  // read only under the next
  std::optional<CasscfGradientStep> casscf_gradient{};  // set: the CASSCF chain
  std::optional<Nevpt2GradientStep> nevpt2_gradient{};  // set: the NEVPT2 chain

  std::filesystem::path json_path{};  // hessian.json

  Result<HessianOutput> run(View<MoleculeOutput> in, NodeResources &node) const;
};

// The CASSCF chain's guard on one displaced solve (below); host only.
Status check_displaced_casscf(size_t nao, std::span<const double> s,
                              std::span<const double> start_active,
                              std::span<const double> active,
                              std::span<const double> reference_ci,
                              std::span<const double> ci, size_t nroots);
```

`run` takes the geometry, charge and multiplicity of the `MoleculeOutput`
and plans `Pipeline`s over

```
molecule -> basis_set -> spherical_transform -> primitive_pair_classes -> eri
  -> overlap -> orthogonalizer -> core_hamiltonian -> guess -> scf [-> gradient]
```

the molecule a private step standing where `MoleculeStep` stood.

- **Analytic.** One pipeline to `[scf]` at R. Its products -- the molecule,
  the basis, the pair classes, the `DensityFitted` B, L and aux basis, and
  the SCF's C (copied to a buffer of its nao × nmo columns), orbital energies
  and n_occ -- are `hessian::df_rhf`'s inputs, as `[gradient]`'s are
  `gradient::df_rhf`'s. The returned `hessian::RhfTerms` holds every term and
  the CPHF's U^x, which `hessian::rhf_dipole_derivative` then reuses for the
  dipole derivative (no solve of its own); the total and the dipole
  derivative leave the step. `asymmetry` is `RhfTerms::asymmetry`, the
  terms' rounding. The dipole integrals stop at d, so for a basis with an f
  shell the dipole derivative and intensities are left empty, with a warning,
  rather than the step failing. An integral-direct `[eri]` is refused (Unsupported) before
  anything runs, and an ROHF SCF after it, as `[gradient]` refuses it.
  Beside a `[casscf_gradient]`, the chain runs to `[casscf]` instead and
  the Hessian is the root's (below, "The CASSCF chain"); beside a
  `[nevpt2_gradient]`, the NEVPT2 energy's there (below, "The NEVPT2
  chain").
- **Numerical.** For each coordinate q_j -- e_x, x = 3 A + k, unless the
  config gives `coordinates` -- one pipeline to `[gradient]` at R + h q_j
  and one at R - h q_j. Column j of the raw Hessian is
  (g(R + h q_j) - g(R - h q_j)) / 2h in the coordinates, Q g, and its
  asymmetry is recorded; the Hessian is reported in Cartesians, Qᵀ H Q.
  Under `ladder`, eight per coordinate instead, and the bilinearity check
  (below, "The ladder and the bilinearity check").

The Hessian kept is (H + Hᵀ) / 2, and its translational sum, which a Hessian
of an energy invariant under translation has zero, is recorded beside it.
Then `geometry::harmonic_analysis` at the molecule's atoms: 3N − 6 (3N − 5
linear) frequencies in cm⁻¹, ascending, an imaginary one negative, and the
Cartesian normal modes, each logged as a table at info, beside its IR
intensity (km/mol, `hessian::infrared_intensities`) when the dipole
derivative was made. The numerical method makes none: its intensities are
empty. Each pipeline is
freed before the next, so the device holds one geometry's products at a
time; its first error -- an SCF that does not converge, `[gradient]`'s
refusal of an open shell -- is returned as it is.

- **Reads the outer `MoleculeOutput`**, so beside an `[optimize]` (whose
  `final.xyz` the molecule is then loaded from) it runs at the optimum, after
  the optimization: the frequencies' Hessian. It is ordered by that product,
  as any step reading the molecule is. Away from a stationary point the
  frequencies are still computed, the rigid-body motions projected out, but
  are not a molecule's vibrations.
- **The numerical error budget.** The central difference is O(h²) in the
  gradient's third derivative, and the SCF's noise in the gradient is
  divided by 2h. At the default h = 1e-3 bohr and the SCF converged as
  `[gradient]`'s test converges it (`energy_tolerance = 1e-12`,
  `commutator_tolerance = 1e-10`), PySCF's own central difference of
  `df.grad.rhf` sits 4.4e-7 from its analytic Hessian on the skew water
  (`tools/emit/hessian_numerical_reference.py`). The step does not tighten
  `[scf]` itself: a config that wants the Hessian to 1e-6 tightens it, as for
  `[gradient]`. The analytic Hessian is the SCF's only as far as the SCF
  converged too, and its CPHF runs to `hessian::df_rhf`'s default 1e-10.
- **SAD at every geometry**, the configured `[guess]` made afresh, as
  `[optimize]`'s.
- **The chain writes nothing.** `scf.molden` is cleared in its copy, and the
  option-less steps run at their defaults.
- **`Node`-shaped**, because each method runs a `Pipeline` on the node,
  co-run on the executor worker (`../pipeline/README.md`). The analytic
  method is one SCF (or CASSCF) and the Hessian; the numerical one 6N
  SCF + gradient runs, 18 for water.
- **`hessian.json`**, when `json_path` is set:

  ```json
  {"method": "analytic",
   "hessian": [[0.71, ...], ...],
   "asymmetry": 1.4e-14, "translational_sum": 8.7e-12,
   "frequencies": [1776.03, 4113.30, 4211.68],
   "modes": [[0.0, -0.07, ...], ...],
   "dipole_derivative": [[0.39, 0.0, ...], ...],
   "intensities": [80.81, 21.17, 60.56]}
  ```

  the Hessian (hartree / bohr²) as 3N rows of 3N, the frequencies (cm⁻¹) and
  the normal modes, one row of 3N per frequency (`geometry::Vibrations`'s
  layout); when made, the dipole derivative (e) as 3N rows of x, y, z and the
  IR intensities (km/mol), one per frequency. The numerical method adds its
  `"displacement"`, the CASSCF chain the `"root"` differentiated, the
  NEVPT2 chain its `"variant"` (`"sc"` or `"pc"`), and either method, when
  given, its `"coordinates"`.

## The CASSCF chain

With `casscf_gradient` set (pipeline/config sets it when the config has a
`[casscf_gradient]`), the Hessian is that root's of the DF-CASSCF,
state-specific or one root of a state average, by either method.

### Analytic

One pipeline to `[casscf]` at R:

```
molecule -> ... -> guess -> scf -> casscf
```

and its `CasscfOutput` back into the `mcscf::Solution` the Hessians take,
as `[casscf_gradient]` builds it for `mcscf::df_gradient` (the orbitals and
CI vectors on the device, gamma the weighted average of the roots' 1-RDMs):

- **one root**: `hessian::df_casscf` (`../../hessian/README.md`), the
  state-specific Hessian; `[casscf_gradient]` `root` must be 0.
- **a state average**: `hessian::df_sacasscf` at the CASSCF's own weights,
  root `[casscf_gradient]` `root` -- equal weights (with eq. rot) or all
  pairwise different (the rotations in λ).

Its `hessian` is the step's raw Hessian (already symmetrized) and its
`asymmetry` the step's; the largest CP residual is logged beside the root's
energy. Both refuse what they cannot take, and the step returns it as it
is: a saddle (Divergence: the CP solve met negative curvature, as a
state-specific excited root does), two roots within 1e-8 hartree
(Numerical), weights equal for some pairs only (InvalidConfig). A root the
CASSCF does not have is InvalidConfig before the Hessian runs. No dipole
derivative is made, so no IR intensities: the product's and `hessian.json`'s
are empty and absent, and the frequencies are logged alone.

As with the RHF, the analytic Hessian is the CASSCF energy's only as far as
the CASSCF converged: it assumes stationarity and does not check it. The
GPU cases set `[casscf]` `conv_tol_grad = 1e-9`.

### Numerical

The numerical method differences the analytic gradient of the root,
`mcscf::df_gradient`, over 6N displaced CASSCF solves:

```
reference: molecule -> ... -> guess -> scf -> casscf
displaced: molecule -> ... -> eri -> overlap -> core_hamiltonian
             -> reference orbitals -> casscf -> casscf_gradient
```

A displaced CASSCF is a nonlinear solve, and it can converge somewhere else
than the reference's solution moved -- quietly, with a gradient that differences
into a wrong Hessian. Building the design's check met two such failures
(`casscf_hessian.tex` section 11 item 1): a weakly correlated CAS(4,4) that
never converged, and a start that was not orthonormal converging O(h) off with no
error. So:

- **One reference, picked once.** The chain to `[casscf]` runs at the
  geometry first, from the SCF and `[casscf]`'s selection (by index or
  AVAS). Its orbitals and CI vectors are the reference.
- **Each displaced solve starts from the reference's orbitals,
  orthonormalized in the displaced metric**: a private step of this module,
  standing where the `ScfStep` stood, hands the CASSCF C (Cᵀ S(x) C)^-1/2,
  the orthonormal set nearest the reference's orbitals in the displaced
  overlap, which the CasscfStep takes by index as they come (its `ncas` and
  `nelecas` the reference's, no `active`, no `avas`): `[optimize]`'s carried
  orbitals (`../optimize/README.md`), from one fixed reference. Its
  ScfOutput is not an SCF's: the energy is NaN and `d` zero, and only the
  CasscfStep reads it.
- **The CI starts from the reference's, and is checked against it.** The
  displaced CasscfStep's `ci_guess` is the reference's CI vectors
  (`mcscf::solve`'s `ci_guess`, #657), over the same active orbitals moved,
  so the first macro iteration's Davidson starts at the reference's roots
  rather than from its own guesses. Each macro iteration still solves for
  the lowest `nroots` roots of its Hamiltonian, so the start moves the
  iteration counts and not which roots are found; what can change is which
  roots those are, and that is checked. With and without the guess, every
  displaced solve of the four GPU cases below converges to the same roots
  (by their CI overlap) and energies (to 1e-9 of |E_r|, two independently
  converged solves);
  `PipelineCasscfHessianCiGuess` prints the macro-iteration counts.
- **`check_displaced_casscf` refuses a solve that left the reference's
  solution**, as Numerical, naming the geometry: each converged active
  orbital must lie more in the start's active space than out of it (its
  weight Σ_u ⟨start_u|S(x)|t⟩² above 1/2), and each root must be more the
  reference's root of the same index than anything else
  (⟨reference_r|c_r⟩² above 1/2). Both are 1 − O(h²) for a solve that
  followed the reference and about 0 for an orbital swapped in from the core
  or the virtuals, or roots that changed order or let in one from outside
  the average. Its gradient is computed, then refused, never differenced.
- **An unconverged displacement is refused.** `mcscf::solve`'s
  MaxIterationsExceeded is the run's error, its message prefixed with the
  geometry -- `[hessian] at the reference geometry: ...` or `[hessian] at
  coordinate x displaced by +0.001 bohr: ...` -- as is every other error of
  a chain.
- **Density fitting only.** `[casscf_gradient]` takes nothing else, so an
  integral-direct `eri` is refused (Unsupported) before anything runs, by
  either method, here and by pipeline/config.
- **Converge the CASSCF.** The gradient is the CASSCF energy's only as far
  as the CASSCF converged, and its noise is divided by 2h: the GPU cases
  below set `[casscf]` `conv_tol_grad = 1e-9` and `conv_tol = 1e-12`.

### Either method

- **The outer `[casscf]` and `[casscf_gradient]` run too**, once, at the
  molecule: tables are requests. They are the same chain as the reference,
  so the outer CASSCF is the reference solution (the GPU cases check its
  energy).
- **The reference is whichever solution the pick converges to.** A CASSCF
  can have several minima (and saddles) within reach of a pick; the step
  works at the one `[casscf]` finds (and, numerically, holds every
  displaced solve to it). Which one that is is the config's to check, by
  its energy. The analytic method refuses a saddle; the numerical one
  differences around it.
- **Beside an `[optimize]`** over the same `[casscf_gradient]`, the step
  reads the optimum's molecule, as the RHF's does, and picks the active
  space afresh there from the configured selection: the optimizer's carried
  orbitals do not reach it. The GPU case below checks the pick at the
  optimum finds the optimized root.

## The NEVPT2 chain

With `nevpt2_gradient` set (pipeline/config sets it when the config has a
`[nevpt2_gradient]`), the Hessian is that SC- or PC-NEVPT2 total energy's,
E_cas + E_corr, on the single-state DF-CASSCF, by either method.

### Analytic

The default (#818). One pipeline to `[casscf]` at R, as the CASSCF chain's
analytic method runs it, its `CasscfOutput` back into the same
`mcscf::Solution`, and the variant's analytic Hessian there
(`../../hessian/README.md`, "The SC-NEVPT2 Hessian" and "The PC-NEVPT2
Hessian"): `hessian::df_sc_nevpt2` for `variant = "sc"`,
`hessian::df_pc_nevpt2` for `"pc"`, the same calls `[nevpt2_gradient]`
makes for the gradient's (batch 0, PC's floor 0). Its `hessian` is the
step's raw Hessian (already symmetrized) and its `asymmetry` the step's;
the NEVPT2 energy and the Z-vector's residual are logged. Then the step's
symmetrizing, its translational sum, `geometry::harmonic_analysis` and
`hessian.json`, as for the other analytic Hessians; no dipole derivative,
so no IR intensities.

- **`coordinates`, if given, are the call's.** The Hessian is taken in
  them, q_i = Σ_x Q_ix R_x, and reported in Cartesians, Qᵀ H Q, as the
  numerical method reports its own. At a symmetric geometry that is not
  bookkeeping: an empty SC perturber that a direction populates makes the
  HVP non-linear in the direction, so a symmetry-breaking Cartesian
  direction mixes the symmetric block into the broken one. Give the
  symmetry-adapted coordinates there. The only analytic Hessian that takes
  them.
- **Refusals are returned as they are**, the step's error, naming what
  failed: SC's pairs of coordinates on which the empty perturbers' energy
  is not a quadratic form (at `nh3_cs_sto3g` in its symmetry-adapted
  coordinates, the whole A″ × A″ off-diagonal block, ten pairs: #894's
  kink, where that block has no second derivative); PC's class kept below
  1e-10, its Schur lift within 1e-3 bohr, and a degenerate inactive pair a
  coordinate splits; the CP solve's saddle (Divergence). No Hessian, so no
  `hessian.json`.
- **Stationarity is assumed, not checked**, as for the CASSCF chain:
  converge `[casscf]` tightly (`conv_tol_grad = 1e-10` in the example and
  the GPU cases).

### Numerical

Central differences of `nevpt2::sc_gradient` / `pc_gradient` over the 6N
displaced geometries (`nuclear_hessian.tex` section 4, Algorithm 1 in
Cartesian form). It is the CASSCF chain's numerical method with
`[nevpt2_gradient]` where `[casscf_gradient]` stood:

```
reference: molecule -> ... -> guess -> scf -> casscf
displaced: molecule -> ... -> eri -> overlap -> core_hamiltonian
             -> reference orbitals -> casscf -> nevpt2_gradient
```

- **The CASSCF chain's guards, as they are.** One reference picked at R;
  each displaced solve from the reference's orbitals orthonormalized in the
  displaced metric and its CI; `check_displaced_casscf` on each; an
  unconverged solve, or any error of a chain (PC's refusals among them: no
  positive metric eigenvalue, a pole), the run's error, naming the geometry.
  The displaced CasscfStep canonicalizes its core and virtuals, as the
  gradient needs.
- **The gradient's hazards are logged per displacement**, at info: the
  NEVPT2 energy, max |z^c|, the smallest inactive gap (what z^c divides by)
  and the Z-vector's residual and iterations; and once, after the 6N, the
  worst of each. A z^c that grows across the displacements is a near
  degeneracy of the canonical inactive orbitals, where the energy itself is
  not smooth (`test/gtest/data/nevpt2_hessian/README.md`): the differences
  there are not a Hessian.
- **The step h.** The default 1e-3 bohr is the CASSCF chain's; under the
  ladder (the default here) it is the coarsest step, and the fixture's
  0.03 makes the finest 0.0075. The gradient is the NEVPT2 energy's only as
  far as the CASSCF converged (its Lagrangian assumes stationarity), so
  converge `[casscf]` tightly; the GPU cases below set
  `conv_tol_grad = 1e-10`. SC's energy kinks wherever a perturber's norm
  passes near zero or two canonical inactive orbitals nearly cross
  (#894), so a step must stay inside the smooth region: the skew fixture
  was chosen where it is, at 0.06 bohr.

### Either method

**What it covers, refused before anything runs** (here and by
pipeline/config), because `hessian::df_sc_nevpt2` and `df_pc_nevpt2` take
none of it and the numerical method is validated on the same ground: an
integral-direct `eri` (Unsupported), a state-averaged `casscf` (`nroots`
above 1, Unsupported), a `quasi_degenerate` gradient (Unsupported:
QD-NEVPT2 has no Hessian here), a `frozen_core` other than 0
(InvalidConfig: the Hessian is all-electron), and a `casscf_gradient`
beside it (InvalidConfig: one energy). The outer `[casscf]` and
`[nevpt2_gradient]` run too, once, at the molecule, as the CASSCF chain's
do.

## The ladder and the bilinearity check

`nuclear_hessian.tex` §3's diagnostic (#812): at a symmetric geometry an
NEVPT2 energy can have a second derivative along every line and still no
Hessian, because along a pair of directions that leave the symmetry it is
not a quadratic form (SC's empty perturbers, #894). A difference Hessian
does not see that; this does. It follows #803's fixture,
`tools/nevpt2_hessian/` (`stencil.py`), over the analytic gradient:

- **The ladder**, `ladder = true` (the default under `[nevpt2_gradient]`,
  off otherwise). Column j at s = h, h/2 and h/4, h the `displacement`, is
  the fourth-order central stencil (g(−2s) − 8 g(−s) + 8 g(s) − g(2s)) / 12s
  along q_j: eight gradients per coordinate (h/2's 2s is h's s), not two.
  Per element, the ratio (D(h) − D(h/2)) / (D(h/2) − D(h/4)) is 16 where
  truncation dominates. It is judged, in (12.8, 19.2), only where the finer
  difference is past 10 floors (the tree's visibility rule; nearer the
  floor the ratio is the noise's) -- below that it is "below the floor",
  null in `"ratios"`, never unsettled. The element's bar is |D(h/2) −
  D(h/4)|, or the floor where that is below it. A judged ratio outside the
  window is *unsettled*: logged at warn and listed, not refused. The
  Hessian is the finest rung's, symmetrized. (The fixture judges past one
  floor and flags a coarse difference far above a sub-floor fine one; at
  this floor, with the gradient's own noise far below the fixture's, ratios
  between one and ten floors read 0.3 to −46 on the skew case: noise.)
- **The noise floor: the difference noise, measured.** σ_g, the
  gradient's noise, is the larger of two measured terms, both in
  `hessian.json`'s `"ladder"."noise"` with the one that won (`"from"`):
  - *restarts*: at +h along the first four coordinates, the gradient from
    the reference's start and from four restarts, each the reference's
    orbitals rotated by up to 1e-3 rad in every non-redundant pair (a fixed
    seed), reconverged; the root mean square of the per-component sample
    standard deviations (the fixture's measurement, on the gradient). It
    sees the spread of where a solve lands, about 1e-13 here, and not what
    every solve leaves unconverged.
  - *solves*: what the production tolerances leave in every gradient,
    measured. At the same four points, the production solve is re-solved
    with every tolerance 100 times tighter -- `[casscf]`'s conv_tol and
    conv_tol_grad (its CI's floor follows conv_tol_grad,
    `mcscf::ci_tolerance_for`) and the NEVPT2 Z-vector's
    (`Nevpt2GradientStep::zvector_tolerance`, 1e-10 to 1e-12) -- started
    from the production solve's orbitals and CI. σ_solves is the root mean
    square over the points and components of g_production − g_tight: the
    root mean square, as the restarts term, because the floor's model is an
    independent error of size σ in each gradient a stencil reads.
  - *the bound, rejected as the floor*: the gradient is ∂L/∂R of a
    Lagrangian stationary in the parameters x and multipliers z, so a
    residual r moves it by r · dx/dR to first order, at most ‖r‖ ‖dx/dR‖
    (‖r‖ the largest CASSCF ‖g‖ + CI residual + Z-vector residual of the
    production solves, ‖dx/dR‖ the largest angle a solve from the
    reference's start turned its active orbitals and root by, per bohr,
    from `check_displaced_casscf`'s measures). It is kept in `hessian.json`
    (`"solve_bound"`) as a check: the measurement must lie under it. As the
    floor it was 70 to 360 times the measurement (Cauchy–Schwarz is loose),
    judged no ratio on any case and hid nh3 SC's (8,11).

  Measured (CUDA, RTX 3080, one run each):

  | case | σ restarts | σ solves (measured) | bound | floor | judged | ratios |
  |---|---|---|---|---|---|---|
  | `h2co_skew_sto3g` SC | 4.1e-14 | 1.45e-12 | 1.04e-10 | 5.5e-9 | 8 of 144 | [16.006, 16.070] |
  | `h2co_skew_sto3g` PC | 3.8e-14 | 1.52e-12 | 1.04e-10 | 5.8e-9 | 8 of 144 | [15.998, 16.063] |
  | `nh3_cs_sto3g` SC | 1.0e-13 | 2.92e-13 | 8.0e-11 | 1.1e-9 | 17 of 144 | [15.053, 16.578] |
  | `nh3_cs_sto3g` PC | 7.8e-14 | 2.90e-13 | 7.2e-11 | 1.1e-9 | 16 of 144 | [15.915, 16.034] |

  The measurement won every time. With the restarts' term alone (1e-13 at
  most) ratios one to ten floors up read 0.3 to −46 on the skew case: the
  noise the restarts do not see.

  One stencil at s multiplies independent noise by √(Σ w²) / s, so the
  finer difference carries at most σ_g (gain(h/2) + gain(h/4)); the floor
  is 20 σ_g (gain(h/2) + gain(h/4)), so a one-sigma noise moves a ratio by
  5% (the fixture's factor). The RHF chain has no orbitals to restart
  from, so the ladder is the CASSCF chains' only (Unsupported over the
  RHF's).
- **Eq. (bilinear)** on each pair (a, b) of `pairs`: the same ladder of the
  directional derivative u · g along u = (q_a + q_b) / √2, its finest rung
  against (H_aa + H_bb + H_ab + H_ba) / 2 of the finest rung's own
  (unsymmetrized) elements. The pair fails when the residual is above u's
  bar plus half the four elements' bars: the difference noise and the
  truncation both. `pairs` defaults to every pair under
  `[nevpt2_gradient]`, up to 12 coordinates (66 pairs, 528 gradients;
  above, none, logged at warn), and to none otherwise; a list names its own,
  and `pairs = []` none. Eight gradients each.
- **A failing pair withholds the frequencies.** The Hessian has no normal
  modes where it is not a bilinear form: `geometry::harmonic_analysis` is
  not run, `hessian.json` is written without `"frequencies"` and `"modes"`
  (with the Hessian and every ladder, for the diagnosis), and the step
  returns Numerical naming the pairs, so the run fails rather than publish a
  frequency. An error rather than a flag, so nothing downstream reads a
  Hessian that is not one.
- **No point group is detected.** A failure on Cartesian pairs names only
  the pair. The config's `coordinates` (n orthonormal rows, each a
  Cartesian displacement) put the check in symmetry-adapted ones, which name
  the failing block: the fixture's, for ammonia.
- **What it finds on the fixture's symmetric case, and why that is more
  than the fixture's three pairs.** On `nh3_cs_sto3g` SC (CUDA, one run,
  644 gradients in 57 s), σ_g = 2.9e-13 and the floor 1.1e-9. Every A′ and
  cross-irrep pair passes, and all ten A″ × A″ pairs fail:

  | pair | residual | allowed | ×allowed | its ladder's ratio | fixture's residual / allowed |
  |---|---|---|---|---|---|
  | (7,8) | +4.36e-6 | 4.82e-8 | 90 | below floor | −3.5e-7 / 3.3e-7, fails |
  | (7,9) | +2.08e-7 | 7.09e-8 | 2.9 | below floor | 1.1e-7 / 2.9e-7 |
  | (7,10) | −1.01e-6 | 9.00e-8 | 11 | 15.36 | 1.1e-7 / 2.4e-7 |
  | (7,11) | −8.33e-8 | 6.26e-8 | 1.33 | below floor | −4.7e-8 / 1.1e-7 |
  | (8,9) | −3.66e-6 | 8.57e-9 | 427 | below floor | −3.1e-7 / 2.9e-6 |
  | (8,10) | −1.72e-6 | 4.72e-9 | 365 | below floor | −2.3e-6 / 1.8e-6, fails |
  | (8,11) | +3.77e-7 | 4.65e-9 | 81 | below floor | −5.0e-7 / 2.8e-7, fails |
  | (9,10) | +5.98e-7 | 1.43e-8 | 42 | below floor | 1.6e-8 / 1.0e-6 |
  | (9,11) | −9.04e-8 | 1.29e-8 | 7.0 | below floor | 1.9e-8 / 2.4e-7 |
  | (10,11) | −9.64e-8 | 4.72e-9 | 20 | below floor | −5.6e-7 / 6.2e-7 |

  The fixture records three of them, (7,8), (8,10) and (8,11). Its
  gradient is a δ = 5e-3 central stencil of energies, and off the mirror
  that stencil straddles #894's kink (SC's empty perturbers, whose energy
  is t² Q(direction)): its points leave the line at a distance comparable
  to the ladder's finest step, 0.0075, so its A″ gradients carry errors
  that make its bars there 1e-7 to 3e-6, wide enough to hide seven of the
  ten. The analytic gradient has no stencil of its own, so its bars sit at
  the truncation and the noise and the whole A″ × A″ block shows, as the
  analytic route found (#816's refusal of the whole block). The test holds
  the fixture's three as failing and every failure inside A″ × A″ (#812's
  amended acceptance), not "exactly ten": (7,11) at 1.33 times its
  allowance is too close to call either way. Its pairs' own ladders are
  mostly below the judging line: their finer differences are 1e-10 to
  3e-8.
- **`hessian.json`** under the ladder adds `"ladder"`: `"steps"` (h, h/2,
  h/4), `"rungs"` (the three D, each n rows, element [i][j] q_i · D along
  q_j, unsymmetrized, in the coordinates), `"ratios"` (null below the
  floor), `"bar"`, `"unsettled"` ([i, j]) and `"noise"` (`"sigma_g"`,
  `"floor"`, the points, starts and kick); `"bilinearity"`, per pair its
  `"pair"`, `"ladder"`, `"ratio"`, `"settled"`, `"directional"`,
  `"predicted"`, `"residual"`, `"allowed"` and `"fails"`; and
  `"failing_pairs"`. With `coordinates`, `"coordinates"` too.

## Config

[`examples/hessian.toml`](../../../examples/hessian.toml) is a whole config:
the analytic DF-RHF Hessian of water in cc-pVDZ, and its `hessian.json`.

```toml
[hessian]
method = "analytic"    # the default; or "numerical"
displacement = 1e-3    # bohr, positive: the numerical method's only
```

With a `[casscf_gradient]` in the config, `[casscf]` and it are copied in
too, and the Hessian is that root's, analytic (the default) or numerical
(`[gradient]` beside it is refused, InvalidConfig):

```toml
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"
[casscf]
ncas = 4
nelecas = 4
active = [2, 3, 5, 6]
nroots = 2
weights = [0.7, 0.3]
conv_tol = 1e-12
conv_tol_grad = 1e-9
[casscf_gradient]
root = 0
[hessian]
method = "analytic"    # or "numerical"
```

With a `[nevpt2_gradient]` in the config instead, `[casscf]` and it are
copied in, and the Hessian is that NEVPT2 energy's. Analytic, the default,
[`examples/nevpt2_hessian.toml`](../../../examples/nevpt2_hessian.toml) is
a whole config: the SC-NEVPT2 Hessian of the skew formaldehyde of #803's
`h2co_skew_sto3g` fixture, CASSCF(4,4) in STO-3G, its frequencies and
`hessian.json` (about 6 s on one card). The geometry is C1 on purpose: at
`nevpt2.toml`'s symmetric water (cc-pVDZ) SC is refused, Numerical (#952):
a B2 displacement mixes two virtuals 1.4e-3 Eh apart within 1e-3 bohr, and
SC's B2 element there (14063 cm⁻¹ against PC's 4028) describes a narrower
region than a vibration (`../../hessian/README.md`, "Mixing inactive
pairs"). PC is not refused there. It takes `coordinates` and nothing else
of the numerical method's:

```toml
[nevpt2_gradient]
variant = "sc"         # or "pc"
[hessian]
method = "analytic"    # the default
coordinates = [[1, 0, 0, ...], ...]  # optional: n orthonormal rows of n
```

By either method `quasi_degenerate` (Unsupported), a `frozen_core` other
than 0 (InvalidConfig), a state average (`[nevpt2_gradient]`'s own
refusal, Unsupported) and an integral-direct `[eri]` (Unsupported) are
refused. Numerically:

```toml
[casscf]
ncas = 4
nelecas = 4
conv_tol = 1e-12
conv_tol_grad = 1e-10
[nevpt2_gradient]
variant = "pc"         # or "sc"
[hessian]
method = "numerical"
displacement = 0.03    # bohr: the ladder's h, so its finest step is 0.0075
ladder = true          # the default here; false for 2 gradients per coordinate
pairs = [[7, 8]]       # the default here: every pair, up to 12 coordinates
coordinates = [[1, 0, 0, ...], ...]  # optional: n orthonormal rows of n
```

`ladder` and `pairs` are the numerical method's, and `coordinates` the
numerical method's or the analytic NEVPT2 Hessian's (each refused beside
any other analytic one, InvalidConfig); `pairs` needs the ladder
(InvalidConfig without it), and the ladder a CASSCF chain (Unsupported over
the RHF's). `coordinates` must be n square rows (InvalidConfig); that n is
the molecule's 3 n_atom and the rows orthonormal (to 1e-10) is checked by
the step before anything runs, as are pairs inside n and none named twice.

Two of `[gradient]`, `[casscf_gradient]` and `[nevpt2_gradient]` beside a
`[hessian]` are refused (InvalidConfig).

The chain is the config's own `[basis_set]`, `[eri]`, `[orthogonalizer]`,
`[guess]` and `[scf]`, copied in by `pipeline/config` as `[optimize]`'s is;
one loading `from` a file is refused. The analytic method over an
integral-direct `[eri]` -- the default one -- is refused there too
(Unsupported), and so is a `displacement` beside it.

## Tests

- The step's contract and its config: `test/pipeline/hessian_host.cpp`
  (`static_assert`s on the run shape, the View and the product; `method`
  read, analytic by default, and refused; `displacement` read and refused;
  `ladder`, `pairs` and `coordinates` read, the ladder over every pair by
  default beside a `[nevpt2_gradient]` and off otherwise, and their
  refusals;
  beside the analytic method among the refusals; the analytic method over
  integral-direct ERIs refused; the chain copied from the tables; `from`
  refused on a chain table; and beside `[optimize]`, the molecule it reads
  loaded from `final.xyz`); the CASSCF chain copied in beside a
  `[casscf_gradient]`, and its refusals (`[gradient]`
  beside it, no `[casscf]` space, an integral-direct `[eri]`); the analytic
  method beside a `[casscf_gradient]` (by default and by name, and beside an
  `[optimize]` holding the same CASSCF chain); the analytic method beside a
  `[nevpt2_gradient]`, SC and PC (by default and by name, no ladder, its
  `coordinates` read, the numerical keys refused) and its refusals before
  running (a state average, an integral-direct `[eri]` written or by
  default, `frozen_core` 1 and `"auto"`, `quasi_degenerate`);
  `examples/nevpt2_hessian.toml` read as the binary reads it; and
  `check_displaced_casscf` over a
  non-orthonormal basis, passing a solve that followed and refusing an active
  orbital swapped out and two roots exchanged -- the forced refusals.
- `test/gtest/pipeline_hessian_gpu.cpp`, TOML configs in cc-pVDZ /
  cc-pVDZ-JKFIT run by `Calculation`:
  - `AnalyticAndNumericalDfRhfHessiansMatchPyscf`: the skew water by both
    methods, each `hessian.json` against the product, against
    `pyscf.df.hessian.rhf` over the same aux set
    (`test/gtest/data/hessian_numerical/hessian_numerical.json`) to 1e-5
    (numerical) and 1e-7 (analytic), against each other to 1e-5 (the central
    differences' error), and their frequencies to 0.1 cm⁻¹; the asymmetries
    and translational sums checked and printed.
  - `OptimizedWaterFrequenciesMatchPyscf`: the water optimized from
    `data/gradient/gradient.json`'s start (Opt=Tight's thresholds, as
    `pipeline_optimize_gpu.cpp`), then the analytic `[hessian]` at the
    optimum, its frequencies against PySCF's `thermo.harmonic_analysis` at
    PySCF's DF-RHF minimum (`data/rhf_hessian/rhf_hessian.json`'s
    `h2o_ccpvdz_opt`) to 0.1 cm⁻¹.

Measured on the RTX 3080 (CUDA): at the skew water the numerical Hessian is
3.6e-7 from PySCF (asymmetry 3.7e-7, translational sum 2.0e-8) in 2.0 s;
the analytic one 4.2e-11 from PySCF and 3.6e-7 from the numerical one
(asymmetry 9.3e-15, translational sum 8.7e-12) in 1.3 s, its frequencies
2.3e-3 cm⁻¹ from the numerical one's. The optimized water's frequencies
(1776.032, 4113.305, 4211.682 cm⁻¹) are 2.2e-4 cm⁻¹ from PySCF's, the
optimization and the Hessian 2.0 s together.

- `test/gtest/pipeline_casscf_hessian_gpu.cpp`, the CASSCF chain, TOML
  configs run by `Calculation`, against `data/casscf_hessian/`
  (`tools/emit/casscf_hessian_reference.py`, the same cc-pVDZ-JKFIT fit):
  the design's stretched water, DF-CASSCF(4,4), root 0, `[casscf]` at
  `conv_tol_grad = 1e-9`. Each case first holds the outer CASSCF -- the
  reference's chain -- to the fixture's E_SA to 1e-8, so the Hessian is
  differenced around the fixture's minimum and not the saddle PySCF's own
  solve finds (both two-root cases and cc-pVDZ) or another minimum: the
  (0.7, 0.3) average reaches three by index from the RHF orbitals (E_SA
  −75.80024 from [2, 3, 5, 6], −75.81939 from [2, 4, 5, 6], the fixture's
  −75.81993 from [3, 4, 5, 6]), so its case picks [3, 4, 5, 6] and the
  others [2, 3, 5, 6]. Then the Hessian against the fixture's `relaxed_fd`
  (a fourth-order difference of the relaxed gradient) to the issue's 1e-5,
  `hessian.json` against the product, and the asymmetry and translational
  sum under 1e-5, all printed. `h2o_631g`, state-specific, is the fast case;
  `h2o_631g_sa2` (0.5, 0.5), `h2o_631g_sa2_unequal` (0.7, 0.3) and
  `h2o_ccpvdz` are slow-gated. Beside them, `[casscf]` cut to two macro
  iterations: MaxIterationsExceeded, `[hessian] at the reference geometry:
  ...`. And `PipelineCasscfHessianCiGuess`, per case (`h2o_631g` fast, the
  others slow-gated): every one of the 6N displaced solves from the
  reference's orbitals run with the reference's CI as `ci_guess` and
  without, held to the same roots (CI overlap squared above 1/2) and
  energies (1e-9 of |E_r|: two independently converged solves, not one by
  two paths), the counts printed and not gated. Measured on the gfx1200
  (HIP), one run:

| Case | macro iterations, with / without | Davidson iterations, with / without | max \|ΔE_r\| / \|E_r\| |
|---|---|---|---|
| `h2o_631g` | 66 / 66 | 668 / 840 | 3.4e-15 |
| `h2o_631g_sa2` | 66 / 66 | 647 / 754 | 1.3e-14 |
| `h2o_631g_sa2_unequal` | 62 / 60 (1 fewer, 3 more) | 606 / 659 | 3.4e-13 |
| `h2o_ccpvdz` | 72 / 72 | 705 / 888 | 3.2e-15 |

  The guess buys Davidson iterations (7-21%), not macro iterations: those
  stop on ||g||, which the first CI's start does not move, and a
  displacement's count shifts by one between runs at the threshold (an
  earlier run had `h2o_631g_sa2_unequal` at 66 / 66).

Measured on the gfx1200 (HIP), against `relaxed_fd`:

| Case | max \|H − relaxed_fd\| | asymmetry | translational sum | time |
|---|---|---|---|---|
| `h2o_631g` | 3.9e-8 | 6.3e-8 | 2.0e-8 | 4.6 s |
| `h2o_631g_sa2` | 1.6e-7 | 6.5e-8 | 4.4e-8 | 17.9 s |
| `h2o_631g_sa2_unequal` | 7.4e-8 | 1.9e-8 | 1.2e-8 | 13.9 s |
| `h2o_ccpvdz` | 5.7e-8 | 1.9e-8 | 9.5e-9 | 5.5 s |

  The analytic method, `hessian::df_casscf` / `df_sacasscf` through the
  step, against `relaxed_fd` to 1e-7 (all in the fast tier):
  `h2o_631g` and `h2o_ccpvdz` state-specific, and root 1 of both two-root
  averages (equal weights, with eq. rot; and (0.7, 0.3), the rotations in
  λ). `StateAveragedRootAnalyticMatchesNumerical` (slow-gated) runs the
  analytic `[hessian]` for root 1 of `h2o_631g_sa2`, then the step again,
  numerically, at the Calculation's molecule: the two to 1e-5, their
  frequencies to 0.1 cm⁻¹. `OptimizedWaterAnalyticFrequenciesMatchNumerical`
  (fast) optimizes the CASSCF water of `data/casscf_gradient/casscf_optimize.json`
  from its start (as `pipeline_optimize_gpu.cpp` does), checks the optimum's
  energy and that the outer CASSCF, picked afresh there, is the optimized
  root, then the analytic `[hessian]` at the optimum -- three positive
  frequencies, in the product and `hessian.json` -- and the numerical one
  beside it, to 1e-5 and 0.1 cm⁻¹.

Measured on the gfx1200 (HIP), analytic:

| Case | max \|H − relaxed_fd\| | asymmetry | translational sum | time |
|---|---|---|---|---|
| `h2o_631g` | 1.8e-9 | 2.3e-10 | 8.1e-11 | 1.9 s |
| `h2o_ccpvdz` | 9.0e-11 | 8.4e-11 | 1.0e-10 | 3.2 s |
| `h2o_631g_sa2` root 1 | 1.6e-9 | 1.9e-15 | 3.0e-12 | 8.5 s |
| `h2o_631g_sa2_unequal` root 1 | 9.7e-10 | 5.6e-13 | 1.0e-11 | 6.0 s |

Analytic against numerical: `h2o_631g_sa2` root 1, 2.4e-7, frequencies
7.1e-4 cm⁻¹ (24 s both); the optimized water, 3.0e-7, frequencies
2.3e-3 cm⁻¹ (1717.748, 3741.433, 3852.114 cm⁻¹), optimization and analytic
Hessian 4.8 s, the numerical one 4.2 s. The CUDA card was not run. No
OpenMolcas comparison (the issue's optional one) was made.

- `test/gtest/pipeline_nevpt2_hessian_gpu.cpp`, the NEVPT2 chain and its
  ladder (#804, #812), against #803's `test/gtest/data/nevpt2_hessian/`
  (STO-3G / cc-pVDZ-JKFIT, `conv_tol_grad = 1e-10`), each run's outer
  CASSCF and NEVPT2 energies first held to the fixture's to 1e-8. Slow:
  `ScSkewLadderIsBilinearAndMatchesFixture` and
  `PcSkewLadderIsBilinearAndMatchesFixture` (`h2co_skew_sto3g`, every
  block defined: the config's ladder at h = 0.03 over all 66 pairs; no pair
  fails, the frequencies are published, every judged ratio in (12.8, 19.2),
  no unsettled element, and the finest rung's Hessian within 2 × (the
  fixture's `bar` + its own) of the fixture's `hessian`);
  `ScSymmetricFailsOnlyItsAPrimePrimeBlock` (`nh3_cs_sto3g` in the
  fixture's coordinates: the fixture's three pairs fail, every failing pair
  is in A″ × A″, the step returns Numerical naming them and `hessian.json`
  has no frequencies) and `PcSymmetricIsBilinear` (no pair fails). Fast:
  the config's refusals (`quasi_degenerate`; `frozen_core` 1 and `"auto"`;
  a state average; integral-direct ERIs; a `[casscf_gradient]` beside it)
  and the step's own, run directly on a planned step changed after
  planning (the ladder under the analytic method among them), each
  returned at once naming no geometry. Measured on the RTX 3080 (CUDA), one run each: the skew SC 133 s and PC
  150 s (644 gradients each), no pair failing (the worst, (0,2), at 0.07 of
  its allowance), every judged ratio in [15.998, 16.070],
  the Hessian within 0.025 of its allowance of the fixture's; nh3 SC 57 s,
  the ten A″ × A″ pairs failing and nothing else, (8,11) at 81 times its
  allowance; nh3 PC 62 s, none (worst 0.07). The noise table above has
  each σ term.

- `test/gtest/pipeline_nevpt2_analytic_hessian_gpu.cpp`, the analytic
  method over the NEVPT2 chain (#818), its wiring and not the Hessian (that
  is `hessian::df_sc_nevpt2`'s and `df_pc_nevpt2`'s own tests'), on #803's
  fixtures with `[casscf]` at `conv_tol_grad = 1e-10`:
  `ScSkewMatchesTheDirectCall` (fast) and `PcSkewMatchesTheDirectCall`
  (slow-gated): `h2co_skew_sto3g` run by `Calculation`, `hessian.json`
  and the product against the variant's call made directly on the
  Calculation's outer products (the same chain the step reruns) to 1e-10
  of the largest element, every frequency published;
  `ScSymmetricRefusalNamesTheAPrimePrimeBlock` (fast): `nh3_cs_sto3g` in
  the fixture's coordinates, the step's error Numerical naming exactly the
  ten pairs of the A″ × A″ off-diagonal block, and no `hessian.json`.
  Measured, one run each, gfx1200 (`--preset hip`) / RTX 3080
  (`--preset default`): SC 1.0e-12 / 1.5e-13 of the largest element,
  10.1 / 7.9 s; PC 8.1e-12 / 7.3e-12, 10.2 / 8.4 s; the refusal 5.3 /
  3.1 s. `examples/nevpt2_hessian.toml` through the binary gives the same
  six frequencies on both cards, 6 / 5 s.

- `test/gtest/infrared_gpu.cpp`'s `PipelineInfrared`: the analytic
  `[hessian]` at PySCF's DF-RHF minimum of water
  (`data/dipole_derivative/dipole_derivative.json`'s `h2o_ccpvdz_opt`), its
  dipole derivative against PySCF's to 1e-8, frequencies to 0.1 cm⁻¹ and IR
  intensities to 0.1 km/mol (the issue's bar), and `hessian.json` holding the
  same.

## Files

| File | Role |
|------|------|
| `hessian.cppm` | Module interface: `HessianMethod`, `HessianOutput`, `HessianStep` |
| `hessian.cpp` | `check_displaced_casscf`; the private reference-orbitals step; the ladder's stencil, verdict and restarts; `HessianStep::run`: its refusals, the analytic Hessian (RHF with its dipole derivative, CASSCF, SC- or PC-NEVPT2) over one nested pipeline, or the nested pipelines along each coordinate (RHF, CASSCF or NEVPT2 gradients) and their central differences, or the ladder, its noise and the bilinearity check; the two checks, the harmonic analysis, the IR intensities and `hessian.json` |
| `CMakeLists.txt` | The module library |
