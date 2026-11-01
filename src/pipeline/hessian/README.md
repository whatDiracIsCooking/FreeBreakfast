# Pipeline: Hessian

The closed-shell RHF Cartesian Hessian at the pipeline's molecule -- or,
numerically, a DF-CASSCF root's (below, "The CASSCF chain") -- its harmonic
frequencies and normal modes, and (analytic) the IR intensities. Two methods:

- **analytic**, the default: `hessian::df_rhf` (`../../hessian/`) at the SCF
  of one run of the chain from the molecule to `[scf]`, a nested `Pipeline`.
  Density-fitted only.
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
holds, and `free_breakfast.pipeline.gradient` and
`free_breakfast.pipeline.casscf_gradient`, which re-export every step the two
chains rerun, each a member, and the `MoleculeOutput` `run` reads.

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

  // The chain, copied in from the config's own tables by pipeline/config.
  BasisSetStep basis_set{};
  EriStep eri{};
  OrthogonalizerStep orthogonalizer{};
  GuessStep guess{};
  ScfStep scf{};
  CasscfStep casscf{};                                  // read only under the next
  std::optional<CasscfGradientStep> casscf_gradient{};  // set: the CASSCF chain

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
- **Numerical.** For each coordinate x = 3 A + k, one pipeline to
  `[gradient]` at R + h e_x and one at R - h e_x. Row x of the raw Hessian
  is (g(R + h e_x) - g(R - h e_x)) / 2h, and its asymmetry is recorded.

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
  method is one SCF and the Hessian; the numerical one 6N SCF + gradient
  runs, 18 for water.
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
  `"displacement"`, and the CASSCF chain the `"root"` differenced.

## The CASSCF chain

With `casscf_gradient` set (pipeline/config sets it when the config has a
`[casscf_gradient]`), the numerical method differences the analytic gradient
of that root of the DF-CASSCF, `mcscf::df_gradient`, state-specific or one
root of a state average, over 6N displaced CASSCF solves:

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
- **The CI is checked against the reference's, not started from it.**
  `mcscf::solve` solves each macro iteration's CI for the lowest `nroots`
  roots of its Hamiltonian, the first from its own guesses, and takes no
  starting vectors; those roots are the eigenvectors whatever the start, so
  a reference start would change the Davidson's iteration count and not
  which roots it finds. What can change is which roots those are, and that
  is checked.
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
- **Density fitting only, numerical only.** `[casscf_gradient]` takes
  nothing else, so an integral-direct `eri` is refused before anything runs,
  here and by pipeline/config; the analytic CASSCF Hessian is not wired into
  the step, so `method = "analytic"` beside it is refused too
  (Unsupported).
- **Converge the CASSCF.** The gradient is the CASSCF energy's only as far
  as the CASSCF converged, and its noise is divided by 2h: the GPU cases
  below set `[casscf]` `conv_tol_grad = 1e-9` and `conv_tol = 1e-12`.
- **The outer `[casscf]` and `[casscf_gradient]` run too**, once, at the
  molecule: tables are requests. They are the same chain as the reference,
  so the outer CASSCF is the reference solution (the GPU cases check its
  energy).
- **The reference is whichever solution the pick converges to.** A CASSCF
  can have several minima (and saddles) within reach of a pick; the step
  differences around the one `[casscf]` finds, and holds every displaced
  solve to it. Which one that is is the config's to check, by its energy.

## Config

```toml
[hessian]
method = "analytic"    # the default; or "numerical"
displacement = 1e-3    # bohr, positive: the numerical method's only
```

With a `[casscf_gradient]` in the config, `[casscf]` and it are copied in
too, and the numerical method differences that root's gradient (the
analytic one beside it is refused, Unsupported, and `[gradient]` beside it,
InvalidConfig):

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
method = "numerical"
```

The chain is the config's own `[basis_set]`, `[eri]`, `[orthogonalizer]`,
`[guess]` and `[scf]`, copied in by `pipeline/config` as `[optimize]`'s is;
one loading `from` a file is refused. The analytic method over an
integral-direct `[eri]` -- the default one -- is refused there too
(Unsupported), and so is a `displacement` beside it.

## Tests

- The step's contract and its config: `test/pipeline/hessian_host.cpp`
  (`static_assert`s on the run shape, the View and the product; `method`
  read, analytic by default, and refused; `displacement` read and refused,
  beside the analytic method among the refusals; the analytic method over
  integral-direct ERIs refused; the chain copied from the tables; `from`
  refused on a chain table; and beside `[optimize]`, the molecule it reads
  loaded from `final.xyz`); the CASSCF chain copied in beside a
  `[casscf_gradient]`, and its refusals (the analytic method, `[gradient]`
  beside it, no `[casscf]` space); and `check_displaced_casscf` over a
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
  ...`.

Measured on the gfx1200 (HIP), against `relaxed_fd`:

| Case | max \|H − relaxed_fd\| | asymmetry | translational sum | time |
|---|---|---|---|---|
| `h2o_631g` | 3.9e-8 | 6.3e-8 | 2.0e-8 | 4.6 s |
| `h2o_631g_sa2` | 1.6e-7 | 6.5e-8 | 4.4e-8 | 17.9 s |
| `h2o_631g_sa2_unequal` | 7.4e-8 | 1.9e-8 | 1.2e-8 | 13.9 s |
| `h2o_ccpvdz` | 5.7e-8 | 1.9e-8 | 9.5e-9 | 5.5 s |

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
| `hessian.cpp` | `check_displaced_casscf`; the private reference-orbitals step; `HessianStep::run`: the analytic Hessian and dipole derivative over one nested pipeline, or the 6N nested pipelines and their central differences; the two checks, the harmonic analysis, the IR intensities and `hessian.json` |
| `CMakeLists.txt` | The module library |
