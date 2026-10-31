# Pipeline: Hessian

The closed-shell RHF Cartesian Hessian at the pipeline's molecule, its
harmonic frequencies and normal modes, and (analytic) the IR intensities. Two
methods:

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
holds, and `free_breakfast.pipeline.gradient`, which re-exports every step the
chain reruns, each a member, and the `MoleculeOutput` `run` reads.

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

  std::filesystem::path json_path{};  // hessian.json

  Result<HessianOutput> run(View<MoleculeOutput> in, NodeResources &node) const;
};
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
  `"displacement"`.

## Config

```toml
[hessian]
method = "analytic"    # the default; or "numerical"
displacement = 1e-3    # bohr, positive: the numerical method's only
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
  loaded from `final.xyz`).
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
| `hessian.cpp` | `HessianStep::run`: the analytic Hessian and dipole derivative over one nested pipeline, or the 6N nested pipelines and their central differences; the two checks, the harmonic analysis, the IR intensities and `hessian.json` |
| `CMakeLists.txt` | The module library |
