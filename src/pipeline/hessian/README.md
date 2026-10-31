# Pipeline: Hessian

The closed-shell RHF Cartesian Hessian at the pipeline's molecule,
semi-numerically: central differences of the analytic gradient of the chain
from the molecule to `[gradient]`, rerun as a nested `Pipeline` at each of the
6N displaced geometries, as `[optimize]` reruns it at each evaluation
(`../optimize/README.md`). It needs no kernel of its own, and it is the
reference the analytic DF-RHF Hessian (`../../hessian/`) is tested against
(the design's first step), so it depends on none of it.

## Module

`free_breakfast.pipeline.hessian`, target `free_breakfast::pipeline::hessian`.
It re-exports `free_breakfast.pipeline.gradient`, which re-exports every step
the chain reruns, each a member, and the `MoleculeOutput` `run` reads.

## The step

```cpp
enum class HessianMethod : uint8_t { Numerical };

struct HessianOutput {
  std::vector<double> hessian;    // 3N x 3N, row-major, x = 3 A + k, symmetrized
  double asymmetry;               // max |H - H^T| before symmetrizing
  double translational_sum;       // max over x, l of |sum_B H[x][3 B + l]|
};

struct HessianStep {
  static constexpr std::string_view kName = "hessian";

  HessianMethod method{HessianMethod::Numerical};
  double displacement{1e-3};      // h, bohr

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

`run` takes the geometry, charge and multiplicity of the `MoleculeOutput`,
and for each coordinate x = 3 A + k plans and runs one `Pipeline` over

```
molecule -> basis_set -> spherical_transform -> primitive_pair_classes -> eri
  -> overlap -> orthogonalizer -> core_hamiltonian -> guess -> scf -> gradient
```

at R + h e_x and one at R - h e_x, the molecule a private step standing
where `MoleculeStep` stood. Row x of the raw Hessian is
(g(R + h e_x) - g(R - h e_x)) / 2h. Its asymmetry is recorded, the Hessian
kept is (H + Hᵀ) / 2, and its translational sum, which a Hessian of an energy
invariant under translation has zero, is recorded beside it. Each pipeline
is freed before the next, so the device holds one geometry's products at a
time; its first error -- an SCF that does not converge, `[gradient]`'s
refusal of an open shell -- is returned as it is.

- **Reads the outer `MoleculeOutput`**, so beside an `[optimize]` (whose
  `final.xyz` the molecule is then loaded from) it runs at the optimum, after
  the optimization: the frequencies' Hessian. It is ordered by that product,
  as any step reading the molecule is.
- **The error budget.** The central difference is O(h²) in the gradient's
  third derivative, and the SCF's noise in the gradient is divided by 2h. At
  the default h = 1e-3 bohr and the SCF converged as `[gradient]`'s test
  converges it (`energy_tolerance = 1e-12`, `commutator_tolerance = 1e-10`),
  PySCF's own central difference of `df.grad.rhf` sits 4.4e-7 from its
  analytic Hessian on the skew water (`tools/emit/hessian_numerical_reference.py`).
  The step does not tighten `[scf]` itself: a config that wants the Hessian
  to 1e-6 tightens it, as for `[gradient]`.
- **SAD at every geometry**, the configured `[guess]` made afresh, as
  `[optimize]`'s.
- **Any `[eri]`.** The chain holds the config's `EriStep`, so the gradient is
  `gradient::df_rhf` or `gradient::rhf`, whichever `[eri]` asks for; the
  analytic method this milestone adds is density-fitted.
- **The chain writes nothing.** `scf.molden` is cleared in its copy, and the
  option-less steps run at their defaults.
- **`Node`-shaped**, because each displacement runs a `Pipeline` on the node,
  co-run on the executor worker (`../pipeline/README.md`). 6N SCF + gradient
  runs: 18 for water.
- **`hessian.json`**, when `json_path` is set:

  ```json
  {"method": "numerical", "displacement": 0.001,
   "hessian": [[0.71, ...], ...],
   "asymmetry": 1.2e-7, "translational_sum": 3.4e-8}
  ```

  the Hessian (hartree / bohr²) as 3N rows of 3N.

## Config

```toml
[hessian]
method = "numerical"   # required: the only method so far
displacement = 1e-3    # bohr, positive
```

The chain is the config's own `[basis_set]`, `[eri]`, `[orthogonalizer]`,
`[guess]` and `[scf]`, copied in by `pipeline/config` as `[optimize]`'s is;
one loading `from` a file is refused, since the chain is rerun at every
displacement.

## Tests

- The step's contract and its config: `test/pipeline/hessian_host.cpp`
  (`static_assert`s on the run shape, the View and the product; `method`
  and `displacement` read and refused; the chain copied from the tables;
  `from` refused on a chain table; and beside `[optimize]`, the molecule it
  reads loaded from `final.xyz`).
- `test/gtest/pipeline_hessian_gpu.cpp`: a TOML config for the skew water in
  cc-pVDZ / cc-pVDZ-JKFIT, run by `Calculation`, its `hessian.json` against
  `pyscf.df.hessian.rhf` over the same aux set
  (`test/gtest/data/hessian_numerical/hessian_numerical.json`) to 1e-5, the
  asymmetry and the translational sum checked and printed.

## Files

| File | Role |
|------|------|
| `hessian.cppm` | Module interface: `HessianMethod`, `HessianOutput`, `HessianStep` |
| `hessian.cpp` | `HessianStep::run`: the 6N nested pipelines, the central differences, the two checks and `hessian.json` |
| `CMakeLists.txt` | The module library |
