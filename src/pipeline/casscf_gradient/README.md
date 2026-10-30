# Pipeline: CasscfGradient

The analytic nuclear gradient of one root of the DF-CASSCF: dE/dR for every
atom, and each term it sums. The numerics are all
`../../mcscf/gradient/`'s (`mcscf::df_gradient`); the step refuses what that
cannot take, wires the pipeline's products into it, and writes
`casscf_gradient.json`.

## Module

`free_breakfast.pipeline.casscf_gradient`, target
`free_breakfast::pipeline::casscf_gradient`. It re-exports
`free_breakfast.mcscf.gradient` (the product holds `mcscf::GradientTerms`) and
`free_breakfast.pipeline.casscf` (which re-exports every product in `run`'s
signature, the `EriOutput` and the `CasscfOutput` among them). `run` is
defined in the implementation unit, so `io.json` is a PRIVATE link.

## The step

```cpp
struct CasscfGradientOutput {
  mcscf::GradientTerms gradient;  // total, kinetic, potential, overlap,
                                  // two_electron, nuclear_repulsion: each n_atom x 3
};

struct CasscfGradientStep {
  static constexpr std::string_view kName = "casscf_gradient";

  int32_t root = 0;                   // in CasscfOutput::energies' order
  std::filesystem::path json_path{};  // casscf_gradient.json, set by pipeline/config

  Result<CasscfGradientOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
           CoreHamiltonianOutput, CasscfOutput> in,
      DeviceResources &res) const;
};
```

`run`:

1. **The refusal**, before anything runs on the device: an integral-direct
   `EriOutput` is `Unsupported`, naming the reason. The Casscf step runs
   `mcscf::DensityFitting::Full` exactly when the `EriOutput` is fitted, so
   this one check is also the check that the CASSCF is the DF-CASSCF the
   formula differentiates. `pipeline/config` refuses the same config when it
   is read; the step refuses it again for a pipeline built by hand.
2. **The solution, back.** `df_gradient` takes `mcscf::solve`'s `Solution`,
   and the `CasscfOutput` is that solution with its core and virtuals
   canonicalized, which changes neither the 1-RDM D nor W = C sym(F) Cᵀ: the
   step copies C and the CI vectors back onto the device, and sums the
   weighted γ from the per-root `rdm1` and the `weights`. Γ is the product's
   `rdm2`, the weighted 2-RDM the Casscf step keeps for this.
3. **The gradient**: `mcscf::df_gradient` at `root`, over the Molecule, the
   orbital basis and its primitive-pair classes, the core Hamiltonian, and
   the `DensityFitted` product's aux basis, B and L. Its memory, its refusals
   and its terms are `../../mcscf/gradient/README.md`'s: a state average of
   more than one root is `Unsupported` there today, and a `root` past the
   CASSCF's `InvalidConfig`. When it takes a state average, this step does
   too, with no change: the `Solution` it builds carries every root.
4. **`casscf_gradient.json`**, when `json_path` is set:

   ```json
   {
     "root": 0,
     "energy": -76.077835817,
     "total": [[0.0, 0.0, -0.0185], ...],
     "terms": {"kinetic": [...], "potential": [...], "overlap": [...],
               "two_electron": [...], "nuclear_repulsion": [...]}
   }
   ```

   The root, its energy (hartree), then the gradient and each term
   (hartree / bohr), one row of three per atom in the Molecule's order: the
   layout of `test/gtest/data/casscf_gradient/casscf_gradient.json`'s
   `roots[i].total` and `roots[i].parts`.

- **`root` is the only option.** What the gradient is over is the pipeline's:
  the aux basis is `[eri]`'s, the active space and convergence `[casscf]`'s.
  The formula is the gradient only at a converged CASSCF, where W carries the
  orbitals' response, so a config that wants it to 1e-8 tightens `[casscf]`
  (`conv_tol_grad = 1e-9`, as the test does).
- **`Device`-shaped**: B, L, C and the CI vectors live on the device.

## Config

```toml
[eri]
mode = "density_fitting"
[casscf]
ncas = 4
nelecas = 4
[casscf_gradient]   # root = 0 by default
```

## Tests

- The step's contract is checked by `static_assert`, in
  `test/pipeline/steps_host.cpp`; `[casscf_gradient]`'s parsing, its output
  path and the integral-direct refusal in `test/pipeline/config_host.cpp`.
- `test/gtest/pipeline_casscf_gradient_gpu.cpp`: TOML configs run by
  `Calculation`, their `casscf_gradient.json` against PySCF's
  `df.grad.casscf` (`test/gtest/data/casscf_gradient/`): the energy, the
  total and every term to 1e-8, for H₂O in cc-pVDZ / cc-pVDZ-JKFIT, CAS(4,4)
  (`h2o_ccpvdz`), and the skewed H₂O (`h2o_skew_ccpvdz`, every component
  nonzero). A root the CASSCF does not have is `InvalidConfig`, with no
  `casscf_gradient.json` written. Each case is a fraction of a second, so
  both are in the fast set. No state-averaged root is tested until
  `df_gradient` takes one.

## Files

| File | Role |
|------|------|
| `casscf_gradient.cppm` | Module interface: `CasscfGradientOutput`, `CasscfGradientStep` |
| `casscf_gradient.cpp` | `CasscfGradientStep::run`: the refusal, the `Solution`, `df_gradient`, and `casscf_gradient.json` |
| `CMakeLists.txt` | The module library |
