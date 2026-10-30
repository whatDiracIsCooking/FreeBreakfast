# Pipeline: Gradient

The closed-shell RHF nuclear gradient at the SCF's orbitals: dE/dR for every
atom, and each term it sums, over either ERI kind `[eri]` builds. The
numerics are all `../../gradient/`'s (`gradient::df_rhf` density-fitted,
`gradient::rhf` integral-direct); the step refuses what those cannot take,
wires the pipeline's products into them, and writes `gradient.json`.

## Module

`free_breakfast.pipeline.gradient`, target `free_breakfast::pipeline::gradient`.
It re-exports `free_breakfast.gradient` (the product holds
`gradient::RhfTerms`) and `free_breakfast.pipeline.scf` (which re-exports every
product in `run`'s signature, the `EriOutput` among them). `run` is defined in
the implementation unit, so `io.json` is a PRIVATE link.

## The step

```cpp
struct GradientOutput {
  gradient::RhfTerms gradient;  // total, kinetic, potential, overlap, coulomb,
                                // exchange, nuclear_repulsion: each n_atom x 3
};

struct GradientStep {
  static constexpr std::string_view kName = "gradient";

  std::filesystem::path json_path{};  // gradient.json, set by pipeline/config

  Result<GradientOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
           ScfOutput> in,
      DeviceResources &res) const;
};
```

`run`:

1. **The refusal**, before anything runs on the device: an ROHF `ScfOutput`
   (n_alpha ≠ n_beta) is `Unsupported`, naming its reason, since both
   entry points are the closed-shell gradient.
2. **The gradient**, on the `EriOutput`'s alternative (`../eri/README.md`),
   over the Molecule, the orbital basis and its primitive-pair classes, and
   the `ScfOutput`'s C, orbital energies and n_alpha occupied orbitals:
   - `DensityFitted`: **`gradient::df_rhf`**, over its placed aux basis, B
     and L.
   - `IntegralDirect`: **`gradient::rhf`**, over its K classes (the
     `ordered_pair_classes` the integral-direct SCF's exchange reads). Its
     two-electron derivatives are unscreened, O(N⁴) in the primitive pairs.

   Their memory, their refusals and their terms are
   `../../gradient/README.md`'s; `RhfTerms` is the same for both, so is
   `gradient.json`.
3. **`gradient.json`**, when `json_path` is set:

   ```json
   {
     "energy": -76.026751518,
     "total": [[0.0, 0.0, 0.0123], ...],
     "terms": {"kinetic": [...], "potential": [...], "overlap": [...],
               "coulomb": [...], "exchange": [...], "nuclear_repulsion": [...]}
   }
   ```

   The SCF energy (hartree), then the gradient and each term (hartree / bohr),
   one row of three per atom in the Molecule's order, whichever ERI kind:
   the layout of `test/gtest/data/gradient/gradient.json`'s `df_rhf` and
   `rhf` cases.

- **No options.** What the gradient is over is the pipeline's: the ERI kind
  and aux basis are `[eri]`'s, and the convergence `[scf]`'s. The formula is
  the gradient only at a converged SCF, where W carries the orbitals'
  response, so a config that wants it to 1e-8 tightens `[scf]`
  (`energy_tolerance = 1e-12`, `commutator_tolerance = 1e-10`, as the test
  does). Integral-direct, add `fock_rebuild_interval = 1`: the incremental
  Fock builds stall short of that commutator (`MaxIterationsExceeded`).
- **The refusal is the step's, not `pipeline/config`'s**: it needs a
  product, as ROHF is the molecule's multiplicity.
- **`Device`-shaped**: B, L, the K classes and C live on the device.

## Config

```toml
[gradient]   # no options; either [eri] mode
```

## Tests

- The step's contract is checked by `static_assert`, in
  `test/pipeline/steps_host.cpp`; `[gradient]`'s parsing and its output
  path in `test/pipeline/config_host.cpp`.
- `test/gtest/pipeline_gradient_gpu.cpp`: TOML configs for H₂O in cc-pVDZ,
  run by `Calculation`, their `gradient.json` against PySCF
  (`test/gtest/data/gradient/gradient.json`): the energy, the total and
  every term to 1e-8, density-fitted over cc-pVDZ-JKFIT against
  `df.grad.rhf` (`df_rhf`'s `h2o_ccpvdz`) and integral-direct against
  `scf.grad.rhf` (`rhf`'s `h2o_ccpvdz`). A config with H₂O⁺ (ROHF) is
  `Unsupported`, naming the reason, with no `gradient.json` written.

## Files

| File | Role |
|------|------|
| `gradient.cppm` | Module interface: `GradientOutput`, `GradientStep` |
| `gradient.cpp` | `GradientStep::run`: the refusal, `df_rhf` or `rhf`, and `gradient.json` |
| `CMakeLists.txt` | The module library |
