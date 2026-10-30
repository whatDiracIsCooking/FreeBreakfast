# Pipeline: Gradient

The closed-shell RHF nuclear gradient at the SCF's orbitals: dE/dR for every
atom, and each term it sums. The numerics are all `../../gradient/`'s
(`gradient::df_rhf`); the step refuses what that cannot take, wires the
pipeline's products into it, and writes `gradient.json`.

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

1. **The refusals**, before anything runs on the device, each `Unsupported`
   and naming its reason: an ROHF `ScfOutput` (n_alpha ≠ n_beta), since
   `df_rhf` is the closed-shell gradient; and an integral-direct `EriOutput`,
   whose gradient is not wired in yet.
2. **`gradient::df_rhf`** over the Molecule, the orbital basis and its
   primitive-pair classes, the `DensityFitted` alternative's placed aux basis,
   B and L (`../eri/README.md`), and the `ScfOutput`'s C, orbital energies and
   n_alpha occupied orbitals. Its memory, its refusals and its terms are
   `../../gradient/README.md`'s.
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
   one row of three per atom in the Molecule's order: the layout of
   `test/gtest/data/gradient/gradient.json`'s `df_rhf` cases.

- **No options.** What the gradient is over is the pipeline's: the aux basis
  is `[eri]`'s, and the convergence `[scf]`'s. The formula is the gradient
  only at a converged SCF, where W carries the orbitals' response, so a
  config that wants it to 1e-8 tightens `[scf]` (`energy_tolerance = 1e-12`,
  `commutator_tolerance = 1e-10`, as the test does).
- **The refusals are the step's, not `pipeline/config`'s.** Both need a
  product: ROHF is the molecule's multiplicity, and the integral-direct
  refusal goes when that gradient is wired in, with nothing to undo in the
  front end.
- **`Device`-shaped**: B, L and C live on the device.

## Config

```toml
[gradient]   # no options; [eri] mode = "density_fitting" is required
```

## Tests

- The step's contract is checked by `static_assert`, in
  `test/pipeline/steps_host.cpp`; `[gradient]`'s parsing and its output
  path in `test/pipeline/config_host.cpp`.
- `test/gtest/pipeline_gradient_gpu.cpp`: a TOML config for H₂O in cc-pVDZ /
  cc-pVDZ-JKFIT, run by `Calculation`, its `gradient.json` against PySCF's
  `df.grad.rhf` (`test/gtest/data/gradient/gradient.json`, `df_rhf`'s
  `h2o_ccpvdz`): the energy, the total and every term to 1e-8. Configs with
  an integral-direct `[eri]` and with H₂O⁺ (ROHF) are each `Unsupported`,
  naming the reason, with no `gradient.json` written.

## Files

| File | Role |
|------|------|
| `gradient.cppm` | Module interface: `GradientOutput`, `GradientStep` |
| `gradient.cpp` | `GradientStep::run`: the refusals, `df_rhf`, and `gradient.json` |
| `CMakeLists.txt` | The module library |
