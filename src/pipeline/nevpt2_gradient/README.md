# Pipeline: Nevpt2Gradient

The analytic nuclear gradient of the SC-NEVPT2 total energy E_cas + E_corr
of the single-state DF-CASSCF: dE/dR for every atom, each term it sums, and
the Lagrangian's diagnostics. The numerics are all
`../../nevpt2/gradient/`'s (`nevpt2::sc_gradient`); the step refuses what
that cannot take, wires the pipeline's products into it, and writes
`nevpt2_gradient.json`. Its shape is `../casscf_gradient/`'s.

## Module

`free_breakfast.pipeline.nevpt2_gradient`, target
`free_breakfast::pipeline::nevpt2_gradient`. It re-exports
`free_breakfast.nevpt2.gradient` (the product holds `nevpt2::ScGradient`)
and `free_breakfast.pipeline.casscf` (which re-exports every product in
`run`'s signature, the `EriOutput`, the `ScfOutput` and the `CasscfOutput`
among them). `run` is defined in the implementation unit, so `io.json` is a
PRIVATE link.

## The step

```cpp
enum class Nevpt2Variant : uint8_t { Sc, Pc };

struct Nevpt2GradientOutput {
  nevpt2::ScGradient gradient;  // terms (total and five terms, each n_atom x 3),
                                // energies (E_corr's classes), zc, multipliers,
                                // asymmetry
};

struct Nevpt2GradientStep {
  static constexpr std::string_view kName = "nevpt2_gradient";

  int32_t root = 0;                         // the single-state CASSCF's only root
  Nevpt2Variant variant = Nevpt2Variant::Sc;  // Pc refused until #564
  std::filesystem::path json_path{};        // nevpt2_gradient.json, set by pipeline/config

  Result<Nevpt2GradientOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
           CoreHamiltonianOutput, ScfOutput, CasscfOutput> in,
      DeviceResources &res) const;
};
```

`run`:

1. **The refusals**, before anything runs on the device, each
   `Unsupported` naming the reason: `variant` PC (below); `[nevpt2]`'s
   three, for the same reasons (`../nevpt2/README.md`) -- an open-shell
   `ScfOutput` (`n_alpha != n_beta`), a molecule of multiplicity other than
   1, and an integral-direct `EriOutput`, naming `[eri]`; and a state
   average, a `CasscfOutput` of more than one root, which `sc_gradient` does
   not take. A `root` other than 0 is then `InvalidConfig`. `pipeline/config`
   refuses each of these when the config is read
   (`../config/README.md`), so a run does not converge an SCF and a CASSCF
   first; the step refuses them again for a pipeline built by hand.
2. **The solution, back**, as the CasscfGradient step builds it: C and the
   CI vector copied back onto the device, γ the root's `rdm1` and Γ the
   product's `rdm2`. The `CasscfOutput`'s core and virtuals are
   canonicalized, which `sc_gradient` needs: with one root, the
   state-averaged Fock it is canonical over is the root's own, so its
   orbital energies are f's diagonal there, as PySCF's NEVPT2 reads them.
3. **The gradient**: `nevpt2::sc_gradient` over the Molecule, the orbital
   basis and its primitive-pair classes, the core Hamiltonian, and the
   `DensityFitted` product's aux basis, B and L. Its terms, its multipliers
   and its gate are `../../nevpt2/gradient/README.md`'s.
4. **`nevpt2_gradient.json`**, when `json_path` is set:

   ```json
   {
     "root": 0,
     "variant": "sc",
     "energy": -76.227225846, "e_casscf": -76.077824893, "e_corr": -0.149400953,
     "total": [[0.0066, 0.0039, -0.0163], ...],
     "terms": {"kinetic": [...], "potential": [...], "overlap": [...],
               "two_electron": [...], "nuclear_repulsion": [...]},
     "max_zc": 1.877e-3, "min_inactive_gap": 2.995e-3,
     "pcg_iterations": 18, "pcg_residual": ...,
     "asymmetry": {"core_core": ..., "active_active": ..., "virtual_virtual": ...,
                   "active_core": ..., "virtual_core": ..., "virtual_active": ...}
   }
   ```

   The root and the variant; the total energy, the CASSCF root's and E_corr
   (hartree); the gradient and each term (hartree / bohr), one row of three
   per atom in the Molecule's order, the layout of
   `test/gtest/data/nevpt2_gradient/nevpt2_gradient.json`'s
   `gradients.sc.richardson`; max |z^c| and the smallest core-core or
   virtual-virtual orbital-energy gap (`null` when neither space has two
   orbitals); the Z-vector's PCG iterations and residual; and X^L's
   antisymmetric residue per block, each below the 1e-8 gate `sc_gradient`
   refuses past. `main` prints the energy, the diagnostics and the total per
   atom.

- **`root` must be 0.** The option is there, as `[casscf_gradient]`'s, for
  when a state average is taken; until then the CASSCF has one root.
- **The CASSCF's convergence is the gradient's.** The Lagrangian is
  stationary only at a converged CASSCF, so a config that wants the gradient
  to 1e-6 of the numerical one tightens `[casscf]`
  (`conv_tol_grad = 1e-9`, as the test does).
- **Memory.** `sc_gradient` takes batch 0, each class's whole external
  extent in one slab, as `nevpt2_gradient_gpu.cpp` runs it: the `[nevpt2]`
  step's slab budget needs `nevpt2::MoBlocks` this step never builds, and
  the cases this gradient is validated on are far from the card's memory.
  A larger system that does not fit fails its allocation (an abort, as
  every failed device allocation is).
- **`Device`-shaped**: B, L, C and the CI vector live on the device.

## PC

`variant = "pc"` is `Unsupported`, naming #564, in two places: `read_step`'s
`Nevpt2GradientStep` branch in `../config/config.cpp`, and the first check in
`Nevpt2GradientStep::run`. Wiring PC after #564 merges is: in `read_step`,
map `"pc"` to `Nevpt2Variant::Pc` instead of refusing it; in `run`, replace
the refusal with the PC gradient's call where `sc_gradient` is called, taking
the same arguments and returning the same `nevpt2::ScGradient`, and write
`"pc"` as the JSON's `variant`. Nothing else in the step depends on the
variant.

## Config

```toml
[eri]
mode = "density_fitting"
[casscf]
ncas = 4
nelecas = 4
conv_tol_grad = 1e-9
[nevpt2_gradient]   # root = 0, variant = "sc" by default
```

## Tests

- `[nevpt2_gradient]`'s parsing, its output path and each refusal at config
  read in `test/pipeline/config_host.cpp`.
- `test/gtest/pipeline_nevpt2_gradient_gpu.cpp`: a TOML config run by
  `Calculation` for `test/gtest/data/nevpt2_gradient/`'s skewed H₂O
  (`h2o_skew_ccpvdz`, cc-pVDZ / cc-pVDZ-JKFIT, CAS(4,4), every component
  nonzero): its `nevpt2_gradient.json` against `nevpt2::sc_gradient` called
  directly over the same products (every term, E_corr and the diagnostics,
  to 1e-10, round-off), and against the reference's numerical SC gradient
  per component to 1e-6; E_corr to 1e-9 and the CASSCF energy to 1e-8.
  About half a second on an RTX 3080 (numerical 3.4e-9, direct 6.4e-14), so
  it is in the fast set.

## Files

| File | Role |
|------|------|
| `nevpt2_gradient.cppm` | Module interface: `Nevpt2Variant`, `Nevpt2GradientOutput`, `Nevpt2GradientStep` |
| `nevpt2_gradient.cpp` | `Nevpt2GradientStep::run`: the refusals, the `Solution`, `sc_gradient`, and `nevpt2_gradient.json` |
| `CMakeLists.txt` | The module library |
