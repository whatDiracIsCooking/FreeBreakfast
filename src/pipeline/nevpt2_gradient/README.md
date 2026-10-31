# Pipeline: Nevpt2Gradient

The analytic nuclear gradient of the SC- or PC-NEVPT2 total energy
E_cas + E_corr of the single-state DF-CASSCF: dE/dR for every atom, each
term it sums, and the Lagrangian's diagnostics. The numerics are all
`../../nevpt2/gradient/`'s (`nevpt2::sc_gradient`, `nevpt2::pc_gradient`);
the step refuses what they cannot take, wires the pipeline's products into
them, and writes `nevpt2_gradient.json`. Its shape is `../casscf_gradient/`'s.

## Module

`free_breakfast.pipeline.nevpt2_gradient`, target
`free_breakfast::pipeline::nevpt2_gradient`. It re-exports
`free_breakfast.nevpt2.gradient` (the product holds `nevpt2::ScGradient` or
`nevpt2::PcGradient`)
and `free_breakfast.pipeline.casscf` (which re-exports every product in
`run`'s signature, the `EriOutput`, the `ScfOutput` and the `CasscfOutput`
among them). `run` is defined in the implementation unit, so `io.json` is a
PRIVATE link.

## The step

```cpp
enum class Nevpt2Variant : uint8_t { Sc, Pc };

struct Nevpt2GradientOutput {
  // The variant's: terms (total and five terms, each n_atom x 3), energies
  // (E_corr's classes), zc, multipliers, asymmetry.
  std::variant<nevpt2::ScGradient, nevpt2::PcGradient> gradient;
};

struct Nevpt2GradientStep {
  static constexpr std::string_view kName = "nevpt2_gradient";

  int32_t root = 0;                         // the single-state CASSCF's only root
  Nevpt2Variant variant = Nevpt2Variant::Sc;  // sc_gradient, or pc_gradient
  std::filesystem::path json_path{};        // nevpt2_gradient.json, set by pipeline/config

  Result<Nevpt2GradientOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
           CoreHamiltonianOutput, ScfOutput, CasscfOutput> in,
      DeviceResources &res) const;
};
```

`run`:

1. **The refusals**, before anything runs on the device, each
   `Unsupported` naming the reason: `[nevpt2]`'s three, for the same reasons (`../nevpt2/README.md`) -- an open-shell
   `ScfOutput` (`n_alpha != n_beta`), a molecule of multiplicity other than
   1, and an integral-direct `EriOutput`, naming `[eri]`; and a state
   average, a `CasscfOutput` of more than one root, which neither gradient
   takes. A `root` other than 0 is then `InvalidConfig`. `pipeline/config`
   refuses each of these when the config is read
   (`../config/README.md`), so a run does not converge an SCF and a CASSCF
   first; the step refuses them again for a pipeline built by hand.
2. **The solution, back**, as the CasscfGradient step builds it: C and the
   CI vector copied back onto the device, γ the root's `rdm1` and Γ the
   product's `rdm2`. The `CasscfOutput`'s core and virtuals are
   canonicalized, which both gradients need: with one root, the
   state-averaged Fock it is canonical over is the root's own, so its
   orbital energies are f's diagonal there, as PySCF's NEVPT2 reads them.
3. **The gradient**: `nevpt2::sc_gradient`, or `nevpt2::pc_gradient` for
   `variant = "pc"`, over the Molecule, the orbital
   basis and its primitive-pair classes, the core Hamiltonian, and the
   `DensityFitted` product's aux basis, B and L. Its terms, its multipliers
   and its gate are `../../nevpt2/gradient/README.md`'s. A class PC refuses
   (no gap at the cut, a non-positive λ + Δ) is the energy's error, returned
   unchanged, so `[optimize]` stops on it.
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
   `gradients.sc.richardson` (`gradients.pc.richardson` for PC); max |z^c| and the smallest core-core or
   virtual-virtual orbital-energy gap (`null` when neither space has two
   orbitals); the Z-vector's PCG iterations and residual; and X^L's
   antisymmetric residue per block, each below the 1e-8 gate the gradient
   refuses past. PC's file adds one key, `"dropped"`: the metric modes each
   class cut as null, eight in PySCF's class order (`PcClassEnergies`); SC's
   has no such key. `main` prints the variant, the energy, the diagnostics
   and the total per atom.

- **`root` must be 0.** The option is there, as `[casscf_gradient]`'s, for
  when a state average is taken; until then the CASSCF has one root.
- **The CASSCF's convergence is the gradient's.** The Lagrangian is
  stationary only at a converged CASSCF, so a config that wants the gradient
  to 1e-6 of the numerical one tightens `[casscf]`
  (`conv_tol_grad = 1e-9`, as the test does).
- **Memory.** Both gradients take batch 0, each class's whole external
  extent in one slab, as `nevpt2_gradient_gpu.cpp` runs it: the `[nevpt2]`
  step's slab budget needs `nevpt2::MoBlocks` this step never builds, and
  the cases this gradient is validated on are far from the card's memory.
  A larger system that does not fit fails its allocation (an abort, as
  every failed device allocation is).
- **`Device`-shaped**: B, L, C and the CI vector live on the device.

## PC

`variant = "pc"` maps to `Nevpt2Variant::Pc` in `read_step`
(`../config/config.cpp`), and `run` then calls `nevpt2::pc_gradient` where
it would call `sc_gradient`, with the same arguments. The product's
`gradient` holds whichever was called; the JSON is SC's shape with
`"variant": "pc"` and `"dropped"`, and `[optimize]` reads the total and
E_corr off either alternative. Nothing else in the step depends on the
variant.

## Config

```toml
[eri]
mode = "density_fitting"
[casscf]
ncas = 4
nelecas = 4
conv_tol_grad = 1e-9
[nevpt2_gradient]   # root = 0, variant = "sc" by default ("pc" the other)
```

## Tests

- `[nevpt2_gradient]`'s parsing, its output path and each refusal at config
  read in `test/pipeline/config_host.cpp`.
- `test/gtest/pipeline_nevpt2_gradient_gpu.cpp`, SC: a TOML config run by
  `Calculation` for `test/gtest/data/nevpt2_gradient/`'s skewed H₂O
  (`h2o_skew_ccpvdz`, cc-pVDZ / cc-pVDZ-JKFIT, CAS(4,4), every component
  nonzero): its `nevpt2_gradient.json` against `nevpt2::sc_gradient` called
  directly over the same products (every term, E_corr and the diagnostics,
  to 1e-10, round-off), and against the reference's numerical SC gradient
  per component to 1e-6; E_corr to 1e-9 and the CASSCF energy to 1e-8.
  About half a second on an RTX 3080 (numerical 3.4e-9, direct 6.4e-14), so
  it is in the fast set.
- The same file, PC: the same run with `variant = "pc"`, against
  `nevpt2::pc_gradient` called directly (every term, E_corr, the diagnostics
  and `dropped`, round-off) and against `gradients.pc.richardson` per
  component to 1e-6; E_corr to 1e-9 against `e_corr.pc`.
  About half a second on an RTX 3080 too (numerical 3.5e-9, direct 4.5e-14),
  so it is in the fast set.

## Files

| File | Role |
|------|------|
| `nevpt2_gradient.cppm` | Module interface: `Nevpt2Variant`, `Nevpt2GradientOutput`, `Nevpt2GradientStep` |
| `nevpt2_gradient.cpp` | `Nevpt2GradientStep::run`: the refusals, the `Solution`, `sc_gradient` or `pc_gradient`, and `nevpt2_gradient.json` |
| `CMakeLists.txt` | The module library |
