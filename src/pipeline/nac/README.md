# Pipeline: Nac

The nonadiabatic (derivative) couplings d_IJ = ⟨I|∂J/∂R⟩ between pairs of
roots of the equal-weight DF-SA-CASSCF, each with its numerator h_IJ and its
CSF term d^CSF_IJ. The numerics are all `../../mcscf/nac/`'s
(`mcscf::df_nac`, whose README has the bra/ket order, the signs and the
options); the step refuses what that cannot take, wires the pipeline's
products into it once per pair, and writes `nac.json`.

## Module

`free_breakfast.pipeline.nac`, target `free_breakfast::pipeline::nac`. It
re-exports `free_breakfast.mcscf.nac` (the step holds `mcscf::NacOptions`, the
product `mcscf::Nac`) and `free_breakfast.pipeline.casscf` (which re-exports
every product in `run`'s signature, the `EriOutput` and the `CasscfOutput`
among them). `run` is defined in the implementation unit, so `io.json` is a
PRIVATE link.

## The step

```cpp
struct NacOutput {
  std::vector<std::array<int32_t, 2>> pairs;  // (I, J) of each coupling
  std::vector<mcscf::Nac> couplings;          // numerator (h_IJ), csf, total: each n_atom x 3
};

struct NacStep {
  static constexpr std::string_view kName = "nac";
  static constexpr double kEqualWeights = 1e-10;

  std::vector<std::array<int32_t, 2>> pairs{};  // empty: every I < J
  mcscf::NacOptions options{};                  // mult_ediff, use_etfs
  std::filesystem::path json_path{};            // nac.json, set by pipeline/config

  Result<NacOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
           CoreHamiltonianOutput, CasscfOutput> in,
      DeviceResources &res) const;
};
```

`run`:

1. **The refusals**, before anything runs on the device: an integral-direct
   `EriOutput` and a CASSCF whose weights span more than `kEqualWeights` are
   `Unsupported`, naming the reason; a CASSCF of fewer than two roots, and a
   pair whose roots are equal or not both the CASSCF's, `InvalidConfig`. The
   Casscf step runs `mcscf::DensityFitting::Full` exactly when the
   `EriOutput` is fitted, so the first check is also the check that the
   CASSCF is the DF-CASSCF the couplings are of. Unequal weights wait on
   `df_nac` (`../../mcscf/nac/README.md` "The numerator" says why).
   `pipeline/config` refuses the same config when it is read; the step
   refuses it again for a pipeline built by hand.
2. **The solution, back**, as `../casscf_gradient/` builds it for
   `df_gradient`: C and the CI vectors copied back onto the device, the
   weighted γ summed from the per-root `rdm1` and the `weights`, Γ the
   product's weighted `rdm2`. The core and virtuals' canonicalization moves
   no coupling.
3. **The couplings**: `mcscf::df_nac` for each pair, in `pairs`' order (left
   empty: (0, 1), (0, 2), …, (1, 2), …), over the Molecule, the orbital basis
   and its primitive-pair classes, the core Hamiltonian, and the
   `DensityFitted` product's aux basis, B and L, with `options`. Each pair
   logs its gap, its largest |d| and the Z-vector's products. Its memory and
   its other refusals are `df_nac`'s: a gap below 1e-10 hartree without
   `mult_ediff` is `Unsupported` there.
4. **`nac.json`**, when `json_path` is set:

   ```json
   {
     "mult_ediff": false,
     "use_etfs": false,
     "energies": [-113.9049288585, -113.7550787538],
     "couplings": [
       {"bra": 0, "ket": 1, "energy_gap": 0.1498501047,
        "h": [[...], ...], "csf": [[...], ...], "total": [[...], ...],
        "zvector": {"iterations": 10, "residual": 8.1e-11}}
     ]
   }
   ```

   The options; the CASSCF's root energies (hartree); then per pair I, J,
   E_J − E_I, h_IJ (hartree / bohr) and d^CSF_IJ (1 / bohr), both unscaled
   whatever the options, and `total` as the options ask (by default d_IJ =
   h_IJ / (E_J − E_I) + d^CSF_IJ, 1 / bohr), one row of three per atom in
   the Molecule's order; and the Z-vector's conjugate-gradient products and
   residual. The keys are `test/gtest/data/casscf_nac/casscf_nac.json`'s
   `couplings[i]`'s of the same names.

- **The couplings change sign with either root's CI phase**, as the CASSCF
  carries it; nothing fixes a phase. h_IJ is symmetric in I and J, d_IJ and
  d^CSF_IJ antisymmetric, so `pairs` names (J, I) only for the reverse sign.
- **What they are over is the pipeline's**: the aux basis is `[eri]`'s, the
  active space, roots and convergence `[casscf]`'s. They are the CASSCF's
  couplings only at its stationary point, so a config that wants them to
  1e-8 tightens `[casscf]` (`conv_tol_grad = 1e-10`, as the test does).
- **`Device`-shaped**: B, L, C and the CI vectors live on the device.

## Config

```toml
[molecule]
xyz = "h2co.xyz"
[basis_set]
basis = "cc-pvdz"
[eri]
mode = "density_fitting"   # required
aux_basis = "cc-pvdz-jkfit"
[casscf]
ncas = 3
nelecas = 4
nroots = 2                 # at least 2, at equal weights (weights left out)
conv_tol_grad = 1e-10      # the couplings are the CASSCF's only at convergence
[nac]
pairs = [[0, 1]]           # (bra, ket); left out, every I < J
mult_ediff = false         # the default
use_etfs = false           # the default
```

`free-breakfast -i <dir> -o <out> nac.toml` runs it from the command line,
with `h2co.xyz` in `<dir>`: the reference's H₂CO,

```
4
h2co_ccpvdz_sa2
C  0.0  0.0            0.0
O  0.0  0.0            1.2028198004
H  0.0  0.9366436633  -0.5794490459
H  0.0 -0.9366436633  -0.5794490459
```

(Angstrom), and writes `<out>/nac.json` beside `casscf.molden` and
`scf.molden` (the `nac.json` above is that run's, on the gfx1200).

## Tests

- The step's contract is checked by `static_assert`, and `[nac]`'s parsing,
  its output path and every refusal (a malformed table, an integral-direct
  `[eri]` written or filled in, a `[casscf]` of one root, a pair past its
  roots, unequal weights) in `test/pipeline/nac_host.cpp`.
- `test/gtest/pipeline_nac_gpu.cpp`: a TOML config run by `Calculation`, its
  `nac.json` against PySCF's DF SA-CASSCF couplings
  (`test/gtest/data/casscf_nac/`, `h2co_ccpvdz_sa2`: H₂CO, SA-2 CAS(4,3),
  cc-pVDZ / cc-pVDZ-JKFIT): the roots' energies and the gap, then h_IJ,
  d^CSF and d_IJ of the default pair (0, 1) to 1e-8, up to the sign the two
  CI phases carry. Then the step run by hand over the same products with
  `pairs = [[1, 0]]` and `use_etfs`, against the reference's (1, 0) `etf`,
  `h` and `csf` under the same sign, and a pair of equal roots refused. Measured on the gfx1200: h_IJ within 8.3e-13, d^CSF 1.1e-12, d_IJ
  6.6e-12, the (1, 0) `etf` 6.9e-12; 1.6 s, so it is in the fast set.

## Files

| File | Role |
|------|------|
| `nac.cppm` | Module interface: `NacOutput`, `NacStep` |
| `nac.cpp` | `NacStep::run`: the refusals, the `Solution`, `df_nac` per pair, and `nac.json` |
| `CMakeLists.txt` | The module library |
