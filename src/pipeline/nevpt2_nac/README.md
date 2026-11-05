# Pipeline: Nevpt2Nac

The SC- or PC-QD-NEVPT2 nonadiabatic (derivative) couplings d_PQ =
⟨Ψ_P|∂Ψ_Q/∂R⟩ between pairs of QD states of the DF-SA-CASSCF, at any
weights, each with its pieces: h_PQ, d^CAS, d^PT2 and the CSF part. The
numerics are all `../../nevpt2/nac/`'s (`nevpt2::qd_sc_nac` and
`qd_pc_nac`, whose README has the bra/ket order, the signs and the options);
the step refuses what those cannot take, forms the effective Hamiltonian's
eigenvalues and R once, wires the pipeline's products into the coupling once
per pair, and writes `nevpt2_nac.json`.

## Module

`free_breakfast.pipeline.nevpt2_nac`, target
`free_breakfast::pipeline::nevpt2_nac`. It re-exports
`free_breakfast.nevpt2.nac` (the product holds `nevpt2::QdNac`) and
`free_breakfast.pipeline.casscf` (which re-exports every product in `run`'s
signature, the `EriOutput`, the `ScfOutput` and the `CasscfOutput` among
them). The step's `variant` is `[nevpt2_gradient]`'s `Nevpt2Variant` and its
`options` `mcscf::NacOptions`, so `pipeline.nevpt2_gradient` and `mcscf.nac`
are PUBLIC links. `run` is defined in the implementation unit, so `io.json`,
`recipes.diagonalize` and the NEVPT2 energy and gradient layers (H_eff and R)
are PRIVATE.

## The step

```cpp
struct Nevpt2NacOutput {
  std::vector<double> energies;               // H_eff's eigenvalues, ascending
  std::vector<double> vectors;                // R, state K's column at [K n, (K + 1) n)
  std::vector<std::array<int32_t, 2>> pairs;  // (P, Q) of each coupling
  std::vector<nevpt2::QdNac> couplings;       // numerator (h_PQ), cas, pt2, csf, total: each n_atom x 3
};

struct Nevpt2NacStep {
  static constexpr std::string_view kName = "nevpt2_nac";

  std::vector<std::array<int32_t, 2>> pairs{};  // empty: every P < Q
  Nevpt2Variant variant = Nevpt2Variant::Sc;    // SC or PC
  mcscf::NacOptions options{};                  // mult_ediff, use_etfs
  std::filesystem::path json_path{};            // nevpt2_nac.json, set by pipeline/config

  Result<Nevpt2NacOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
           CoreHamiltonianOutput, ScfOutput, CasscfOutput> in,
      DeviceResources &res) const;
};
```

`run`:

1. **The refusals**, before anything runs on the device: an open-shell
   `ScfOutput`, a molecule of multiplicity other than 1 and an
   integral-direct `EriOutput` are `Unsupported`, naming the reason (the
   `[nevpt2]` step's, for its reasons); a CASSCF of fewer than two roots,
   and a pair whose states are equal or not both H_eff's (one per root),
   `InvalidConfig`. The Casscf step runs `mcscf::DensityFitting::Full`
   exactly when the `EriOutput` is fitted, so the integral-direct check is
   also the check that the CASSCF is the DF-SA-CASSCF the couplings are of.
   Any weights are taken. `pipeline/config` refuses the same config when it
   is read; the step refuses it again for a pipeline built by hand.
2. **The solution, back**, as `../nevpt2_gradient/` builds it for the QD
   gradients: C (core and virtuals canonical over the SA Fock) and the CI
   vectors copied back onto the device, the weighted γ summed from the
   per-root `rdm1` and the `weights`, Γ the product's weighted `rdm2`.
3. **H_eff and R**, once: `nevpt2::qd_inputs` handed to
   `nevpt2::sc_effective_hamiltonian` (or `pc_effective_hamiltonian` for
   `variant = "pc"`), H_eff = diag(E_r) + ½ (C + Cᵀ), diagonalized, and each
   column signed with its largest-magnitude component positive: the
   interstate coupling's own formation of them, so the pairs' indices name
   the states the coupling takes. **A PC refusal** (a class's metric with no
   positive eigenvalue, or a pole: an intruder, |t| > 1, or a λ + Δ at
   round-off) is that effective Hamiltonian's
   `Numerical` error, naming the class and the state, returned unchanged
   before any Z-vector is solved; there is no SC to keep, and no
   `nevpt2_nac.json` is written.
4. **The couplings**: `nevpt2::qd_sc_nac` or `qd_pc_nac` for each pair, in
   `pairs`' order (left empty: (0, 1), (0, 2), …, (1, 2), …), over the
   Molecule, the orbital basis and its primitive-pair classes, the core
   Hamiltonian, and the `DensityFitted` product's aux basis, B and L, with
   `options`, each class's slabs in one batch (batch 0, as the
   `[nevpt2_gradient]` step runs its QD gradients). Each pair logs E_Q − E_P,
   its largest |d| and the three Z-vectors' products. Its memory and its
   other refusals are the coupling's: QD states within
   `mcscf::kRootGapFloor` (1e-8 hartree), and two coupled CASSCF roots
   closer than that, are `Numerical` there.
5. **`nevpt2_nac.json`**, when `json_path` is set:

   ```json
   {
     "variant": "sc",
     "mult_ediff": false,
     "use_etfs": false,
     "e_casscf": [...],
     "qd_energies": [...],
     "r": [[...], ...],
     "couplings": [
       {"bra": 0, "ket": 1, "energy_gap": ...,
        "h": [[...], ...], "cas": [[...], ...], "pt2": [[...], ...],
        "csf": [[...], ...], "total": [[...], ...],
        "heff_gap": ..., "root_gap": ..., "max_zeta": ...,
        "zvector": {"iterations": [..., ..., ...], "residual": [..., ..., ...]}}
     ]
   }
   ```

   The variant and the options; the CASSCF's root energies and H_eff's
   eigenvalues (hartree); R, one array per QD state K over the CASSCF roots
   (`r[K][M]` = R_MK); then per pair P, Q, E_Q − E_P from those eigenvalues,
   h_PQ (hartree / bohr), d^CAS, d^PT2 and the CSF part (1 / bohr), all
   unscaled whatever the options, and `total` as the options ask (by
   default d_PQ = h_PQ / (E_Q − E_P) + d^CAS + d^PT2, 1 / bohr), one row of
   three per atom in the Molecule's order; and the diagnostics: |E_Q − E_P|
   as the coupling's own H_eff has it, the smallest coupled CASSCF root gap
   (`null` when T^PQ couples none), max |ζ| of h_PQ's Lagrangian, and the
   three Z-vectors' (h_PQ's, d^CAS's, d^PT2's) products and residuals.

- **H_eff and R are the step's own**, as `[nevpt2_gradient]`'s are its: the
  step does not read a `[nevpt2]` step's `Nevpt2Output`, and plans none. A
  step's `View` is fixed, so naming that product would make every
  `[nevpt2_nac]` plan and run a `[nevpt2]` with `quasi_degenerate`, every
  root's state-specific NEVPT2 and both QD schemes, for the eigenvalues and
  R this step forms in one effective Hamiltonian. They agree with the ones
  inside each coupling to the atomics' round-off (the test holds the gap
  to 1e-10).
- **The couplings change sign with the CI phases**, through R: d_PQ by
  s_m(P) s_m(Q), m(P) the root carrying R_P's largest component
  (`../../nevpt2/nac/README.md`, "Conventions"); nothing fixes a phase.
  h_PQ is symmetric in P and Q, d_PQ antisymmetric, so `pairs` names
  (Q, P) only for the reverse sign.
- **What they are over is the pipeline's**: the aux basis is `[eri]`'s, the
  active space, roots, weights and convergence `[casscf]`'s. They are the
  CASSCF's couplings only at its stationary point, so a config that wants
  them to 1e-8 tightens `[casscf]` (`conv_tol_grad = 1e-10`, as the test
  does).
- **`Device`-shaped**: B, L, C and the CI vectors live on the device.

## Config

[`examples/qd_nevpt2_nac.toml`](../../../examples/qd_nevpt2_nac.toml) is a
whole config, over [`examples/h2co.xyz`](../../../examples/h2co.xyz): `[nac]`'s
H₂CO in cc-pVDZ / cc-pVDZ-JKFIT, with `[eri] mode = "density_fitting"`
(required) and a `[casscf]` of two roots at `conv_tol_grad = 1e-10`. The
table's own keys:

```toml
[nevpt2_nac]
pairs = [[0, 1]]           # (bra, ket) QD states; left out, every P < Q
variant = "sc"             # the default; "pc" the other
mult_ediff = false         # the default
use_etfs = false           # the default
```

`pairs`, `mult_ediff` and `use_etfs` are `[nac]`'s; `variant` is
`[nevpt2_gradient]`'s key, with its two values. **`frozen_core` is
refused**, `InvalidConfig` naming it when the config is read: the coupling
is all-electron (h_PQ and the PT2 transition density take no frozen core,
`../../nevpt2/nac/README.md`), so a frozen-core coupling would be the wrong
one rather than an approximation of the right one. `[nevpt2]` and
`[nevpt2_gradient]` take the key (#875); a `[nevpt2]` with a frozen core
beside this step still leaves the coupling all-electron.
`free-breakfast -i examples -o <out> examples/qd_nevpt2_nac.toml` runs it,
and writes `<out>/nevpt2_nac.json` beside `casscf.molden` and `scf.molden`;
`main` prints each QD state's energy and R, and each pair's gap,
diagnostics and `total` per atom.
On the RTX 3080 the whole run takes 1.2 s: the gap 0.1581181479 hartree,
and on planar H₂CO the coupling's only nonzero components are the two
hydrogens' out-of-plane x, ∓0.3059 / bohr. It is not in `test/examples/`:
`rhf.toml` and `hessian.toml` already take every path of `main.cpp` but
this step's printout (arguments, file I/O, a Molden file and a JSON one),
and the step's numbers and file are held by the GPU test below, against the
direct call.

## Tests

- The step's contract is checked by `static_assert`, and `[nevpt2_nac]`'s
  parsing, its output path and every refusal at config read (a malformed
  table, an unknown `variant`, an integral-direct `[eri]` written or filled
  in, a `[casscf]` of one root, a pair past H_eff's states, a triplet) in
  `test/pipeline/nevpt2_nac_host.cpp`, which also reads a `[casscf]` at
  unequal weights and checks no `[nevpt2]` is planned.
- `test/gtest/pipeline_nevpt2_nac_gpu.cpp`: a TOML config run by
  `Calculation` for `../../../test/gtest/data/qd_nevpt2/`'s
  `h8_ring_sto3g_sa3` (STO-3G / cc-pVDZ-JKFIT, CAS(6,4) averaged over three
  roots, at #696's bars), at pair (1, 2), both indices past 0, so an index
  or ordering error between H_eff's states and the pair wired through shows;
  once SC, once PC. Its `nevpt2_nac.json` against the product exactly, R's
  sign rule, and the product against `nevpt2::qd_sc_nac` / `qd_pc_nac`
  called directly over the same products: h, cas, pt2, csf and total each
  to 1e-10 of its own largest element, and the step's H_eff gap at the pair
  against the call's `heff_gap` to 1e-10 of the largest eigenvalue. Then
  the step run by hand over the same products with a pair of equal states
  and one past H_eff's, each `InvalidConfig` before anything runs.
  Measured on the RTX 3080 (CUDA), relative to each quantity's largest
  element: SC h 1.5e-13, cas 5.1e-15, pt2 1.2e-11, csf 1.1e-14, total
  1.7e-13; PC h 2.0e-13, cas 7.1e-15, pt2 2.4e-13, csf 1.2e-15, total
  1.8e-13; the H_eff gap 2.1e-16 in both. 0.5 s (SC) and 0.7 s (PC) in the
  tier's GPU batch, so both are in the fast set.

## Files

| File | Role |
|------|------|
| `nevpt2_nac.cppm` | Module interface: `Nevpt2NacOutput`, `Nevpt2NacStep` |
| `nevpt2_nac.cpp` | `Nevpt2NacStep::run`: the refusals, the `Solution`, H_eff and R, `qd_sc_nac` or `qd_pc_nac` per pair, and `nevpt2_nac.json` |
| `CMakeLists.txt` | The module library |
