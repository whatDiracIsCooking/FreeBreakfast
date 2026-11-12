# `cope_rearrangement/step_4/`

Step 4 of the chair Cope study: the PT2-level work on **free-breakfast's own
DF-PC-NEVPT2**. The plan is the [study README](../README.md)'s
"Step 4 with free-breakfast". Done so far: **4a, the implementation check.**

## Step 4a: free-breakfast against PySCF/block2

**Result.** At all 23 of step 1's geometries, free-breakfast's DF-CASSCF(6,6)
and PC-NEVPT2 agree with PySCF and block2.
- Against step 2's numbers as they stand: CASSCF to ≤ 6.6e-12 Eh, PC-NEVPT2
  to ≤ 1.1e-7 Eh.
- That 1.1e-7 is two layers of convergence difference, not either code. Each
  layer is taken away in turn below. Given the same orbitals and the same CI
  vector, the two codes agree to ≤ 4.1e-10 Eh in PC E_corr and ≤ 9.2e-12 Eh in SC.

The bar is 1e-7 Ha, the one this tree's own PC-NEVPT2 tests hold against block2
(`test/gtest/nevpt2_energy_gpu.cpp`).
- **The implementation comparison** (same orbitals and CI) is inside it, by
  more than two orders of magnitude.
- **Layer 0** (each code's own convergence) is not quite: 1.11e-7 at
  R = 1.50 Å. That's step 2's orbital and CI convergence, which layers 1 and 2
  measure. The bar was not raised to cover it.

**Speed.** One point (DF-RHF, CASSCF from the projected orbitals,
SC- and PC-NEVPT2) takes about 1.5 s on an RTX 3080. The 22-point batch took
34 s of wall clock. Step 2 took about 95 s a point on 4 CPU cores (CASSCF
26 s, block2's PC 69 s).

### Method

| | |
|---|---|
| geometries | step 1's 23, unchanged (DF-CASSCF(6,6)/cc-pVDZ, relaxed under C2h at fixed R16 = R34) |
| basis / aux | this tree's `cc-pvdz` / `cc-pvdz-jkfit`. Same span as the PySCF built-ins steps 1–2 used (study README, "Steps 1–2's basis is not this tree's"), so energies compare directly |
| start | step 1's orbitals, projected onto this tree's AOs (below) and handed to `[casscf]`'s `orbitals`; its frontier default then takes columns 20–25 as active |
| CASSCF | `conv_tol` 1e-11, `conv_tol_grad` 1e-6, as step 2's `pt2.py`. Converged in 2–3 macro iterations at every point, to ‖g‖ 1.7e-10 to 9.4e-7 |
| NEVPT2 | `[nevpt2]` defaults: SC and PC, no frozen core (step 2 freezes none either) |

**Moving orbitals between the two codes** takes two factors:
- **The contraction.** PySCF's built-in cc-pVDZ and this tree's JSON are
  different contractions of one span. Project with
  C_json = S_json⁻¹ S_json,builtin C_builtin, which is exact.
- **The normalisation.** This tree never renormalises a contracted AO (#1220),
  so its AO i is PySCF's normalised AO i times D_i, the contraction's norm
  under the published coefficients (within ~1e-6 of 1 in cc-pVDZ). So
  C_tree = D⁻¹ C_json, and back again with D.
  `energies.py`'s `contraction_norms` computes D.

Without D, a free-breakfast orbital read in PySCF is off from orthonormal by
2.4e-6, and the CASCI energy on it is off by 8e-5. With D, the round trip is
exact to 1e-13 one way and 2e-10 the other.

### Layer 0: against step 2's numbers

Each code converged its own CASSCF. ΔE = free-breakfast − step 2.

| R / Å | E(CASSCF) / Eh | E(PC-NEVPT2) / Eh | ΔE_CASSCF | ΔE_corr PC | ΔE_corr SC |
|---:|---:|---:|---:|---:|---:|
| 1.50 | −232.986334554 | −233.764395605 | −3.5e-12 | −1.1e-07 | −1.1e-07 |
| 1.55 | −232.993089747 | −233.772230254 | −2.7e-12 | +5.7e-08 | +5.7e-08 |
| 1.60 | −232.996097141 | −233.777211261 | −1.2e-12 | −4.3e-08 | −4.3e-08 |
| 1.65 | −232.996700677 | −233.780646315 | −7.7e-13 | −5.0e-08 | −5.3e-08 |
| 1.70 | −232.996017971 | −233.783400930 | −1.4e-13 | +8.9e-08 | +8.9e-08 |
| 1.75 | −232.994919203 | −233.785805643 | +2.3e-13 | +3.9e-08 | +4.1e-08 |
| 1.80 | −232.993970895 | −233.787772572 | −1.7e-13 | −8.2e-08 | −8.3e-08 |
| 1.85 | −232.993423006 | −233.789131682 | −2.2e-12 | +3.4e-08 | +3.3e-08 |
| 1.90 | −232.993289772 | −233.789857976 | −1.8e-12 | −4.7e-08 | −4.8e-08 |
| **1.95** | −232.993469831 | **−233.790031942** | +6.3e-13 | −4.7e-08 | −4.1e-08 |
| 2.00 | −232.993834306 | −233.789770626 | −1.6e-12 | +3.0e-08 | +3.1e-08 |
| 2.05 | −232.994269290 | −233.789158649 | +8.5e-14 | +1.2e-08 | +1.1e-08 |
| 2.10 | −232.994689557 | −233.788254742 | −1.6e-12 | +2.4e-08 | +2.4e-08 |
| 2.15 | −232.995039117 | −233.787108886 | −6.0e-13 | +8.7e-09 | +8.3e-09 |
| 2.20 | −232.995286580 | −233.785758832 | −1.5e-12 | +6.9e-09 | +6.9e-09 |
| 2.25 | −232.995419542 | −233.784230763 | −2.6e-12 | +2.7e-09 | +2.2e-09 |
| 2.30 | −232.995438743 | −233.782556109 | −1.1e-12 | +2.6e-10 | −7.7e-11 |
| 2.35 | −232.995353697 | −233.780761567 | −2.8e-12 | −2.2e-08 | −2.1e-08 |
| 2.40 | −232.995178372 | −233.778855173 | −6.6e-12 | −4.5e-08 | −4.1e-08 |
| 2.45 | −232.994930847 | −233.776921272 | −2.2e-12 | −2.8e-09 | −7.0e-09 |
| 2.50 | −232.994626716 | −233.774929503 | −2.1e-12 | +2.5e-08 | −7.1e-09 |
| 2.55 | −232.994282519 | −233.772920500 | +5.7e-14 | +6.2e-09 | +9.0e-09 |
| 2.60 | −232.993912663 | −233.770915660 | +0.0e+00 | −9.7e-09 | −1.2e-08 |

- **CASSCF agrees to 1e-11 everywhere**: the same state, from the same
  orbitals, in a different code and basis representation.
- **The NEVPT2 differences have random signs, and SC tracks PC point by
  point.** That is the signature of a shared input, not of either PT2 code.
  The shared input is the orbitals: step 2 stopped at ‖g‖ ≈ 1e-6, and NEVPT2,
  not variational in the orbitals, carries that to first order.
- **Step 2's single PC-NEVPT2 minimum at 1.95 Å is reproduced.** The
  differences are three orders of magnitude below the curve's
  1.7e-4 Eh (0.11 kcal/mol) step from 1.90 to 1.95 Å.

SC is recorded but not judged on its own: SC changes with the active orbital
choice, so a difference could be a convention. Step 2 ran it on the active
natural orbitals. Given the same orbitals and CI, free-breakfast's SC matches
that to ~1e-11 (below), so the two use the same active orbitals here.

### Layers 1 and 2: the same orbitals, then the same CI

`crosscheck.py` takes free-breakfast's converged orbitals and runs PySCF's
DF-CASCI(6,6) on them, with no orbital optimisation, then step 2's own PC
(block2) and SC (PySCF) code. That removes the orbital layer.

What is left still has SC tracking PC. The second layer is the CI vector.
free-breakfast's CASSCF converges its CI to a residual scaled to
`conv_tol_grad` (`mcscf::ci_tolerance_for`, about 1e-8 absolute at
`conv_tol_grad` 1e-6). That leaves it ~1e-6 in norm from PySCF's CI, which
is converged to 1e-13. The energy is quadratic in that error, so ~1e-12; NEVPT2
is linear in it, so ~1e-8. The second run hands PySCF/block2 free-breakfast's
own CI vector (`casscf_ci.npy`, already in PySCF's string order) on the same
orbitals. What is left after that is the two implementations' difference.

ΔE = free-breakfast − PySCF/block2. "own CI" is PySCF's CASCI CI; "fb CI" is
free-breakfast's.

| R / Å | ΔE | own CI: ΔE_corr PC | own CI: ΔE_corr SC | ‖Δci‖ | fb CI: ΔE_corr PC | fb CI: ΔE_corr SC |
|---:|---:|---:|---:|---:|---:|---:|
| 1.50 | +2.3e−10 | −2.6e−08 | −2.8e−08 | 6.2e-07 | +1.6e−11 | −1.6e−12 |
| 1.55 | +2.3e−10 | −1.1e−08 | −1.1e−08 | 2.1e-07 | +2.8e−11 | −1.3e−12 |
| 1.60 | +2.4e−10 | −3.8e−09 | −4.2e−09 | 1.4e-07 | +6.4e−11 | −2.7e−12 |
| 1.65 | +2.4e−10 | −1.0e−08 | −1.3e−08 | 6.0e-07 | +6.4e−11 | −2.3e−12 |
| 1.70 | +2.4e−10 | −4.7e−09 | −6.7e−09 | 2.7e-07 | +4.1e−10 | −3.0e−12 |
| 1.75 | +2.4e−10 | −7.9e−09 | −7.7e−09 | 3.5e-07 | −3.7e−12 | −3.9e−12 |
| 1.80 | +2.4e−10 | −6.7e−09 | −6.7e−09 | 4.3e-07 | −4.8e−12 | −4.3e−12 |
| 1.85 | +2.4e−10 | −5.1e−09 | −4.6e−09 | 4.6e-07 | −5.6e−12 | −5.4e−12 |
| 1.90 | +2.4e−10 | −2.8e−09 | −2.0e−09 | 3.9e-07 | −6.1e−12 | −6.1e−12 |
| 1.95 | +2.4e−10 | −1.2e−09 | +1.7e−10 | 3.8e-07 | −6.8e−12 | −6.6e−12 |
| 2.00 | +2.4e−10 | +2.3e−09 | +2.4e−09 | 4.5e-07 | −6.3e−12 | −6.4e−12 |
| 2.05 | +2.4e−10 | +3.9e−09 | +3.3e−09 | 6.6e-07 | −6.9e−12 | −7.4e−12 |
| 2.10 | +2.4e−10 | +9.1e−09 | +9.0e−09 | 6.4e-07 | −7.3e−12 | −7.2e−12 |
| 2.15 | +2.4e−10 | −5.4e−09 | −4.5e−09 | 3.0e-07 | −7.9e−12 | −7.2e−12 |
| 2.20 | +2.4e−10 | −1.3e−09 | −1.3e−09 | 2.6e-07 | −7.8e−12 | −7.7e−12 |
| 2.25 | +2.4e−10 | −7.0e−09 | −7.6e−09 | 1.0e-06 | −8.3e−12 | −8.5e−12 |
| 2.30 | +2.4e−10 | −1.1e−08 | −1.1e−08 | 9.6e-07 | −8.0e−12 | −8.0e−12 |
| 2.35 | +2.4e−10 | −1.6e−08 | −1.5e−08 | 9.1e-07 | −8.2e−12 | −8.4e−12 |
| 2.40 | +2.4e−10 | −2.3e−08 | −2.1e−08 | 1.1e-06 | −8.6e−12 | −8.7e−12 |
| 2.45 | +2.4e−10 | −7.9e−11 | −4.7e−09 | 6.8e-07 | −8.4e−12 | −9.2e−12 |
| 2.50 | +2.4e−10 | +2.7e−08 | −5.3e−09 | 1.0e-06 | +2.8e−10 | −8.8e−12 |
| 2.55 | +2.4e−10 | −5.2e−09 | −2.2e−09 | 4.6e-07 | +5.5e−11 | −9.2e−12 |
| 2.60 | +2.5e−10 | −3.1e−09 | −4.1e−09 | 5.9e-07 | −9.3e−12 | −9.1e−12 |

Reading it:
- **Own CI: up to 2.8e-8, with SC tracking PC.** That's the CI layer. The
  CI distance ‖Δci‖ runs 1.4e-7 to 1.1e-6.
- **fb CI: SC to ≤ 9.2e-12 and PC to ≤ 4.1e-10.** That's the implementations'
  own difference. SC is at round-off. PC is ≤ 1e-10 at 20 of 23 points; the
  three larger ones (1.70, 2.50, 2.55 Å: 4.1e-10, 2.8e-10, 5.5e-11) are
  untraced. block2 drops near-singular PC overlap eigenvalues by its own
  threshold, and free-breakfast by its conditioning floor, which would give
  differences of this size. Both are far below the 1e-7 bar.
- **ΔE is a constant +2.4e-10 at every R**, the CASSCF energy on the same
  orbitals and CI. It's untraced too. The CODATA Bohr difference (3e-11
  relative) and the digits of D are both of a size that could give it.
  Being constant, it cancels in every relative energy.

**A note for anyone using `[nevpt2]` for 1e-8-level energies.** The NEVPT2
energy's precision is limited by the CASSCF's CI convergence, which
`conv_tol_grad` sets, not by `conv_tol`. At the default-like 1e-6 used
here, that limit is ~2e-8 Eh. Tighten `conv_tol_grad` to tighten it.

### Files

| File | What |
|---|---|
| `energies.py` | `prepare` (PySCF: the geometry, step 1's orbitals projected with D, the config) → `inputs/R*/`; `run` (the binary) → `runs/R*/`; `collect` → `energies.json`, `energies.csv` |
| `run.sh` | `energies.py run` holding the CUDA card (`with_gpu_card`), one run per card |
| `crosscheck.py` | layers 1 and 2 → `crosscheck/R*.json` |
| `energies.json`, `energies.csv` | per point: free-breakfast's CASSCF, SC and PC; step 1's and step 2's; the differences |
| `crosscheck/R<r>.json` | per point: both cross-check layers, the CI distance, block2's PC class energies |

`inputs/` and `runs/` are gitignored: `prepare` and `run` rebuild them in
under a minute.

### Regenerating

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    experimental/cope_rearrangement/step_4/energies.py prepare
experimental/cope_rearrangement/step_4/run.sh            # needs the built binary
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    experimental/cope_rearrangement/step_4/energies.py collect
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    experimental/cope_rearrangement/step_4/crosscheck.py 1.50 1.55 ... --threads 10
```

`run` takes about 35 s for all 23 points. `crosscheck.py` runs block2's PC
twice a point, about 2.5 min a point on 10 threads.
