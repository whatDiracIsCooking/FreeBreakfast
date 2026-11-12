# `cope_rearrangement/step_4/`

Step 4 of the chair Cope study: the PT2-level work on **free-breakfast's own
DF-PC-NEVPT2**. The plan is the [study README](../README.md)'s
"Step 4 with free-breakfast". Done so far: **4a, the implementation check**;
**4b, the PT2-level stationary point, its Hessian and the mode-displaced
walks**, with the CASSCF contrast; and **4d, the IRC** that replaces 4b's
mode-displaced walks as the connectivity check.

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

## Step 4b: the C2h stationary points, their Hessians, the IRC stand-in

**Result.** PC-NEVPT2 has **one C2h stationary point, at R16 = R34 =
1.9223 Å, with exactly one imaginary frequency, 475.2i cm⁻¹, of a_u
symmetry**: a concerted chair TS. Displacing ± along that mode and
minimising reaches the two mirror-image 1,5-hexadiene minima (R = 1.551 Å /
4.269 Å), with energy falling at every accepted step and no stop at a
diyl-like structure. The electronic barrier from that minimum is
**0.0497427 Eh, 31.21 kcal/mol** (DF-PC-NEVPT2/cc-pVDZ, no ZPE).

CASSCF(6,6) has **two** C2h stationary points: the diyl at R = 1.6423 Å, a
**minimum** (all 42 frequencies real, its lowest a_u 216.0 cm⁻¹), and the
"aromatic" one at R = 2.2836 Å, an **a_u saddle** (one imaginary frequency,
701.3i cm⁻¹, a_u). The diyl is 0.80 kcal/mol below the aromatic saddle.
So CASSCF's chair surface is stepwise through a diyl intermediate, and
PC-NEVPT2's is concerted: the study's deciding test comes out as expected.

The two checks that replace a symmetry projection held at the PC-NEVPT2 TS:
the wavefunction stayed C2h to round-off, and the geometry to 2e-6 bohr. One
walk, CASSCF from 2.30 Å, did leave C2h. That is the a_u saddle's negative
curvature amplifying round-off, as the Hessian confirms, not a wavefunction
breaking symmetry. Its walk ended at 1,5-hexadiene, and the aromatic point
was reached by a restart (below).

**Two things did not work as planned:**
- **The CASSCF walk from 2.30 Å left C2h.** The study README expected
  round-off to stay far below the convergence threshold. It did not:
  |R16 − R34| grew about 2× per step from 2e-11 Å. The a_u stationary point
  was then taken from that walk's evaluation 20, symmetrised by hand: one
  manual symmetrisation, recorded as such.
- **The analytic PC-NEVPT2 Hessian ran out of device memory uncapped**
  on the 10 GB RTX 3080. Under `[memory] max_device_mem_per_gpu = "8GiB"`
  it ran (below, "The Hessian's memory").

### Method

| | |
|---|---|
| level | DF-CASSCF(6,6) and DF-PC-NEVPT2 on it, cc-pVDZ / cc-pVDZ-JKFIT, no frozen core, as step 4a |
| starts | step 1's C2h geometries at R = 1.95 (PC), 1.65 and 2.30 Å (CASSCF), with step 4a's projected orbitals (`energies.projected`: the contraction projection and D) handed to `[casscf]`'s `orbitals`; the frontier default takes columns 20–25 as active |
| walks | `[optimize]` over `[nevpt2_gradient]` `variant = "pc"` or over `[casscf_gradient]`; the optimiser's BFGS from 0.5 × identity, trust radius 0.3 bohr (max 1.0), rigid-body motion projected out, all defaults |
| convergence | Gaussian's Opt=Tight: max force 1.5e-5, rms force 1e-5 Eh/bohr, max step 6e-5, rms step 4e-5 bohr; `max_steps` 300. The `[optimize]` table takes only these four and `max_steps`, so nothing tighter than Opt=Tight is configurable without code |
| CASSCF | `conv_tol` 1e-12; `conv_tol_grad` 1e-9 on the walks (the tree's own optimisation tests' choice for Opt=Tight), 1e-10 for the Hessians and their single points |
| symmetry | none imposed. No projection anywhere; the one exception is the CASSCF 2.30 Å restart's start geometry, symmetrised once by hand |
| Hessians | `[hessian]` `method = "analytic"` at each walk's final geometry, from the CASSCF orbitals the walk's outer `[casscf]` converged there. PC: `pc_floor` at its default 1e-10, and `[memory] max_device_mem_per_gpu = "8GiB"` |
| IRC stand-in | the TS geometry ± the imaginary normal mode (`hessian.json`'s Cartesian mode, phase so that + lengthens R16), scaled to **0.2 bohr in its largest Cartesian component**; R16/R34 at the start 2.1349/1.7099 Å (+) and 1.7099/2.1349 Å (−); orbitals the TS CASSCF's; then `[optimize]` over PC-NEVPT2 as above |
| symmetry analysis | the four C2h operations of each walk's start geometry (`step_2/au.py`'s `operations`, at 1e-3 bohr), their projectors onto each irrep; the a_u subspace includes the C2-axis translation, which the optimiser keeps at zero |

### The stationary points

| | method | R16 / Å | R34 / Å | E / Eh | evaluations | imaginary | imaginary mode: a_u weight, \|cos\| with R16 − R34 |
|---|---|---:|---:|---:|---:|---:|---|
| **TS** | PC-NEVPT2 | 1.9223 | 1.9223 | **−233.7919654996** | 32 | **1: 475.2i** | 1 − 8e-10, 0.923 |
| diyl | CASSCF | 1.6423 | 1.6423 | −232.9967169074 | 20 | 0 | |
| aromatic | CASSCF | 2.2836 | 2.2836 | −232.9954445232 | 1 (restart) | 1: 701.3i | 1 − 1e-16, 0.870 |

The lowest frequencies (cm⁻¹; irrep by the largest projector weight, which
is within 4e-9 of 1 for each of the 12 lowest modes analysed);
`frequencies.csv` has all 42 of each:

| | 1 | 2 | 3 | 4 | 5 | 6 | a_u modes | highest |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| PC TS | 475.2i a_u | 202.5 a_u | 296.1 a_g | 369.2 | 388.9 | 441.4 | 475.2i, 202.5, 887.5, … | 3258.6 |
| CASSCF diyl | 216.0 a_u | 289.6 b_u | 370.4 a_g | 441.4 | 482.1 | 535.8 | 216.0, 692.9, … | 3288.6 |
| CASSCF aromatic | 701.3i a_u | 150.3 a_g | 170.0 a_u | 326.0 | 353.9 | 443.3 | 701.3i, 170.0, 741.5, … | 3384.1 |

What these numbers mean:
- **The PC-NEVPT2 TS R, 1.922 Å**, is between CASPT2's too-short ~1.75 Å and
  MR-CISD/MR-AQCC's 1.9–2.0 Å (study README). It is 0.03 Å short of step 2's
  1.95 Å, which was the minimum of PC-NEVPT2 single points along CASSCF's
  relaxed C2h scan; relaxing the rest of the geometry at PC-NEVPT2 moves it
  and lowers E by 1.93e-3 Eh.
- **The imaginary mode is a_u to 8e-10** and 92% along R16 − R34: the
  reaction mode, forming C1–C6 while breaking C3–C4. The next a_u mode is
  real (202.5 cm⁻¹), so the a_u block has exactly one negative eigenvalue.
- **The CASSCF aromatic point's E is the restart's**, the same as the
  original walk's evaluation 20 (−232.9954445236) to 4e-13.
- **The PC Hessian's asymmetry before symmetrising is 1.4e-6 Eh/bohr²**, and
  its translational sum 1.3e-6, against ≤ 2.2e-10 for both CASSCF Hessians.
  Untraced. It is small beside the curvatures that decide anything here (the
  imaginary mode's mass-weighted eigenvalue is ~5 orders larger), but it is
  four orders above CASSCF's. Candidates are the near-degenerate inactive pair
  (smallest inactive gap 4.1e-6 Eh, logged by the gradient) and the PC
  conditioning floor.

### The mode-displaced walks (the IRC stand-in)

| walk | start E / Eh | evaluations | end R16 / Å | end R34 / Å | end E / Eh | ΔE from TS |
|---|---:|---:|---:|---:|---:|---:|
| TS + mode | −233.79978529 | 76 | 4.2694 | 1.5510 | −233.8417081727 | −0.0497427 Eh, −31.214 kcal/mol |
| TS − mode | (the same) | 75 | 1.5510 | 4.2693 | −233.8417081728 | −0.0497427 Eh, −31.214 kcal/mol |

- **Both reach 1,5-hexadiene**: one interallylic distance a C–C single bond
  (1.551 Å), the other 4.27 Å, and active natural occupations of two π bonds
  and a σ bond (1.983, 1.916, 1.909 / 0.089, 0.087, 0.017).
- **They are mirror images.** The + end is the − end's image under the TS's
  σ_h to 8.6e-5 bohr RMS per coordinate (7.0e-5 after a Kabsch alignment),
  and under its inversion to 1.7e-4. That is at the Opt=Tight step threshold.
- **No intermediate.** Every evaluation of both walks was accepted and the
  energy fell monotonically; R16 rose monotonically from 2.13 Å while R34
  fell to C–C at once (1.59 Å by evaluation 5). Neither walk passes a point
  with both distances near 1.6 Å, and neither stops before the hexadiene.
- **Converged to max force 1.6e-6 and 1.2e-6 Eh/bohr.** Step 6's
  Hessians show both are minima ([step 6](../step_6/README.md#1-the-end-points-are-minima)).
- **The conformer.** The walks relax past the chair-like geometry the IRC
  reaches, to R16 = 4.27 Å, through 70-odd low-gradient steps. This is the
  minimum the descent found from the IRC's end, not a search for the global
  minimum conformer. The CASSCF 2.30 Å walk that left C2h found another
  hexadiene minimum (R = 1.565/4.518 Å).

**The barrier**, E(TS) − E(that minimum), is **0.0497427 Eh = 31.21
kcal/mol** at DF-PC-NEVPT2/cc-pVDZ, electronic, no ZPE. Experiment's ΔH‡ is
≈ 33.5 kcal/mol (study README); the ZPE and thermal correction and the
reference conformer are step 6's. For contrast, the CASSCF aromatic saddle
is 0.0774018 Eh = 48.57 kcal/mol above the CASSCF hexadiene minimum that its
broken-symmetry walk reached.

### The two checks that replace a symmetry projection

**(a) Geometry drift**, per evaluation of each symmetric walk
(`stationary.json`'s `trajectory`: |R16 − R34| and the RMS distance from the
C2h-symmetrised geometry, split by irrep):

| walk | evaluations | max \|R16 − R34\| / Å | max C2h deviation / bohr | at the end / bohr | what grew |
|---|---:|---:|---:|---:|---|
| PC from 1.95 | 32 | 3.2e-6 | 2.1e-5 (evaluation 23) | 2.3e-6 | b_g, alternating in sign, ×2.5–3 a step from evaluation 8 to 23, then damped |
| CASSCF from 1.65 | 20 | 1.1e-10 | 1.1e-9 | 1.1e-9 | b_g, alternating in sign, ×2.2 a step from evaluation 11 |
| CASSCF from 2.30 | 110 | 2.97 | 0.73 | (left C2h) | **a_u, one sign, ×~2 a step from 2e-11 at evaluation 1; past 1e-4 Å at evaluation 21; the walk then fell to 1,5-hexadiene** |
| CASSCF restart at 2.28 | 1 | 0 | 3e-13 | 3e-13 | (converged at its first evaluation) |

Two kinds of growth, told apart by the sign of the departure's projection
on a fixed direction:
- **Alternating sign: the optimiser's overshoot**, not a symmetry
  instability. BFGS starts from 0.5 × identity, and its updates see only
  the steps it takes, which are totally symmetric to round-off. So in each
  non-symmetric direction it keeps the start's inverse curvature, and in a
  direction stiffer than 1 Eh/bohr² each step overshoots and amplifies.
  In the PC walk the overshoot grew until BFGS learned the direction, then
  damped. The Hessians confirm it: the PC TS's and the CASSCF diyl's b_g
  frequencies are all real.
- **One sign: negative curvature.** The CASSCF 2.30 Å walk's a_u departure
  grew monotonically. Its Hessian has the 701i a_u mode, so the walk was
  sliding off the saddle it was converging on. **This contradicts the study
  README's estimate** that round-off stays far below the ~1e-5 gradient
  threshold over the dozen or two iterations a walk takes. This walk's gradient was small at
  the start (max force 1.9e-4), so it took 20 steps to get near
  convergence. That is long enough for a ×2-a-step growth to take 2e-11 to
  1e-5. The PC walk, on a saddle too, did not show it: |R16 − R34| stayed ≤
  3.2e-6 Å over its 32 steps.

**(b) The gradient's non-symmetric part at each stationary point**, from the
outer gradient step the run wrote there (`nevpt2_gradient.json` /
`casscf_gradient.json`). Norms in Eh/bohr; d is the geometry's departure
from C2h in that irrep (bohr), H the point's own Hessian:

| point | a_g | b_g | a_u | b_u | max over irreps \|g − H d\| |
|---|---:|---:|---:|---:|---:|
| PC, step 1's exact C2h 1.95 Å start | 3.4e-2 | 9.1e-13 | 5.5e-12 | 1.4e-12 | (no Hessian; d = 0) |
| PC TS | 1.5e-5 | 9.1e-7 | 1.6e-5 | 7.9e-6 | 1.7e-10 |
| CASSCF diyl | 1.5e-5 | 7.3e-9 | 7.3e-10 | 2.5e-9 | 1.3e-12 |
| CASSCF aromatic (restart) | 2.7e-5 | 1.1e-12 | 8.7e-13 | 7.2e-13 | 5.1e-13 |

- **At an exactly C2h geometry the a_u gradient is round-off** (5.5e-12 at
  the PC start, 8.7e-13 at the symmetrised CASSCF restart). Neither CASSCF
  nor PC-NEVPT2 broke the wavefunction's symmetry at these geometries.
- **At the PC TS the a_u gradient is 1.6e-5**, at the convergence
  threshold, not at round-off. **It is entirely the geometry's**: the
  non-symmetric part of g equals H d, the Hessian times the geometry's own
  departure from C2h (2.1e-6 bohr RMS a_u), to 1.7e-10 or better in every
  irrep. What remains after the geometry is accounted for is round-off, so
  the wavefunction kept C2h. The departure is the overshoot above, damped to
  the step threshold.

### The active space

Active natural occupations, descending, with each natural orbital's C2h
irrep (PySCF's `label_orb_symm`, at the symmetrised geometry, which is
within 3e-6 bohr of the run's):

| point | occupations (irreps) |
|---|---|
| PC TS, 1.922 Å | 1.961 a_g, 1.933 a_u, 1.822 b_u, 0.180 a_g, 0.066 b_g, 0.038 b_u |
| CASSCF diyl, 1.642 Å | 1.978 a_g, 1.974 a_u, **1.442 b_u, 0.558 a_g**, 0.025 b_g, 0.023 b_u |
| CASSCF aromatic, 2.284 Å | 1.947 a_g, 1.887 b_u, 1.849 a_u, 0.151 b_g, 0.113 a_g, 0.053 b_u |
| PC hexadiene, either end | 1.983, 1.916, 1.909, 0.089, 0.087, 0.017 (C1) |

- **Each stationary point matches step 1's occupations at the nearest R**
  (step 1 at 1.95: 1.959 a_g, 1.929 a_u, 1.821 b_u, 0.181 a_g′, 0.070 b_g,
  0.040 b_u′; at 1.65: b_u 1.455 / a_g′ 0.545; at 2.30: a_u 1.844 / b_g 0.156).
  The irreps per occupation are step 1's 2a_g + 2b_u + a_u + b_g.
  **No jump**: the same active space as step 1's, with the b_u/a_g′ pair
  open at the diyl and the a_u/b_g pair the more open at the aromatic point.
- **At the PC TS the b_u/a_g′ pair is the more open (0.180)**, as at
  step 1's 1.95 Å, with the a_u/b_g pair at 0.066.
- The walks themselves logged no per-step occupations. The first and last
  points of each walk are on the same solution: energies are continuous,
  and the outer step rerun at the optimum (from the start's orbitals) gives
  the walk's last energy to 1e-13, and the Hessian run (from the optimum's
  orbitals) to 7e-13, for the PC TS.

### The Hessian's memory

The uncapped analytic PC-NEVPT2 Hessian (`pc_ts_hessian`) aborted after
194 s with `cudaErrorMemoryAllocation (out of memory)` at
`src/nevpt2/gradient/operands.cppm:95` (a `zeroed` device buffer). The
CASSCF part and the gradient had run. Under `[memory]
max_device_mem_per_gpu = "8GiB"` (`pc_ts_hessian_capped`), pre-flight
reported a 6.074 GiB device peak at `[hessian]`, and the run finished in
776 s.

The cause, found separately: the Hessian's minimum is 6.07 GiB in the pool,
4.17 GiB of it the 48-coordinate tangent stack t_b. Uncapped, its chunks
grow to fill the budget to within 64 MiB of the card. On top of that comes
about 2 GiB of device memory outside the pool: the driver's stack
reservation for kernels of up to 23 KB a thread, which the budget does not
count. That outside-pool growth is what ran the card out. The capped run
peaked at 8842 MiB, about 340 MiB short of what the card leaves free, so
8 GiB passed with little to spare. **For a rerun on a 10 GB card use
7 GiB**: it still passes pre-flight (6.69 GiB of room) and leaves ~2 GiB for
the outside-pool memory. The uncapped OOM is a free-breakfast issue (the
budget should leave room for the outside-pool reservation), reported, not
worked around in code.

**Since #1259** the budget leaves the kernels' stack its room, capped or
not. The uncapped `pc_ts_hessian`, rerun at `geometries/pc_ts.xyz`, then
completed in 749 s with a process peak of 8490 MiB. Its frequencies match
`pc_ts_hessian_capped`'s to a median of 3e-5 cm⁻¹, and at most 0.047 cm⁻¹
(the 3170 cm⁻¹ mode). The imaginary one is 475.1518i. Pre-flight now
refuses the 8 GiB cap on this card, because the cap plus the 2.2 GiB the
stack can take must fit what is free. A capped rerun needs at most about
7 GiB.

### What is untraced

- The PC Hessian's 1.4e-6 asymmetry and translational sum (above).
- ~~Whether the hexadiene end points are minima~~: traced in [step 6](../step_6/README.md#1-the-end-points-are-minima). Both are minima (42 real frequencies, the lowest 58.1 cm⁻¹, numerical Hessians: the analytic PC Hessian refuses them, #1268), and the two spectra agree to 0.016 cm⁻¹. Neither is the lowest conformer: a C1 gauche lies 0.245 kcal/mol below.
- The CASSCF walks' b_g overshoot factor implies a b_g direction with Cartesian curvature
  above 1 Eh/bohr² that BFGS had not learned; the direction itself was not
  identified.
- ~~The IRC stand-in is not an IRC~~: replaced by [step 4d](#step-4d-the-irc)'s
  IRC, which descends monotonically from the TS into the same two
  mirror-image minima, to within two and five steps of them.

### Files

| File | What |
|---|---|
| `stationary.py` | `prepare` (the inputs per run, `RUNS_SPEC`), `run` (the binary) → `runs/<name>/`, `analyse` → `stationary.json`, `frequencies.csv`, `geometries/` |
| `run.sh` | as for 4a; `run.sh stationary.py NAME ...` runs `stationary.py run` holding the CUDA card |
| `stationary.json` | settings; per run: the trajectory with drift by irrep, the final geometry's R and energy, the gradient by irrep, `gradient_from_geometry` (g against H d), the natural occupations and irreps, the frequencies and the lowest 12 modes' irrep weights and R16 − R34 overlap; the barrier and the mirror check |
| `frequencies.csv` | every harmonic frequency of the three Hessians |
| `geometries/*.xyz` | each walk's final geometry (Å): `pc_ts`, `casscf_1.65`, `casscf_2.30_restart`, `pc_plus`, `pc_minus`, and `casscf_2.30` (the hexadiene its broken-symmetry walk reached) |

Runs (`RUNS_SPEC`): `pc_point_1.95` (one PC gradient at step 1's 1.95 Å),
`pc_ts`, `casscf_1.65`, `casscf_2.30`, `casscf_2.30_restart`, the Hessians
`pc_ts_hessian` (uncapped, OOM), `pc_ts_hessian_capped`,
`casscf_1.65_hessian`, `casscf_2.30_hessian`, and `pc_plus`, `pc_minus`.

### Regenerating

Each `prepare` reads what the runs before it wrote, so in this order, from
the repo root (`py` is
`UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools python`, `s4` is
`experimental/cope_rearrangement/step_4`):

```sh
py $s4/stationary.py prepare pc_point_1.95 pc_ts casscf_1.65 casscf_2.30
$s4/run.sh stationary.py pc_point_1.95 pc_ts casscf_1.65 casscf_2.30
py $s4/stationary.py prepare casscf_2.30_restart pc_ts_hessian_capped casscf_1.65_hessian
$s4/run.sh stationary.py casscf_2.30_restart casscf_1.65_hessian pc_ts_hessian_capped
py $s4/stationary.py prepare casscf_2.30_hessian pc_plus pc_minus
$s4/run.sh stationary.py casscf_2.30_hessian pc_plus pc_minus
py $s4/stationary.py analyse
```

Wall-clock times on the RTX 3080: the PC gradient 3 s; the walks 74 s
(PC), 31 s (CASSCF 1.65), 291 s (CASSCF 2.30, 110 evaluations), 3 s
(restart), 403 s and 377 s (the two displaced PC walks, 76 and 75
evaluations); the Hessians 776 s (PC, capped), 170 s and 157 s (CASSCF);
the uncapped PC Hessian 194 s to its OOM. 41 min in all. A walk evaluation is
about 2.5 s at PC-NEVPT2 and 2.7 s at CASSCF.

## Step 4d: the IRC

**Result.** The intrinsic reaction coordinate from step 4b's PC-NEVPT2 TS
replaces the IRC stand-in. It is free-breakfast's `[irc]`
(Gonzalez–Schlegel GS2 in mass-weighted Cartesians) over the same
DF-PC-NEVPT2 gradient. This run, on main after #1300, is the final one;
"Three runs" below has the two before it.
- **Both sides fall monotonically from the TS to the hexadiene minimum**, by
  31.2 kcal/mol over 13.3–13.9 amu^½ bohr.
  - The forward end is 2.5e-6 Eh (0.002 kcal/mol) and 0.20 amu^½ bohr
    (two steps) from `pc_plus`.
  - The backward end is 1.6e-5 Eh (0.010 kcal/mol) and 0.51 amu^½ bohr from
    `pc_minus`.
- **There is no stationary point along the way.** The gradient norm rises
  from the TS to one maximum, 0.0412 Eh/bohr at |s| = 1.9, and then falls to
  the end. The one wobble is in the basin, 0.09 kcal/mol above the minimum,
  where it is the sphere's tolerance and not a feature (below).
- **The diyl region is never approached.** The longer of R16 and R34 is
  ≥ 1.922 Å everywhere on the path, and that minimum is at the TS itself. As
  one bond forms, the other breaks, from the first step.
- **The path keeps C2**, as the a_u mode does: its C2-breaking part is
  ≤ 2.9e-4 bohr, against an a_u part of ~0.5 bohr at the ends.
- **Each end, polished at Opt=Tight, is step 4b's own minimum.** The forward
  end reaches `pc_plus` (RMSD 1.2e-4 Å, ΔE 9e-7 kcal/mol). The backward end
  reaches `pc_minus` (1.0e-4 Å, 1e-7 kcal/mol). That minimum is step 6's C2
  `g_mm` conformer.

So the PC-NEVPT2 chair Cope is **concerted** by an IRC that runs from the TS
into both hexadiene minima, not only by the Hessian at its stationary point
and a minimisation from beside it.

**Where it stops, and why there.** Both sides stop on `energy_rose`: the
next sphere's minimum was above the last point, so it was not kept.
- **That is the resolution limit of the sphere's own threshold.** Near the
  minimum |g| falls to the order of `sphere_gradient` (1e-4): 7.7e-5 at the
  forward end and 2.0e-4 at the backward one. The sphere's minimum is then
  located only to the threshold, a step's energy change (~1e-6 Eh) is
  smaller than that uncertainty, and the walk stops.
- **The forward side got closer.** It reached |g| below the threshold
  before stopping, the backward side not quite; hence two steps short
  against five.
- **The rest is the `[optimize]`.** It takes 55 and 58 evaluations from
  E 0.002 and 0.010 kcal/mol above the minimum, through the soft torsion.
- **Nothing to file.** Tighter `sphere_gradient` and `max_force` would walk
  further; the defaults were kept so this is the tree's default IRC.

### Method

| | |
|---|---|
| level | DF-CASSCF(6,6) and DF-PC-NEVPT2 on it, cc-pVDZ / cc-pVDZ-JKFIT, no frozen core; `conv_tol` 1e-12, `conv_tol_grad` 1e-9 (step 4b's walk settings) |
| start | `geometries/pc_ts.xyz` (step 4b's TS); its analytic PC-NEVPT2 Hessian from step 6's `ts_hessian` (the same geometry, uncapped since #1259), loaded with `[hessian] from`; that run's CASSCF orbitals in `[casscf] orbitals`. `[irc]` carries them from point to point as `[optimize]` does |
| `[irc]` | `step_size` 0.1 amu^½ bohr, `max_points` 400 a side, `direction` both; `max_force` 4.5e-4, `sphere_gradient` 1e-4, `sphere_displacement` 1e-3, `max_sphere_iterations` 20 (the defaults) |
| code | main at a9935cb6, with `[irc]`'s #1284, #1289, #1290 and #1300 (below) |
| single points | at every fifth point of each side and its last: DF-CASSCF + `[nevpt2]`, each from the orbitals of the single point before it (the TS's at s = 0). They give the natural occupations, and check that the walk stayed on one CASSCF solution |
| the ends | `[optimize]` at Opt=Tight (step 4b's thresholds), from each side's last point with its single point's orbitals |
| symmetry | none imposed. The path is analysed in the TS's frame (`[irc]` holds the centre of mass and takes no rotating step), by projecting the displacement from the TS onto the TS's C2h irreps (`stationary.deviation_by_irrep`) |

**The step size.** The straight mass-weighted distance from the TS to step
4b's hexadiene end is 12.4 amu^½ bohr, so 0.1 (Gaussian's and the tree's
default) needed well over 100 points a side. Raising `max_points` to 400
lets the walk stop on its own rather than on the cap, and it did, at 139
and 133 points.

0.1 was kept rather than raised, for three reasons:
- **The cost is acceptable.** The whole IRC took 46 min, about 4.7 s an
  evaluation.
- **The spheres are cheap where the path is steep.** Over points 1–98, 1.35
  evaluations a point. Only the flat basin past R ≈ 3.9 Å costs more, 2–13.
- **The curve is resolved.** E(s) and R(s) are smooth at that spacing
  through the TS region.

The converged tangential gradient is ≤ 9.6e-5 Eh/(amu^½ bohr) at every
point.

### The path

E relative to the TS (−233.7919654996 Eh), R in Å, |g| the projected
Cartesian gradient's 2-norm (Eh/bohr), "sphere" the evaluations its
hypersphere took. The full table is `irc.csv`; the plot is `irc.svg`.

| s | ΔE / kcal mol⁻¹ | R16 | R34 | \|g\| | sphere |
|---:|---:|---:|---:|---:|---:|
| −13.3 (end) | −31.204 | 1.5518 | 4.1934 | 2.03e-4 | 6 |
| −12.0 | −31.112 | 1.5538 | 4.0153 | 5.01e-4 | 2 |
| −8.0 | −29.973 | 1.5606 | 3.3499 | 3.26e-3 | 1 |
| −2.0 | −9.647 | 1.6713 | 2.2609 | 4.09e-2 | 1 |
| **0 (TS)** | **0** | **1.9223** | **1.9223** | 2.3e-5 | — |
| 1.0 | −2.698 | 2.0821 | 1.7823 | 2.93e-2 | 1 |
| 2.0 | −9.647 | 2.2609 | 1.6713 | 4.09e-2 | 1 |
| 3.0 | −16.622 | 2.4526 | 1.6075 | 3.10e-2 | 1 |
| 5.0 | −25.354 | 2.8246 | 1.5723 | 1.40e-2 | 1 |
| 8.0 | −29.973 | 3.3499 | 1.5606 | 3.26e-3 | 1 |
| 10.0 | −30.780 | 3.7000 | 1.5572 | 1.12e-3 | 1 |
| 12.0 | −31.101 | 4.0003 | 1.5539 | 5.48e-4 | 5 |
| 13.9 (end) | −31.212 | 4.2395 | 1.5513 | 7.7e-5 | 9 |

| side | points after the TS | stop | sphere evaluations: all, points 1–98, past 98 | energy rises | max C2-breaking / bohr | max \|E_sp − E_irc\| / Eh |
|---|---:|---|---|---:|---:|---:|
| forward (R16 opens) | 139 | `energy_rose` | 323, 132, 191 | 0 | 2.4e-4 | 5.4e-11 |
| backward (R34 opens) | 133 | `energy_rose` | 264, 132, 132 | 0 | 2.9e-4 | 4.8e-11 |

Reading it:
- **Monotone, with one gradient maximum.** E falls at every point. |g|
  peaks at |s| = 1.9 and falls after.
- **The one wobble is in the flat basin.** At forward point 122 (s = 12.2),
  |g| is 4.86e-4, and point 123 has 4.89e-4, 0.6% more. That is 0.09
  kcal/mol above the minimum, where the sphere's tangential tolerance
  (1e-4) is a fifth of |g|. The backward side has no such wobble. A diyl
  intermediate would show as a gradient minimum approaching zero, followed
  by a rise of real size. Nothing like that appears before the basin.
- **Bond formation and breaking are synchronous near the TS, and the formed
  bond completes first.** By |s| = 3 the new bond is at 1.607 Å and the
  breaking one at 2.45 Å; by |s| = 5, 1.572 and 2.82. The rest of the
  descent opens the broken bond against a soft potential: the last
  8.9 amu^½ bohr gain 5.9 kcal/mol.
- **The two sides are mirror images through the steep part.** For
  |s| ≤ 10, E(+s) and E(−s) agree to 1.4e-7 Eh and R16(+s) and R34(−s) to
  1.7e-5 Å. In the basin they part (1.8e-5 Eh, 1.5e-2 Å by |s| = 12), as
  the spheres resolve the flat valley only to their tolerance.
- **C2 is kept.** The C2-breaking part grows from 1e-6 bohr at the TS to
  ≤ 2.9e-4 at the ends, against an a_u part of ~0.5 bohr: round-off and the
  sphere's tolerance, not a departure. The end minimum is itself C2 (`g_mm`).
- **One CASSCF solution throughout.** Every single point, each started from
  the one before it, reproduces the IRC's own energy to ≤ 5.4e-11 Eh.

### Natural occupations along the path

The active natural occupations from the single points. Both sides are the
same to ≤ 1.1e-4, and to ≤ 1e-5 through the steep part.

| \|s\| | R short / long, Å | n1 | n2 | n3 | n4 | n5 | n6 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 (TS) | 1.922 / 1.922 | 1.9608 | 1.9329 | 1.8219 | 0.1803 | 0.0657 | 0.0385 |
| 1.0 | 1.782 / 2.082 | 1.9659 | 1.9282 | 1.8463 | 0.1555 | 0.0709 | 0.0332 |
| 2.0 | 1.671 / 2.261 | 1.9743 | 1.9217 | 1.8767 | 0.1220 | 0.0802 | 0.0251 |
| 3.0 | 1.608 / 2.453 | 1.9789 | 1.9186 | 1.8897 | 0.1054 | 0.0867 | 0.0207 |
| 5.0 | 1.572 / 2.825 | 1.9813 | 1.9167 | 1.8991 | 0.0936 | 0.0909 | 0.0185 |
| 10.0 | 1.557 / 3.700 | 1.9822 | 1.9159 | 1.9074 | 0.0905 | 0.0863 | 0.0176 |
| 13.9 (end) | 1.551 / 4.240 | 1.9825 | 1.9157 | 1.9090 | 0.0888 | 0.0867 | 0.0174 |

**The diradical character falls smoothly from the TS.** At the TS the most
open pair (step 4b: b_u / a_g′) is at 0.180. It closes steadily to the
hexadiene's two π bonds and one σ bond (0.089, 0.087, 0.017), which are
step 4b's at the minimum. There is no point where a pair opens toward the
CASSCF diyl's 1.44 / 0.56 (step 4b).

### The ends against step 4b and step 6

| | forward end | backward end |
|---|---|---|
| IRC's last point | E −233.8417056566, R16/R34 4.240/1.551 | E −233.8416921663, R16/R34 1.552/4.193 |
| to the minimum | 2.5e-6 Eh (0.0016 kcal/mol), 0.20 amu^½ bohr | 1.6e-5 Eh (0.010 kcal/mol), 0.51 amu^½ bohr |
| `[optimize]` (Opt=Tight) | 55 evaluations, max force 4.5e-6, E −233.8417081712 | 58 evaluations, max force 2.0e-6, E −233.8417081730 |
| R16/R34 | 4.2693/1.5510 | 1.5510/4.2694 |
| dihedrals C2C3C4C5, C1C2C3C4, C3C4C5C6 | 64.59, −118.41, −118.41 | (the mirror) |
| against step 4b's | `pc_plus`: ΔE 9e-7 kcal/mol, RMSD 1.2e-4 Å | `pc_minus`: ΔE −1e-7 kcal/mol, RMSD 1.0e-4 Å |
| against step 6's conformers | `g_mm` (C2) to 1.0e-4 Å; every other ≥ 1.0 Å | the mirror of `g_mm` |

The two optimised ends agree with each other to 1.8e-9 Eh, and are mirror
images to 1.3e-4 Å RMSD (Kabsch with a reflection, after relabelling the
C1–C6 bonded end by the TS's inversion). **So the IRC and the stand-in
agree on what the TS connects:** the same mirror-image pair of `g_mm`
minima. The IRC now reaches them itself, to within its spheres'
resolution.

The barrier is unchanged: 0.0497427 Eh, 31.21 kcal/mol, electronic, from
this minimum (step 4b).

### Three runs

The IRC was run three times with the same inputs, as `[irc]` changed under
it. The path through the steep part is the same in all three (to 4e-6 Eh
over the shared arc); what changed is where it stops and what it costs.

| run | code | stop (forward / backward) | points | to the minimum: ΔE, mass-weighted distance | sphere evaluations, points 1–98 (forward / backward) | IRC wall |
|---|---|---|---:|---|---|---:|
| 1 | 664af84b (as merged in #1279) | `minimum_within_step` | 98 / 97 | 0.48 / 0.51 kcal/mol, 3.87 / 3.97 | 132 / 129 | 1060 s |
| 2 | 6a9a6abe: #1284, #1289, #1290 | `minimum` | 111 / 111 | 0.23 / 0.22 kcal/mol, 2.63 / 2.61 | 217 / 211 | 2294 s |
| 3 | a9935cb6: + #1300 | `energy_rose` | 139 / 133 | 0.002 / 0.010 kcal/mol, 0.20 / 0.51 | 132 / 132 | 2736 s |

- **Run 1.** `minimum_within_step` tested the line minimum along the
  steepest descent, which in this curving valley is near while the path
  goes on. It fired ~39 steps short (#1278). #1284 changed it to require
  |g| halved.
- **Run 2.** The early stop was gone, but `minimum` at the default
  `max_force` fired ~26 steps short (#1294). The later spheres' new first
  trial (#1290) took 1.64× the evaluations (#1295). #1289, which keeps the
  first sphere off the TS, changed nothing visible here.
- **Run 3.** #1300 makes the model's step the first trial again, with the
  far pole as the fallback, and requires |g| halved for `minimum` too. The
  steep part costs what run 1 did (1.35 evaluations a point), and the
  walk runs on into the basin until the sphere threshold limits it.
- **Run 3's extra wall clock over run 1 is the basin.** Past point 98 it
  takes 191 and 132 evaluations for 41 and 35 more points.

Runs 1 and 2 are kept, untracked, under `runs/before_1284/`,
`runs/after_1290/` (and the same under `inputs/`). Run 1's `irc.json`,
`irc.csv` and `irc.svg` are in git at 47ab41e4.

### Files

| File | What |
|---|---|
| `irc.py` | `prepare` (the `[irc]` input: the TS geometry, `ts_hessian`'s `hessian.json` and orbitals), `run` (the binary; a single point or end walk is prepared just before it runs, from the run before it), `names` (every run, in order), `analyse` → `irc.json`, `irc.csv`, `irc.svg`, `geometries/irc_*.xyz` |
| `irc.json` | settings; per side the stop, the point count, the end, energy rises, points in the diyl region, interior gradient minima, the sphere histogram, max C2 breaking, max single-point difference; the optimised end with its comparison to `pc_plus`/`pc_minus` and step 6's ten conformers; the two ends' mirror check; wall clocks |
| `irc.csv` | every point, backward end to forward end (s ascending): s, E, ΔE (kcal/mol), \|g\|, max force, sphere evaluations, tangential gradient, R16, R34, C2-breaking and a_u parts (bohr), and at single-point rows E_sp − E_irc and the six natural occupations |
| `irc.svg` | E(s) and R16/R34(s) (stdlib SVG, as step 2's plot; matplotlib is not in `uv.lock`) |
| `geometries/irc_forward.xyz`, `irc_backward.xyz` | the path, one frame a point, the TS first |
| `geometries/irc_forward_optimum.xyz`, `irc_backward_optimum.xyz` | each end minimised |

### Regenerating

From the repo root, after step 6's `ts_hessian` has run (`py` as above):

```sh
python3 $s4/irc.py prepare irc
$s4/run.sh irc.py irc
$s4/run.sh irc.py $(python3 $s4/irc.py names | cut -d' ' -f2-)
py $s4/irc.py analyse
```

The CPU side was held to 8 cores with `taskset -c 0-7` around `run.sh`.
Wall clock on the RTX 3080 (run 3):
- the IRC, 2736 s for both sides: 587 sphere evaluations plus the two TS
  evaluations, ~4.7 s each;
- the 57 single points, 179 s;
- the two end optimisations, 254 s and 266 s.

57 min in all.
