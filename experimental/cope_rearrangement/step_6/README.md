# `cope_rearrangement/step_6/`

Step 6 of the chair Cope study compares with experiment. On
[step 4b](../step_4/README.md#step-4b-the-c2h-stationary-points-their-hessians-the-irc-stand-in)'s
PC-NEVPT2 surface it does five things:

- checks that the hexadiene end points are minima;
- finds the equilibrium reactant;
- computes the chair barrier's ΔH‡, ΔS‡ and ΔG‡;
- locates the boat TS;
- adds two checks of step 4b's TS:
  - an energy-only check of its curvature;
  - PC walks from the two CASSCF C2h stationary points.

The level is free-breakfast's DF-PC-NEVPT2/cc-pVDZ (cc-pVDZ-JKFIT) on
DF-CASSCF(6,6) throughout. Secondary kinetic isotope effects were in the
plan and were dropped, so nothing here computes them.

## Results

| | result |
|---|---|
| **end points** | both minima: 42 real frequencies, the lowest 58.1 cm⁻¹; pc_plus and pc_minus agree to 0.016 cm⁻¹ (median 0.002). The numerical Hessians were the result while the analytic PC Hessian refused them (a degenerate inactive pair, [#1268](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/1268)). Since #1280 it takes them. Since #1306 (#1296), pc_plus's analytic Hessian is within 0.24 bars of its ladder on every element (it was 2.91) and within 8e-5 cm⁻¹ in frequency. Its asymmetry is 3.0e-8 (it was 1.1e-6). Since #1314 (a near-degenerate cluster taken whole) it is 0.21 bars and 3.6e-8: what is left is the `pc_floor` cut's floor, [#1318](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/1318) |
| **reference conformer** | ten conformer starts gave ten distinct minima, all with zero imaginary frequencies, spread over 2.03 kcal/mol. The lowest is a C1 gauche (g_pm), **0.245 kcal/mol below step 4b's minimum** (a C2 gauche, which is the g_mm start's minimum). ΔH‡ uses the Boltzmann ensemble of all ten |
| **chair ΔH‡ / ΔS‡ / ΔG‡**, 505 K | **29.30 kcal/mol / −15.14 eu / 36.94 kcal/mol** (ensemble, unscaled RRHO). Experiment: 33.5 ± 0.5 / −13.8 ± 1 / 40.5 at 505 K. ΔH‡ is 4.2 kcal/mol low; ΔS‡ is 1.3 eu more negative |
| **boat TS** | C2v at R16 = R34 = 2.1718 Å, one imaginary frequency (493.7i cm⁻¹, of B symmetry: antisymmetric under C2 and under the mirror through C2 and C5, symmetric under the mirror that exchanges the allyls). It stayed C2v (largest departure 6.9e-5 bohr, damped to 8.5e-7) |
| **boat − chair**, 505 K | ΔΔH‡ **10.24 kcal/mol**, ΔΔS‡ 3.22 eu, ΔΔG‡ 8.62. Experiment (Goldstein & Benzon): ΔΔH‡ 11.2 ± 2.0, ΔΔS‡ 10.8 ± 3.7, so ΔΔG‡ ≈ 5.8 at 505 K. ΔΔH‡ is within the experimental bar; ΔΔS‡ is not |
| **energy-only check of the TS** | PC energies alone at x_TS ± δq (δ = 0.01, 0.02 bohr) along the lowest a_u, a_g, b_g and b_u modes. The a_u one falls on both sides, symmetric to 6.6e-9 Eh, which is the residual gradient's 6.63e-9. Each second difference matches qᵀHq to ≤ 3.9e-5 Eh/bohr² (≤ 2.2e-4 relative) at δ = 0.01, inside the truncation-plus-noise bar |
| **PC walks from the CASSCF C2h points** | from the CASSCF diyl (1.6423 Å) and the CASSCF aromatic saddle (2.2836 Å), both walks end at step 4b's TS: R = 1.9223 Å, E within ±1.8e-9 Eh, 6.0e-5 and 3.7e-5 bohr RMS from it. C2h held. **The PC surface has one C2h stationary point**: no diyl minimum, no second saddle |

The chair ΔH‡ being 4 kcal/mol low is a property of the method, not the
numerics: the electronic barrier from the lowest conformer is 31.46
kcal/mol, and ZPE and thermal enthalpy take 1.83 kcal/mol off it. Step 5's
cc-pVTZ and CCSD(T) single points are what would say how much of the gap is
basis and how much is PC-NEVPT2(6,6).

## Method

| | |
|---|---|
| level | DF-CASSCF(6,6) and DF-PC-NEVPT2, cc-pVDZ / cc-pVDZ-JKFIT, all electrons correlated, as step 4b |
| walks | `[optimize]` over `[nevpt2_gradient]` `variant = "pc"`, Gaussian's Opt=Tight (max force 1.5e-5, rms 1e-5 Eh/bohr, max step 6e-5, rms 4e-5 bohr), `max_steps` 300, CASSCF `conv_tol` 1e-12, `conv_tol_grad` 1e-9 |
| Hessians | CASSCF `conv_tol_grad` 1e-10; PC at `[hessian]`'s default `pc_floor` 1e-10; **no `[memory]` cap**. Three kinds (below): analytic, numerical with the ladder, and one central difference |
| memory | `memory.json` per run: the card's `memory.used` sampled every 0.5 s by `nvidia-smi`, the process's peak taken as the peak minus the idle reading before the run. The container sees no per-process list |
| CASSCF starts | always a free-breakfast CASSCF's converged orbitals, carried with the atoms where the geometry moved (below). The one exception is `ts_walk`, step 4b's chain rerun from step 1's PySCF orbitals (projected) exactly as step 4b did. It was finished before this step's rule, and was not redone |
| thermochemistry | `thermo.py`: ideal gas, rigid rotor, harmonic oscillator, 1 atm. **Unscaled frequencies**, no scale factor anywhere. Most-abundant-isotope masses (¹H 1.00782503223, ¹²C 12). The TS's imaginary mode is left out |
| ensemble | Q = Σ_i (n_i/σ_i) q_i over distinct minima, with n optical isomers and σ the rotational symmetry number (2/\|G\| for every group met here). ΔH‡ = H‡ − Σ w_i H_i and ΔG‡ = G‡ + kT ln Σ (n_i/σ_i) e^(−G_i/kT), with the TS's own n/σ (C2h: 1/2) |
| temperatures | 505 K, the middle of Doering, Toscano & Beasley's 480–531 K (below), and 298.15 K |

### The CASSCF starts

`experiment.py prepare` writes each run's `orbitals.npy` from a converged
free-breakfast CASSCF (`casscf_orbitals.npy`). `[casscf]`'s frontier
default then takes columns 20–25 as active.

| run | start |
|---|---|
| `ts_walk` | step 1's 1.95 Å geometry and its PySCF orbitals, projected (`energies.projected`), as step 4b. **The only PySCF start** |
| `ts_hessian` | step 4b's `pc_ts.xyz`, `ts_walk`'s orbitals |
| `plus_walk`, `minus_walk` | TS ± its imaginary mode (step 4b's IRC stand-in), `ts_hessian`'s orbitals |
| end-point Hessians | step 4b's `pc_plus.xyz` / `pc_minus.xyz`, the matching walk's orbitals |
| conformers | `pc_plus.xyz` with the three dihedrals set by rigid rotations, and `plus_walk`'s orbitals rotated with the atoms: each atom's AO block × that rotation's spherical-harmonic representation (fitted on points, exact to 1e-9) |
| boat | a C2v boat built from step 4b's TS allyls, and `ts_walk`'s orbitals carried with the rigid motion that took each TS atom to its boat atom |
| CASSCF-point walks | each point's free-breakfast CASSCF, converged first from `ts_walk`'s orbitals (`diyl_sp`, `aromatic_sp`) |
| energy check | `ts_hessian`'s orbitals |

**Each start converged to the intended (6,6) space**, by natural
occupations:

| structures | active natural occupations |
|---|---|
| every hexadiene minimum (11 walks) | 1.982–1.983, 1.913–1.917, 1.909–1.913 / 0.086–0.089, 0.086–0.087, 0.017: two π bonds and the C3–C4 σ, as step 4b's ends |
| boat TS | 1.950, 1.874, 1.870 / 0.130, 0.126, 0.049 |
| CASSCF diyl, `diyl_sp` | 1.978, 1.974, **1.442, 0.558**, 0.025, 0.023. CASSCF E −232.9967169074, step 4b's to all 10 decimals |
| CASSCF aromatic, `aromatic_sp` | 1.947, 1.887, 1.849, 0.151, 0.113, 0.053. CASSCF E −232.9954445232, step 4b's to all 10 decimals |
| both CASSCF-point walks' ends | 1.961, 1.933, 1.822, 0.180, 0.066, 0.038: the PC TS's (step 4b) |

So free-breakfast selected the right space at every geometry, and PySCF was
needed nowhere past step 4b's own start.

## 1. The end points are minima

**The analytic PC Hessian refused both, until #1280.** At `pc_plus` (`plus_hessian_analytic`, kept as `runs/plus_hessian_analytic_refused_1268/`):

```
hessian::df_pc_nevpt2: the canonical inactive orbitals are not differentiable at a degenerate
pair the displacement splits: (coordinate 0, orbitals 5 and 4, gap 7.96e-07, f -5.53e-03), ...
```

The refusal comes after 239 s, on all 48 coordinates; `pc_minus` gives the
same at a gap of 8.07e-07, after 241 s. Orbitals 4 and 5 are the 1s of C1
and C6, which are 4.27 Å apart: a C2-equivalent pair that interacts almost
not at all. They are split by 8e-7 Eh, under the call's 1e-6 Eh rule.
`src/hessian/README.md` documents this refusal as conservative. The step
filed it as [#1268](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/1268)
and worked around it in no code.

**The numerical Hessian**, central differences of the analytic PC gradient
on the tree's default ladder (h = 1e-3, 5e-4, 2.5e-4 bohr, fourth-order
stencils, 404 displaced gradients), is the result. It shows the surface is
smooth at the pair:

| | pc_plus | pc_minus |
|---|---:|---:|
| imaginary frequencies | **0** | **0** |
| lowest 3 / cm⁻¹ | 58.12, 99.40, 110.92 | 58.14, 99.41, 110.92 |
| highest / cm⁻¹ | 3301.94 | 3301.95 |
| rigid-body eigenvalues before projection / cm⁻¹ | −0.67 to 0.83 | −0.40 to 1.01 |
| asymmetry / translational sum, Eh/bohr² | 8.0e-8 / 1.0e-7 | 7.6e-8 / 8.4e-8 |
| max \|D(h/2) − D(h/4)\|, Eh/bohr² | 6.4e-8 | 6.0e-8 |
| ladder ratios judged | 0 of 2304 | 0 of 2304 |
| over the 404 gradients: max \|z^c\|, smallest gap | 6.137e-3, 7.86e-7 | 6.137e-3, 7.86e-7 |
| wall clock / process peak | **2261 s / 1816 MiB** | **2178 s / 1824 MiB** |

- **No ratio was judged** because every element's finer difference is under
  ten noise floors (floor 2.5e-7, σ_g 2.2e-12). So in h the Hessian is
  converged to its noise. The rungs agree to 6.4e-8 against a largest
  element of 0.72.
- **z^c does not grow** across the displacements: 6.137e-3 against the
  undisplaced 6.131e-3. `pipeline/hessian`'s README names a growing z^c as
  the sign that the energy itself is not smooth.
- **The two spectra agree** to 0.016 cm⁻¹ at most (at 99.4 cm⁻¹), median
  0.002 cm⁻¹. The energies agree to 1.1e-10 Eh. The bar is set by
  geometry: the two geometries are mirror images only to step 4b's 8.6e-5
  bohr RMS, the Opt=Tight step threshold. The connectivity check confirms
  C3–C4 bonded at plus and C1–C6 bonded at minus.
- The translations and rotations are projected out exactly: the
  diagonalisation is in their orthogonal complement (`thermo.harmonic`). Our
  frequencies at the tree's isotope-averaged masses reproduce
  `hessian.json`'s to 1.5e-9 cm⁻¹. The other numbers here use
  most-abundant masses.

No imaginary mode appeared, so no displace-and-reminimise was needed.

### The analytic Hessian at the end points, since #1280 and #1306

[#1280](https://github.com/whatDiracIsCooking/FreeBreakfast/pull/1280)
(closing #1268) lets the analytic PC-NEVPT2 Hessian take a degenerate
inactive pair. Its gate was HIP-only, so this is the CUDA confirmation on
the RTX 3080. `plus_hessian_analytic` and `minus_hessian_analytic` were
rerun as before: the same geometries and orbitals, CASSCF `conv_tol_grad`
1e-10, no `[memory]` cap. Both completed.

[#1306](https://github.com/whatDiracIsCooking/FreeBreakfast/pull/1306)
(closing [#1296](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/1296))
then fixed the precision. Every core or virtual pair under 1e-4 Eh is now
taken by #1280's pair term rather than divided by its gap.
`plus_hessian_analytic` was rerun on it. `minus_hessian_analytic` is its
mirror image, and is still the #1280 run. Against the numerical ladder
Hessians at the same geometries:

| | pc_plus, since #1306 | pc_plus, #1280 only | pc_minus, #1280 only |
|---|---:|---:|---:|
| inactive pairs taken (closest gap / Eh) | 2 (7.96e-7, and 9.9e-6) | 1 (7.96e-7) | 1 (8.07e-7) |
| imaginary frequencies | 0 | 0 | 0 |
| lowest 3 / cm⁻¹ (isotope-averaged masses, `hessian.json`) | 58.105, 99.365, 110.884 | 58.105, 99.365, 110.884 | 58.120, 99.381, 110.891 |
| max \|ν_analytic − ν_numerical\| / cm⁻¹ | **7.9e-5** (at 1719 cm⁻¹) | 0.0014 (at 1315 cm⁻¹) | 0.0022 (at 3201 cm⁻¹) |
| median \|Δν\| / cm⁻¹ | 1.3e-5 | 1.4e-5 | 3.1e-5 |
| max \|H_analytic − H_numerical\|, Eh/bohr² | **6.0e-8** | 7.3e-7 | 7.8e-7 |
| worst element / `ladder.bar` | **0.24** | 2.91 | 3.12 |
| elements over 2 / over 1 bar (of 2304) | **0 / 0** | 21 / 67 | 14 / 32 |
| asymmetry / translational sum, analytic, Eh/bohr² | **3.0e-8 / 1.7e-8** | 1.1e-6 / 9.7e-7 | 1.6e-6 / 6.4e-7 |
| wall clock / process peak | 833 s / 7935 MiB | 836 s / 7715 MiB | 841 s / 7887 MiB |

- **Since #1306 the analytic Hessian is inside the ladder's 2-bar policy
  everywhere, with room to spare.** The worst element is 0.24 bars (C2
  z–z diagonal), against 2.91 before. The per-atom worst ranges from 0.03 to
  0.24, with no trace of the pair: C1, C6 and H13 had carried 2.9 bars.
  `ladder.bar` is the per-element floor, ≈ 2.5e-7 Eh/bohr² everywhere.
- **The asymmetry fell 36-fold, to 3.0e-8**, under the numerical ladder's
  own 8.0e-8, but not to the ≲ 1e-8 that #1306 expected. Its closest
  divided pair is a core pair 2.5e-4 Eh apart, just past the new 1e-4 Eh
  reach. It is not that pair (below, "Since #1314").
- **The frequencies move by 0.0014 cm⁻¹ at most** (median 2.2e-5), so the
  #1280 results stand in frequency. That is a twelfth of the 0.016 cm⁻¹
  between the mirror images.
- **The pair term costs nothing measurable**: 833 s against 836 s, and
  the process peak is within the run-to-run spread of the idle reading.
- **The analytic Hessian is 2.7× faster than the ladder** (833 s against
  2261 s) **and 4.4× larger in memory** (7.9 against 1.8 GiB).

History: under #1280 alone, the elements missed the ladder by up to 2.91
(plus) and 3.12 (minus) bars, on the degenerate pair's atoms. The
asymmetry was 1.1–1.6e-6. Both were the divided pairs' rounding (#1296).

### Since #1314: a near-degenerate cluster taken whole

[#1314](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/1314)
found that at the chair TS and the boat the carbon 1s chain together: four
of them are joined by pairs under the 1e-4 Eh reach. #1306 took disjoint
pairs, the closest first, and divided the cross pairs that shared an
orbital with a taken one: 8.5e-5 and 8.9e-5 Eh at the TS, 3.4e-5 to
5.4e-5 Eh at the boat. Since #1314 such a cluster is taken whole, every
pair within it (`src/hessian/README.md` "A near-degenerate cluster"). The
TS and the boat now take 7 pairs (they took 3), and nothing under the reach
is divided. plus has no cluster, and takes the same 2 pairs as before.

The three Hessians were rerun on it on the RTX 3080, with the card held
(both CUDA locks), the same inputs as the `*_before_1314` runs:

| Hessian | pairs taken | asymmetry, before → now | translational sum, before → now | max \|ΔH\| against before | max / median \|Δν\| against before, cm⁻¹ | wall clock, before → now | process peak, before → now |
|---|---|---:|---:|---:|---:|---:|---:|
| `plus_hessian_analytic` | 2 → 2 | 3.0e-8 → 3.6e-8 | 1.7e-8 → 1.9e-8 | 1.1e-7 | 1.1e-4 / 1.0e-5 | 833 → 880 s | 7935 → 8303 MiB |
| `ts_hessian` | 3 → 7 | 1.2e-8 → 3.8e-9 | 7.1e-9 → 2.6e-9 | 2.1e-8 | 1.7e-5 / 1.1e-6 | 792 → 893 s | 7749 → 8322 MiB |
| `boat_hessian` | 3 → 7 | 1.06e-7 → 7.8e-8 | 4.5e-8 → 4.2e-8 | 2.2e-7 | 5.9e-4 / 7.7e-6 | 806 → 901 s | 7785 → 8332 MiB |

Against the references:

- **plus against its ladder** (`plus_hessian`): the worst element is 0.21
  bars (it was 0.24), none over 1 bar, max \|ΔH\| 5.2e-8, and the
  frequencies within 6.9e-5 cm⁻¹ (median 1.5e-5).
- **The boat against its difference Hessian** (`boat_hessian_difference`):
  max \|ΔH\| 6.2e-7 (it was 6.7e-7), and the spectra within 0.0020 cm⁻¹
  (it was 0.0021; median 2.6e-4).

**The cluster fix does not move the Cope asymmetry beyond its run-to-run
spread.** An earlier run of the same binary, on the same inputs, without
the card lock (kept as `*_1314_unlocked`), gave plus 2.9e-8, the TS 2.8e-8
and the boat 1.26e-7. Its Hessians differ from these runs by 8.3e-8, 4.0e-8
and 4.5e-7 element by element. That is as much as these runs differ from
the before-runs, or more, so this step cannot measure the fix's effect.
It is measured on the small ladders instead: `ch4_td_631g` (Td methane,
two exact t2 triples, which #1268 refused) at 0.031 bars and
`ch4_split_631g` at 0.218 bars, both in this tree's tests.

**What sets the Cope asymmetry is the `pc_floor` cut, not a pair.** At plus
all of it is in the Hessian-vector product's part, and it scales as one over
the smallest kept metric eigenvalue (1.5e-10 at the step's `pc_floor`,
1e-10): 3.4e-8, 1.1e-9 and 1.8e-10 at `pc_floor` 1e-10, 1e-8 and 1e-7. Those
`dev_plus_*` runs used temporary instrumentation, not committed. Tightening
the CP and Z-vector solves 100-fold changes nothing, and neither does taking
every pair under 3e-3 Eh. The cut's run-to-run rounding is also what scatters
the asymmetry above. That floor is
[#1318](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/1318)'s.
The default `pc_floor` stays at 1e-10: raising it changes the PC surface.

**Cost.** Wall clock rose 6% at plus and 12–13% at the TS and the boat. The
process peak rose about 370–570 MiB at all three. plus's pairs did not change, so
its 6% and its 370 MiB are the build, not the fix: the before-runs predate
#1304 and were built with clang 20, these with clang 23. The TS's and the
boat's extra 6–7% over plus's is the cluster's products: 7 pairs, against 3.

The frequencies move by 6e-4 cm⁻¹ at most. ΔH‡, ΔS‡ and ΔG‡, the boat's
imaginary frequency, and the energy check's table (section 5) are
unchanged at their printed digits.

The thermochemistry below still uses the numerical end-point Hessians.

## 2. The reference conformer

Ten starts cover the conformer classes: about C3–C4, anti (180°) or gauche
(60°); about each C2–C3, skew (±120°) or cis (0°, C=C eclipsing C3–C4).
The symmetry-distinct combinations are four anti and six gauche
(`experiment.py`'s `CONFORMERS`). Each start is step 4b's `pc_plus` with the
three dihedrals set, minimised at PC-NEVPT2 to Opt=Tight, then given a
Hessian.

| conformer | start (C2C3C4C5, C1C2C3C4, C3C4C5C6) | minimum's dihedrals | group | E − pc_plus / kcal mol⁻¹ | evaluations | imaginary | lowest / cm⁻¹ | weight at 505 K |
|---|---|---|---|---:|---:|---:|---:|---:|
| **g_pm** | 60, 120, −120 | 65.4, 116.8, −115.8 | C1 | **−0.245** | 82 | 0 | 67.7 | **0.273** |
| a_pp | 180, 120, 120 | 176.9, 114.9, 114.9 | C2 | −0.114 | 66 | 0 | 75.0 | 0.153 |
| a_pm | 180, 120, −120 | 179.9, 114.7, −114.8 | Ci | −0.061 | 60 | 0 | 72.2 | 0.150 |
| g_mm = **pc_plus** (step 4b) | 60, −120, −120 | 64.6, −118.4, −118.4 | C2 | 0 | 68 | 0 | 58.1 | 0.153 |
| g_pp | 60, 120, 120 | 61.6, 110.1, 110.1 | C2 | 0.264 | 65 | 0 | 68.6 | 0.087 |
| a_pc | 180, 120, 0 | 178.4, 115.1, 0.4 | C1 | 0.518 | 82 | 0 | 86.3 | 0.098 |
| g_pc | 60, 120, 0 | 70.5, 117.7, 10.8 | C1 | 1.159 | 81 | 0 | 79.4 | 0.038 |
| a_cc | 180, 0, 0 | 179.9, 0.0, 0.0 | C2h | 1.269 | 48 | 0 | 81.4 | 0.007 |
| g_mc | 60, −120, 0 | 69.6, −132.2, 5.7 | C1 | 1.301 | 94 | 0 | 83.2 | 0.036 |
| g_cc | 60, 0, 0 | 73.9, −0.1, −0.1 | C2 | 1.789 | 58 | 0 | 107.3 | 0.006 |

- **The reference is the Boltzmann ensemble of all ten.** Within 0.5
  kcal/mol of the lowest there are five minima, and at 505 K kT is 1.0
  kcal/mol, so no single conformer is the reactant. The lowest, g_pm,
  carries 27% of the ensemble at 505 K and 33% at 298 K. Every minimum was
  given a Hessian.
- **Step 4b's minimum is the g_mm conformer.** The g_mm walk landed on it
  (E within 4e-7 kcal/mol, the same dihedrals), and it is 0.245 kcal/mol
  above g_pm.
- **a_pm is Ci only to 0.011 bohr.** Its walk stopped (Opt=Tight met) on a
  flat Ci-breaking direction, at dihedrals 114.68 / −114.83. It is counted
  as Ci (n/σ = 1); counted as C1 it would carry twice the weight. Every other
  structure holds its group to ≤ 0.005 bohr or misses any group by more than
  0.3 bohr.
- **The conformer Hessians are single central differences** (h = 1e-3
  bohr, 96 gradients, ~580 s and ~1.8 GiB each). The lowest, g_pm, was also
  run analytically, because its closest inactive pair is 2.7e-4 Eh apart
  and the analytic call takes it. The two agree to **0.0017 cm⁻¹ at most**
  (median 3.5e-4). The difference Hessian's asymmetry is 2.7e-7, against the
  analytic one's 7.9e-9. Four of the symmetric minima, pc_plus among them
  (a_pp, a_cc, g_pp, g_mm), have an inactive pair below 1e-6 Eh (2.6e-7 to
  7.9e-7). There the analytic call refused until #1280, and now takes them
  (below). a_pm and g_cc have one at 1.9e-6 and 1.4e-6.

**The four, analytically, since #1280.** Each conformer's analytic Hessian
was run at its walk's final geometry and compared with its single central
difference (h = 1e-3 bohr), at the same geometry. g_mm's partner is
`plus_hessian`, the ladder at the same minimum in another frame, so for g_mm
only frequencies compare. All four completed with no imaginary frequency.
a_pp was rerun since #1306; the other three are the #1280 runs:

| conformer | closest pair / Eh | max / median \|Δν\| / cm⁻¹ | max \|ΔH\| | asymmetry, analytic / difference | wall / process peak |
|---|---:|---|---:|---|---|
| a_pp, since #1306 (2 pairs taken) | 2.7e-7 | 0.0014 / 3.4e-4 | 4.8e-7 | 1.1e-8 / 2.6e-7 | 861 s / 7840 MiB |
| a_pp, #1280 only | 2.7e-7 | 0.0099 / 4.4e-4 | 2.4e-6 | 1.1e-6 / 2.6e-7 | 847 s / 7906 MiB |
| a_cc | 5.0e-7 | 0.0017 / 3.8e-4 | 4.8e-7 | 5.1e-7 / 2.9e-7 | 853 s / 7821 MiB |
| g_pp | 2.6e-7 | 0.0034 / 5.0e-4 | 4.8e-7 | 3.7e-8 / 8.2e-7 | 878 s / 7908 MiB |
| g_mm | 7.9e-7 | 0.0113 / 1.4e-3 (against the ladder at pc_plus) | — | 1.7e-6 / — | 852 s / 7820 MiB |
| g_pm (before, no pair) | 2.7e-4 | 0.0017 / 3.5e-4 | 4.8e-7 | 7.9e-9 / 2.7e-7 | 807 s / 8063 MiB |

- **a_cc and g_pp agree with their difference Hessians as g_pm does**, to
  4.8e-7 element by element, which is the difference Hessian's own
  truncation and noise. **Since #1306 so does a_pp** (4.8e-7, 0.0014 cm⁻¹).
  It takes two pairs, at 2.7e-7 and 6.2e-6 Eh. Its closest divided pair is
  a core pair 2.5e-4 Eh apart. Its asymmetry is 1.1e-8 and its translational
  sum 4.9e-9. The rerun moved its frequencies by 0.0086 cm⁻¹ at most
  (median 1e-5).
- **g_mm's 0.011 cm⁻¹** compares two Hessians at geometries 7e-5 Å apart
  and so is no tighter a test.
- History: under #1280 alone, a_pp was five times worse than the others
  (2.4e-6, 0.0099 cm⁻¹), with a 1.1e-6 asymmetry. Its 6.2e-6 Eh pair was
  divided then. a_cc and g_mm (asymmetries 5.1e-7 and 1.7e-6) are #1280 runs, not
  rerun ([#1296](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/1296)).

## 3. ΔH‡, ΔS‡, ΔG‡ against experiment

The TS is `ts_hessian`, the uncapped analytic Hessian at step 4b's
`pc_ts.xyz` (one imaginary mode, 475.15i cm⁻¹, a_u). Unscaled RRHO,
kcal/mol and cal/(mol K):

| | T / K | ΔH‡ | ΔS‡ | ΔG‡ | k / s⁻¹ |
|---|---:|---:|---:|---:|---:|
| **ensemble of ten** | 505 | **29.30** | **−15.14** | **36.94** | 1.1e-3 |
| lowest conformer alone (g_pm) | 505 | 29.63 | −11.90 | 35.64 | 4.0e-3 |
| ensemble of ten | 298.15 | 29.69 | −14.11 | 33.90 | 8.8e-13 |
| lowest conformer alone | 298.15 | 29.91 | −11.17 | 33.24 | 2.7e-12 |
| **experiment** (Doering, Toscano & Beasley) | 480–531 | **33.5 ± 0.5** | **−13.8 ± 1** | 40.5 at 505 K | |

- **ΔH‡ is 4.2 kcal/mol below experiment**, eight times its stated bar.
  The breakdown at 505 K, from the lowest conformer:
  - electronic barrier: 31.46 kcal/mol;
  - ZPE: 89.64 (TS) against 90.36 (reactant), so −0.72;
  - the rest of the thermal enthalpy: −1.11;
  - so ΔH‡ = 29.63.
  The ensemble's thermal averaging takes off another 0.33.
- **ΔS‡ agrees within about 1.3 σ**: −15.14 against −13.8 ± 1. The
  symmetry factors alone (n/σ of 1/2 for the TS, 2 for g_pm) put
  R ln 4 = 2.75 eu into the lowest-only number. The ensemble value is the
  one that corresponds to the measurement, which is of the equilibrium
  mixture.
- **ΔG‡ at 505 K is 3.5 kcal/mol low** (36.94 against 40.5), so k is about
  30 times too fast. That is the ΔH‡ error, partly offset by ΔS‡.
- **The 298 K numbers are an extrapolation** beyond the experiment's range.
  The experimental ones there assume ΔH‡ and ΔS‡ constant.

**The numpy thermochemistry against PySCF's `thermo`** at the TS, at the
same masses and σ: ZPE −1.6e-11 Eh, H 3.9e-9, S 5.3e-11 Eh/K, G −1.2e-8 Eh
(7e-6 kcal/mol), the size of differences in the physical constants.

## 4. The boat TS

**The start**: an exactly C2v boat at R16 = R34 = 2.0 Å, built from step
4b's TS allyls with the second allyl the mirror image of the first
(`boat_c2v`). The walk took 31 evaluations, 82 s, to R16 = R34 = **2.1718
Å**, E = −233.7750528465 Eh. That is **10.61 kcal/mol above the chair TS**,
electronic.

- **It stayed C2v.** The departure from the start's four operations grew
  ×~3 a step from 2.6e-11 bohr, peaked at 6.9e-5 at evaluation 20, then
  damped to 8.5e-7. The sign alternated on a fixed direction, step after
  step. That is the BFGS overshoot step 4b describes (an unlearned stiff
  non-symmetric direction), not a negative curvature pulling it off.
- **The Hessian** (`boat_hessian`, analytic, uncapped, rerun since #1314: **901 s,
  8332 MiB process peak**) has **exactly one imaginary frequency, 493.7i cm⁻¹**. The
  next is 122.4.
- **The mode's symmetry**: χ(C2) = −1, χ(σ through C2 and C5) = −1,
  χ(σ exchanging the allyls) = +1, each to 9e-11. With the C2 axis as z and
  B1 symmetric under the allyl-exchanging plane, it is **B1**. It
  lengthens one interallylic distance as it shortens the other: the
  reaction mode, as at the chair.
- **Since #1306 the boat's analytic Hessian agrees with its difference
  Hessian** (`boat_hessian_difference`: the same point, 96 gradients, 252 s,
  1793 MiB). The spectra agree to **0.0020 cm⁻¹ at most** (median 2.6e-4)
  and 6.2e-7 element by element (since #1314; 0.0021 and 6.7e-7 under
  #1306), as g_pm's do. Both have one imaginary mode, 493.72i. The #1306
  rerun moved the analytic spectrum by 0.17 cm⁻¹ at most, on the 3194 cm⁻¹
  C–H stretch where the two had disagreed.
- **Its asymmetry is 7.8e-8, the largest left in the step.** It is down
  from 1.0e-5, and under the difference Hessian's 2.7e-7. Under #1306 it
  was 1.06e-7, with four carbon 1s chained by pairs under 1e-4 Eh and the
  cross pairs divided, the closest at 3.4e-5 Eh. Since #1314 that cluster
  is taken whole (7 pairs) and nothing under the reach is divided. The
  asymmetry is now within its run-to-run spread (7.8e-8 and 1.26e-7 in two
  runs), which is the `pc_floor` cut's floor
  ([#1318](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/1318);
  section 1, "Since #1314").
- History: before #1306 the asymmetry was 1.0e-5. The 2.4e-6 Eh pair was
  divided, and the analytic and difference spectra differed by 0.17 cm⁻¹.
  It was 2.4e-4 kcal/mol in ZPE, so the thermochemistry below is unchanged
  at its printed digits.

**Boat − chair** (both TSs, unscaled RRHO, the chair at n/σ 1/2 and the
boat C2v at 1/2):

| T / K | ΔΔE_el | ΔΔH‡ | ΔΔS‡ | ΔΔG‡ |
|---:|---:|---:|---:|---:|
| 505 | 10.61 | **10.24** | **3.22** | **8.62** |
| 298.15 | 10.61 | 9.99 | 2.56 | 9.22 |
| experiment (Goldstein & Benzon) | | **11.2 ± 2.0** | **10.8 ± 3.7** | ≈ 5.8 at 505 K |

- **ΔΔH‡ agrees**: 10.24 against 11.2 ± 2.0, inside the bar.
- **ΔΔS‡ does not**: 3.2 against 10.8 ± 3.7, two experimental sigmas off.
  So ΔΔG‡ is 8.6 against 5.8. The experimental boat values are indirect:
  they come from bicyclo[2.2.0]hexane's thermal inversion, not from
  hexadiene itself (below), and its ΔS‡ carries ±3.6 eu.
- **The brief's "ΔΔH‡ ≈ 5.8 kcal/mol, Goldstein & Benzon, JACS 1977" is
  not that paper's number.** The paper is JACS 1972. Its ΔΔH‡ is 11.2. The
  ≈ 5.8 is a free-energy gap: Doering & Roth's 1962 product-ratio
  estimate was ΔΔG‡ 5.7, and Goldstein & Benzon's numbers give ΔΔG‡ ≈ 5.8
  at 505 K.

## 5. The energy-only check of the PC TS

This check uses PC-NEVPT2 energies alone (`[nevpt2]`, no derivative code),
which step 4a validated against PySCF and block2. They are taken at
x_TS ± δq, with q `ts_hessian`'s normal mode scaled to largest Cartesian
component 1, and δ = 0.01 and 0.02 bohr. 17 single points, 2 s each, from
`ts_hessian`'s orbitals.
- The second difference d2 = E₊ + E₋ − 2E₀ gives D(δ) = d2/δ² → qᵀHq.
- The first difference E₊ − E₋ → 2δ g·q, with g the TS's residual gradient.
- The bar on D(0.01) is the truncation, (D(0.02) − D(0.01))/3, plus 2σ of
  the noise.
- The noise is √6 σ_E/δ², with σ_E = 1e-10 Eh: the convergence at
  `conv_tol_grad` 1e-10, and the 10 printed decimals.
- E₀ at the TS is −233.7919654996, step 4b's to all 10 decimals. It is
  7.4e-10 below `ts_hessian`'s energy, which runs at `pc_floor` 1e-10
  where `[nevpt2]` runs at 0.

| mode | ν / cm⁻¹ | E(±0.01) − E₀ / Eh | E(±0.02) − E₀ / Eh | qᵀHq | D(0.01) − qᵀHq | bar | Richardson − qᵀHq |
|---|---:|---|---|---:|---:|---:|---:|
| a_u (0) | 475.2i | −2.1240e-5 / −2.1246e-5 | −8.4959e-5 / −8.4972e-5 | −0.424845 | −1.3e-5 | 1.5e-5 | −2.3e-5 |
| a_g (2) | 296.1 | +8.866e-6 / +9.193e-6 | +3.504e-5 / +3.726e-5 | 0.180550 | +3.9e-5 | 6.1e-5 | −1.6e-5 |
| b_u (3) | 369.2 | +4.1230e-6 / +4.1209e-6 | +1.6493e-5 / +1.6489e-5 | 0.082433 | +5.6e-6 | 1.1e-5 | −7.8e-8 |
| b_g (4) | 388.9 | +1.8923e-5 / +1.8923e-5 | +7.5699e-5 / +7.5698e-5 | 0.378459 | +6.1e-6 | 1.4e-5 | −2.9e-6 |

(Eh/bohr² per unit q; for each eigenmode qᵀHq equals λ qᵀMq, the
frequency's own eigenvalue, to the printed digits.)

- **The a_u mode is a maximum along q**, both sides lower, and **the
  others are minima**: the TS's signature, from energies alone.
- **Every D(0.01) is inside its bar.** In frequency terms the worst, a_g's
  2.2e-4 relative, is 1.1e-4 in ν, or 0.03 cm⁻¹.
- **The first differences are the residual gradient's.** a_u: 6.60e-9
  against 2δ g·q = 6.63e-9; b_u: 2.10e-9 against 2.19e-9; b_g: 3.00e-10
  against 3.00e-10. So "symmetric to round-off" holds once the TS's own
  1.6e-5 Eh/bohr a_u gradient (step 4b) is counted. a_g's first difference
  (−3.3e-7 at 0.01, −2.2e-6 at 0.02, a ratio of 6.8) is the cubic term,
  which symmetry allows only for a_g.
- **One miss, recorded.** The Richardson value (4D(0.01) − D(0.02))/3
  misses its noise-only bar (6.5e-6) for a_u (2.3e-5) and a_g (1.6e-5). Two
  steps cannot separate the δ⁴ remainder from noise: a_u's D(0.01) lies
  below qᵀHq and D(0.02) above it. Since #1306 the Hessian's own error no
  longer covers these misses. Its bound is each element's error taken as
  the asymmetry before symmetrising (3.8e-9 since #1314; 1.2e-8 under
  #1306), times (Σ|q_i|)². That is 2.3e-7 (a_u) and 4.9e-7 (a_g), 7.1e-7
  and 1.5e-6 under #1306, so what is left is the δ⁴ remainder.
  Before #1306 the asymmetry was 1.25e-6, and the bound of 7.5e-5 and
  1.6e-4 covered both.
- The table's qᵀHq and Richardson columns are from the rerun `ts_hessian`
  (#1314). They moved by ≤ 2e-6 against the run before #1306, and by
  ≤ 2e-7 against #1306's. The modes are
  phased by the displaced points' own geometries: the largest-component
  rule ties between symmetry-equivalent atoms.

## 6. PC walks from the CASSCF C2h stationary points

Step 4b found two CASSCF C2h stationary points: the diyl minimum (1.6423 Å)
and the aromatic saddle (2.2836 Å). Here each is the start of a PC walk,
with its own CASSCF orbitals (`diyl_sp`, `aromatic_sp`). At the starts,
PC-NEVPT2 puts the diyl 7.40 kcal/mol and the aromatic point 5.55 kcal/mol
above the PC TS.

| from | evaluations | wall | end R16 / R34 / Å | E − E(TS) / Eh | RMS from TS / bohr | max C2h departure / bohr (at) | final departure | max \|R16 − R34\| / Å |
|---|---:|---:|---|---:|---:|---|---:|---:|
| CASSCF diyl, 1.6423 Å | 31 | 78 s | 1.92229 / 1.92229 | +1.8e-9 | 6.0e-5 | 1.1e-4 (18) | 1.5e-6 | 4.8e-6 |
| CASSCF aromatic, 2.2836 Å | 35 | 86 s | 1.92239 / 1.92229 | −1.8e-9 | 3.7e-5 | 2.8e-5 (23) | 1.4e-5 | 9.8e-5 |

- **Both reach step 4b's TS** (1.9223 Å, −233.7919654998 Eh), with its
  natural occupations (1.961, 1.933, 1.822 / 0.180, 0.066, 0.038). The
  energies agree within the convergence. The geometries agree to the
  Opt=Tight step threshold (6e-5 bohr max).
- **C2h held.** In both walks the departure alternated in sign on a fixed
  direction, ×2–3 a step, and then damped: the BFGS overshoot, as in the
  boat walk. The aromatic walk ends 1.4e-5 bohr from C2h, with |R16 − R34|
  at 9.8e-5 Å at most. That is larger than the TS walk's 3.2e-6 Å, and
  still below the step threshold.
- **So PC-NEVPT2 has a single C2h stationary point.** Started on CASSCF's
  diyl minimum, the PC walk goes uphill in R (1.64 → 1.92 Å) and down in
  energy, to the concerted TS. Started on CASSCF's aromatic saddle, it goes
  downhill in R to the same point. Neither CASSCF structure survives as a
  stationary point at PC-NEVPT2. Combined with step 4b's one imaginary
  a_u frequency, the TS is the only C2h stationary point, and it is a
  saddle.

## The experimental references

Each was checked against Crossref.

- **W. von E. Doering, V. G. Toscano, G. H. Beasley**, "Kinetics of the
  Cope rearrangement of 1,1-dideuteriohexa-1,5-diene", *Tetrahedron*
  **27**, 5299–5306 (1971).
  [doi:10.1016/S0040-4020(01)91694-1](https://doi.org/10.1016/S0040-4020(01)91694-1).
  - ΔH‡ = 33.5 ± 0.5 kcal/mol and ΔS‡ = −13.8 ± 1 eu, as quoted with this
    paper as their source by Khuong, Houk et al., *J. Org. Chem.* **68**,
    572 (2003), [doi:10.1021/jo020595b](https://doi.org/10.1021/jo020595b).
  - The temperature: the paper's abstract, as indexed, reports kinetics at
    four temperatures, which read as about 207, 217, 237 and 258 °C
    (480–531 K).
  - The paper itself sits behind a paywall and could not be read. The
    numbers and the range are secondhand, and 505 K is that range's
    middle.
- **M. J. Goldstein, M. S. Benzon**, "Boat and chair transition states of
  1,5-hexadiene", *J. Am. Chem. Soc.* **94**, 7147–7149 (1972),
  [doi:10.1021/ja00775a046](https://doi.org/10.1021/ja00775a046), with its
  companion **94**, 5119 (1972),
  [doi:10.1021/ja00769a075](https://doi.org/10.1021/ja00769a075).
  - Boat ΔH‡ = 44.7 ± 2.0 kcal/mol and ΔS‡ = −3.0 ± 3.6 eu, also from
    Khuong et al. 2003.
  - They were found indirectly, from the stereochemical inversion of
    bicyclo[2.2.0]hexane. Hence boat − chair: ΔΔH‡ = 11.2 ± 2.0 and
    ΔΔS‡ = 10.8 ± 3.7.
  - Crossref lists no 1977 Goldstein & Benzon paper on this.

## Files

| File | What |
|---|---|
| `experiment.py` | `prepare` (each run's inputs: `RUNS_SPEC`), `run` (the binary, with wall clock and memory sampling) → `runs/<name>/`, `analyse` |
| `analyse.py` | every finished run → `results.json`, `frequencies.csv`, `geometries/`: harmonic analysis, conformer grouping, point groups and mode irreps, the ensemble thermochemistry, the boat, the energy check, the CASSCF-point walks, the PySCF thermo check |
| `thermo.py` | harmonic analysis with the rigid-body space removed exactly, RRHO, the Boltzmann ensemble, Eyring |
| `run.sh` | `experiment.py run NAME ...` holding the CUDA card (`with_gpu_card`), at most 8 OpenMP threads |
| `results.json` | everything above, per run and per part |
| `frequencies.csv` | every harmonic frequency (most-abundant masses) of every Hessian |
| `geometries/*.xyz` | each walk's final geometry (Å) |

`inputs/` and `runs/` are not committed (`.gitignore`). The configs are
regenerated by `prepare`. The runs are listed below, all from `RUNS_SPEC`:

| kind | runs |
|---|---|
| walks | `ts_walk`, `plus_walk`, `minus_walk`, `boat_walk`, `diyl_walk`, `aromatic_walk`, `conf_{a_pp,a_pm,a_pc,a_cc,g_pp,g_mm,g_pm,g_pc,g_mc,g_cc}` |
| analytic Hessians | `ts_hessian`, `boat_hessian`, `conf_g_pm_hessian_analytic`; since #1280, `plus_hessian_analytic`, `minus_hessian_analytic` and `conf_{a_pp,a_cc,g_pp,g_mm}_hessian_analytic` (before it, the first two were refused; their runs are kept as `*_refused_1268`); since #1306, `ts_hessian`, `boat_hessian`, `plus_hessian_analytic` and `conf_a_pp_hessian_analytic` rerun (the earlier runs kept as `*_before_1306`); since #1314, `ts_hessian`, `boat_hessian` and `plus_hessian_analytic` rerun again (the #1306 runs kept as `*_before_1314`, a first #1314 run without the card lock as `*_1314_unlocked`; the `dev_*` runs are #1314's probes, some on temporary instrumentation) |
| numerical, ladder | `plus_hessian`, `minus_hessian` |
| numerical, one difference | `conf_*_hessian` (nine; g_mm's minimum is `plus_hessian`'s), `boat_hessian_difference` |
| single points | `diyl_sp`, `aromatic_sp`, `fd_ts` and the 16 `fd_{au,ag,bg,bu}_{p,m}{01,02}` |

### Regenerating

Each `prepare` reads what the runs before it wrote, so run them in this
order. From the repo root, `py` is
`UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools python` and `s6` is
`experimental/cope_rearrangement/step_6`:

```sh
py $s6/experiment.py prepare ts_walk && $s6/run.sh ts_walk
py $s6/experiment.py prepare ts_hessian && $s6/run.sh ts_hessian
py $s6/experiment.py prepare plus_walk minus_walk boat_walk && $s6/run.sh plus_walk minus_walk boat_walk
py $s6/experiment.py prepare plus_hessian minus_hessian boat_hessian boat_hessian_difference \
  diyl_sp aromatic_sp fd_ts fd_au_p01 ...            # every fd_* run
py $s6/experiment.py prepare conf_a_pp ...           # the ten conf_* walks
$s6/run.sh <all of the above>
py $s6/experiment.py prepare diyl_walk aromatic_walk conf_a_pp_hessian ... conf_g_pm_hessian_analytic
$s6/run.sh diyl_walk aromatic_walk conf_a_pp_hessian ...
py $s6/experiment.py analyse
```

The whole step is about 5 GPU-hours on the RTX 3080. Most of that is the
two ladder Hessians (37 min each) and the ten conformer walks and Hessians
(5–9 and 10 min each).

| Hessian | method | wall / s | process peak / MiB |
|---|---|---:|---:|
| `ts_hessian` (since #1314; 792 s / 7749 MiB under #1306, 756 s / 8024 MiB before it) | analytic | 893 | 8322 |
| `boat_hessian` (since #1314; 806 s / 7785 MiB under #1306, 746 s / 8068 MiB before it) | analytic | 901 | 8332 |
| `conf_g_pm_hessian_analytic` | analytic | 807 | 8063 |
| `plus_hessian` | numerical, ladder | 2261 | 1816 |
| `minus_hessian` | numerical, ladder | 2178 | 1824 |
| `conf_*_hessian` (9) | numerical, one difference | 553–601 | 1783–1836 |
| `boat_hessian_difference` | numerical, one difference | 252 | 1793 |
| `plus_hessian_analytic`, `minus_hessian_analytic` (before #1280) | analytic, refused | 239, 241 | 7110, 7278 |
| `plus_hessian_analytic`, `minus_hessian_analytic` (since #1280) | analytic | 836, 841 | 7715, 7887 |
| `plus_hessian_analytic`, `conf_a_pp_hessian_analytic` (since #1306) | analytic | 833, 861 | 7935, 7840 |
| `plus_hessian_analytic` (since #1314; a newer build, section 1) | analytic | 880 | 8303 |
