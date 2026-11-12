# `cope_rearrangement/step_3/`

Step 3 of the chair Cope study: the relaxed 2D (R16, R34) surface, at
DF-PC-NEVPT2 and, for the contrast, DF-CASSCF(6,6), both cc-pVDZ /
cc-pVDZ-JKFIT. Each grid point holds R16 (C1–C6) and R34 (C3–C4) fixed and
relaxes everything else. It is free-breakfast's `[optimize]` with
`constraints` (#1260) over the PC-NEVPT2 or CASSCF gradient. The study
README's "Step 3, after step 4" has why.

**Result.** The two surfaces have the shapes the study predicted.

- **PC-NEVPT2 has one stationary point in the whole region:** a saddle on
  the diagonal at R16 = R34 = 1.922 Å, step 4b's TS.
  - The grid fit puts it within 1e-3 Å and 0.002 kcal/mol of 4b's
    optimised TS (1.9223 Å, −233.7919655 Eh).
  - Along the diagonal E falls monotonically from 1.50 Å to it and rises
    after. There is no diyl well: the diagonal at 1.65 Å is 5.2 kcal/mol
    above the TS.
  - Its curvatures: −114 kcal/mol/Å² across the diagonal (R16 − R34), +54
    along it.
- **CASSCF(6,6) is stepwise.** It has:
  - a diyl **minimum** on the diagonal at 1.649 Å (4b: 1.642);
  - two mirror-image **saddles** off the diagonal at (1.68, 1.81) Å,
    flanking the well, 1.1 kcal/mol above 4b's diyl energy (1.4 above
    the fit's);
  - a **maximum** on the diagonal at 1.89 Å;
  - the "aromatic" a_u **saddle** on the diagonal at 2.269 Å (4b: 2.284),
    0.04 kcal/mol from 4b's.
- **The IRC lies along the PC surface's valley.** Step 4d's path, overlaid,
  leaves the saddle across the diagonal. At every one of its 97 points in
  the grid, the relaxed surface interpolated to the path's (R16, R34) is
  below the IRC's energy, as a minimum over everything else must be, by
  0.004–0.45 kcal/mol (median 0.24): that is the bilinear interpolation's
  error on a 0.05 Å grid, plus the relaxation of the path's other
  coordinates.
- **Every point converged:** 378 of 378 at PC and 105 of 105 at CASSCF.
  Every point held its two distances to 1e-12 Å and kept C2 to ≤ 1.8e-4
  bohr. The diagonal points also kept the inversion, to ≤ 2.6e-5 bohr.

![PC-NEVPT2 surface](surface_pc.svg)
![CASSCF surface](surface_casscf.svg)

## Method

| | |
|---|---|
| level | DF-CASSCF(6,6), and DF-PC-NEVPT2 on it; cc-pVDZ / cc-pVDZ-JKFIT, no frozen core; `conv_tol` 1e-12, `conv_tol_grad` 1e-9 (step 4b's walk settings) |
| grid | PC: R16, R34 ∈ 1.50–2.80 Å at 0.05 Å (27 × 27). CASSCF: the same range at 0.1 Å (14 × 14) |
| symmetry of the grid | E(R16, R34) = E(R34, R16): the chair's inversion swaps the two bonds. So only R16 ≤ R34 is run (378 PC points, 105 CASSCF) and the rest is the mirror |
| each point | `[optimize]` with `constraints = [{ atoms = [0, 5], distance = R16 }, { atoms = [2, 3], distance = R34 }]`, Gaussian's default thresholds (max force 4.5e-4, rms force 3e-4 Eh/bohr, max step 1.8e-3, rms step 1.2e-3 bohr), `max_steps` 200 |
| sweep | the diagonal outward from (1.90, 1.90), the grid point nearest step 4b's TS, which starts from `step_4/geometries/pc_ts.xyz` with step 6's `ts_hessian` orbitals. Then each row from its diagonal point, (R, R + h), (R, R + 2h), … Every point starts from its neighbour's converged geometry and CASSCF orbitals; a point whose neighbour did not converge starts from the nearest converged one before it on the same line |
| symmetry of each point | none imposed. A diagonal start is C2h and an off-diagonal one C2. The gradient keeps that symmetry, and both constraints map to themselves under C2 (C1↔C6, C3↔C4), so the walks stay C2. Checked at every point (below) |

**The thresholds are Gaussian's default, not step 4b's Opt=Tight.** The
surface is read at the 0.1 kcal/mol level. At max force 4.5e-4 Eh/bohr, a
point is within ~g²/2k of its constrained minimum along any mode with
curvature k. For k ≥ 0.05 Eh/bohr² that is ≤ 2e-6 Eh, 1e-3 kcal/mol.
Softer modes, such as the torsions of the long-R region, can leave
more. The scatter in the grid (below) bounds it.

**Sizing.** Six timed points (0.1 Å hops, before the grid was fixed) took
18–43 s each, about 30 s a point. On that basis the first plan was 1.50–3.20 Å
for both grids: 630 PC points (~5 h) and 171 CASSCF points.

The R16 = 1.50 row, running out to R34 = 3.20 Å, then cost 50–115 s a point.
Its evaluations took 4.2 s against 2.3 s near the TS, with both the CASSCF
and the PC gradient slower. The card was not throttled and not shared. At
that rate the full grid projected to ~8.7 h, over the 6–8 h budget.

So after 73 points both grids were cut to 1.50–2.80 Å: 378 PC points and
105 CASSCF ones. The points already run are on the new grid, by the same
names and starts. The R16 = 1.50 row's points past 2.80 Å (2.85–3.20) were
run but are not part of the surface; the analysis ignores them.

The range takes the diagonal from below the CASSCF diyl (1.64 Å) to well past
the CASSCF "aromatic" point (2.28 Å). It takes the off-diagonal from below
the C–C single bond (1.55 Å) to 2.8 Å, which covers the IRC's first
4.8 amu^½ bohr each side: 25 kcal/mol of its 30.7.

**The grid's stationary points** are read off a quadratic fitted by least
squares to each 3 × 3 patch. Where the fit's stationary point falls inside
the central cell, it is kept, located, and classified by the fit's Hessian:
minimum, saddle or maximum. On an analytic double well it recovers the
minima and the saddle exactly. On the grid it is as good as the grid
spacing and each point's convergence allow, so its locations are ±h/2 at
best, and its energies the fit's.

## Results

### The stationary points

From the grid fits (`surface.json`'s `grid_stationary_points`). E is the
fit's; "4b" is step 4b's optimised stationary point at the same level.

| level | kind | R16 / Å | R34 / Å | E / Eh | against 4b | curvatures / kcal mol⁻¹ Å⁻² |
|---|---|---:|---:|---:|---|---|
| PC-NEVPT2 | saddle | 1.922 | 1.922 | −233.7919625 | TS: +2.9e-6 Eh (0.002 kcal/mol), ΔR < 1e-3 Å | −114, +54 |
| CASSCF | minimum | 1.649 | 1.649 | −232.9972706 | diyl: −5.5e-4 Eh (−0.35 kcal/mol), ΔR +0.007 Å | +309, +389 |
| CASSCF | saddle ×2 | 1.682 | 1.811 | −232.9949710 | (not located in 4b) | −122, +112 |
| CASSCF | maximum | 1.888 | 1.888 | −232.9932891 | | −283, −38 |
| CASSCF | saddle | 2.269 | 2.269 | −232.9955113 | aromatic: −6.7e-5 Eh (−0.04 kcal/mol), ΔR −0.015 Å | −244, +16 |

- **The PC saddle is the TS to the grid fit's precision**: 0.05 Å spacing,
  and a quadratic that reproduces a smooth saddle.
- **The CASSCF fits are coarser**, from a 0.1 Å grid. At the diyl, the well
  is stiff (~300–400 kcal/mol/Å²) and its fitted minimum lies at the patch's
  edge (offset 0.49 cell), so the fit's energy there is 0.35 kcal/mol below
  4b's exact one.
- **The CASSCF stepwise route** is hexadiene → off-diagonal saddle → diyl →
  mirror saddle → hexadiene. Its saddles are 1.1 kcal/mol above 4b's exact
  diyl, so the diyl is a shallow intermediate. The aromatic saddle, 0.80
  kcal/mol above the diyl in 4b, is a second, symmetric route over the
  diagonal. The maximum between the two diagonal stationary points is what
  separates them.

### The diagonal, E(R, R)

ΔE in kcal/mol from each grid's lowest point, which is on its edge
(PC (1.55, 2.80), CASSCF (1.6, 2.8)) and not a stationary point. Only the
shape along R is meant.

| R / Å | 1.50 | 1.60 | 1.65 | 1.70 | 1.80 | 1.90 | 1.95 | 2.00 | 2.10 | 2.20 | 2.30 | 2.40 | 2.60 | 2.80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PC | 41.78 | 33.10 | 30.54 | 28.59 | 26.18 | **25.34** | 25.36 | 25.58 | 26.59 | 28.17 | 30.19 | 32.53 | 37.62 | 42.50 |
| CASSCF | 44.69 | **38.57** | | 38.63 | 39.90 | 40.33 | | 39.99 | 39.46 | 39.08 | **38.99** | 39.15 | 39.95 | 40.89 |

PC has one minimum on the diagonal, which is the saddle across it. CASSCF
has two, the diyl near 1.65 and the aromatic point near 2.3, with a maximum
between them.

### The reference wavefunction on the diagonal

Active natural occupations from each point's CASSCF. For the PC grid that
is the CASSCF the PC-NEVPT2 rests on, at the PC-relaxed geometry.

| R / Å | PC grid: n3 / n4 | CASSCF grid: n3 / n4 |
|---|---|---|
| 1.60 | 1.432 / 0.568 | 1.376 / 0.624 |
| 1.65 | 1.527 / 0.474 | |
| 1.90 | 1.810 / 0.192 | 1.790 / 0.212 |
| 2.30 | 1.841 / 0.159 (n5 0.111) | 1.845 / 0.156 (n5 0.112) |

**The diradical pair opens near 1.6 Å on both surfaces.** On the PC grid's
reference it reaches 1.43 / 0.57. What differs is the energy: CASSCF puts a
minimum there and PC-NEVPT2 does not. The dynamic correlation, not a
different reference, removes the diyl, which is the study's premise.

### Convergence and cost

| | points | converged | evaluations, median / max | wall per point, median / max | total | max C2-breaking / bohr |
|---|---:|---:|---|---|---:|---:|
| PC | 378 | 378 | 11 / 36 | 37 s / 123 s | 4.28 h | 1.5e-4 |
| CASSCF | 105 | 105 | 13 / 27 | 26 s / 62 s | 0.83 h | 1.8e-4 |

Plus the R16 = 1.50 row's eight points past 2.80 Å, 0.2 h, from before the
grid was cut. No point failed, so none started from anything but its
direct neighbour.

## Files

| File | What |
|---|---|
| `surface.py` | `run METHOD [NAME ...]` (prepare and run each point in sweep order, from its neighbour; skips one done), `names METHOD` (the sweep order), `analyse` → `surface.json`, `surface_<method>.csv`, `surface_<method>.svg` |
| `contour.py` | the contour plot, stdlib SVG (matplotlib is not in `uv.lock`; step 2's plot is the precedent) |
| `run.sh` | `surface.py run` holding the CUDA card, as step 4's and step 6's |
| `surface.json` | settings; per method the run status, the points that did not converge, the grid's stationary points, the diagonal E(R), and totals |
| `surface_<method>.csv` | every point of the full square (the mirrored half marked `mirrored = 1`): status, E, ΔE from the grid minimum (kcal/mol), evaluations, max force, wall clock, the final distances, the C2-breaking measure, the six natural occupations |
| `surface_<method>.svg` | the contour plot, the IRC (step 4d) overlaid on the PC one |

`inputs/` and `runs/` are gitignored.

### Regenerating

From the repo root, after step 6's `ts_hessian` (whose orbitals seed the
sweep) and step 4d's `irc.csv` (the overlay):

```sh
taskset -c 0-7 experimental/cope_rearrangement/step_3/run.sh pc
taskset -c 0-7 experimental/cope_rearrangement/step_3/run.sh casscf
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    python experimental/cope_rearrangement/step_3/surface.py analyse
```
