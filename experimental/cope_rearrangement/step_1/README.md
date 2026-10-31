# `cope_rearrangement/step_1/`

Setup and step 1 of the chair Cope rearrangement of 1,5-hexadiene: the
C2h chair guess, the symmetry-labelled CAS(6,6), and the **relaxed C2h
CASSCF(6,6) scan of R = R16 = R34**, 1.50 to 2.60 Å in 0.05 Å steps. PySCF
throughout, geomeTRIC as the optimiser. Nothing here is built; it is a
generator and its output, like `compress_rdm3/`.

Steps 2 on (NEVPT2 and the a_u curvature, the 2D surface, the PT2 TS,
cross-checks) are not done here. `orbitals/` is what step 2 starts from.

## Method

| | |
|---|---|
| numbering | C1=C2–C3–C4–C5=C6, C3–C4 breaking, C1–C6 forming; atoms 0–5 are C1…C6 |
| basis / aux | cc-pVDZ / cc-pVDZ-JKFIT, **density-fitted** throughout (DF-RHF, DF-CASSCF and its gradient) |
| guess | `cope.chair_c2h`: two allyls stacked in parallel planes, central carbons on opposite sides, exact C2h by construction |
| B3LYP | DF-B3LYP/cc-pVDZ (PySCF's B3LYP, the VWN-RPA one Gaussian uses), minimised under C2h: R16 = R34 = **1.985 Å**, E = −234.566429021 |
| active space | CAS(6,6), 1A_g singlet (`fix_spin_(ss=0)`), per irrep **2a_g + 2b_u + 1a_u + 1b_g** |
| scan | at each R, geomeTRIC `$set distance 1 6 R` and `3 4 R`; PySCF symmetrises every step to C2h; energy converged to 1e-7 Eh, gradients to geomeTRIC's defaults |

**How the active MOs are picked.** At the B3LYP geometry, the DF-RHF
canonicals are ranked irrep by irrep by their weight on the six carbons' p
functions along each carbon's own allyl-plane normal (Löwdin basis,
`cope.pi_character`). The top 1a_g, 1b_u and 1a_u occupied and 1a_g, 1b_u
and 1b_g virtual are taken (`setup.json` lists them). C2h CASSCF cannot rotate
orbitals across irreps, so the per-irrep counts hold for the whole scan.
Every point asserts it anyway.

**Propagation.** `scan.py` converges R = 1.95 Å first, from the B3LYP geometry
and the setup orbitals. It then runs two sweeps at once, down to 1.50 and up
to 2.60, each with one CASSCF gradient scanner. That scanner carries
geometry, orbitals and CI vector from each converged point into the next.

**The jump check.** At every point `scan.py` records the singular values of
the active-space overlap with its neighbour, using the cross-geometry AO
overlap. `scan.json`'s `flags` would list any value below 0.9, any occupation
step above 0.1, and any unconverged geometry. The smallest value in the whole
scan is 0.9965, and `flags` is empty.

## Result

E relative to the lowest point (R = 1.65 Å). Occupations are the active
natural orbitals, diagonalised irrep by irrep.

| R / Å | E / Eh | ΔE / kcal mol⁻¹ | a_g | b_u | a_u | b_g | a_g′ | b_u′ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.50 | −232.98633455 | 6.51 | 1.986 | 1.237 | 1.985 | 0.015 | 0.763 | 0.014 |
| 1.55 | −232.99308975 | 2.27 | 1.983 | 1.302 | 1.982 | 0.018 | 0.698 | 0.017 |
| 1.60 | −232.99609714 | 0.38 | 1.980 | 1.375 | 1.978 | 0.022 | 0.625 | 0.020 |
| **1.65** | **−232.99670068** | **0.00** | 1.977 | 1.455 | 1.974 | 0.026 | 0.545 | 0.024 |
| 1.70 | −232.99601797 | 0.43 | 1.973 | 1.538 | 1.968 | 0.031 | 0.462 | 0.027 |
| 1.75 | −232.99491920 | 1.12 | 1.970 | 1.619 | 1.962 | 0.037 | 0.382 | 0.031 |
| 1.80 | −232.99397090 | 1.71 | 1.966 | 1.690 | 1.955 | 0.044 | 0.311 | 0.034 |
| 1.85 | −232.99342301 | 2.06 | 1.964 | 1.747 | 1.947 | 0.052 | 0.254 | 0.036 |
| *1.90* | *−232.99328977* | *2.14* | 1.961 | 1.790 | 1.938 | 0.061 | 0.212 | 0.038 |
| 1.95 | −232.99346983 | 2.03 | 1.959 | 1.821 | 1.929 | 0.070 | 0.181 | 0.040 |
| 2.00 | −232.99383431 | 1.80 | 1.957 | 1.842 | 1.919 | 0.080 | 0.160 | 0.042 |
| 2.05 | −232.99426929 | 1.53 | 1.955 | 1.858 | 1.909 | 0.090 | 0.144 | 0.044 |
| 2.10 | −232.99468956 | 1.26 | 1.953 | 1.868 | 1.898 | 0.101 | 0.133 | 0.046 |
| 2.15 | −232.99503912 | 1.04 | 1.951 | 1.876 | 1.886 | 0.113 | 0.125 | 0.048 |
| 2.20 | −232.99528658 | 0.89 | 1.950 | 1.881 | 1.874 | 0.127 | 0.120 | 0.050 |
| 2.25 | −232.99541954 | 0.80 | 1.948 | 1.885 | 1.860 | 0.141 | 0.115 | 0.051 |
| **2.30** | **−232.99543874** | **0.79** | 1.946 | 1.887 | 1.844 | 0.156 | 0.112 | 0.053 |
| 2.35 | −232.99535370 | 0.85 | 1.944 | 1.889 | 1.828 | 0.173 | 0.110 | 0.055 |
| 2.40 | −232.99517837 | 0.96 | 1.943 | 1.890 | 1.810 | 0.192 | 0.108 | 0.057 |
| 2.45 | −232.99493085 | 1.11 | 1.941 | 1.891 | 1.790 | 0.211 | 0.107 | 0.059 |
| 2.50 | −232.99462672 | 1.30 | 1.939 | 1.892 | 1.769 | 0.233 | 0.106 | 0.061 |
| 2.55 | −232.99428252 | 1.52 | 1.937 | 1.892 | 1.747 | 0.255 | 0.105 | 0.063 |
| 2.60 | −232.99391266 | 1.75 | 1.935 | 1.893 | 1.723 | 0.279 | 0.105 | 0.065 |

Every geometry converged in 7–11 steps, and R16 and R34 hit their targets to
within 3e-6 Å.

**CASSCF behaves as the guide's sanity check says it should.** E(R) under
C2h has two minima:
- a **diyl-like one at R ≈ 1.65 Å**;
- an **"aromatic" one at R ≈ 2.30 Å**, 0.79 kcal/mol above it.

A 2.1 kcal/mol ridge at R ≈ 1.90 Å separates them. The guide expects ~1.6 and
~2.2 Å. This is the spurious two-region structure that step 2's NEVPT2 is
meant to remove.

**One expectation in the guide does not hold: the diradical pair is not the
same pair at both ends.**
- **Short R.** The occupation that moves toward 1/1 is the **b_u/a_g′ pair**
  (1.455/0.545 at the diyl minimum, 1.237/0.763 at 1.50 Å), and the a_u/b_g
  pair goes *toward* 2/0. This is the physics, not a broken active space. The
  diyl's radical centres are C2 and C5, where the allyl nonbonding MO ψ2 has
  its node. Their in-phase and out-of-phase combinations are therefore built
  from ψ1/ψ3 and transform as a_g and b_u.
- **Long R.** **a_u/b_g** opens up (1.723/0.279 at 2.60 Å), heading for the
  two-allyl-radical limit, where ψ2's in-phase and out-of-phase combinations
  become degenerate.

Neither crossover is a jump: every occupation moves smoothly, and every
neighbour overlap is ≥ 0.9965.

Step 2 should watch both pairs, not only a_u/b_g.

## Files

| File | What |
|---|---|
| `cope.py` | the shared pieces: geometry builder, active-MO picker, natural occupations, overlap check |
| `setup.py` | B3LYP C2h chair → `b3lyp_c2h_chair.xyz`; DF-CASSCF(6,6) there → `setup_orbitals.npy`, `setup.json` |
| `scan.py` | the scan → `points/R*.json`, `orbitals/R*.npy`, `scan.json`, `scan.csv` |
| `points/R<r>.json` | one point: energy, R16/R34 reached, steps, residual gradient, active irreps, natural occupations, neighbour overlap, geometry (Å, PySCF's C2h frame, atom order as above) |
| `orbitals/R<r>.npy` | its converged DF-CASSCF MO coefficients (134 × 134; ncore 20, then the six active MOs), for step 2 |
| `scan.csv`, `scan.json` | every point in R order, plus the jump-check flags |

`logs/` (the PySCF/geomeTRIC output) is gitignored.

## Regenerating

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    experimental/cope_rearrangement/step_1/setup.py --threads 4
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    experimental/cope_rearrangement/step_1/scan.py --threads 4
```

On 4 cores, setup takes about 10 minutes (14 B3LYP steps) and the scan about
2.5 hours: the 1.95 Å point, then two 2-thread sweeps at ~7 minutes a point.
`scan.py --collect` rebuilds `scan.json` and `scan.csv` from `points/`
without recomputing anything.
