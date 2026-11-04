# Single points at Park's SI geometries: the all-electron core floor

Phase 1 of #860. Each point is an SA3-QD-PC-NEVPT2 single point
(`[nevpt2] quasi_degenerate = true` over a three-root `[casscf]`) at one of
Park's own optimized geometries, compared with the absolute energies he
lists beside it. Our optimizations play no part. This tree correlates every
inactive orbital, and he froze the core ("The frozen core approximation is
invoked for all the calculations", acs.jctc.9b00919 §4). So per molecule
and basis, Δ = E_ours − E_his is his missing core correlation, plus
aux-set, density-fitting and convergence differences:

- **the mean of Δ** is an offset that every relative quantity cancels;
- **the spread of Δ** over the geometries (max − min, per state) is the
  floor under this milestone's relative gates. Those are tier 1's per-MECI
  basis-set shift (bar 0.02 eV) and tier 3's 13–14 CI vs 11–12 CI sign
  (dead band 0.03 eV). A tier's claim stands on measured ground only if the
  spread is at most ⅓ of its bar.

## The floor against the tier bars

The floor rows are Δ = PC − his PC at PC points only. Ethylene's S0 is
left out of them, because PC-QD was refused there (#874) and its Δ would
be SC − his PC. The rows marked "not a floor measurement" add that S0
point back in, so the SC − PC difference is mixed into them. They are
shown only to make that visible.

| molecule, basis | points | spread of Δ (meV), states 0 / 1 / 2 | bar it underpins | ⅓ of the bar |
|---|---|---|---|---|
| ethylene, aug-cc-pVDZ, **floor** | Py, Et, Hm (PC) | 3.7 / 5.0 / 3.7 | tier 1, 20 meV | 6.7 meV |
| ethylene, aug-cc-pVTZ, **floor** | Py, Et, C3v, Hm (PC) | 19.4 / 21.5 / 18.8 | tier 1, 20 meV | 6.7 meV |
| ethylene, aug-cc-pVDZ, all points (includes SC at S0; not a floor measurement) | + S0 (SC) | 4.7 / 63.9 / 9.5 | – | – |
| ethylene, aug-cc-pVTZ, all points (includes SC at S0; not a floor measurement) | + S0 (SC) | 19.4 / 66.8 / 18.8 | – | – |
| PSB3, cc-pVDZ, **floor** | all 4 (PC) | 8.0 / 8.1 / 7.9 | tier 3, 30 meV | 10 meV |
| PSB3, cc-pVTZ, **floor** | all 4 (PC) | 39.8 / 46.1 / 42.7 | tier 3, 30 meV | 10 meV |

The bars are unchanged, and the verdicts belong to the milestone owner.
What the numbers say:

- **Tier 1 (ethylene).** Over the PC MECIs, the aug-cc-pVDZ floor is
  3.7–5.0 meV, below ⅓ of the bar. That is over 3 of 5 points: C3v is
  #865, and S0 is #874. The aug-cc-pVTZ floor is 18.8–21.5 meV, which is
  comparable to the bar itself. Tier 1's shift is referenced to the S0
  minimum, so **it is not measurable as PC until #874 lands**. A shift
  between MECIs needs no S0. One such shift is E(MECI) − E(Py MECI), taken
  from aug-cc-pVDZ to aug-cc-pVTZ, at his geometries:

  | MECI − Py MECI, shift aug-cc-pVDZ → aug-cc-pVTZ | ours | his | ours − his |
  |---|---|---|---|
  | Et − Py | −0.0487 eV | −0.0440 eV | −4.7 meV |
  | Hm − Py | −0.0776 eV | −0.0614 eV | −16.2 meV |
  | C3v − Py | – (#865) | – | – |

  At aug-cc-pVTZ alone, E(MECI) − E(Py MECI) differs from his by −5.3
  (Et), −8.5 (C3v) and −20.5 meV (Hm). At aug-cc-pVDZ it differs by −0.5
  (Et) and −4.2 meV (Hm). Each is the floor acting on a MECI−MECI energy,
  with no optimization involved.
- **Tier 3 (PSB3): the gate is a sign.** At his geometries:
  - cc-pVDZ: E(13–14 CI) − E(11–12 CI) is −0.195 eV for us and −0.200 eV
    for him.
  - cc-pVTZ: the same quantity is +0.105 eV for us and +0.074 eV for him.

  **The signs agree in both bases.** At cc-pVTZ the core floor contributes
  +31 meV to that value (+4.9 meV at cc-pVDZ). That moves the magnitude,
  not the sign: +0.105 eV is well outside the 0.03 eV dead band. The
  cc-pVTZ floor's spread (40–46 meV) does not make the sign gate
  non-discriminating. A "reproduced" claim for tier 3 must carry the
  +31 meV core-floor contribution to the cc-pVTZ value. The cc-pVDZ floor
  (8 meV) is below ⅓ of the dead band.

The spread grows by about 4–6× from double to triple zeta, in both molecules.
The mean Δ grows by about 6× (ethylene −5.4 → −35.1 mEh, PSB3 −14.9 →
−92.8 mEh). That is what core correlation should do: cc-pVDZ and
aug-cc-pVDZ have almost no functions that can correlate a 1s shell, and the
TZ sets have some. So the spread looks like the geometry dependence of the
core correlation energy. Phase 2 (#859) tests this: with the C and N 1s
frozen, both the mean and the spread should collapse.

## The points

[`geometries/`](geometries/) holds his 18 structures, converted from bohr
at 1 bohr = 0.529177210903 Å, with no other change and his atom order
(PSB3: C1…C5, N, then H). Each `.xyz` comment line cites its page: J. W.
Park, *J. Chem. Theory Comput.* (2020), acs.jctc.9b00919, Supporting
Information, pp. S2–S14. Each config gives his three energies in its header.

| file stem | structure | basis / aux | SI page |
|---|---|---|---|
| `ethylene_adz_{s0,py,et,c3v,hm}` | S0 min; Py, Et, C3v, Hm MECIs | aug-cc-pVDZ / aug-cc-pVDZ-JKFIT | S2–S4 |
| `ethylene_atz_{s0,py,et,c3v,hm}` | the same | aug-cc-pVTZ / aug-cc-pVTZ-JKFIT | S5–S6 |
| `psb3_dz_{s0,s1,ci1112,ci1314}` | FC (S0 min), S1 min, 11–12 CI, 13–14 CI | cc-pVDZ / cc-pVDZ-JKFIT | S8–S10 |
| `psb3_tz_{s0,s1,ci1112,ci1314}` | the same | cc-pVTZ / cc-pVTZ-JKFIT | S12–S14 |

The method in each config: DF-RHF (`max_iterations = 400`), then DF-SA3-CASSCF.
Ethylene is (2e,2o) at `conv_tol_grad = 1e-10`. PSB3 is (6e,6o) with
`charge = 1`, at `conv_tol_grad = 1e-9`, because at 1e-10 the PSB3 CASSCF
can stall (#867). Then `[nevpt2] quasi_degenerate = true`. The aug JKFIT
sets are #845's. Run from the root of the tree:

```sh
bash -c '. devtools/lib.sh; with_gpu_card cuda build/src/free-breakfast \
  -i docs/reproduction/park2019/single_points -o <out> \
  docs/reproduction/park2019/single_points/<stem>.toml'
```

### Status of each point

15 of 18 have a PC-QD-NEVPT2 result. In each of those 15 logs, `main` printed
the three `PC-QD-NEVPT2 state` lines, with no `PC-QD-NEVPT2: refused` and no
PC-refusal warning. Every one ended with `completed: 0 errors, 0 warnings`.

| point | status |
|---|---|
| ethylene aug-cc-pVDZ C3v | **not run: blocked by #865** (RHF at the C3v geometries) |
| ethylene aug-cc-pVDZ S0, aug-cc-pVTZ S0 | RHF, CASSCF and SC-QD complete; **PC-QD refused** (below) |
| the other 15 | complete; PC-QD not refused |

### PC-QD refused at ethylene's S0 minimum

At both bases, S0's run logs
`PC-QD-NEVPT2: refused (nevpt2::pc_effective_hamiltonian: state 2: Sr has a
denominator λ + Δ = -1.223e-01 that is not positive)` at aug-cc-pVDZ, and
`... λ + Δ = -1.388e-01 ...` at aug-cc-pVTZ. Per root, PC-NEVPT2 root 2 is
refused the same way: Sr, −1.265e-01 and −1.421e-01. Each run ends
`completed: 0 errors, 2 warnings`. This is not #853's metric cut. It is the
other refusal rule, a non-positive denominator
([`src/nevpt2/energy/README.md`](../../../../src/nevpt2/energy/README.md),
"The cut, and refusals"). It fires on root 2, the π*² (Z) state, which is
about 13–14 eV up at the planar minimum. His SI does report PC-QD energies
there. The CASSCF is the right one: the two active natural orbitals are π
and π* (below). SC-QD's state 2 also lands 5.2 mEh (aug-cc-pVDZ) and
35.1 mEh (aug-cc-pVTZ) below his, which matches the MECIs' offsets. So this
is a refusal by this tree's rule, not a wrong active space. SC-QD's Δ at S0
is shown in parentheses and italics below and left out of every mean and
spread. Tier 1's energies are referenced to the S0 minimum, so this point
blocks tier 1 as well. It is #874.

## Active spaces, and how they were checked

The check is a π-character metric on the Molden files the run writes
(`scf.molden` and `casscf.molden`). For each MO, take the p-coefficient
weight on each heavy atom along that atom's local normal. The local normal
is the normal of the plane through the atom's three bonded neighbours, or,
for a two-coordinate carbene carbon, of its two bonds. Divide by the MO's
total p weight on heavy atoms. Diffuse shells (any exponent below 0.1) are
left out, so a Rydberg orbital does not read as π. A local normal keeps the
metric meaningful at twisted and pyramidalized geometries. A value of 1 is
pure π, and 0 is pure σ.

- **PSB3 (6e,6o).** The default frontier orbitals are not the π set at
  seven of the eight geometries. At the FC point in cc-pVDZ, for example, RHF
  orbitals 19 and 24 (0-based) have π character 0.000 and 0.005. So every
  PSB3 config names `active`: the three highest occupied and three lowest
  virtual RHF orbitals of π character at that geometry.

  | point | `active` | point | `active` |
  |---|---|---|---|
  | dz S0 | 18, 20, 21, 22, 23, 28 | tz S0 | 18, 20, 21, 22, 25, 29 |
  | dz S1 | 19, 20, 21, 22, 23, 27 | tz S1 | 18, 20, 21, 22, 25, 28 |
  | dz 11–12 CI | 19, 20, 21, 22, 23, 29 | tz 11–12 CI | 19, 20, 21, 22, 23, 30 |
  | dz 13–14 CI | 19, 20, 21, 22, 23, 24 | tz 13–14 CI | 19, 20, 21, 22, 23, 27 |

  #848's `[17, 20, 21, 22, 24, 27]` does not carry over: RHF orbital 17 has
  π character ≤ 0.04 at every one of these eight geometries. Two choices
  were less clear-cut, the cc-pVTZ 13–14 CI's sixth orbital (27, π 0.47)
  and the cc-pVDZ 11–12 CI's (29, π 0.70), because twisted virtuals mix.
  So the converged space is what counts. In every one of the eight
  `casscf.molden` files, all six active natural orbitals have π character
  ≥ 0.91 (1.000 at both minima in both bases). Their natural occupations
  are 1.95–1.96, 1.69–1.76, 1.29–1.46, 0.67–0.78, 0.15–0.21 and
  0.04–0.06, the
  three-bonding, three-antibonding pattern. Every CASSCF converged in 6–7
  macro iterations to |g| ≤ 1.0e-9.
- **Ethylene (2e,2o).** The default frontier (RHF orbitals 7 and 8) is
  used, and no config names `active`. The RHF LUMO at the planar minimum
  is a diffuse in-plane orbital (π character 0.000), but the CASSCF does
  not stay there. Both active natural orbitals in `casscf.molden` have π
  character 1.000 at S0 in both bases. At Py and Hm they have 0.70–0.98.
  At Et (π 0.00 and 1.00) they are the carbene's σ lone pair and its empty
  p, the (2e,2o) partner of π/π* along that path. At C3v (aug-cc-pVTZ,
  0.57 and 0.43) they are two equal-weight mixtures. Under SA3 over N, V
  and Z, both occupations are 1.00 at every point. The other check is Δ
  itself: within each basis it agrees across states and geometries to
  0.2 mEh (aug-cc-pVDZ) and 0.9 mEh (aug-cc-pVTZ), which a wrong active
  space would not give.

## Δ = E_ours − E_his, PC-QD-NEVPT2

E_ours is taken from the step's `PC-QD-NEVPT2 energies` log line, at 12
decimals. E_his is from the SI, at 10 decimals.

### Ethylene, aug-cc-pVDZ

| geometry | E₀ ours (Eh) | Δ₀ (mEh) | Δ₁ (mEh) | Δ₂ (mEh) |
|---|---|---|---|---|
| S0 min | PC-QD refused | *(−5.3336)* | *(−3.1568)* | *(−5.2004)* |
| Py MECI | −78.1771391607 | −5.3759 | −5.3242 | −5.4150 |
| Et MECI | −78.1669729351 | −5.3709 | −5.3690 | −5.4296 |
| C3v MECI | blocked by #865 | – | – | – |
| Hm MECI | −78.1523106473 | −5.5056 | −5.5067 | −5.5503 |
| **mean** (mEh) | | −5.4174 | −5.4000 | −5.4650 |
| **mean** (eV) | | −0.1474 | −0.1469 | −0.1487 |
| **spread** (mEh) | | 0.1347 | 0.1825 | 0.1353 |
| **spread** (meV) | | 3.7 | 5.0 | 3.7 |

### Ethylene, aug-cc-pVTZ

| geometry | E₀ ours (Eh) | Δ₀ (mEh) | Δ₁ (mEh) | Δ₂ (mEh) |
|---|---|---|---|---|
| S0 min | PC-QD refused | *(−35.3485)* | *(−33.1018)* | *(−35.0967)* |
| Py MECI | −78.2805163173 | −34.8284 | −34.7644 | −34.9346 |
| Et MECI | −78.2721363957 | −34.9897 | −34.9901 | −35.0151 |
| C3v MECI | −78.2680573075 | −35.1094 | −35.1094 | −35.1001 |
| Hm MECI | −78.2585240966 | −35.5422 | −35.5551 | −35.6268 |
| **mean** (mEh) | | −35.1174 | −35.1048 | −35.1692 |
| **mean** (eV) | | −0.9556 | −0.9552 | −0.9570 |
| **spread** (mEh) | | 0.7137 | 0.7907 | 0.6922 |
| **spread** (meV) | | 19.4 | 21.5 | 18.8 |

### PSB3, cc-pVDZ

| geometry | E₀ ours (Eh) | Δ₀ (mEh) | Δ₁ (mEh) | Δ₂ (mEh) |
|---|---|---|---|---|
| S0 min (FC) | −249.0641989341 | −15.0885 | −15.1286 | −15.1232 |
| S1 min | −249.0502380331 | −14.7944 | −14.8317 | −14.8338 |
| 11–12 CI | −248.9480208102 | −15.0334 | −15.0273 | −15.0450 |
| 13–14 CI | −248.9551950687 | −14.8480 | −14.8499 | −14.8544 |
| **mean** (mEh) | | −14.9411 | −14.9594 | −14.9641 |
| **mean** (eV) | | −0.4066 | −0.4071 | −0.4072 |
| **spread** (mEh) | | 0.2941 | 0.2969 | 0.2894 |
| **spread** (meV) | | 8.0 | 8.1 | 7.9 |

### PSB3, cc-pVTZ

| geometry | E₀ ours (Eh) | Δ₀ (mEh) | Δ₁ (mEh) | Δ₂ (mEh) |
|---|---|---|---|---|
| S0 min (FC) | −249.3848148202 | −93.5332 | −93.7279 | −93.7327 |
| S1 min | −249.3762056130 | −92.1057 | −92.2958 | −92.3177 |
| 11–12 CI | −249.2714348484 | −93.2370 | −93.1799 | −93.3173 |
| 13–14 CI | −249.2675474682 | −92.0704 | −92.0354 | −92.1623 |
| **mean** (mEh) | | −92.7366 | −92.8098 | −92.8825 |
| **mean** (eV) | | −2.5235 | −2.5255 | −2.5275 |
| **spread** (mEh) | | 1.4628 | 1.6925 | 1.5705 |
| **spread** (meV) | | 39.8 | 46.1 | 42.7 |

## Relative energies beside his

E₁ − E₀ is the MECI gap at a MECI, and S1 − S0 at a minimum. The last two
columns are state 0 relative to state 0 at the S0 minimum, as in his
Tables 1 and 2. For ethylene, S0's values come from SC-QD, because PC-QD
was refused there (#874). Every value that depends on that SC-QD point is
in italics. Those values mix SC − PC into the comparison, so they are not
PC measurements.

### Ethylene, aug-cc-pVDZ

| geometry | E₁ − E₀ ours | E₁ − E₀ his | E₀ − E₀(S0 min) ours (eV) | his (eV) |
|---|---|---|---|---|
| S0 min | *7.5801 eV* | 7.5208 eV | 0 | 0 |
| Py MECI | 1.589 meV | 0.182 meV | *4.4259* | 4.4270 |
| Et MECI | 0.104 meV | 0.054 meV | *4.7025* | 4.7035 |
| Hm MECI | 0.936 meV | 0.967 meV | *5.1015* | 5.1062 |

### Ethylene, aug-cc-pVTZ

| geometry | E₁ − E₀ ours | E₁ − E₀ his | E₀ − E₀(S0 min) ours (eV) | his (eV) |
|---|---|---|---|---|
| S0 min | *7.7041 eV* | 7.6430 eV | 0 | 0 |
| Py MECI | 1.765 meV | 0.024 meV | *4.5231* | 4.5090 |
| Et MECI | 0.114 meV | 0.124 meV | *4.7511* | 4.7414 |
| C3v MECI | 0.005 meV | 0.005 meV | *4.8621* | 4.8556 |
| Hm MECI | 0.243 meV | 0.595 meV | *5.1216* | 5.1268 |

### PSB3, cc-pVDZ

| geometry | E₁ − E₀ ours | E₁ − E₀ his | E₀ − E₀(S0 min) ours (eV) | his (eV) |
|---|---|---|---|---|
| S0 min (FC) | 4.2442 eV | 4.2453 eV | 0 | 0 |
| S1 min | 3.7019 eV | 3.7029 eV | 0.3799 | 0.3719 |
| 11–12 CI | 0.235 meV | 0.070 meV | 3.1614 | 3.1599 |
| 13–14 CI | 0.235 meV | 0.286 meV | 2.9661 | 2.9596 |

### PSB3, cc-pVTZ

| geometry | E₁ − E₀ ours | E₁ − E₀ his | E₀ − E₀(S0 min) ours (eV) | his (eV) |
|---|---|---|---|---|
| S0 min (FC) | 4.2334 eV | 4.2387 eV | 0 | 0 |
| S1 min | 3.8707 eV | 3.8758 eV | 0.2343 | 0.1954 |
| 11–12 CI | 1.981 meV | 0.426 meV | 3.0852 | 3.0772 |
| 13–14 CI | 1.139 meV | 0.186 meV | 3.1910 | 3.1512 |

At his MECIs, our gaps are up to 1.8 meV (ethylene Py) and 2.0 meV (PSB3
11–12 CI, cc-pVTZ), against his 0.02–1 meV. Correlating the core moves the
seam slightly away from his geometry.

### The tiers' own quantities at his geometries

These are the gated quantities themselves. Both sides are evaluated at his
structures, so any difference is the floor (core correlation plus aux and
DF) acting on that quantity, with no optimization involved. A MECI's energy
is the mean of states 0 and 1.

| quantity | ours | his | ours − his |
|---|---|---|---|
| tier 3, E(13–14 CI) − E(11–12 CI), cc-pVDZ | −0.1952 eV | −0.2002 eV | +4.9 meV |
| tier 3, E(13–14 CI) − E(11–12 CI), cc-pVTZ | +0.1054 eV | +0.0739 eV | **+31.4 meV** |
| tier 1, S0-referenced shift aug-cc-pVDZ → aug-cc-pVTZ (Py, Et, C3v, Hm) | not measurable as PC until #874 | +0.0819, +0.0379, +0.0162, +0.0205 eV | – |

His shifts are computed from the SI's 10-decimal energies, which is why
they differ in the last digit from Table 1's rounding. Tier 1's shift is
referenced to the S0 minimum, where PC-QD was refused (#874). An SC-QD
stand-in for S0 would mix the SC − PC difference into the shift, so none
is given. The MECI−MECI shifts above need no S0.

## What was run

- **Card and build:** RTX 3080 (CUDA, 10 GiB), `devtools/cpp-tier.sh
  --preset default --no-test` (`cpp-tier PASS rev=4f84cfc preset=default`),
  the tree at 4f84cfc. Every run held the card with
  `with_gpu_card cuda`.
- **Wall time:** 1–3 s for each ethylene point and 4–5 s (cc-pVDZ) or
  14–18 s (cc-pVTZ) for each PSB3 point, not counting the wait for the
  card.
- Run outputs (logs, Molden files) stay out of git, and the numbers above
  are quoted from them.
