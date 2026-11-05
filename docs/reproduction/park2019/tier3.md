# Tier 3: PSB3's Table 2 in cc-pVDZ and cc-pVTZ

This is the reachable half of Park's Table 2 (#850): PSB3's Franck–Condon (FC)
point, its S1 equilibrium geometry, and its 11–12CI and 13–14CI MECIs. The
level is SA-QD-PC-NEVPT2 over DF-SA3-CASSCF(6,6), in cc-pVDZ (124 basis
functions, aux cc-pVDZ-JKFIT) and cc-pVTZ (292, aux cc-pVTZ-JKFIT). Both
counts match his. [tier2a.md](tier2a.md) has the molecule, the carbon
numbering and the method.

Every number is from one card, CUDA (RTX 3080, 10 GiB), with
`--preset default` at 728da1a (`cpp-tier PASS rev=728da1a preset=default`).
Each run held the card with `with_gpu_card cuda`
([README.md](README.md), "How the runs are made"). No C++ changed, and no
tests were run for this tier.

## The headline: the sign flips

**E(13–14CI) − E(11–12CI)**, each MECI's energy taken as the mean of QD
states 0 and 1 at our own converged structure:

| | ours | his (SI) | ours − his |
|---|---|---|---|
| cc-pVDZ | **−0.1950 eV** | −0.2002 eV | +5.2 meV |
| cc-pVTZ | **+0.1079 eV** | +0.0739 eV | +34.0 meV |

- **Negative at cc-pVDZ and positive at cc-pVTZ, as his are.** The 13–14CI
  is the lower intersection in double zeta and the higher one in triple zeta.
- **The cc-pVTZ value is past the dead band.** It is +0.108 eV, against a
  band of |value| < 0.03 eV. The gate is conclusive in both bases.
- **The core floor.** He freezes the core; this tree correlates it.
  [#860](single_points/README.md) measured that difference on this same
  quantity at his own geometries. At cc-pVTZ the all-electron core adds
  **+31.4 meV** (ours +0.1054 against his +0.0739 eV); at cc-pVDZ it adds
  +4.9 meV. Our optimized +0.1079 eV is that +0.1054 eV moved by another
  +2.5 meV through reoptimization. Take the +31 meV floor out and about
  +0.077 eV is left, which is his value. **The floor changes the size of
  the cc-pVTZ value, not its sign.**

The settings that produced it:

- `[casscf] conv_tol_grad = 1e-9`. At 1e-10, `mcscf::solve` stalls on PSB3
  (#867). At 1e-9 the bound's orbital term s_o ‖g‖ is ≤ 2.3e-10 Eh, far
  below `max_gap`.
- `[nevpt2_gradient] quasi_degenerate = true, variant = "pc"`.
- The MECIs use `meci = [0, 1]`, `max_gap = 1e-7` Eh, and Opt=Tight's four
  criteria (max/rms force 1.5e-5/1e-5 Eh/bohr, max/rms displacement
  6e-5/4e-5 bohr), as tier 2a's do.

## The runs

The cc-pVDZ FC, 13–14CI and 11–12CI are tier 2a's (#848) and were not
rerun. This tier ran the three cc-pVTZ structures and the cc-pVDZ S1 walk.
The S1 walk did not converge (see [below](#the-s1-minimum)), so the S1 row
comes from #860's single points at his geometries in both bases, and no S1
walk was made at cc-pVTZ.

| | config | evaluations | energy (Eh) | max / rms force at the end (Eh/bohr) |
|---|---|---|---|---|
| FC, DZ | [`tier2a_psb3_fc_qdpc.toml`](tier2a_psb3_fc_qdpc.toml) | 16 | −249.0642095767 | 6.0e-6 / 2.9e-6 |
| 13–14CI, DZ | [`tier2a_psb3_1314_qdpc.toml`](tier2a_psb3_1314_qdpc.toml) | 39 | −248.9551975430 | 6.7e-6 / 2.4e-6 |
| 11–12CI, DZ | [`tier2a_psb3_1112_qdpc.toml`](tier2a_psb3_1112_qdpc.toml) | 39 | −248.9480325354 | 4.9e-6 / 2.0e-6 |
| S1, DZ | [`tier3_psb3_s1_ccpvdz_qdpc.toml`](tier3_psb3_s1_ccpvdz_qdpc.toml) | **stopped at 32** | – | – |
| **FC, TZ** | [`tier3_psb3_fc_ccpvtz_qdpc.toml`](tier3_psb3_fc_ccpvtz_qdpc.toml) | **42** | −249.3850724517 | 2.9e-6 / 9.9e-7 |
| **13–14CI, TZ** | [`tier3_psb3_1314_ccpvtz_qdpc.toml`](tier3_psb3_1314_ccpvtz_qdpc.toml) | **44** | −249.2677255170 | 4.8e-6 / 2.4e-6 |
| **11–12CI, TZ** | [`tier3_psb3_1112_ccpvtz_qdpc.toml`](tier3_psb3_1112_ccpvtz_qdpc.toml) | **38** | −249.2716925153 | 6.5e-6 / 2.5e-6 |

Notes on the table:

- **Energies.** A MECI's energy is the mean of states 0 and 1. An FC
  energy is QD state 0.
- **No refusals.** All six converged walks ended
  `completed: 0 errors, 0 warnings`. A PC refusal from
  `qd_pc_intersection` (or from the state-following gradient) would have
  stopped a walk, so PC was refused at no geometry along any of them. Nor
  did #874's denominator refusal or #867's stall appear.
- **Starts.** Each cc-pVTZ walk starts from Park's own cc-pVTZ structure,
  `single_points/geometries/psb3_tz_*.xyz` (his SI, pp. S12–S14, bohr to Å).
  These files match the SI to 5e-11 Å.
- **First evaluations.** Each cc-pVTZ walk's first evaluation reproduces
  #860's single point at that geometry. At FC, E₀ is −249.3848148203
  against −249.3848148202 Eh. At the MECIs, the pair mean is #860's E₀
  plus half its gap.
- **Wall time.** Each cc-pVTZ walk took 16–43 minutes, counting the wait
  for the card.

### The MECIs' trails and gap bounds

| | 13–14CI, DZ | 11–12CI, DZ | **13–14CI, TZ** | **11–12CI, TZ** |
|---|---|---|---|---|
| `heff_gap` at his structure, evaluation 0 (Eh) | 8.6e-6 | 8.6e-6 | 4.18e-5 | 7.28e-5 |
| `heff_gap` at the end (Eh) | 3.83e-11 | 2.89e-10 | **9.60e-11** | **1.60e-10** |
| `gradient_norm` at the end | 8.04e-11 | 5.28e-11 | 4.14e-10 | 1.35e-10 |
| `ci_residual_norm` at the end | 1.94e-10 | 2.15e-10 | 2.08e-10 | 2.05e-10 |
| s_o, s_c at the end | 0.426, 5.95 | 0.279, 33.0 | 0.549, 6.92 | 0.301, 40.4 |
| **gap bound** s_o ‖g‖ + s_c √3 ρ (Eh) | 2.04e-9 | 1.23e-8 | **2.72e-9** | **1.44e-8** |
| its orbital / CI terms (Eh) | 3.4e-11 / 2.0e-9 | 1.5e-11 / 1.2e-8 | 2.3e-10 / 2.5e-9 | 4.1e-11 / 1.4e-8 |
| min `pair_overlap` (evaluation) | 0.9999984 (1) | 0.9999998 (2) | **0.9999917 (2)** | **0.9999999 (8)** |
| max `gradient_norm` / `ci_residual_norm` over the walk | 6.7e-10 / 2.5e-10 | 8.4e-10 / 2.5e-10 | 8.9e-10 / 2.5e-10 | 7.8e-10 / 2.5e-10 |
| third-state margin, min over the walk (Eh) | 0.1235 | 0.1082 | 0.1239 | 0.1090 |
| CASSCF root gap, min over the walk (Eh) | 0.0549 | 0.0495 | 0.0540 | 0.0520 |

- **The bound.** s_o and s_c are `optimize.json`'s
  `gap_sensitivity_orbital` and `gap_sensitivity_ci` (#844), at n = 3. At
  every MECI the bound is at least six times below `max_gap` (1e-7). It is
  dominated by the CI term, and the 11–12CI's s_c is 5.5–5.8× the
  13–14CI's in both bases.
- **Branching space.** It was two-dimensional at every evaluation of all
  four walks.
- **Overlap.** `pair_overlap` never fell below 0.99999 (the floor is 0.9).
- **The cc-pVTZ FC walk** follows QD state 0, with a matched
  `state_overlap` of at least 0.9999988 throughout. Its CASSCF root gap
  stays at or above 0.0251 Eh, and its final H_eff gap to state 1 is
  0.157 Eh.

### The active spaces

Each config names `active` as #860 chose it at the same start geometry
([single_points/README.md](single_points/README.md), "Active spaces"):

| start | `active` (0-based RHF MOs) |
|---|---|
| S1, DZ | 19, 20, 21, 22, 23, 27 |
| FC, TZ | 18, 20, 21, 22, 25, 29 |
| 13–14CI, TZ | 19, 20, 21, 22, 23, 27 |
| 11–12CI, TZ | 19, 20, 21, 22, 23, 30 |

At each converged cc-pVTZ structure, the final `casscf.molden`'s six active
natural orbitals were checked with #860's π-character metric (the
heavy-atom p weight along each atom's local normal). Every one is π:

| structure | π character of the six | natural occupations |
|---|---|---|
| FC, TZ | 1.000 (all six) | 1.959 – 0.051 |
| 13–14CI, TZ | ≥ 0.959 | 1.964 – 0.041 |
| 11–12CI, TZ | ≥ 0.912 | 1.963 – 0.038 |

The cc-pVDZ structures' minima are tier 2a's: 1.000 (FC), 0.965 (13–14CI)
and 0.920 (11–12CI). The S1 row's single points are #860's (≥ 0.91 at
every point).

## Table 2 beside his

Energies are in eV, relative to S0 at the FC structure, as in his Table 2.
Each basis uses its own FC.

- **Ours.** FC, 13–14CI and 11–12CI are our optimized structures. **The S1
  row is a single point at his S1 geometry**, #860's, and not optimized
  (see [below](#the-s1-minimum)). S2 at a MECI is the pair's upper state
  plus `optimize.json`'s third-state margin. At FC it is
  `nevpt2_gradient.json`'s `qd_energies[2]`.
- **His.** Computed from the SI's 10-decimal absolute energies, so a value
  may differ from his printed table in the last digit. His printed values
  are in parentheses.

### cc-pVDZ

| | S0 ours | S1 ours | S2 ours | S0 his | S1 his | S2 his |
|---|---|---|---|---|---|---|
| FC | 0.00 | 4.25 | 5.56 | 0.00 | 4.25 | 5.55 |
| S1 (single point at his geometry) | 0.38 | 4.08 | 5.23 | 0.37 | 4.07 | 5.23 |
| 13–14CI | 2.97 | 2.97 | 6.33 | 2.96 | 2.96 | 6.32 |
| 11–12CI | 3.16 | 3.16 | 6.11 | 3.16 | 3.16 | 6.11 |

### cc-pVTZ

| | S0 ours | S1 ours | S2 ours | S0 his | S1 his | S2 his |
|---|---|---|---|---|---|---|
| FC | 0.00 | 4.27 | 5.66 | 0.00 | 4.24 | 5.61 |
| S1 (single point at his geometry) | 0.24 | 4.11 | 5.24 | 0.20 | 4.07 | 5.20 |
| 13–14CI | 3.19 | 3.19 | 6.60 | 3.15 | 3.15 | 6.53 |
| 11–12CI | 3.09 | 3.09 | 6.08 | 3.08 | 3.08 | 6.05 |

### His three footnotes

He labels footnotes a and b "E(S1) − E(MECI)", but his printed values are
the MECI's energy minus S1's energy at the S1 geometry (−1.11 at cc-pVDZ is
2.96 − 4.07). They are computed here the way his numbers are.

| eV | DZ ours | DZ his | ours − his | TZ ours | TZ his | ours − his |
|---|---|---|---|---|---|---|
| a: E(13–14CI) − E_S1(S1 geometry) | −1.1157 | −1.1150 (−1.11) | −0.7 meV | −0.9188 | −0.9200 (−0.92) | +1.2 meV |
| b: E(11–12CI) − E_S1(S1 geometry) | −0.9207 | −0.9149 (−0.91) | −5.8 meV | −1.0267 | −0.9939 (−0.99) | −32.8 meV |
| c: E(13–14CI) − E(11–12CI) | **−0.1950** | −0.2002 (−0.20) | +5.2 meV | **+0.1079** | +0.0739 (+0.07) | +34.0 meV |

- **cc-pVDZ.** Every entry agrees with his to 0.01 eV, and the footnotes to
  6 meV.
- **cc-pVTZ.** Our entries lie 8–46 meV above his for S0 and S1 (72 meV
  for S2 at the 13–14CI). Footnotes b and c differ by about 33 meV. This is the size of #860's cc-pVTZ core floor
  (spread 40–46 meV), and footnote c's +34.0 meV is the +31.4 meV it adds
  at his geometries plus 2.5 meV from reoptimization. A cc-pVTZ set can
  correlate a 1s shell where cc-pVDZ almost cannot.
- **What is not in this table.** Footnotes a and b mix our optimized MECIs
  with a single point at his S1 geometry. Like the S1 row, they are not a
  fully self-consistent "ours" value.

## Against his absolute energies

His runs freeze the core; ours correlate every inactive orbital
([README.md](README.md), "What cannot match"). So every absolute energy here
lies below his. E_ours − E_his, in mEh, for states 0 / 1 / 2:

| | cc-pVDZ | cc-pVTZ |
|---|---|---|
| FC | −15.099 / −14.860 / −14.779 | −93.791 / −92.530 / −92.110 |
| S1 (single point at his geometry) | −14.794 / −14.832 / −14.834 | −92.106 / −92.296 / −92.318 |
| 13–14CI | −14.850 / −14.861 / −14.533 | −92.248 / −92.255 / −91.139 |
| 11–12CI | −15.045 / −15.048 / −14.900 | −93.495 / −93.510 / −92.572 |

The absolute energies (Eh), ours then his (SI pp. S8–S10 and S12–S14):

| | state 0 | state 1 | state 2 |
|---|---|---|---|
| FC, DZ | −249.0642095767 / −249.0491104000 | −248.9079590478 / −248.8930991441 | −248.8598928546 / −248.8451138449 |
| S1, DZ | −249.0502380331 / −249.0354436267 | −248.9141969621 / −248.8993652796 | −248.8718529572 / −248.8570191947 |
| 13–14CI, DZ | −248.9551975430 / −248.9403470698 | −248.9551975429 / −248.9403365431 | −248.8314059398 / −248.8168734078 |
| 11–12CI, DZ | −248.9480325355 / −248.9329874544 | −248.9480325352 / −248.9329848732 | −248.8395370751 / −248.8246366903 |
| FC, TZ | −249.3850724517 / −249.2912816246 | −249.2280435139 / −249.1355139022 | −249.1771336466 / −249.0850239583 |
| S1, TZ | −249.3762056130 / −249.2840999522 | −249.2339612767 / −249.1416654535 | −249.1924935746 / −249.1001758879 |
| 13–14CI, TZ | −249.2677255171 / −249.1754770559 | −249.2677255170 / −249.1754702166 | −249.1425582841 / −249.0514189025 |
| 11–12CI, TZ | −249.2716925154 / −249.1781978062 | −249.2716925152 / −249.1781821379 | −249.1616130453 / −249.0690413237 |

The offset is about −15 mEh at cc-pVDZ and −93 mEh at cc-pVTZ. Both match
#860's means at his geometries (−14.9 and −92.7 mEh). Within each basis,
over states 0 and 1 of the four structures, the offset varies by 0.31 mEh
(8.3 meV) at DZ and 1.69 mEh (46 meV) at TZ. With state 2 included it
varies by 0.57 and 2.65 mEh. That variation is the floor that enters every
relative number above.

## Geometries

The converged structures are checked in, in Park's atom order (C11…C15, N,
then H):

- cc-pVTZ, this tier:
  [`psb3_fc_ccpvtz.xyz`](geometries/psb3_fc_ccpvtz.xyz),
  [`psb3_1314_meci_ccpvtz.xyz`](geometries/psb3_1314_meci_ccpvtz.xyz) and
  [`psb3_1112_meci_ccpvtz.xyz`](geometries/psb3_1112_meci_ccpvtz.xyz);
- cc-pVDZ, tier 2a's:
  [`psb3_fc_ccpvdz.xyz`](geometries/psb3_fc_ccpvdz.xyz),
  [`psb3_1314_meci_ccpvdz.xyz`](geometries/psb3_1314_meci_ccpvdz.xyz) and
  [`psb3_1112_meci_ccpvdz.xyz`](geometries/psb3_1112_meci_ccpvdz.xyz).

### RMSD against his

The RMSD is Kabsch over all 14 atoms, against his structure at the same
basis:

| Å | FC | 13–14CI | 11–12CI |
|---|---|---|---|
| cc-pVDZ | 0.0020 | 0.0034 | 0.0030 |
| cc-pVTZ | 0.0097 | 0.0094 | 0.0110 |

At cc-pVTZ every backbone bond here is 0.003–0.009 Å shorter than his (see
the next table), which is most of the RMSD. That is the direction core
correlation pulls a bond, and cc-pVTZ is the set in which it can act.

### The Figure-3 bond lengths

His Figure 3 prints the backbone bond lengths of these structures. Its
numbers are in the image only, so his column here is from his SI
coordinates.

| Å, ° | basis | C11=C12 | C12–C13 | C13=C14 | C14–C15 | C15=N | C12–C13–C14–C15 | H–C11–C12–C13 |
|---|---|---|---|---|---|---|---|---|
| FC, ours | DZ | 1.3622 | 1.4400 | 1.3838 | 1.4165 | 1.3176 | 180.0 | 180.0 / 0.0 |
| FC, his | DZ | 1.3633 | 1.4413 | 1.3850 | 1.4177 | 1.3184 | 180.0 | 180.0 / 0.0 |
| FC, ours | TZ | 1.3448 | 1.4221 | 1.3671 | 1.3975 | 1.3063 | 180.0 | 180.0 / 0.0 |
| FC, his | TZ | 1.3499 | 1.4288 | 1.3736 | 1.4037 | 1.3110 | 180.0 | 180.0 / 0.0 |
| S1, his only | DZ | 1.4067 | 1.4007 | 1.5046 | 1.3995 | 1.3558 | 179.2 | 179.9 / 0.1 |
| S1, his only | TZ | 1.3868 | 1.4140 | 1.4451 | 1.4084 | 1.3420 | 180.0 | 180.0 / 0.0 |
| 13–14CI, ours | DZ | 1.3795 | 1.4184 | **1.4641** | 1.4578 | 1.2982 | −109.2 | −172.2 / 4.7 |
| 13–14CI, his | DZ | 1.3796 | 1.4212 | 1.4663 | 1.4573 | 1.2999 | −109.4 | −172.3 / 4.8 |
| 13–14CI, ours | TZ | 1.3534 | 1.4126 | **1.4395** | 1.4472 | 1.2824 | −111.9 | −171.6 / 5.0 |
| 13–14CI, his | TZ | 1.3569 | 1.4209 | 1.4481 | 1.4511 | 1.2879 | −111.9 | −171.6 / 5.1 |
| 11–12CI, ours | DZ | **1.4125** | 1.3682 | 1.4251 | 1.3866 | 1.3461 | 179.9 | 96.6 / −95.4 |
| 11–12CI, his | DZ | 1.4138 | 1.3696 | 1.4262 | 1.3883 | 1.3467 | 180.0 | 96.3 / −95.7 |
| 11–12CI, ours | TZ | **1.4009** | 1.3504 | 1.4092 | 1.3678 | 1.3364 | 179.9 | 96.0 / −95.0 |
| 11–12CI, his | TZ | 1.4068 | 1.3564 | 1.4163 | 1.3741 | 1.3412 | 179.9 | 95.9 / −95.2 |

Against the FC structure at the same basis, the twisted bond is elongated
by:

- **13–14CI:** C13=C14, by 0.080 Å at DZ and 0.072 Å at TZ (his 0.081 and
  0.075). His prose says "~0.07 Å" for this bond.
- **11–12CI:** C11=C12, by 0.050 Å at DZ and 0.056 Å at TZ (his 0.051 and
  0.057). His prose says "~0.05 Å".

## The S1 minimum

**At our level, his planar S1 structure is not a minimum.**
[`tier3_psb3_s1_ccpvdz_qdpc.toml`](tier3_psb3_s1_ccpvdz_qdpc.toml) follows
QD state 1 from his cc-pVDZ S1 geometry, and the walk does not converge:

- **Evaluations 0–19.** The walk sits at his structure:
  - E₁ = −248.91420 Eh, which reproduces #860's single point there,
    −248.9141969621 Eh, at evaluation 0;
  - max |g| is 1e-4 to 7e-4 Eh/bohr;
  - the matched state overlap is ≥ 0.99998;
  - the H_eff gap to state 0 is 0.042 Eh and the CASSCF root gap
    0.022 Eh.
- **Evaluations 20–31.** The gradient grows to 1e-2 to 3e-2 Eh/bohr, and E₁
  falls steadily to −248.94152 Eh by evaluation 31. That is **27 mEh
  (0.74 eV) below** his S1 minimum, and still descending. The overlap dips
  to 0.936 at evaluation 28.
- **Evaluation 32.** The QD state-overlap guard stops the walk:
  `qd_state_overlap: the QD state's overlap with the last geometry's is
  0.743456, below 0.9: it is lost (a crossing), and is not followed onto
  another surface`. The run ends `completed: 1 errors, 0 warnings`.
  Every CASSCF along the way converged (the last at |g| 7.3e-11), and
  there was no PC refusal.

The likely reading is that his structure is a **planar stationary point
that is a saddle along a torsion**. His optimizer stayed on it. An
unconstrained optimizer leaves it once round-off breaks the planar
symmetry, and then descends toward a twisted S1 structure and a crossing
with S0. This tree has no symmetry-constrained optimizer, so the planar
point cannot be held. Two things were not changed to get past this: the
overlap floor, and every other criterion.

A failed `[optimize]` run writes no geometry: no `optimize.json`,
`optimize.xyz` or `final.xyz` (#877). So which torsion opened is not
recorded.

Table 2's S1 row therefore comes from #860's single points at his S1
geometries in both bases. It is labelled as such, and no S1 walk was made
at cc-pVTZ. S1 is not part of the gated sign.

## cc-pVQZ and cc-pV5Z

His Table 2 has two more columns, cc-pVQZ (570 functions) and cc-pV5Z (986),
and his headline PSB3 numbers are cc-pV5Z. Both are out of reach:

- **The f ceiling.** The orbital basis stops at f:
  `kMaxAngularMomentum = 3` in
  [`src/basis/angular_momentum.h`](../../../src/basis/angular_momentum.h).
  cc-pVQZ carries g functions and cc-pV5Z carries g and h, so both need
  g/h orbital kernels in FelizJueves first.
- **Not carried.** Neither set is in `data/basis/`.

His trend from cc-pVTZ to cc-pV5Z continues in the direction measured here:
E(13–14CI) − E(11–12CI) goes +0.07 → +0.17 → +0.21 eV. Only its sign
change, between DZ and TZ, is reproduced in this tier.
