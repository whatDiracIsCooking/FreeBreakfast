# Tier 2a: PSB3's geometries and its 13–14CI

PSB3, the penta-2,4-dieniminium cation (H₂C=CH–CH=CH–CH=NH₂⁺), is the rhodopsin
protonated Schiff base model of Park's Figure 2. Here it is at his level,
SA-QD-PC-NEVPT2 over DF-SA3-CASSCF(6,6)/cc-pVDZ with cc-pVDZ-JKFIT. That is
124 basis functions, his count. The active space is the six π orbitals, and
the average is over the three lowest singlets. This tier converges three
structures (#848):

- the Franck–Condon (FC) geometry, the minimum of QD state 0;
- the **13–14CI**, the MECI of QD states 0 and 1, which Figure 5's loop
  (tier 2b) and tier 3 start from;
- the 11–12CI, which tier 3 needs and which came for free from the same
  source.

Every number is from one card, CUDA (RTX 3080), with `--preset default` at
01146ab ([README.md](README.md), "How the runs are made").

## Numbering

The carbons carry the rhodopsin chromophore's numbers, counted from the
Schiff base: C15 is the iminium carbon, and the terminal CH₂ is C11.
C13=C14 is therefore the central C=C, one bond away from C15, and C11=C12
is the terminal one. Park's SI lists the atoms as C11, C12, C13, C14, C15,
N, then the hydrogens: two on C11, one each on C12 to C15, and two on N.
Every file here taken from his SI or converged from it keeps that order.
The hand-built cross-check files below list N first.

## Where the starts come from

Park's Supporting Information gives every optimized QD-NEVPT2 geometry, in
bohr, with all three states' absolute energies (J. W. Park, JCTC (2020),
acs.jctc.9b00919, Supporting Information, section C, pp. S8–S11). His three
cc-pVDZ structures are converted to Å (× 0.529177210903) and checked in as
the starts. Each file's comment line carries his three energies.

| file | his structure |
|---|---|
| [`geometries/psb3_fc_start.xyz`](geometries/psb3_fc_start.xyz) | S0 |
| [`geometries/psb3_1314_start.xyz`](geometries/psb3_1314_start.xyz) | 13–14 CI |
| [`geometries/psb3_1112_start.xyz`](geometries/psb3_1112_start.xyz) | 11–12 CI |

Before the SI turned up, the starts were hand-built. They are kept as a
cross-check:

- [`psb3_fc_idealized.xyz`](geometries/psb3_fc_idealized.xyz) is planar
  all-trans PSB3 built from textbook values: C=N 1.30, C–C 1.41/1.42 and
  C=C 1.38/1.36 Å, C–H 1.08, N–H 1.01 Å, and every angle 120°.
- [`tier2a_psb3_fc_casscf.toml`](tier2a_psb3_fc_casscf.toml) optimizes it to
  root 0 of the SA3-CASSCF: 32 evaluations, at Gaussian's default
  thresholds, giving [`psb3_fc_casscf.xyz`](geometries/psb3_fc_casscf.xyz).
  This was run at e424b1c, before this tier rebased onto 01146ab.
- [`psb3_1314_handbuilt_start.xyz`](geometries/psb3_1314_handbuilt_start.xyz)
  is that minimum with C13=C14 twisted 90° and set to 1.45 Å.
- From it, [`tier2a_psb3_1314_casscf.toml`](tier2a_psb3_1314_casscf.toml)
  finds the bare-CASSCF 13–14 MECI,
  [`psb3_1314_casscf_meci.xyz`](geometries/psb3_1314_casscf_meci.xyz):
  53 evaluations, gap 6.3e-9 Eh, mean −248.160024936719 Eh.
  - That run was made by #858's session (PR #872, the fix that let it
    converge), on CUDA, with exactly this config and start file. The start
    file was then named `psb3_1314_start.xyz`, and this tier did not rerun it.
  - Its first attempt, before #858, stopped at evaluation 7: the
    intersection's Z-vector had a true residual of 1.19e-10, just above its
    1e-10 bar.

## The active space

The six active orbitals are picked by index, once, at each start
(`../../../src/pipeline/optimize/README.md`, "The active space is picked
once"). The frontier orbitals around the RHF's HOMO are not all π here. To
find the π ones, each RHF orbital's π character was measured: the weight of
its heavy-atom p coefficients along each atom's local plane normal, the same
metric #860 uses. The six orbitals with π ≈ 1 were taken:

| start | `active` (0-based RHF MOs) |
|---|---|
| S0 (`tier2a_psb3_fc_qdpc.toml`) | 18, 20, 21, 22, 23, 28 |
| 13–14 CI (`tier2a_psb3_1314_qdpc.toml`) | 19–24, the frontier default, so no key |
| 11–12 CI (`tier2a_psb3_1112_qdpc.toml`) | 19, 20, 21, 22, 23, 29 |

These are the same sets #860 chose independently at the same three
geometries. The hand-built ones differ: [17, 20, 21, 22, 24, 27] at the
idealized start, and the frontier at the hand-built twist.

At each converged structure, the outer `[casscf]` makes its own fresh pick
from these indices. Its six active natural orbitals there are all π:

| structure | minimum π character | natural occupations |
|---|---|---|
| FC | 1.000 | 1.957 – 0.056 |
| 13–14CI | 0.965 | 1.962 – 0.045 |
| 11–12CI | 0.920 | 1.960 – 0.041 |

## The runs

All three QD-PC configs use the same settings:

- `[casscf] conv_tol_grad = 1e-9`, not tier 0's 1e-10. At 1e-10,
  `mcscf::solve` stalls at |g| ≈ 3e-10 on PSB3 and the walk fails midway
  (#867).
- Opt=Tight's four force and displacement criteria.
- The MECIs add `max_gap = 1e-7` Eh.

A PC refusal by `qd_pc_intersection` would stop a walk with the step's
error. None did: every walk ended `completed: 0 errors, 0 warnings`, so PC
was refused at no geometry along any of them. The (6e,6o) refusal seen
before #853 (PR #868), `Sr's metric has no gap at its cut`, is gone.

| | FC | **13–14CI** | 11–12CI |
|---|---|---|---|
| config | [`tier2a_psb3_fc_qdpc.toml`](tier2a_psb3_fc_qdpc.toml) | [`tier2a_psb3_1314_qdpc.toml`](tier2a_psb3_1314_qdpc.toml) | [`tier2a_psb3_1112_qdpc.toml`](tier2a_psb3_1112_qdpc.toml) |
| evaluations | 16 | **39** | 39 |
| energy (Eh) | −249.0642095767 | **−248.9551975430** (pair mean) | −248.9480325354 (pair mean) |
| `heff_gap` at the end (Eh) | 0.156 (to state 1) | **3.83e-11** | 2.89e-10 |
| `gradient_norm` at the end | – | **8.04e-11** | 5.28e-11 |
| `ci_residual_norm` at the end | – | **1.94e-10** | 2.15e-10 |
| gap bound s_o ‖g‖ + s_c √3 ρ (Eh) | – | **2.04e-9** | 1.23e-8 |
| s_o, s_c at the end | – | 0.426, 5.95 | 0.279, 33.0 |
| min `pair_overlap` (evaluation) | – | **0.9999984** (1) | 0.9999998 (2) |
| max / rms force at the end (Eh/bohr) | 6.0e-6 / 2.9e-6 | 6.7e-6 / 2.4e-6 | 4.9e-6 / 2.0e-6 |
| third-state margin, min over the walk (Eh) | – | 0.1235 | 0.1082 |
| CASSCF root gap, min over the walk (Eh) | 0.0228 | 0.0549 | 0.0495 |

Notes on the table:

- **The FC walk** follows QD state 0, with a matched overlap of 1.00000000
  throughout.
- **The 13–14CI trail.**
  - The gap is already 8.6e-6 Eh at Park's geometry (evaluation 0).
  - It is 6.8e-9 by evaluation 5. The mean then creeps down the seam by
    another 1.3e-6 Eh, and the walk converges at evaluation 38.
  - Over the whole trail, `gradient_norm` ≤ 6.7e-10, `ci_residual_norm`
    ≤ 2.5e-10, `pair_overlap` ≥ 0.9999984, and the branching space is
    two-dimensional at every evaluation.
- **The gap bound** is the last evaluation's s_o ‖g‖ + s_c √n ρ
  (`gap_sensitivity_orbital`, `gap_sensitivity_ci`, n = 3), the bound #844
  put into the output. At both MECIs it sits well under `max_gap` and is
  dominated by the CI term, s_c √3 ρ. Tier 2b's gate is built from these
  numbers.

The converged structures are checked in:

- [`psb3_fc_ccpvdz.xyz`](geometries/psb3_fc_ccpvdz.xyz) (FC);
- [`psb3_1314_meci_ccpvdz.xyz`](geometries/psb3_1314_meci_ccpvdz.xyz)
  (13–14CI);
- [`psb3_1112_meci_ccpvdz.xyz`](geometries/psb3_1112_meci_ccpvdz.xyz)
  (11–12CI).

## It is the 13–14CI

| Å, ° | FC | 13–14CI | 11–12CI |
|---|---|---|---|
| C11=C12 | 1.3622 | 1.3795 | **1.4125** |
| C12–C13 | 1.4400 | 1.4184 | 1.3682 |
| C13=C14 | 1.3838 | **1.4641** | 1.4251 |
| C14–C15 | 1.4165 | 1.4578 | 1.3866 |
| C15=N | 1.3176 | 1.2982 | 1.3461 |
| C12–C13–C14–C15 | 180.0 | **−109.2** | 179.9 |
| H–C11–C12–C13 | 180.0 / 0.0 | −172.2 / 4.7 | **96.6 / −95.4** |

- **The 13–14CI.** C13=C14 is elongated by **0.080 Å** against the FC
  structure's, and the torsion that broke is C12–C13–C14–C15, from 180° to
  −109°. The terminal CH₂ stays within 8° of the plane.
- **The 11–12CI.** C11=C12 is elongated by 0.050 Å, and the CH₂ is twisted
  to ±96°.
- **Against Park's own structures** the elongations agree: 0.081 and
  0.051 Å from his SI geometries. His prose says "~0.07" for the 13–14
  bond, so the 0.080 here differs from his text, not from his coordinates.
- **By RMSD** (Kabsch, all 14 atoms) the converged structures sit 0.0034 Å
  (13–14CI), 0.0030 Å (11–12CI) and 0.0020 Å (FC) from his. The 13–14CI is
  0.77 Å from his 11–12 CI.

## Against Park's energies

His runs freeze the core; ours correlate every inactive orbital
([README.md](README.md), "What cannot match"). So each absolute energy here
lies below his. The offset is the same to 0.24 mEh at all three
structures, so it cancels in every energy difference:

| Eh | ours | his (SI) | ours − his |
|---|---|---|---|
| FC, state 0 | −249.0642095767 | −249.0491104000 | −15.099 mEh |
| 13–14CI, pair mean | −248.9551975430 | −248.9403418065 | −14.856 mEh |
| 11–12CI, pair mean | −248.9480325354 | −248.9329861638 | −15.046 mEh |

The offsets match #860's single points at his own geometries
([single_points/README.md](single_points/README.md), mean −14.94 mEh),
which is expected: the converged structures are within 0.004 Å of his. The
walks' first evaluations, made at his geometries, reproduce #860's E₀ to
1e-10 Eh at S0 (−249.0641989341) and at the 13–14 CI (−248.9551950687).

| eV | ours | his |
|---|---|---|
| E(13–14CI) − E(FC) | 2.966 | 2.960 |
| E(11–12CI) − E(FC) | 3.161 | 3.160 |
| E(13–14CI) − E(11–12CI) | **−0.195** | −0.200 |

The last row is tier 3's cc-pVDZ sign, and it is negative, as his is.
Park's own MECIs are converged to gaps of 1.05e-5 Eh (13–14) and 2.6e-6 Eh
(11–12). At our level his 13–14 CI starts at a gap of 8.6e-6 Eh.

The bare-CASSCF 13–14 MECI from the hand-built start is a different
structure, 0.32 Å from his. It twists the same bond (C12–C13–C14–C15 at
−89.5°) but with the bond alternation of the chain shifted:

- C14–C15 is 1.330 Å and C15=N is 1.370 Å;
- C13=C14 is 1.433 Å, an elongation of 0.063 Å against the CASSCF minimum.

That is CASSCF's own seam, not the QD-PC-NEVPT2 one. It confirms the basin
and does not stand in for the MECI.
