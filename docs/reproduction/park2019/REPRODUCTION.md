# What of Park's QD-NEVPT2 reproduced, and what could not

#851. This page is the summary of milestone 26: the reproduction of J. W. Park,
*Analytical Gradient Theory for Quasidegenerate N-Electron Valence State
Perturbation Theory (QD-NEVPT2)*, *J. Chem. Theory Comput.* (2020),
[DOI 10.1021/acs.jctc.9b00919](https://doi.org/10.1021/acs.jctc.9b00919),
together with his Supporting Information (cited as "SI p. S*n*"). It
answers one question: which of his results this tree reproduces outright,
which it cannot, and why. The tables are not repeated here. Each section
gives the result and links the tier document that has the tables, configs
and geometries.

**In short.**
- **The method is his.** With the C and N 1s frozen, our single points at
  his own SI geometries give his absolute QD-PC-NEVPT2 energies to within
  0.52 µEh, at all 18 points and every state.
- **Every gated quantity passed.** These are tier 0's walk, tier 1's two
  basis shifts, tier 2a's structure, tier 2b's amplitude and its bound, and
  tier 3's sign flip.
- **What could not be done:** his cc-pVQZ and cc-pV5Z columns, the methods
  he compares against, his timings and pHBI. Also not reproduced as
  optimizations: three of his stationary points. His PSB3 S1 "minimum",
  his ethylene Hm and his ethylene C3v are each a finding here, not a
  failure.
- **What the work fixed.** It found and fixed six defects in the tree
  (#853, #856, #858, #864, #865 and #874). Nine issues are still open.

## Contents

- [The method identification](#the-method-identification)
- [How the numbers were made](#how-the-numbers-were-made)
- [The single-point headline](#the-single-point-headline)
- [What reproduced, tier by tier](#what-reproduced-tier-by-tier)
- [Convergence and refusals, as observed](#convergence-and-refusals-as-observed)
- [What had to be rerun, and what was not](#what-had-to-be-rerun-and-what-was-not)
- [What could not be reproduced, and why](#what-could-not-be-reproduced-and-why)
- [Where the tree is ahead](#where-the-tree-is-ahead)
- [The diagnostic he did not have](#the-diagnostic-he-did-not-have)
- [What the reproduction hardened](#what-the-reproduction-hardened)
- [Where the record disagrees with itself](#where-the-record-disagrees-with-itself)

## The method identification

His "QD-NEVPT2" is **SA-QD-PC-NEVPT2**:

- **PC, state-averaged Fock.** Paper p. 11: "we consider only the
  SA-QD-PC-NEVPT2 theory … Hereafter, the SA-QD-PC-NEVPT2 theory is
  referred to as QD-NEVPT2 for brevity". Paper p. 10: "In the current work,
  we exclusively use the SA-NEVPT2 scheme", which is Dyall's H over the
  *state-averaged* generalized Fock, with a state-specific projection
  manifold (eq. 14). SI p. S1: "In all calculations, the partially
  contracted (PC) variant with state-averaged orbitals are used."
- **Symmetrization.** H_eff is symmetrized as ½ (H_MN + H_NM) (eq. 17).
  The state energy is E_P = Σ_MN R_PPMN H_MN, with
  R_PQMN = ½ (R_MP R_NQ + R_MQ R_NP) (eqs. 19–20).
- **The gradient.** It comes from a Lagrangian with a Z-vector, density
  fitted over JKFIT. The aug JKFIT sets carry tempering functions from
  MOLPRO's library (SI p. S1).
- **Frozen core.** "The frozen core approximation is invoked for all the
  calculations" (p. 18).

The same theory in this tree is `[nevpt2_gradient] quasi_degenerate = true,
variant = "pc"`, or `[nevpt2] quasi_degenerate = true` for an energy:

- **Shared orbitals.** QD is built on the CASSCF's own orbitals, core and
  virtuals canonical over the **state-averaged** Fock: "Not the per-root
  Fock the state-specific results use"
  ([`src/pipeline/nevpt2/README.md`](../../../src/pipeline/nevpt2/README.md),
  step 1 of the QD section).
- **Symmetrization.** **H_IJ = E_I δ_IJ + ½ (C_IJ + C_JI)** (the same
  README, step 4).
- **The gradient.** `nevpt2::qd_pc_gradient`'s W form takes any symmetric
  W ([`src/nevpt2/gradient/README.md`](../../../src/nevpt2/gradient/README.md),
  "PC at any symmetric W"). A state is W = u_K u_Kᵀ, and the interstate
  W^PQ = ½ (u_P u_Qᵀ + u_Q u_Pᵀ) is his R_PQMN of eq. 20
  ([`docs/nevpt2/nevpt2.tex`](../../nevpt2/nevpt2.tex)).
- **The branching space.** `nevpt2::qd_pc_intersection` builds g_P, g_Q and
  h_PQ from one set of H_eff eigenvectors
  ([`src/nevpt2/intersection/README.md`](../../../src/nevpt2/intersection/README.md)).
- **Frozen core**, since #859: `[nevpt2] frozen_core = "auto"`, the 1s of
  each C and N, for energies only.

The decisive evidence is numerical. Frozen-core single points at his 18 SI
structures reproduce his absolute energies to within 0.52 µEh, with nothing
fitted ([the next section but one](#the-single-point-headline)). The two
differ in implementation (BAGEL with an in-house gradient against this
tree), in hardware, and in the aux sets, which were rebuilt here (#845).
Agreement at that level fixes the contraction (PC), the Fock (state
averaged), the symmetrization, the frozen-core count (C/N 1s, BAGEL's
default, an inference #860 tested), and the aux sets with their tempering
functions.

## How the numbers were made

- **One card.** Every number in this directory comes from one card: CUDA,
  RTX 3080 (10 GiB), `--preset default`. Each run held the whole card
  (`with_gpu_card cuda`). [README.md](README.md), "How the runs are made",
  gives the command and the reason (rows of one table from two backends
  would put a cross-backend difference inside the gated quantity).
- **No code changed here.** No C++ changed for any of the tier documents,
  and none for this one. No ctest was run for this page.
- **The build behind each tier.** The tiers ran on different builds, so
  they include different fixes:

| results | build | includes |
|---|---|---|
| tier 0 ([README.md](README.md)) | 79b2bc7 | none of the milestone's fixes |
| #860 phase 1 ([single_points/README.md](single_points/README.md)) | 4f84cfc | #844, #845, #853, #856 |
| tier 2a ([tier2a.md](tier2a.md)) | 01146ab | the same, and #858/#864 |
| tier 3 ([tier3.md](tier3.md)) | 728da1a | the same as tier 2a |
| tier 2b ([tier2b.md](tier2b.md)) | 446cf30, rerun at 97207e7 | tier 2a's; the rerun also has #874 |
| #860 phase 2 ([single_points/frozen_core.md](single_points/frozen_core.md)) | #859's branch, 8516b9f and the change | the same, and #874 |
| tier 1 ([tier1.md](tier1.md)), ethylene starts ([ethylene_starts.md](ethylene_starts.md)) | e1868b8 (and a0e7b9c for the Hm walks) | every fix, #859 and #865 included (a0e7b9c lacks only #865) |

[Below](#what-had-to-be-rerun-and-what-was-not) says why the older builds'
results still stand.

- **Traceability.** Each number below can be traced to a checked-in
  config and geometry under this directory. Three sets of files were
  checked in by #851, after their runs, from the recipes the tier
  documents wrote down. They reproduce the recorded runs and were not
  rerun:
  - **#860 phase 2's 18 frozen-core single points:**
    [`single_points/<stem>_fc.toml`](single_points/), each its
    all-electron twin `<stem>.toml` with `frozen_core = "auto"` appended
    ([frozen_core.md](single_points/frozen_core.md), "What was run").
  - **Tier 1's 12 single points:**
    [`tier1_ethylene_{s0,py,et}_aug{dz,tz}_sp.toml`](.) (all-electron) and
    `…_sp_fc.toml` (frozen core). Each is the optimization config with
    `[nevpt2_gradient]` and `[optimize]` replaced by `[nevpt2]
    quasi_degenerate = true`, and `xyz` pointed at the converged structure
    ([tier1.md](tier1.md), "The runs").
  - **Tier 0's MECI:**
    [`geometries/ethylene_tier0_pc_meci.xyz`](geometries/ethylene_tier0_pc_meci.xyz),
    the `final.xyz` that [README.md](README.md) quotes.

  Run outputs (logs, `optimize.json`, Molden files) are not in git. The
  tier documents quote them.

## The single-point headline

**With the C and N 1s frozen, our QD-PC-NEVPT2 energies at Park's own SI
geometries equal his absolute energies to within 0.52 µEh, at all 18 points
and all three states.**

- **The points.** Ethylene S0, Py, Et, C3v and Hm at aug-cc-pVDZ and
  aug-cc-pVTZ (SI pp. S2–S7), and PSB3 S0, S1, 11–12 CI and 13–14 CI at
  cc-pVDZ and cc-pVTZ (SI pp. S8–S15).
- **Who ran them.** #860 phase 2 ran 17 of them. The 18th, C3v at
  aug-cc-pVDZ, was run by #847 once #865 had landed.
- **The largest miss** is 0.52 µEh (PSB3 cc-pVTZ 11–12 CI, state 1). The
  largest spread over the geometries of one state is 0.014 meV.
- **The tiers' own quantities at his geometries:**
  - Tier 3's E(13–14 CI) − E(11–12 CI) is −0.20015 eV at cc-pVDZ (his
    −0.20016) and +0.07392 eV at cc-pVTZ (his +0.07392).
  - Tier 1's Py and Et shifts are +0.08187 and +0.03791 eV (his +0.08187
    and +0.03790).
- **All-electron, for contrast.** Phase 1 put E_ours − E_Park at −5.4 mEh
  (ethylene aug-cc-pVDZ), −35.1 (aug-cc-pVTZ), −14.9 (PSB3 cc-pVDZ) and
  −92.7 to −92.9 mEh (cc-pVTZ). Its spread over the geometries was 3.7 to
  46 meV.

Source: [single_points/frozen_core.md](single_points/frozen_core.md); the
configs and geometries are in [`single_points/`](single_points/README.md).

This is the most direct reproduction available. No optimization of ours is
involved. It confirms two inferences:

- **The frozen-core count.** It is the C/N 1s, which the paper does not
  state.
- **The aux sets.** #845's aug-cc-pV{D,T}Z-JKFIT, with MOLPRO's tempering
  functions, are the ones he used.

It also means the old "frozen-core floor" no longer limits any comparison
that can be made at his geometries.

## What reproduced, tier by tier

| tier | gated quantity | bar | measured | verdict |
|---|---|---|---|---|
| 0 | PC-QD not refused along a MECI walk (ethylene, (2e,2o), cc-pVDZ) | the walk converges; a PC refusal would stop it | 83 evaluations, `completed: 0 errors, 0 warnings` | **pass** |
| 1 | aug-cc-pVDZ → aug-cc-pVTZ shift of ΔE(MECI − S0), Py, frozen core | within 0.02 eV of his +0.08 | +0.0820 eV (ours − his +0.002) | **pass** |
| 1 | the same, Et | within 0.02 eV of his +0.04 | +0.0379 eV (ours − his −0.002) | **pass** |
| 1 | C3v and Hm shifts | reported, not gated (rulings on #847) | +0.0147 and +0.0190 eV at his geometries (his SI +0.0162, +0.0205) | – |
| 2a | the converged PSB3 MECI is the 13–14CI | C13=C14 elongated by about 0.07 Å against FC, and the C12–C13–C14–C15 torsion broken | +0.080 Å; torsion 180° → −109.2° | **pass** |
| 2b | the convergence bound on the Figure-5 gap | below 20% of the oscillation amplitude | 2.56e-9 Eh against 1.11e-5 Eh | **pass** |
| 2b | the oscillation amplitude | below 0.02 eV | 0.0015 eV (half peak-to-peak), 0.0030 eV (peak-to-peak) | **pass** |
| 3 | sign of E(13–14CI) − E(11–12CI), cc-pVDZ | negative | −0.1950 eV | **pass** |
| 3 | the same, cc-pVTZ | positive, with \|value\| ≥ 0.03 eV (otherwise inconclusive) | +0.1079 eV | **pass** |

### Tier 0: PC-QD survives a MECI walk

[README.md](README.md), "Tier 0" (#843).
[`tier0_ethylene_pc.toml`](tier0_ethylene_pc.toml) runs ethylene's MECI at
QD-PC-NEVPT2 over DF-SA3-CASSCF(2,2)/cc-pVDZ.

- **The result.** It converged in 83 evaluations with no refusal, to a gap
  of 8.07e-8 Eh.
- **Against SC.** The PC MECI is 0.79 meV below the SC one from the same
  start.
- **Why convergence is the gate.** `qd_pc_intersection` has no SC to fall
  back on, so a refusal would have stopped the walk.

### Tier 1: ethylene's Table 1

[tier1.md](tier1.md) (#847). Six walks were optimized all-electron: S0, Py
and Et at both bases. They start from his SI geometries.

- **The gated shift** is computed from frozen-core single points at our
  converged structures: **Py +0.0820 eV and Et +0.0379 eV**, against his
  +0.08 and +0.04. Both pass the 0.02 eV bar. His SI's 10-decimal energies
  give +0.08187 and +0.03790. Ours differ from those by +0.10 and +0.02 meV.
- **Why frozen core.** The all-electron shifts are larger, +0.0977 and
  +0.0490 eV. #860 found that the all-electron core floor at aug-cc-pVTZ is
  19–22 meV, the size of the bar. So the bar is not discriminating
  without frozen core (#860's verdict comment).
- **Why our structures are good enough.** Frozen-core gradients do not
  exist (#875), so the energies are taken at the all-electron structures.
  The error that brings is second order in the displacement (S0 is a
  minimum, and a MECI's mean is stationary along the seam). Measured, it is
  ≤ 1.7 meV per structure, and +0.10 meV (Py) and +0.02 meV (Et) on the
  shift ([tier1.md](tier1.md), "Why all-electron geometries are good
  enough here").
- **C3v and Hm are reported, not gated.** They are frozen-core single
  points at his geometries, referenced to our S0. Referenced to his S0,
  they are his values to ≤ 0.01 meV.

### Tier 2a: PSB3 and the 13–14CI

[tier2a.md](tier2a.md) (#848). This is SA3-CASSCF(6,6)/cc-pVDZ with 124
basis functions, his count. Each walk starts from his SI structure.

- **The walks.** The 13–14CI converged in 39 evaluations to a gap of
  3.83e-11 Eh. The FC and the 11–12CI walks converged too, with no refusal
  on any of them.
- **The structure.** C13=C14 is **0.080 Å** longer than at FC, and the
  torsion that broke is C12–C13–C14–C15 (180° → −109.2°). His own
  coordinates give 0.081 Å. His prose says "~0.07".
- **Against his structures.** The RMSD to his is 0.0034 Å (13–14CI),
  0.0030 Å (11–12CI) and 0.0020 Å (FC).
- **Against his energies.** E(13–14CI) − E(FC) is 2.966 eV (his 2.960).

### Tier 2b: Figure 5, artifact-free with a bound

[tier2b.md](tier2b.md) (#849). The scan is a 24-point loop of radius
0.002 Å in the branching plane of **our** 13–14CI.

- **The gap.** Its maximum is **0.0123 eV** (his ~0.014). It oscillates
  by **0.0015 eV** half peak-to-peak, or 0.0030 eV peak-to-peak (his
  ~0.005; the paper does not say which reading).
- **The bound,** s_o ‖g‖ + s_c √3 ρ at the worst point, is 2.56e-9 Eh
  (7.0e-8 eV). That is 2.3e-4 of the 20% bar. Both gates pass, each on the
  reading that is harder to pass.
- **No refusal anywhere.** `pc_qd_refusal` is a real gate here: in the
  `[nevpt2]` step, SC would otherwise stand in silently. None of the 24
  logs contains "refused".
- **The curve is the cone's.** A linear two-state model built from the
  MECI's own g and h predicts 12.35–9.29 meV, and the scan measures
  12.34–9.32 meV.
- **Our oscillation is a third of his.** That is a property of the cone's
  shape: our MECI is 0.0034 Å from his, and our correlation is
  all-electron. It is not an artifact. An artifact would show as a cusp or
  a collapse of the gap, and none appears.

### Tier 3: the sign flip

[tier3.md](tier3.md) (#850). This is Table 2's cc-pVDZ and cc-pVTZ columns
(124 and 292 basis functions).

- **The sign flips.** E(13–14CI) − E(11–12CI) is **−0.1950 eV at cc-pVDZ**
  (his −0.2002) and **+0.1079 eV at cc-pVTZ** (his +0.0739). The cc-pVTZ
  value is outside the 0.03 eV dead band, so both halves of the gate are
  conclusive.
- **The core floor in the cc-pVTZ value.** It is all-electron, so it
  carries the core floor. #860 measured that contribution at his geometries
  as +31.4 meV, and reoptimization adds +2.5 meV. With the core frozen, at
  his geometries, the value is his to 0.00 meV
  ([the headline](#the-single-point-headline)). The floor changes the size
  of the cc-pVTZ value, not its sign.

## Convergence and refusals, as observed

These are the values at the end of each walk, read from the logs and
`optimize.json`, not the configs' settings.

**What the configs ask for.** Every walk asks for Opt=Tight's four criteria:
max/rms force 1.5e-5/1e-5 Eh/bohr and max/rms displacement 6e-5/4e-5 bohr.
Every MECI walk adds `max_gap` = 1e-7 Eh. The CASSCF's `conv_tol_grad` is
1e-10 for ethylene and **1e-9 for PSB3**. When these runs were made, 1e-10
stalled `mcscf::solve` at |g| ≈ 3e-10 on PSB3 and failed the walk (the
ruling on #848 and #850). #867 has since fixed that (PR #896: the CI
solver's tolerance was relative to |E| ≈ 248 Eh, and it now tracks
`conv_tol_grad`), and PSB3 cc-pVDZ now converges at 1e-10. The recorded PSB3
runs were **not** rerun at 1e-10. Their gap bounds below are stated at 1e-9.
The one gated bound, tier 2b's, is 2.56e-9 Eh against a 1.11e-5 Eh bar.
The MECI bounds elsewhere (up to 1.44e-8 Eh at tier 3's cc-pVTZ) are
reported, not gated.

| walk | evaluations | final gap (Eh) | max force (Eh/bohr) | CASSCF ‖g‖ / ρ | gap bound (Eh) | refusal status |
|---|---|---|---|---|---|---|
| tier 0, ethylene PC MECI | 83 | 8.07e-8 | 5.67e-7 | 2.96e-11 / 4.61e-14 | not formed (s_o, s_c predate #844) | none; 0 errors, 0 warnings |
| tier 1, six walks | 7–25 | 2.98e-11 to 5.24e-9 (MECIs) | ≤ 9.2e-6 | over the trails ≤ 8.5e-11 / ≤ 1.1e-13 | ≤ 1.9e-12 | none; all 0 errors, 0 warnings |
| tier 1, C3v aug-cc-pVDZ | stopped at evaluation 10 | 6.6e-10 | – | ≤ 4e-12 | – | stopped by `mcscf::root_pair_multipliers`, CASSCF roots 7.873e-9 Eh apart (#892) |
| tier 2a, 13–14CI / 11–12CI | 39 / 39 | 3.83e-11 / 2.89e-10 | 6.7e-6 / 4.9e-6 | 8.04e-11 / 1.94e-10; 5.28e-11 / 2.15e-10 | 2.04e-9 / 1.23e-8 | none |
| tier 2a, FC | 16 | – | 6.0e-6 | – | – | none |
| tier 2b, 24 points | `max_steps = 0` | 3.43e-4 to 4.53e-4 (the scan) | – | worst 9.34e-11 / 2.45e-10 | 2.56e-9 (2.39e-9 on the #874 rerun) | none; no "refused" in any log |
| tier 3, cc-pVTZ 13–14CI / 11–12CI | 44 / 38 | 9.60e-11 / 1.60e-10 | 4.8e-6 / 6.5e-6 | 4.14e-10 / 2.08e-10; 1.35e-10 / 2.05e-10 | 2.72e-9 / 1.44e-8 | none |
| tier 3, cc-pVTZ FC | 42 | – | 2.9e-6 | – | – | none |
| tier 3, cc-pVDZ S1 | stopped at evaluation 32 | – | – | last CASSCF 7.3e-11 | – | stopped by the QD state-overlap guard (overlap 0.743456); `completed: 1 errors`; no PC refusal |

Single points:

- **#860 phase 1** (all-electron, 4f84cfc). PC-QD was refused at ethylene
  S0 in both bases: `state 2: Sr has a denominator λ + Δ = -1.223e-01`
  (aug-cc-pVDZ) and `-1.388e-01` (aug-cc-pVTZ). That was #874. The SC-QD
  value stood in, with `completed: 0 errors, 2 warnings`. The other 15
  points completed with no refusal. C3v at aug-cc-pVDZ was blocked by
  #865's RHF stall.
- **#860 phase 2** (frozen core). All 17 logs end `completed: 0 errors,
  0 warnings`, with no refusal. So does the 18th point, run by #847.
- **Tier 1's 12 single points** all completed with no PC-QD refusal.

Every final gap is below `max_gap`, and every bound is at least six times
below it. The bounds are stated in full in
[the diagnostic section](#the-diagnostic-he-did-not-have).

## What had to be rerun, and what was not

Rerun, or rerun after a fix:

- **Tier 2b, after #874.** The tables are from a build at 446cf30. #874
  (8516b9f) then replaced PC's rule against non-positive denominators with
  one against poles, and all 24 points were rerun at 97207e7. Every one
  completed with no refusal, and every gap agrees with the table to
  5.2e-11 Eh. The bound from the rerun's worst ‖g‖ and ρ is 2.39e-9 Eh
  ([tier2b.md](tier2b.md), last section).
- **Ethylene S0's single points, after #874.** Phase 1 had them refused.
  After #874, the all-electron PC-QD Δ was measured as −5.515 / −5.505 /
  −5.497 mEh (aug-cc-pVDZ), inside the MECIs' band
  ([`src/nevpt2/energy/README.md`](../../../src/nevpt2/energy/README.md),
  "Poles, not signs"). Phase 2 then ran them frozen-core.
- **C3v at aug-cc-pVDZ, after #865.** This single point was run by #847
  once #865 had landed.
- **#846's C3v CASSCF walk** was refused by `df_intersection` until #856.
  After that, its final fresh RHF exited 6 until #865.
- **Tier 2a's CASSCF 13–14 MECI** (a basin check, not a gated number).
  Before #858 it stopped at evaluation 7, on a Z-vector residual of
  1.19e-10 against a 1e-10 bar. #858's session reran it after the fix
  (PR #872): 53 evaluations.
- **#846's Py and Et walks** were run again at a0e7b9c. The evaluation
  counts, energies, gaps and overlaps were identical.

Not rerun, and why the results stand:

- **Tier 0** ran at 79b2bc7, before every fix in this milestone, #853's
  and #874's changes to PC's refusal rules included.
  - **Reproduced later.** #846's QD-PC walk from Park's Hm geometry, at
    a0e7b9c, which has #853, #858/#864 and #874, converged onto tier 0's
    MECI: 0.0000 Å apart, and 1.7e-8 Eh ([ethylene_starts.md](ethylene_starts.md)).
  - **What is missing.** Tier 0's gap bound was never formed. The run
    predates #844, which put s_o and s_c into `optimize.json`.
- **Tiers 2a and 3** ran before #874 (01146ab and 728da1a). #874 changes
  only which modes PC refuses, not the energy of a mode it keeps. The
  energy README measured the largest amplitude |t| at Park's four cc-pVDZ
  PSB3 SI geometries as 0.04–0.08, against the new refusal threshold of 1.
  #874's PR (#883) reports the same list: its largest |t| over Park's
  geometries, 0.274, is ethylene's aug-cc-pVDZ S0. Neither covers PSB3 at
  cc-pVTZ, so no margin is recorded for tier 3's cc-pVTZ walks. Tier 2b's rerun around the 13–14CI
  moved no gap by more than 5.2e-11 Eh. This is an argument, not a rerun:
  the tier 2a and tier 3 walks themselves were not repeated on a post-#874
  build.

## What could not be reproduced, and why

| item | why | where |
|---|---|---|
| Table 2's cc-pVQZ and cc-pV5Z columns, and his headline cc-pV5Z PSB3 numbers | Neither set is in `data/basis/`. cc-pVQZ needs g and cc-pV5Z needs g and h, above the orbital ceiling `kMaxAngularMomentum = 3` (f). Both need g/h orbital kernels in FelizJueves first. His trend continues past cc-pVTZ (+0.07 → +0.17 → +0.21 eV); only its DZ → TZ sign change is reproduced. | [`src/basis/angular_momentum.h`](../../../src/basis/angular_momentum.h); [tier3.md](tier3.md), "cc-pVQZ and cc-pV5Z" |
| Figure 6's comparators (SS-QD-NEVPT2 with state-specific Dyall, uniform-FOIS QD-NEVPT2, MS-CASPT2, XMS-CASPT2) | None of these methods exists in this tree. Only his own scheme's panel is reachable. | the milestone description, "Out of scope" |
| Table 5's timings | His are 18 Xeon cores with MPI, and ours are one GPU, so they are not comparable. A per-stage breakdown of our own would be the meaningful version; it was not made. | the same |
| pHBI (Tier 4) | 518 basis functions, (12e,11o) = 213,444 determinants, at three geometries. Reachable on paper, multi-day on a 10 GB card; it wants a single-point timing first, and none was made. [Below](#where-the-tree-is-ahead) is why it is a question of time, not of a storage wall. Since done for single points (#904, [single_points/phbi.md](single_points/phbi.md)): (4e,3o), (12e,11o) and (12e,12o)/cc-pVTZ match his to ≤ 0.14 µEh, the (12e,11o) one in 141 s on an A100; (14e,12o) is not his space, and aug-cc-pVTZ is #905. | the same; SI pp. S24–S38 have his geometries |
| **PSB3's S1 minimum**, as an optimization | **At our level his planar S1 structure is not a minimum.** The cc-pVDZ walk from it sits there for 20 evaluations, then descends: E₁ falls 27 mEh (0.74 eV) below his S1 energy, and the QD state-overlap guard stops it at evaluation 32 (overlap 0.743). The likely reading is a planar saddle along a torsion: his optimizer stayed on it, and ours, with no symmetry constraint, slides off. Table 2's S1 row comes from #860's single points at his S1 geometries instead. S1 is not part of the gated sign. | [tier3.md](tier3.md), "The S1 minimum"; #877 for the trail that was lost |
| **Ethylene's Hm MECI**, as an optimization | **His Hm is a seam point, not a seam minimum, at our level.** At his aug-cc-pVDZ Hm geometry we reproduce his gap (3.44e-5 Eh against his 3.55e-5) and its height above Py (0.676 eV against his 0.68). But a tight QD-PC-NEVPT2/aug-cc-pVDZ walk from it descends to Py, ending 0.0001 Å and 2.5e-9 Eh from our Py. It does the same at cc-pVDZ and at SA3-CASSCF. His Hm is converged loosely: gap 3.55e-5 / 2.19e-5 Eh, against ≤ 7e-6 at Py, Et and C3v. That fits a stop on the seam's slope. Not ruled out: a frozen-core seam minimum, since frozen-core gradients (#875) do not exist yet. | [ethylene_starts.md](ethylene_starts.md); the Hm ruling on #847 |
| **Ethylene's C3v MECI** at QD-PC-NEVPT2, as an optimization | **The degenerate CASSCF reference stops the gradient.** The aug-cc-pVDZ walk from his C3v was healthy: the H_eff gap fell from 7.15e-7 to 6.6e-10 Eh. Then the CASSCF's own ¹E pair became degenerate (7.873e-9 Eh), and `mcscf::root_pair_multipliers` divides by that gap. Whether the QD-NEVPT2 gradient has a finite limit inside a degenerate reference pair is a theory question (#892). The SA3-CASSCF C3v walk does converge, after #856 and #865. | [tier1.md](tier1.md), "C3v and Hm"; [ethylene_starts.md](ethylene_starts.md) |

### Frozen core is no longer out of scope

The milestone began with frozen core out of scope. This tree correlated
every inactive orbital, so every gate was chosen as a trend, a sign or a
bound to cancel an asserted 0.02–0.05 eV floor. Two things changed that:

- **#860 phase 1 measured the floor.** It replaced the assertion with
  numbers. The all-electron spread over his geometries is 3.7–5.0 meV
  (ethylene aug-cc-pVDZ), 18.8–21.5 meV (aug-cc-pVTZ), 7.9–8.1 meV (PSB3
  cc-pVDZ) and 39.8–46.1 meV (cc-pVTZ). The aug-cc-pVTZ spread is the size
  of tier 1's 20 meV bar, so frozen core became a blocker for tier 1.
  Tier 3's gate is a sign, and it was not blocked.
- **#859 added frozen-core energies** (PR #888):
  - **The key.** `[nevpt2] frozen_core = <count>` or `"auto"`, which is
    PySCF's chemcore table: 2 orbitals for ethylene and 6 for PSB3.
  - **What it changes.** The frozen orbitals leave the six classes with a
    core index. They stay in the core density that Dyall's H is built over.
  - **Canonicalization.** The core is canonicalized whole, then its lowest
    orbitals are frozen, as BAGEL does.
  - **The result.** #860 phase 2 then gave
    [the headline](#the-single-point-headline).

**Frozen-core gradients are #875 (open).** `[nevpt2_gradient]` refuses
`frozen_core` until #875 lands. So every walk here is all-electron:

- Tier 1's gated shift is a frozen-core energy at an all-electron
  structure, with the second-order argument above, measured.
- Tiers 2a, 2b and 3 are all-electron throughout.

## Where the tree is ahead

**Park's bottleneck** is the CI derivative in his section 3 ("Acceleration
of CI Derivative Contraction", pp. 16–18):

- **The 4RDM derivative.** The S⁽⁻¹⁾′_r and S⁽⁺¹⁾′_i classes need it
  (eq. 55). Storing it is "quickly untractable for the active space beyond
  (8e,8o), which needs 658 GB of memory".
- **What he does instead.** His eqs. 56–58 avoid it. They form
  intermediates from A and ⟨K|E_ij,kl|0⟩, so that "only the RDM derivatives
  up to the second order are required".
- **What is left: the 2RDM derivative,** O(N_det N_act⁴), stored with a
  multipassing algorithm. "With 10 GB of memory dedicated to the RDM
  derivatives, 12 passes are needed … for the (12e,12o) active space", and
  362 passes at (14e,14o). Each pass "requires redundant evaluations of
  2RDM derivative".
- **His proposed fix.** "Probably, contracting 2RDM derivatives with
  two-electron integrals when constructing them will solve this issue."

**What this tree does is not f3ca/f3ac.**

- **f3ca and f3ac** are integral-contracted **4-RDMs of one CI vector**,
  not derivatives
  ([`src/nevpt2/README.md`](../../../src/nevpt2/README.md),
  [`src/nevpt2/rdm/README.md`](../../../src/nevpt2/rdm/README.md)). They
  keep the explicit 4-RDM out of the energy, where it "never exists here".
  That is the energy-side analogue of his eq. 56, and of PySCF's
  `_contract4pdm`. It is not the part he was still stuck on.
- **The reverse-mode adjoint, `nevpt2::rdms_adjoint`, is what removes his
  bottleneck.** This was checked against the source, as follows.
  - **What reaches ȳ.** The class energies' reverse pass ends in bars on
    dm3, f3ca and f3ac (n⁶ each).
    [`src/nevpt2/gradient/README.md`](../../../src/nevpt2/gradient/README.md)
    ("dm1 and dm2 are traces of dm3", "The reverse pass"): dm1̄ and dm2̄
    fold into dm3̄ by the adjoint of the trace. `rdms_adjoint` takes dm3̄,
    f3cā and f3ac̄ onto ȳ and (tu|vw)'s bar. "No sigma vector is needed".
  - **How the adjoint runs.** It runs the RDM build backwards over the
    build's own tiles of determinants
    ([`src/nevpt2/rdm/README.md`](../../../src/nevpt2/rdm/README.md),
    "The adjoint"). Per tile it rebuilds R = ⟨K|E_tu|c⟩ and
    L2 = ⟨c|E_pq E_rs|K⟩ as the forward build does, transposes each GEMM,
    and gathers onto the CI bar over each determinant's replacement lists.
    "The 4-RDM is never formed, and nothing is atomic."
  - **No stored RDM derivative.** No RDM-derivative tensor over the whole
    determinant space is stored at all. L2, n⁴ per determinant, is the same
    object as his ⟨K|E_ij,kl|0⟩. It exists one tile at a time, and the
    forward build's consume W = (tu|vw) · L2 contracts it with the
    two-electron integrals inside the tile. That is the contraction his
    closing sentence proposes, and the adjoint is its transpose.
  - **The memory is the energy's.** The adjoint's workspace is fixed n⁴
    and n⁶ regions plus (4 n² + 2 n⁴) doubles per determinant of the tile,
    against the forward build's (2 n² + n⁴). Under one byte budget its
    tiles are therefore about half as wide. Nothing in the gradient
    outgrows the energy's own tiled build by more than that factor.

**This qualifies the issue's wording.** #851 says there is "nothing to
multipass". The source does not quite say that:

- **The tiles are passes too.** The forward build's CAS(12,12) case took
  9 tiles on a 16 GiB card (`Cas1212TilesAtTheDefaultBudget`,
  [`src/nevpt2/rdm/README.md`](../../../src/nevpt2/rdm/README.md)).
- **Each pass costs something.** In the adjoint, every tile walks every
  determinant's replacement lists to gather onto the CI bar.
- **What differs is what a pass repeats.** Each tile's L2 is built once per
  call, from that tile's determinants alone. No 2RDM derivative is stored
  and re-evaluated pass after pass, and the integral contraction he
  proposed happens inside the tile.

That is why pHBI is a question of time here, not a storage wall. The
forward has since been timed at pHBI's (12e,11o), in single points
(#904, [single_points/phbi.md](single_points/phbi.md)): 2 tiles and
18.6 s per root on an A100. The adjoint has not.

## The diagnostic he did not have

### The gap bound

At a MECI, a reported gap could be the gap of an unconverged CASSCF.
`nevpt2::qd_pc_intersection` returns the pair's **gap sensitivities** s_o
and s_c, the Z-vector multipliers of the gap's two elements. With the
CASSCF's residuals ‖g‖ (orbital) and ρ (CI), they bound how far the
reported gap sits from the exact surface's:

    |δgap| ≤ s_o ‖g‖ + s_c √n ρ

([`src/nevpt2/intersection/README.md`](../../../src/nevpt2/intersection/README.md),
"The gap sensitivities"; in `optimize.json` since #844).

| MECI | gap (Eh) | s_o | s_c | bound (Eh) | source |
|---|---|---|---|---|---|
| ethylene Py, Et, aug-cc-pV{D,T}Z | 2.98e-11 to 5.24e-9 | 0.048–0.192 | 1.6e-17 to 3.1e-15 | 1.3e-13 to 1.9e-12 | [tier1.md](tier1.md) |
| PSB3 13–14CI, cc-pVDZ | 3.83e-11 | 0.426 | 5.95 | 2.04e-9 | [tier2a.md](tier2a.md) |
| PSB3 11–12CI, cc-pVDZ | 2.89e-10 | 0.279 | 33.0 | 1.23e-8 | [tier2a.md](tier2a.md) |
| PSB3 13–14CI, cc-pVTZ | 9.60e-11 | 0.549 | 6.92 | 2.72e-9 | [tier3.md](tier3.md) |
| PSB3 11–12CI, cc-pVTZ | 1.60e-10 | 0.301 | 40.4 | 1.44e-8 | [tier3.md](tier3.md) |

What it shows:

- **Our MECIs are resolved.** Every bound is at least six times below
  `max_gap`. At PSB3 the bound is dominated by the CI term. The 11–12CI's
  s_c is 5.5–5.8× the 13–14CI's in both bases.
- **His MECIs are converged to gaps far above these bounds.** His gaps are
  1.05e-5 Eh (PSB3 13–14) and 2.6e-6 (11–12), with 3.55e-5 at ethylene Hm
  and ≤ 7e-6 at Py, Et and C3v. He reports no bound.

### Figure 5, with an error bar

His Figure 5 gives the amplitude as "~0.005 eV" with no statement of
resolution. Ours is **0.0015 eV half peak-to-peak (5.54e-5 Eh), with the
convergence bound ±2.56e-9 Eh (7.0e-8 eV)**. The bound is formed from the
MECI's s_o and s_c and the worst scan point's ‖g‖ and ρ, each taken
separately.

- **The assumption.** The bound assumes s_o and s_c barely change over the
  loop. Each point's own intersection gives s_o of 0.4234–0.4293 and s_c
  of 5.751–6.061, within 2% and 3% of the MECI's. The worst bound formed
  point by point is 2.57e-9 Eh.
- **A separate systematic: the sampling.** At 15° sampling the scan can
  miss the true extrema by at most about 3% of the half amplitude, about
  0.05 meV ([tier2b.md](tier2b.md)). That sampling error, not convergence,
  is the larger uncertainty on the amplitude, by about three orders.

### The gradient against finite differences

His single-state PC-NEVPT2 paper validated the analytic gradient against a
two-point finite difference to within 5.0×10⁻⁶ a.u. (J. W. Park, arXiv
1907.10180v3, §4). Here the PC-QD gradient at PSB3 was checked against
re-converged central differences with Richardson extrapolation, to a
measured noise floor. FC passes 42 of 42 components, the largest miss
8.7e-9. The 90° twist passes 42 of 42 too
([`src/nevpt2/energy/README.md`](../../../src/nevpt2/energy/README.md),
"The derivative across a continuum"). One component, atom 4 x at the
twist, misses by 6.3e-9 on average, inside its bar; whether that is
CASSCF stopping bias or analytic is #862 (open).

## What the reproduction hardened

Defects found and fixed:

| issue | PR | defect, and fix |
|---|---|---|
| #853 | #868 | PC's metric gap rule refused every PSB3 CAS(6,6) geometry. The modes at the cut are real and carry no energy. The fix is a BAGEL-style cut, max(1e-13, 100 ε √d) of the largest eigenvalue, with no gap test, shipped with an FD check of the PC-QD gradient at PSB3. |
| #856 | #863 | The SA weights were recovered from the roots' 1-RDMs by least squares. That drifted past 1e-8 near ethylene's C3v and refused the walk. Now `mcscf::Solution` carries the weights. |
| #858, #864 | #872 | The fixed 1e-10 Z-vector bars (CASSCF intersection, and the QD-NEVPT2 Lagrangian) were crossed by a floor from the inexact CI response inside each H̃ product. Now H̃'s response is solved to max(1e-2 t, 1e-10), and the one-column path checks its true residual too. |
| #865 | #890 | RHF stalled at ethylene's C3v geometries, because every DIIS extrapolation read as singular under an absolute pivot test. Now a singular subspace restarts with scaled residuals. |
| #874 | #883 | PC refused negative denominators, which refused ethylene S0's state 2, a state Park reports. Now it refuses only poles: an amplitude above 1, or a denominator at round-off. |

Features the milestone added on the way: #844 (the branching space, and s_o
and s_c in `optimize.json`), #845 (aug-cc-pV{D,T}Z-JKFIT, PR #871) and #859
(frozen-core energies, PR #888); #867, the CASSCF gradient floor at PSB3 (PR #896, after these runs: see "What the configs ask for").

Still open:

| issue | what |
|---|---|
| #862 | A 6e-9 FD residual in the PC-QD gradient at twisted PSB3 (atom 4 x): CASSCF stopping bias, or analytic? |
| #875 | Frozen-core NEVPT2 and QD-NEVPT2 gradients. |
| #892 | The QD-NEVPT2 gradient at a degenerate CASSCF reference pair (ethylene C3v). |
| #866 | Expose mcscf's convergence knobs as options-struct parameters. Not in the milestone. |
| #869 | Flaky: `Nevpt2NumericalHeff.H2coSkewSa3`'s PC finite differences are intermittently noisy. Not in the milestone. |
| #870 | Flaky in the HIP GPU batch: `McscfTransitionGradient` h2co_skew_ccpvdz_sa3, and the batch log drops the assertion. Not in the milestone. |
| #877 | `[optimize]` writes no `optimize.json` or `optimize.xyz` when a walk fails partway. Not in the milestone. |
| #878 | `Nevpt2QdIntersection.Psb3TrueResidual` times out (120 s) in the HIP slow tier. Not in the milestone. |

## Where the record disagrees with itself

None of these changes a verdict above. Each is stated rather than smoothed
over.

- **Stale text, now corrected (#851).** Three places in this directory
  had fallen behind, and were corrected with this page:
  - [README.md](README.md)'s "What cannot match" said this tree has no
    frozen-core option and that the paper has no Supporting Information.
    #859 and the SI (the milestone's update of 2026-11-04) superseded both.
    Its tier-0 text waited on s_o and s_c from #844, which tier 0 never
    received. That is now stated there.
  - [single_points/README.md](single_points/README.md) (phase 1) said
    tier 1's shift is "not measurable as PC until #874 lands", and showed
    ethylene S0 as refused. Both were true at 4f84cfc. The page now notes
    that #874 landed and that tier 1 measured the shift.
  - [ethylene_starts.md](ethylene_starts.md) said "the gated per-MECI
    basis shift covers Py, Et and C3v", which was the Hm ruling's wording.
    The C3v ruling on #847 that followed removed C3v. The page now cites
    that ruling, and [tier1.md](tier1.md) gates Py and Et only.
- **The milestone's tier 1 premise.** It assumed the core correlation
  "cancels to second order" in the shift. #860 showed it does not: the
  all-electron shifts are 15.7 (Py) and 11.1 meV (Et) larger than the
  frozen-core ones. Hence the frozen-core gated shift.
- **The milestone's floor.** It asserted a floor of about 0.02–0.05 eV. As
  a *spread* across geometries, #860 measured 4–46 meV. As an *offset*,
  the all-electron energies sit 0.15 eV (ethylene aug-cc-pVDZ) to 2.5 eV
  (PSB3 cc-pVTZ) below his. The gates were built to cancel the offset, and
  that worked; the spread is what reached tier 1's bar.
- **Park's own text against his data:**
  - He describes the 13–14CI elongation as "~0.07 Å". His coordinates give
    0.081 Å at cc-pVDZ and 0.075 Å at cc-pVTZ, and we measure 0.080 and
    0.072 Å.
  - His Table 2 footnotes a and b are labelled "E(S1) − E(MECI)", but the
    printed values are E(MECI) − E_S1 ([tier3.md](tier3.md), "His three
    footnotes").
  - His SI exists. The milestone and the early issues were written
    assuming it did not.
- **Tier counts against the plan.** Tier 1 planned ten optimizations and
  ran six, plus one C3v walk that stopped. Tier 3 planned eight table
  entries from optimizations. Six are optimized structures; the two S1
  entries are #860's single points at his geometries, because the cc-pVDZ
  S1 walk did not converge and no cc-pVTZ S1 walk was run.
