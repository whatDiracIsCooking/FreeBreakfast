# Tier 1: ethylene's Table 1 at QD-PC-NEVPT2/aug-cc-pV{D,T}Z

#847. This is Park's Table 1: ethylene's S0 minimum and its S1/S0 MECIs at
SA-QD-PC-NEVPT2 over DF-SA3-CASSCF(2,2), in aug-cc-pVDZ (aug-cc-pVDZ-JKFIT)
and aug-cc-pVTZ (aug-cc-pVTZ-JKFIT). The source is J. W. Park, *J. Chem.
Theory Comput.* (2020), acs.jctc.9b00919: Table 1, Figure 1 and p. 19.
Each walk starts from his optimized geometry at the same basis, taken from
the Supporting Information, pp. S2–S6 (`single_points/geometries/`). Each
walk is optimized all-electron. The energies compared are frozen-core
single points at the converged structures.

**Result: the gated aug-cc-pVDZ → aug-cc-pVTZ shift passes for both gated
MECIs.**

- **Py:** +0.0820 eV, against his +0.08 (Table 1), a difference of
  +0.002 eV.
- **Et:** +0.0379 eV, against his +0.04, a difference of −0.002 eV.

The bar is 0.02 eV. The C3v and Hm rows are reported, not gated, per
#847's rulings (below).

## The runs

Six optimizations: the S0 minimum (QD state 0), the Py MECI and the Et MECI
(`meci = [0, 1]`), at each basis. The configs are
[`tier1_ethylene_{s0,py,et}_aug{dz,tz}.toml`](.). All six share the same
settings:

- `[nevpt2_gradient] quasi_degenerate = true, variant = "pc"`;
- `[casscf] conv_tol_grad = 1e-10`;
- Opt=Tight's four criteria, and `max_gap = 1e-7` Eh for the MECIs (tier 0's
  criteria).

The converged structures are checked in as
[`geometries/ethylene_{s0,py,et}_aug{dz,tz}.xyz`](geometries/), each a
`final.xyz`. A seventh walk,
[`tier1_ethylene_c3v_augdz.toml`](tier1_ethylene_c3v_augdz.toml), is kept
as the evidence of the C3v stop below.

| walk | evaluations | wall | final energy (Eh) | final gap (Eh) | max / rms force (Eh/bohr) | max / rms displacement (bohr) |
|---|---|---|---|---|---|---|
| S0, aug-cc-pVDZ | 7 | 4 s | −78.339974309 | – | 1.3e-6 / 4.8e-7 | 2.8e-6 / 1.0e-6 |
| Py, aug-cc-pVDZ | 8 | 9 s | −78.177129717 | 1.43e-9 | 5.4e-6 / 3.1e-6 | 4.8e-5 / 1.7e-5 |
| Et, aug-cc-pVDZ | 9 | 10 s | −78.166971616 | 5.24e-9 | 9.2e-6 / 4.3e-6 | 3.5e-5 / 1.6e-5 |
| S0, aug-cc-pVTZ | 9 | 32 s | −78.446998234 | – | 5.4e-7 / 2.1e-7 | 1.4e-5 / 8.6e-6 |
| Py, aug-cc-pVTZ | 25 | 172 s | −78.280562853 | 2.28e-9 | 1.0e-6 / 4.1e-7 | 1.2e-5 / 4.6e-6 |
| Et, aug-cc-pVTZ | 16 | 111 s | −78.272192974 | 2.98e-11 | 4.6e-6 / 2.6e-6 | 4.9e-5 / 1.7e-5 |

A MECI's energy is the mean of QD states 0 and 1. Every log ends
`completed: 0 errors, 0 warnings`. A PC refusal would have stopped its walk,
so convergence is the statement that PC was refused at no evaluation
(README.md, "Tier 0").

Then, at each of the six converged structures, there are two
SA3-QD-PC-NEVPT2 single points. One is all-electron, `[nevpt2]
quasi_degenerate = true`. The other is frozen-core, which adds
`frozen_core = "auto"`; each frozen-core log reports
`NEVPT2: 2 frozen core orbitals, 5 correlated`. Each single point is the
optimization config with `[nevpt2_gradient]` and `[optimize]` replaced by
that `[nevpt2]`, and `xyz` pointed at the converged geometry. They are
checked in as
[`tier1_ethylene_{s0,py,et}_aug{dz,tz}_sp.toml`](.) (all-electron) and
`…_sp_fc.toml` (frozen core), written from that recipe after the runs
(#851) and not rerun. All 12 ended
`completed: 0 errors, 0 warnings`, with no PC-QD refusal. The all-electron
energies equal the walks' final energies to the 12 decimals printed.

## The gated shift

The shift is ΔE_TZ − ΔE_DZ, where ΔE = E(MECI) − E(S0 at the S0 minimum),
both taken at the same basis and the same level. His Table 1 lists
4.51 / 4.43 (Py) and 4.74 / 4.70 eV (Et) for aug-cc-pVTZ / aug-cc-pVDZ.
So a positive shift means the MECI rises with the basis.

| MECI | ΔE aug-cc-pVDZ (eV) | ΔE aug-cc-pVTZ (eV) | shift, frozen-core (eV) | his Table 1 | ours − his | verdict (bar 0.02 eV) | shift, all-electron (eV) |
|---|---|---|---|---|---|---|---|
| Py | 4.42673 | 4.50870 | **+0.08197** | +0.08 | +0.0020 | **pass** | +0.09771 |
| Et | 4.70369 | 4.74161 | **+0.03792** | +0.04 | −0.0021 | **pass** | +0.04905 |

Computed from his SI's 10-decimal energies, his own shifts are +0.08187
(Py) and +0.03790 eV (Et). Ours differ from those by +0.10 and +0.02 meV.

The all-electron shifts are shown for comparison only. They are 15.7 (Py)
and 11.1 meV (Et) larger than the frozen-core ones. That is #860's core
floor: about 5 meV at aug-cc-pVDZ and 19–22 meV at aug-cc-pVTZ, and it
does not cancel between S0 and a MECI. The frozen-core shifts are the
like-for-like comparison with his.

### Why all-electron geometries are good enough here

Frozen-core MECI gradients do not exist yet (#875), so the frozen-core
energies are taken at the all-electron structures, not at frozen-core
optima. The error that brings is second order:

- **S0 minimum.** E_FC is stationary at its own minimum. So moving from
  that minimum to the all-electron one changes E_FC at second order in the
  displacement.
- **MECI.** The two-state mean energy is stationary along the seam. A
  displacement within the seam therefore enters at second order.
- **The branching plane.** A displacement out of the seam, into the
  branching plane, enters the mean at first order, through the seam's tilt.
  But that component is what opens the frozen-core gap, so it is bounded by
  that gap. The gap is 1.77 meV at Py, 0.10 meV at Et/aug-cc-pVDZ and
  0.018 meV at Et/aug-cc-pVTZ.

The effect can be measured, not just argued. With the core frozen, this
tree reproduces Park's energies at his own geometries to ≤ 0.52 µEh
([`single_points/frozen_core.md`](single_points/frozen_core.md)), and his
geometries are his frozen-core optima. So E_FC(our all-electron structure)
− E_FC(his structure) is the geometry error itself:

| structure | aug-cc-pVDZ (meV) | aug-cc-pVTZ (meV) |
|---|---|---|
| S0 minimum, state 0 | +0.08 | +1.55 |
| Py MECI, mean of 0, 1 | −0.30 | +1.27 |
| Et MECI, mean of 0, 1 | +0.23 | +1.71 |

Each is at most 1.7 meV. Most of each cancels in the shift, which is a
difference of differences. The net effect on the shift is +0.10 meV (Py)
and +0.02 meV (Et), the residuals above. That is under 1% of the 20 meV
bar. The
Py mean falls 0.30 meV below his MECI at aug-cc-pVDZ. That is possible
because our structure is off his seam: the gap there is 1.77 meV, which is
the first-order branching-plane term the third bullet bounds.

## Table 1

Each energy is in eV, relative to state 0 at our S0 minimum at the same
basis and level. Columns S0, S1 and S2 are QD states 0, 1 and 2. At the
S0 minimum, S1 and S2 are vertical excitations; at a MECI, S0 ≈ S1. His
values are from the SI's absolute energies, relative to his S0 minimum.

The C3v and Hm rows are not our optimizations. They are single points at
**his** geometries, referenced to our S0 minimum. Their frozen-core
absolutes are #859's ([`single_points/frozen_core.md`](single_points/frozen_core.md)),
except C3v at aug-cc-pVDZ, which is new here. Their all-electron absolutes
are #860's phase 1 ([`single_points/README.md`](single_points/README.md)),
again except C3v at aug-cc-pVDZ, which is new here.

### aug-cc-pVDZ

| structure | S0 FC | S1 FC | S2 FC | S0 AE | S1 AE | S2 AE | S0 his | S1 his | S2 his |
|---|---|---|---|---|---|---|---|---|---|
| S0 minimum | 0 | 7.5321 | 13.1622 | 0 | 7.5324 | 13.1627 | 0 | 7.5208 | 13.1380 |
| Py MECI | 4.4258 | 4.4276 | 8.9145 | 4.4312 | 4.4312 | 8.9173 | 4.4270 | 4.4272 | 8.9171 |
| Et MECI | 4.7036 | 4.7037 | 6.0227 | 4.7076 | 4.7076 | 6.0250 | 4.7035 | 4.7036 | 6.0228 |
| C3v, his geometry | 4.8393 | 4.8394 | 6.0036 | 4.8429 | 4.8429 | 6.0058 | 4.8394 | 4.8394 | 6.0037 |
| Hm, his geometry | 5.1061 | 5.1071 | 8.2791 | 5.1066 | 5.1075 | 8.2784 | 5.1062 | 5.1071 | 8.2792 |

### aug-cc-pVTZ

| structure | S0 FC | S1 FC | S2 FC | S0 AE | S1 AE | S2 AE | S0 his | S1 his | S2 his |
|---|---|---|---|---|---|---|---|---|---|
| S0 minimum | 0 | 7.6864 | 13.5095 | 0 | 7.6893 | 13.5145 | 0 | 7.6430 | 13.4054 |
| Py MECI | 4.5078 | 4.5096 | 8.8968 | 4.5289 | 4.5289 | 8.9133 | 4.5090 | 4.5090 | 8.9030 |
| Et MECI | 4.7416 | 4.7416 | 6.0532 | 4.7567 | 4.7567 | 6.0676 | 4.7414 | 4.7415 | 6.0540 |
| C3v, his geometry | 4.8541 | 4.8541 | 6.0260 | 4.8692 | 4.8692 | 6.0414 | 4.8556 | 4.8556 | 6.0276 |
| Hm, his geometry | 5.1253 | 5.1259 | 8.2470 | 5.1286 | 5.1289 | 8.2481 | 5.1268 | 5.1274 | 8.2486 |

How the columns compare:

- **The vertical excitations at S0 differ from his by 11–104 meV**, while
  the MECI rows agree with his to a few meV. That is because our S0
  minimum is the all-electron one. At aug-cc-pVTZ its C=C bond is
  0.0052 Å shorter than his (below). A vertical excitation at a minimum is
  not stationary in the geometry, so it carries the geometry difference at
  first order.
- **The S0 → MECI energies** are stationary, as argued above. As
  two-state means, the frozen-core ones agree with his to ≤ 0.4 meV at
  aug-cc-pVDZ and ≤ 1.6 meV at aug-cc-pVTZ. At Py the individual states
  differ from his by up to 1.2 meV, because our structure sits off his
  frozen-core seam: the gap there is 1.8 meV.

The all-electron absolutes, against his frozen-core ones, are −5.5 mEh
(aug-cc-pVDZ) and −35 mEh (aug-cc-pVTZ) apart: his missing core
correlation, as in #860 phase 1.

### The C3v and Hm shifts, reported and not gated

| MECI | frozen-core shift (eV) | all-electron shift (eV) | his, SI (eV) | his Table 1 |
|---|---|---|---|---|
| C3v, his geometries | +0.0147 | +0.0263 | +0.0162 | +0.02 |
| Hm, his geometries | +0.0190 | +0.0217 | +0.0205 | +0.02 |

The C3v and Hm energies are at his geometries, and the S0 reference is at
ours. So the C3v and Hm frozen-core shifts carry the S0 reference's +1.55
meV (aug-cc-pVTZ) and +0.08 meV (aug-cc-pVDZ) geometry error from the
table above. Referenced to his S0 geometry instead, they are his to
≤ 0.01 meV: +0.01620 (C3v) and +0.02048 eV (Hm)
([`single_points/frozen_core.md`](single_points/frozen_core.md)).

## C3v and Hm: not optimized, and why

- **Hm** (#847's Hm ruling). From his aug-cc-pVDZ Hm geometry, a tight
  QD-PC-NEVPT2/aug-cc-pVDZ walk (#846's) descends the seam to Py, 0.0001 Å
  from #846's own Py. It collapses the same way at cc-pVDZ and at
  SA3-CASSCF. His Hm is loosely converged: his gap is 3.55e-5 /
  2.19e-5 Eh, against ≤ 7e-6 at Py, Et and C3v. Park's Hm is a seam point,
  but not a seam minimum at our level, so no Hm walk was run.
- **C3v** (#847's C3v ruling, #892).
  [`tier1_ethylene_c3v_augdz.toml`](tier1_ethylene_c3v_augdz.toml),
  started from his aug-cc-pVDZ C3v, stops at the gradient of evaluation 10
  (0-based):

  ```
  mcscf::root_pair_multipliers: roots 0 and 1 are 7.873e-09 hartree apart,
  under 1e-08: the pair's multiplier divides by the gap
  ```

  The QD walk itself was healthy:
  - The H_eff gap went from 7.15e-7 to 6.6e-10 Eh in nine steps.
  - The mean energy went from −78.1620013 to −78.1620050 Eh.
  - `pair_overlap` was ≥ 0.9999996, and the third-state margin was
    0.0427 Eh.
  - Every CASSCF converged to |g| ≤ 4e-12.

  It is the **CASSCF's** own ¹E pair that becomes degenerate on the
  approach to C3v. Its root gap went from 2e-6 Eh at evaluation 0 to
  below 1e-8, and the intersection's CASSCF root-pair multiplier divides
  by that gap. Whether the QD-NEVPT2 gradient has a finite limit inside a
  degenerate reference pair is a theory question, filed as #892. The walk
  wrote no `optimize.json` (#877), and the aug-cc-pVTZ C3v walk was not
  run.

## The MECIs' gap bounds and overlap trails

Each bound is `s_o ‖g‖ + s_c √n ρ` (n = 3) at the last evaluation, from
`optimize.json`'s `gap_sensitivity_orbital`, `gap_sensitivity_ci`,
`gradient_norm` and `ci_residual_norm` (#844).

| MECI | gap (Eh) | s_o | ‖g‖ | s_c | ρ | bound (Eh) |
|---|---|---|---|---|---|---|
| Py, aug-cc-pVDZ | 1.43e-9 | 0.183 | 2.6e-12 | 1.6e-17 | 7.4e-14 | 4.8e-13 |
| Et, aug-cc-pVDZ | 5.24e-9 | 0.048 | 2.8e-12 | 2.9e-15 | 3.0e-14 | 1.3e-13 |
| Py, aug-cc-pVTZ | 2.28e-9 | 0.192 | 9.6e-12 | 1.7e-17 | 3.8e-14 | 1.9e-12 |
| Et, aug-cc-pVTZ | 2.98e-11 | 0.048 | 7.2e-12 | 3.1e-15 | 3.4e-14 | 3.5e-13 |

Every bound is ≤ 1.9e-12 Eh: below its own reported gap, and more than
four orders below `max_gap`.

| MECI | `pair_overlap` minimum (evaluation) | last | gap at the start (Eh) | third-state margin, minimum (Eh) | CASSCF root gap, minimum (Eh) | max ‖g‖ / max ρ over the trail |
|---|---|---|---|---|---|---|
| Py, aug-cc-pVDZ | 0.99999991 (1) | 0.9999999998 | 5.84e-5 | 0.1648 | 0.0278 | 2.7e-11 / 7.4e-14 |
| Et, aug-cc-pVDZ | 0.999999986 (3) | 0.99999999997 | 3.84e-6 | 0.0484 | 2.28e-4 | 3.5e-11 / 8.1e-14 |
| Py, aug-cc-pVTZ | 0.9999993 (2) | 0.99999999999 | 6.49e-5 | 0.1610 | 0.0284 | 8.5e-11 / 9.0e-14 |
| Et, aug-cc-pVTZ | 0.9999988 (2) | 0.99999999999 | 4.18e-6 | 0.0482 | 1.90e-4 | 5.8e-11 / 1.1e-13 |

The S0 walks' QD `state_overlap` stayed ≥ 0.99999997 (aug-cc-pVDZ) and
≥ 0.9999996 (aug-cc-pVTZ). The floor is 0.9. Starting at his geometry, the
all-electron gap is 1.6–1.8 meV (Py) and 0.1 meV (Et).

## The converged structures against his

RMSD is over all six atoms, in Å, after the best rigid superposition
(Kabsch). His structures are the SI's (bohr → Å, `single_points/geometries/`).
Figure 1 reports bond lengths, and its numbers are not available as text,
so the same parameters are tabulated here from the SI coordinates. Atoms
are numbered C0, C1, H2–H5 in his order.

| structure | RMSD (Å) | r(C–C) ours / his (Å) | r(C–H) ours / his (Å) | angle ours / his (°) |
|---|---|---|---|---|
| S0, aug-cc-pVDZ | 0.0014 | 1.3431 / 1.3446 | 1.0938 / 1.0949 (all four) | – |
| S0, aug-cc-pVTZ | 0.0049 | 1.3206 / 1.3257 | 1.0797 / 1.0828 (all four) | – |
| Py, aug-cc-pVDZ | 0.0012 | 1.4324 / 1.4343 | C1–H4 1.1591 / 1.1603, C1–H5 1.1220 / 1.1231, C0–H 1.1010, 1.1046 / 1.1019, 1.1055 | ∠C0C1H4 92.36 / 92.32 |
| Py, aug-cc-pVTZ | 0.0050 | 1.4116 / 1.4178 | C1–H4 1.1429 / 1.1468, C1–H5 1.1067 / 1.1107, C0–H 1.0857, 1.0895 / 1.0897, 1.0932 | ∠C0C1H4 92.90 / 92.97 |
| Et, aug-cc-pVDZ | 0.0014 | 1.4464 / 1.4481 | C0–H 1.1059, 1.1160, 1.1059 / 1.1069, 1.1169, 1.1070; C1–H5 1.0815 / 1.0826 | ∠C0C1H5 152.63 / 152.62 |
| Et, aug-cc-pVTZ | 0.0050 | 1.4282 / 1.4344 | C0–H 1.0929, 1.1023, 1.0929 / 1.0952, 1.1048, 1.0952; C1–H5 1.0635 / 1.0692 | ∠C0C1H5 153.62 / 153.49 |

Every all-electron bond is shorter than his frozen-core one:

- at aug-cc-pVDZ, by 0.0009–0.0019 Å;
- at aug-cc-pVTZ, by 0.0023–0.0062 Å.

That is the expected sign and size of core-valence correlation, which
aug-cc-pVTZ can partly describe and aug-cc-pVDZ hardly at all. It is the
same 4–6× growth #860 found in the energy floor. The tabulated angles agree
to ≤ 0.13°.

## What was run

- **Card and build.** RTX 3080 (CUDA, 10 GiB), `devtools/cpp-tier.sh
  --preset default --no-test` (`cpp-tier PASS rev=e1868b8 preset=default`),
  the tree at e1868b8. No ctest was run: this PR changes no code. Every run
  held the card with `with_gpu_card cuda`, as in README.md "How the runs
  are made". The 12 single points ran under one hold.
- Run outputs (logs, `optimize.json`, Molden files) stay out of git. The
  numbers above are quoted from them.
