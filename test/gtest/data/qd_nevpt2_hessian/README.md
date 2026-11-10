# qd_nevpt2_hessian golden data

Semi-numerical nuclear Hessians of this tree's SC- and PC-QD-NEVPT2 states,
the oracle the analytic QD Hessian (#959, #965–#967) is gated on
(`docs/nevpt2/nevpt2.tex`, Table qdhesschecks's semi-numerical row): a
fourth-order central stencil at steps h, h/2 and h/4 over the tree's own
analytic QD gradient G(W) (`nevpt2::qd_sc_gradient`, `qd_pc_gradient`,
milestone 20), every displaced point re-converged through a continuity
guard. Written by `tools/nevpt2_hessian/qd_nevpt2_hessian.py` over the C++
driver `free-breakfast-qd-nevpt2-hessian` (`tools/qd_nevpt2_hessian/`)
(#953); both docstrings are the method. Never hand-edited.

No other program has a QD-NEVPT2 gradient, so the gradient differenced is
the tree's, which `../qd_nevpt2_gradient/` ladder-validates against the
tree's own energy (whose reference points `../qd_nevpt2/` pins to
independent code). This data establishes that a Hessian is the derivative
of that gradient: what an analytic Hessian assembled from second-order
pieces must reproduce.

## Cases

The system, basis, active space and weights of the first two are
`../qd_nevpt2/qd_nevpt2.json`'s; the third is the same geometry at
0.4/0.2/0.4, its reference point `qd_weight_search.json`'s (below).
cc-pVDZ / cc-pVDZ-JKFIT, CAS(4,3), three roots, all-electron, Cartesian
coordinates (C1: every block defined).

| case | weights | scheme, h (bohr) | bar, max (Eh/bohr²) | floor | worst asymmetry | inactive gaps (core, virtual) | H_eff gaps |
|---|---|---|---|---|---|---|---|
| `h2co_skew_ccpvdz_sa3` | equal | SC, 3.125e-4 | 5.0e-7 | 1.1e-7 – 5.0e-7 | 8.5e-9 | 0.072, 7.6e-3 | 0.1545, 0.1986 |
| | | PC, 2.5e-3 | 6.4e-8 | 1.3e-8 – 6.4e-8 | 1.3e-9 | | 0.1546, 0.1985 |
| `h2co_skew_ccpvdz_sa3_weighted` | 0.4/0.4/0.2 | PC, 2.5e-3 | 1.4e-7 | 2.4e-8 – 1.4e-7 | 2.2e-8 | 2.6e-3, 3.6e-3 | 0.1566, 0.1907 |
| `h2co_skew_ccpvdz_sa3_w424` | 0.4/0.2/0.4 | SC, 3.125e-4 | 3.5e-7 | 1.2e-7 – 3.5e-7 | 1.1e-8 | 0.084, 6.5e-3 | 0.1489, 0.2004 |

**For #965's unequal-weight check**: SC on `h2co_skew_ccpvdz_sa3_w424`, PC
on `h2co_skew_ccpvdz_sa3_weighted`. The equal-weight check reads both
schemes on `h2co_skew_ccpvdz_sa3`.

Per scheme, every one of the three states' Hessians (`states`) and every
pair's (`pairs`, (0,1), (0,2), (1,2)). Every element of every ladder settles:
each finer difference D(h/2) − D(h/4) lies below the measured floor, and
each coarser one D(h) − D(h/2) below 19.2 floors (the largest is 1.0e-7, SC
state 0 of the equal-weight case; 2.5e-8, the pair (0,2), on
`_w424`), so no ratio is judged and every bar is its floor. Every Hessian is
symmetric to its bar. The pair sums equal their two states' sums within the
three bars (G is linear in W; each W has its own Z-vector solve).

**The step.** The guard holds a matched root's |CI overlap| within 1e-3 of 1
at every point (the library's `kCiTol`), and on these cases that overlap
falls off by 9.3e-3 at 0.02 bohr along O z (h = 0.01's 2h). So h ≤ 2.5e-3
(max displacement 5e-3, worst miss 5.9e-4). PC's ladder at 2.5e-3 is
already inside its floor everywhere.

**SC needs a far smaller step, and settles on the equal-weight case only**
(`kQdSmooth` in `test/gtest/nevpt2_gradient_gpu.cpp`; #724, #894). Measured
on `h2co_skew_ccpvdz_sa3`: at h = 2.5e-3, SC's six quantities miss on 195
elements and 11 pairs' symmetry (ratios from −100 to 90, rung differences
up to 2.8e-5), where PC, at the same points, misses nothing; at
6.25e-4, one element, H[1][1] (C y, C y) of state 0 and of the pair (0,1),
reads 11.1 and 12.2 against (12.8, 19.2); at 3.125e-4, the committed step,
every element is inside its floor. On `h2co_skew_ccpvdz_sa3_weighted`, whose
two closest core orbitals are 2.6e-3 Eh apart, SC at 3.125e-4 still misses
on 62 elements of state 1 (22), state 2 (1) and the pairs (0,1) and (1,2),
mostly along C y and C z (ratios 9.7 to 12.7, rung differences up to
1.1e-4): its curvature has not reached the
asymptotic range at a step the floor still resolves, so the case carries PC
only. SC's unequal-weight reference is `h2co_skew_ccpvdz_sa3_w424`, below.

## SC's unequal-weight case (#1060)

`qd_weight_search.json` (`tools/nevpt2_hessian/qd_weight_search.py`, PySCF
over `tools/emit/qd_nevpt2_reference.py`'s SC-QD-NEVPT2, the point polished
to |g| < 1e-10) screens the skew geometry at sixteen weightings, at the
reference point alone: the inactive gaps, per root and class the smallest
norm the guard keeps and how many kept norms are under 1e-6, the root gaps
and SC's spectrum. The weightings fall on two SA-CASSCF solutions:

| branch | weightings | E₀ (Eh) | closest core pair | closest virtual pair |
|---|---|---|---|---|
| the equal-weight one's | equal; 0.4/0.3/0.3, 0.3/0.4/0.3, 0.3/0.3/0.4, 0.2/0.4/0.4, **0.4/0.2/0.4**, 0.35/0.35/0.3, 0.3/0.35/0.35, 0.25/0.25/0.5, 0.2/0.3/0.5 | −113.82 to −113.86 | 0.053 – 0.084 | 6.2e-3 – 9.6e-3 |
| 0.4/0.4/0.2's | 0.4/0.4/0.2, 0.5/0.3/0.2, 0.5/0.25/0.25, 0.6/0.2/0.2, 0.45/0.35/0.2, 0.38/0.38/0.24 | −113.896 to −113.902 | 2.6e-3 – 1.3e-2 | 3.6e-3 – 5.6e-3 |

Every weighting of the second branch gives root 3 at most 0.25, and its
third root sits about 0.1 Eh above the first branch's. The SC energy's
canonical-orbital response divides by these pairs' gaps, so its curvature
changes on a scale of the gap. That fits 0.4/0.4/0.2's slow ratios
(9.7–12.7 at 3.125e-4). The smallest kept norms do not separate the
branches. Si's per root are 3e-6, 2e-7, 5e-7 at 0.4/0.2/0.4, against
4e-6, 1e-7, 2e-7 at equal weights (which settles) and 2e-7, 7e-8, 7e-7 at
0.4/0.4/0.2. Every candidate keeps some tuple between 5e-14 and 5e-13.

0.4/0.2/0.4 is the weighting with the widest core gap, 0.084 Eh (virtual
6.5e-3), and its ratio of 2 between weights exercises θ. Its SC ladder at
the equal-weight case's h = 3.125e-4 settles on the first run: every rung
difference is under 2.5e-8, against floors of 1.2e-7 to 3.5e-7 (σ_g 3.2e-13
to 9.7e-13). Every Hessian is symmetric to its bar (worst 1.1e-8). The pair
sums hold. The guard's worst values are prediction 0.57 |dR|², active space
4.8e-7, CI 9.1e-6 and u_K 5.4e-10. The tree's reference point matches the
search's PySCF one to 2.0e-12 (CASSCF) and 1.6e-12 (SC's H_eff)
(`reference.agreement`). No other weighting was laddered. Only SC is
written for this case; PC's unequal-weight reference is
`h2co_skew_ccpvdz_sa3_weighted`.

### Keys (`qd_weight_search.json`)

| key | meaning |
|---|---|
| `atoms` (bohr), `basis`, `ncas`, `nelecas` | the geometry every candidate shares |
| `tolerances` | the guard (1e-14) and the kept norm counted `small` (1e-6) |
| `chosen` | the case laddered |
| `candidates` | per weighting: `weights`, `casscf_energies`, `casscf_gaps`, `next_root_gap`, `inactive_gaps` (`core`, `virtual`: the smallest SA-canonical spacing), `sc_energies` (H_eff's eigenvalues), `smallest_kept` (per class but Sijrs, per root, the smallest kept `norm` and the `small` count) |

## The fast tier's case (#1243)

`h4_631g_sa2.json`: the smallest C1 case with every part of the analytic QD
Hessian, for the representatives of the three QD Hessian families
(`df_qd_sc_nevpt2_hessian_gpu`, `df_qd_pc_nevpt2_hessian_gpu`,
`df_qd_nevpt2_crossing_hessian_gpu`). #1207 timed the analytic QD Hessian at
6 to 50 s on the cases above, most of it in steps that run over every
Cartesian coordinate whatever block is asked for, so the
cost is the atoms', and four atoms is the floor for a C1 case. Two stretched,
twisted H₂ units (bonds 1.70 and 1.51 bohr), no symmetry, 6-31G /
cc-pVDZ-JKFIT, CAS(2,2) over the RHF's frontier σ and σ*, SA2 at equal
weights: a core of one and five virtuals. In STO-3G the one virtual leaves
κ_oo no rows; in 6-31G each part of the Hessian is visible, theta's share
too, on the block the tests take. The roots are 0.368 Eh apart (H_eff 0.379
SC and PC), the third root 0.374 above the second, and the closest
virtuals 0.084 Eh apart (0.082 over every point).

| scheme, h (bohr) | quantities | σ_g | floor (every bar) | worst asymmetry |
|---|---|---|---|---|
| SC, 3.125e-4 | states 0, 1; pair (0, 1) | 5.9e-14 to 1.9e-13 | 2.2e-8 to 7.0e-8 | 8.4e-10 |
| PC, 5e-3 | states 0, 1; pair (0, 1) | 5.7e-14 to 1.9e-13 | 1.3e-9 to 4.4e-9 | 1.4e-10 |

Every element of every ladder settles inside its floor on the run written:
the largest finer difference is 0.12 floors, the largest coarser one 0.46
(the PC pair). No ratio is judged and every bar is its floor. Guard, worst
over the 212 points: prediction 0.081 |dR|², active space 1.6e-5, CI 3.0e-5.
A core of one has no core pair, so `inactive_gaps.core` and
`guard.core_gap` are null.

**PC's step is 5e-3, twice the C1 cases'.** At 2.5e-3 the PC pair's column
along H1 y missed on its finest rung only: D(h/2) − D(h/4) of 3.4e-9 to
4.6e-9 against a floor of 2.6e-9 (ratios 0.004 to 0.006, and those four
elements asymmetric by up to 1.004 bars), with D(h) − D(h/2) at 2e-11.
That is noise at the ±h/4 points past what the restarts measured, which a
step twice as long halves; both states settled at 2.5e-3 too. The guard
holds at 5e-3 (the CI overlap's worst miss is 3.0e-5 of its 1e-3).

**The reference point** is held to PySCF's: `h4_631g_sa2.pyscf.json`,
written by `tools/nevpt2_hessian/qd_pyscf_point.py` over
`tools/emit/qd_nevpt2_reference.py`'s polished SA-CASSCF and SC- and
PC-QD-NEVPT2 (`casscf_energies`, `sc_energies`, `pc_energies`, the
`inactive_gaps` and `next_root_gap`). The tree's matches it to 1.4e-13
(CASSCF), 1.6e-13 (SC) and 1.6e-13 (PC) (`reference.agreement`).

The keys are the Keys table's.

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/nevpt2_hessian/qd_pyscf_point.py          # seconds, no GPU
tools/nevpt2_hessian/qd_nevpt2_hessian.py --only h4_631g_sa2   # GPU
```

About 75 s on the gfx1200 (HIP), both schemes.

## The noise floor

σ_g per quantity is measured: the reference geometry re-converged from the
orbitals of the ±h points of the first two columns (four restarts), and the
reference and those four points solved again with the Z-vectors at 3e-11
rather than 3e-12, each gradient against its partner's; σ_g is the RMS over
components and the nine pairs (`noise.pairs`, `noise.differences`). The
floor is 20 σ_g (gain(h/2) + gain(h/4)), gain(s) the stencil's noise gain
`stencil.noise_gain`, as `../nevpt2_hessian/`'s. Measured σ_g: 2.8e-13 to
3.0e-12 Eh/bohr.

**On a symmetry-adapted case** (#1093's `h2nb_sto3g_sa2_hessian`) σ_g is
the larger of the restarts' and the one the point group's zeros imply. An
element H[i][j] whose coordinates belong to different irreps is zero by
symmetry, so each rung's value there is pure stencil noise, σ_g gain(s); the
RMS of value / gain(s) over those elements and the three rungs is σ_g
measured a second way (`noise.sigma_forbidden`, beside
`noise.sigma_restarts`). The restarts at R0 miss noise that the displaced
points carry: on that case SC's read 4.2e-15 against the zeros' 4.6e-14.
Its restarts start from the first two columns that are not a rigid
translation or rotation (a restart from a rigidly moved start re-converges
to the reference's own orbitals); on a Cartesian case those are the first
two columns, as before.

Every point is polished to CASSCF ‖g‖ < 3e-12 by the sampler; the rerun the
gradient is taken over starts there and stops at its own ‖g‖ (worst 5.0e-11
and 1.0e-10), which the restarts' spread includes.

## What each point is

The driver's (`tools/qd_nevpt2_hessian/main.cpp`):

- **The chain and its guard**: `tools/qd_nevpt2_gradient`'s `ChainSampler`,
  polished: the DF-SA-CASSCF re-converged from the reference's orbitals,
  each root matched to the reference's by `mcscf::state_overlaps`, the
  active space's overlap and each matched root's within 1e-3 of 1, PC's cut
  the reference's at every coupling; each root's energy within 2.0 |dR|² of
  its first-order prediction from `mcscf::df_gradient`; each H_eff
  eigenstate matched to the reference's by the overlap of u_K, never by
  rank, within 1e-3 of 1.
- **The gradients**: the point's chain rerun from its converged orbitals,
  its roots matched again, and one several-W `qd_*_gradient` at W^K = u_K
  u_K^T for each state (so the state is tracked) and W^P + W^Q for each pair,
  in the reference's root basis carried into the rerun's, Z-vectors to
  3e-12.

Worst over every point (`guard`): prediction 0.59 |dR|², active space
4.1e-5, CI 5.9e-4, u_K 5.2e-7, rerun energies 3.3e-12 Eh off the sampler's.
The reference energies match `../qd_nevpt2/`'s goldens to 1.2e-11
(`reference.agreement`).

## Keys

| key | meaning |
|---|---|
| `provenance` | the generator, the driver, the source case. No date or host |
| `basis`, `aux_basis` | file and sha256 |
| `atoms` (bohr), `ncas`, `nelecas`, `active`, `weights` | the case; `active` null for the RHF's frontier orbitals |
| `schemes`, `pairs` | what is differenced |
| `reference` | at R0: CASSCF energies and gaps, ‖g‖, inactive gaps, PC's cut per coupling, per scheme H_eff (n × n, root basis), its eigenvalues, gaps and eigenvectors (state K at [K n, (K + 1) n)); `agreement` against `../qd_nevpt2/` |
| `steps` | per scheme `h`, the `ladder` h, h/2, h/4, the eight `displacements` |
| `stencil` | `offsets` and `weights` |
| `coordinates` | `labels`, `vectors` (row j = q_j, Cartesian; the identity) |
| `tolerances` | the ratio window, the floor factor, the Z-vector tolerances, the noise's columns |
| `guard` | the worst guard values above, per scheme the u_K miss and the Z-vectors' true residuals |
| `sc`, `pc` | per scheme `states` (one per state, `state`) and `pairs` (one per pair, `pair`), each below |

Per quantity (H[i][j] = q_iᵀ ∂G/∂q_j):

| key | meaning |
|---|---|
| `gradient` | G at R0, n_atom × 3 flat |
| `hessian` | the finest step's, symmetrized |
| `bar` | per element, the larger of H[i][j]'s and H[j][i]'s `column_bar`: \|D(h/2) − D(h/4)\|, or the floor where that is below it. **The tolerance an analytic Hessian is held to** |
| `ladder` | D(h), D(h/2), D(h/4), unsymmetrized |
| `ratios` | (D(h) − D(h/2)) / (D(h/2) − D(h/4)); null below the floor |
| `asymmetry` | \|D(h/4) − D(h/4)ᵀ\| |
| `noise` | σ_g, the floor, the pairs and their differences |
| `points` | per column, the gradient (in the coordinates) at each displacement: the ladder recomputes from these |

`test/pytest/test_nevpt2_hessian.py` recomputes every ladder, floor, bar and
verdict from these, and checks symmetry, the pair sums and the guard.

## Regenerating

From a build of the tree, on a GPU:

```sh
tools/nevpt2_hessian/qd_nevpt2_hessian.py [--only CASE ...] \
    [--points DIR --keep]
```

About 6 minutes per ladder (106 points, both schemes' gradients) on the
RTX 3080 (CUDA), so 18 minutes for the first two cases; `_w424`'s SC ladder
took 4 minutes on the gfx1200 (HIP). `qd_weight_search.py` (the `tools`
group, no GPU) writes `qd_weight_search.json` first, which `_w424`'s
agreement reads; `--points DIR` alone re-judges kept
points with no GPU. A miss prints `STUCK` with the numbers and writes nothing
for that case. Not byte for byte: the GPU sums reassociate run to run, within
the noise above.

## The SA3 crossing (#1059)

Ethylene's DF-SA3-CASSCF(2,2) (every singlet of the CAS, equal weights),
cc-pVDZ / cc-pVDZ-JKFIT, at the minimum-energy conical intersection of QD
states 0 and 1: one case per scheme, each at its own scheme's MECI. At a
cone E_0 and E_1 have no Hessian (`docs/nevpt2/nevpt2.tex`, Eq. qd-pair and
Eq. qd-lowdin), so each case ladders only what is smooth there while the
third state stays gapped:

- **the pair sum** G(W^0 + W^1), the projector onto the pair's tracked
  eigenspace: Eq. qd-pair's left side (`pairs`, (0,1));
- **the seam** G(W) at a fixed W = ½ s (V_0 V_1ᵀ + V_1 V_0ᵀ) over an
  orthonormal basis V of the pair's eigenspace, a non-projector W whose
  fixed-W Hessian is Eq. qd-lowdin's first term (`seams`, (0,1)), recorded
  as laddered (below);
- **the third state**, W^2 = u_2 u_2ᵀ, tracked by u_2 (`states`, 2).

| case | scheme, h (bohr) | MECI: evaluations, gap (Eh) | third state above the pair | bar, max (Eh/bohr²): pair, seam, state 2 | worst asymmetry |
|---|---|---|---|---|---|
| `ethylene_ccpvdz_sa3_sc_meci` | SC, 3.125e-4 | 97, 7.9e-8 | 0.1813 | 5.0e-7, 4.8e-8, 2.9e-7 | 1.9e-9 |
| `ethylene_ccpvdz_sa3_pc_meci` | PC, 2.5e-3 | 83, 8.1e-8 | 0.1813 | 6.2e-8, 5.6e-8, 3.6e-8 | 3.0e-10 |

**The MECIs** are `ethylene_meci.json`, written first by
`tools/nevpt2_hessian/qd_meci.py`: the tree's `[optimize]` MECI mode
(`src/pipeline/optimize/README.md`, "The QD-NEVPT2 MECI") through the
`free-breakfast` binary, `examples/qd_meci.toml`'s config at each `variant`,
from `examples/ethylene.xyz`. Each is converged by the step's own five
end-point criteria (`max_gap` 1e-7 and Opt=Tight's four), which the record
holds per variant with the last evaluation's values; the final geometry is
`final.xyz`'s, in bohr. The third state's margin there is 0.1813 Eh, far
above `mcscf::kRootGapFloor` (1e-8); over every laddered point its gap from
the pair stays above 0.180 Eh (`guard.<scheme>_third_state_gap`), and the
generator refuses a case where it does not. The CASSCF roots under the
pair are 0.029 Eh apart. The driver's fresh reference point reproduces
`[optimize]`'s last QD energies to 1.4e-13 (SC) and 4.7e-13 (PC)
(`reference.agreement`); its pair gap is 7.9e-8 and 8.1e-8
(`reference.<scheme>.pair_gap`).

**Tracking the pair by its eigenspace.** At a crossing u_0 and u_1 are not
defined one by one, and a step of any size turns them inside their
eigenspace: over the laddered points the better of their overlaps with R_0
or R_1 falls to 0.77 (SC) and 0.73 (PC) (`guard.<scheme>_pair_rotation`, at
most 1 − 1/√2), which a u_K match would refuse. The driver's `crossing`
instead matches the two-state space (`tracked_pair`: the two displaced
states of largest projection onto span{R_0, R_1}, and 1 − σ_min of their
2 × 2 overlap, the larger principal angle's cosine), within 1e-3 like
every other match: its worst miss is 2.4e-10 (SC) and 1.4e-8 (PC)
(`guard.<scheme>_pair_miss`), the third state's own 1 − |overlap| to
1e-17, as it must be over three states. Those two values were re-judged
at the recorded points (#1112): the miss first read 1.1e-8 and 1.7e-8,
the rounding floor of 1 − σ_min taken from M's own σ², and nothing
else in the records changed.

**The step**: SC at #953's 3.125e-4, PC at 2.5e-3; both settled on the first
run. Every element of every ladder settles inside its floor (σ_g 1.3e-13 to
1.4e-12); no ratio is judged, every bar is its floor, and every Hessian is
symmetric to its bar. The largest coarse rung difference is the SC seam's,
2.8e-7 against its floor of 4.8e-8 (the rule allows 19.2 floors). Guard,
worst over every point: prediction 0.34 |dR|², active space 1.3e-5, CI
3.1e-4, rerun energies 2.7e-13 Eh off the sampler's. About 2.3 minutes per
case on the RTX 3080 (154 points).

**The seam's W, and its gauge** (#1098). Eq. qd-lowdin's first term is
term (i) at W = ½(V_0 V_1ᵀ + V_1 V_0ᵀ) for an orthonormal basis V of the
pair's eigenspace, and its elements are V's. The eigensolver's R_0, R_1 are
not a stable V: at a gap of 8e-8 each rerun returns them turned inside the
eigenspace (1.9e-7 rad SC, 5.8e-7 PC, measured on #1091's regeneration) and
signed afresh, and with W = ½(R_0 R_1ᵀ + R_1 R_0ᵀ) that rerun missed SC's
seam with its sign flipped and PC's by 9.19 bars. The driver's W depends on
the eigenspace's projector alone: V the Löwdin-orthonormalized projections
of CASSCF roots r_0 < r_1 onto span{R_0, R_1} (the pair of roots whose
2 × 2 block of [R_0 R_1] has the largest |det|),
`hessian::df_qd_sc_nevpt2_crossing`'s basis of the eigenspace. The CASSCF
roots' own phases still flip V_0 or V_1, so W is multiplied by s, the sign
of the reference H_eff[r_0][r_1], which carries the same two phases
(|H_eff[0][1]| is 1.1e-6 SC and 2.3e-6 PC; the driver refuses one below
1e-10): W = ½ s (V_0 V_1ᵀ + V_1 V_0ᵀ) is one operator whatever either
solve's phases. Each seam entry records the W it was laddered at (`w`,
n × n row-major in the record's root basis), `roots` (r_0, r_1) and
`sign`; a consumer carries `w` into its own root phases, as it carries
H_eff. Here V is within 2e-3 of CASSCF roots 0 and 1, so W is about
½ s (e_0 e_1ᵀ + e_1 e_0ᵀ).

A second run of the generator over both cases reproduced every element of
every ladder within 0.029 of the committed bars (the seams 0.029 SC and
0.002 PC; the pair sums and the third state 0.003 or less). Its SC
reference came back with CASSCF root 1's phase flipped: its W read the
opposite sign in its own basis, and carried into the committed one matched
`w` to 7e-14.

The keys are the Keys table's, plus `crossing` (the tracked pair), `seams`,
`meci` (the record, the evaluations, the criteria and the final values from
`ethylene_meci.json`), per scheme `seams` (one per seam, `seam`, as a pair's
entry, with its `w`, `roots` and `sign`), `reference.<scheme>.pair_gap`,
`third_state_gap` and `seams` (the driver's W per seam), and in `guard`
`<scheme>_pair_miss`, `<scheme>_pair_rotation` and `<scheme>_third_state_gap`.
`ethylene_meci.json` holds per variant the `config`, `start`, `steps`,
`criteria` (value, bar, met), `final` (the head's energies and gap, the last
evaluation's margins, gaps, sensitivities and convergence, and g_P, g_Q,
h_PQ as the step wrote them), `system` (what the crossing case reads) and
every `evaluations` entry's energies, gap, four norms and margin.

```sh
tools/nevpt2_hessian/qd_meci.py
tools/nevpt2_hessian/qd_nevpt2_hessian.py \
    --only ethylene_ccpvdz_sa3_sc_meci ethylene_ccpvdz_sa3_pc_meci
```

## The C2v A1/B2 coupling case (#1058)

`h2nb_sto3g_sa2.json` and `h2nb_sto3g_sa2.analysis.json`: the check of
`docs/nevpt2/nevpt2.tex`'s symmetric geometries (Table qdhesschecks, "a C2v
SA2 with A1/B2 roots"). There, a model space whose roots have different
irreps that a vibration couples can lose its Hessian in the coupling itself,
one order before the diagonal: SC through a tuple empty for one ket and
populated for the other (Eq. qd-scfail), PC through a null mode of a metric
that the coordinate lifts (Eq. qd-pcfail). This case records where each
variant is expected to be refused, for #959, #962 and #964–#966.

**The case.** Aminoborylene H₂N–B, planar in yz with the C2 axis z: B–N
1.36 Å, N–H 1.00 Å, angle BNH 115° (`atoms`, bohr). STO-3G / cc-pVDZ-JKFIT,
all-electron, CAS(4,3) over the RHF's frontier π (b1), σ (a1) and empty
in-plane p (b2), two roots at equal weights: 1A1 and 1B2, 0.197 Eh apart
(SC's H_eff 0.190). The vibrations are 3a1 + b1 + 2b2, and the two b2 ones
couple the roots. Every inactive pair is at least 0.15 Eh apart (core 0.27,
virtual 0.15). In cc-pVDZ (explored at angle 123°) two virtuals of A1 and
B2 are 8.7e-4 Eh apart, and SC's coupling curves on that scale: C₀₁ / s grows
from 0.085 to 0.200 Eh/bohr between s = 2.5e-3 and 1.6e-4. The tree's reference point matches the
analysis's PySCF one to 6.7e-13 (CASSCF), 5.1e-13 (SC's H_eff) and 4.0e-13 (PC's)
(`reference.agreement`).

**The coordinates** (`coordinates`) are an orthonormal C2v-adapted basis of
the 12 Cartesian displacements, irrep by irrep: the translations and
rotations first (T_z; R_z; T_x, R_y; T_y, R_x), then the vibrations ("A1 1"
… "B2 2"). `qd_nevpt2_hessian.symmetry_coordinates` builds them. A
pipeline `[hessian]` takes them as its `coordinates` rows, so a block is
named by its labels. The coupled pair is `coupled`, B2 1 and B2 2.

**The ladders.** C₀₁ = H_eff[0][1] (symmetrized, in the reference's root
basis, signs carried) at every point of the fourth-order stencil at h, h/2
and h/4 along B2 1, B2 2 and u = (B2 1 + B2 2)/√2, through the driver's
`"gradients": false` mode: the same chain and guard as the Hessians, no
gradient. Each first-derivative ladder is judged as a Hessian element is,
over σ measured from twelve re-solves (each positive displacement started
from its negative image's orbitals). The bilinearity of the pair is
D(u) − (D(B2 1) + D(B2 2))/√2 at the finest rung, against u's bar plus
(bar(B2 1) + bar(B2 2))/√2.

| scheme, h | σ | floor (every bar) | D(B2 1) | D(B2 2) | D(u) | bilinearity residual / allowed |
|---|---|---|---|---|---|---|
| SC, 4e-5 | 1.2e-15 | 3.4e-9 | −3.900612e-3 | 2.269345e-3 | −1.153480e-3 | 1.7e-10 / 8.2e-9 |

Every rung difference is under 2.0e-10, below the floor, so no ratio is
judged.

**SC's step.** At h = 2.5e-3 the ladder along B2 1 reads a ratio of −1.6
(|D| 3.90052e-3, 3.90049e-3, 3.90051e-3; differences 3.6e-8 and −2.3e-8).
Along that ray |C₀₁| / s drifts by 1.3e-7 Eh/bohr between s = 1.25e-3 and
2e-5, and follows its s² term only below about 1e-4. That is not the norm
guard: in a diagnostic build with the guard at 1e-16 instead of 1e-14, the
same points moved by under 1e-14 Eh/bohr. So h = 4e-5 (points 1e-5 to 8e-5),
where the CI guard's worst miss is 6.7e-8 of its 1e-3.

**PC is not laddered.** Along either b2 vibration the lifted modes below
pass PC's cut (1e-13 of the largest) once s exceeds about 2e-5, so the
sampler refuses the first displaced point as a rank change of the cut
(`pc_probe`: the reference, and +h along B2 1). The computed PC C₀₁ has a
jump there, not a smooth surface. Its expected refusals come from the
analysis alone, as the issue asks.

### The expected refusals

`expected_refusals`, per variant, a list of `{states, coordinates}` blocks:

- **SC: none.** C₀₁'s ladder is bilinear on (B2 1, B2 2) to 1.7e-10 of
  8.2e-9. The analysis agrees: every tuple empty for one ket and populated
  for the other is in Sij, a one-function class at CAS(4,3)'s two active
  holes. Its quotient is the one function's Dyall energy η whatever the
  direction (ρ_t is constant, so Eq. qd-scfail's term is linear), and no
  tuple of a class of several functions is one-sided (`analysis.sc.rho_defects`
  is empty). Nothing in the totally symmetric block is expected either.
- **PC: (0, 1) on (B2 1, B2 1), (B2 1, B2 2) and (B2 2, B2 2).** 1B2's
  metrics have modes cut at R0 that both b2 vibrations lift at second order,
  each with a first-order overlap to 1A1 (`analysis.pc.lifts`, at s = 1e-3
  and 2e-3, every rate constant to 2%):

  | class (ket 1B2) | cut at R0 | λ / σ_max / s², B2 1 / B2 2 | max_t \|⟨φ\|V\|1A1⟩\| | max_t \|⟨φ\|V\|1B2⟩\| / s |
  |---|---|---|---|---|
  | Sijr | 1 | 4.25e-4 / 4.82e-4 | 1.6e-3 | 7.3e-4 / 7.7e-4 |
  | Srs | 3 | 4.25e-4 / 4.82e-4 | 0.149 | 1.5e-3 / 1.6e-3 |
  | Sir | 10 | 1.89e-4 / 2.14e-4 | 3.3e-3 | 4.0e-3 / 4.3e-3 |

  These are the nevpt2.tex picture exactly: λ = O(h²), ⟨φ|V|M⟩ = O(1),
  ⟨φ|V|N⟩ = O(h). Nothing lifts along T_y or R_x, and 1A1's cut modes do
  not lift. A block is listed when both of its coordinates lift
  (`pc_expected`). A translation or rotation carries no coupling at first
  order, so a block with one never is.

### The one-sided tuples, and the guarded one #962 reads

`analysis.sc.one_sided`: 13 tuples, all Sij (t = i n_core + j over the
SA-canonical core, PySCF's order). Six are empty for 1A1 and seven for 1B2,
each at R0 below 2e-17 for its ket and 1.2e-7 to 2.9e-4 for the other root.
Both b2 vibrations populate each one at first order (N / s² 1.6e-10 to
1.3e-4),
with a transition tangent |Ṫ^t₀₁| up to 1.1e-4 (B2 1) and 2.0e-4 (B2 2), for
tuple 18 = (3, 3), empty for 1B2. Those tangents are linear in s to 2%.

But a one-function class keeps every tuple at η, so the tree's guard drops
none of them, and `ScEffectiveHamiltonianTangent::empty` lists none.
`guarded_tuple` (`analysis.sc.chosen`) is therefore **null**: this case has
no norm-guarded tuple empty for one ket and populated for the other. On the
same molecule, CAS(2,3) over σ, p and π* has them in Srsi, a class of three
functions. There the tree's SA-CASSCF and PySCF's converge to different
stationary points (roots 1.1e-3 Eh apart), so it is not written. Its
Srsi tuples are linear anyway: see "The search for an SC ρ refusal" below.

### Keys (`h2nb_sto3g_sa2.json`)

| key | meaning |
|---|---|
| `provenance`, `basis`, `aux_basis` | as above |
| `atoms` (bohr), `ncas`, `nelecas`, `active`, `weights`, `irreps` | the case; `active` null for the frontier orbitals |
| `schemes`, `state_pairs`, `coupled` | what is laddered |
| `coordinates` | `labels`, `irreps`, `vectors` (row j = q_j, Cartesian, orthonormal) |
| `reference` | at R0: CASSCF energies and gaps, ‖g‖, inactive gaps; SC's H_eff, eigenvalues, gaps and eigenvectors; PC's H_eff and its cut per coupling (`dropped`, n² rows of the eight classes); `agreement` against the analysis |
| `steps`, `stencil`, `tolerances` | as above |
| `guard` | per scheme, the worst guard values over every point |
| `pc_probe` | the driver over both schemes at R0 and at +h along the first coupled coordinate: exit code and last message |
| `sc` | per state pair: `directions` (each its `vector`, C at each displacement `values`, the `ladder`, `ratio`, `settled`, `bar`), `bilinearity` (per rung `residuals`, the finest `residual`, `allowed`, `fails`), `noise` (σ, floor, the re-solve pairs and differences) |
| `expected_refusals` | per variant the expected-refused `{states, coordinates}` blocks |
| `guarded_tuple` | the analysis's `sc.chosen` |

C₀₁'s sign follows the reference roots' phases, which a rerun may flip; the
magnitudes, bars and verdicts do not depend on it.

### Keys (`h2nb_sto3g_sa2.analysis.json`)

| key | meaning |
|---|---|
| `coordinates`, `steps` | the coordinates (the same as the record's) and the displacements read |
| `tolerances` | the guard, the populated norm, the lifted fraction, the agreement of the rates at s and 2s, the overlap floor, ρ's spread |
| `reference` | PySCF's CASSCF energies, SC's and PC's H_eff (symmetrized, flat), the SA-canonical core and virtual orbital energies |
| `sc.guarded` | per class and ket, how many tuples the guard drops at R0 |
| `sc.one_sided` | each tuple empty for `ket` and populated for `bra`: class, flat `tuple`, its `index` and `shape`, `one_function`, both norms, per b2 coordinate `tangent` \|T\|/s and `population` N/s² at both steps, per coupled coordinate `rho` (H/N), `rho_spread`, `rho_defect`, `first_order` |
| `sc.chosen`, `sc.rho_defects` | the guarded tuple of largest Ṫ in a class of several functions (or null), and the one-sided tuples with a ρ defect |
| `pc.dropped`, `pc.lifts`, `pc.lifted_coordinates` | the cut modes per ket and class at R0; per coordinate, ket and class the lifted modes (`modes`, `lift` λ/σ_max/s², `bra_overlap`, `ket_overlap_rate`, and the three verdicts); the coordinates along which a mode lifts with a nonzero overlap |

`test/pytest/test_nevpt2_hessian.py` recomputes every ladder, the floor, the
verdicts, the bilinearity and both variants' expected refusals from these,
and (slow, `tools` group) reruns the analysis's reference point.

### Regenerating

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/nevpt2_hessian/qd_coupling_analysis.py     # about a minute, no GPU
tools/nevpt2_hessian/qd_nevpt2_hessian.py --only h2nb_sto3g_sa2   # GPU, seconds
```

The analysis runs on one BLAS thread and is byte-identical on a rerun. The
ladders reproduce to their noise.

### The search for an SC ρ refusal (#1064): none found

`h2nb_sto3g_sa2`'s SC expected-refusal set is empty, so the "exactly the
expected-refusal blocks" checks of #959, #962 and #965 show only that
nothing is refused falsely. #1064 looked for a physical case where SC's
off-diagonal failure (`docs/nevpt2/nevpt2.tex`, Eq. qd-scfail) should fire.
It found none. The constructed-input refusal test of #962
(`Nevpt2QdHvpEmpty` in `test/gtest/nevpt2_qd_hvp_gpu.cpp`) remains the
only coverage of that refusal.

**What a case needs.** All three of:

1. A tuple empty for ket N by symmetry and populated for bra M, in a class
   of several functions. Sij at two active holes and Srs at two active
   electrons are one-function classes: their quotient is η in every
   direction, so C_MN is linear.
2. A Rayleigh quotient ρ_t(q) of ∂_qΦ^N_t that differs between two
   coordinates of the coupling irrep. A planar XY₂ has only one b2
   vibration, so it cannot qualify. With the roles swapped (empty bra) the
   term is linear.
3. An SA-CASSCF that stays at the symmetric point, with inactive pairs
   apart.

**The screen.** Each candidate is a `COUPLING_CASES`-shaped entry: frontier
active space, two roots at equal weights, cc-pVDZ-JKFIT. It ran through
`qd_coupling_analysis.py`'s `at_reference`, with PySCF's polished
SA-CASSCF from the C2v RHF. Recorded per candidate: the polish's least
orbital-Hessian eigenvalue, which must clear `POLISH_CURVATURE` = 1e-3, and
the one-sided tuples per class. A candidate with a one-sided tuple in a
class of several functions then ran the whole `analyse`. The reported ρ
is H/N of that tuple at s = 1e-3 and 2e-3 along B2 1 and B2 2,
extrapolated to R0 as ρ(s) + (ρ(s) − ρ(2s))/3.

H₂N–B is `H2NB`. H₂N–Al copies its shape with Al–N 1.75 Å. The b2
vibrations are B2 1 and B2 2. Every one-sided tuple of a class of several
functions is empty for 1B2 (ket 1).

| candidate (STO-3G but where noted) | active (irreps) | one-sided tuples | least orbital-Hessian eigenvalue; inactive gaps (core, virtual) | ρ at R0, B2 1 / B2 2 (relative difference) | verdict |
|---|---|---|---|---|---|
| H₂NB CAS(2,2) | a1, b2 | Srs 10, Sij 24 | 0.070; 0.072, 0.147 | — | one-function classes only |
| H₂NB CAS(2,3) | a1, b2, b1 | Srsi 4, Srs 9 | 0.020; 0.071, 0.146 | 0.3039808960 / 0.3039808960 (1.7e-10) | ρ constant: linear (Ṫ 4.9e-7, 1.2e-6) |
| H₂NB CAS(2,4) | a1, b2, b1, a1 | Srsi 2, Srs 4 | 7.6e-3; 0.071, 0.185 | 0.3087984664 / 0.3087984664 (2.8e-10) | ρ constant: linear (Ṫ 2.0e-7, 5.5e-7) |
| H₂NB CAS(2,5) | a1, b2, b1, a1, b2 | Srsi 1, Srs 1 | **5.9e-4**; 0.071, one virtual | 0.3088984136 / 0.3088984136 (1.0e-10) | soft orbital mode, and ρ constant |
| H₂NB CAS(4,4) | b1, a1, b2, b1 | none | 2.3e-3; 0.27, 0.19 | — | no one-sided tuple |
| H₂NB CAS(4,5) | b1, a1, b2, b1, a1 | — | polish stalls at \|g\| 7.6e-5; CASCI 1.8e-7 off it | — | no stationary point reached |
| H₂NB CAS(6,5) | a1, b1, a1, b2, b1 | none | 0.021; 0.41, 0.16 | — | no one-sided tuple |
| H₂NB 6-31G CAS(4,3) | b1, a1, b2 | Sij 11 | 3.9e-3; 0.27, **3.8e-3** | — | one-function class only; virtuals near-degenerate |
| H₂NAl CAS(2,3) | a1, b2, b1 | Srsi 7, Srs 9 | 3.7e-3; **2.5e-4**, 0.10 | 0.1417995473 / 0.1417995472 (4.1e-10) | ρ constant; core near-degenerate |
| H₂NAl CAS(4,4) | b1, a1, b2, b1 | none | 0.010; 6.8e-4, 0.22 | — | no one-sided tuple |

**Why it fails.** Every candidate whose one-sided tuples lie in a class of
several functions (Srsi each time) falls under nevpt2.tex's
parallel-derivatives exception. At s = 1e-3, ρ already agrees between B2 1
and B2 2 to 6e-8 to 1.2e-7, and that difference is its s² term. Extrapolated to
R0, ρ agrees to 1e-10 to 4e-10. Then C_MN is bilinear, like
`h2nb_sto3g_sa2`'s Sij, and nothing should be refused. Adding a second a1
active orbital (CAS(2,4)) or a second b2 one (CAS(2,5)) does not change
that. With four or more active electrons the one-sided tuples go away
entirely. The tangents are also small, 5.7e-8 to 1.2e-6. The SA-CASSCF
stability of the CAS(2,3) point (the bistability #962 met) therefore
decides nothing and was not rechecked. The orbital Hessian above is
PySCF's at fixed CI, so a positive value does not rule out an orbital-CI
saddle.

#1058's earlier rejects still stand: H₂CO (A1/A2, no a2 vibration),
vinylidene and cis-diazene (relax below C2v), H₄ CAS(2,2) (the B2 root's
`df_gradient` `pcg_solve` failure), and H₂NB cc-pVDZ (two virtuals of A1
and B2 8.7e-4 Eh apart).

## The C2v A1/B2 whole Hessians (#1093)

`h2nb_sto3g_sa2_hessian.json`: the state Hessians of the coupling case
above (the same system, `coordinates` and, to 1.7e-13, reference point,
`reference.agreement`), laddered as the C1 cases' are, for #965's and
#966's check on the symmetric geometry. Each scheme skips what has no smooth
surface to ladder, and records it (`not_laddered`):

| scheme, h (bohr) | laddered | not laddered | σ_g (restarts, zeros) | floor (every bar) | worst asymmetry |
|---|---|---|---|---|---|
| SC, 3.125e-4 | state 0; T_z, A1 1–3, R_z, T_x, R_y, T_y, R_x | B1 1, B2 1, B2 2; state 1, the pair | 4.2e-15, 4.6e-14 | 1.7e-8 | 4.3e-9 |
| PC, 2.5e-3 | states 0, 1 and the pair; every coordinate but B2 1, B2 2 | B2 1, B2 2 | 2.0e-13 to 5.9e-13, 1.0e-13 to 2.4e-13 | 8.9e-9 to 2.7e-8 | 2.6e-9 |

Every element of every ladder settles inside its floor on the first run at
these steps: the largest finer difference is 0.72 floors (SC), the largest
coarser one 0.30. No ratio is judged and every bar is its floor. The PC
pair is its two states' sum within the three bars. Guard, worst over the
172 points: prediction 0.49 |dR|², active space 8.8e-6, CI 3.2e-4.

**PC skips the b2 vibrations.** Its sampler refuses a point along B2 1
from 2e-5 bohr and along B2 2 from 1.5e-5, a rank change of the cut as the
lifted modes above pass it (accepted at 1.5e-5 and 1e-5). Every block the
two make is an expected refusal of the analytic PC Hessian. Along every other
coordinate the sampler took ±5e-3.

**SC skips the three vibrations that lower the symmetry**, which populate
perturbers empty by symmetry at second order (`docs/nevpt2/nevpt2.tex`,
Sec. hess-refusals, "SC: empty perturbers"). Along them the semi-numerical
surface crosses over on a scale of 1e-5 to 1e-4 bohr, and no step settles:

- B1 1: G·q / s for state 0 reads −0.045571 at s = 1e-5, −0.045964 at 2e-5,
  −0.046022 at 4e-5 and −0.046032 at 8e-5, then −0.0460337 at 1.56e-4,
  −0.0460342 at 3.125e-4 and −0.0460342 at 6.25e-4. (B1 1, B1 1) reads
  ratio 0.14 at h = 4e-5 (rungs differ by 7.3e-5 and 5.1e-4) and 0.26 at
  3.125e-4 (5.9e-7 and 2.3e-6).
- B2 1, B2 2: at 3.125e-4, state 1's (B2 1, B2 1) reads ratio 2.7 (2.4e-5
  and 9.0e-6), (B2 1, B2 2) 12.7 and (B2 2, B2 2) 10.3, and state 0's
  (B2 1, B2 1) 1.8. Their only check is C₀₁'s ladders above.

State 1 and the pair are not laddered for SC. The analytic SC Hessian
refuses state 1 by its conditioning rule (root 1's Si, #1102), and its
ladder along B1 1 reads a ratio of 8.3 at 3.125e-4.

The keys are the Keys table's, with `coordinates` the coupling case's
(`labels`, `irreps`, `vectors`), `steps.<scheme>.coordinates` the labels
laddered (each quantity's matrices are over those, in that order),
`not_laddered` per scheme (`coordinates`, `states`, `pairs`), and in each
`noise` `sigma_restarts` and `sigma_forbidden`.

```sh
tools/nevpt2_hessian/qd_nevpt2_hessian.py --only h2nb_sto3g_sa2_hessian
```

About 2 minutes on the gfx1200 (HIP), both schemes. It reads
`h2nb_sto3g_sa2.json`'s reference point, so that case is written first.
