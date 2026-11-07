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

The system, basis, active space and weights of each are
`../qd_nevpt2/qd_nevpt2.json`'s; cc-pVDZ / cc-pVDZ-JKFIT, CAS(4,3), three
roots, all-electron, Cartesian coordinates (C1: every block defined).

| case | weights | scheme, h (bohr) | bar, max (Eh/bohr²) | floor | worst asymmetry | inactive gaps (core, virtual) | H_eff gaps |
|---|---|---|---|---|---|---|---|
| `h2co_skew_ccpvdz_sa3` | equal | SC, 3.125e-4 | 5.0e-7 | 1.1e-7 – 5.0e-7 | 8.5e-9 | 0.072, 7.6e-3 | 0.1545, 0.1986 |
| | | PC, 2.5e-3 | 6.4e-8 | 1.3e-8 – 6.4e-8 | 1.3e-9 | | 0.1546, 0.1985 |
| `h2co_skew_ccpvdz_sa3_weighted` | 0.4/0.4/0.2 | PC, 2.5e-3 | 1.4e-7 | 2.4e-8 – 1.4e-7 | 2.2e-8 | 2.6e-3, 3.6e-3 | 0.1566, 0.1907 |

Per scheme, every one of the three states' Hessians (`states`) and every
pair's (`pairs`, (0,1), (0,2), (1,2)). Every element of every ladder settles:
each finer difference D(h/2) − D(h/4) lies below the measured floor, and
each coarser one D(h) − D(h/2) below 19.2 floors (the largest is 1.0e-7, SC
state 0), so no ratio is judged and every bar is its floor. Every Hessian is
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
only, and SC's unequal-weight reference is still to find (#953's
"Remaining").

## The noise floor

σ_g per quantity is measured: the reference geometry re-converged from the
orbitals of the ±h points of the first two columns (four restarts), and the
reference and those four points solved again with the Z-vectors at 3e-11
rather than 3e-12, each gradient against its partner's; σ_g is the RMS over
components and the nine pairs (`noise.pairs`, `noise.differences`). The
floor is 20 σ_g (gain(h/2) + gain(h/4)), gain(s) the stencil's noise gain
`stencil.noise_gain`, as `../nevpt2_hessian/`'s. Measured σ_g: 2.8e-13 to
3.0e-12 Eh/bohr.

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
RTX 3080 (CUDA), so 18 minutes in all; `--points DIR` alone re-judges kept
points with no GPU. A miss prints `STUCK` with the numbers and writes nothing
for that case. Not byte for byte: the GPU sums reassociate run to run, within
the noise above.
