# qd_nevpt2_gradient golden data

Numerical nuclear gradients of this tree's SC- and PC-QD-NEVPT2 energies
E_K (every eigenstate K of the symmetrized H_eff), and of each DF-SA-CASSCF
root's energy, written by `free-breakfast-qd-nevpt2-gradient`
(`tools/qd_nevpt2_gradient/main.cpp`, over the chain, guard and Richardson of
its module library `free_breakfast.tools.qd_nevpt2_gradient`). No file here
is hand-edited. `test/gtest/nevpt2_numerical_heff_gpu.cpp` checks that
library's in-process ∇H_eff (#783) against `h6_631g_sa2` and
`h2co_skew_ccpvdz_sa3`: each g_K = R_Kᵀ(∇H_eff)R_K within twice the committed
error bar of E_K's difference.

No other program has a QD-NEVPT2 energy, and the oracle has to differentiate
the same functional the analytic gradient does: the same B, the core and
virtuals canonical over the state-averaged Fock, and the same PC cut and gap
rule. So the energy differenced is the tree's own. At every geometry the tool
runs the chain `free-breakfast` runs for `[casscf]` + density-fitted `[eri]`
(cc-pVDZ-JKFIT) + `[nevpt2] quasi_degenerate = true`, as one nested
`Pipeline`. The energies at the reference are gated against independent
code already (`../qd_nevpt2/`, whose goldens each case's reference point
matches: `reference.agreement`, below). This data establishes only that a
gradient is the derivative of that energy. The bar a consumer holds an
analytic gradient to is **~1e-6 hartree/bohr**, not the tree's usual 1e-7,
because the oracle is itself a difference. Every number here has an error
bar of at most 1e-7 (below).

## Cases

The geometry, basis, active space and weights of each are
`../qd_nevpt2/qd_nevpt2.json`'s.

| case | roots, weights | SC/PC | steps (bohr) | points | inactive gaps (core, virtual) | H_eff gaps SC (PC) | worst ‖g‖ |
|---|---|---|---|---:|---|---|---|
| `h2co_skew_ccpvdz_sa3` | 3, equal | both | 1e-3, 5e-4, 2.5e-4 | 18 + 72 | 0.072, 7.6e-3 | 0.1545, 0.1986 (0.1546, 0.1985) | 8.4e-11 |
| `h8_ring_sto3g_sa3` | 3, equal | both | 1e-4, 5e-5, 2.5e-5 | 18 | —, 1.8e-3 | 0.1416, 4.7e-3 (0.1400, 6.0e-3) | 5.0e-11 |
| `h6_631g_sa2` | 2, equal | both | 1e-3, 5e-4, 2.5e-4 | 18 | 0.28, 0.023 | 0.2907 (0.2898) | 5.0e-12 |
| `h8_sto3g_sa3` | 3, equal | SC | 5e-4, 2.5e-4, 1.25e-4 | 18 | 0.039, 0.10 | 0.7512, 0.0152 | 6.6e-11 |
| `h2co_ccpvdz_sa2` | 2, equal | both | 1e-3, 5e-4, 2.5e-4 | 18 | 0.020, 8.8e-5 | 0.1581 (0.1582) | 9.2e-11 |
| `h2co_ccpvdz_sa2_weighted` | 2, 0.7/0.3 | both | 1e-3, 5e-4, 2.5e-4 | 18 | 0.012, 2.0e-3 | 0.1589 (0.1587) | 6.1e-11 |
| `h2co_skew_ccpvdz_sa3_weighted` | 3, 0.4/0.4/0.2 | both | 1e-3, 5e-4, 2.5e-4 | 18 | 2.6e-3, 3.6e-3 | 0.1555, 0.2029 (0.1566, 0.1907) | 9.0e-11 |
| `h8_ring12_ccpvdz_cas66_sa2` | 2, equal | both | 4e-4, 2e-4, 1e-4 | 18 + 144 | —, 2.2e-3 | 0.0867 (0.0861) | < 1e-10 |

The inactive gaps are the smallest gap between two canonical core, or two
canonical virtual, orbital energies of the state-averaged Fock at the
reference (— for a core of one orbital); `guard.inactive_gap` has the
smallest seen at any point. A near-degenerate pair is why the two rings
take small steps: SC is not invariant to the pair's rotation, so its
energy curves sharply on the length scale of the gap. `h2co_ccpvdz_sa2`'s
pair is of two irreps and never mixes along its totally symmetric
directions (below), though it comes within 5.5e-7 of degenerate there.
The weighted three-root case's two closest core orbitals are 2.6e-3 apart.
H_eff's gaps are between neighbouring eigenvalues (hartree): `h8_ring`'s
states 1 and 2 are 4.7 mEh apart.

- **Directional** (every case): g · u along three unit vectors u in the 3N
  space (`directions`), drawn from `std::mt19937_64` seeded with `seed`
  (696), translation and rotation about the centroid projected out,
  orthonormalized. Each u is first projected onto the totally symmetric
  displacements of the geometry's symmetry (the operations of D2h about the
  coordinate axes that map the atoms onto themselves). A C1 case has the
  identity alone. The two C2v formaldehyde cases' directions are therefore
  totally symmetric, which loses nothing: their energies are invariant
  under the group, so the gradient is totally symmetric. But a
  symmetry-breaking step mixes canonical virtuals of different irreps,
  which in `h2co_ccpvdz_sa2` are 8.8e-5 hartree apart. SC, not invariant
  to that rotation, curves too sharply along such a step for any step the
  noise allows (`nevpt2_gradient.json`'s symmetric water, in
  `../README.md`). Three random directions catch a wrong component with
  probability 1, at 6 energies each.
- **Full** (`h2co_skew_ccpvdz_sa3`, `h8_ring12_ccpvdz_cas66_sa2`): every
  Cartesian component, 6 × 3N energies: the localizer once a directional
  check fails. Any other case can be run in full by adding it to the
  tool's `kCases` with `full` set.
- **`h8_sto3g_sa3` is SC only**, as #696 lists it. Its PC was refused when
  the issue was written, and it passes since #687 (`../qd_nevpt2/README.md`):
  a later run may add it.
- **`h8_ring12_ccpvdz_cas66_sa2`** is the one case over six active orbitals,
  where the index ranges of the Γ⁽³⁾ and 4-RDM-digest contractions of the
  S⁽⁻¹⁾′_r and S⁽⁺¹⁾′_i classes come apart. Skew H₂CO at CAS(6,6) and (8,6)
  was tried first; every variant's PC is refused by the gap rule.

## What each point is

- **The reference.** RHF (SAD guess), then the DF-SA-CASSCF from the RHF's
  frontier orbitals at the case's weights, to ‖g‖ < 1e-10 and |ΔE| < 1e-13
  (`conv_tol_grad`, `conv_tol`), and QD-NEVPT2 over it. Each root's analytic
  DF-SA-CASSCF gradient by `mcscf::df_gradient` is `reference.casscf_gradient`.
- **A displaced point** re-converges the DF-SA-CASSCF from the reference's
  orbitals, Löwdin-orthonormalized in the displaced overlap (as `[optimize]`
  carries them), at the same weights, bars and fitting set, then the same
  QD-NEVPT2. The NEVPT2 energy is not variational in the orbitals, so a
  residual ‖g‖ puts an error linear in it into E_corr. Hence 1e-10, where
  `mcscf::Options`' default is 1e-6. 1e-11 stalls at displaced
  formaldehyde points near 6e-11.
- **Matching, not ranking.** Each displaced root is matched to the
  reference's by `mcscf::state_overlaps` (the largest |S_IJ|, which must be
  a permutation), and each eigenstate K of H_eff by the overlap of u_K
  carried into the matched root basis. E_K is invariant to the roots' and
  u_K's signs (W_MN and C_MN both pick up s_M s_N), so nothing is signed.

## The continuity guard

#555's (`../README.md`, `nevpt2_gradient.json`), ported. Every displaced
point must, or nothing is written:

1. have converged to the bars above (`guard.gradient_norm`, the largest ‖g‖
   reached anywhere);
2. have each root's energy within 2.0 h² of E_r + g_r · ΔR, g_r the analytic
   gradient (`guard.prediction`, the worst in units of h²);
3. have an active space whose smallest singular value of overlap with the
   reference's, through the AO cross overlap, is within 1e-3 of 1
   (`guard.subspace`, as 1 − σ_min);
4. have each matched root's |CI overlap| within 1e-3 of 1 (`guard.ci`), and
   each matched u_K's (`guard.sc_vectors`, `guard.pc_vectors`);
5. for PC, cut exactly the reference's modes in every coupling's metrics
   (`reference.pc_dropped`, as a multiset over the pairs), so no step
   crosses a rank change.

`guard.inactive_gap` is the smallest gap between two canonical core, or
virtual, orbital energies seen at any point. That is the quantity SC's
gradient divides by, and the reason for the rings' small steps.

## The differences, and the error bar

Per component (direction or Cartesian), family (`casscf` per root, `sc` and
`pc` per H_eff eigenstate) and index:

- `central`: D(h) = (E(+h) − E(−h)) / 2h at each of `steps` (bohr, halving);
- `richardson`: (4 D(h/2) − D(h)) / 3 over the two finest, **the number to
  test against**;
- `error_bar`: |R(fine) − R(coarse)|, the distance from the same Richardson
  step over the two coarser (#555's), **plus `energy_noise` / 2h** at the
  finest step;
- `analytic` (`casscf` only): g_r · u from `mcscf::df_gradient`.

**The energy noise.** The tree's GPU sums are not bit-reproducible (atomics
in `integrals/active_space`'s J and `recipes/einsum`). Each point's CASSCF
stops at its own ‖g‖ below 1e-10, which a root's energy (not variational in
a state average) and E_corr feel to first order. So each energy carries
noise of its own, mostly reproducible per point, that #555's error bar
alone does not see. `energy_noise` per case is set above two measurements:

| case | `energy_noise` | rerun spread | per-point (CASSCF) |
|---|---:|---:|---:|
| `h2co_skew_ccpvdz_sa3` | 1e-11 | 4.3e-12 | 3.2e-12 |
| `h2co_skew_ccpvdz_sa3_weighted` | 1e-11 | 9.0e-12 | 9.3e-13 |
| `h2co_ccpvdz_sa2` | 1e-11 | 9.8e-13 | 4.8e-13 |
| `h2co_ccpvdz_sa2_weighted` | 1e-11 | 1.3e-12 | 1.7e-11 |
| `h8_sto3g_sa3` | 1e-11 | 2.2e-14 | 1.8e-11 |
| `h8_ring12_ccpvdz_cas66_sa2` | 8e-12 | 7.0e-12 | 6.0e-12 |
| `h2o_skew_ccpvdz` (cross-check) | 2e-12 | 1.4e-12 | 3.4e-13 |
| `h6_631g_sa2` | 1e-12 | 1.6e-14 | 9.1e-14 |
| `h8_ring_sto3g_sa3` | 1e-12 | 3.6e-14 | 7.1e-13 |

The rerun spread is the largest difference of any point's energy between
two full runs. The per-point error is the largest |R − analytic| · 2h of the
CASSCF difference at the finest step: at the committed steps for most cases.
Two came from development runs at smaller steps, where the noise is the
larger part of it: `h2co_ccpvdz_sa2_weighted` at 8e-4–2e-4 (4.3e-8 · 4e-4,
reproduced on two runs) and `h8_sto3g_sa3` at 2.5e-4–6.25e-5, whose CASSCF
error bar of 1.4e-7 there is 1.8e-11 over 2h. SC and PC feel a
residual ‖g‖ to first order where a root's energy feels it only through the
state average's non-stationarity: at half the committed steps,
formaldehyde's SC and PC error bars reached 1.2e-7, about 1e-11 over 2h.
So every formaldehyde case, and `h8_sto3g_sa3`, carries 1e-11. A rerun
spread under the per-point error says most of the noise repeats from run
to run, as an error per point should; that is why #555's error bar alone,
a difference of two Richardson values of the same points, does not see it.

**Checks before writing**, every case, family and component, or nothing is
written:

- Richardson against the finest D, within 2e-7 (the finest D's truncation);
- the error bar under 1e-7;
- the ladder ratio (D(4h) − D(2h)) / (D(2h) − D(h)) in (3.2, 4.8) wherever
  |D(2h) − D(h)| is above both 1e-7 and 5 `energy_noise` / h. Below that,
  the noise alone moves the ratio by the window's half width. On the noisier
  formaldehyde cases this leaves few or no components to check
  (`ladder_components`), and the error bar then carries the evidence;
- the full gradients' translational sums within the sum of their
  components' error bars (`translation_over_error_bars`);
- each root's CASSCF Richardson within its error bar of the analytic
  gradient (`analytic_over_error_bar`), which checks the differencing
  machinery itself.

**Measured** (both full runs passed every check): Richardson against the
finest D at most 9.7e-8 (`h2co_skew_ccpvdz_sa3_weighted`'s CASSCF, the
largest truncation); error bars at most 9.4e-8 (that case's PC), 9.3e-8 on
the CAS(6,6) ring, 6.3e-8 or less elsewhere, so every committed number is
inside 1e-7 against the consumers' ~1e-6; CASSCF against analytic within
0.43 of the error bar everywhere (largest 3.0e-8, the CAS(6,6) ring; 3.6e-9
or less on formaldehyde); translational sums within 0.25 of the error bars
(4.5e-8 the largest, the CAS(6,6) ring's SC); the ladder ratio 3.95–4.07
on the 7 components whose rung difference clears its floor (the rest sit
under the noise's floor, where the error bar is the evidence); the guard's worst prediction
0.94 h² (skew H₂CO) against 2.0, active subspace 1.2e-6, CI 2.3e-5, u_K
2.2e-6 (`h8_sto3g_sa3`), all against 1e-3, and every PC cut unchanged at
every point. The reference energies match `../qd_nevpt2/`'s goldens to
1.2e-11 (formaldehyde, polished there), 5.6e-8 (the hydrogen cases, which
`qd_nevpt2_reference.py` converges but does not polish).

## Reproducibility

Not byte for byte: the GPU sums above reassociate run to run. Two full runs
on the RX 9060 XT (HIP) differ by at most 9.0e-12 hartree in any energy and
4.7e-8 hartree/bohr in any Richardson gradient, per case:

| case | energies | Richardson gradients |
|---|---:|---:|
| `h2co_skew_ccpvdz_sa3` | 4.3e-12 | 6.2e-9 |
| `h2co_skew_ccpvdz_sa3_weighted` | 9.0e-12 | 2.5e-8 |
| `h2co_ccpvdz_sa2` | 9.8e-13 | 1.6e-9 |
| `h2co_ccpvdz_sa2_weighted` | 1.3e-12 | 3.7e-9 |
| `h8_ring12_ccpvdz_cas66_sa2` | 7.0e-12 | 4.7e-8 |
| `h2o_skew_ccpvdz` | 1.4e-12 | 5.3e-9 |
| `h8_ring_sto3g_sa3` | 3.6e-14 | 7.3e-10 |
| `h8_sto3g_sa3` | 2.2e-14 | 1.2e-10 |
| `h6_631g_sa2` | 1.6e-14 | 2.1e-11 |

So a rerun reproduces the energies to ~1e-12–1e-11 hartree and the
gradients to ~1e-9–1e-8 (the CAS(6,6) ring, whose step is the smallest of the
cc-pVDZ cases, and the three-root weighted formaldehyde), each inside its
error bar. The runs above are the committed files and a second run of the
same binary; the weighted three-root row, and the H₂O and H₆ rows' check,
add a third, on the tree rebased onto a later main (H₂O 1.3e-12 and 2.1e-9,
H₆ 1.9e-14 and 3.0e-11). The ring and the
cross-check were then rerun once more with their noise terms raised to
cover these spreads (8e-12 from 4e-12, 2e-12 from 1e-12), which is the
committed data for those two.

## The cross-check against PySCF

`single_state_h2o_skew_ccpvdz.json`: `nevpt2_gradient.json`'s
`h2o_skew_ccpvdz` (single-state DF-CASSCF(4,4) over RHF orbitals 2, 3, 5, 6,
its steps, every Cartesian component) through the same chain without
`quasi_degenerate`, each root's own SC- and PC-NEVPT2. Its SC and PC
Richardson gradients must match that file's PySCF- and block2-built ones,
per component, within the two error bars combined. Ours carries its own
noise term (1e-12, over its 9e-13 rerun spread). That file's carries the
noise its README states for its SC and PC energies, about 1e-11 hartree,
over 2h at its finest step (4e-8). `cross_check` holds the largest
difference and its largest ratio to the combined bars. Measured: SC within 5.1e-9 of PySCF's (0.09 of the bars), PC within 4.3e-9 of block2's (0.08); without either noise term the same differences sit at 0.9–2.8 of #555's bars alone, run to run, which is what the noise term is for. The CASSCF
difference is checked against the analytic gradient there, as everywhere.

## Files

One `<case>.json` per case, and `single_state_h2o_skew_ccpvdz.json`:

| Key | Meaning |
|---|---|
| `name`, `basis`, `aux_basis`, `atoms` (bohr), `ncore`, `ncas`, `nelecas`, `active`, `weights` | the case; `active` null for the RHF's frontier orbitals |
| `schemes` | `["sc", "pc"]` or `["sc"]` |
| `steps`, `energy_noise` | above |
| `reference` | at the reference geometry: `casscf_energies`, `sc_energies`, `pc_energies` (H_eff's eigenvalues, ascending; one root: its SC/PC total), `gradient_norm`, `casscf_gradient` (per root, n_atom × 3), `inactive_gaps`, `pc_dropped` (per coupling [I n + J], each class's cut), `sc_heff_gaps`, `pc_heff_gaps` (neighbouring eigenvalues), `sc_heff_vectors`, `pc_heff_vectors` (eigenvector K at [K n, (K + 1) n) over the roots), `agreement` (against `../qd_nevpt2/qd_nevpt2.json`'s, or `../nevpt2_gradient/nevpt2_gradient.json`'s, energies) |
| `seed`, `directions` | the three unit vectors u, 3N each, atom-major |
| `directional`, `full` | per family a list over its roots/states of `central` (per step), `richardson`, `error_bar`, `analytic`; one entry per direction, or n_atom × 3 rows; `checks` (above); `displaced`, per component and step the `[−h, +h]` pair of each point's energies and ‖g‖, so every difference can be redone |
| `guard` | the worst guard values over every point, and the `points` count |
| `cross_check` | the single-state file only |

## Regenerating

From a HIP or CUDA build, on a GPU:

```sh
build-hip/tools/qd_nevpt2_gradient/free-breakfast-qd-nevpt2-gradient \
    test/gtest/data [case ...]
```

writes the cases named, or all of them. It takes about 25 minutes on the RX 9060 XT, 20 of them the CAS(6,6) ring's 162 points (about 7 s each, where a formaldehyde point takes under 1 s). A refused case writes
nothing and fails the run, naming the check.
