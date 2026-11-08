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
stationary points (roots 1.1e-3 Eh apart), so it is not written.

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
