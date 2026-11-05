# nevpt2_hessian golden data

Semi-numerical nuclear Hessians of the single-state SC- and PC-NEVPT2 (and
DF-CASSCF) energies, the oracle milestone 24's analytic Hessians
(`hessian::df_sc_nevpt2`, `hessian::df_pc_nevpt2`) are gated on: a
fourth-order central stencil at steps h, h/2 and h/4 over the fourth-order
central gradient of re-converged PySCF energies (PC by the C++ step's cut),
every point through a continuity guard. Written by
`tools/nevpt2_hessian/nevpt2_hessian.py` (#803), whose docstring is the
method; never hand-edited.

No program has an NEVPT2 Hessian, and no analytic NEVPT2 gradient exists
outside this tree's C++, so the reference gradient is the one
`../nevpt2_gradient/` already is (a difference of re-converged energies),
and nothing here ran the code it gates. Same fitting set as the tree:
`data/aux_basis/cc-pVDZ-JKFIT.json`.

## Cases

All STO-3G / cc-pVDZ-JKFIT, single state, all-electron (`frozen_core` 0).

| case | kind | CAS, active | h, δ (bohr) | inactive gaps (core, virtual) | schemes | bar, max (Eh/bohr²) | judged ratios | worst \|ratio − 16\| |
|---|---|---|---|---|---|---|---|---:|
| `h2co_skew_sto3g` | skew (C1) | (4,4), 6–9 | 0.03, 5e-3 | 0.140, 0.341 | CASSCF, SC, PC | 2.0e-7, 1.8e-7, 1.8e-7 | 8, 10, 10 | 0.78 |
| `nh3_cs_sto3g` | symmetric (Cs) | (2,2), 4–5 | 0.03, 5e-3 | 0.285, 0.289 | CASSCF, SC (A′ block), PC | 5.5e-8, 5.7e-6 (A″ failed), 5.5e-8 | 8, 4, 10 | 0.96 |
| `h2co_neardeg_sto3g` | near-degenerate virtual pair | (4,4), 6–9 | 0.04, 5e-3 | 0.185, **0.0060** | CASSCF, PC | 1.2e-7, 1.9e-7 | 19, 24 | 1.25 |

Every judged ratio of every element (and of every pair's directional
ladder) lies in (12.8, 19.2); the rest are below their noise floor (2.6e-8
to 6.2e-8), and their bar is that floor. Every Hessian is symmetric to its
bar and bilinear on every coordinate pair, but SC's A″ block of the
symmetric case.

- **`h2co_skew_sto3g`**: every block defined. Found by a search for a
  geometry where SC is smooth on the stencil's scale: SC's Rayleigh
  quotient kinks wherever a perturber's norm passes near zero or two
  canonical inactive orbitals nearly cross, on scales of 0.01 bohr in
  every larger-basis case tried (water and formaldehyde in cc-pVDZ,
  ammonia and water in 6-31G, H₆ in 6-31G), so none gave an SC ladder. At h = 0.04 one of its directional ladders,
  SC along (C y + C z)/√2, reads 28 (a kink 0.06–0.08 bohr out), so it
  takes h = 0.03.
- **`nh3_cs_sto3g`**: mirror plane xz, H1 in it, H3 H2's image; the
  coordinates are symmetry-adapted (7 A′, 5 A″; `coordinates`), so the
  failing block is named. SC has empty perturbers by symmetry (Sijr 10,
  Srsi 5, Sij 6, …), whose energy along a displacement t that breaks the
  mirror is t² Q(direction), Q a ratio of quadratic forms (#894): a second
  derivative along every line, but no bilinear form. SC's A″ × A″ block
  fails: `failing_pairs` [7,8], [8,10], [8,11] (N y with H1 y, H1 y with
  H2x − H3x and H2z − H3z), residual up to 5.0e-7 against 2.8e-7, and four
  pairs asymmetric past their bar; it is recorded, not a Hessian. SC's A′
  block, and PC and CASSCF everywhere, are held to every check. PC has no
  lifted null mode here: its dropped modes are the CAS(2,2) structural ones
  at every displaced point.
- **`h2co_neardeg_sto3g`**: the two virtuals 6.0e-3 apart at R0, and 5.6e-5
  at the closest displaced point, for #817's κ_oo. PC only (and CASSCF):
  SC is not invariant to rotations within the pair and does not ladder
  there. Not a stationary point (|g| up to 0.19).

## Keys

| key | meaning |
|---|---|
| `provenance` | PySCF, block2, numpy versions, threads (1), command. No date or host: a rerun is byte-identical |
| `basis`, `aux_basis` | file and sha256 (and `naux`) |
| `atoms`, `ncore`, `ncas`, `nelecas`, `active` | geometry (bohr) and active space (`active`: the RHF orbitals `sort_mo` takes) |
| `energies`, `e_corr`, `classes` | at R0: CASSCF, SC, PC totals; per class SC norm and energy, PC energy, SC's empty perturbers (norm ≤ 1e-14) and PC's dropped modes |
| `inactive_gaps` | smallest canonical core and virtual gaps at R0 |
| `casscf_analytic_gradient` | PySCF's DF-CASSCF gradient at R0 (`checks.casscf_gradient_vs_analytic`: the stencil's CASSCF gradient against it) |
| `steps` | `h`, the `ladder` h, h/2, h/4, the eight `displacements`, the gradient's `delta`, the stencil's `offsets` and `weights` |
| `coordinates` | `labels`, `irreps` (`irreps_order[0]` totally symmetric) and `vectors` (row j = coordinate j, Cartesian, orthonormal; the identity for the C1 cases) |
| `may_fail` | the quantities held on the totally symmetric block only |
| `casscf`, `sc`, `pc` | per scheme (below) |
| `noise`, `tolerances`, `guard`, `checks` | σ_E's measurement, every bound, the worst guard values, the reference-point checks |

Per scheme, in the coordinates (H[i][j] = q_iᵀ ∂g/∂q_j; a Cartesian
Hessian is Qᵀ H Q with Q's rows `coordinates.vectors`):

| key | meaning |
|---|---|
| `gradient` | Cartesian gradient at R0 (n_atom × 3) |
| `hessian` | the finest step's, symmetrized |
| `bar` | per element, the larger of H[i][j]'s and H[j][i]'s `column_bar`: \|D(h/2) − D(h/4)\|, or the noise floor where that is below it. **The tolerance an analytic Hessian is held to** |
| `ladder` | D(h), D(h/2), D(h/4), unsymmetrized |
| `ratios` | (D(h) − D(h/2)) / (D(h/2) − D(h/4)); null below the floor |
| `asymmetry` | \|D(h/4) − D(h/4)ᵀ\| |
| `noise` | σ_E and the floor |
| `points` | per column, the gradient (in the coordinates) at each displacement: the ladder recomputes from these |
| `bilinearity` | per pair: the directional ladder along (q_i + q_j)/√2, its derivatives, ratio, residual against (H_ii + H_jj + H_ij + H_ji)/2 and the allowed |
| `failing_pairs`, `asymmetric_pairs`, `failed_blocks` | the misses (empty but SC in `nh3_cs_sto3g`) |

## Regenerating

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
        tools/nevpt2_hessian/nevpt2_hessian.py

6 768 energies a case (48 for the gradient at R0, 12 columns × 8 points ×
48, 66 pairs × 8 × 4) plus 40 for the noise; with the default (and cap) of
8 one-thread workers, about 4.5 min for each formaldehyde and 2 min for
ammonia, 12 min in all. A miss prints `STUCK` with the numbers and writes
nothing for that case.
