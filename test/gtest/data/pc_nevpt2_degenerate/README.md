# pc_nevpt2_degenerate golden data

The degenerate-inactive-pair references of the analytic PC-NEVPT2 Hessian
(#1268) and of the PC-QD-NEVPT2 state Hessian (#1277, the `qd_` cases):
this tree's own numerical `[hessian]` through the `free-breakfast` binary,
the h, h/2, h/4 ladder of fourth-order central differences of the analytic
PC gradient (or the PC-QD state's, tracked by its eigenvector) (`src/pipeline/hessian/README.md`, "The ladder and the
bilinearity check"), at the step's default `pc_floor` (1e-10), Cartesian
coordinates. Written by `tools/nevpt2_hessian/degenerate_pair_ladder.py`,
whose docstring is the method; never hand-edited.

The analytic Hessian takes these pairs through their off-diagonal f, read
in turned bases (`src/hessian/README.md`, "A degenerate inactive pair");
the ladder differences gradients that never see that construction, so it
is an independent check of it. Each ladder settled: no ratio unsettled,
every element's bar the measured noise floor.

| case | what | pairs under 1e-6 Eh | floor (Eh/bohr²) |
|---|---|---|---|
| `nh3_c3v_631g` | C3v ammonia, 6-31G / cc-pVDZ-JKFIT, CAS(8,7) over 1–7 | virtual e (10, 9), (12, 11), 3e-14 | 4.2e-7 |
| `nh3_near_631g` | the same, H1 moved 2e-6 bohr along x | the same, 7.4e-8, no symmetry | 3.2e-7 |
| `nh3_split_631g` | the same, H1 moved 3e-5 bohr along x | none: the same pairs at 1.1e-6 and 2.9e-6 Eh, inside PC's 1e-4 Eh reach (#1296) | 3.1e-7 |
| `hexatriene_sto3g` | all-trans hexatriene (C2h), STO-3G / cc-pVDZ-JKFIT, CAS(6,6) over 19–24 | the terminal carbons' 1s (5, 4), 4.6e-7 | 1.2e-7 |
| `qd_nh3_c3v_631g_sa2` | `nh3_c3v_631g` over an SA-2 CAS(8,7), equal weights, PC-QD state 1 | virtual e (9, 8), (11, 10) at f^SA, 1e-11 | 4.1e-7 |
| `qd_nh3_near_631g_sa2` | the same, H1 moved 2e-6 bohr along x | the same, 2.3e-7, no symmetry | 4.1e-7 |

## Keys

The start, in `nevpt2_hessian/`'s format so the test's harness reads it
unchanged: `name`, `atoms` (`symbol`, `coord` in bohr), `charge`, `basis`
and `aux_basis` (`name`), `ncas`, `nelecas`, `active`. Then `pc_floor`,
and `pc`: `hessian` (the finest rung's, symmetrized, [i][j] in
Eh/bohr²), `bar` (per element, [i][j]), `noise` and `unsettled` (the
ladder's), `asymmetry`. A `qd_` case adds `nroots`, `weights`, `state`
(H_eff's eigenvalues ascending) and `qd` (the step's tracking report).
`provenance` has the commit, the date and the config the binary ran.

The ammonia ladder's noise step re-solves the Z-vector 100 times tighter,
held at what H̃'s products resolve (1.3e-11 here, #1304). At a flat
1e-12 the solve once stopped at 1.6e-12 on the gfx1200 and the run ended
there. At `nh3_split_631g` it stopped every time, which is why that case
came after the floor.
