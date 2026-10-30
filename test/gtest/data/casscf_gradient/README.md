# `casscf_gradient/`

`casscf_gradient.json`, written by `tools/emit/casscf_gradient_reference.py`
and never hand-edited: PySCF's analytic nuclear gradients of the
density-fitted CASSCF, `df.grad.casscf` for one state and `df.grad.sacasscf`
at **every** root of a state average, the references the DF-CASSCF gradient
tests read (#542, #543, #544).

## Cases

All cc-pVDZ over `data/aux_basis/cc-pVDZ-JKFIT.json`:

| `name` | Molecule | CAS | ncore | Roots, weights | `from` |
|---|---|---|---|---|---|
| `h2o_ccpvdz` | H₂O, C₂ᵥ | (4,4), `active` [2, 3, 5, 6] | 3 | 1 | `h2o_ccpvdz` |
| `h2o_skew_ccpvdz` | H₂O off symmetry (`df_reference.py`'s `WATER_SKEW`) | (4,4), `active` [2, 3, 5, 6] | 3 | 1 | — |
| `h2co_ccpvdz_sa2` | H₂CO | (4,3), frontier | 6 | 2 singlets, [0.5, 0.5] | `h2co_ccpvdz_sa2` |

`from` names the `../casscf/casscf_df.json` case with the same molecule,
start, active space and fitting set; the tool checks its energies against
that file's (each root to 1e-6, the weighted energy to 1e-9). The skewed
water has no such twin; it is there so no gradient component vanishes by
symmetry. PySCF refuses unequal weights in SA gradients, so
`h2co_ccpvdz_sa2_weighted` is not carried.

## How each case is made

1. The exact RHF and the start (`sort_mo` over `active`), as
   `casscf_reference.py` makes them.
2. The DF-CASSCF over a **DF-RHF `_scf` sharing its `with_df`**:
   `mcscf.CASSCF(scf.RHF(mol).density_fit(auxbasis=...), ...)
   .density_fit(auxbasis=...)`, with `casscf_reference.py`'s tolerances.
   Over an exact `_scf`, PySCF's DF gradient would build its generalized
   Fock (W) with exact J and K while the energy is fitted. The tool checks the
   fit is exactly the file's shells.
3. mc1step and mc2step from the start, each converged (`conv_tol` 1e-12,
   `conv_tol_grad` 1e-7), polished by Newton steps to |g| < 1e-10 (H's least
   eigenvalue above 1e-3), then put in canonical orbitals (below) with the
   CI re-solved densely. The two must reach one minimum (energies to 1e-9),
   and their gradients agree to 1e-9.
4. The gradient. A state average's Lagrange solve runs at `conv_rtol` 1e-12
   (PySCF's default is 1e-7).

### The orbitals

`mc.canonicalize(sort=True, cas_natorb=True)` at the state-averaged 1-RDM:
core and virtual orbitals are eigenvectors of the state-averaged Fock in
their own space, ascending, and active orbitals are the state-averaged
natural orbitals, occupation descending. The tool refuses a pair closer than
1e-5 in any of the three, so each orbital is fixed up to its sign. Signs are
PySCF's, in PySCF's AO order. A test in its own orbitals matches them by
`orbitals` and aligns signs (or compares magnitudes). The gradients do not
depend on this; `z` and `ci` do.

## Keys

Per case:

| Key | Meaning |
|---|---|
| `name`, `from`, `basis`, `charge`, `electrons` | As in `../casscf/` |
| `aux_basis` | `name`, `file` (from the repository root), `sha256`, `naux` |
| `atoms` | `symbol`, `z`, `coord` (bohr), in gradient row order |
| `nao`, `nmo` | Spherical AOs; orbitals |
| `ncore`, `ncas`, `nelecas`, `active` | The active space; `active` as in `../casscf/` (null when frontier) |
| `weights` | One per root |
| `energies`, `energy` | Each root's energy at the minimum; the weighted energy |
| `orbitals` | `core_energies`, `virtual_energies` (state-averaged Fock eigenvalues), `natural_occupations` (active, descending) |
| `ci` | Each root's CI vector at the minimum, an n_alpha-strings × n_beta-strings matrix in PySCF's `cistring` order, unit norm |
| `pairs` | SA only: the non-redundant (p, q), p > q, of the multipliers, in PySCF's packed (row-major) order |
| `roots` | One record per root, below |
| `checks` | What the tool measured, below |

Per root (`roots[i]`):

| Key | Meaning |
|---|---|
| `root`, `energy` | Its index and energy |
| `total` | PySCF's gradient, n_atom × 3, hartree/bohr |
| `parts` | The same split into `kinetic`, `potential` (nuclear attraction, both halves), `overlap` (−W at S′), `two_electron` (J, K, the active 2-RDM term and the auxiliary-basis response) and `nuclear_repulsion`; they sum to `total` |
| `responses` | SA only: `hamiltonian` (root I's RDMs at the SA orbitals), `orbital` (z_o's term) and `ci` (z_c's term), summing to `total` |
| `z` | SA only: `orbital` (z_o, aligned with `pairs`), `ci` (z_c, one matrix per root r of the average, shaped as `ci`), `residual` (‖A z + b‖) and `ci_overlap` (max \|⟨z_c,r\|c_s⟩\|) |

The parts are PySCF's own gradient code run once per part with every other
family of derivative integrals zeroed (hcore split into kinetic and the
rest), so each is the contraction of PySCF's derivative integrals with its
converged densities. For an SA root they include the multiplier terms.

The multipliers are PySCF's `Lvec` for root I: A z = −b, b the gradient of
root I's energy (`newton_casscf.gen_g_hop`'s, orbital then CI), A the
state-averaged Hessian (the same), projected off the state-averaged CI space.
κ_pq = z_pq and κ_qp = −z_pq for a pair (p, q). This tree packs pairs
column-major (`src/response/README.md`) and PySCF row-major, so match by
(p, q), not by position. A and b are both in PySCF's convention, whose
orbital blocks are each half this tree's H and g; scaling A and b together
leaves z unchanged. A solver whose CI coordinates are normalized differently
from `gen_g_hop`'s gets a z_c scaled accordingly.

## Checks

Every case records, and the tool asserts:

| `checks` key | Asserted below | Measured (this file) |
|---|---|---|
| `rerun` (fresh PySCF gradient object vs `total`) | 1e-12 SS, 1e-9 SA | 0 |
| `parts_residual` (parts vs `total`) | 1e-10 | ≤ 1.7e-14 |
| `translation` (max over k of \|Σ_A G[A][k]\|) | 1e-8 | ≤ 5.2e-14 |
| `mc1step_mc2step` (the two minima's gradients) | 1e-9 | ≤ 6.4e-12 |
| `finite_difference` (directional derivative along 2 random unit displacements vs a 5-point difference, step 5e-3 bohr, of each root's fitted energy) | 1e-7 | ≤ 2.2e-10 |

## Regenerating

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/emit/casscf_gradient_reference.py
```

About 3 minutes. BLAS and OpenMP run at one thread and no date or host is
written, so a rerun on the same PySCF and numpy rewrites the file byte for
byte. `test/pytest/test_casscf_gradient_reference.py` checks the invariants
with the standard library alone.

## `casscf_optimize.json`

Written by `tools/emit/casscf_optimize_reference.py`, never hand-edited: the
reference minimum of the `[optimize]` CASSCF chain's GPU case (#546). One
case, `h2o_ccpvdz`: H₂O in cc-pVDZ over `cc-pVDZ-JKFIT`, the state-specific
CAS(4,4) over `active` [2, 3, 5, 6], from `h2o_skew_ccpvdz`'s off-symmetry
geometry (`start`, with `start_energy`) to PySCF's DF-CASSCF minimum (`atoms`,
bohr, and `energy`).

geomeTRIC is not in the tools environment, so the minimum is **Newton's
method over `df.grad.casscf`**, its Hessian a central difference of that
gradient (step 1e-3 bohr) recomputed every step, rigid translations and
rotations projected out, converged to max |dE/dR| < 1e-7 hartree/bohr
(`max_force`, `newton_steps`, `optimizer`). Each energy and gradient is the
CASSCF built as above, polished to |g| < 1e-10 and differentiated at the
polished orbitals; every geometry after the start carries the previous one's
orbitals, Loewdin-orthonormalized in the new overlap. At the minimum the
CASSCF is rerun from a fresh RHF and `sort_mo`, and must reach the same energy
to 1e-9 (`fresh_start_energy`).

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/emit/casscf_optimize_reference.py
```
