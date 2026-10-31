# Z-vector

The Lagrange multipliers of one root's gradient in a state-averaged CASSCF.
The SA energy is stationary in the orbitals and the CI, but root I's energy
E_I is not, so its nuclear gradient needs the multipliers z = (z_o, z_c) of
the SA stationarity conditions:

    A z = −b,   b = (g_I, y),

with A the SA Hessian (orbital and CI blocks) and g_I the orbital gradient of
E_I at the SA orbitals. For a CASSCF gradient b's CI part y is zero: every
root is an eigenvector of the same active-space H₀. A Lagrangian with more in
it (NEVPT2's, #563) has a nonzero one, passed as `ci_rhs`. Eliminating z_c
leaves the orbital block's Schur complement, the CI-relaxed Hessian of
[`../relaxed_hessian/`](../relaxed_hessian/README.md):

    H̃ z_o = −g_I − H_oc w,   w_r = −(H₀ − E_r)⁻¹ P y_r,
    z_c,r = −(H₀ − E_r)⁻¹ P H′(z_o) c_r + w_r,

with P = 1 − Σ_s c_s c_sᵀ. A null `ci_rhs` is y = 0, w = 0: the CASSCF call.

## The CI right-hand side

The equations are, in full,

    H z_o + H_oc z_c = −g_I,
    (H₀ − E_r) z_c,r + P H′(z_o) c_r = −P y_r   for every root r,

H_oc z_c the gradient over the packed pairs of z_c's transition RDMs (weighted
as the average) and H′(x) the active-space Hamiltonian's first derivative
along κ(x). **The CI rows are unweighted and unscaled**: the coordinates of
PySCF's `gen_g_hop` (c → c + δc), the ones the CASSCF's z_c has always been
in. A caller writing its CI block as the tex's J_cc = 2 P (H − E) P
(`nuclear_gradient.tex`, Eq. cp) must scale its right-hand side to match.
That CI row is twice this one (its J_cκ, 2 P (H[κ] − ⟨H[κ]⟩) c, is twice
P H′ c too), so for the same z, its orbital rows being this tree's, it
passes y = ½ P_c(y + y^c). Left unscaled, z_c and z_o both come out wrong,
not just rescaled. y is projected off every root here, so a component along
a root is ignored; it must be in the roots' spin, as a Lagrangian's CI
derivative is, since H₀ − E_r need not be positive along another spin's
directions.

## Module

`free_breakfast.mcscf.zvector`, target `free_breakfast::mcscf::zvector`.
**Internal to the MCSCF layer**: `free_breakfast.mcscf` does not re-export
it, and the SA gradient links it PRIVATE. Everything is in
`namespace free_breakfast::mcscf`.

| Name | What it is |
|------|------------|
| `ZVector` | `orbital` (z_o over the packed pairs), `ci` (z_c, dim × nroots, each off every root), `iterations` (H̃ products) and `residual` (‖H̃ z_o + g_I + H_oc w‖) |
| `zvector` | The solve, from a `RelaxedHessian`, the `response::Hessian` and `ci::Roots` it was created over, g_I on the device and an optional CI right-hand side y (dim × nroots) |
| `CpSolver` | How `zvectors` solves: `PerColumn`, `Schur` or `Coupled` (below) |
| `ZVectors` | `zvectors`' result: `orbital` (size() × ncol), `ci` (ncol blocks of dim × nroots), and per column `iterations` and `residuals` |
| `zvectors` | The same equations on ncol right-hand sides at once: g (size() × ncol) and an optional y (ncol blocks), by a `CpSolver` |

## The solve

0. With a CI right-hand side, `RelaxedHessian::couple`: w into `delta()`
   (kept as z_c's start) and H_oc w, which is added to g_I.
1. Conjugate gradients on H̃ z_o = −(g_I + H_oc w) from zero, preconditioned by
   diag(H)⁻¹ (`response::Hessian::diagonal`; at a minimum diag(H) ≥ diag(H̃)
   > 0), to ‖r‖ ≤ `tolerance` (1e-10 by default) in at most
   `max_iterations` (200) products. H̃ is positive definite at a minimum;
   a direction it is not positive along is a Divergence error.
2. One more product at z_o, for the true residual: the recurrence's r drifts
   from it, H̃'s products being exact only to the CI response's conjugate
   gradients (1e-8 of their right-hand side).
3. That product's `respond` leaves δ(z_o) in `RelaxedHessian::delta()`, and
   z_c is δ + w with every root projected out: the other roots' part `respond`
   adds is the state average's rotations among its roots, which PySCF's
   Lagrange equations project away (and which, at equal weights, cancel in H̃).

g_I is `response::fock`'s gradient at root I's own RDMs (`ci::Rdm::rdm12` at
weight 1 on root I) and the SA orbitals. H, g and so H̃ are each twice PySCF's
`gen_g_hop` ones, which leaves z_o as PySCF's; z_c is in the CI coordinates of
PySCF's `gen_g_hop` (c → c + δc, unnormalized), so it is PySCF's `Lvec` CI
part as it is.

## Many right-hand sides

A Hessian solves these equations once per nuclear coordinate, E_λλ λ^x =
−E_λx for 3N right-hand sides over one A (`.temp/hessian/casscf_hessian.tex`
§4.3). `zvectors` takes them as a block, each column in the convention
above (the caller assembles the response as Λᵀ·rhs, so the CI rows' scaling
cancels), and solves each to ‖r‖ ≤ `tolerance` by one of three methods:

- **`PerColumn`**: `zvector` on each column in turn.
- **`Schur`**: `zvector`'s conjugate gradients on every column in lockstep.
  A sweep gathers the active columns' directions and takes one block product
  of H̃ (`RelaxedHessian::apply` over k columns: H over the block in one call,
  then every column's CI response by one lockstep conjugate gradients over
  all k · nroots vectors, one sigma of the block per step), and a column
  stops on its own residual as in `zvector`. A CI right-hand side is the
  block `couple`. Each column is `zvector`'s on it alone to the CI response's
  summation order: 1e-13 in the test below.
- **`Coupled`**: the unreduced equations in (z_o, z_c), as PySCF solves its
  CP-MCSCF equations, with no CI solve inside a product. Their CI rows are
  scaled by D_r = 2 w_r, which makes the operator symmetric (H_oc δ is the
  gradient of 2 Σ_r w_r ⟨δ_r|H′(x)|c_r⟩, so H_oc = (D H_co)ᵀ) and, at a
  minimum, positive definite on the space off the roots:

      [ H        H_oc   ] [z_o]     [ g     ]
      [ D H_co   D H_cc ] [z_c] = − [ D P y ],   H_cc,r = P (H₀ − E_r) P.

  Preconditioned conjugate gradients on both blocks at once, the
  preconditioner block-diagonal: diag(H)⁻¹ as `zvector`'s, and
  1 / (D_r max(|diag(H₀)_i − E_r|, 10⁻²)) on the CI rows, projected off the
  roots. A product is `RelaxedHessian::apply_coupled` over the active
  columns. Its residual is the norm of both blocks of the unscaled
  equations. It needs every weight positive, and H₀ − E_r positive along
  every direction the diagonal preconditioner reaches: in the full
  determinant space (not spin-adapted) a lower root of another spin than the
  averaged ones would make it indefinite (a Divergence error).

The CI rows are z_c's, so all three return the same z_c (off every root).
Schur and PerColumn solve H̃, whose products are good only to the CI
response's `response_tolerance` (1e-8 of its right-hand side): with a CI
right-hand side their true residual, ‖H̃ z_o + b‖ from one more product, sits
near 1e-9 rather than the 1e-10 asked for, and z by that much. Coupled solves
the exact equations, and agrees with `zvector` over an H̃ whose responses are
solved to 1e-12 to 3e-11 relative.

At unequal weights the rotations among the averaged roots belong in A
(#613); none of the three carries them yet, and Schur and PerColumn keep the
other-root part of H̃'s CI response that Coupled's projected CI space leaves
out, so the methods agree at equal weights and for one root.

### Which is the default

**`Schur`**, because it was the fastest of the three against `PerColumn`
everywhere it was measured; Coupled is slower on the larger molecule, so it
is not the default. The wall clock of 3N right-hand sides
(`bench/mcscf_zvectors.cpp`, `free_breakfast_mcscf_zvectors`), each an
orbital column uniform in [−0.1, 0.1) and a CI one P H_y c for a random
spin-free H_y, over one DF-CASSCF(4,4) (cc-pVDZ / cc-pVDZ-JKFIT, frontier
active space, equal weights) and one H̃ for all three; the best of two
solves after a warm-up, on the **RTX 3080 (CUDA)**, the card held. The
gfx1200 was not measured.

| System | Columns | `PerColumn` | `Schur` | `Coupled` |
| --- | --- | --- | --- | --- |
| water, 1 root | 9 | 0.567 s | 0.299 s (1.90×) | 0.234 s (2.42×) |
| water, SA-2 | 9 | 1.175 s | 0.436 s (2.70×) | 0.311 s (3.78×) |
| octatetraene (C₈H₁₀, 18 atoms, nmo 162, naux 790), 1 root | 54 | 41.66 s | 40.20 s (1.04×) | 44.15 s (0.94×, slower) |
| octatetraene, SA-2 | 54 | 46.71 s | 41.45 s (1.13×) | 52.14 s (0.90×, slower) |

Schur takes exactly PerColumn's products (1588 and 1587 over octatetraene's
54 columns) and agrees with it to 6e-13 (one root) and 6e-10 in z_c of
order 100 (SA-2). Coupled takes 10 to 28% more products (1756, 2027), each
cheaper, and agrees with PerColumn to 1e-8 to 8e-8, PerColumn's own error:
its true residual is 5e-9 to 1.3e-8 where Coupled's is below 1e-10.

What separates them is what a product costs. On water the CI side is most
of it, and lockstep batching (Schur) or not solving it (Coupled) pays. On
octatetraene H's product dominates: `response::Hessian::apply` builds each
column's J and K on its own even over a block, so the batched J/K of
density_fitting/jk (#595) is not reached from here, and Schur saves only the
CI side (4% and 11%) while Coupled's extra products cost more than its
cheaper ones save. A block J/K inside `response::Hessian::apply` would help
Schur and Coupled alike, Coupled more (its products are mostly that).

## Tests

`test/gtest/mcscf_zvector_gpu.cpp`, on every state-averaged case of
`test/gtest/data/casscf_gradient/` (PySCF's `df.grad.sacasscf`): the
DF-CASSCF by `mcscf::solve`, put in the reference's canonical orbitals with
the CI re-solved there, then per root the residual, z_o by (p, q) and z_c
against the reference's to 1e-7 (both in magnitude: the orbitals' signs are
not PySCF's), and z_c off every root.

With a random CI right-hand side, y_r = P H_y c_r for a random spin-free
H_y (so in the roots' spin), on `h2o_ccpvdz` (state specific) and `h2co_ccpvdz_sa2`: the coupled equations above, each block built
from its parts (`response::Hessian::apply` and `response_gradient` of z_c's
transition RDMs; H₀'s sigma and H′(z_o)'s), to 1e-7 for a y of order 1 (the
CI response's products are good to 1e-8 of their right-hand side, which
bounds the solve's true residual too), and z_c off every root.

`test/gtest/mcscf_zvectors_gpu.cpp`, on `h2o_ccpvdz` (state specific) and
`h2co_ccpvdz_sa2` (DF-CASSCF from the Casci start, the CI re-solved at its
orbitals): three columns, root 0's gradient then random orbital right-hand
sides, each with a random CI one y_r = P H_y c_r. Schur against `zvector` on
each column alone over the same H̃, Coupled against `zvector` over an H̃ whose
CI responses are solved to 1e-12, each to 1e-10 of the column's largest
|z_o| and |z_c|, and z_c off every root; and no columns refused.

## Files

| File | Role |
|------|------|
| `zvector.cppm` | Module interface: `ZVector`, `zvector`, `CpSolver`, `ZVectors`, `zvectors` |
| `zvector.cpp` | The preconditioned conjugate gradients and z_c's projection; `zvectors`' three methods |
| `CMakeLists.txt` | The module `free_breakfast.mcscf.zvector` |
