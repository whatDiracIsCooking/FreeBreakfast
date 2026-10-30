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

## Files

| File | Role |
|------|------|
| `zvector.cppm` | Module interface: `ZVector`, `zvector` |
| `zvector.cpp` | The preconditioned conjugate gradients and z_c's projection |
| `CMakeLists.txt` | The module `free_breakfast.mcscf.zvector` |
