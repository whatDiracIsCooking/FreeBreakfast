# Z-vector

The Lagrange multipliers of one root's gradient in a state-averaged CASSCF.
The SA energy is stationary in the orbitals and the CI, but root I's energy
E_I is not, so its nuclear gradient needs the multipliers z = (z_o, z_c) of
the SA stationarity conditions:

    A z = −b,   b = (g_I, 0),

with A the SA Hessian (orbital and CI blocks) and g_I the orbital gradient of
E_I at the SA orbitals. b's CI part is zero: every root is an eigenvector of
the same active-space H₀. Eliminating z_c leaves the orbital block's Schur
complement, the CI-relaxed Hessian of [`../relaxed_hessian/`](../relaxed_hessian/README.md):

    H̃ z_o = −g_I,   z_c,r = −(H₀ − E_r)⁻¹ (1 − Σ_s c_s c_sᵀ) H′(z_o) c_r.

## Module

`free_breakfast.mcscf.zvector`, target `free_breakfast::mcscf::zvector`.
**Internal to the MCSCF layer**: `free_breakfast.mcscf` does not re-export
it, and the SA gradient links it PRIVATE. Everything is in
`namespace free_breakfast::mcscf`.

| Name | What it is |
|------|------------|
| `ZVector` | `orbital` (z_o over the packed pairs), `ci` (z_c, dim × nroots, each off every root), `iterations` (H̃ products) and `residual` (‖H̃ z_o + g_I‖) |
| `zvector` | The solve, from a `RelaxedHessian`, the `response::Hessian` and `ci::Roots` it was created over, and g_I on the device |

## The solve

1. Conjugate gradients on H̃ z_o = −g_I from zero, preconditioned by
   diag(H)⁻¹ (`response::Hessian::diagonal`; at a minimum diag(H) ≥ diag(H̃)
   > 0), to ‖r‖ ≤ `tolerance` (1e-10 by default) in at most
   `max_iterations` (200) products. H̃ is positive definite at a minimum;
   a direction it is not positive along is a Divergence error.
2. One more product at z_o, for the true residual: the recurrence's r drifts
   from it, H̃'s products being exact only to the CI response's conjugate
   gradients (1e-8 of their right-hand side).
3. That product's `respond` leaves δ(z_o) in `RelaxedHessian::delta()`, and
   z_c is δ with every root projected out: the other roots' part `respond`
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

## Files

| File | Role |
|------|------|
| `zvector.cppm` | Module interface: `ZVector`, `zvector` |
| `zvector.cpp` | The preconditioned conjugate gradients and z_c's projection |
| `CMakeLists.txt` | The module `free_breakfast.mcscf.zvector` |
